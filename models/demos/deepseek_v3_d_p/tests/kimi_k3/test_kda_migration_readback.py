# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 prefill on Galaxy publishes a migration table whose KDA entries read back the right state.

The serving path end to end, in one process: the production adapter builds the runtime and allocates the KV
cache with its SP-sharded KDA slabs, the traced runtime prefills 11 chunks x 5120 = 56320 tokens of the 1M
code_debug prompt, and ``runtime.build_kv_chunk_table`` publishes the table to protobuf. The table is imported
back and ``kda_state_readback.score_layers`` (the read-back tool's own walk) reads every segment of every KDA
layer from the one chip its entry names and scores the reassembled state against the head/tail golden at 56320
tokens (row 54). Each layer's read-back must also equal, bit for bit, the carry the layer itself imports from the
slabs (``import_layer``: copy + SP all-gather), which pins the table to the exact bytes on the right chip.

Reads go through the table (``read_device_chunk``), not the tool's device-less UMD path, because this process
holds the devices and a second, bare UMD cluster pins 1 GiB of host memory per chip (~32 GiB on a Galaxy).

    KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e \
    TT_KIMI_K3_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill \
    scripts/run_safe_pytest.sh \
      'models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_migration_readback.py::test_published_kda_state_reads_back[blackhole-kimi_k3-torus-xy-8x4-L4]' -vv -s
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import read_slab_layer
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import deallocate_state
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
from models.demos.deepseek_v3_d_p.tt.runners import kda_state_readback
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import decode_tail

CHUNK = 5120
NUM_CHUNKS = 11
REAL_LEN = CHUNK * NUM_CHUNKS  # 56320 = 55 golden periods: head row 54
# The KV cache length: a multiple of the chunk and of 8192, so every KV slot starts on DRAM bank 0 (kv_bank_alignment).
MAX_SEQ_LEN = 81920
SLOT = 0
SP_AXIS, TP_AXIS = 0, 1
# The read-back tool's sanity bar, not an accuracy gate: the carry drifts with prefix length. The bit-for-bit
# comparison with the slabs is the exact check; the PCCs are the result.
PCC_BAR = 0.9


def _token_ids(trace_dir: Path) -> list[int]:
    with open(trace_dir / "metadata.json") as handle:
        ids = json.load(handle)["token_ids"]
    assert len(ids) >= REAL_LEN, f"{trace_dir} holds {len(ids)} tokens, need {REAL_LEN}"
    return ids[:REAL_LEN]


def _packed_metadata_msg(mesh_device, slot_id: int, actual_start: int, actual_end: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([slot_id, actual_start, actual_end], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _carry_as_global(mesh_device, geometry, carry):
    """The layer's imported native carry as the read-back's global forms: recurrent ``[H, D, D]`` ([k, v], as
    stored) and golden-order convolution ``[K-1, 3*H*D]``. The carry is TP-sharded and SP-replicated after the
    import's gather; SP row 0 of each TP column is taken."""
    rows, cols = tuple(mesh_device.shape)
    rec_shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(carry.recurrent)]
    conv_shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(carry.convolution)]
    recurrent = torch.cat([rec_shards[col].reshape(-1, geometry.head_dim, geometry.head_dim) for col in range(cols)])
    per_chip = geometry.local_heads * geometry.head_dim
    tails = [
        decode_tail(conv_shards[col], KDA_CONVOLUTION_PAGE_CHANNELS).reshape(geometry.conv_history, -1)
        for col in range(cols)
    ]
    convolution = torch.cat(
        [torch.cat([t[:, branch * per_chip : (branch + 1) * per_chip] for t in tails], dim=-1) for branch in range(3)],
        dim=-1,
    )
    return recurrent, convolution


@pytest.mark.parametrize("num_layers", [4, 24], ids=["L4", "L24"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(KimiK3Config),
                l1_small_size=KimiK3Config.L1_SMALL_SIZE,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k3"], indirect=True, ids=["kimi_k3"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 requires Blackhole")
@pytest.mark.timeout(0)
def test_published_kda_state_reads_back(
    variant, config_only, mesh_device, device_params, weight_cache_path, num_links, num_layers, tmp_path
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    golden_dir = Path(variant.kda_golden_default)
    trace_dir = Path(variant.prefill_trace_default)
    if not (golden_dir / "kda").is_dir() or not (trace_dir / "metadata.json").is_file():
        pytest.skip(f"needs the KDA head/tail golden {golden_dir} and the prompt {trace_dir}")
    token_ids = _token_ids(trace_dir)

    hf_config = config_only
    hf_config.max_seq_len = MAX_SEQ_LEN
    sp, tp = tuple(mesh_device.shape)
    params = PrefillRunParams(
        mesh_shape=(sp, tp),
        num_layers=num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK,
        num_users=1,
        capacity_factor=8,
        num_links=num_links,
        gate_mode_name=variant.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=weight_cache_path / f"{sp}x{tp}",
        tp_shard_kv=True,
        use_trace=True,
    )
    # A missing file would be converted and written into the (shared) weight cache; refuse instead.
    assert variant.transformer_cls.check_cache_complete(
        params.weight_cache_path,
        num_layers,
        experts_per_chip=KimiK3Config.NUM_ROUTED_EXPERTS // (sp * tp),
        first_k_dense=KimiK3Config.NUM_DENSE_LAYERS,
        kv_only_last_layer=True,
    ), f"TTNN cache incomplete for {num_layers} layers at {params.weight_cache_path}"
    runtime = variant.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = variant.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    slabs = kv_caches.kda_states
    geometry = kimi_k3_kda_geometry((sp, tp), SP_AXIS, TP_AXIS)
    assert slabs is not None and slabs.geometry == geometry
    assert geometry.stored_heads == geometry.num_heads // (sp * tp), "the slabs must be SP-sharded"
    try:
        runtime.compile(kv_caches)
        runtime.capture_trace(kv_caches)
        logger.info(f"captured; prefilling {NUM_CHUNKS} chunks of {CHUNK} tokens, {num_layers} layers")
        for c in range(NUM_CHUNKS):
            start, end = c * CHUNK, (c + 1) * CHUNK
            msg = _packed_metadata_msg(mesh_device, SLOT, start, end)
            runtime.prefill_chunk(
                runtime.make_chunk_input(token_ids[start:end]),
                kv_caches,
                slot_id=SLOT,
                actual_start=start,
                actual_end=end,
                request_id=c,
                metadata_msg=msg,
            )
            ttnn.deallocate(msg)
            logger.info(f"chunk {c} [{start}:{end}) replayed")
        ttnn.synchronize_device(mesh_device)

        published = runtime.build_kv_chunk_table(kv_caches, str(tmp_path / "kv_chunk_table.pb"))
        table = ttnn.experimental.disaggregation.import_from_protobuf_file(published)
        logger.info(f"table {published}: {table.num_configs()} configs, {table.num_device_groups()} device groups")

        row = REAL_LEN // kda_state_readback.GOLDEN_PERIOD - 1
        results, worst = kda_state_readback.score_layers(
            table,
            variant,
            geometry,
            num_layers=num_layers,
            slot=SLOT,
            row=row,
            golden_dir=golden_dir,
            read_chunk=kda_state_readback.table_chunk_reader(table),
        )

        # The read-back must be the bytes the layer itself resumes from, for every KDA layer, both kinds.
        kda_layers = [layer for layer in KimiK3Config.kda_layer_ids() if layer < num_layers]
        assert tuple(slabs.layer_ids) == tuple(kda_layers)
        for layer in kda_layers:
            carry = read_slab_layer(mesh_device, slabs, SLOT, layer)
            want_rec, want_conv = _carry_as_global(mesh_device, geometry, carry)
            deallocate_state(carry)
            for config_id, kind, want in ((1, "kda_recurrent", want_rec), (2, "kda_convolution", want_conv)):
                assert variant.cache_kind(config_id) == kind
                got, status = kda_state_readback.read_layer(
                    table, config_id, kind, layer, SLOT, geometry, kda_state_readback.table_chunk_reader(table)
                )
                assert status == "ok", f"{kind} layer {layer}: {status}"
                assert torch.equal(got, want), f"{kind} layer {layer}: table read-back != the layer's imported carry"
            logger.info(f"layer {layer}: table read-back == imported carry, bit for bit (both kinds)")
    finally:
        runtime.release_trace()
        del runtime
        gc.collect()

    checked = [r for r in results if r["status"] == "ok"]
    assert len(checked) == 2 * len(kda_layers), f"read back {len(checked)} layer states: {results}"
    for kind, score in worst.items():
        logger.info(f"min pcc {kind}: {score:.6f} over {len(kda_layers)} layers")
    failed = [(r["kind"], r["layer"], r["pcc"]) for r in checked if not r["pcc"] >= PCC_BAR]
    assert not failed, f"layer states below {PCC_BAR}: {failed}"
