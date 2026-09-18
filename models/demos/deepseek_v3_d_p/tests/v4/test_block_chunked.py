# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill PCC for TtV4Block against a vLLM golden, teacher-forced.

One layer, real weights, the golden's own activations. The layer is fed
``decoder_output_layer_{idx-1}`` a chunk at a time and its accumulated output is compared against
``decoder_output_layer_{idx}``, so the measured error is this layer's, with nothing inherited from
the layers below -- which is what ``tests/v4/test_block.py`` cannot do, since it invents its input.

Two axes, and only one of them is teacher-forced. Along depth the input is the golden's, so no
cross-layer error arrives. Along the sequence the block runs free: chunk c+1 attends over the cache
that chunk c wrote, and nothing golden is injected between them, because that accumulation is the
subject.

Nothing is seeded. The attention owns its state and allocates it zeroed (``TtHCA.alloc_state``), so
the run starts at token 0 with an empty cache and builds it as it goes; the golden's cache is a
comparison target, never an input.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import v4_block_state_dict, v4_mhc_weights
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tests.v4.test_block import _pack_streams, _test_config, _unpack_streams
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Block
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs

CHUNK = PREFILL_CHUNK_TOKENS  # 5120 tokens per chunk, the production configuration
SEQ_CACHE = 55 * 1024  # 56320, the length the golden was captured at

# The golden each variant is graded against.
_GOLDEN = {DeepSeekV4ProConfig: golden.V4_PRO, DeepSeekV4FlashConfig: golden.V4_FLASH}

# One row per (layer, attention, MoE) the device can build.
# TODO add a CSA row once CSA has a device implementation: layers 2, 4, 6 ... are CSA in both
# models, so that kind is entirely ungraded today.
_PRO_CASES = [
    pytest.param(1, "heavily_compressed_attention", "hash_moe", id="L1-hca-hash"),
    pytest.param(3, "heavily_compressed_attention", "moe", id="L3-hca-topk"),
]
# Flash has no HCA + hash_moe layer: num_hash_layers is 3, its first two layers are sliding and its
# third is CSA.
_FLASH_CASES = [
    pytest.param(1, "sliding_attention", "hash_moe", id="L1-swa-hash"),
    pytest.param(3, "heavily_compressed_attention", "moe", id="L3-hca-topk"),
]

_BLOCK_PCC = 0.99
_CACHE_PCC = 0.999


def run_chunked_block_v4(mesh_device, device_params, num_links, variant, layer_idx, attn_kind, mlp_kind, n_chunks):
    # Both are asserts rather than skips: this row is meant for CI, and a skipped row reads as a
    # green one.
    gold = _GOLDEN[variant]
    trace = golden.resolve_trace(gold)
    assert trace is not None, f"no golden trace at {gold.trace}; ${gold.trace_env} overrides the path"
    checkpoint = golden.resolve_checkpoint(gold)
    assert checkpoint is not None, f"no checkpoint at {gold.checkpoint}; ${' / $'.join(gold.ckpt_envs)} override it"

    topology = per_axis_topology(device_params["fabric_config"])
    tp_factor = mesh_device.shape[1]
    ms = tuple(mesh_device.shape)
    total_len = n_chunks * CHUNK
    assert total_len <= SEQ_CACHE, f"{n_chunks} chunks ({total_len}) exceed the golden's {SEQ_CACHE}"

    config, model_cfg = _test_config(variant, layer_idx)
    config.max_seq_len = SEQ_CACHE
    # The row names the layer and its pair; this is where the config has to agree.
    assert (config.layer_types[layer_idx], config.mlp_layer_types[layer_idx]) == (attn_kind, mlp_kind), (
        f"case asks for layer {layer_idx} to be {attn_kind}/{mlp_kind}, the config gives "
        f"{config.layer_types[layer_idx]}/{config.mlp_layer_types[layer_idx]}"
    )
    gate_mode = GateComputeMode.HASH_DEVICE if mlp_kind == "hash_moe" else GateComputeMode.DEVICE_FP32
    logger.info(
        f"[v4 chunked] {variant.__name__} layer {layer_idx}: {attn_kind} / {mlp_kind} / " f"{n_chunks} x {CHUNK} tokens"
    )

    ref = golden.v4_layer_from_checkpoint(config, layer_idx, checkpoint)
    block = TtV4Block(
        mesh_device=mesh_device,
        config=config,
        model_cfg=model_cfg,
        state_dict=v4_block_state_dict(ref, config),
        layer_idx=layer_idx,
        seq_len=CHUNK,
        max_seq_len=SEQ_CACHE,
        attn_reference=ref["attn"],
        mhc_weights=v4_mhc_weights(ref),
        num_links=num_links,
        topology=topology,
        sp_axis=0,
        tp_axis=1,
        gate_fallback_mode=gate_mode,
        weight_cache_path=None,  # real weights come off the reference module, not a ttnn cache
    )

    n = config.hc_mult
    out_accum = torch.zeros(total_len, n, config.hidden_size, dtype=torch.float32)
    mesh_device.enable_program_cache()
    programs = mesh_device.num_program_cache_entries()

    for chunk in range(n_chunks):
        start = chunk * CHUNK
        rows = trace.layer_input(layer_idx, start, start + CHUNK).reshape(1, CHUNK, n, config.hidden_size)
        tt_in = ttnn.from_torch(
            _pack_streams(rows, tp_factor),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        tt_out = block(
            tt_in,
            actual_isl=CHUNK,
            actual_start=start,
            input_ids=trace.token_ids(CHUNK, start),
        )
        full = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
        out_accum[start : start + CHUNK] = _unpack_streams(full, n, tp_factor)[0].float()
        ttnn.synchronize_device(mesh_device)

        # Every chunk drives identical device shapes, so one compiled program has to serve them all;
        # a new entry means an op attribute moved with the chunk, which is what breaks trace later.
        now = mesh_device.num_program_cache_entries()
        if chunk:
            assert now == programs, f"chunk {chunk} compiled {now - programs} new program(s) ({programs} -> {now})"
        programs = now
        logger.info(f"  chunk {chunk} done (start={start})")

    truth = trace.decoder_output(layer_idx, 0, total_len).reshape(total_len, n, config.hidden_size)
    _, pcc = comp_pcc(truth.float(), out_accum)
    logger.info(f"[v4 chunked L{layer_idx} {attn_kind} / {mlp_kind}] block output PCC vs golden: {pcc}")

    assert pcc >= _BLOCK_PCC, f"chunked block output PCC {pcc:.6f} < {_BLOCK_PCC}"

    # What the run left in the state, which the output PCC would not see until a later chunk read a
    # corrupted tail. Only HCA and CSA keep a compressed cache; SWA has none, so its rows grade the
    # output alone.
    if attn_kind != "heavily_compressed_attention":
        return

    # The state is replicated, so replica 0 is the whole thing in sequence order, and `capacity`
    # carries tile padding and write headroom past the real entries.
    rate = config.compress_rates[attn_kind]
    entries = -(-total_len // rate)
    device_cache = ttnn.to_torch(
        block.attn_state.compressed_kv, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    ).float()[0, 0, :entries]
    golden_cache = trace.compressed_entries(layer_idx, entries)
    nope = config.head_dim - config.qk_rope_head_dim  # 448, the golden's kv_nope_dim
    # pe_interleave=False: V4's compressed rope is natively interleaved on both sides, so the golden
    # needs no re-basing (build_rope_table repeat_interleaves the table it uploads).
    nope_pcc, pe_pcc = cache_half_pccs(golden_cache, device_cache, nope, pe_interleave=False)
    logger.info(
        f"[v4 chunked L{layer_idx}] compressed cache PCC over {entries} entries: "
        f"nope={nope_pcc:.6f} pe={pe_pcc:.6f}"
    )

    worst_half = min(nope_pcc, pe_pcc)
    assert worst_half >= _CACHE_PCC, f"compressed cache PCC {worst_half:.6f} < {_CACHE_PCC}"


def _mesh_params(payload: int):
    """The 8x4 torus row, with the variant's own fabric payload."""
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=payload,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("layer_idx, attn_kind, mlp_kind", _PRO_CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_block_chunked(mesh_device, device_params, num_links, layer_idx, attn_kind, mlp_kind, n_chunks):
    run_chunked_block_v4(
        mesh_device, device_params, num_links, DeepSeekV4ProConfig, layer_idx, attn_kind, mlp_kind, n_chunks
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _mesh_params(DeepSeekV4FlashConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("layer_idx, attn_kind, mlp_kind", _FLASH_CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_block_chunked(mesh_device, device_params, num_links, layer_idx, attn_kind, mlp_kind, n_chunks):
    run_chunked_block_v4(
        mesh_device, device_params, num_links, DeepSeekV4FlashConfig, layer_idx, attn_kind, mlp_kind, n_chunks
    )
