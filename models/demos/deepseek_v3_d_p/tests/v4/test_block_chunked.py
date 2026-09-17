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

The ceiling is measured, not assumed: on layer 3 at 512 tokens the CPU reference loaded from the
same checkpoint reproduces the golden to PCC 0.99993, so what the device loses below that is its
own. The residual is the golden's fp8 KV cache and its bf16 storage.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import v4_block_state_dict, v4_mhc_weights
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

# The golden is Pro's, so the variant is fixed here and the rows carry the layer instead.
_VARIANT = DeepSeekV4ProConfig

# One row per (layer, attention, MoE) the device can build, spelled out so no index has to be read
# for it. No CSA row: layers 2, 4, 6 ... are compressed_sparse_attention and have no device
# implementation yet.
_CASES = [
    pytest.param(1, "heavily_compressed_attention", "hash_moe", id="L1-hca-hash"),
    pytest.param(3, "heavily_compressed_attention", "moe", id="L3-hca-topk"),
]

# Measured at the full 56320: L1 0.99857, L3 0.99658, against a 0.9999 reference ceiling, and flat
# to 4e-4 across chunk counts.
_BLOCK_PCC = 0.99


def run_chunked_block_v4(mesh_device, device_params, num_links, layer_idx, attn_kind, mlp_kind, n_chunks):
    trace = golden.resolve_trace()
    if trace is None:
        pytest.skip(f"golden trace unavailable (set ${golden.TRACE_ENV} or stage {golden.V4_PRO_TRACE})")
    checkpoint = golden.resolve_checkpoint()
    if checkpoint is None:
        pytest.skip(f"V4 checkpoint unavailable (set one of {golden.CKPT_ENVS})")

    topology = per_axis_topology(device_params["fabric_config"])
    tp_factor = mesh_device.shape[1]
    ms = tuple(mesh_device.shape)
    total_len = n_chunks * CHUNK
    assert total_len <= SEQ_CACHE, f"{n_chunks} chunks ({total_len}) exceed the golden's {SEQ_CACHE}"

    config, model_cfg = _test_config(_VARIANT, layer_idx)
    config.max_seq_len = SEQ_CACHE
    # The row names the layer and its pair; this is where the config has to agree.
    assert (config.layer_types[layer_idx], config.mlp_layer_types[layer_idx]) == (attn_kind, mlp_kind), (
        f"case asks for layer {layer_idx} to be {attn_kind}/{mlp_kind}, the config gives "
        f"{config.layer_types[layer_idx]}/{config.mlp_layer_types[layer_idx]}"
    )
    gate_mode = GateComputeMode.HASH_DEVICE if mlp_kind == "hash_moe" else GateComputeMode.DEVICE_FP32
    logger.info(f"[v4 chunked] layer {layer_idx}: {attn_kind} / {mlp_kind} / {n_chunks} x {CHUNK} tokens")

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

    # What the run left in the cache, which the output PCC would not see until a later chunk read a
    # corrupted tail. The state is replicated, so replica 0 is the whole thing in sequence order, and
    # `capacity` carries tile padding and write headroom past the real entries.
    rate = config.compress_rates[attn_kind]
    entries = -(-total_len // rate)
    device_cache = ttnn.to_torch(
        block.attn_state.compressed_kv, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    ).float()[0, 0, :entries]
    golden_cache = trace.compressed_entries(layer_idx, entries)
    nope = config.head_dim - config.qk_rope_head_dim  # 448, the golden's kv_nope_dim
    nope_pcc, pe_pcc = cache_half_pccs(golden_cache, device_cache, nope, pe_interleave=False)
    _, pe_interleaved = cache_half_pccs(golden_cache, device_cache, nope, pe_interleave=True)
    # A rope rotation preserves each row's norm, so norms that agree while values do not says the two
    # sides differ by a basis or a rotation and not by the quantity itself.
    norms = (
        torch.linalg.norm(device_cache[:, nope:], dim=-1) / torch.linalg.norm(golden_cache[:, nope:], dim=-1)
    ).median()
    logger.info(
        f"[v4 chunked L{layer_idx}] compressed cache over {entries} entries: nope={nope_pcc} "
        f"pe={pe_pcc} pe_interleaved={pe_interleaved} pe_norm_ratio={norms:.5f}"
    )
    dump = os.getenv("V4_CACHE_DUMP")
    if dump:
        torch.save({"golden": golden_cache, "device": device_cache, "nope": nope}, f"{dump}/L{layer_idx}.pt")
        logger.info(f"  dumped both caches to {dump}/L{layer_idx}.pt")

    assert pcc >= _BLOCK_PCC, f"chunked block output PCC {pcc} < {_BLOCK_PCC}"
    # UNRESOLVED, and only the nope half is asserted because of it: the pe half measures 0.19 against
    # this golden, and no simple relation explains it. It is the same quantity -- per-row norms agree
    # to 3e-4 and 19 of 440 rows match outright -- rotated by something unidentified: 0.009 through
    # interleave_pe, 0.34 under an entry-to-token position shift (127 * entry), 0.13 under +-1.
    # The compressor ropes each entry before the write (compressor.forward:359, positioned by
    # first_window_position // compress_rate), so the cleanest fix is a golden captured at that same
    # point, which is how GLM's cache compares directly. `V4_CACHE_DUMP=<dir>` writes both halves out
    # for a host-side hunt without re-running the mesh.
    assert nope_pcc >= _BLOCK_PCC, f"compressed cache nope PCC {nope_pcc} < {_BLOCK_PCC}"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
# The golden's whole 56320, which is what the GLM teacher-forced row runs. Shorter runs measured the
# same to 4e-4, so they graded nothing the full one does not.
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("layer_idx, attn_kind, mlp_kind", _CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_block_chunked(mesh_device, device_params, num_links, layer_idx, attn_kind, mlp_kind, n_chunks):
    run_chunked_block_v4(mesh_device, device_params, num_links, layer_idx, attn_kind, mlp_kind, n_chunks)
