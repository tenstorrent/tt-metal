# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer: catch the chunked-prefill op-to-op (host-dispatch) gap for Kimi K2.6 under Tracy.

This is a heavily trimmed-down cousin of test_kimi_prefill_transformer_chunked_no_pcc (see
test_prefill_transformer_chunked.py::run_chunked_transformer_no_pcc). Everything unrelated to
*capturing an op2op trace* has been cut: no golden-trace / PCC comparison, no KV-depth preload, no
perf-baseline gating, no chunk-timing table, no host-side MLA/FFN timing breakdown, no
DeepSeek/GLM/padded-chunk variants. It:

  1. builds a `num_layers`-layer Kimi chunked-prefill transformer (default 2 = one dense + one MoE
     layer -- the smallest config that exercises BOTH regions),
  2. drives ONE 5120-token chunk for exactly two iterations (iteration 0 = cold / JIT-compiles every
     kernel + populates the program cache; iteration 1 = warm -- the one to inspect), and
  3. brackets each iteration with `iter_{i}_start` / `iter_{i}_end` Tracy signposts.

The model itself unconditionally emits `forward_layer_{i}_start` / `forward_layer_{i}_end`
(tt_prefill_transformer.py) and `MLA_START` / `MLA_END` (mla.py) signposts on every forward call, so
no extra instrumentation is needed to get a per-layer / MLA-vs-FFN breakdown of the warm iteration's
op2op gap:

    python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op <tracy_output_dir> --iter 1

Requires an 8x4 Blackhole mesh and (env from the task):
    TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/moonshotai/Kimi-K2.6-Cache
    KIMI_K2_6_HF_MODEL=/mnt/models/moonshotai/Kimi-K2.6

Reproduce (context: https://github.com/tenstorrent/tt-metal/issues/50932). --op-support-count can
stay tiny here (2 layers x 1 chunk x 2 iters is nowhere near the profiler-buffer overflow that forces
20000 on the full L61 sweep):

    python -m tracy -v -r --sync-host-device --op-support-count 2000 \\
        -o generated/profiler/kimi_op2op_min \\
        -m pytest "models/demos/deepseek_v3_d_p/tests/test_kimi_prefill_op2op_min.py::test_kimi_prefill_op2op_min[kimi-mesh-8x4-L2]" \\
        -s --timeout=0

    python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op generated/profiler/kimi_op2op_min --iter 1
"""

import gc
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

CHUNK = 5 * 1024  # 5120 tokens per chunk (matches test_prefill_transformer_chunked.CHUNK)
SEQ_CACHE = 2 * CHUNK  # smallest slab-aligned KV cache that can hold one chunk starting at kv_actual=0


@pytest.mark.parametrize(
    "num_layers", [2], ids=["L2"]
)  # layer 0 dense + layer 1 MoE: smallest config with both regions
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                # MoE routing all-gather needs a small L1_SMALL region for its global semaphores; see
                # test_kimi_prefill_transformer_chunked for the rationale.
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_prefill_op2op_min(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    num_links,
    topology,
):
    """Build a `num_layers`-layer Kimi chunked-prefill transformer and run ONE 5120-token chunk for
    two iterations under Tracy, so the warm iteration's op2op gap can be inspected with
    perlayer_op2op.py. No PCC, no golden trace, no perf gate -- purely a Tracy capture target."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    config = config_only

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"
    chunk_local = CHUNK // sp

    config.max_seq_len = SEQ_CACHE
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    vocab_size = config.vocab_size

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    logger.info(f"building {num_layers}-layer Kimi chunked-prefill transformer (mesh={mesh_shape})")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,  # per-chunk size -> MoE/FFN dispatch buffers
        max_seq_len=SEQ_CACHE,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=True,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    # Deterministic in-vocab tokens for the one chunk-aligned position (kv_actual=0), laid out
    # block-cyclic chip-major and SP-sharded on dim 0. No golden trace / correctness check needed --
    # this repro only cares about op dispatch timing, not numerics.
    token_ids = torch.arange(CHUNK, dtype=torch.int64) % vocab_size
    positions = rotated_chip_positions(0, sp, chunk_local)
    flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
    chunk_tok_host = token_ids[flat].reshape(sp, 1, chunk_local)

    mesh_device.enable_program_cache()

    for it in range(2):  # iter 0 = cold (JIT compile), iter 1 = warm -- inspect this one (--iter 1)
        tt_tokens = ttnn.from_torch(
            chunk_tok_host,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )
        # Device-idle boundary for the profiler post-processor: sync, then bracket the region with
        # signposts so perlayer_op2op.py can isolate this iteration's ops.
        ttnn.synchronize_device(mesh_device)
        signpost(f"iter_{it}_start")
        transformer.forward(
            tt_tokens,
            tt_kvpe_cache,
            actual_isl=CHUNK,
            actual_start=0,
            actual_end=CHUNK,
            cache_user_id=0,
            return_intermediates=False,
        )
        ttnn.synchronize_device(mesh_device)
        signpost(f"iter_{it}_end")
        ttnn.deallocate(tt_tokens)
        # Flush the on-device profiler buffer between iterations so it can't overflow across iters
        # (irrelevant at this tiny op count, but kept for parity with the full no-PCC sweep).
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(mesh_device)
        logger.info(f"iter {it} done")

    logger.success(f"Kimi op2op-min repro done (num_layers={num_layers}); inspect with perlayer_op2op.py --iter 1")
