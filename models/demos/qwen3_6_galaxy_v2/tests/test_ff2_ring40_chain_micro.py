# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-test isolating the FF2 chain of the fused ring-40 decode MLP.

The fused ``_fuse_rs`` branch in ``tt/llama_decoder.py::_mlp_decode_qwen36``
(env ``QWEN36_FUSE_RS_MATMUL=1``, default OFF) passes FF12 + the FF2
``all_gather_matmul`` then crashes on the FINAL non-fused collective:
``TT_FATAL: Unsupported memory layout for output tensor
TensorMemoryLayout::INTERLEAVED`` at
``tt_ccl.line_all_reduce(w2_out_ring, cluster_axis=0,
memory_config=ttnn.DRAM_MEMORY_CONFIG, ...)`` — because ``w2_out_ring``
(the all_gather_matmul output) is L1 WIDTH-SHARDED
(``REDUCE_SCATTER_OUT_RING40_MEMCFG``) but the all-reduce is asked for a
DRAM-INTERLEAVED output, and the all-reduce decomposition does not support
L1-sharded-in -> interleaved-out.

The unfused else-branch works because its ``w2_out`` is DRAM-interleaved
before the all-reduce. FIX (mirror the unfused branch exactly): convert the
fused FF2 matmul output to DRAM-interleaved before ``line_all_reduce``.

This drives ONLY the FF2 chain exactly as ``_fuse_rs`` does:
  synthetic SwiGLU output (M=32, 640-wide, REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG)
  -> reshard to the AG input layout (30-core WIDTH_SHARDED [32,32])
  -> tt_ccl.all_gather_matmul(... w2_ring40 L1 ..., mm_memory_config=
     REDUCE_SCATTER_OUT_RING40_MEMCFG, program_config=FF2_RING40_PROGCFG, ...)
  -> tt_ccl.line_all_reduce(cluster_axis=0, ...)

The definitive RED is the full 1L fused run
(``QWEN36_FUSE_RS_MATMUL=1 ... profile_decode_eager.py``), which FATALs at
this exact all-reduce. In isolation the no-fix path
(``QWEN36_FF2_CHAIN_NO_FIX=1``) is CCL-state-dependent: the decode
``all_reduce_async`` may tolerate the sharded-in/interleaved-out for a short
call sequence, so the no-fix branch here is a diagnostic toggle, not the
primary gate. With the fix (default) it asserts PCC > 0.99 vs torch (gather
640 col-shards -> 2560, matmul w2 -> 1280, sum over cluster_axis=0 rows =
all-reduce) — confirming the DRAM-interleaved round-trip preserves the math.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff2_ring40_chain_micro.py -s -x
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_CLUSTER_SHAPE = (8, 4)
_CLUSTER_AXIS = 1


@pytest.mark.hardware
def test_ff2_ring40_chain_micro(bh_glx_mesh):
    mesh_device = bh_glx_mesh

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",  # ensures the decode-mode ring-40 weights are built
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    pattern = ["linear_attention"]
    state_dict = _load_full_state_dict(_SNAPSHOT)
    out = {}
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    for k, v in state_dict.items():
        if k.startswith(all_pfx):
            if k.startswith(pfx0):
                out[k] = v
            else:
                continue
        else:
            out[k] = v
    state_dict = out

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, pattern, 1, paged_attention_config)

    mlp = model.layers[0].feed_forward
    mc = model.model_config
    num_links = mc["GALAXY_NUM_LINKS"]
    fused_ck = args.compute_kernel_config_hifi2

    M = 32
    dim_per_tp = args.dim_per_tp  # 1280
    ff_n_ring40 = mc["FF_N_RING40"]  # 2560
    swiglu_per_col = ff_n_ring40 // _CLUSTER_SHAPE[_CLUSTER_AXIS]  # 640

    # ---- Synthetic SwiGLU output: M=32, 640-wide, per (row,col) device.
    # This is exactly what the fused branch hands to the FF2 chain: each device
    # holds its own 640-col slice (post FF12 reduce-scatter across cols). The
    # all_gather (cluster_axis=1) gathers the 4 col-shards -> 2560 then matmuls
    # w2 (K=2560, N=1280); line_all_reduce sums over the 8 rows.
    torch.manual_seed(0)
    swiglu_full = torch.randn(*_CLUSTER_SHAPE, M, swiglu_per_col)  # [8,4,32,640]
    ff_ring = ttnn.from_torch(
        swiglu_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=mc["REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG"],
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )
    print(f"[ff2_chain] ff_ring (SwiGLU out, 640-wide) shape = {list(ff_ring.shape)} dtype={ff_ring.dtype}")
    print(f"[ff2_chain] mlp.w2_ring40 shape = {list(mlp.w2_ring40.shape)} dtype={mlp.w2_ring40.dtype}")

    # JIT-reshard the DRAM ring-40 w2 to L1 WIDTH-SHARDED over the 40 ring cores.
    w2_ring40_l1 = ttnn.to_memory_config(mlp.w2_ring40, mc["W2_RING40_L1_MEMCFG"])

    # ---- Reshard SwiGLU output to the AG input layout (exactly as _fuse_rs) ----
    ag_in_crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0),
        30,
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        ),
        row_wise=True,
    )
    ag_in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ag_in_crs, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ag_out_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mc["RING40_AG_INTERIM_CRS"], [32, 640], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ff_ag_in = ttnn.to_memory_config(ff_ring, ag_in_memcfg)

    # ---- FF2 fused all_gather_matmul (ring-40, PCC-validated wrapper) ----
    w2_out_ring = mlp.tt_ccl.all_gather_matmul(
        ff_ag_in,
        w2_ring40_l1,
        dim=3,
        cluster_axis=_CLUSTER_AXIS,
        num_links=num_links,
        ag_memory_config=ag_out_memcfg,
        mm_memory_config=mc["REDUCE_SCATTER_OUT_RING40_MEMCFG"],
        program_config=mc["FF2_RING40_PROGCFG"],
        compute_kernel_config=fused_ck,
        dtype=ttnn.bfloat8_b,
        global_cb=None,
        buffer_key="FF2_AG_MM_CHAIN_MICRO",
    )
    ttnn.synchronize_device(mesh_device)
    print(
        f"[ff2_chain] w2_out_ring shape = {list(w2_out_ring.shape)} memcfg layout = {w2_out_ring.memory_config().memory_layout}"
    )

    # ---- FIX: convert the fused FF2 matmul output to DRAM-interleaved BEFORE
    # the all-reduce (mirror the unfused else-branch's DRAM-interleaved w2_out).
    # WITHOUT this line the line_all_reduce raises the
    # "Unsupported memory layout ... INTERLEAVED" FATAL (the RED).
    # Set QWEN36_FF2_CHAIN_NO_FIX=1 to skip the fix and reproduce the RED FATAL.
    if os.environ.get("QWEN36_FF2_CHAIN_NO_FIX", "0") == "1":
        w2_dram = w2_out_ring  # L1 width-sharded -> RED: Unsupported INTERLEAVED out
    else:
        w2_dram = ttnn.to_memory_config(w2_out_ring, ttnn.DRAM_MEMORY_CONFIG)
        w2_out_ring.deallocate(True)

    # ---- Final FF2 collective: all_reduce over rows (cluster_axis=0) -> DRAM ----
    w2_red = mlp.tt_ccl.line_all_reduce(
        w2_dram,
        cluster_axis=0,
        num_links=min(3, num_links),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        buffer_key="FF2",
        batch_size=1,
    )
    if w2_dram is not w2_out_ring:
        w2_dram.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    print(f"[ff2_chain] w2_red shape = {list(w2_red.shape)} memcfg layout = {w2_red.memory_config().memory_layout}")

    # ---- Torch golden ----
    # Per device, recover the on-device w2 weight slice [2560, 1280] (it is the
    # SAME across cluster_axis=1 cols — replicated K-pad? no: ShardTensor2dMesh
    # splits per (row,col); recover it per device to avoid re-deriving the split).
    # all-gather along cluster_axis=1: each col contributes its 640 slice ->
    # gathered A = concat over cols -> [8, 32, 2560]; then A @ w2[row,col] is the
    # SAME A for all cols in a row, producing a per-(row,col) partial that is then
    # summed over the 8 rows (cluster_axis=0).
    w2_dev = [ttnn.get_device_tensors(mlp.w2_ring40)[i].cpu().to_torch().float() for i in range(32)]
    w2_dev = [w.reshape(w.shape[-2], w.shape[-1]) for w in w2_dev]  # [2560, 1280] each

    # gathered activation per row = concat of the 4 cols' 640 slices.
    gathered = {}
    for r in range(_CLUSTER_SHAPE[0]):
        cols = [swiglu_full[r, c] for c in range(_CLUSTER_SHAPE[1])]  # each [32,640]
        gathered[r] = torch.cat(cols, dim=-1)  # [32, 2560]

    # per-(row,col) matmul partial, then sum over rows = all-reduce result (same on
    # every row of a given col). Compare each device's output to the col's row-sum.
    # First compute per col the sum over rows: sum_r gathered[r] @ w2_dev[r*4+c].
    col_allreduce = {}
    for c in range(_CLUSTER_SHAPE[1]):
        acc = None
        for r in range(_CLUSTER_SHAPE[0]):
            w = w2_dev[r * _CLUSTER_SHAPE[1] + c]  # [2560,1280]
            part = gathered[r] @ w  # [32,1280]
            acc = part if acc is None else acc + part
        col_allreduce[c] = acc  # [32,1280]

    pcc = None
    for i, t in enumerate(ttnn.get_device_tensors(w2_red)):
        r = i // _CLUSTER_SHAPE[1]
        c = i % _CLUSTER_SHAPE[1]
        got = t.cpu().to_torch().float().reshape(M, -1)[:, :dim_per_tp]
        ref = col_allreduce[c]
        eq, pcc = comp_pcc(got, ref)
        assert eq, f"device {i} (row {r}, col {c}) FAILED FF2-chain PCC: {pcc}"
    print(f"[ff2_chain] FF2-chain all-reduce PCC OK across all 32 devices; last msg: {pcc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
