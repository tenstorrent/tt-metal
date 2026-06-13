# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tradeoff probe: col-REPLICATED decode MLP vs col-SHARDED, on the 32-chip galaxy.

Decode is CCL-bound (~3x compute at M=32). The col-axis (cluster_axis=1, 4 cols)
collectives exist only because the hidden dim is col-SHARDED (dim_per_tp=1280).
If we REPLICATE the hidden dim across the 4 cols (cols redundant), those col-axis
collectives vanish — at the cost of 4x redundant matmul work (free-ish at M=32).

This measures the tradeoff directly on the MLP:

  SHARDED (current decode path):
    w1/w3 (K=1280/col) -> line_reduce_scatter(cols) -> line_all_gather(cols)
    -> w2 -> line_all_reduce(rows)               # 3 CCLs (2 col + 1 row), K=1280 matmuls
  REPLICATED (proposed):
    w1/w3 LOCAL (K=5120 full) -> SwiGLU -> w2 LOCAL -> line_all_reduce(rows)
                                                  # 1 CCL (row only), K=5120 matmuls (4x)

WIN if (4x matmul time increase) < (2 col-CCLs saved). Both compared vs the same
torch MLP for PCC (synthetic weights — timing is layout-dependent, not value-dependent).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_mlp_replicated_vs_sharded_micro.py -s -x
"""
from __future__ import annotations

import os
import time

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

_CLUSTER = (8, 4)  # (rows, cols)
_ITERS = int(os.environ.get("QWEN36_TRADEOFF_ITERS", "20"))


def _time(fn, mesh, iters=_ITERS):
    fn()  # warmup / compile
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e6 / iters  # us/iter


@pytest.mark.hardware
def test_mlp_replicated_vs_sharded(bh_glx_mesh):
    mesh = bh_glx_mesh
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
    }.items():
        os.environ.setdefault(_k, _v)
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    sd = _load_full_state_dict(_SNAPSHOT)
    pfx0 = "model.language_model.layers.0."
    allp = "model.language_model.layers."
    sd = {k: v for k, v in sd.items() if (not k.startswith(allp)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh, sd, ["linear_attention"], 1, paged)
    model.switch_mode("decode")
    tt_ccl = model.layers[0].feed_forward.tt_ccl
    ck = args.compute_kernel_config_hifi2

    M = 32
    dim = args.dim  # 5120
    inter = args.intermediate_dim  # 17408
    rows, cols = _CLUSTER
    inter_per_row = inter // rows  # 2176

    torch.manual_seed(0)
    w1 = torch.randn(dim, inter) * 0.02
    w3 = torch.randn(dim, inter) * 0.02
    w2 = torch.randn(inter, dim) * 0.02
    x = torch.randn(1, 1, M, dim) * 0.5
    ref = (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2  # [1,1,32,5120]

    # ===================== REPLICATED layout =====================
    # x replicated everywhere; w1/w3 row-shard inter (over 8 rows), replicate dim over cols;
    # w2 row-shard inter-K (over 8 rows), replicate dim-N over cols.
    x_rep = ttnn.from_torch(
        x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _shard_rep(t, shard_dim):  # shard `shard_dim` over rows (axis0), replicate over cols (axis1)
        return ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(shard_dim, None), mesh_shape=_CLUSTER),
        )

    w1_rep = _shard_rep(w1, -1)  # [5120, 2176] per chip
    w3_rep = _shard_rep(w3, -1)
    w2_rep = _shard_rep(w2, -2)  # [2176, 5120] per chip

    out_rep_holder = {}

    def rep_mlp():
        h1 = ttnn.linear(
            x_rep, w1_rep, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        h3 = ttnn.linear(
            x_rep, w3_rep, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sg = ttnn.mul(
            h1,
            h3,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h1.deallocate(True)
        h3.deallocate(True)
        op = ttnn.linear(
            sg, w2_rep, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sg.deallocate(True)
        out = tt_ccl.line_all_reduce(op, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        op.deallocate(True)
        out_rep_holder["o"] = out
        return out

    t_rep = _time(rep_mlp, mesh)
    o = out_rep_holder["o"]
    got = ttnn.get_device_tensors(o)[0].cpu().to_torch().float().reshape(M, dim)
    eq, msg = comp_pcc(got, ref.reshape(M, dim))
    print(
        f"[tradeoff] REPLICATED: {t_rep:8.1f} us/iter   PCC {msg.split('PCC:')[-1].strip()}   (1 CCL row-all_reduce, K=5120 matmuls)",
        flush=True,
    )

    # ===================== SHARDED layout (current unfused decode pattern) =====================
    # x col-sharded (dim over cols); w1/w3 dims=(-1,-2) [inter over rows, dim-K over cols];
    # w2 dims=(-2,-1) [inter-K over rows, dim-N over cols]. CCL: col RS + col gather + row all_reduce.
    x_sh = ttnn.from_torch(
        x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=_CLUSTER),
    )  # dim over cols

    def _shard_2d(t, d0, d1):
        return ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(d0, d1), mesh_shape=_CLUSTER),
        )

    w1_sh = _shard_2d(w1, -1, -2)  # [1280, 2176] per chip
    w3_sh = _shard_2d(w3, -1, -2)
    w2_sh = _shard_2d(w2, -2, -1)  # [2176, 1280] per chip

    out_sh_holder = {}

    def sh_mlp():
        h1 = ttnn.linear(
            x_sh, w1_sh, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        h3 = ttnn.linear(
            x_sh, w3_sh, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        h1r = tt_ccl.line_reduce_scatter(h1, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dim=3)
        h3r = tt_ccl.line_reduce_scatter(h3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dim=3)
        h1.deallocate(True)
        h3.deallocate(True)
        sg = ttnn.mul(
            h1r,
            h3r,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h1r.deallocate(True)
        h3r.deallocate(True)
        gg = tt_ccl.line_all_gather(sg, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dim=3)
        sg.deallocate(True)
        op = ttnn.linear(
            gg, w2_sh, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gg.deallocate(True)
        out = tt_ccl.line_all_reduce(op, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        op.deallocate(True)
        out_sh_holder["o"] = out
        return out

    t_sh = _time(sh_mlp, mesh)
    osh = out_sh_holder["o"]
    # sharded out is col-sharded dim (1280/col); gather to compare
    got_sh = torch.cat(
        [ttnn.get_device_tensors(osh)[c].cpu().to_torch().float().reshape(M, -1) for c in range(cols)], dim=-1
    )
    eq_sh, msg_sh = comp_pcc(got_sh[:, :dim], ref.reshape(M, dim))
    print(
        f"[tradeoff] SHARDED:    {t_sh:8.1f} us/iter   PCC {msg_sh.split('PCC:')[-1].strip()}   (3 CCLs: 2 col + 1 row, K=1280 matmuls)",
        flush=True,
    )

    print(
        f"\n[tradeoff] dims: M={M} dim={dim} inter={inter} rows={rows} cols={cols} inter/row={inter_per_row}",
        flush=True,
    )
    print(
        f"[tradeoff] === REPLICATED {t_rep:.0f} us  vs  SHARDED {t_sh:.0f} us  ->  {'REPLICATED WINS' if t_rep < t_sh else 'SHARDED WINS'} (delta {t_rep - t_sh:+.0f} us) ===",
        flush=True,
    )
    print("[tradeoff] (replicated: 4x weight-DRAM-read, 1 CCL; sharded: 1/4 weight read, 3 CCLs)", flush=True)
    assert eq, f"replicated MLP PCC failed: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
