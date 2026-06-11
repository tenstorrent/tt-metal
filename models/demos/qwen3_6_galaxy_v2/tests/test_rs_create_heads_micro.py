# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-test for the batch-DP QKVG user-scatter (seam A).

The fused llama_rs_create_heads kernel hardcodes head_dim=128/q_heads=8, so it
mis-scatters qwen's head_dim=256/6-slot QKVG (verified: scrambled per-user
output). Seam A therefore uses a PLAIN reduce_scatter(cluster_axis=1, dim=2) to
(a) complete the cross-column hidden sum AND (b) scatter the 32 users -> 8/col,
then reuses qwen's existing flat split/QK-norm/RoPE.

This validates that reduce_scatter exactly:
  1. SUMS across the 4 columns (identical input on each col → 4× output), and
  2. SCATTERS the user dim (32 → 8/col) preserving per-user identity (identical
     input users → identical 8 output users — the property that was BROKEN by
     the fused op, maxdiff=16).

No model build needed → ~20 s.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate \\
      && python -m pytest --noconftest -v -s \\
         models/demos/qwen3_6_galaxy_v2/tests/test_rs_create_heads_micro.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn

_N_USERS = 32
_TOTAL_PC = 2048  # qwen per-chip QKVG width (3Q+3G+1K+1V) × 256
_N_COLS = 4
_SLICE = _N_USERS // _N_COLS  # 8 users/col


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
def test_reduce_scatter_user_scatter(bh_glx_mesh):
    # Input [1,1,32,2048] with all 32 user-rows IDENTICAL, REPLICATED across the
    # whole mesh (so every column holds the same data → reduce sums to 4×).
    torch.manual_seed(0)
    one_user = torch.randn(1, 1, 1, _TOTAL_PC, dtype=torch.bfloat16)
    x_torch = one_user.expand(1, 1, _N_USERS, _TOTAL_PC).contiguous()
    x = ttnn.from_torch(
        x_torch,
        device=bh_glx_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    # Seam A's exact op: reduce_scatter on the user dim across the 4 columns.
    out = ttnn.reduce_scatter(
        x,
        dim=2,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Read column 0's device-0 shard: [1,1,8,2048] (this col's 8 users).
    o0 = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(_SLICE, _TOTAL_PC)
    print(f"[rs] out dev0 shape (8 users × 2048): absmean={o0.abs().mean():.4f} absmax={o0.abs().max():.4f}")

    # (2) per-user identity: all 8 output users identical (the property the fused op BROKE).
    u0 = o0[0]
    maxdiff = max((o0[u] - u0).abs().max().item() for u in range(1, _SLICE))
    print(f"[rs] per-user maxdiff vs user0 = {maxdiff:.5f}  (must be ~0)")

    # (1) reduce sum: identical input replicated on 4 cols → output ≈ 4 × input.
    exp = 4.0 * one_user.float().reshape(_TOTAL_PC)
    rel = (u0 - exp).abs().max().item() / (exp.abs().max().item() + 1e-6)
    print(
        f"[rs] user0 vs 4×input rel-maxdiff = {rel:.4f}  (out absmax={u0.abs().max():.3f} exp absmax={exp.abs().max():.3f})"
    )

    assert maxdiff < 1e-2, f"output users DIFFER (maxdiff={maxdiff}) — scatter bug"
    assert rel < 0.05, f"reduce sum wrong (rel={rel}) — expected ~4× (bf16 tol)"
    print("[rs] PASS — reduce_scatter sums 4 cols and scatters 32→8/col with per-user identity")
