# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TEMP profile + correctness harness (uncommitted): neighborhood attention, op vs gather.

Feeds the SAME q/k/v to both device backends and to the na3d_torch host reference. Asserts each
matches the reference (correctness), then times the warmed-up executor (perf). Single chip,
replicated — isolates the attention kernel/algorithm difference, which is where the ~1000× lives.
Add "fused" as a third backend value later.
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.tt_dit.layers.na3d import na3d_torch, neighborhood_attention_3d
from models.tt_dit.utils.check import assert_quality


def _timeit(fn, mesh, iters=3):
    o = fn()
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / iters


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "grid, kernel, heads, head_dim",
    [
        ((4, 4, 4), (3, 3, 3), 1, 64),
        ((6, 8, 8), (3, 5, 5), 2, 64),
        ((13, 68, 120), (11, 11, 11), 4, 64),
    ],
    ids=["tiny", "mid", "small"],
)
def test_na3d_fused(*, mesh_device, grid, kernel, heads, head_dim):
    """Fused-gather NA3D backend, standalone correctness (B2.2 bring-up). Feeds the same q/k/v to the
    fused backend and the na3d_torch reference; asserts PCC 0.999. Tiny grids iterate fast on device."""
    T, H, W = grid
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    ref = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    q_tt, k_tt, v_tt = (
        ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    out = neighborhood_attention_3d(q_tt, k_tt, v_tt, kernel_size=kernel, scale=1.0, backend="fused")
    assert tuple(out.shape) == tuple(ref.shape), f"{tuple(out.shape)} != {tuple(ref.shape)}"
    assert_quality(ref, ttnn.to_torch(out).float(), pcc=0.999)
    ttnn.deallocate(out)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("backend", ["op", "gather", "fused"], ids=["op", "gather", "fused"])
@pytest.mark.parametrize("grid", [(13, 68, 120), (13, 136, 240)], ids=["small", "big4x"])
def test_na3d_op_vs_gather(*, mesh_device, backend, grid):
    T, H, W = grid
    heads, head_dim = 4, 64
    kernel = (11, 11, 11)
    N = T * H * W
    kw = tuple(min(k, d) for k, d in zip(kernel, grid))
    window = kw[0] * kw[1] * kw[2]

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    ref = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    q_tt, k_tt, v_tt = (
        ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )

    def run():
        return neighborhood_attention_3d(q_tt, k_tt, v_tt, kernel_size=kernel, scale=1.0, backend=backend)

    # Correctness: this backend vs the host neighborhood-attention reference.
    out = run()
    assert tuple(out.shape) == tuple(ref.shape), f"{tuple(out.shape)} != {tuple(ref.shape)}"
    assert_quality(ref, ttnn.to_torch(out).float(), pcc=0.999)
    ttnn.deallocate(out)

    # Perf: warmed-up executor time.
    t = _timeit(run, mesh_device)
    ideal_tflop = N * heads * window * 2 * head_dim * 2 / 1e12  # (QK + AV) MACs * 2 flop/MAC
    print(
        f"\n[na3d {backend}] grid={grid} N={N:,} window={window}: "
        f"{t * 1000:8.1f} ms  ({t * 1e9 / N:7.1f} ns/site)  "
        f"ideal {ideal_tflop:.2f} TFLOP -> {ideal_tflop / t:6.2f} TFLOP/s effective\n"
    )
