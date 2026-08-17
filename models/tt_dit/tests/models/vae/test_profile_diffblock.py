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
@pytest.mark.parametrize("grid, kernel", [((4, 4, 4), (3, 3, 3)), ((8, 8, 8), (5, 5, 5))], ids=["s4", "s8"])
def test_fused_q_offset(*, mesh_device, grid, kernel):
    """Fused NA3D with a nonzero windowed_q_token_offset (the W-SP crux): run the fused kernel on a Q
    SUB-BAND (rows [off, S)) against FULL K/V, telling it its global offset, and check it matches the
    na3d_torch reference's corresponding rows. If this holds, spatial-W SP is just plumbing."""
    T, H, W = grid
    heads, head_dim = 2, 64
    S = T * H * W
    off = S // 2
    assert off % 32 == 0, "offset must be tile-aligned"
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    ref = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, S, heads * head_dim)  # [1, S, width]

    def kv_wrow(x):  # full K/V -> [1, heads, T*H, W*head_dim] ROW_MAJOR (fused's W-row page layout)
        x = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(ttnn.reshape(x, (1, S, heads, head_dim)), (0, 2, 1, 3))  # [1, heads, S, hd]
        return ttnn.reshape(x, (1, heads, T * H, W * head_dim))

    k_tt, v_tt = kv_wrow(k), kv_wrow(v)
    q_full = ttnn.from_torch(q, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    q_full = ttnn.permute(ttnn.reshape(q_full, (1, S, heads, head_dim)), (0, 2, 1, 3))  # [1, heads, S, hd]
    q_band = ttnn.to_layout(ttnn.slice(q_full, [0, 0, off, 0], [1, heads, S, head_dim]), ttnn.TILE_LAYOUT)

    grid_dev = mesh_device.compute_with_storage_grid_size()
    attended = ttnn.transformer.scaled_dot_product_attention(
        q_band,
        k_tt,
        v_tt,
        is_causal=False,
        neighborhood_3d=(T, H, W, *kernel),
        neighborhood_gather=True,
        windowed_q_token_offset=off,
        scale=1.0,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_dev.x, grid_dev.y),
            exp_approx_mode=False,
            q_chunk_size=32,
            k_chunk_size=32,
        ),
    )
    got = ttnn.to_torch(attended).float().permute(0, 2, 1, 3).reshape(1, S - off, heads * head_dim)
    assert_quality(ref[:, off:, :], got, pcc=0.999)


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
