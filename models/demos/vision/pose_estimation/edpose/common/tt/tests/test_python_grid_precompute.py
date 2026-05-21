# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test: Python bilinear grid precompute vs C++ prepare_grid_sample_grid.

Verifies bit-level equivalence of the vectorized Python replacement.
"""

import time

import torch
import ttnn


def cpp_precompute(grid_l, H, W, D, N_M, Lq, P):
    grid_host = ttnn.from_torch(grid_l, dtype=ttnn.float32)
    precomputed = ttnn.prepare_grid_sample_grid(
        grid_host, [N_M, H, W, D],
        mode="bilinear", padding_mode="zeros", align_corners=False,
        output_dtype=ttnn.bfloat16,
    )
    precomputed_t = ttnn.to_torch(precomputed)
    return precomputed_t.reshape(N_M, Lq, 1, 6 * P)


def python_precompute(grid_l, H, W, N_M, Lq, P):
    x = grid_l[..., 0]
    y = grid_l[..., 1]

    h_coord = (y + 1.0) * (H * 0.5) - 0.5
    w_coord = (x + 1.0) * (W * 0.5) - 0.5

    h0 = torch.floor(h_coord)
    w0 = torch.floor(w_coord)

    h_frac = h_coord - h0
    w_frac = w_coord - w0

    h0i = h0.to(torch.int32)
    w0i = w0.to(torch.int32)

    h0v = (h0i >= 0) & (h0i < H)
    h1v = ((h0i + 1) >= 0) & ((h0i + 1) < H)
    w0v = (w0i >= 0) & (w0i < W)
    w1v = ((w0i + 1) >= 0) & ((w0i + 1) < W)

    hfi = 1.0 - h_frac
    wfi = 1.0 - w_frac

    wt_nw = (hfi * wfi * (h0v & w0v).float()).to(torch.bfloat16)
    wt_ne = (hfi * w_frac * (h0v & w1v).float()).to(torch.bfloat16)
    wt_sw = (h_frac * wfi * (h1v & w0v).float()).to(torch.bfloat16)
    wt_se = (h_frac * w_frac * (h1v & w1v).float()).to(torch.bfloat16)

    h0_bf16 = h0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)
    w0_bf16 = w0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)

    packed = torch.stack([h0_bf16, w0_bf16, wt_nw, wt_ne, wt_sw, wt_se], dim=-1)
    return packed.reshape(N_M, Lq, 1, 6 * P)


def test_equivalence():
    N_M = 8
    Lq = 1024
    P = 4
    test_levels = [(100, 152), (50, 76), (25, 38), (13, 19), (7, 10)]
    D = 32

    print("=== Python vs C++ grid precompute equivalence test ===\n")

    for H, W in test_levels:
        grid_l = torch.randn(N_M, Lq, P, 2) * 0.5
        grid_l = grid_l.clamp(-1.0, 1.0)

        cpp_out = cpp_precompute(grid_l, H, W, D, N_M, Lq, P)
        py_out = python_precompute(grid_l, H, W, N_M, Lq, P)

        cpp_raw = cpp_out.contiguous().view(torch.uint16)
        py_raw = py_out.contiguous().view(torch.uint16)

        match = (cpp_raw == py_raw).all().item()
        mismatch_count = (cpp_raw != py_raw).sum().item()
        total = cpp_raw.numel()

        print(f"  Level ({H:3d}×{W:3d}): bit-exact={match}, mismatches={mismatch_count}/{total}")

        if not match:
            diff_idx = (cpp_raw != py_raw).nonzero(as_tuple=True)
            for i in range(min(5, mismatch_count)):
                flat = diff_idx[0][i] if len(diff_idx) == 1 else sum(d[i] for d in diff_idx)
                idx = flat.item()
                print(f"    idx={idx}: cpp=0x{cpp_raw.flatten()[idx].item():04x} py=0x{py_raw.flatten()[idx].item():04x}")

    print()


def test_timing():
    N_M = 8
    Lq = 80997
    P = 4
    test_levels = [(100, 152), (50, 76), (25, 38), (13, 19), (7, 10)]
    D = 32

    print("=== Timing comparison (full encoder-scale) ===\n")

    for H, W in test_levels:
        grid_l = torch.randn(N_M, Lq, P, 2).clamp(-1.0, 1.0)

        # Warm up
        python_precompute(grid_l, H, W, N_M, Lq, P)

        t0 = time.perf_counter()
        for _ in range(3):
            cpp_precompute(grid_l, H, W, D, N_M, Lq, P)
        cpp_ms = (time.perf_counter() - t0) / 3 * 1000

        t0 = time.perf_counter()
        for _ in range(3):
            python_precompute(grid_l, H, W, N_M, Lq, P)
        py_ms = (time.perf_counter() - t0) / 3 * 1000

        print(f"  Level ({H:3d}×{W:3d}): C++={cpp_ms:.1f}ms  Python={py_ms:.1f}ms  speedup={cpp_ms/py_ms:.2f}x")

    print()


if __name__ == "__main__":
    test_equivalence()
    test_timing()
    print("Done.")
