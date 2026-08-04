# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Why are the encoder's data-movement ops so slow, and can the shape be chosen better?

After the norm rewrite, block 0's profile is Conv3d 121 ms, **Tilize 114 ms**, Concat 88 ms.
Two things say those are not simply bandwidth-bound:

* block 1's tensors are half block 0's size, yet its Tilize total is the same (104 ms);
* 285 MB in + 285 MB out in ~16 ms is ~36 GB/s, far under what DRAM should give.

So the cost is likely in *how the work is spread*, which the logical shape controls. This
times the same bytes under different shapes, and times ``slice_write`` against ``concat``
for the padding, which is the other 20 % of the block.
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
ITERS = 5

# The two shapes that dominate: block 0 (17,256,256,128) and block 1 (17,128,128,256).
CASES = [(17, 256, 256, 128), (17, 128, 128, 256)]


def _best(fn, device) -> float:
    fn()
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
        ttnn.deallocate(out)
    return best * 1e3


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_tilize_shape(mesh_device):
    """Same bytes, four logical shapes. The norm needs (T,1,H*W,C) tiles, but it can get
    there by tilizing a different view and reshaping for free (that reshape is 0.02 ms)."""
    for T, H, W, C in CASES:
        mb = T * H * W * C * 2 / 1e6
        print(f"\n=== tilize {T}x{H}x{W}x{C}  ({mb:.0f} MB) ===", flush=True)
        for label, shape in [
            ("(T,1,HW,C)   <- shipping", (T, 1, H * W, C)),
            ("(1,1,T*HW,C)", (1, 1, T * H * W, C)),
            ("(T,H,W,C)", (T, H, W, C)),
            ("(1,T,H,W,C)", (1, T, H, W, C)),
        ]:
            x = ttnn.from_torch(
                torch.randn(*shape), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ms = _best(lambda: ttnn.to_layout(x, ttnn.TILE_LAYOUT), mesh_device)
            print(f"  {label:26s} {ms:8.2f} ms   {2 * mb / ms:7.1f} GB/s", flush=True)
            ttnn.deallocate(x)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_pad_strategy(mesh_device):
    """Three reflect-pad concats per conv is three full copies of the activation.
    ``slice_write`` into one preallocated padded buffer should be one."""
    for T, H, W, C in CASES:
        mb = T * H * W * C * 2 / 1e6
        print(f"\n=== pad {T}x{H}x{W}x{C}  ({mb:.0f} MB) ===", flush=True)
        x = ttnn.from_torch(
            torch.randn(1, T, H, W, C), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # (a) shipping: slice the edges, concat W, concat H, concat T.
        def concat_path():
            left = ttnn.slice(x, [0, 0, 0, 1, 0], [1, T, H, 2, C])
            right = ttnn.slice(x, [0, 0, 0, W - 2, 0], [1, T, H, W - 1, C])
            wide = ttnn.concat([left, x, right], dim=3)
            top = ttnn.slice(wide, [0, 0, 1, 0, 0], [1, T, 2, W + 2, C])
            bot = ttnn.slice(wide, [0, 0, H - 2, 0, 0], [1, T, H - 1, W + 2, C])
            tall = ttnn.concat([top, wide, bot], dim=2)
            zeros = ttnn.zeros(
                (1, 2, H + 2, W + 2, C), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
            return ttnn.concat([zeros, tall], dim=1)

        ms = _best(concat_path, mesh_device)
        print(f"  {'3x concat (shipping)':26s} {ms:8.2f} ms", flush=True)

        # (b) one full-size write into a preallocated padded buffer, plus the small borders.
        # slice_write needs rank 4, so fold (1,T,...) -> (T,...).
        x4 = ttnn.reshape(x, (T, H, W, C))

        def slice_write_path():
            out = ttnn.zeros(
                (T + 2, H + 2, W + 2, C), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
            out = ttnn.experimental.slice_write(x4, out, [2, 1, 1, 0], [T + 2, H + 1, W + 1, C], [1, 1, 1, 1])
            return out

        try:
            ms = _best(slice_write_path, mesh_device)
            print(f"  {'slice_write interior':26s} {ms:8.2f} ms", flush=True)
        except Exception as exc:
            print(f"  {'slice_write interior':26s} FAILED {str(exc)[:100]}", flush=True)
        ttnn.deallocate(x)
