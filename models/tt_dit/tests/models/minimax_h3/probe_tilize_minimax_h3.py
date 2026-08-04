# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Why does the same tilize cost 2.8 ms in one place and 21.3 ms in another?

Block 0's profile has seven ``TilizeDeviceOperation`` calls with byte-identical input specs
-- ``(17,1,65536,128)`` bf16 ROW_MAJOR DRAM_INTERLEAVED -> TILE -- costing
``[21.3, 2.8, 21.3, 20.8, 5.2, 21.3, 20.9]`` ms, reproducibly across iterations. Standalone
(``probe_datamovement``) the same tilize is **3.0 ms at 190 GB/s**, so the 21 ms figure is the
anomaly, not the 3 ms one, and it is 26 % of the block.

The only variable left is the *producer* of the input tensor. This times the tilize against
inputs made four different ways.
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.conv_minimax_h3 import MiniMaxH3CausalConv3d

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
T, H, W, C = 17, 256, 256, 128
ITERS = 5


def _best(fn, device) -> float:
    out = fn()
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
        ttnn.deallocate(out)
    return best * 1e3


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_tilize_producer(mesh_device):
    torch.manual_seed(0)

    fresh = ttnn.from_torch(
        torch.randn(T, 1, H * W, C), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    print(f"\n  {'from_torch':34s} {_best(lambda: ttnn.to_layout(fresh, ttnn.TILE_LAYOUT), mesh_device):8.2f} ms")
    print(f"  {'from_torch, ttnn.tilize':34s} {_best(lambda: ttnn.tilize(fresh), mesh_device):8.2f} ms")

    # A real conv3d output, which is what the slow sites tilize.
    conv = MiniMaxH3CausalConv3d(C, C, kernel_size=3, spatial_padding=1, mesh_device=mesh_device, dtype=ttnn.bfloat16)
    conv.load_torch_state_dict({"weight": torch.randn(C, C, 3, 3, 3) * 0.02, "bias": torch.zeros(C)})
    x = ttnn.from_torch(
        torch.randn(1, T, H, W, C), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    conv_out = conv(x)
    flat = ttnn.reshape(conv_out, (T, 1, H * W, C))
    print(
        f"  {'conv3d out -> reshape':34s} {_best(lambda: ttnn.to_layout(flat, ttnn.TILE_LAYOUT), mesh_device):8.2f} ms"
    )
    print(f"  {'conv3d out, ttnn.tilize':34s} {_best(lambda: ttnn.tilize(flat), mesh_device):8.2f} ms")

    # The same bytes copied into a freshly allocated tensor.
    cloned = ttnn.clone(flat)
    print(
        f"  {'conv3d out -> clone':34s} {_best(lambda: ttnn.to_layout(cloned, ttnn.TILE_LAYOUT), mesh_device):8.2f} ms"
    )

    # Tilize the 5D conv output directly, then reshape in TILE (which is free).
    def five_d():
        return ttnn.reshape(ttnn.to_layout(conv_out, ttnn.TILE_LAYOUT), (T, 1, H * W, C))

    print(f"  {'conv3d out 5D -> tilize -> rs':34s} {_best(five_d, mesh_device):8.2f} ms", flush=True)

    # Does DRAM occupancy explain it? In-model, block 0 has ~1 GB live at the residual add
    # (x, h, the 324 MB padded conv input, the conv output), and the same tilize measures
    # 21 ms there against 3 ms here.
    ballast = []
    for step_mb in (300, 600, 900, 1200):
        while sum(285 for _ in ballast) < step_mb:
            ballast.append(
                ttnn.from_torch(
                    torch.randn(T, 1, H * W, C),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )
        ms = _best(lambda: ttnn.to_layout(fresh, ttnn.TILE_LAYOUT), mesh_device)
        print(f"  {f'+{len(ballast) * 285} MB live':34s} {ms:8.2f} ms", flush=True)
