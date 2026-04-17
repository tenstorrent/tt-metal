"""tt-nn MLP for Fast3R encoder/decoder blocks.

Structure matches reference.Mlp: Linear(embed -> hidden) -> GELU -> Linear(hidden -> embed).
"""
from __future__ import annotations

import torch
import ttnn


def to_device_weight(
    device,
    w: torch.Tensor,
    *,
    dtype=ttnn.bfloat8_b,
    transpose: bool = True,
) -> ttnn.Tensor:
    """Upload a torch weight to device in TILE layout.

    torch Linear stores weight as (out, in); ttnn.linear expects (in, out), so we transpose by default.
    """
    if transpose:
        w = w.transpose(-2, -1).contiguous()
    w = w.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) — matches tt-metal convention
    return ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def to_device_bias(device, b: torch.Tensor, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(b.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TtMlp:
    """Holds MLP weights on device. Callable on a (B, N, C) device tensor."""

    def __init__(self, device, fc1_w: torch.Tensor, fc1_b: torch.Tensor, fc2_w: torch.Tensor, fc2_b: torch.Tensor):
        self.fc1_w = to_device_weight(device, fc1_w)
        self.fc1_b = to_device_bias(device, fc1_b)
        self.fc2_w = to_device_weight(device, fc2_w)
        self.fc2_b = to_device_bias(device, fc2_b)

    CORE_GRID = ttnn.CoreGrid(y=10, x=11)
    COMPUTE = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        y = ttnn.linear(
            x, self.fc1_w, bias=self.fc1_b, activation="gelu",
            core_grid=self.CORE_GRID, compute_kernel_config=self.COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
        )
        out = ttnn.linear(
            y, self.fc2_w, bias=self.fc2_b,
            core_grid=self.CORE_GRID, compute_kernel_config=self.COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        y.deallocate(True)
        return out
