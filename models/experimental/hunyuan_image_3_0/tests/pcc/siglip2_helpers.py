# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared constants/helpers for SigLIP2 TTNN PCC tests (tests/pcc/).
# Fixtures live in conftest.py; this module is safe to import from tests.

import os

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import VIT_CONFIG

# Default 1 layer for fast device runs; set HY_VIT_NUM_LAYERS=27 for full stack.
NUM_LAYERS = int(os.environ.get("HY_VIT_NUM_LAYERS", "1"))
B, S = 1, 64
PCC_THR = float(os.environ.get("HY_VIT_PCC_THR", "0.99"))
SPATIAL_SHAPES_HW = ((8, 8),)


def spatial_shapes_to_hw(spatial_shapes: torch.Tensor) -> tuple[tuple[int, int], ...]:
    return tuple((int(spatial_shapes[i][0]), int(spatial_shapes[i][1])) for i in range(spatial_shapes.shape[0]))


def upload_pixel_values(device, pixel_values: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_attention_mask(device, pixel_attention_mask: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        pixel_attention_mask.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_smoke_vision_inputs():
    """Host-side synthetic processor outputs (fixed seed; weights are real)."""
    torch.manual_seed(0)
    patch_dim = VIT_CONFIG["num_channels"] * VIT_CONFIG["patch_size"] ** 2
    pixel_values = torch.randn(B, S, patch_dim, dtype=torch.float32)
    spatial_shapes = torch.tensor([[8, 8]], dtype=torch.long)
    pixel_attention_mask = torch.ones(B, S, dtype=torch.long)
    pixel_attention_mask[0, 48:] = 0  # 48 valid, 16 padded
    return pixel_values, spatial_shapes, pixel_attention_mask
