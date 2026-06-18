# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared fixtures for SigLIP2 TTNN PCC tests (tests/vision/).
# Weights: real HunyuanImage checkpoint (same as VAE tests via ref/weights.MODEL_DIR).
# Host inputs are torch (ref golden); TTNN inputs are uploaded once in tt_vision_inputs.

import os

import pytest
import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import (
    VIT_CONFIG,
    load_aligner,
    load_siglip2_vision,
)
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import Siglip2VisionInputs

# Default 1 layer for fast device runs; set HY_VIT_NUM_LAYERS=27 for full stack.
NUM_LAYERS = int(os.environ.get("HY_VIT_NUM_LAYERS", "1"))
B, S = 1, 64
PCC_THR = float(os.environ.get("HY_VIT_PCC_THR", "0.99"))
SPATIAL_SHAPES_HW = ((8, 8),)


def _spatial_shapes_to_hw(spatial_shapes: torch.Tensor) -> tuple[tuple[int, int], ...]:
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


@pytest.fixture(scope="session")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model_dir():
    index = MODEL_DIR / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"Hunyuan checkpoint not found at {MODEL_DIR} (set HUNYUAN_MODEL_DIR)")
    return MODEL_DIR


@pytest.fixture(scope="module")
def vision_inputs():
    """Host-side synthetic processor outputs (fixed seed; weights are real)."""
    torch.manual_seed(0)
    patch_dim = VIT_CONFIG["num_channels"] * VIT_CONFIG["patch_size"] ** 2
    pixel_values = torch.randn(B, S, patch_dim, dtype=torch.float32)
    spatial_shapes = torch.tensor([[8, 8]], dtype=torch.long)
    pixel_attention_mask = torch.ones(B, S, dtype=torch.long)
    pixel_attention_mask[0, 48:] = 0  # 48 valid, 16 padded
    return pixel_values, spatial_shapes, pixel_attention_mask


@pytest.fixture
def tt_vision_inputs(device, vision_inputs):
    """On-device Siglip2VisionInputs (host upload happens in test infra only)."""
    pixel_values, spatial_shapes, pixel_attention_mask = vision_inputs
    return Siglip2VisionInputs.create(
        upload_pixel_values(device, pixel_values),
        _spatial_shapes_to_hw(spatial_shapes),
        upload_attention_mask(device, pixel_attention_mask),
    )


@pytest.fixture(scope="module")
def ref_vision(model_dir):
    return load_siglip2_vision(model_dir, num_layers=NUM_LAYERS)


@pytest.fixture(scope="module")
def ref_aligner(model_dir):
    return load_aligner(model_dir)


@pytest.fixture(scope="module")
def vision_state_dict(ref_vision):
    return ref_vision.state_dict()


@pytest.fixture(scope="module")
def aligner_state_dict(ref_aligner):
    return ref_aligner.state_dict()
