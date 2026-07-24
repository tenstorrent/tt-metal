# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Full-dimension SigLIP2 PCC: max_num_patches=1024 (32×32) + full 27-layer vision stack.
# Smoke suite in test_siglip2_ttnn.py stays at S=64 / 1L.
#
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_siglip2_full_dim.py -v -s --timeout=10800

from __future__ import annotations

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import (
    ALIGNER_CONFIG,
    VIT_CONFIG,
    load_aligner,
    load_siglip2_vision,
)
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    HunyuanTtLightProjector,
    HunyuanTtSiglip2Vision,
    Siglip2VisionInputs,
    forward_vision_with_aligner,
)

# Full HF / processor dimensions (config.json vit_processor.max_num_patches + vit layers).
FULL_LAYERS = int(os.environ.get("HY_VIT_NUM_LAYERS", str(VIT_CONFIG["num_hidden_layers"])))
FULL_S = int(os.environ.get("HY_VIT_SEQ", "1024"))  # max_num_patches
FULL_HW = (int(FULL_S**0.5), int(FULL_S**0.5))
assert FULL_HW[0] * FULL_HW[1] == FULL_S, f"HY_VIT_SEQ={FULL_S} must be a perfect square"
PCC_THR = float(os.environ.get("HY_VIT_PCC_THR", "0.99"))
B = 1


def assert_pcc(ref: torch.Tensor, tt: torch.Tensor, *, label: str) -> float:
    passing, pcc = comp_pcc(ref, tt, PCC_THR)
    logger.info(f"PCC [{label}]: {pcc}  (threshold {PCC_THR}, passing={passing})")
    assert passing, f"PCC [{label}] {pcc} < {PCC_THR}"
    return pcc


def to_torch_squeezed(t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(t)
    while out.ndim > 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out


@pytest.fixture(scope="function")
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
def full_vision_inputs():
    """Host inputs at processor max_num_patches (1024) with a 32×32 spatial layout."""
    torch.manual_seed(0)
    patch_dim = VIT_CONFIG["num_channels"] * VIT_CONFIG["patch_size"] ** 2
    pixel_values = torch.randn(B, FULL_S, patch_dim, dtype=torch.float32)
    spatial_shapes = torch.tensor([list(FULL_HW)], dtype=torch.long)
    # Keep most patches valid; zero a trailing tile-aligned pad block to exercise mask.
    pixel_attention_mask = torch.ones(B, FULL_S, dtype=torch.long)
    pixel_attention_mask[0, FULL_S - 32 :] = 0
    return pixel_values, spatial_shapes, pixel_attention_mask


@pytest.fixture(scope="module")
def ref_vision_full(model_dir):
    return load_siglip2_vision(model_dir, num_layers=FULL_LAYERS)


@pytest.fixture(scope="module")
def ref_aligner(model_dir):
    return load_aligner(model_dir)


@pytest.mark.slow
def test_vision_full_dim_masked_pcc(device, ref_vision_full, full_vision_inputs):
    """Full 27L vision @ S=1024 (32×32 patches) vs fp32 ref."""
    pixel_values, spatial_shapes, pixel_attention_mask = full_vision_inputs
    with torch.no_grad():
        pt_out = ref_vision_full(pixel_values, pixel_attention_mask, spatial_shapes)

    sd = ref_vision_full.state_dict()
    tt_mod = HunyuanTtSiglip2Vision(device, sd, num_layers=FULL_LAYERS)
    tt_mod.prewarm_pos_geometries([(FULL_HW[0], FULL_HW[1], FULL_S)])
    inputs = Siglip2VisionInputs.create(
        ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        (FULL_HW,),
        ttnn.from_torch(
            pixel_attention_mask.to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    tt_out = to_torch_squeezed(tt_mod(inputs))
    assert pt_out.shape == tt_out.shape == (B, FULL_S, VIT_CONFIG["hidden_size"])
    assert_pcc(pt_out, tt_out, label=f"vision_full_dim_masked_{FULL_LAYERS}L_S{FULL_S}")


@pytest.mark.slow
def test_e2e_vision_aligner_full_dim_pcc(device, ref_vision_full, ref_aligner, full_vision_inputs):
    """Full vision+aligner @ S=1024 / 27L (output embed dim = backbone H=4096)."""
    pixel_values, spatial_shapes, pixel_attention_mask = full_vision_inputs
    with torch.no_grad():
        pt_out = ref_aligner(ref_vision_full(pixel_values, pixel_attention_mask, spatial_shapes))

    vision_tt = HunyuanTtSiglip2Vision(device, ref_vision_full.state_dict(), num_layers=FULL_LAYERS)
    vision_tt.prewarm_pos_geometries([(FULL_HW[0], FULL_HW[1], FULL_S)])
    aligner_tt = HunyuanTtLightProjector(device, ref_aligner.state_dict())
    inputs = Siglip2VisionInputs.create(
        ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        (FULL_HW,),
        ttnn.from_torch(
            pixel_attention_mask.to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    tt_out = to_torch_squeezed(forward_vision_with_aligner(vision_tt, aligner_tt, inputs))
    assert pt_out.shape == tt_out.shape == (B, FULL_S, ALIGNER_CONFIG["n_embed"])
    assert_pcc(pt_out, tt_out, label=f"e2e_vision_aligner_full_dim_{FULL_LAYERS}L_S{FULL_S}")
