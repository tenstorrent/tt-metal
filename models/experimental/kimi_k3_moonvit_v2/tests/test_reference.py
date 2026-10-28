# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the vendored reference, and the config and grid checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel, load_vision_config
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import tpool_patch_merger
from models.experimental.kimi_k3_moonvit_v2.tests.common import GRIDS, SHALLOW_REFERENCE, random_patches
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig, check_grid

REFERENCE_CONFIG = Path(__file__).parents[1] / "reference" / "config.json"


def test_default_config_is_k3():
    assert MoonViTConfig.from_hf_config(REFERENCE_CONFIG) == MoonViTConfig()
    assert MoonViTConfig().head_dim == 128
    assert MoonViTConfig().patch_dim == 588
    assert MoonViTConfig().padded_patch_dim == 608
    assert MoonViTConfig().merged_dim == 4096


@pytest.mark.parametrize("vision_config", [{}, {"pos_emb_interpolation_mode": "bicubic", "attn_bias": True}])
def test_from_hf_config_defaults_match_reference(tmp_path, vision_config):
    """Keys a config.json leaves out resolve the way the reference model builds."""
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": vision_config}))
    config = MoonViTConfig.from_hf_config(tmp_path)
    tower = KimiK3VisionModel(load_vision_config(tmp_path, num_hidden_layers=1)).vision_tower
    block = tower.encoder.blocks[0]
    assert config.pos_emb_interpolation_mode == tower.patch_embed.pos_emb.interpolation_mode
    assert config.patch_embed_proj_bias == (tower.patch_embed.proj.bias is not None)
    assert config.attn_bias == (block.wqkv.bias is not None)
    assert config.linear_bias == (block.mlp.fc0.bias is not None)
    assert config.qkv_hidden_size == block.wqkv.out_features // 3


@SHALLOW_REFERENCE
@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
@torch.no_grad()
def test_forward_shape(shallow_config, shallow_reference, grid):
    pixels, grid_thws = random_patches(shallow_config, grid)
    pixels = pixels.to(torch.bfloat16)
    n = pixels.shape[0] // shallow_config.merge_area
    (tower_out,) = shallow_reference.vision_tower(pixels, grid_thws)
    assert tower_out.shape == (n, shallow_config.merge_area, shallow_config.hidden_size)
    (out,) = shallow_reference(pixels, grid_thws)
    assert out.shape == (n, shallow_config.text_hidden_size)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_patch_merger_order():
    """Token (i, j) holds patches (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) of the raster order."""
    h, w = 4, 6
    x = torch.arange(h * w, dtype=torch.float32)[:, None]
    (merged,) = tpool_patch_merger(x, torch.tensor([[1, h, w]]))
    expected = [
        [r * w + c, r * w + c + 1, (r + 1) * w + c, (r + 1) * w + c + 1] for r in range(0, h, 2) for c in range(0, w, 2)
    ]
    assert merged[..., 0].tolist() == expected


def test_from_hf_config_rejects_null_qkv_hidden_size(expect_error):
    with expect_error(ValueError, "qkv_hidden_size"):
        MoonViTConfig.from_hf_config({"vision_config": {"qkv_hidden_size": None}})


@pytest.mark.parametrize(
    "grid, message",
    [
        ([(1, 8, 8), (1, 4, 4)], "one image"),
        ([(2, 8, 8)], "video"),
        ([(1, 7, 8)], "merge kernel"),
        ([(1, 2, 514)], "exceeds"),
        ([(1, 512, 130)], "exceeds"),
    ],
    ids=["two_images", "video", "odd", "too_wide", "too_many"],
)
def test_rejects_unsupported_grids(config, expect_error, grid, message):
    with expect_error(ValueError, message):
        check_grid(torch.tensor(grid), config)
