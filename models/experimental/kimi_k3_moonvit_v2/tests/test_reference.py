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
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    GRIDS,
    ROW_GRIDS,
    SHALLOW_REFERENCE,
    MeshRows,
    device_config,
    random_patches,
    row_grid,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig, check_grid, resolve_parallel

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


@ROW_GRIDS
def test_row_grids(config, size):
    """Each row-wise device grid is a valid image, and its shape and the quarter one P150 runs of it
    are what its name promises."""
    (grid,) = row_grid(size)
    h, w = check_grid(torch.tensor([grid]), config)
    chip = MeshRows(h * w, config.max_patches, None)
    real = chip.real.stop - chip.real.start
    last_chip = config.max_patches - chip.share
    if size in ("full", "max_side"):
        assert h * w == config.max_patches
        assert (chip.start, real) == (last_chip, chip.share)
    if size == "max_side":
        assert max(h, w) == config.max_side_patches
    if size == "partial":
        assert chip.start == last_chip and real % 32 and real < chip.share
    if size == "small":
        assert chip.start == 0 and real == h * w < chip.share


def test_mesh_rows():
    """On the ring every row; on one chip the quarter of the last ring chip with real rows."""
    x = torch.arange(10.0)[:, None]
    ring = MeshRows(10, 16, sp_axis=1)
    assert torch.equal(ring.take(x), x)
    assert torch.equal(ring.held(x)[:10], x) and not ring.held(x)[10:].any()
    chip = MeshRows(10, 16, sp_axis=None)
    assert (chip.share, chip.start, chip.real) == (4, 8, slice(8, 10))
    assert torch.equal(chip.take(x), x[8:10])
    assert torch.equal(chip.held(x), torch.tensor([[8.0], [9.0], [0.0], [0.0]]))
    # An already padded tensor gives the same rows.
    assert torch.equal(chip.held(torch.nn.functional.pad(x, (0, 0, 0, 6))), chip.held(x))
    assert MeshRows(16, 16, None).start == 12 and MeshRows(3, 16, None).start == 0


def test_one_device_runs_at_most_a_ring_chips_share(config, expect_error):
    resolve_parallel(None, None, None, device_config(config, None))
    with expect_error(ValueError, "one ring chip's share"):
        resolve_parallel(None, None, None, config)
