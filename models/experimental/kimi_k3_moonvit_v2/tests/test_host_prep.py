# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: `tt.host_prep` (the host-side prep for the device) against the reference modules, and `tt.host_fill` (the same
inputs built fast for the device inputs) against `tt.host_prep`, bit for bit."""

from __future__ import annotations

import pytest
import torch

import numpy as np

from models.experimental.kimi_k3_moonvit_v2.tt import host_fill, host_prep, image_prep
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import load_vision_config
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import (
    MoonViT3dPretrainedModel,
    Rope2DPosEmbRepeated,
    VisionTowerConfig,
    apply_rope,
    tpool_patch_merger,
)
from models.experimental.kimi_k3_moonvit_v2.tests.common import GRIDS, random_patches, random_tensor


@pytest.fixture(scope="module")
def tower():
    torch.manual_seed(0)
    vt_config = VisionTowerConfig(load_vision_config())
    return MoonViT3dPretrainedModel(vt_config).eval()


@torch.no_grad()
def test_patch_embed_is_a_matmul(config, tower):
    pixels, _ = random_patches(config, [(1, 4, 6)])
    weight = tower.patch_embed.proj.weight.reshape(config.hidden_size, config.patch_dim)
    torch.testing.assert_close(
        host_prep.patches_to_rows(pixels) @ weight.T, tower.patch_embed.proj(pixels).flatten(1), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
@torch.no_grad()
def test_pos_embeds(config, tower, grid):
    (_, h, w) = grid[0]
    zeros = torch.zeros(h * w, config.hidden_size)
    expected = tower.patch_embed.pos_emb(zeros, torch.tensor(grid))
    actual = host_prep.pos_embeds(tower.patch_embed.pos_emb.weight, h, w, config.pos_emb_interpolation_mode)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_rope_cos_sin_matches_complex_rope(config, grid):
    (_, h, w) = grid[0]
    q = random_tensor(h * w, config.num_attention_heads, config.head_dim, dtype=torch.float32)
    freqs_cis = Rope2DPosEmbRepeated(config.head_dim, config.max_side_patches, config.max_side_patches).get_freqs_cis(
        torch.tensor(grid), device=torch.device("cpu")
    )
    expected, _ = apply_rope(q, q, freqs_cis)

    cos, sin = host_prep.rope_cos_sin(config, h, w)
    actual = host_prep.apply_rope(q.transpose(0, 1), cos, sin).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rotate_pairs_is_the_device_trans_mat(config):
    trans_mat = host_prep.rot_transformation_mat()[0, 0]  # one 32x32 tile, applied per tile of head_dim
    x = random_tensor(5, config.head_dim, dtype=torch.float32)
    tiled = (x.unflatten(-1, (-1, 32)) @ trans_mat).flatten(-2)
    torch.testing.assert_close(host_prep.rotate_pairs(x), tiled)


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
def test_window_order_merges_by_reshape(config, grid):
    """Window order makes the merger a reshape of consecutive rows into the reference's tokens."""
    (_, h, w) = grid[0]
    x = random_tensor(h * w, config.hidden_size, dtype=torch.float32)
    (expected,) = tpool_patch_merger(x, torch.tensor(grid), merge_kernel_size=config.merge_kernel_size)
    window = host_prep.to_window_order(x, h, w, config.merge_kernel_size)
    torch.testing.assert_close(window.reshape(-1, config.merge_area, config.hidden_size), expected)
    torch.testing.assert_close(host_prep.from_window_order(window, h, w, config.merge_kernel_size), x)


def test_prepare_host_inputs(config, tower):
    h, w = 40, 100
    pixels, grid_thws = random_patches(config, [(1, h, w)])
    inputs = host_prep.prepare_host_inputs(config, tower.patch_embed.pos_emb.weight, pixels, grid_thws)
    n, rows = h * w, config.max_patches
    assert inputs["logical_n"] == n
    assert inputs["patches"].shape == (rows, config.padded_patch_dim)
    assert not inputs["patches"][:, config.patch_dim :].any()
    assert inputs["pos_emb"].shape == (rows, config.hidden_size)
    assert inputs["cos"].shape == inputs["sin"].shape == (rows, config.head_dim)
    for key in ("patches", "pos_emb", "cos", "sin"):
        assert not inputs[key][n:].any(), key

    # Every per-row input shares the window order.
    cos, sin = host_prep.rope_cos_sin(config, h, w)
    torch.testing.assert_close((inputs["cos"], inputs["sin"]), host_prep.rope_tables(config, h, w))
    raster = {
        "patches": host_prep.patch_rows(config, pixels),
        "pos_emb": host_prep.pos_embeds(tower.patch_embed.pos_emb.weight, h, w, config.pos_emb_interpolation_mode),
        "cos": cos,
        "sin": sin,
    }
    for key, value in raster.items():
        torch.testing.assert_close(inputs[key][:n], host_prep.to_window_order(value, h, w, config.merge_kernel_size))


@pytest.mark.parametrize("extra", [-1, 1], ids=["fewer", "more"])
def test_prepare_host_inputs_rejects_wrong_row_count(config, tower, expect_error, extra):
    h, w = 4, 6
    pixels, _ = random_patches(config, [(1, h * w + extra, 1)])
    with expect_error(ValueError, "pixel_values"):
        host_prep.prepare_host_inputs(config, tower.patch_embed.pos_emb.weight, pixels, torch.tensor([[1, h, w]]))


# Grids for host_fill: window order needs h and w to be multiples of 2, interpolation is skipped only at 64x64, and the
# extremes of the side limit stress the RoPE tables.
FILL_GRIDS = [(2, 2), (6, 20), (20, 6), (64, 64), (100, 6), (2, 512), (512, 2), (148, 148)]


@pytest.fixture(scope="module")
def fill_weight(config):
    return torch.randn(64, 64, config.hidden_size, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("h,w", FILL_GRIDS, ids=[f"{h}x{w}" for h, w in FILL_GRIDS])
def test_host_fill_matches_prepare_host_inputs(config, fill_weight, h, w):
    """Every input host_fill builds is prepare_host_inputs' (before its padding), bit for bit, and the patches are
    also bit for bit after the cast to the device's bf16."""
    pixels = np.random.default_rng(h * 1000 + w).integers(0, 256, (h * 14, w * 14, 3), dtype=np.uint8)
    pixel_values, grid_thws = image_prep.pixels_to_patches(pixels, config)
    expected = host_prep.prepare_host_inputs(config, fill_weight, pixel_values, grid_thws)
    n = h * w
    for dtype in (torch.bfloat16, torch.float32):
        want = expected["patches"][:n].to(dtype)
        from_pixels = torch.zeros(n, config.padded_patch_dim, dtype=dtype)
        host_fill.normalized_patches_into(from_pixels, pixels, config)
        assert torch.equal(from_pixels, want), f"normalized_patches_into, {dtype}"
        from_values = torch.zeros(n, config.padded_patch_dim, dtype=dtype)
        host_fill.patch_values_into(from_values, pixel_values, h, w, config)
        assert torch.equal(from_values, want), f"patch_values_into, {dtype}"
    pos = host_fill.PosEmbedTable(fill_weight, config).window_rows(h, w)
    assert torch.equal(pos, expected["pos_emb"][:n])
    cos, sin = host_fill.rope_window_rows(config, h, w)
    assert torch.equal(cos, expected["cos"][:n]) and torch.equal(sin, expected["sin"][:n])


def test_pos_embed_table_falls_back_for_other_weights(config):
    """A weight the batched interpolation cannot take (non-contiguous here) goes through pos_embeds itself."""
    weight = torch.randn(config.hidden_size, 64, 64).permute(1, 2, 0)
    table = host_fill.PosEmbedTable(weight, config)
    assert not table.batched
    expected = host_prep.to_window_order(
        host_prep.pos_embeds(weight, 6, 20, config.pos_emb_interpolation_mode), 6, 20, config.merge_kernel_size
    )
    assert torch.equal(table.window_rows(6, 20), expected)
