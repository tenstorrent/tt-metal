# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from models.experimental.alpamayo2_edge.reference.vision import (
    VisionTower,
    vision_cu_seqlens,
    vision_position_embeddings,
)
from models.experimental.alpamayo2_edge.reference.weights import load_weights

PATCH_DIM = 3 * 16 * 16


@pytest.fixture(scope="module")
def visual_weights(model_dir):
    return load_weights(model_dir, "vlm.model.visual.")


@pytest.fixture(scope="module")
def projector_weights(model_dir):
    return load_weights(model_dir, "vlm.model.projector.")


def _tower(cfg, visual_weights, projector_weights, layers):
    return VisionTower(replace(cfg.vision, layers=layers), cfg.projector, visual_weights, projector_weights)


def test_cu_seqlens():
    assert vision_cu_seqlens(torch.tensor([[1, 20, 36], [1, 14, 26]])) == [0, 720, 1084]


def test_cu_seqlens_temporal():
    assert vision_cu_seqlens(torch.tensor([[2, 4, 4]])) == [0, 16, 32]


def test_pos_embed_block_major():
    torch.manual_seed(0)
    pos_weight = torch.randn(256, 1152)
    out = vision_position_embeddings(pos_weight, torch.tensor([[1, 4, 4]]), merge=2)
    assert out.shape == (16, 1152)
    grid = F.interpolate(
        pos_weight.reshape(16, 16, -1).permute(2, 0, 1).unsqueeze(0),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0].permute(1, 2, 0)
    for row, (y, x) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        torch.testing.assert_close(out[row], grid[y, x])
    torch.testing.assert_close(out[4], grid[0, 2])


def test_attention_is_per_image(cfg, visual_weights, projector_weights):
    tower = _tower(cfg, visual_weights, projector_weights, layers=2)
    grid = torch.tensor([[1, 4, 4], [1, 4, 6]])
    torch.manual_seed(0)
    a = torch.randn(16 + 24, PATCH_DIM)
    b = a.clone()
    b[16:] += torch.randn(24, PATCH_DIM)
    with torch.no_grad():
        out_a = tower(a, grid)
        out_b = tower(b, grid)
    assert torch.equal(out_a[:4], out_b[:4])
    assert not torch.equal(out_a[4:], out_b[4:])


def test_tower_shape(cfg, visual_weights, projector_weights):
    tower = _tower(cfg, visual_weights, projector_weights, layers=2)
    grid = torch.tensor([[1, 20, 36]] * 10 + [[1, 14, 26]] * 12)
    n = int(grid.prod(-1).sum())
    assert n == 11568
    patches = torch.randn(n, PATCH_DIM)
    with torch.no_grad():
        feats = tower.encode(patches, grid)
        out = tower.project(feats)
    assert feats.shape == (n, 1152)
    assert out.shape == (2892, 2048)
    assert len(tower.blocks) == 2
