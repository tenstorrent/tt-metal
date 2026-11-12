# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host tests that the vision modules' builders reject weights, modules or configs they cannot run, before any upload.
They are in a file of their own because the device tests ask ttnn for the number of chips when they are collected.
"""
from __future__ import annotations

import torch

from models.demos.minimax_m3.tt.vision.attention import _pad_wo_weight
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed


def test_vision_mlp_rejects_mismatched_weights(expect_error):
    """M3VLMLP raises when fc2 does not take fc1's outputs or a bias does not fit its weight."""
    fc1_weight, fc1_bias, fc2_weight, fc2_bias = torch.zeros(16, 8), torch.zeros(16), torch.zeros(8, 16), torch.zeros(8)
    for weights, message in (
        ((fc1_weight, fc1_bias, torch.zeros(8, 12), fc2_bias), "does not take the 16 outputs of fc1.weight"),
        ((fc1_weight, torch.zeros(12), fc2_weight, fc2_bias), "fc1.bias shape"),
        ((fc1_weight, fc1_bias, fc2_weight, torch.zeros(12)), "fc2.bias shape"),
    ):
        with expect_error(AssertionError, message):
            M3VLMLP(None, *weights)


def test_vision_patch_embed_rejects_other_modules(expect_error):
    """from_torch rejects a module that is not one matmul per patch, such as a Linear or a Conv3d with a bias."""
    kernel = (2, 14, 14)
    for module in (
        torch.nn.Linear(1176, 8, bias=False),
        torch.nn.Conv3d(3, 8, kernel, stride=kernel, bias=True),
        torch.nn.Conv3d(3, 8, kernel, stride=(1, 14, 14), bias=False),
    ):
        with expect_error(AssertionError, "expected a Conv3d with stride equal to its kernel and no bias"):
            M3VLPatchEmbed.from_torch(None, module)


def test_vision_layernorm_rejects_other_modules(expect_error):
    """M3VLLayerNorm rejects a weight or bias of another shape, and from_torch a module that is not a 1-d LayerNorm."""
    dim = 8
    weight, bias = torch.ones(dim), torch.zeros(dim)
    for weights, message in (
        ((torch.ones(1, dim), bias), r"weight must be 1D of size 8, got \(1, 8\)"),
        ((torch.ones(dim + 1), bias), r"weight must be 1D of size 8, got \(9,\)"),
        ((weight, torch.zeros(1, dim)), r"bias must be 1D of size 8, got \(1, 8\)"),
    ):
        with expect_error(AssertionError, message):
            M3VLLayerNorm(None, dim, *weights)
    with expect_error(AssertionError, "expected nn.LayerNorm, got Linear"):
        M3VLLayerNorm.from_torch(None, torch.nn.Linear(dim, dim))
    with expect_error(NotImplementedError, "only supports 1-d normalized_shape"):
        M3VLLayerNorm.from_torch(None, torch.nn.LayerNorm((4, dim)))


def test_vision_attention_rejects_other_out_proj(expect_error):
    """The output projection's weight must take num_heads * head_dim inputs, which the heads are padded from."""
    with expect_error(AssertionError, r"wo weight in dim 161 != 2 \* 80"):
        _pad_wo_weight(torch.zeros(8, 2 * 80 + 1), num_heads=2, head_dim=80, padded_head_dim=96)


def test_vision_model_rejects_other_merge_size(expect_error):
    """M3VLVisionModel, and from_state_dict before it checks the weights, raise ValueError for a spatial_merge_size
    other than image_input.MERGE_SIZE (2)."""
    config = MiniMaxM3VisionConfig(spatial_merge_size=3)
    message = "spatial_merge_size must be 2, as image_input counts tokens with it, got 3"
    with expect_error(ValueError, message):
        M3VLVisionModel(
            None,
            config,
            padded_head_dim=96,
            patch_embed=None,
            pre_layrnorm=None,
            blocks=[],
        )
    with expect_error(ValueError, message):
        M3VLVisionModel.from_state_dict(None, {}, config)
