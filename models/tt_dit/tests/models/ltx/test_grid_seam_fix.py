# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host unit tests for the I2V grid-seam fix: the VAE conv halo seams the latent at the 2x4
mesh boundaries when H/W don't divide evenly across the mesh (e.g. the stage-1 cond/upsampler
at 17x30). The fix replicate-pads to even shards and crops back. These tests lock the pad/crop
math; the device-level seam check lives in the I2V integration test."""

import torch

from models.tt_dit.utils.ltx import ceil_to, pad_hw_replicate


def test_ceil_to_rounds_up_to_multiple():
    assert ceil_to(17, 2) == 18  # uneven s1 latent height over h_factor=2
    assert ceil_to(30, 4) == 32  # uneven s1 latent width over w_factor=4
    assert ceil_to(16, 2) == 16 and ceil_to(28, 4) == 28  # already even -> unchanged
    assert ceil_to(34, 2) == 34 and ceil_to(60, 4) == 60  # full-res latent already even


def test_pad_hw_replicate_makes_even_shards_and_reports_true_dims():
    x = torch.randn(1, 128, 5, 17, 30)  # s1 latent shape (uneven over the 2x4 mesh)
    padded, h, w = pad_hw_replicate(x, 2, 4)
    assert (h, w) == (17, 30)  # true dims returned for the crop-back
    assert padded.shape == (1, 128, 5, 18, 32)
    assert padded.shape[3] % 2 == 0 and padded.shape[4] % 4 == 0  # evenly shardable


def test_pad_hw_replicate_replicates_edges_not_zeros():
    x = torch.randn(1, 8, 2, 17, 30)
    padded, _, _ = pad_hw_replicate(x, 2, 4)
    assert torch.equal(padded[:, :, :, 17, :30], padded[:, :, :, 16, :30])  # bottom row replicated
    assert torch.equal(padded[:, :, :, :17, 30], padded[:, :, :, :17, 29])  # right col replicated
    assert torch.equal(padded[:, :, :, :17, :30], x)  # original content untouched


def test_pad_hw_replicate_noop_when_already_aligned():
    x = torch.randn(1, 128, 5, 16, 28)  # even-res s1 latent (1024x1792)
    padded, h, w = pad_hw_replicate(x, 2, 4)
    assert (h, w) == (16, 28) and padded.shape == x.shape
    assert padded is x  # no copy when no padding needed


def test_encoder_pad_targets_even_latent_and_crop_restores_true_grid():
    # encoder pads pixels by factor*SPATIAL_COMPRESSION so the /32 latent lands on an even shard
    img = torch.randn(1, 3, 1, 544, 960)  # s1 cond image (-> latent 17x30, uneven)
    padded, h, w = pad_hw_replicate(img, 2 * 32, 4 * 32)
    assert padded.shape[3:] == (576, 1024)  # 576/32=18 (%2==0), 1024/32=32 (%4==0)
    assert (h // 32, w // 32) == (17, 30)  # latent crop restores the true grid
