# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side checks of the DiffVAE's shared weight-prep helpers."""

import pytest
import torch

from models.tt_dit.models.vae.diffvae_ops import align_down, device_major_qkv, pad_dim, split_qkv


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_device_major_qkv_is_the_per_head_index_order(tp: int):
    """The reshape/permute form equals the explicit ``[dev][q|k|v][heads/tp]`` index vector."""
    dim, head_dim = 256, 64
    heads = dim // head_dim
    heads_local = heads // tp
    fused = torch.randn(3 * dim, 96)
    order = torch.cat(
        [
            torch.arange(part * dim + h * head_dim, part * dim + (h + 1) * head_dim)
            for d in range(tp)
            for part in range(3)
            for h in range(d * heads_local, (d + 1) * heads_local)
        ]
    )
    assert torch.equal(device_major_qkv(fused, tp), fused[order])
    bias = torch.randn(3 * dim)
    assert torch.equal(device_major_qkv(bias, tp), bias[order])


def test_device_major_qkv_shards_to_each_devices_own_qkv():
    """Column shard ``d`` of the regrouped weight is ``[q_d | k_d | v_d]`` of the shipped one."""
    dim, tp = 8, 2
    fused = torch.arange(3 * dim).float()
    q, k, v = split_qkv(fused)
    regrouped = device_major_qkv(fused, tp)
    for d in range(tp):
        shard = regrouped[d * 3 * dim // tp : (d + 1) * 3 * dim // tp]
        lo, hi = d * dim // tp, (d + 1) * dim // tp
        assert torch.equal(shard, torch.cat([q[lo:hi], k[lo:hi], v[lo:hi]]))


def test_split_qkv_rejects_a_width_not_divisible_by_three(expect_error):
    with expect_error(ValueError, "not divisible by 3"):
        split_qkv(torch.zeros(7, 3))


def test_pad_dim_zero_fills_to_size():
    w = torch.ones(3, 5)
    assert pad_dim(w, 0, 3) is w
    padded = pad_dim(w, 1, 32)
    assert padded.shape == (3, 32)
    assert torch.equal(padded[:, :5], w)
    assert padded[:, 5:].abs().sum() == 0


def test_align_down_is_the_floor_counterpart_of_ceil_to():
    from models.tt_dit.utils.ltx import ceil_to

    for value in range(0, 40):
        for step in (1, 4, 8):
            assert align_down(value, step) <= value < align_down(value, step) + step
            assert ceil_to(value, step) >= value > ceil_to(value, step) - step
