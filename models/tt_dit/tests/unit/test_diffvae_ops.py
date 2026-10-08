# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side checks of the DiffVAE's shared weight-prep helpers."""

import pytest
import torch

from models.tt_dit.models.vae.diffvae_ops import (
    TILE,
    align_down,
    device_major_qkv,
    head_gain_matrix,
    head_mean_matrix,
    pad_dim,
    split_qkv,
)
from models.tt_dit.models.vae.diffvae_rope import pair_swap_matrix


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


def test_packed_lane_norm_and_rope_match_the_per_head_form():
    """The packed-lane path (mean matmul, rsqrt, gain matmul, then a 32-lane pair swap per tile)
    equals per-head RMSNorm * gamma * scale followed by the 64-lane pair-swap RoPE."""
    heads, head_dim, sites, eps, scale = 4, 64, 96, 1e-6, 0.125
    x = torch.randn(sites, heads * head_dim, dtype=torch.float64)
    gamma = torch.rand(head_dim, dtype=torch.float64) + 0.5
    cos = torch.randn(sites, heads * head_dim, dtype=torch.float64)
    sin = torch.randn(sites, heads * head_dim, dtype=torch.float64)

    per_head = x.reshape(sites * heads, head_dim)
    normed = per_head * torch.rsqrt(per_head.pow(2).mean(-1, keepdim=True) + eps) * gamma * scale
    swap = pair_swap_matrix(head_dim).double()
    rows_cos, rows_sin = cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)
    expected = (normed * rows_cos + (normed @ swap) * rows_sin).reshape(sites, -1)

    mean = (x * x) @ head_mean_matrix(heads, head_dim).double() + eps
    factor = torch.rsqrt(mean) @ head_gain_matrix(gamma, heads, scale).double()
    packed = x * factor
    tile_swap = torch.block_diag(*[pair_swap_matrix(TILE).double()] * (heads * head_dim // TILE))
    actual = packed * cos + (packed @ tile_swap) * sin
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
