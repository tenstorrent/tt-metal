# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_wan_fused_rmsnorm_pre_allgather_rejects_non_tile_aligned_width(device, expect_error):
    # The pre op reduces over the padded tile width, so a partial last tile would add its padding to sum(x**2).
    tt_input = ttnn.from_torch(torch.randn(1, 1, 32, 42), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input last dimension must be the same as padded last dimension"):
        ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input, dtype=ttnn.bfloat16)


@pytest.mark.parametrize(
    "tile",
    [ttnn.Tile((16, 16)), ttnn.Tile((32, 32), transpose_tile=True)],
    ids=["16x16", "32x32_transposed"],
)
def test_wan_fused_rmsnorm_pre_allgather_rejects_non_default_tile(device, tile, expect_error):
    # The program factory counts rows and columns in 32x32 tiles with the default face order.
    tt_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=device
    )
    with expect_error(RuntimeError, "Input tensor must use the default 32x32 tile"):
        ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input, dtype=ttnn.bfloat16)


# The pre-allgather factory divides the volume by W.
def test_wan_fused_rmsnorm_pre_allgather_rejects_zero_width(device, expect_error):
    tt_input = ttnn.from_torch(torch.zeros(1, 1, 32, 0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input must have non-zero height and width"):
        ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input, dtype=ttnn.bfloat16)


# compute_output_specs divides the width by num_heads_per_device before validate runs.
def test_wan_fused_rmsnorm_post_allgather_rejects_zero_heads(device, expect_error):
    tt_input = ttnn.from_torch(torch.randn(1, 1, 32, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_stats = ttnn.from_torch(torch.randn(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Number of heads must be greater than 0"):
        ttnn.experimental.wan_fused_rmsnorm_post_allgather(tt_input, tt_stats, num_heads_per_device=0)


# The DiT layernorm Welford factories divide the volume by H * W.
@pytest.mark.parametrize("shape", [(1, 1, 32, 0), (1, 1, 0, 64)], ids=["zero_width", "zero_height"])
def test_dit_layernorm_pre_allgather_rejects_zero_hw(device, expect_error, shape):
    tt_input = ttnn.from_torch(torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    recip_tensor = ttnn.create_layer_norm_reciprocals(device, core_range_set, 64)
    with expect_error(RuntimeError, "Input must have non-zero height and width"):
        ttnn.experimental.dit_layernorm_pre_allgather(tt_input, recip_tensor)


def test_dit_layernorm_post_allgather_rejects_zero_width(device, expect_error):
    tt_input = ttnn.from_torch(torch.zeros(1, 1, 32, 0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_stats = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input must have non-zero height and width"):
        ttnn.experimental.dit_layernorm_post_allgather(tt_input, tt_stats)
