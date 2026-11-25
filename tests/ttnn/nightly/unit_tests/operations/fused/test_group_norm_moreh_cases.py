# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# moreh_group_norm's forward test cases, run on ttnn.group_norm. Unsupported cases skip with their issue.

import pytest
import torch
import torch.nn.functional as F

import ttnn

C_NOT_TILE_ALIGNED = "C must be a multiple of 32: https://github.com/tenstorrent/tt-metal/issues/43826"
NO_MEAN_RSTD = "ttnn.group_norm does not return mean/rstd: https://github.com/tenstorrent/tt-metal/issues/67202"
GAMMA_BETA_WIDTH_NOT_VALIDATED = (
    "gamma/beta width is not validated: https://github.com/tenstorrent/tt-metal/issues/67203"
)

pytestmark = pytest.mark.use_module_device


def run_group_norm(x, num_groups, device, gamma=None, beta=None, **kwargs):
    """Runs ttnn.group_norm on an [N, C, H, W] torch tensor and returns the result as [N, C, H, W]."""
    N, C, H, W = x.shape
    grid = ttnn.determine_expected_group_norm_dram_grid_size(
        device=device, num_channels=C, num_groups=num_groups, input_nhw=N * H * W, num_batches=N
    )
    weight = bias = None
    if gamma is not None:
        weight, bias = ttnn.dram_group_norm_params_from_torch(
            [gamma, beta], C, num_groups, device, core_grid=grid, return_mask=False
        )
    tt_input = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(N, 1, H * W, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_output = ttnn.group_norm(
        tt_input,
        num_groups=num_groups,
        epsilon=1e-5,
        weight=weight,
        bias=bias,
        core_grid=grid,
        inplace=False,
        **kwargs,
    )
    return ttnn.to_torch(tt_output).float().reshape(N, H, W, C).permute(0, 3, 1, 2)


@pytest.mark.parametrize("C, num_groups", [(4, 1), (4, 2), (4, 4)])
@pytest.mark.parametrize("H, W", [(23, 23), (512, 512)])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("compute_mean_rstd", [True, False])
def test_group_norm_moreh_cases(C, num_groups, H, W, affine, compute_mean_rstd, device):
    if C % ttnn.TILE_SIZE:
        pytest.skip(C_NOT_TILE_ALIGNED)
    if compute_mean_rstd:
        pytest.skip(NO_MEAN_RSTD)
    torch.manual_seed(2024)
    N = 2
    x = torch.empty(N, C, H, W).uniform_(-2, 2).bfloat16().float()
    gamma = torch.empty(C).uniform_(-2, 2) if affine else None
    beta = torch.empty(C).uniform_(-2, 2) if affine else None

    expected = F.group_norm(x, num_groups, gamma, beta, eps=1e-5)
    # With core_grid given, num_out_blocks defaults to 1, which overflows L1 at 512x512.
    num_out_blocks = 32 if H * W >= 512 * 512 else -1
    actual = run_group_norm(x, num_groups, device, gamma, beta, num_out_blocks=num_out_blocks)

    # Same tolerances as the moreh test.
    tolerance = 0.13 if (C, num_groups) == (4, 1) and H == 512 and affine else 0.1
    assert torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("N, C, num_groups, H, W", [(2, 8, 4, 32, 32), (1, 32, 8, 64, 64)])
@pytest.mark.parametrize(
    "memory_configs",
    [
        "out_dram_mean_l1_rstd_l1",
        "out_l1_mean_dram_rstd_dram",
        "out_dram_mean_l1_rstd_dram",
        "out_l1_mean_l1_rstd_dram",
    ],
)
def test_group_norm_mean_rstd_memory_config(N, C, num_groups, H, W, memory_configs):
    if C % ttnn.TILE_SIZE:
        pytest.skip(C_NOT_TILE_ALIGNED)
    pytest.skip(NO_MEAN_RSTD)


@pytest.mark.skip(reason=GAMMA_BETA_WIDTH_NOT_VALIDATED)
@pytest.mark.parametrize("affine_width", [32, 256], ids=["narrower", "wider"])
@pytest.mark.parametrize("affine", ["gamma", "beta"])
def test_group_norm_rejects_mismatched_gamma_beta(affine_width, affine, device, expect_error):
    N, C, H, W, num_groups = 2, 128, 32, 32, 4
    grid = ttnn.CoreGrid(y=1, x=1)

    def packed(width):
        return ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(torch.rand(width), width, grid.y),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_input = ttnn.from_torch(torch.rand(N, 1, H * W, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, affine):
        ttnn.group_norm(
            tt_input,
            num_groups=num_groups,
            weight=packed(affine_width if affine == "gamma" else C),
            bias=packed(affine_width if affine == "beta" else C),
            core_grid=grid,
            inplace=False,
        )


@pytest.mark.parametrize("offset", [50.0, 500.0])
def test_group_norm_mean_far_from_zero(offset, device):
    torch.manual_seed(0)
    N, C, H, W, num_groups = 2, 32, 64, 64, 8
    x = (torch.randn(N, C, H, W) + offset).bfloat16().float()

    expected = F.group_norm(x.double(), num_groups, eps=1e-5)
    # Welford: the default mode's group mean is inexact, so it fails here (#53846).
    actual = run_group_norm(x, num_groups, device, use_welford=True)

    assert torch.allclose(actual.double(), expected, rtol=0.05, atol=0.05)


def test_group_norm_num_groups_zero(device, expect_error):
    tt_input = ttnn.from_torch(
        torch.randn(2, 1, 32 * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.group_norm(tt_input, num_groups=0, epsilon=1e-5, inplace=False)
