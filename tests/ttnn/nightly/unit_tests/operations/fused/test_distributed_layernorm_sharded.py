# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Nightly-only parametrizations of the sanity tests in
tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py (#62486)."""

import pytest
import ttnn

import tests.ttnn.unit_tests.operations.fused.test_distributed_layernorm_sharded as base

_SEEDS_NIGHTLY_ONLY = [
    s for s in base.SIMULATED_DISTRIBUTED_LAYERNORM_SEEDS if s not in base.SIMULATED_DISTRIBUTED_LAYERNORM_SANITY_SEEDS
]


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", _SEEDS_NIGHTLY_ONLY)
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [1])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid, grid_offset, output_core_grid", base.SIMULATED_DISTRIBUTED_LAYERNORM_GRIDS)
def test_simulated_distributed_layernorm(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    weights_df,
    seed,
    eps,
    mean,
    std,
    min_pcc,
    max_atol,
    core_grid,
    grid_offset,
    output_core_grid,
):
    base.test_simulated_distributed_layernorm(
        device,
        input_width,
        num_devices,
        is_rmsnorm,
        input_df,
        weights_df,
        seed,
        eps,
        mean,
        std,
        min_pcc,
        max_atol,
        core_grid,
        grid_offset,
        output_core_grid,
    )
