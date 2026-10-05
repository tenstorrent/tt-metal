# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Representative normalisation regressions; exhaustive matrices run in nightly/fused."""

import pytest
import ttnn

from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tests.ttnn.nightly.unit_tests.operations.fused import (
    test_two_pass_group_norm as group_norm,
    test_two_pass_group_norm_dram as group_norm_dram,
    test_two_pass_layer_norm as layer_norm,
    test_two_pass_layer_norm_sharded as sharded_layer_norm,
)
from tests.ttnn.nightly.unit_tests.operations.fused.test_two_pass_layer_norm import enabled_program_cache


def test_two_pass_layer_norm_unrepresentative_anchor(device):
    layer_norm.test_layer_norm_welford_unrepresentative_anchor(device, 128, 0)


@pytest.mark.parametrize("has_residual", [False, True])
def test_two_pass_layer_norm_bf16_large_offset(device, has_residual):
    layer_norm.test_layer_norm_welford_large_offset(device, 256, has_residual)


@pytest.mark.parametrize("has_residual", [False, True])
def test_two_pass_layer_norm_fp32_finalizer(device, has_residual):
    layer_norm.test_layer_norm_welford_fp32_finalizer_large_offset(
        device, 32, 2880 if has_residual else 64, has_residual, has_residual, has_residual
    )


def test_two_pass_layer_norm_row_major_affine_repeated_rows(device):
    layer_norm.test_layer_norm_fp32_residual_with_row_major_affine(device, True, True, True)


@run_for_wormhole_b0_or_blackhole()
def test_two_pass_layer_norm_nonfinite_anchor(device, enabled_program_cache):
    layer_norm.test_layer_norm_fp32_residual_nonfinite_anchor_is_row_local(
        device, enabled_program_cache, 128, float("inf")
    )


def test_two_pass_layer_norm_bfp8_tile_stride(device):
    layer_norm.test_layer_norm_bfp8_compensated_subtraction_tile_stride(
        device, (1, 1, 37, 288), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT
    )


@pytest.mark.parametrize("two_stage", [False, True])
def test_two_pass_sharded_layer_norm_large_offset(device, two_stage):
    sharded_layer_norm.test_layer_norm_sharded_fp32_large_offset(device, 1_000_000.0, two_stage, True, True, True)


def test_two_pass_sharded_layer_norm_low_bits(device):
    sharded_layer_norm.test_layer_norm_sharded_fp32_preserves_centred_low_bits(device, 3, True)


@pytest.mark.parametrize("two_stage", [False, True])
def test_two_pass_sharded_layer_norm_beta_only(device, two_stage):
    sharded_layer_norm.test_sharded_norm_beta_only(device, ttnn.float32, "layer", True, True, two_stage)


@pytest.mark.parametrize("constant", [None, 1e38, 1e-37], ids=["offset", "overflow", "underflow"])
def test_two_pass_group_norm_fp32_large_offset(device, constant):
    group_norm.test_group_norm_sharded_fp32_large_offset(device, True, 16, constant)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "dtype,layout,mask_dtype,has_weight",
    [
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat8_b, False),
        (ttnn.float32, ttnn.TILE_LAYOUT, ttnn.float32, True),
    ],
)
def test_two_pass_group_norm_affine_cache(device, enabled_program_cache, dtype, layout, mask_dtype, has_weight):
    group_norm.test_group_norm_sharded_optional_affine_program_cache(
        device, enabled_program_cache, 1, 128, 16, dtype, layout, mask_dtype, has_weight, True
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_two_pass_group_norm_dram_large_offset(device):
    group_norm_dram.test_group_norm_fp32_large_offset_DRAM(device, True, 2, None)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_two_pass_group_norm_dirty_padding(device):
    group_norm_dram.test_group_norm_non_tile_aligned_dirty_padding_grids_DRAM(device, 2, 1, 3, ttnn.TILE_LAYOUT)
