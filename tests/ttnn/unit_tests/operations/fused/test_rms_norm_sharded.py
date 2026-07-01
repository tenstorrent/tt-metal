# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
    rms_norm_test_main,
    single_stage_param_sets,
    simple_size_params,
    generate_input_tensor,
    ttnn_rms_norm_sharded,
    rms_norm_golden,
    run_sharded_norm_logical_width_multicore,
)
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import is_watcher_enabled

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt", single_stage_param_sets())
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_single_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, two_stage, tensor_type, dtype
):
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
    )


@pytest.mark.parametrize(
    "h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt",
    [(32 * 2, 32 * 4, 2, 2, 2, 1, 1), (32 * 4, 32 * 8, 4, 2, 4, 1, 1), (32 * 8, 32 * 16, 2, 4, 8, 2, 1)],
)
@pytest.mark.parametrize("two_stage", [True])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_two_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, two_stage, tensor_type, dtype
):
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
    )


@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_with_residual(device, two_stage, tensor_type, dtype):
    torch.manual_seed(0)
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=None,
    )


@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_with_bias_only(device, two_stage, tensor_type, dtype):
    """
    Sharded rms_norm with bias only (no weight). Exercises do_beta=1, do_gamma=0 in layernorm_sharded.
    Fails with the old cb_im formula; passes with the fixed formula.
    """
    torch.manual_seed(0)
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    bias = generate_input_tensor(1, w, "random", dtype)
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=None,
        bias=bias[0],
    )


@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_with_weight_and_bias(device, two_stage, tensor_type, dtype):
    torch.manual_seed(0)
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    weight = generate_input_tensor(1, w, "random", dtype)
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=weight[0],
    )


@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_with_weight_and_bias_row_major(device, two_stage, tensor_type, dtype):
    torch.manual_seed(0)
    if is_watcher_enabled() and two_stage is False:
        pytest.skip("Skipping test with watcher enabled, see github issue #37259")

    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = 64, 32, 2, 1, 1, 1, 1

    weight = generate_input_tensor(1, w, "random", dtype)
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=weight[0],
        weight_layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_with_weight_and_bias_and_residual(device, two_stage, tensor_type, dtype):
    torch.manual_seed(0)
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    weight = generate_input_tensor(1, w, "random", dtype)
    rms_norm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=weight[0],
    )


@pytest.mark.parametrize("h,w", [(32, 256)])
def test_rms_norm_sharded_padded(device, h, w):
    """
    Test layer norm on a sharded tensor that is padded with zeros
    in the width dimension.
    Compare against analytic layer norm calculation: (x - mean) / sqrt(var + eps)
    Only tests Welford layernorm, since legacy reduce doesn't give the correct
    result for partially-filled tiles.
    """
    torch.manual_seed(0)

    dtype = torch.bfloat16

    num_cores_h, num_cores_w = 1, 8
    non_zero_columns = 3
    torch_input_tensor = torch.zeros((h, w), dtype=dtype)
    torch_input_tensor[:, :non_zero_columns] = 1.0

    # Create sharded memory config for 2x8 core grid
    shard_height = h // num_cores_h
    shard_width = w // num_cores_w
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Convert to TTNN tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=sharded_mem_config,
    )

    # Run sharded layer norm
    output_ttnn = ttnn_rms_norm_sharded(
        device,
        tt_input_tensor,
        block_ht=1,
        block_wt=1,
        subblock_w=1,
        residual=None,
        weight=None,
    )
    output_ttnn = output_ttnn.to(dtype)

    golden_output = rms_norm_golden(torch_input_tensor, weight=None).to(dtype)

    assert_numeric_metrics(
        golden_output,
        output_ttnn,
        pcc_threshold=0.999,
        rtol=0.031,
        atol=0.052,
        frobenius_threshold=0.010,
    )


@pytest.mark.parametrize("h,w", [(32, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rms_norm_sharded_width_default_config(device, h, w, dtype):
    """
    Test RMS norm with width-sharded input on L1 and interleaved weight on DRAM.
    Uses default config (no explicit program_config).
    """
    torch.manual_seed(0)

    # For WIDTH_SHARDED: height stays full, width is sharded across all cores
    num_cores_h, num_cores_w = 8, 8
    num_cores_total = num_cores_h * num_cores_w  # 64
    shard_height = h  # Full height (32)
    shard_width = w // num_cores_total  # 2048 / 64 = 32

    torch_input_tensor = generate_input_tensor(h, w, "random", dtype)
    torch_weight = generate_input_tensor(1, w, "random", dtype)

    # Create width-sharded memory config for input on L1
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Input tensor: width-sharded on L1
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weight tensor: interleaved on DRAM
    weight = ttnn.from_torch(
        torch_weight[0],
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Uses default config (no explicit program_config)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_output = rms_norm_golden(torch_input_tensor, weight=torch_weight[0]).to(dtype)

    assert_numeric_metrics(
        golden_output,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.031,
        atol=0.052,
        frobenius_threshold=0.010,
    )


# Block sharding mandates that every core must get the same-sized tile-aligned shard:
# shard_w = ceil(w / cores / 32) * 32, so the padded width is cores * shard_w.
# This means the final core's shard may have padding.
# Two categories are covered:
#   - tile-aligned widths whose tiles do not divide evenly across the cores (96 over 2, 224 over 3):
#     every tile on the final core is either fully valid or fully padding, never partial. E.g. 96 over
#     2 gives the final core one fully-valid tile and one fully-padding tile.
#   - non-tile-aligned widths (72 over 2, 200 over 3): the logical columns run out mid-tile, so the
#     final core holds a partially-valid tile followed by a fully-padding tile. E.g. 72 over 2 gives
#     the final core one partially-valid tile (8 of its 32 columns valid) and one fully-padding tile.
# In both, the op must normalize over the logical width, not the padded per-core width.
@pytest.mark.parametrize(
    ("w", "num_cores_w"),
    [(96, 2), (224, 3), (72, 2), (200, 3)],
    ids=["w96_c2", "w224_c3", "w72_c2_nonaligned", "w200_c3_nonaligned"],
)
def test_rms_norm_sharded_uneven_multicore_logical_width(device, w, num_cores_w):
    run_sharded_norm_logical_width_multicore(device, is_rmsnorm=True, w=w, num_cores_w=num_cores_w)
