# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt", single_stage_param_sets())
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_sharded_single_stage(
    device_module, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, two_stage, tensor_type, dtype
):
    device = device_module
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
    device_module, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, two_stage, tensor_type, dtype
):
    device = device_module
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
def test_rms_norm_sharded_with_residual(device_module, two_stage, tensor_type, dtype):
    device = device_module
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see #30455")

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
def test_rms_norm_sharded_with_weight_and_bias(device_module, two_stage, tensor_type, dtype):
    device = device_module
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
def test_rms_norm_sharded_with_weight_and_bias_row_major(device_module, two_stage, tensor_type, dtype):
    device = device_module
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
def test_rms_norm_sharded_with_weight_and_bias_and_residual(device_module, two_stage, tensor_type, dtype):
    device = device_module
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see #30455")

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
def test_rms_norm_sharded_padded(device_module, h, w):
    device = device_module
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

    # Assert that the output is close to the golden output
    rtol = 1.6e-2
    atol = 1e-5
    assert torch.allclose(output_ttnn, golden_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("h,w", [(32, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rms_norm_sharded_width_default_config(device_module, h, w, dtype):
    device = device_module
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

    assert_with_pcc(golden_output, output_tensor, 0.9998)
