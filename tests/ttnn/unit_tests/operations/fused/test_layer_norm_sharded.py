# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
    layernorm_test_main,
    single_stage_param_sets,
    simple_size_params,
    generate_input_tensor,
    ttnn_layer_norm_sharded,
)

@pytest.mark.parametrize("h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt", single_stage_param_sets())
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_single_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, use_welford, two_stage, tensor_type, dtype
):
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
    )


@pytest.mark.parametrize(
    "h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt",
    [(32 * 2, 32 * 4, 2, 2, 2, 1, 1), (32 * 4, 32 * 8, 4, 2, 4, 1, 1), (32 * 8, 32 * 16, 2, 4, 8, 2, 1)],
)
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_two_stage(
    device, h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt, use_welford, two_stage, tensor_type, dtype
):
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_with_residual(device, use_welford, two_stage, tensor_type, dtype):
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see #30455")

    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=None,
        bias=None,
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_with_weight_and_bias(device, use_welford, two_stage, tensor_type, dtype):
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    weight = generate_input_tensor(1, w, "random", dtype)
    bias = generate_input_tensor(1, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=weight[0],
        bias=bias[0],
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random_normal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_with_weight_and_bias_row_major(device, use_welford, two_stage, tensor_type, dtype):
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = 64, 32, 2, 1, 1, 1, 1

    weight = generate_input_tensor(1, w, "random", dtype)
    bias = generate_input_tensor(1, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=None,
        weight=weight[0],
        bias=bias[0],
        weight_bias_layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["ascending_values_repeated_rows", "random"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_sharded_with_weight_and_bias_and_residual(device, use_welford, two_stage, tensor_type, dtype):
    if tensor_type == "random" or tensor_type == "random_normal":
        pytest.skip("Low PCC, see #30455")

    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    residual = generate_input_tensor(h, w, "random_normal", dtype)
    weight = generate_input_tensor(1, w, "random", dtype)
    bias = generate_input_tensor(1, w, "random_normal", dtype)
    layernorm_test_main(
        device,
        h,
        w,
        num_cores_h,
        num_cores_w,
        block_ht,
        block_wt,
        subblock_wt,
        use_welford,
        two_stage,
        tensor_type,
        dtype,
        residual=residual,
        weight=weight[0],
        bias=bias[0],
    )


@pytest.mark.parametrize("use_welford", [True])
def test_layer_norm_sharded_padded(device, use_welford):
    """
    Test layer norm on a sharded tensor that is padded with zeros
    in the width dimension.
    Compare against analytic layer norm calculation: (x - mean) / sqrt(var + eps)
    Only tests Welford layernorm, since legacy reduce doesn't give the correct
    result for partially-filled tiles.
    """
    torch.manual_seed(0)

    h, w = 32, 256
    num_cores_h, num_cores_w = 1, 8
    non_zero_columns = 3
    torch_input_tensor = torch.zeros((h, w), dtype=torch.bfloat16)
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
    output_ttnn = ttnn_layer_norm_sharded(
        device,
        tt_input_tensor,
        use_welford,
        block_ht=1,
        block_wt=1,
        subblock_w=1,
        residual=None,
        weight=None,
        bias=None,
    )

    golden = ttnn.get_golden_function(ttnn.layer_norm)
    golden_output = golden(torch_input_tensor, weight=None, bias=None, eps=1e-12)

    # Assert that the output is close to the golden output
    rtol = 1.6e-2
    atol = 1e-5
    assert torch.allclose(output_ttnn, golden_output, rtol=rtol, atol=atol)
