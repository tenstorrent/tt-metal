# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.pool import golden_adaptive_avg_pool2d, golden_adaptive_max_pool2d


def randomize_tensor(tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map.keys():
        torch_tensor = tensor_map[tensor_shape]
    else:
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tensor_map[tensor_shape] = torch_tensor
    return torch_tensor


def run_adaptive_pool2d(
    device,
    tensor_map,
    input_shape,
    output_size,
    dtype,
    pool_type="avg",
    dram_slice_config=None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sharding=None,
):
    in_n, in_c, in_h, in_w = input_shape
    out_h, out_w = output_size

    # Skip cases where output size is larger than input size (upsampling)
    if output_size is not None:
        input_h, input_w = input_shape[2], input_shape[3]
        output_h, output_w = output_size[0], output_size[1]
        if output_h > input_h or output_w > input_w:
            pytest.skip(f"Adaptive pooling cannot upsample: input {input_h}x{input_w} -> output {output_h}x{output_w}")

    # Skip memory-intensive cases that cause OOM
    if dtype == ttnn.bfloat16 and in_n == 1 and in_c == 64 and in_h == 224 and in_w == 224:
        pytest.skip(f"Skipping memory-intensive case [1, 64, 224, 224] -> [{out_h}, {out_w}] with {dtype} due to OOM")

    torch.manual_seed(0)
    # Create tensor directly in [1, 1, NHW, C] format used by both golden and TTNN ops
    nhwc_shape = (1, 1, in_n * in_h * in_w, in_c)
    torch_input = randomize_tensor(tensor_map, nhwc_shape)

    if dtype == ttnn.bfloat8_b:
        ttnn_input = ttnn.from_torch(torch_input, dtype, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        ttnn_input = ttnn.from_torch(torch_input, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Call the appropriate TTNN function
    if pool_type == "avg":
        ttnn_output = ttnn.adaptive_avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=memory_config,
            dram_slice_config=dram_slice_config,
            applied_shard_scheme=sharding,
        )

        torch_output = golden_adaptive_avg_pool2d(
            input_tensor=torch_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=(out_h, out_w),
        )
    else:  # max
        ttnn_output = ttnn.adaptive_max_pool2d(
            input_tensor=ttnn_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=memory_config,
            dram_slice_config=dram_slice_config,
            applied_shard_scheme=sharding,
        )

        torch_output = golden_adaptive_max_pool2d(
            input_tensor=torch_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=(out_h, out_w),
        )

    ttnn_output = ttnn.to_torch(ttnn_output)

    # Test for equivalence with pool-type-specific tolerances
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if pool_type == "avg":
        rtol = 0.01  # Relaxed rtol for avg pool due to bfloat16 scalar precision limitations

    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(ttnn_output, torch_output, atol=atol, rtol=rtol)
    assert (
        allclose
    ), f"Reference and output tensor are not close. Input: {input_shape}, Output: [{out_h}, {out_w}], Pool: {pool_type}"

    # Max pool has additional strict equality check for bfloat16
    if pool_type == "max" and dtype == ttnn.bfloat16:
        isequal = torch.equal(ttnn_output, torch_output)
        assert (
            isequal
        ), f"Reference and output tensor are not equal for bfloat16. Input: {input_shape}, Output: [{out_h}, {out_w}]"
