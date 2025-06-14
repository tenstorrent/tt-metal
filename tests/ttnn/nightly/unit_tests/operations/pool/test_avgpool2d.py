# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


def randomize_tensor(tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    # if tensor_shape in tensor_map.keys():
    #     torch_tensor = tensor_map[tensor_shape]
    # else:
    #     torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    torch_tensor = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    for n in range(tensor_shape[0]):
        for c in range(tensor_shape[1]):
            for h in range(tensor_shape[2]):
                for w in range(tensor_shape[3]):
                    torch_tensor[n, c, h, w] = h * tensor_shape[3] + w
    return torch_tensor


def run_avg_pool2d(
    device,
    use_program_cache,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    divisor_override,
    count_include_pad,
    shard_scheme,
    run_twice=False,
):
    if stride[0] == 1 and ceil_mode == True:
        pytest.skip("ceil stride")
    if padding[0] == 0 and count_include_pad == True:
        pytest.skip("count pad")

    ## Test setup for both.
    in_n, in_c, in_h, in_w = input_shape
    torch.manual_seed(0)
    torch_input = randomize_tensor(tensor_map, input_shape)

    ## Test setup for Actual.
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    ## Get Expected output.
    torch_output = torch.nn.functional.avg_pool2d(
        torch_input,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    ## Get Actual output
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        applied_shard_scheme=shard_scheme,
    )
    if False:
        ttnn_output = ttnn.avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            count_include_pad=count_include_pad,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            applied_shard_scheme=shard_scheme,
        )

    ## Test teardown for Actual.
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    )
    ttnn_output = ttnn.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W
    ttnn_output = ttnn.to_torch(ttnn_output)

    ## Assertion
    # print(ttnn_output)
    # print(torch_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
    allclose = torch.allclose(ttnn_output, torch_output, rtol=0.02)
    assert allclose, " Reference and output tensor are not close"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Passing with 9x9
        # [1, 32, 28, 28],
        # [1, 64, 28, 28],
        # [1, 128, 28, 28],
        # [1, 256, 28, 28],
        # [1, 288, 28, 28],
        # [1, 384, 28, 28],
        # [1, 512, 28, 28],
        # [1, 576, 28, 28],
        # [1, 640, 28, 28],
        # [1, 800, 28, 28],
        # testing
        [1, 128, 28, 28],
        [1, 320, 28, 28],
        [1, 512, 28, 28],
        [1, 640, 28, 28],
        [1, 800, 28, 28],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # Wide and normal reductions go to normal kernels
        # Large reductions go to large kernels
        # Reductions which are large and wide at the same time
        # go to large kernels
        # (2, 2),
        # (3, 3),
        # (5, 5),
        (8, 8),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "stride",
    (
        (1, 1),
        (2, 2),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        # (1, 1),
        # (2, 2),
        (4, 4),
    ),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
        5,
    ],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "use_program_cache",
    [False],
)
def test_run_avg_pool2d(
    device,
    use_program_cache,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    divisor_override,
    count_include_pad,
    shard_scheme,
):
    if (
        shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        and (tuple(input_shape) == (2, 512, 112, 32) or tuple(input_shape) == (1, 512, 112, 32))
        and divisor_override == None
        and (ceil_mode == True or count_include_pad == False)
    ):
        pytest.skip("Not enough L1 space for the correct calculation of the elements, use different kind of sharding")

    if any(p > k // 2 for p, k in zip(padding, kernel_size)):
        pytest.skip(
            "Known issue with this combination of parameters - RuntimeError: pad should be at most half of kernel size."
        )
    run_avg_pool2d(
        device,
        use_program_cache,
        tensor_map,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
        run_twice=True,
    )
