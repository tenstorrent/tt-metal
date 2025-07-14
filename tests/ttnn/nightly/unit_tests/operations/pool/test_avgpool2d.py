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
    if tensor_shape in tensor_map.keys():
        torch_tensor = tensor_map[tensor_shape]
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    return torch_tensor


def run_avg_pool2d(
    device,
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
    dtype=ttnn.bfloat16,
):
    ## Test setup for both.
    in_n, in_c, in_h, in_w = input_shape
    torch.manual_seed(0)
    torch_input = randomize_tensor(tensor_map, input_shape)
    # counter = 0

    # for c in range(in_c):
    #     for h in range(in_h):
    #         for w in range(in_w):
    #             torch_input[0, c, h, w] = counter
    #             counter += 1

    ## Test setup for Actual.
    if dtype == ttnn.bfloat8_b:
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
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
    if run_twice:
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
    # ttnn_output = ttnn_output[:, :, :, :in_c]
    ## Test teardown for Actual.
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    )
    ttnn_output = ttnn.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W
    ttnn_output = ttnn.to_torch(ttnn_output)
    # print("TTNN output")
    # for n in range(ttnn_output.shape[0]):
    #     for c in range(ttnn_output.shape[1]):
    #         for h in range(ttnn_output.shape[2]):
    #             for w in range(ttnn_output.shape[3]):
    #                 value = ttnn_output[n, c, h, w]
    #                 print(f"Index: ({n}, {c}, {h}, {w}), Value: {value.item()}")

    # print("Torch output")
    # for n in range(torch_output.shape[0]):
    #     for c in range(torch_output.shape[1]):
    #         for h in range(torch_output.shape[2]):
    #             for w in range(torch_output.shape[3]):
    #                 value = torch_output[n, c, h, w]
    #                 print(f"Index: ({n}, {c}, {h}, {w}), Value: {value.item()}")

    ## Assertion
    pcc_thresh = 0.99
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    # TTNN only supports scalars in Bfloat16, so we cannot support rtol lower than 0.01
    # for instance, a 3x3 kernel uses scalar 1/9 = 0.111, which in Bfloat16 is 0.11084
    # so if we fill the tensor with 1s, Torch gets 9 * 0.111 = 0.999 which converted back
    # to Bfloat16 rounds to 1.0 but TTNN gets 9 * 0.11084 = 0.99756 which converted back
    # to Bfloat16 rounds to 0.9961, so the rdiff in this case is 0.0039
    # since the atol default is 0.016 we don't see this issue for low magnitude values, but
    # when using small divisor overrides with large kernels we see much large values which
    # overwhelm the atol and the rtol becomes significant
    rtol = 0.01
    if dtype == ttnn.bfloat8_b:
        atol = 0.35
    assert_with_pcc(torch_output, ttnn_output, pcc_thresh)
    allclose = torch.allclose(ttnn_output, torch_output, atol=atol, rtol=rtol)
    assert allclose, " Reference and output tensor are not close"
=======
    assert_with_pcc(torch_output, ttnn_output, 0.99)
    # allclose = torch.allclose(ttnn_output, torch_output, rtol=0.02)
    # assert allclose, " Reference and output tensor are not close"
>>>>>>> bc0a7e076b (Test case which is not passing for the partial tiles)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        [1, 1, 7, 7],
        [1, 32, 16, 16],
        [1, 290, 10, 10],
        [1, 512, 112, 32],
        [1, 512, 16, 16],
        [1, 800, 16, 16],
        [2, 1, 7, 7],
        [2, 32, 16, 16],
        [2, 290, 10, 10],
        [2, 512, 112, 32],
        [2, 512, 16, 16],
        [2, 800, 16, 16],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # Wide and normal reductions go to normal kernels
        # Large reductions go to large kernels
        # Reductions which are large and wide at the same time
        # go to large kernels
        (2, 2),
        (3, 3),
        # (5, 5),
        # (9, 9),
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
        (1, 2),
        (2, 3),
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
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "use_program_cache",
    [True, False],
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