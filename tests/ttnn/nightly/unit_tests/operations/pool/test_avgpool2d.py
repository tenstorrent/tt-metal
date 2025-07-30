# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


# helper to correct torch output for asymmetric padding
def correct_torch_asym_pad(
    torch_output, input_shape, kernel_size, stride, padding, divisor_override, count_include_pad
):
    in_n, in_c, in_h, in_w = input_shape
    pad_t, pad_b, pad_l, pad_r = padding
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    out_n, out_c, out_h, out_w = torch_output.shape
    padded_h = in_h + pad_t + pad_b
    padded_w = in_w + pad_l + pad_r

    for oh in range(out_h):
        for ow in range(out_w):
            # get the kernel position in the padded input
            top_left_h = oh * stride_h
            top_left_w = ow * stride_w

            # count the number of sticks positions in the kernel which should be used in the divisor,
            # and the number used by torch which is unaware of basic padding since it was applied manually
            valid_sticks = 0
            torch_sticks = 0
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # get the padded coordinates of this stick
                    h = top_left_h + kh
                    w = top_left_w + kw

                    if h < padded_h and w < padded_w:
                        # torch is unaware of basic padding but is aware of ceil mode padding and never includes ceil
                        # mode sticks in the divisor count so only count sticks that are within the basic padded input
                        torch_sticks += 1

                        # get the non-padded coordinates of this stick
                        orig_h = h - pad_t
                        orig_w = w - pad_l

                        # check if this position is should be used in the divisor (within the original input shape)
                        if 0 <= orig_h < in_h and 0 <= orig_w < in_w:
                            valid_sticks += 1

            # apply a divisor correction if there's a mismatch between what torch used as divisor
            # and what should be used based on actual valid sticks
            if valid_sticks != torch_sticks:
                if valid_sticks > 0:
                    torch_output[:, :, oh, ow] *= torch_sticks / valid_sticks
                else:
                    raise ValueError(
                        "no valid sticks found, cannot correct torch output, it is possible padding is too large for the kernel size so we have an entire kernel in the padded region"
                    )

    return torch_output


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
    # handle both 2D and 4D padding
    padding_is_4d = False
    if len(padding) == 2:
        pad_h = int(padding[0] * 2)
        pad_w = int(padding[1] * 2)
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    elif len(padding) == 4:
        padding_is_4d = True
        pad_t, pad_b, pad_l, pad_r = padding
        pad_h = pad_t + pad_b
        pad_w = pad_l + pad_r
    else:
        raise ValueError(f"Padding must be 2D or 4D tuple, got {len(padding)}D")

    in_n, in_c, in_h, in_w = input_shape
    kernel_h, kernel_w = kernel_size
    torch.manual_seed(0)
    torch_input = randomize_tensor(tensor_map, input_shape)

    if pad_t > kernel_h / 2 or pad_b > kernel_h / 2 or pad_l > kernel_w / 2 or pad_r > kernel_w / 2:
        pytest.skip("padding is too large for the kernel size")

    if (in_h + pad_h) < kernel_h or (in_w + pad_w) < kernel_w:
        pytest.skip("kernel is too large for the padded tensor")

    if dtype == ttnn.bfloat8_b:
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    if padding_is_4d:
        # apply padding manually to torch tensor since torch doesn't support asymmetric padding
        torch_input_padded = torch.nn.functional.pad(
            torch_input,
            (pad_l, pad_r, pad_t, pad_b),
            mode="constant",
            value=0,
        )
        torch_padding = [0, 0]  # no additional padding since we already padded manually
    else:
        torch_input_padded = torch_input  # no manual padding needed
        torch_padding = padding  # use original padding
    # run torch avg_pool2d
    torch_output = torch.nn.functional.avg_pool2d(
        torch_input_padded,
        kernel_size,
        stride,
        torch_padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    # apply correction for asymmetric padding when needed
    if padding_is_4d and divisor_override is None and count_include_pad is False:
        torch_output = correct_torch_asym_pad(
            torch_output,
            input_shape,
            kernel_size,
            stride,
            (pad_t, pad_b, pad_l, pad_r),
            divisor_override,
            count_include_pad,
        )

    # run ttnn avg_pool2d
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=[pad_t, pad_b, pad_l, pad_r],
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
            padding=[pad_t, pad_b, pad_l, pad_r],
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
    pcc_thresh = 0.985
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        [1, 32, 16, 16],
        [1, 512, 112, 32],
        [1, 512, 16, 16],
        [1, 800, 16, 16],
        [2, 32, 16, 16],
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
        (5, 5),
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
        (3, 4),
        (1, 0, 1, 2),
        (1, 4, 3, 2),
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
