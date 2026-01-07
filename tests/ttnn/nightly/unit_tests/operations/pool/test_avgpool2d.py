# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math

from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_HIFI4,
    COMPUTE_KERNEL_CONFIG_LOFI,
)
from models.common.utility_functions import is_blackhole
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import HS

SliceWidth = ttnn.Op2DDRAMSliceWidth
SliceHeight = ttnn.Op2DDRAMSliceHeight


# helper to correct torch output for asymmetric padding
def correct_torch_asym_pad(
    torch_output, input_shape, kernel_size, stride, padding, divisor_override, count_include_pad
):
    _, _, in_h, in_w = input_shape
    pad_t, pad_b, pad_l, pad_r = padding
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    _, _, out_h, out_w = torch_output.shape
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
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tensor_map[tensor_shape] = torch_tensor
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
    in_dtype=ttnn.bfloat16,
    nightly_skips=True,
    out_dtype=ttnn.bfloat16,
    output_layout=ttnn.ROW_MAJOR_LAYOUT,
    compute_kernel_config=None,
    use_reshaped_tensor=True,
    dram_slice_config=None,
    config_tensor_in_dram=False,
):
    in_n, in_c, in_h, in_w = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h = dilation_w = 1  # avg pool does not yet support dilation

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

    if (out_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat4_b) and output_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("BFLOAT8_B/BFLOAT4_B output data format is not supported with ROW_MAJOR layout")

    if nightly_skips:
        if divisor_override is not None:
            if count_include_pad or ceil_mode:
                pytest.skip(
                    "divisor_override paired with count_include_pad or ceil_mode is trivial and not useful to test"
                )
        if count_include_pad and padding == (0, 0):
            pytest.skip("count_include_pad paired with no padding is trivial and not useful to test")
        if in_dtype == ttnn.bfloat8_b:
            if stride == (2, 2) or padding == (1, 1):
                pytest.skip("Skip for stride (2, 2) and padding (1, 1) for BF8!")
            if kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (9, 9) for BF8!")
        if ceil_mode:
            if stride == (1, 1):
                pytest.skip("ceiling mode with stride (1, 1) is trivial and not useful to test")
            if kernel_size == (3, 3) or kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (3, 3) and (9, 9) for ceil mode!")

    if pad_t > kernel_h / 2 or pad_b > kernel_h / 2 or pad_l > kernel_w / 2 or pad_r > kernel_w / 2:
        pytest.skip("padding is too large for the kernel size")

    if (in_h + pad_h) < kernel_h or (in_w + pad_w) < kernel_w:
        pytest.skip("kernel is too large for the padded tensor")

    ceil_mode_out_shape_adj = False
    if ceil_mode:
        out_h = math.ceil((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        if ((out_h - 1) * stride_h) >= (in_h + pad_t):
            ceil_mode_out_shape_adj = True
            out_h -= 1
        if ((out_w - 1) * stride_w) >= (in_w + pad_l):
            ceil_mode_out_shape_adj = True
            out_w -= 1
    else:
        out_h = math.floor((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1

    # using non-zero seed to avoid random spike in floating point error on single element of the
    # 1x256x56x56 tensor with divisor_override=5 and 5x5 kernel resulting in rtol=0.015 for that element
    torch.manual_seed(1e3)
    torch_input = randomize_tensor(tensor_map, input_shape)
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
    torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
    if in_dtype == ttnn.bfloat8_b:
        assert use_reshaped_tensor == True
        ttnn_input = ttnn.from_torch(torch_input_reshaped, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        if use_reshaped_tensor:
            ttnn_input = ttnn.from_torch(
                torch_input_reshaped, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        else:
            ttnn_input = ttnn.from_torch(
                torch_input_permuted, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
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
        memory_config=None,
        applied_shard_scheme=shard_scheme,
        dtype=out_dtype,
        output_layout=output_layout,
        compute_kernel_config=compute_kernel_config,
        dram_slice_config=dram_slice_config,
        config_tensor_in_dram=config_tensor_in_dram,
    )

    if run_twice:
        ttnn.deallocate(ttnn_output, True)
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
            memory_config=None,
            applied_shard_scheme=shard_scheme,
            dtype=out_dtype,
            output_layout=output_layout,
            compute_kernel_config=compute_kernel_config,
            dram_slice_config=dram_slice_config,
            config_tensor_in_dram=config_tensor_in_dram,
        )

    # apply padding manually to torch tensor since torch doesn't support asymmetric padding
    if padding_is_4d:
        assert (
            not ceil_mode_out_shape_adj
        ), "current test infrastructure does not support ceil mode output shape adjustments with 4D padding"
        torch_input_padded = torch.nn.functional.pad(
            torch_input,
            (pad_l, pad_r, pad_t, pad_b),  # torch is padding in the order (left, right, top, bottom)
            mode="constant",
            value=0,
        )
        torch_padding = [0, 0]  # use zero padding for torch avg pool since we are padding manually
    else:
        torch_input_padded = torch_input
        torch_padding = padding
    # run torch avg_pool2d
    torch_output = torch.nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )(torch_input_padded)

    # adjust the TTNN output to match the expected shape
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    )  # N, H, W, C
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W

    # apply correction to TORCH output for asymmetric padding when needed
    torch_needs_correction = padding_is_4d and divisor_override is None and count_include_pad is False
    if torch_needs_correction:
        torch_output = correct_torch_asym_pad(
            torch_output,
            input_shape,
            kernel_size,
            stride,
            (pad_t, pad_b, pad_l, pad_r),
            divisor_override,
            count_include_pad,
        )

    # test for equivalence
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    # TTNN supports scalars only in Bfloat16 and from recently it uses
    # tie-to-even rounding for fp32->bf16 scalar conversion, which improves accuracy
    # compared to the previous truncation method. For a 3x3 kernel using scalar 1/9:
    # the new rounding converts 1/9 → 0.111328125 in bf16, and 9 × 0.111328125 = 1.001953125,
    # which rounds back to 1.0 in bf16. This is much better than the old truncation method
    # which gave 1/9 → 0.11084 → 9 × 0.11084 = 0.99756 → 0.99609375 in bf16.
    # However, numerical differences still occur in complex operations due to:
    # different rounding at intermediate computation steps
    # and accumulation order differences in pooling operations
    # These factors compound, especially with small divisor overrides and large kernels,
    # requiring relaxed rtol thresholds for robust comparisons.
    rtol = 0.01
    if compute_kernel_config is not None:
        if compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi:
            atol = 0.045  # LOFI has less precise accumulation, so relax atol further
    if in_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat4_b:
        atol = 0.35
    # Ensure both tensors have the same dtype for comparison
    if out_dtype != ttnn.bfloat16:
        ttnn_output = ttnn_output.to(torch.bfloat16)
    allclose = torch.allclose(ttnn_output, torch_output, atol=atol, rtol=rtol)
    assert allclose


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # model shapes
        [1, 64, 112, 112],
        [8, 32, 132, 20],
        [1, 256, 56, 56],
        [1, 512, 28, 28],
        [1, 192, 264, 40],
        # # wide non-4 multiple tests
        [1, 800, 32, 32],
        [1, 576, 32, 32],
        # C partial tile test
        [1, 16, 12, 12],
        [1, 1, 56, 56],
        [2, 290, 10, 10],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (3, 3),  # 1 face 1 chunk
        (5, 5),  # 2 faces 1 chunk
        (7, 7),  # 2 chunks
        (9, 9),  # 3 chunks
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
        (1, 1),
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
        # only test height sharding, max pool tests the other schemes
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
def test_run_avg_pool2d(
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
    in_dtype,
):
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
        in_dtype=in_dtype,
        run_twice=True,
        config_tensor_in_dram=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_reshaped_tensor", [True, False])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_shape, shard_startegy",
    (
        (
            ([1, 256, 56, 56], HS),
            ([1, 512, 28, 28], HS),
            ([1, 192, 264, 40], HS),
            ([1, 800, 32, 32], HS),
            ([1, 576, 32, 32], HS),
            ([1, 16, 12, 12], HS),
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (3, 3),
        (5, 5),
        (7, 7),
    ),
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
def test_avg_pool2d_output_formats_and_layouts(
    device,
    tensor_map,
    input_shape,
    shard_startegy,
    kernel_size,
    use_reshaped_tensor,
    out_dtype,
    output_layout,
    in_dtype,
):
    padding = (0, 0)
    stride = (1, 1)

    if not use_reshaped_tensor:
        if in_dtype == ttnn.bfloat8_b:
            pytest.skip("BFLOAT8_B input data format is not supported without reshaped tensor")
        if out_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat4_b:
            pytest.skip("skip BFLOAT8_B/BFLOAT4_B output data format for non-reshaped tensor")
        if output_layout == ttnn.TILE_LAYOUT:
            pytest.skip("skip TILE_LAYOUT output layout for non-reshaped tensor")

    run_avg_pool2d(
        device,
        tensor_map,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode=False,
        divisor_override=None,
        count_include_pad=False,
        shard_scheme=shard_startegy,
        in_dtype=in_dtype,
        output_layout=output_layout,
        out_dtype=out_dtype,
        config_tensor_in_dram=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    ([1, 64, 112, 112],),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((3, 3),),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize(
    "compute_kernel_config",
    [COMPUTE_KERNEL_CONFIG_HIFI4, COMPUTE_KERNEL_CONFIG_HIFI2, COMPUTE_KERNEL_CONFIG_LOFI],
)
def test_avg_pool2d_compute_kernel_config(
    device,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    compute_kernel_config,
):
    run_avg_pool2d(
        device,
        tensor_map,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode=False,
        divisor_override=None,
        count_include_pad=False,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        in_dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        config_tensor_in_dram=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, num_slices",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        ([2, 32, 1024, 1024], 8),
        ([1, 320, 384, 384], 6),
        ([1, 256, 81, 81], 2),
    ),
)
@pytest.mark.parametrize(
    "kernel_size, padding",
    (
        # Wide and normal reductions go to normal kernels
        # Large reductions go to large kernels
        # Reductions which are large and wide at the same time
        # go to large kernels
        [(2, 2), (0, 0)],
        [(2, 2), (1, 1)],
        [(3, 3), (2, 2)],
        [(4, 4), (0, 0)],
        [(4, 4), (2, 2)],
        [(5, 5), (2, 2)],
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [True, False],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [True, False],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "slice_type",
    [SliceWidth, SliceHeight],
)
def test_avg_pool2d_dram(
    device,
    tensor_map,
    input_shape,
    num_slices,
    kernel_size,
    stride,
    padding,
    divisor_override,
    ceil_mode,
    count_include_pad,
    shard_scheme,
    in_dtype,
    slice_type,
):
    if slice_type == SliceHeight and input_shape[3] >= 256:
        pytest.skip("Skip height slice for inputs with large width")
    dram_slice_config = ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=slice_type)

    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
        in_dtype=in_dtype,
        nightly_skips=False,
        dram_slice_config=dram_slice_config,
        run_twice=True,
    )
