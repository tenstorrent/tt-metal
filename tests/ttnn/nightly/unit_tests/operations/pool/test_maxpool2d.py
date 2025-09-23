# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import ttnn
import pytest
import math

from models.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def tensor_map(request):
    tensor_map = {}

    return tensor_map


def randomize_torch_tensor(tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in tensor_map.keys():
        torch_tensor = tensor_map[tensor_shape]
    else:
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tensor_map[tensor_shape] = torch_tensor

    return torch_tensor


def run_max_pool(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    in_dtype,
    memory_config=None,
    shard_scheme=None,
    ceil_mode=False,
    in_place=False,
    nightly_skips=True,
    out_dtype=ttnn.bfloat16,
    output_layout=ttnn.ROW_MAJOR_LAYOUT,
):
    in_n, in_c, in_h, in_w = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

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

    # skips to avoid unimportant combinations
    if ceil_mode:
        if stride == (1, 1):
            pytest.skip("ceiling mode with stride (1, 1) is trivial and not useful to test")

    if dilation_h > 1 or dilation_w > 1:
        effective_kernel_h = dilation_h * (kernel_h - 1) + 1
        effective_kernel_w = dilation_w * (kernel_w - 1) + 1
        padded_input_h = in_h + pad_t + pad_b
        padded_input_w = in_w + pad_l + pad_r
        if effective_kernel_h > padded_input_h or effective_kernel_w > padded_input_w:
            pytest.skip("Effective kernel size cannot exceed padded input size")
    # skips to speed up nightly test
    if nightly_skips:
        if in_dtype == ttnn.bfloat8_b:
            if stride == (2, 2) or padding == (1, 1):
                pytest.skip("Skip for stride (2, 2) and padding (1, 1) for BF8!")
            if kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (9, 9) for BF8!")
        if ceil_mode:
            if kernel_size == (3, 3) or kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (3, 3) and (9, 9) for ceil mode!")
        if dilation != (1, 1) and stride != (1, 1):
            pytest.skip("Skip for dilation with stride != (1, 1), also skips ceil mode for dilation!")

    if pad_t > kernel_h / 2 or pad_b > kernel_h / 2 or pad_l > kernel_w / 2 or pad_r > kernel_w / 2:
        pytest.skip("padding is too large for the kernel size")

    if (in_h + pad_h) < kernel_h or (in_w + pad_w) < kernel_w:
        pytest.skip("kernel is too large for the padded tensor")

    out_n = in_n
    out_c = in_c
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

    torch.manual_seed(0)
    torch_input = randomize_torch_tensor(tensor_map, input_shape)
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    if in_dtype == ttnn.bfloat8_b:
        ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
        torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
        ttnn_input = ttnn.from_torch(torch_input_reshaped, in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        ttnn_input = ttnn.from_torch(torch_input_permuted, in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # run ttnn maxpool2d
    ttnn_output = ttnn.max_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=[pad_t, pad_b, pad_l, pad_r],  # ttnn is padding in the order (top, bottom, left, right)
        dilation=dilation,
        memory_config=memory_config,
        applied_shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
        in_place_halo=in_place,
        deallocate_input=True,
        reallocate_halo_output=True,
        dtype=out_dtype,
        output_layout=output_layout,
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
            value=float("-inf"),
        )
        torch_padding = [0, 0]  # use zero padding for torch avg pool since we are padding manually
    else:
        torch_input_padded = torch_input
        torch_padding = padding
    # run torch maxpool2d
    torch_output = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=ceil_mode,
    )(torch_input_padded)

    # adjust the TTNN output to match the expected shape
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(out_n, out_h, out_w, out_c)  # N, H, W, C
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))  # N, C, H, W

    # test for equivalance
    pcc_thresh = 1.0
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if in_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat8_b:
        pcc_thresh = 0.997
        atol = 0.35
    if out_dtype == ttnn.bfloat4_b:
        pcc_thresh = 0.93
        atol = 0.5
        rtol = 1.0
    assert_with_pcc(ttnn_output, torch_output, pcc_thresh)
    if out_dtype != ttnn.bfloat16:
        ttnn_output = ttnn_output.to(torch.bfloat16)
    allclose = torch.allclose(ttnn_output, torch_output, atol=atol, rtol=rtol)
    assert allclose
    if in_dtype == ttnn.bfloat16 and out_dtype == ttnn.bfloat16:
        isequal = torch.equal(ttnn_output, torch_output)
        assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  ## NCHW
    (
        (  # resnet shapes
            [1, 64, 112, 112],
            [16, 64, 112, 112],
            # hpr shapes
            [8, 32, 132, 20],
            [32, 32, 264, 40],
            [4, 16, 1056, 160],
            [16, 16, 528, 80],
            # wide for vgg
            [1, 256, 56, 56],
            [1, 512, 28, 28],
            # wide yolo kernel
            [1, 512, 10, 10],
            [1, 96, 112, 112],
            [1, 192, 132, 20],
            # wide non-8 multiple tests
            [1, 800, 32, 32],
            [1, 640, 32, 32],
            [1, 576, 32, 32],
            [1, 384, 32, 32],
            # C partial tile test
            [1, 16, 12, 12],
            [1, 1, 56, 56],
            [2, 290, 10, 10],
            [1, 280, 10, 10],
            # partial grid tests
            [1, 32, 10, 10],  # BH
            [1, 32, 6, 6],  # WH
        )
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
    "padding",
    (
        (0, 0),
        (1, 1),
        (1, 4, 3, 2),
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
    "dilation",
    (
        (1, 1),
        (2, 2),
    ),
)
@pytest.mark.parametrize(
    "in_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
        True,
    ],
)
def test_run_max_pool_height_shard(
    input_shape, kernel_size, padding, stride, dilation, device, tensor_map, in_dtype, ceil_mode
):
    run_max_pool(
        input_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  ## NCHW
    (
        (
            [1, 2048, 28, 28],
            [1, 1024, 6, 6],
            [1, 2048, 132, 20],
            [2, 4096, 10, 16],
            [1, 32768, 10, 10],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (5, 5),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (1, 2),
        (4, 2),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "dilation",
    (
        (1, 1),
        (3, 1),
    ),
)
@pytest.mark.parametrize(
    "in_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
    ],
)
def test_run_max_pool_width_shard(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    in_dtype,
    ceil_mode,
):
    run_max_pool(
        input_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  ## NCHW
    (
        (
            [1, 256, 56, 56],
            [1, 128, 10, 14],
            [1, 512, 8, 6],
            [1, 256, 132, 20],
            [1, 4096, 10, 10],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (6, 6),
        (10, 10),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (3, 2),
        (4, 5),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "dilation",
    (
        (1, 1),
        (4, 3),
    ),
)
@pytest.mark.parametrize(
    "in_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
    ],
)
def test_run_max_pool_block_shard(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    in_dtype,
    ceil_mode,
):
    run_max_pool(
        input_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  ## NCHW
    (
        (
            [8, 64, 112, 112],
            [1, 512, 10, 10],
        )
    ),
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_run_max_pool_mem_config(
    input_shape,
    device,
    tensor_map,
    memory_config,
):
    run_max_pool(
        input_shape,
        (3, 3),
        (1, 1),
        (2, 2),
        (1, 1),
        device,
        tensor_map,
        ttnn.bfloat16,
        memory_config=memory_config,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  ## NCHW
    (([1, 512, 10, 10],)),  ## yolov4 shapes
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (5, 5),
        (9, 9),
        (13, 13),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (2, 2),
        (4, 4),
        (6, 6),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "dilation",
    ((1, 1),),
)
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool_yolov4(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    in_dtype,
):
    run_max_pool(input_shape, kernel_size, padding, stride, dilation, device, tensor_map, in_dtype)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    (([1, 256, 54, 54],)),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((3, 3),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize("stride", ((2, 2),))
@pytest.mark.parametrize(
    "dilation",
    ((1, 1),),
)
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_run_max_pool_squeeze_net_model(
    input_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    tensor_map,
    in_dtype,
    ceil_mode,
):
    run_max_pool(
        input_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        tensor_map,
        in_dtype,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_shape, shard_startegy",
    (
        (
            ([1, 64, 112, 112], HS),
            ([1, 280, 10, 10], HS),
            ([1, 384, 32, 32], HS),
            ([1, 256, 132, 20], BS),
            ([1, 512, 8, 6], BS),
            ([2, 4096, 10, 16], WS),
            ([1, 32768, 10, 10], WS),
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (3, 3),
        (5, 5),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "in_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_max_pool2d_output_formats_and_layouts(
    device, tensor_map, input_shape, shard_startegy, kernel_size, out_dtype, output_layout, in_dtype
):
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)

    run_max_pool(
        input_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        tensor_map,
        in_dtype,
        shard_scheme=shard_startegy,
        out_dtype=out_dtype,
        output_layout=output_layout,
        nightly_skips=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 37888}], indirect=True)
@pytest.mark.parametrize(
    "input_shape_nchw",
    (([1, 128, 256, 512],)),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((3, 3),),
)
@pytest.mark.parametrize(
    "padding",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "dilation",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_panoptic_maxpool_sliced(device, input_shape_nchw, kernel_size, padding, dilation, stride, dtype, tensor_map):
    num_slices = 4

    batch_size, channels, input_h, input_w = input_shape_nchw
    assert channels % num_slices == 0, "Channels must be divisible by num_slices"
    slice_channels = channels // num_slices

    logger.info(f"Running Panoptic MaxPool2D with Channel Slicing (slices={num_slices})")

    torch.manual_seed(0)
    torch_input_nchw = randomize_torch_tensor(tensor_map, input_shape_nchw)

    torch_output = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(torch_input_nchw)

    out_h, out_w = torch_output.shape[2], torch_output.shape[3]

    ttnn_input_nhwc = ttnn.from_torch(
        torch_input_nchw.permute(0, 2, 3, 1), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype
    )
    output_slices = []
    for i in range(num_slices):
        start_idx = i * slice_channels
        end_idx = (i + 1) * slice_channels

        x_slice = ttnn.slice(ttnn_input_nhwc, [0, 0, 0, start_idx], [batch_size, input_h, input_w, end_idx])

        x_slice_reshaped = ttnn.reshape(x_slice, (1, 1, batch_size * input_h * input_w, slice_channels))
        ttnn.deallocate(x_slice)

        x_slice_pooled = ttnn.max_pool2d(
            x_slice_reshaped,
            batch_size=batch_size,
            input_h=input_h,
            input_w=input_w,
            channels=slice_channels,
            kernel_size=list(kernel_size),
            stride=list(stride),
            padding=list(padding),
            dilation=list(dilation),
            ceil_mode=False,
        )
        ttnn.deallocate(x_slice_reshaped)

        x_slice_output_nhwc = ttnn.reshape(x_slice_pooled, (batch_size, out_h, out_w, slice_channels))
        ttnn.deallocate(x_slice_pooled)
        x_slice_output_nhwc = ttnn.to_memory_config(x_slice_output_nhwc, ttnn.DRAM_MEMORY_CONFIG)
        output_slices.append(x_slice_output_nhwc)

    ttnn.deallocate(ttnn_input_nhwc)
    ttnn_output_nhwc = ttnn.concat(output_slices, dim=3)
    for s in output_slices:
        ttnn.deallocate(s)

    ttnn_output_torch = ttnn.to_torch(ttnn_output_nhwc)
    ttnn_output_torch_nchw = torch.permute(ttnn_output_torch, (0, 3, 1, 2))
    passed, pcc_score = assert_with_pcc(ttnn_output_torch_nchw, torch_output, pcc=0.999)
    logger.info(f"PCC Score: {pcc_score}")
    assert passed, f"PCC check failed. PCC: {pcc_score}"
