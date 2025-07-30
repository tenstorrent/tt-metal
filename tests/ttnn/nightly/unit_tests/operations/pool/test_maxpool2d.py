# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math

from models.utility_functions import is_wormhole_b0, is_x2_harvested
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def torch_tensor_map(request):
    torch_tensor_map = {}

    return torch_tensor_map


def randomize_torch_tensor(torch_tensor_map, tensor_shape):
    tensor_shape = tuple(tensor_shape)
    if tensor_shape in torch_tensor_map.keys():
        torch_tensor = torch_tensor_map[tensor_shape]
    else:
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
        torch_tensor_map[tensor_shape] = torch_tensor

    return torch_tensor


def run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    torch_tensor_map,
    dtype,
    memory_config=None,
    shard_scheme=None,
    ceil_mode=False,
    in_place=False,
    nightly_skips=True,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size

    # handle both 2D and 4D padding
    if len(padding) == 2:
        pad_h = int(padding[0] * 2)
        pad_w = int(padding[1] * 2)
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    elif len(padding) == 4:
        pad_t, pad_b, pad_l, pad_r = padding
        pad_h = pad_t + pad_b
        pad_w = pad_l + pad_r
    else:
        raise ValueError(f"Padding must be 2D or 4D tuple, got {len(padding)}D")

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if pad_t > kernel_h / 2 or pad_b > kernel_h / 2 or pad_l > kernel_w / 2 or pad_r > kernel_w / 2:
        pytest.skip("padding is too large for the kernel size")

    if (in_h + pad_h) < kernel_h or (in_w + pad_w) < kernel_w:
        pytest.skip("kernel is too large for the padded tensor")

    if ceil_mode:
        out_h = math.ceil((in_h + pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    else:
        out_h = math.floor((in_h + pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    cores_x = device.core_grid.x
    cores_y = device.core_grid.y
    max_cores = cores_x * cores_y

    # skips to speed up nightly test
    if nightly_skips:
        if dtype == ttnn.bfloat8_b:
            if stride == (2, 2):
                pytest.skip("Skip for stride (2, 2) for BF8!")
            if kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (9, 9) for BF8!")
            if ceil_mode:
                pytest.skip("Skip for ceil mode for BF8!")
        if ceil_mode:
            if stride == (1, 1):
                pytest.skip("Skip for stride (1, 1) for ceil mode!")
            if kernel_size == (9, 9):
                pytest.skip("Skip for kernel size (9, 9) for ceil mode!")

    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED or shard_scheme is None:
        if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
            pytest.skip("This case runs out of memory")
        if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
            pytest.skip("This case runs out of memory on Wormhole b0")
        if (
            stride == (1, 1)
            and (act_shape == [16, 64, 112, 112] or act_shape == [4, 16, 1056, 160] or act_shape == [16, 16, 528, 80])
            and is_wormhole_b0()
        ):
            pytest.skip("This case runs out of memory on Wormhole b0")
        if kernel_h > 5 and kernel_w > 5 and act_shape == [16, 64, 112, 112] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    # construct the tensor in NCHW shape
    act = randomize_torch_tensor(torch_tensor_map, act_shape)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = h * in_w + w
    # torch.save(act, "act.pt")
    # act = torch.load("act.pt")

    # this op expects input tensor as { N, 1, H * W, C }
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    pre_shard = shard_scheme == None

    ttact_device = ttnn.to_device(ttact, device)
    if pre_shard:
        parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            compute_grid_size=device.compute_with_storage_grid_size(),
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            enable_channels_padding=False,
            is_shard_height_tile_multiple=False,
            is_shard_width_tile_multiple=False,
        )
        sharded_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttact_device.shape,
            parallel_config=parallel_config,
            tile_size=32 if dtype == ttnn.bfloat8_b else 1,
        )
        ttact_device = ttnn.to_memory_config(ttact_device, sharded_memory_config)
    # run ttnn maxpool2d
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_t, pad_b, pad_l, pad_r],  # ttnn is padding in the order (top, bottom, left, right)
        dilation=[dilation_h, dilation_w],
        memory_config=memory_config,
        applied_shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
        in_place_halo=in_place,
    )

    output_host = output.cpu()
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_host))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    # apply padding manually to torch tensor since torch doesn't support asymmetric padding
    act_padded = torch.nn.functional.pad(
        act,
        (pad_l, pad_r, pad_t, pad_b),  # torch is padding in the order (left, right, top, bottom)
        mode="constant",
        value=-float("inf"),
    )
    # run torch maxpool2d
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=[0, 0],  # always use zero padding we are padding manually
        dilation=dilation,
        return_indices=False,
        ceil_mode=ceil_mode,
    )(act_padded)

    # test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])

    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    pcc_thresh = 1.0
    if dtype == ttnn.bfloat8_b:
        pcc_thresh = 0.9994

    passing, pcc = assert_with_pcc(output_pytorch, golden_pytorch, pcc_thresh)

    logger.debug(f"Passing: {passing}, PCC: {pcc}")

    # do more rigorous comparision for each element
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(output_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose
    assert isclose
    if dtype == ttnn.bfloat16:
        assert isequal

    if memory_config:
        logger.debug(f"Output memory config: {memory_config}")
        assert ttnn.get_memory_config(output) == memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## resnet shapes
            [1, 64, 112, 112],
            [16, 64, 112, 112],
            ## hpr shapes
            [8, 32, 132, 20],
            [32, 32, 264, 40],
            [4, 16, 1056, 160],
            [16, 16, 528, 80],
            ## wide for vgg
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
            # C=16 test
            [1, 16, 12, 12],
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
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "dtype",
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
    act_shape, kernel_size, padding, stride, dilation, device, torch_tensor_map, dtype, ceil_mode
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [1, 2048, 28, 28],
            [1, 1024, 6, 6],
            [2, 2048, 132, 20],
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
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "dtype",
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
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    torch_tensor_map,
    dtype,
    ceil_mode,
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
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
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "dtype",
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
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    torch_tensor_map,
    dtype,
    ceil_mode,
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [8, 64, 112, 112],
            [1, 512, 10, 10],
        )
    ),
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_run_max_pool_mem_config(
    act_shape,
    device,
    torch_tensor_map,
    memory_config,
):
    run_max_pool(
        act_shape, (3, 3), (1, 1), (2, 2), (1, 1), device, torch_tensor_map, ttnn.bfloat16, memory_config=memory_config
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (([1, 512, 10, 10],)),  ## yolov4 shapes
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (5, 5),
        (9, 9),
        (13, 13),
        # (3, 3),
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
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool_yolov4(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    torch_tensor_map,
    dtype,
):
    run_max_pool(act_shape, kernel_size, padding, stride, dilation, device, torch_tensor_map, dtype)


@pytest.mark.skip("See GH issue #12285")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override, xfail",
    (
        (
            1,
            32,
            9,
            9,
            3,
            3,
            1,
            1,
            1,
            1,
            {
                "num_cores_nhw": 2,
            },
            False,
        ),
        (
            1,
            32,
            17,
            17,
            3,
            3,
            1,
            1,
            1,
            1,
            {
                "num_cores_nhw": 4,
            },
            False,
        ),
        (
            1,
            32,
            17,
            17,
            3,
            3,
            2,
            2,
            1,
            1,
            {
                "num_cores_nhw": 2,
            },
            False,
        ),
        (
            2,
            32,
            16,
            16,
            3,
            3,
            1,
            1,
            1,
            1,
            {
                "num_cores_nhw": 3,
            },
            False,
        ),
        (
            2,
            32,
            23,
            23,
            3,
            3,
            2,
            2,
            1,
            1,
            {
                "num_cores_nhw": 3,
            },
            False,
        ),
        (
            1,
            32,
            23,
            23,
            3,
            3,
            1,
            1,
            1,
            1,
            {
                "num_cores_nhw": 10,
            },
            True,
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_pool_core_nondivis(
    device,
    batch_size,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    xfail,
    dtype,
):
    if xfail:
        pytest.xfail()

    act_shape = (batch_size, input_channels, input_height, input_width)
    in_n, in_c, in_h, in_w = batch_size, input_channels, input_height, input_width
    kernel_h, kernel_w = filter_height, filter_width
    dilation_h, dilation_w = 1, 1

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        pytest.skip("Invalid case")

    if (kernel_h == 3 and pad_h != 1) or (kernel_h == 2 and pad_h != 0):
        pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    if in_c % 16 != 0:
        pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")

    if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
        pytest.skip("This case runs out of memory on Wormhole b0")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        if (in_h * in_w) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input height * width should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    pre_shard = True
    # pre_shard = False

    ttact_device = ttnn.to_device(ttact, device)
    if pre_shard:
        parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            compute_grid_size=device.compute_with_storage_grid_size(),
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            enable_channels_padding=False,
            is_shard_height_tile_multiple=True,
            is_shard_width_tile_multiple=True,
        )
        sharded_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttact_device.shape,
            parallel_config=parallel_config,
            tile_size=32 if dtype == ttnn.bfloat8_b else 1,
        )
        ttact_device = ttnn.to_memory_config(ttact_device, sharded_memory_config)
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
    )

    # interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    # output = ttnn.to_memory_config(output, interleaved_mem_config)
    output_host = output.cpu()
    output_pytorch_padded = ttnn.to_torch(output_host)
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        (kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W
    # torch.save(output_pytorch, "output_pytorch.pt")
    # torch.save(golden_pytorch, "golden_pytorch.pt")
    passing, pcc = assert_with_pcc(output_pytorch, golden_pytorch)

    logger.debug(f"Passing: {passing}, PCC: {pcc}")

    ## do more rigorous comparision for each element
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(output_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose
    assert isclose
    if dtype == ttnn.bfloat16:
        assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",
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
@pytest.mark.parametrize("dilation", ((1, 1),))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_run_max_pool_squeeze_net_model(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    torch_tensor_map,
    dtype,
    ceil_mode,
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        torch_tensor_map,
        dtype,
        ceil_mode=ceil_mode,
    )
