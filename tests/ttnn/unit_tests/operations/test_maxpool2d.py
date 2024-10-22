# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math

from models.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn


def run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    memory_config=None,
    shard_scheme=None,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if shard_scheme != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
            pytest.skip("Invalid case")
        if (kernel_h == 3 and pad_h != 1) or (kernel_h == 2 and pad_h != 0):
            pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    cores_x = device.core_grid.x
    cores_y = device.core_grid.y
    max_cores = cores_x * cores_y

    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED or shard_scheme is None:
        if in_c % 16 != 0:
            pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")
        if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
            pytest.skip("This case runs out of memory on Grayskull")
        if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
            pytest.skip("This case runs out of memory on Wormhole b0")

    if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if in_c < max_cores:
            pytest.skip("Width sharding requires channles >= cores")
        if in_c / max_cores < 16:
            pytest.skip("Width sharding requires large enough channels to shard (at least 16 per core)")

    if shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        if in_c < cores_x:
            pytest.skip("Block sharding requires channles >= cores")
        if in_c / cores_x < 16:
            pytest.skip("Block sharding requires large enough channels to shard (at least 16 per core)")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # act = torch.ones(act_shape, dtype=torch.bfloat16)
    # act = torch.arange(0, volume(act_shape), dtype=torch.bfloat16).reshape(act_shape)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = 1 + n + h + w + c # + torch.rand(1) * 0.15
    # torch.save(act, "act.pt")
    # act = torch.load("act.pt")

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        if (in_h * in_w) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input height * width should be multiple of 32")
        if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED and (in_c / max_cores) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input channels / max_cores should be multiple of 32")
        if shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED and (in_c / cores_x) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input channels / cores_x should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    pre_shard = shard_scheme == None

    ttact_device = ttnn.to_device(ttact, device)
    if pre_shard:
        parallel_config = ttnn._ttnn.operations.conv2d.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            device=device,
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            is_out_tiled=False,
        )
        sharded_memory_config = ttnn._ttnn.operations.conv2d.create_sharded_memory_config_from_parallel_config(
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
        memory_config=memory_config,
        applied_shard_scheme=shard_scheme,
    )

    output_host = output.cpu()
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_host))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])

    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W
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

    if memory_config:
        logger.debug(f"Output memory config: {memory_config}")
        assert ttnn.get_memory_config(output) == memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## resnet shapes
            [1, 64, 112, 112],
            [4, 64, 112, 112],
            [8, 64, 112, 112],
            [16, 64, 112, 112],
            # [20, 64, 112, 112],   ## oom
            ## hpr shapes
            [8, 32, 132, 20],
            [16, 32, 132, 20],
            [32, 32, 132, 20],
            [64, 32, 132, 20],
            [128, 32, 132, 20],
            # [256, 32, 132, 20],   ## oom
            [8, 32, 264, 40],
            [16, 32, 264, 40],
            [32, 32, 264, 40],
            # [64, 32, 264, 40],    ## oom
            # [128, 32, 264, 40],   ## oom
            # [256, 32, 264, 40],   ## oom
            [4, 16, 1056, 160],
            # [8, 16, 1056, 160],     ## oom
            # [16, 16, 1056, 160],    ## oom
            # [32, 16, 1056, 160],    ## oom
            # [64, 16, 1056, 160],    ## oom
            # [128, 16, 1056, 160],   ## oom
            # [256, 16, 1056, 160],   ## oom
            [8, 16, 528, 80],
            [16, 16, 528, 80],
            # [32, 16, 528, 80],  ## oom
            # [64, 16, 528, 80],  ## oom
            # [128, 16, 528, 80], ## oom
            # [256, 16, 528, 80], ## oom
            ## wide for vgg
            [1, 256, 56, 56],
            [1, 512, 28, 28],
            [1, 512, 14, 14],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (2, 2),
        (3, 3),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        (1, 1),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    use_program_cache,
):
    run_max_pool(act_shape, kernel_size, padding, stride, dilation, device, dtype)


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
    memory_config,
    use_program_cache,
):
    run_max_pool(act_shape, (3, 3), (1, 1), (2, 2), (1, 1), device, ttnn.bfloat16, memory_config=memory_config)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [1, 512, 28, 28],
            [1, 512, 14, 14],
            [1, 1024, 6, 6],
            [1, 2048, 6, 6],
            [1, 4096, 6, 6],
            [4, 1024, 40, 40],
            [2, 2048, 40, 40],
            [8, 4096, 10, 16],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (2, 2),
        (3, 3),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        (1, 1),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool_width_shard(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    use_program_cache,
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (
            [1, 256, 56, 56],
            [1, 256, 28, 28],
            [1, 256, 14, 14],
            [1, 256, 10, 14],
            [1, 512, 8, 6],
            [1, 1024, 6, 6],
            [1, 2048, 4, 6],
            [4, 512, 40, 40],
            [2, 1024, 40, 40],
            [8, 2048, 10, 16],
            ## resnet shapes
            [1, 64, 112, 112],
            [4, 64, 112, 112],
            [8, 64, 112, 112],
            [16, 64, 112, 112],
            ## hpr shapes
            [8, 32, 132, 20],
            [16, 32, 132, 20],
            [32, 32, 132, 20],
            [64, 32, 132, 20],
            [128, 32, 132, 20],
            [8, 32, 264, 40],
            [16, 32, 264, 40],
            [32, 32, 264, 40],
            [4, 16, 1056, 160],
            [8, 16, 528, 80],
            [16, 16, 528, 80],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (2, 2),
        (3, 3),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        (1, 1),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool_block_shard(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    use_program_cache,
):
    run_max_pool(
        act_shape,
        kernel_size,
        padding,
        stride,
        dilation,
        device,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
    dtype,
    use_program_cache,
):
    run_max_pool(act_shape, kernel_size, padding, stride, dilation, device, dtype)


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
    use_program_cache,
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

    if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
        pytest.skip("This case runs out of memory on Grayskull")

    if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
        pytest.skip("This case runs out of memory on Wormhole b0")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # act = torch.ones(act_shape, dtype=torch.bfloat16)
    # act = torch.arange(0, volume(act_shape), dtype=torch.bfloat16).reshape(act_shape)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = 1 + n + h + w + c # + torch.rand(1) * 0.15
    # torch.save(act, "act.pt")
    # act = torch.load("act.pt")

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
        parallel_config = ttnn._ttnn.operations.conv2d.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            device=device,
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            is_out_tiled=True,
        )
        sharded_memory_config = ttnn._ttnn.operations.conv2d.create_sharded_memory_config_from_parallel_config(
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
