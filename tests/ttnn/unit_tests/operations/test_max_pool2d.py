# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.utility_functions import is_wormhole_b0, is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
## max-pool params:
## kernel_h, kernel_w
## stride_h, stride_w
## pad_h, pad_w
## dilation_h, dilation_w
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## resnet shapes
            [1, 64, 112, 112],
            [4, 64, 112, 112],
            [8, 64, 112, 112],
            [16, 64, 112, 112],
            # [20, 64, 112, 112],
            ## hpr shapes
            [8, 32, 132, 20],  ## pass
            [16, 32, 132, 20],  ## pass
            [32, 32, 132, 20],  ## pass
            [64, 32, 132, 20],  ## pass
            [128, 32, 132, 20],  ## pass
            # [256, 32, 132, 20],   ## oom
            [8, 32, 264, 40],  ## pass
            [16, 32, 264, 40],  ## pass
            [32, 32, 264, 40],  ## pass
            # [64, 32, 264, 40],    ## oom
            # [128, 32, 264, 40],   ## oom
            # [256, 32, 264, 40],   ## oom
            [4, 16, 1056, 160],  ## pass
            # [8, 16, 1056, 160],     ## oom
            # [16, 16, 1056, 160],    ## oom
            # [32, 16, 1056, 160],    ## oom
            # [64, 16, 1056, 160],    ## oom
            # [128, 16, 1056, 160],   ## oom
            # [256, 16, 1056, 160],   ## oom
            [8, 16, 528, 80],  ## pass
            [16, 16, 528, 80],  ## pass
            # [32, 16, 528, 80],  ## oom
            # [64, 16, 528, 80],  ## oom
            # [128, 16, 528, 80], ## oom
            # [256, 16, 528, 80], ## oom
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
@pytest.mark.parametrize(
    "nblocks",
    (1,),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    nblocks,
    device,
    dtype,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        pytest.skip("Invalid case")

    if (kernel_h == 3 and pad_h != 1) or (kernel_h == 2 and pad_h != 0):
        pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    if out_w % nblocks != 0:
        pytest.skip(f"Unsupported case when out_w ({out_w}) % nblocks ({nblocks}) != 0")

    if in_c % 16 != 0:
        pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")

    if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
        pytest.skip("This case runs out of memory on Grayskull")

    if in_n >= 16 and in_c >= 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
        pytest.skip("This case runs out of memory on Wormhole b0")

    if (
        is_wormhole_b0()
        and act_shape == [16, 64, 112, 112]
        and kernel_size == (3, 3)
        and padding == (1, 1)
        and stride == (2, 2)
        and dilation == (1, 1)
        and dtype == ttnn.bfloat16
    ):
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

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
    act_shape = (in_n, 1, in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    reader_patterns_cache = {}
    max_pool = ttnn.MaxPool2d(
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        dtype=dtype,
        device=device,
        batch_size=in_n,
        input_height=in_h,
        input_width=in_w,
        reader_patterns_cache=reader_patterns_cache,
    )

    if dtype == ttnn.bfloat8_b:
        if (in_h * in_w) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input height * width should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)
    ttact_d = max_pool.copy_input_to_device(ttact)

    out_d = max_pool(ttact_d)
    out_padded = max_pool.copy_output_from_device(out_d)

    # clear the cache maps
    reader_patterns_cache.clear()

    out_pytorch_padded = ttnn.to_torch(out_padded)
    out_pytorch = out_pytorch_padded[:, :, :, :in_c]

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
    out_pytorch = out_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, H, W
    assert_with_pcc(out_pytorch, golden_pytorch)

    ## do more rigorous comparision for each element
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(out_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(out_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(out_pytorch, golden_pytorch)

    assert allclose
    assert isclose
    if dtype == ttnn.bfloat16:
        assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override, xfail",
    (
        (1, 32, 9, 9, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 2, "snap_to_tile": True}, False),
        (1, 32, 17, 17, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4, "snap_to_tile": True}, False),
        (1, 32, 17, 17, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 2, "snap_to_tile": True}, False),
        (2, 32, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 3, "snap_to_tile": True}, False),
        (2, 32, 23, 23, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 3, "snap_to_tile": True}, False),
        (1, 32, 23, 23, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 10, "snap_to_tile": True}, True),
    ),
)
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
):
    if xfail:
        pytest.xfail()

    torch.manual_seed(0)

    if True:
        v = batch_size * input_height * input_width
        act = (
            torch.arange(v, dtype=torch.bfloat16)
            .repeat(input_channels)
            .reshape(input_channels, batch_size, input_height, input_width)
            .permute(1, 0, 2, 3)
        )
    else:
        act = torch.randn((batch_size, input_channels, input_height, input_width), dtype=torch.bfloat16)
    golden = torch.nn.functional.max_pool2d(
        act, (filter_height, filter_width), stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )

    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(batch_size, 1, input_height * input_width, input_channels)

    reader_patterns_cache = {}
    max_pool = ttnn.MaxPool2d(
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dtype=ttnn.bfloat16,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache=reader_patterns_cache,
        parallel_config_override=config_override,
    )

    ttact = ttnn.from_torch(act_reshaped, ttnn.bfloat16)
    ttact_d = max_pool.copy_input_to_device(ttact)
    out_d = max_pool(ttact_d)
    out_padded = max_pool.copy_output_from_device(out_d)
    out_pytorch_padded = ttnn.to_torch(out_padded)
    out_pytorch = out_pytorch_padded[:, :, :, :input_channels]
    out_pytorch = out_pytorch.reshape(golden.shape[0], golden.shape[2], golden.shape[3], golden.shape[1])
    out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    ## test for equivalance
    out_pytorch = out_pytorch.reshape(golden.shape)
    assert_with_pcc(out_pytorch, golden)
    assert torch.allclose(out_pytorch, golden)
