# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math

from loguru import logger


import torch


from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc

from functools import reduce
import operator
import ttnn


def volume(shape):
    return reduce(operator.mul, shape, 1)


## max-pool params:
## kernel_h, kernel_w
## stride_h, stride_w
## pad_h, pad_w
## dilation_h, dilation_w
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## [1, 32, 32, 32],
            [1, 64, 64, 64],
            [1, 64, 112, 112],
            # [2, 64, 64, 64],
            # [8, 64, 64, 64],
            [8, 64, 128, 128],
            # [4, 64, 112, 112],
            [8, 64, 112, 112],
            # [9, 32, 32, 32],
            # [9, 64, 64, 64],
            # [9, 1, 128, 128],
            # [16, 32, 32, 32],
            # [16, 64, 64, 64],
            # [16, 64, 112, 112],
            # [16, 1, 128, 128],
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (1, 1),
        (3, 3),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),  ## default
        (1, 1),
    ),
)
@pytest.mark.parametrize(
    "stride",
    (
        (1, 1),  ## default
        (2, 2),
    ),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
    ),
    ids=[
        "in_DRAM",
        #  "in_L1",
        "in_HEIGHT_SHARDED",
    ],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
    ),
    ids=[
        "out_DRAM",
        #  "out_L1",
        "out_HEIGHT_SHARDED",
    ],
)
@pytest.mark.parametrize(
    "nblocks",
    (
        1,  ## default
        # 4,
        # 8,
        # 28, # for perf
        # 56,
    ),
)
@pytest.mark.parametrize(
    "use_multicore",
    (
        False,
        True,
    ),
)
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    in_mem_config,
    out_mem_config,
    nblocks,
    use_multicore,
    device,
):
    # ttnn.device.EnableMemoryReports()

    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        pytest.skip("Invalid case")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    if out_w % nblocks != 0:
        pytest.skip(f"Unsupported case when out_w ({out_w}) % nblocks ({nblocks}) != 0")

    if in_c != 64:
        pytest.skip("Current maxpool writer needs nchannels to be 64!")

    if use_multicore and (padding != (1, 1) or stride != (2, 2) or kernel_size != (3, 3)):
        pytest.skip("Multicore version only supports Resnet50 configs for now.")

    if nblocks > 1 and in_mem_config.is_sharded() and use_multicore:
        pytest.skip("nblocks > 1 is not properly supported with multicore sharded input")

    interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    if (out_mem_config.is_sharded() or in_mem_config.is_sharded()) and not use_multicore:
        pytest.skip("Unsupported sharding in single core")

    out_nhw = (out_h * out_w) * in_n
    compute_grid_size = device.compute_with_storage_grid_size()
    # Resnet specific config pulled from max_pool_multi_core.cpp:get_num_cores
    is_resnet_shape = out_nhw in (3136, 6272, 12544, 25088)
    ncores_for_resnet_shape = 98
    ncores_on_n300 = 56

    if is_resnet_shape and (compute_grid_size.x * compute_grid_size.y < ncores_for_resnet_shape):
        pytest.skip(
            f"Skipping over Resnet specific config where parallelization does not fit on core grid {compute_grid_size}"
        )

    # if (compute_grid_size.x * compute_grid_size.y) == ncores_on_n300:
    #     pytest.skip(f"Skipping on N300 (8x7 core grid) due to bug https://github.com/tenstorrent/tt-metal/issues/5458")

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    torch.manual_seed(0)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # act = torch.ones(act_shape, dtype=torch.bfloat16)
    # act = torch.arange(0, volume(act_shape), dtype=torch.bfloat16).reshape(act_shape)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = n + c + h + w

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (in_n, 1, in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    act_shape_padded = (in_n, 1, in_h * in_w, _nearest_32(in_c))
    act_padding = (0, act_shape_padded[3] - act_shape[3])
    act_padded = torch.nn.functional.pad(act_reshaped, act_padding, value=0xFF7F)
    assert act_shape_padded == act_padded.shape

    ttact = ttnn.Tensor(
        act_padded.flatten().tolist(),
        act_shape_padded,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    ncores = 1
    grid_size = [1, 1]
    if in_mem_config.is_sharded():
        in_height = in_n * in_h * in_w
        out_nhw = in_n * out_h * out_w
        ## NOTE: these should match the max_pool op code for now. Hardcoded Resnet shapes only.
        if out_nhw == 1024:
            ncores = 32
            grid_size = [12, 3]
        elif out_nhw == 2048 or out_nhw == 4096 or out_nhw == 8192 or out_nhw == 16384 or out_nhw == 32768:
            ncores = 64
            grid_size = [12, 6]
        elif out_nhw == 3136 or out_nhw == 6272 or out_nhw == 12544 or out_nhw == 25088:
            ncores = 98
            grid_size = [12, 9]
        else:
            assert False
        if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
            pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

        ttact = ttact.to(device, interleaved_mem_config)
        ttact = ttnn.interleaved_to_sharded(
            ttact,
            grid_size,
            [in_height // ncores, act_padded.shape[-1]],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        ttact = ttact.to(device, in_mem_config)

    out_padded = ttnn.max_pool2d(
        ttact,
        in_n,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        memory_config=out_mem_config,
        nblocks=nblocks,
        use_multicore=use_multicore,
    )
    if out_mem_config.is_sharded():
        out_padded = ttnn.sharded_to_interleaved(out_padded, interleaved_mem_config)
    out_padded = out_padded.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    out_shape_padded = out_padded.get_legacy_shape()
    out_pytorch_padded = out_padded.to_torch().reshape(tuple(out_shape_padded))  ## N, 1, HW, C
    out_pytorch = out_pytorch_padded[:, :, :, :in_c]
    out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, 1, HW

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    out_pytorch = out_pytorch.reshape(golden_pytorch.shape)
    passing = torch.allclose(out_pytorch, golden_pytorch)  ##, rtol=1e-01, atol=1e-01)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    passing = torch.allclose(out_pytorch, golden_pytorch)  ##, rtol=1e-01, atol=1e-01)
    logger.info(f"Passing PCC = {passing_pcc}")
    logger.info(f"Output PCC = {output_pcc}")
    # print(f'OUTPUT: {out_pytorch}')
    # print(f'GOLDEN: {golden_pytorch}')

    assert passing
    assert passing_pcc
    assert passing
