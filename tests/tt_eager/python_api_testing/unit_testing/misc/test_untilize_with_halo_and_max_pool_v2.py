# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math

from loguru import logger

import torch

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_max_pool import (
    TTPyMaxPool,
    SlidingWindowOpParamsWithParallelConfig,
)

import tt_lib as ttl
from tt_lib.utils import _nearest_32
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0


def volume(shape):
    vol = 1.0
    for d in shape:
        vol *= d
    return vol


@skip_for_wormhole_b0()
## max-pool params:
## kernel_h, kernel_w
## stride_h, stride_w
## pad_h, pad_w
## dilation_h, dilation_w
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## Only resnet shapes supported for now in untilize with halo + maxpool
            [1, 64, 64, 64],
            [1, 64, 112, 112],
            [4, 64, 112, 112],
            [8, 64, 112, 112],
            [16, 64, 112, 112],
            [20, 64, 112, 112],
        )
    ),
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
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "nblocks",
    (1,),
)
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
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
    # ttl.device.EnableMemoryReports()
    if act_shape[0] >= 16 and dtype == ttl.tensor.DataType.BFLOAT16:
        pytest.skip("Configuration does not fit in L1")

    in_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)

    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        logger.info("Invalid case")
        pytest.skip()

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    if out_w % nblocks != 0:
        logger.info(f"Unsupported case when out_w ({out_w}) % nblocks ({nblocks}) != 0")
        pytest.skip()

    if in_c != 64:
        logger.info("Current maxpool writer needs nchannels to be 64!")
        pytest.skip()

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        ttl.tensor.BufferType.DRAM if act_shape[0] > 8 else ttl.tensor.BufferType.L1,
    )
    assert out_mem_config.is_sharded() and in_mem_config.is_sharded()

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
    #                 act[n, c, h, w] = 1 + n + h + w + c  ##+ torch.rand(1) * 0.15

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (in_n, 1, in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    act_shape_padded = (in_n, 1, in_h * in_w, _nearest_32(in_c))
    act_padding = (0, act_shape_padded[3] - act_shape[3])
    act_padded = torch.nn.functional.pad(act_reshaped, act_padding, value=0xF7FF)
    assert act_shape_padded == act_padded.shape

    ncores_nhw = 1
    grid_size = (1, 1)
    shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 0))})
    in_nhw = in_n * in_h * in_w
    out_nhw = in_n * out_h * out_w
    ## NOTE: these should match the max_pool op code for now. Hardcoded Resnet shapes only.
    if out_nhw == 1024:
        ncores_nhw = 32
        grid_size = (12, 3)
        shard_grid = ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 1)),
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 2), ttl.tensor.CoreCoord(7, 2)),
            }
        )
    elif out_nhw == 2048 or out_nhw == 4096 or out_nhw == 8192 or out_nhw == 16384 or out_nhw == 32768:
        ncores_nhw = 64
        grid_size = (12, 6)
        shard_grid = ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 4)),
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 5), ttl.tensor.CoreCoord(3, 5)),
            }
        )
    elif (
        out_nhw == 3136
        or out_nhw == 6272
        or out_nhw == 12544
        or out_nhw == 25088
        or out_nhw == 50176
        or out_nhw == 62720
    ):
        if is_wormhole_b0():
            pytest.skip("Unsupported grid size for WH")
        ncores_nhw = 98
        grid_size = (12, 9)
        shard_grid = ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 7)),
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 8), ttl.tensor.CoreCoord(1, 8)),
            }
        )
    else:
        assert False

    sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        window_h=kernel_h,
        window_w=kernel_w,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        num_cores_h=grid_size[1],
        num_cores_w=grid_size[0],
        num_cores_nhw=ncores_nhw,
    )
    pad_val = 0xF7FF

    shard_spec = ttl.tensor.ShardSpec(
        shard_grid, [in_nhw // ncores_nhw, act_padded.shape[-1]], ttl.tensor.ShardOrientation.ROW_MAJOR, False
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, shard_spec
    )

    ttact_sharded = (
        ttl.tensor.Tensor(
            act_padded.flatten().tolist(),
            act_shape_padded,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, sharded_mem_config)
    )
    assert in_h * in_w == act_shape_padded[-2]
    assert kernel_w == kernel_h and stride_w == stride_h and pad_w == pad_h and dilation_w == dilation_h

    max_pool_reader_patterns_cache = {}
    max_pool = TTPyMaxPool(sliding_window_op_params, device, max_pool_reader_patterns_cache, pad_val=pad_val)

    out_padded = max_pool(ttact_sharded)
    out_padded = out_padded.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    # Clear the cache maps
    # halo_reader_patterns_cache.clear()
    max_pool_reader_patterns_cache.clear()

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
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    logger.info(f"Passing PCC = {passing_pcc}")
    logger.info(f"Output PCC = {output_pcc}")

    # torch.save(out_pytorch, "output.pt")
    # torch.save(golden_pytorch, "golden.pt")

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        atol = 0.35

    allclose = torch.allclose(out_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(out_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(out_pytorch, golden_pytorch)

    assert passing_pcc
    assert allclose
    assert isclose
    if dtype == ttl.tensor.DataType.BFLOAT16:
        assert isequal
