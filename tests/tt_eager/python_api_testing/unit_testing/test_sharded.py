# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def test_sharded_tile(device):
    N = 1
    C = 1
    H = 100352
    W = 64
    num_cores = 98
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, num_cores, [H // num_cores, W], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    assert torch.equal(tt_og, tt_got_back)


def test_sharded_rm(device):
    N = 1
    C = 1
    H = 100352
    W = 64
    num_cores = 98
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, num_cores, [H // num_cores, W], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(tt_og, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[100352, 98], [25088, 98]])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_sharded_untilize(H, num_cores, in_sharded, out_sharded, device):
    N = 1
    C = 1
    W = 64
    if out_sharded and not in_sharded and H == 100352:
        pytest.skip("Unsupported config for sharding")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            num_cores,
            [H // num_cores, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    passing, output = comp_equal(x, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
def test_sharded_tilize(H, num_cores, device):
    N = 1
    C = 1
    W = 64
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        num_cores,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    yt_tilized = ttl.tensor.tilize(
        yt,
        output_mem_config=ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
        use_multicore=True,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt_tilized,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    passing, output = comp_equal(x, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize(
    "in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"]
)
@pytest.mark.parametrize(
    "out_sharded", [True, False], ids=["out_sharded", "out_unsharded"]
)
@pytest.mark.parametrize("M, num_cores", [[25088, 98]])
@pytest.mark.parametrize("N", [64, 256])
def test_sharded_matmul(device, in0_sharded, out_sharded, M, N, num_cores):
    K = 64
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            num_cores,
            [M // num_cores, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        )

    program_config = ttl.operations.primary.get_mcast_1d_config(
        in0_t, in1_t, True, None, False, out_sharded
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skip("Sharded tensors do not work with program cache")
def test_sharded_program_cache(device, use_program_cache):
    N = 1
    C = 1
    H = 25088
    W = 64
    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16().float()
    x2 = (-1 * torch.arange(N * C * H * W)).reshape((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, 98, [H // 98, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    xt2 = (
        ttl.tensor.Tensor(
            x2.reshape(-1).tolist(),
            x2.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt2 = ttl.tensor.interleaved_to_sharded(
        xt2, 98, [H // 98, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    zt2 = ttl.tensor.sharded_to_interleaved(
        yt2,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )
    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_og2 = xt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_got_back2 = zt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    assert torch.equal(tt_og, tt_got_back)
    assert torch.equal(tt_og2, tt_got_back2)
