# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, divup, is_grayskull


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@pytest.mark.parametrize(
    "W, Z, Y, X",
    ([1, 1, 128, 64], [1, 71, 128, 64], [32, 1, 32, 64], [32, 71, 32, 64]),
)
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("sincos_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_rotary_embedding_prefill(W, Z, Y, X, cache_size, in_sharded, out_sharded, input_dtype, sincos_dtype, device):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    else:
        out_mem_config = ttl.tensor.MemoryConfig()

    xt = ttl.tensor.Tensor(x, input_dtype)
    if xt.get_legacy_shape()[-2] % 32 == 0 and xt.get_legacy_shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    elif input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttl.tensor.Layout.TILE:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.get_legacy_shape()[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttl.tensor.CoreRangeSet(
                ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttl.tensor.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.get_legacy_shape()[-1],
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
    else:
        xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint, output_mem_config=out_mem_config)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached)

    p, o = comp_pcc(pt_out[0], tt_got_back[0])
    logger.info(o)
    assert p


@pytest.mark.parametrize(
    "W, Z, Y, X",
    ([1, 1, 32, 64], [1, 71, 32, 64], [1, 1, 64, 64], [1, 71, 64, 64], [1, 32, 32, 64], [1, 2, 32, 64]),
)
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("token_idx", [0, 128, 129, 1024, 1025])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("sincos_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_rotary_embedding_decode(
    W, Z, Y, X, cache_size, token_idx, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    else:
        out_mem_config = ttl.tensor.MemoryConfig()

    xt = ttl.tensor.Tensor(x, input_dtype)
    if xt.get_legacy_shape()[-2] % 32 == 0 and xt.get_legacy_shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    elif input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttl.tensor.Layout.TILE:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.get_legacy_shape()[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttl.tensor.CoreRangeSet(
                ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttl.tensor.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.get_legacy_shape()[-1],
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
    else:
        xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint, token_idx, out_mem_config)
    if out_sharded:
        xtt = ttl.tensor.sharded_to_interleaved(xtt)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 128, 64)])
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("in_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.FLOAT32])
@pytest.mark.parametrize("sincos_dtype", [ttl.tensor.DataType.FLOAT32])
def test_rotary_embedding_prefill_fp32(
    W, Z, Y, X, cache_size, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    else:
        out_mem_config = ttl.tensor.MemoryConfig()

    xt = ttl.tensor.Tensor(x, input_dtype)
    if xt.get_legacy_shape()[-2] % 32 == 0 and xt.get_legacy_shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    elif input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttl.tensor.Layout.TILE:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.get_legacy_shape()[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttl.tensor.CoreRangeSet(
                ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttl.tensor.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.get_legacy_shape()[-1],
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
    else:
        xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint, output_mem_config=out_mem_config)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached)

    p, o = comp_pcc(pt_out[0], tt_got_back[0])
    logger.info(o)
    assert p


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 32, 64)])
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("token_idx", [0, 128])
@pytest.mark.parametrize("in_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.FLOAT32])
@pytest.mark.parametrize("sincos_dtype", [ttl.tensor.DataType.FLOAT32])
def test_rotary_embedding_decode_fp32(
    W, Z, Y, X, cache_size, token_idx, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    else:
        out_mem_config = ttl.tensor.MemoryConfig()

    xt = ttl.tensor.Tensor(x, input_dtype)
    if xt.get_legacy_shape()[-2] % 32 == 0 and xt.get_legacy_shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    elif input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttl.tensor.Layout.TILE:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.get_legacy_shape()[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttl.tensor.CoreRangeSet(
                ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttl.tensor.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.get_legacy_shape()[-1],
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
    else:
        xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, sincos_dtype).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint, token_idx, out_mem_config)
    if out_sharded:
        xtt = ttl.tensor.sharded_to_interleaved(xtt)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p
