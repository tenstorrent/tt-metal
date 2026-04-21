# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CI-style coverage for ``ttnn.experimental.rotary_embedding_hf`` mirroring ``test_rotary_embedding.py``.

Layout mapping from the legacy rotary tests (``[W, Z, Y, X]`` with cos/sin ``[1, 1, cache, X]``):

- **Prefill** (``is_decode=False``): HF expects ``[1, num_heads, seq, head_dim]``. Treat ``num_heads = W * Z``,
  ``seq = Y``, ``head_dim = X`` by reshaping activations to ``[1, W * Z, Y, X]``. Cos/sin stay
  ``[1, 1, cache_size, X]`` (kernel requires ``cache_size >= Y``).

- **Decode-style legacy cases** (single cos/sin position ``token_idx`` broadcast across ``Y``): the HF decode API
  is ``[1, batch, num_heads, head_dim]`` with sharded tensors, which does not map 1:1 onto every legacy
  ``Y > 1`` decode tensor. Those cases are covered here by the **same** golden math using prefill mode with
  cos/sin tiled to ``[1, 1, Y, X]`` where every sequence row matches ``token_idx`` (equivalent broadcast to
  legacy ``rotary_embedding(..., token_idx)``).

The HF op requires TILE layout and ``head_dim`` divisible by 64; ``test_rotary_embedding_hf_row_major`` skips
because row-major inputs are rejected by validation.

See ``ttnn/cpp/.../rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp`` for shape rules.
"""

import torch
import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, divup, is_blackhole
from ttnn.types import BlackholeComputeKernelConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    """Legacy-style reference (matches ``test_rotary_embedding``)."""
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def _hf_rope_compute_kernel_config():
    cls = BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _to_hf_prefill_shape(x_wzyd: torch.Tensor) -> torch.Tensor:
    w, z, y, x = x_wzyd.shape
    return x_wzyd.reshape(1, w * z, y, x)


def _from_hf_prefill_shape(x_1hsd: torch.Tensor, w: int, z: int) -> torch.Tensor:
    _, h, y, x = x_1hsd.shape
    assert h == w * z
    return x_1hsd.reshape(w, z, y, x)


@pytest.mark.parametrize(
    "W, Z, Y, X",
    ([1, 1, 128, 64], [1, 71, 128, 64], [32, 1, 32, 64], [32, 71, 32, 64]),
)
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("sincos_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_rotary_embedding_hf_prefill(
    W, Z, Y, X, cache_size, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    else:
        out_mem_config = ttnn.MemoryConfig()

    x_hf = _to_hf_prefill_shape(x)
    xt = ttnn.Tensor(x_hf, input_dtype)
    if xt.padded_shape[-2] % 32 == 0 and xt.padded_shape[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttnn.TILE_LAYOUT:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.padded_shape[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)
    else:
        xt = xt.to(device)

    cost = ttnn.Tensor(cos_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    rope_cfg = _hf_rope_compute_kernel_config()
    xtt = ttnn.experimental.rotary_embedding_hf(
        xt,
        cost,
        sint,
        is_decode=False,
        memory_config=out_mem_config,
        compute_kernel_config=rope_cfg,
    )

    tt_got_back = _from_hf_prefill_shape(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(), W, Z)

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
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("sincos_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_rotary_embedding_hf_decode(
    W, Z, Y, X, cache_size, token_idx, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    else:
        out_mem_config = ttnn.MemoryConfig()

    x_hf = _to_hf_prefill_shape(x)
    cos_row = cos_cached[:, :, token_idx : token_idx + 1, :].expand(1, 1, Y, -1).contiguous()
    sin_row = sin_cached[:, :, token_idx : token_idx + 1, :].expand(1, 1, Y, -1).contiguous()

    xt = ttnn.Tensor(x_hf, input_dtype)
    if xt.padded_shape[-2] % 32 == 0 and xt.padded_shape[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttnn.TILE_LAYOUT:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.padded_shape[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)
    else:
        xt = xt.to(device)

    cost = ttnn.Tensor(cos_row, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_row, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    rope_cfg = _hf_rope_compute_kernel_config()
    xtt = ttnn.experimental.rotary_embedding_hf(
        xt,
        cost,
        sint,
        is_decode=False,
        memory_config=out_mem_config,
        compute_kernel_config=rope_cfg,
    )
    if out_sharded:
        xtt = ttnn.sharded_to_interleaved(xtt)

    tt_got_back = _from_hf_prefill_shape(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(), W, Z)

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p


@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 128, 64)])
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("in_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("sincos_dtype", [ttnn.float32])
def test_rotary_embedding_hf_prefill_fp32(
    W, Z, Y, X, cache_size, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    else:
        out_mem_config = ttnn.MemoryConfig()

    x_hf = _to_hf_prefill_shape(x)
    xt = ttnn.Tensor(x_hf, input_dtype)
    if xt.padded_shape[-2] % 32 == 0 and xt.padded_shape[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttnn.TILE_LAYOUT:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.padded_shape[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)
    else:
        xt = xt.to(device)

    cost = ttnn.Tensor(cos_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    rope_cfg = _hf_rope_compute_kernel_config()
    xtt = ttnn.experimental.rotary_embedding_hf(
        xt,
        cost,
        sint,
        is_decode=False,
        memory_config=out_mem_config,
        compute_kernel_config=rope_cfg,
    )

    tt_got_back = _from_hf_prefill_shape(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(), W, Z)

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached)

    p, o = comp_pcc(pt_out[0], tt_got_back[0])
    logger.info(o)
    assert p


@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 32, 64)])
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("token_idx", [0, 128])
@pytest.mark.parametrize("in_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("sincos_dtype", [ttnn.float32])
def test_rotary_embedding_hf_decode_fp32(
    W, Z, Y, X, cache_size, token_idx, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()

    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    else:
        out_mem_config = ttnn.MemoryConfig()

    x_hf = _to_hf_prefill_shape(x)
    cos_row = cos_cached[:, :, token_idx : token_idx + 1, :].expand(1, 1, Y, -1).contiguous()
    sin_row = sin_cached[:, :, token_idx : token_idx + 1, :].expand(1, 1, Y, -1).contiguous()

    xt = ttnn.Tensor(x_hf, input_dtype)
    if xt.padded_shape[-2] % 32 == 0 and xt.padded_shape[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()

    if in_sharded or out_sharded:
        if xt.get_layout() != ttnn.TILE_LAYOUT:
            pytest.skip("Sharding support required tile size")
        num_blocks = xt.volume() // xt.padded_shape[-1] // 32
        compute_grid_size = device.compute_with_storage_grid_size()
        for i in range(compute_grid_size.x * compute_grid_size.y, 0, -1):
            if num_blocks % i == 0:
                num_cores = i
                break

        if in_sharded:
            Ht = divup(num_blocks, num_cores)
            shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
                [
                    Ht * 32,
                    xt.padded_shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)
    else:
        xt = xt.to(device)

    cost = ttnn.Tensor(cos_row, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_row, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    rope_cfg = _hf_rope_compute_kernel_config()
    xtt = ttnn.experimental.rotary_embedding_hf(
        xt,
        cost,
        sint,
        is_decode=False,
        memory_config=out_mem_config,
        compute_kernel_config=rope_cfg,
    )
    if out_sharded:
        xtt = ttnn.sharded_to_interleaved(xtt)

    tt_got_back = _from_hf_prefill_shape(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(), W, Z)

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p


@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 64, 64)])
@pytest.mark.parametrize("cache_size", [2048])
def test_rotary_embedding_hf_row_major(W, Z, Y, X, cache_size, device):
    """``rotary_embedding_hf`` validates TILE layout; row-major inputs are unsupported."""
    pytest.skip("rotary_embedding_hf requires TILE layout inputs (see device op validate)")
