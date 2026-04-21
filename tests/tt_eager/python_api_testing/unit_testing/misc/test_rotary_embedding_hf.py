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

- **Prefill vs batch / positions**: In this codebase prefill tensors are ``[1, num_heads, seq_len, head_dim]``;
  the leading ``1`` is not a multi-user batch axis. Different sequence positions use different rows of
  ``cos/sin`` along ``seq_len``—there is no shared ``token_idx`` across users. Per-batch **decode** positions
  (different users at different cached steps) are what ``is_decode=True`` and ``[1, batch, …]`` cos/sin fix
  relative to the legacy op (see ``test_rotary_embedding_hf_decode_per_batch_position``).

The HF op requires TILE layout and ``head_dim`` divisible by 64; ``test_rotary_embedding_hf_row_major`` skips
because row-major inputs are rejected by validation.

See ``ttnn/cpp/.../rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp`` for shape rules.
"""

import torch
import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, divup, is_blackhole, nearest_32
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


# Thirty-two cache indices in [0, 2048), non-contiguous and spread out (not ``torch.arange(32)``).
_DECODE_PER_BATCH_POSITION_INDICES = [
    0,
    57,
    113,
    179,
    241,
    307,
    367,
    431,
    499,
    563,
    641,
    701,
    769,
    823,
    887,
    947,
    1009,
    1063,
    1129,
    1187,
    1249,
    1301,
    1367,
    1429,
    1483,
    1549,
    1601,
    1663,
    1723,
    1789,
    1847,
    2005,
]


def _torch_hf_rope_decode_broadcast_heads(x_1bhd: torch.Tensor, cos_1b1d: torch.Tensor, sin_1b1d: torch.Tensor):
    """HF decode RoPE: cos/sin ``[1, batch, 1, head_dim]`` broadcast across ``num_heads``."""
    cos_e = cos_1b1d.expand(-1, -1, x_1bhd.shape[2], -1)
    sin_e = sin_1b1d.expand(-1, -1, x_1bhd.shape[2], -1)
    return (x_1bhd * cos_e) + (rotate_half(x_1bhd) * sin_e)


def _decode_qk_heads_mem_config(device, batch: int, num_heads: int, head_dim: int) -> ttnn.MemoryConfig:
    """HEIGHT-sharded L1 for decode Q/K (aligned with ``test_rotary_embedding_hf_decode_baseline``)."""
    padded_heads = nearest_32(num_heads)
    shard_h = padded_heads
    if is_blackhole():
        return ttnn.create_sharded_memory_config(
            shape=(shard_h, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_hf_cos_sin_sharded(device, batch: int, head_dim: int, cos_torch, sin_torch, *, dtype):
    """Shard cos/sin like ``HfRotarySetupNew.get_rot_mats`` (HEIGHT ``(TILE_SIZE, head_dim)`` on batch grid)."""
    core_grid = device.compute_with_storage_grid_size()
    num_cores = min(batch, core_grid.x * core_grid.y)
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pad_h = ttnn.TILE_SIZE - cos_torch.shape[2]
    if pad_h > 0:
        z = torch.zeros(1, batch, pad_h, head_dim, dtype=cos_torch.dtype, device=cos_torch.device)
        cos_torch = torch.cat([cos_torch, z], dim=2)
        sin_torch = torch.cat([sin_torch, z], dim=2)
    cos_interleaved = ttnn.from_torch(
        cos_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_interleaved = ttnn.from_torch(
        sin_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if batch % ttnn.TILE_SIZE != 0:
        cos_interleaved = cos_interleaved[:, :batch, :, :]
        sin_interleaved = sin_interleaved[:, :batch, :, :]
    cos_tensor = ttnn.interleaved_to_sharded(cos_interleaved, mem_config)
    sin_tensor = ttnn.interleaved_to_sharded(sin_interleaved, mem_config)
    return cos_tensor, sin_tensor


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


def test_rotary_embedding_hf_decode_per_batch_position(device):
    """Decode mode: each batch row uses a different cached position (legacy op could not do this).

    ``cos_cache`` / ``sin_cache`` are ``[1, batch, 1, head_dim]`` with one row per user; positions are taken
    from a full ``[1, 1, cache_size, head_dim]`` cache at thirty-two **non-contiguous** indices spread across
    ``[0, cache_size)`` (see ``_DECODE_PER_BATCH_POSITION_INDICES``).
    """
    assert len(_DECODE_PER_BATCH_POSITION_INDICES) == 32
    torch.manual_seed(0)

    batch = 32
    num_heads = 8
    head_dim = 128
    cache_size = 2048
    dtype = ttnn.bfloat16

    for pos in _DECODE_PER_BATCH_POSITION_INDICES:
        assert 0 <= pos < cache_size

    cos_full = torch.randn(1, 1, cache_size, head_dim, dtype=torch.float32)
    sin_full = torch.randn(1, 1, cache_size, head_dim, dtype=torch.float32)

    cos_rows = torch.stack([cos_full[0, 0, pos, :] for pos in _DECODE_PER_BATCH_POSITION_INDICES], dim=0)
    sin_rows = torch.stack([sin_full[0, 0, pos, :] for pos in _DECODE_PER_BATCH_POSITION_INDICES], dim=0)
    cos_1b1d = cos_rows.unsqueeze(0).unsqueeze(2)  # [1, batch, 1, head_dim]
    sin_1b1d = sin_rows.unsqueeze(0).unsqueeze(2)

    torch_input = torch.randn(1, batch, num_heads, head_dim, dtype=torch.float32)
    torch_golden = _torch_hf_rope_decode_broadcast_heads(torch_input, cos_1b1d, sin_1b1d)

    padded_heads = nearest_32(num_heads)
    inp_for_dev = torch_input
    if padded_heads != num_heads:
        pad_h = padded_heads - num_heads
        z = torch.zeros(1, batch, pad_h, head_dim, dtype=torch_input.dtype)
        inp_for_dev = torch.cat([torch_input, z], dim=2)

    qk_mem = _decode_qk_heads_mem_config(device, batch, num_heads, head_dim)
    input_tensor = ttnn.from_torch(
        inp_for_dev.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qk_mem,
    )
    cos_tt, sin_tt = _decode_hf_cos_sin_sharded(
        device,
        batch,
        head_dim,
        cos_1b1d.to(torch.bfloat16),
        sin_1b1d.to(torch.bfloat16),
        dtype=dtype,
    )

    rope_cfg = _hf_rope_compute_kernel_config()
    out_tt = ttnn.experimental.rotary_embedding_hf(
        input_tensor,
        cos_tt,
        sin_tt,
        is_decode=True,
        compute_kernel_config=rope_cfg,
    )
    out_torch = ttnn.to_torch(out_tt).to(torch.float32)
    if padded_heads != num_heads:
        out_torch = out_torch[:, :, :num_heads, :]

    p, o = comp_pcc(torch_golden, out_torch)
    logger.info(o)
    assert p

    ttnn.deallocate(out_tt)
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)


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
