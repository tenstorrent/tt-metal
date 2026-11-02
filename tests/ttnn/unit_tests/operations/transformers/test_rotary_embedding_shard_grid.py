# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, nearest_32
from ttnn.types import BlackholeComputeKernelConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached):
    seq_len = x.shape[-2]
    cos = cos_cached[:, :, :seq_len, ...]
    sin = sin_cached[:, :, :seq_len, ...]
    return (x * cos) + (rotate_half(x) * sin)


def _hf_rope_compute_kernel_config():
    cls = BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _decode_qk_heads_mem_config(device, batch, num_heads, head_dim):
    batch_grid = ttnn.num_cores_to_corerangeset(batch, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(nearest_32(num_heads), head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _offset_height_shard_config(X):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3))})
    shard_spec = ttnn.ShardSpec(shard_grid, [32, X], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_rotary_embedding_single_tile_matches_torch(input_dtype, device):
    # X == 32 takes the single-tile matmul kernel, where the input and trans_mat CB formats only
    # coincide for bfloat16/bfloat8_b inputs.
    torch.manual_seed(0)
    x = torch.randn([1, 1, 32, 32]).bfloat16().float()
    cos_cached = torch.randn([1, 1, 2048, 32]).bfloat16().float()
    sin_cached = torch.randn([1, 1, 2048, 32]).bfloat16().float()

    xt = ttnn.from_torch(x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cost = ttnn.from_torch(cos_cached, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    sint = ttnn.from_torch(sin_cached, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.experimental.rotary_embedding(xt, cost, sint))

    p, o = comp_pcc(apply_rotary_pos_emb(x, cos_cached, sin_cached), out, 0.999)
    logger.info(o)
    assert p


@pytest.mark.parametrize("X", [128, 32])
def test_rotary_embedding_prefill_offset_shard_grid(X, device):
    # Shard grid not rooted at (0, 0): runtime args must land on the shard cores, on both the
    # cache-miss and cache-hit paths.
    torch.manual_seed(0)
    in_mem_config = _offset_height_shard_config(X)
    cos_cached = torch.randn([1, 1, 2048, X]).bfloat16().float()
    sin_cached = torch.randn([1, 1, 2048, X]).bfloat16().float()
    cost = ttnn.from_torch(cos_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sint = ttnn.from_torch(sin_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for _ in range(2):
        x = torch.randn([1, 1, 128, X]).bfloat16().float()
        xt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem_config
        )
        out = ttnn.experimental.rotary_embedding(xt, cost, sint, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        p, o = comp_pcc(apply_rotary_pos_emb(x, cos_cached, sin_cached), ttnn.to_torch(out))
        logger.info(o)
        assert p


@pytest.mark.parametrize("X", [32, 64])
def test_rotary_embedding_hf_prefill_offset_shard_grid(X, device):
    # Shard grid not rooted at (0, 0): runtime args must land on the shard cores.
    torch.manual_seed(0)
    x = torch.randn([1, 1, 128, X]).bfloat16().float()
    cos_cached = torch.randn([1, 1, 2048, X]).bfloat16().float()
    sin_cached = torch.randn([1, 1, 2048, X]).bfloat16().float()

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_offset_height_shard_config(X)
    )
    cost = ttnn.from_torch(cos_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sint = ttnn.from_torch(sin_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.rotary_embedding_hf(
        xt,
        cost,
        sint,
        is_decode_mode=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_hf_rope_compute_kernel_config(),
    )

    p, o = comp_pcc(apply_rotary_pos_emb(x, cos_cached, sin_cached), ttnn.to_torch(out))
    logger.info(o)
    assert p


def _decode_hf_inputs(device, trig_mem):
    batch, num_heads, head_dim = 8, 8, 64
    input_tensor = ttnn.from_torch(
        torch.randn(1, batch, nearest_32(num_heads), head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_decode_qk_heads_mem_config(device, batch, num_heads, head_dim),
    )
    cos_tt, sin_tt = (
        ttnn.from_torch(
            torch.randn(1, batch, ttnn.TILE_SIZE, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=trig_mem,
        )
        for _ in range(2)
    )
    return input_tensor, cos_tt, sin_tt


def test_rotary_embedding_hf_decode_rejects_cos_sin_grid_mismatch(device, expect_error):
    # The sharded decode factory binds the cos/sin CBs on the input's cores, so the shard grids must match.
    shifted_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(7, 1))})
    trig_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shifted_grid, [ttnn.TILE_SIZE, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tensor, cos_tt, sin_tt = _decode_hf_inputs(device, trig_mem)

    with expect_error(RuntimeError, "Cos/sin shard grid .* must match input shard grid"):
        ttnn.experimental.rotary_embedding_hf(
            input_tensor, cos_tt, sin_tt, is_decode_mode=True, compute_kernel_config=_hf_rope_compute_kernel_config()
        )


def test_rotary_embedding_hf_decode_rejects_width_sharded_cos_sin(device, expect_error):
    # The sharded decode factory reads full head_dim rows of cos/sin per core, so they must be height sharded.
    trig_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    trig_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(trig_grid, [8 * ttnn.TILE_SIZE, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tensor, cos_tt, sin_tt = _decode_hf_inputs(device, trig_mem)

    with expect_error(RuntimeError, "Cos/sin must be HEIGHT_SHARDED in decode mode"):
        ttnn.experimental.rotary_embedding_hf(
            input_tensor, cos_tt, sin_tt, is_decode_mode=True, compute_kernel_config=_hf_rope_compute_kernel_config()
        )


def test_rotary_embedding_hf_decode_rejects_cos_sin_orientation_mismatch(device, expect_error):
    batch_grid = ttnn.num_cores_to_corerangeset(8, device.compute_with_storage_grid_size(), row_wise=True)
    trig_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(batch_grid, [ttnn.TILE_SIZE, 64], ttnn.ShardOrientation.COL_MAJOR),
    )
    input_tensor, cos_tt, sin_tt = _decode_hf_inputs(device, trig_mem)

    with expect_error(RuntimeError, "Cos/sin shard orientation .* must match input shard orientation"):
        ttnn.experimental.rotary_embedding_hf(
            input_tensor, cos_tt, sin_tt, is_decode_mode=True, compute_kernel_config=_hf_rope_compute_kernel_config()
        )


def test_rotary_embedding_hf_decode_rejects_oversized_cos_sin_shard(device, expect_error):
    # Eight users on eight cores read one cos/sin tile row per core, so a two-row shard would leave
    # the later users reading padding.
    batch_grid = ttnn.num_cores_to_corerangeset(8, device.compute_with_storage_grid_size(), row_wise=True)
    trig_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(batch_grid, [2 * ttnn.TILE_SIZE, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tensor, cos_tt, sin_tt = _decode_hf_inputs(device, trig_mem)

    with expect_error(RuntimeError, "Cos/sin shard height 64 must be 32"):
        ttnn.experimental.rotary_embedding_hf(
            input_tensor, cos_tt, sin_tt, is_decode_mode=True, compute_kernel_config=_hf_rope_compute_kernel_config()
        )
