# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generic SDPA memory-config sweep for dots.ocr-style attention.

This is intentionally SDPA-only: no vision tower, no layer-41 module, no QKV/O
projection weights.  It mirrors the matmul config tests by running the same
SDPA shape with different input/output memory configurations and letting TTNN
surface unsupported combinations.

Run one config:

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_sdpa_perf.py \
        -k q_l1_height_kv_dram_interleaved_out_l1_height -s

Run with Tracy:

    python -m tracy -r -p -v -m pytest -s \
        models/experimental/tt_symbiote/tests/test_dots_ocr_sdpa_perf.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


BATCH = 1
NUM_HEADS = 8
SEQ_LEN = 1024
HEAD_DIM = 128
Q_CHUNK_SIZE = 256
K_CHUNK_SIZE = 512
GRID_X = 8
GRID_Y = 8
NUM_ITERS = 1
PCC_TARGET = 0.99
SDPA_INTERLEAVED_ONLY_REASON = (
    "ttnn.transformer.scaled_dot_product_attention prefill currently requires "
    "Q/K/V operands to be DRAM/L1 interleaved; sharded operands trip "
    "`Operands to SDPA need to be DRAM/L1 interleaved`."
)


@dataclass(frozen=True)
class SDPAConfig:
    q_mem: ttnn.MemoryConfig
    k_mem: ttnn.MemoryConfig
    v_mem: ttnn.MemoryConfig
    out_mem: ttnn.MemoryConfig
    program_config: ttnn.SDPAProgramConfig
    compute_kernel_config: ttnn.DeviceComputeKernelConfig


def _compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _program_config(grid_x: int = GRID_X, grid_y: int = GRID_Y):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        q_chunk_size=Q_CHUNK_SIZE,
        k_chunk_size=K_CHUNK_SIZE,
        exp_approx_mode=True,
    )


def _l1_sharded_mem(shape, grid_x: int, grid_y: int, strategy: ttnn.ShardStrategy):
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _dram_core_range_set(device, num_banks: int | None = None):
    dram_grid = device.dram_grid_size()
    grid_x = int(dram_grid.x)
    grid_y = int(dram_grid.y)
    total_banks = grid_x * grid_y
    if num_banks is None or num_banks >= total_banks:
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_x - 1, grid_y - 1),
                )
            }
        )
    ranges = set()
    for bank_id in range(num_banks):
        coord = ttnn.CoreCoord(bank_id % grid_x, bank_id // grid_x)
        ranges.add(ttnn.CoreRange(coord, coord))
    return ttnn.CoreRangeSet(ranges)


def _dram_width_sharded_mem(device, shape):
    """Width-shard a 4D SDPA tensor across DRAM banks.

    Keep each physical shard tile-sized.  With HEAD_DIM=128, using all DRAM
    banks can create 16-wide shards on some systems, which fails before SDPA's
    own validation.  Four banks gives 32-wide shards and lets the intended SDPA
    interleaved-only check surface.
    """
    _, heads, seq_len, head_dim = shape
    dram_grid = device.dram_grid_size()
    num_banks = min(int(dram_grid.x) * int(dram_grid.y), max(1, head_dim // 32))
    head_dim_padded = math.ceil(head_dim / (32 * num_banks)) * 32 * num_banks
    shard_shape = [heads * seq_len, head_dim_padded // num_banks]
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            _dram_core_range_set(device, num_banks), shard_shape, ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def _dram_height_sharded_mem(device, shape):
    """Height-shard a 4D SDPA tensor across all DRAM banks."""
    _, heads, seq_len, head_dim = shape
    dram_grid = device.dram_grid_size()
    num_banks = int(dram_grid.x) * int(dram_grid.y)
    height = heads * seq_len
    height_padded = math.ceil(height / (32 * num_banks)) * 32 * num_banks
    shard_shape = [height_padded // num_banks, head_dim]
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(_dram_core_range_set(device), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _cfg_all_dram_interleaved_out_l1(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return SDPAConfig(
        q_mem=ttnn.DRAM_MEMORY_CONFIG,
        k_mem=ttnn.DRAM_MEMORY_CONFIG,
        v_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


def _cfg_q_l1_height_kv_dram_interleaved_out_l1_height(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    out_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    return SDPAConfig(
        q_mem=q_mem,
        k_mem=ttnn.DRAM_MEMORY_CONFIG,
        v_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=out_mem,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


def _cfg_q_l1_height_kv_dram_width_out_l1_height(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    kv_mem = _dram_width_sharded_mem(device, shape)
    out_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    return SDPAConfig(
        q_mem=q_mem,
        k_mem=kv_mem,
        v_mem=kv_mem,
        out_mem=out_mem,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


def _cfg_q_dram_width_kv_l1_height_out_l1_height(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q_mem = _dram_width_sharded_mem(device, shape)
    kv_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    out_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    return SDPAConfig(
        q_mem=q_mem,
        k_mem=kv_mem,
        v_mem=kv_mem,
        out_mem=out_mem,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


def _cfg_q_dram_height_kv_l1_height_out_l1_height(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q_mem = _dram_height_sharded_mem(device, shape)
    kv_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    out_mem = _l1_sharded_mem(shape, GRID_X, GRID_Y, ttnn.ShardStrategy.HEIGHT)
    return SDPAConfig(
        q_mem=q_mem,
        k_mem=kv_mem,
        v_mem=kv_mem,
        out_mem=out_mem,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


def _cfg_q_l1_block_kv_dram_width_out_l1_block(device):
    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    # BLOCK sharding over an 8-wide grid would split HEAD_DIM=128 into
    # 16-wide physical shards, which is invalid for tile layout.  Keep the
    # block-sharded tensor tile-aligned so SDPA's own interleaved-only
    # validation is the error this config records.
    q_mem = _l1_sharded_mem(shape, 4, GRID_Y, ttnn.ShardStrategy.BLOCK)
    kv_mem = _dram_width_sharded_mem(device, shape)
    out_mem = _l1_sharded_mem(shape, 4, GRID_Y, ttnn.ShardStrategy.BLOCK)
    return SDPAConfig(
        q_mem=q_mem,
        k_mem=kv_mem,
        v_mem=kv_mem,
        out_mem=out_mem,
        program_config=_program_config(),
        compute_kernel_config=_compute_kernel_config(device),
    )


CONFIGS: list[tuple[str, Callable]] = [
    ("all_dram_interleaved_out_l1", _cfg_all_dram_interleaved_out_l1),
    # ("q_l1_height_kv_dram_interleaved_out_l1_height", _cfg_q_l1_height_kv_dram_interleaved_out_l1_height),
    # ("q_l1_height_kv_dram_width_out_l1_height", _cfg_q_l1_height_kv_dram_width_out_l1_height),
    # ("q_dram_width_kv_l1_height_out_l1_height", _cfg_q_dram_width_kv_l1_height_out_l1_height),
    # ("q_dram_height_kv_l1_height_out_l1_height", _cfg_q_dram_height_kv_l1_height_out_l1_height),
    # ("q_l1_block_kv_dram_width_out_l1_block", _cfg_q_l1_block_kv_dram_width_out_l1_block),
]
CONFIG_IDS = [name for name, _ in CONFIGS]
XFAIL_CONFIGS = {name: SDPA_INTERLEAVED_ONLY_REASON for name, _ in CONFIGS if name != "all_dram_interleaved_out_l1"}


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_dots_ocr_sdpa_memory_configs(device, config_name: str, cfg_builder: Callable):
    torch.manual_seed(0)

    shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    q_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    k_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    v_torch = torch.randn(shape, dtype=torch.bfloat16) * 0.1

    q = k = v = last_out = None
    expected_xfail_reason = XFAIL_CONFIGS.get(config_name)
    try:
        cfg = cfg_builder(device)
        q = ttnn.from_torch(
            q_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg.q_mem,
        )
        k = ttnn.from_torch(
            k_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg.k_mem,
        )
        v = ttnn.from_torch(
            v_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg.v_mem,
        )

        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=cfg.program_config,
            compute_kernel_config=cfg.compute_kernel_config,
            memory_config=cfg.out_mem,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

        start = time.perf_counter()
        for _ in range(NUM_ITERS):
            if last_out is not None:
                ttnn.deallocate(last_out)
            last_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=cfg.program_config,
                compute_kernel_config=cfg.compute_kernel_config,
                memory_config=cfg.out_mem,
            )
        ttnn.synchronize_device(device)
        avg_us = (time.perf_counter() - start) * 1e6 / NUM_ITERS
    except RuntimeError as e:
        msg = str(e)
        for tensor in (last_out, q, k, v):
            if tensor is not None:
                ttnn.deallocate(tensor)
        if expected_xfail_reason and "Operands to SDPA need to be DRAM/L1 interleaved" in msg:
            pytest.xfail(expected_xfail_reason)
        raise

    if expected_xfail_reason:
        pytest.fail(f"{config_name} unexpectedly ran; expected SDPA to reject sharded Q/K/V operands")

    out_torch = ttnn.to_torch(last_out).to(torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_torch.to(torch.float32),
        k_torch.to(torch.float32),
        v_torch.to(torch.float32),
        is_causal=False,
    )
    pcc_passed, pcc = comp_pcc(ref, out_torch, PCC_TARGET)

    logger.info(
        f"[{config_name}] shape={shape} LoFi BFP8/BFP8=>BFP8 "
        f"grid={GRID_X}x{GRID_Y} q_chunk={Q_CHUNK_SIZE} k_chunk={K_CHUNK_SIZE} "
        f"avg={avg_us:.1f} us pcc={float(pcc):.4f}"
    )

    out_shape = tuple(out_torch.shape)
    ttnn.deallocate(last_out)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    assert out_shape == shape
    assert pcc_passed, f"{config_name} PCC {float(pcc):.4f} below target {PCC_TARGET}"
