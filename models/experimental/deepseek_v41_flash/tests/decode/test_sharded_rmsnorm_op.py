# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit and perf tests for ``ttnn.experimental.deepseek.sharded_rmsnorm``.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_sharded_rmsnorm_op.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm

PCC = 0.9999
EPS = 1e-6


def _torch_rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x_f32 = x.float()
    w_f32 = weight.float()
    return x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps) * w_f32


def _make_width_sharded_mem_config(core_grid: ttnn.CoreRangeSet, shard_h: int, shard_w: int) -> ttnn.MemoryConfig:
    shard_spec = ttnn.ShardSpec(core_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _make_inputs(device, d: int, core_grid: ttnn.CoreRangeSet, input_layout: ttnn.Layout, seed: int = 0):
    torch.manual_seed(seed)
    num_cores = core_grid.num_cores()
    assert d % num_cores == 0
    shard_w = d // num_cores
    assert shard_w % ttnn.TILE_SIZE == 0

    x = torch.randn(1, 1, 1, d, dtype=torch.bfloat16)
    w = torch.randn(1, 1, 1, d, dtype=torch.bfloat16)

    in_shard_h = ttnn.TILE_SIZE if input_layout == ttnn.TILE_LAYOUT else 1
    in_mem = _make_width_sharded_mem_config(core_grid, in_shard_h, shard_w)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=in_mem)
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x, w, x_tt, w_tt


def _resolve_output_spec(core_grid: ttnn.CoreRangeSet, shard_w: int, output_mode: str):
    if output_mode == "rm_sharded":
        return _make_width_sharded_mem_config(core_grid, 1, shard_w), ttnn.ROW_MAJOR_LAYOUT
    if output_mode == "rm_dram":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
    if output_mode == "tile_sharded":
        return _make_width_sharded_mem_config(core_grid, ttnn.TILE_SIZE, shard_w), ttnn.TILE_LAYOUT
    raise ValueError(f"Unknown output_mode: {output_mode}")


@pytest.mark.parametrize("input_layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=("in_tile", "in_rm"))
@pytest.mark.parametrize("output_mode", ("rm_sharded", "rm_dram", "tile_sharded"))
def test_sharded_rmsnorm_d5120_8cores(device, reset_seeds, input_layout, output_mode):
    d = 5120
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    shard_w = d // 8

    x, w, x_tt, w_tt = _make_inputs(device, d, core_grid, input_layout, seed=42)
    ref = _torch_rmsnorm_fp32(x, w, EPS)

    out_mem, out_layout = _resolve_output_spec(core_grid, shard_w, output_mode)
    mem_arg = (
        None
        if (output_mode == "rm_sharded" and input_layout == ttnn.ROW_MAJOR_LAYOUT)
        or (output_mode == "tile_sharded" and input_layout == ttnn.TILE_LAYOUT)
        else out_mem
    )

    out_tt = ttnn.experimental.deepseek.sharded_rmsnorm(
        x_tt, w_tt, epsilon=EPS, memory_config=mem_arg, output_layout=out_layout
    )
    assert out_tt.layout == out_layout
    assert out_tt.memory_config() == out_mem

    got = ttnn.to_torch(out_tt).float().reshape(ref.shape)
    passing, pcc = comp_pcc(ref, got, PCC)
    logger.info(f"D=5120 N=8 ({input_layout}, {output_mode}): PCC {pcc}")
    assert passing, f"D=5120 N=8 ({input_layout}, {output_mode}): PCC {pcc} < {PCC}"


@pytest.mark.parametrize(
    "d, grid_kind",
    (
        (512, "rect_4x4"),
        (512, "row_16x1"),
        (128, "row_4x1"),
        (128, "non_rect_3plus1"),
    ),
)
@pytest.mark.parametrize("input_layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=("in_tile", "in_rm"))
@pytest.mark.parametrize("output_mode", ("rm_sharded", "rm_dram", "tile_sharded"))
def test_sharded_rmsnorm_geometries(device, reset_seeds, d, grid_kind, input_layout, output_mode):
    if grid_kind == "rect_4x4":
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    elif grid_kind == "row_16x1":
        core_grid = ttnn.num_cores_to_corerangeset(16, device.compute_with_storage_grid_size(), row_wise=True)
    elif grid_kind == "row_4x1":
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    elif grid_kind == "non_rect_3plus1":
        core_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1)),
            }
        )
    else:
        raise ValueError(grid_kind)

    shard_w = d // core_grid.num_cores()
    x, w, x_tt, w_tt = _make_inputs(device, d, core_grid, input_layout, seed=17)
    ref = _torch_rmsnorm_fp32(x, w, EPS)

    out_mem, out_layout = _resolve_output_spec(core_grid, shard_w, output_mode)
    out_tt = ttnn.experimental.deepseek.sharded_rmsnorm(
        x_tt, w_tt, epsilon=EPS, memory_config=out_mem, output_layout=out_layout
    )
    assert out_tt.layout == out_layout
    assert out_tt.memory_config() == out_mem

    got = ttnn.to_torch(out_tt).float().reshape(ref.shape)
    passing, pcc = comp_pcc(ref, got, PCC)
    logger.info(f"D={d} {grid_kind} ({input_layout}, {output_mode}): PCC {pcc}")
    assert passing, f"D={d} {grid_kind} ({input_layout}, {output_mode}): PCC {pcc} < {PCC}"


@pytest.mark.parametrize("output_mode", ("rm_sharded", "rm_dram", "tile_sharded"))
def test_sharded_rmsnorm_program_cache_reuse(device, reset_seeds, output_mode):
    d = 5120
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    shard_w = d // 8
    out_mem, out_layout = _resolve_output_spec(core_grid, shard_w, output_mode)

    device.enable_program_cache()
    base_entries = device.num_program_cache_entries()

    # Keep prior tensors alive on each iteration so fresh tensors allocate at distinct addresses.
    live_tensors = []
    for step in range(3):
        x, w, x_tt, w_tt = _make_inputs(device, d, core_grid, ttnn.TILE_LAYOUT, seed=100 + step)
        ref = _torch_rmsnorm_fp32(x, w, EPS)
        out_tt = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tt, w_tt, epsilon=EPS, memory_config=out_mem, output_layout=out_layout
        )
        live_tensors.extend((x_tt, w_tt, out_tt))
        got = ttnn.to_torch(out_tt).float().reshape(ref.shape)
        passing, pcc = comp_pcc(ref, got, PCC)
        logger.info(f"cache reuse step {step} ({output_mode}): PCC {pcc}")
        assert passing, f"cache reuse step {step} ({output_mode}): PCC {pcc} < {PCC}"
        assert device.num_program_cache_entries() == base_entries + 1


def test_sharded_rmsnorm_perf(device, reset_seeds):
    d = 5120
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    shard_w = d // 8
    warmup = 2
    iters = 8

    x, w, x_tile_tt, w_tt = _make_inputs(device, d, core_grid, ttnn.TILE_LAYOUT, seed=7)
    ref = _torch_rmsnorm_fp32(x, w, EPS)

    baseline_norm = DeepSeekV4RMSNorm(w.reshape(-1), EPS, device, sharded=True)
    rm_dram_mem = ttnn.DRAM_MEMORY_CONFIG
    tile_sharded_mem = _make_width_sharded_mem_config(core_grid, ttnn.TILE_SIZE, shard_w)

    for _ in range(warmup):
        y_base = baseline_norm(x_tile_tt)
        y_tile = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tile_tt, w_tt, epsilon=EPS, memory_config=tile_sharded_mem, output_layout=ttnn.TILE_LAYOUT
        )
        y_dram = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tile_tt, w_tt, epsilon=EPS, memory_config=rm_dram_mem, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.deallocate(y_base)
        ttnn.deallocate(y_tile)
        ttnn.deallocate(y_dram)

    for _ in range(iters):
        y_base = baseline_norm(x_tile_tt)
        ttnn.deallocate(y_base)

    for _ in range(iters):
        y_tile = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tile_tt, w_tt, epsilon=EPS, memory_config=tile_sharded_mem, output_layout=ttnn.TILE_LAYOUT
        )
        ttnn.deallocate(y_tile)

    for _ in range(iters):
        y_dram = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tile_tt, w_tt, epsilon=EPS, memory_config=rm_dram_mem, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.deallocate(y_dram)

    # Final correctness spot-check.
    y_tile = ttnn.experimental.deepseek.sharded_rmsnorm(
        x_tile_tt, w_tt, epsilon=EPS, memory_config=tile_sharded_mem, output_layout=ttnn.TILE_LAYOUT
    )
    y_dram = ttnn.experimental.deepseek.sharded_rmsnorm(
        x_tile_tt, w_tt, epsilon=EPS, memory_config=rm_dram_mem, output_layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.synchronize_device(device)

    for label, out_tt in (("tile_sharded", y_tile), ("rm_dram", y_dram)):
        got = ttnn.to_torch(out_tt).float().reshape(ref.shape)
        passing, pcc = comp_pcc(ref, got, PCC)
        logger.info(f"perf spot-check {label}: PCC {pcc}")
        assert passing, f"perf spot-check {label}: PCC {pcc} < {PCC}"

    if hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.ReadDeviceProfiler(device)
