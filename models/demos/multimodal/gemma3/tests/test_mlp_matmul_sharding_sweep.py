# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone sweep of sharding/program-config strategies for the two Gemma3 prefill MLP matmuls
seen in Tracy:

    FF1/FF3 (gate/up):  b={2} x 512 x 2560 x 10240   (LoFi, BFP4 weight)
    FF2     (down):     128 x 10240 x 2560            (HiFi2, BF16 weight)

Each strategy is a self-contained recipe of (input mem config, weight mem config, output mem config,
program config). This keeps the config families comparable in one parametrized test:

  - dram_2d_8x8    : baseline / current model path. DRAM-interleaved activation/output, DRAM
                     width-sharded weight, 2D multicast on the 8x8 grid (FF1/FF3 -> 64 active cores,
                     FF2 -> 32).
  - dram_2d_13x10  : increased-core-count variant of the current model path. Same layout, 2D multicast
                     program config on the full Blackhole 13x10 (130-core) grid.
  - l1_width_1d    : L1 width-sharded activation/output, DRAM-INTERLEAVED weight, 1D multicast
                     (mcast_in0) on an 80-core grid. The configuration that gave the best FF2 perf.

The test is shape/validity + perf oriented (zeros input, output-shape assert only); run it under Tracy
to compare per-strategy device time. Numerical correctness (PCC) is intentionally out of scope here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import get_out_subblock_w

TILE_SIZE = 32


@dataclass(frozen=True)
class Gemma3MlpMatmulCase:
    name: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    weight_dtype: ttnn.DataType
    output_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    math_approx_mode: bool
    fp32_dest_acc_en: bool
    profiler_active_cores: int  # reference only: active cores in the current 8x8 model path

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (*self.input_shape[:-1], self.weight_shape[-1])

    @property
    def m(self) -> int:
        return self.input_shape[-2]

    @property
    def k(self) -> int:
        return self.input_shape[-1]

    @property
    def n(self) -> int:
        return self.weight_shape[-1]

    @property
    def flattened_height(self) -> int:
        return math.prod(self.input_shape[:-1])


GEMMA3_MLP_MATMUL_CASES = (
    Gemma3MlpMatmulCase(
        name="ff1_ff3_prefill_b2_512_2560_10240",
        input_shape=(1, 2, 512, 2560),
        weight_shape=(1, 1, 2560, 10240),
        weight_dtype=ttnn.bfloat4_b,
        output_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        profiler_active_cores=64,
    ),
    Gemma3MlpMatmulCase(
        name="ff2_prefill_128_10240_2560",
        input_shape=(1, 1, 128, 10240),
        weight_shape=(1, 1, 10240, 2560),
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        profiler_active_cores=32,
    ),
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _largest_divisor(value: int, max_divisor: int = 8) -> int:
    for divisor in range(max_divisor, 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _core_range_set(grid: tuple[int, int]) -> ttnn.CoreRangeSet:
    grid_cols, grid_rows = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))})


def _skip_if_grid_exceeds_device(device, grid: tuple[int, int]) -> None:
    device_grid = device.compute_with_storage_grid_size()
    grid_cols, grid_rows = grid
    if grid_cols > device_grid.x or grid_rows > device_grid.y:
        pytest.skip(f"grid={grid} exceeds device grid=({device_grid.x}, {device_grid.y})")


def _dram_width_sharded_weight_config(device, weight_shape: tuple[int, int, int, int]) -> ttnn.MemoryConfig:
    """The model's real FFN weight layout: width-sharded across DRAM banks."""
    k = weight_shape[-2]
    n = weight_shape[-1]
    dram_grid_size = device.dram_grid_size()
    if dram_grid_size.y != 1:
        pytest.skip(f"Gemma3 MLP weight sharding assumes a 1D DRAM grid, got {dram_grid_size}")
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    padded_n = math.ceil(n / (TILE_SIZE * dram_grid_size.x)) * (TILE_SIZE * dram_grid_size.x)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_grid_size.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _l1_width_sharded_mem_config(shape: tuple[int, int, int, int], grid: tuple[int, int]) -> ttnn.MemoryConfig:
    """L1 width-sharded config: split the last dim across grid cores in tile-aligned shards."""
    grid_cols, grid_rows = grid
    num_cores = grid_cols * grid_rows
    height = math.prod(shape[:-1])
    width = shape[-1]
    if height % TILE_SIZE != 0:
        pytest.skip(f"height={height} must be tile-aligned")
    if width % num_cores != 0 or (width // num_cores) % TILE_SIZE != 0:
        pytest.skip(f"width={width} must split into tile-aligned shards across {num_cores} cores")
    shard_spec = ttnn.ShardSpec(_core_range_set(grid), (height, width // num_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _compute_kernel_config(device, matmul_case: Gemma3MlpMatmulCase):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=matmul_case.math_fidelity,
        math_approx_mode=matmul_case.math_approx_mode,
        fp32_dest_acc_en=matmul_case.fp32_dest_acc_en,
        packer_l1_acc=True,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuiltStrategy:
    input_mem_config: ttnn.MemoryConfig
    weight_mem_config: ttnn.MemoryConfig
    output_mem_config: ttnn.MemoryConfig
    program_config: object
    num_cores: int
    layout_summary: str


@dataclass(frozen=True)
class ShardingStrategy:
    name: str
    kind: str  # "dram_2d" | "l1_width_1d"
    grid: tuple[int, int] | None = None  # (cols, rows) for fixed-grid strategies


STRATEGIES = (
    # Baseline: the current model path / program grid (FF1/FF3 -> 64 active cores, FF2 -> 32).
    ShardingStrategy("dram_2d_8x8_64c", "dram_2d", (8, 8)),
    ShardingStrategy("dram_2d_13x10_130c", "dram_2d", (13, 10)),
    ShardingStrategy("l1_width_1d_interleaved_wt_10x8_80c", "l1_width_1d", (10, 8)),
)


def _build_dram_2d(device, case: Gemma3MlpMatmulCase, grid: tuple[int, int]) -> BuiltStrategy:
    _skip_if_grid_exceeds_device(device, grid)
    grid_cols, grid_rows = grid
    if case.k % (TILE_SIZE * grid_rows) != 0:
        pytest.skip(f"k={case.k} must divide evenly across grid_rows={grid_rows}")
    per_core_M = math.ceil(case.m / (TILE_SIZE * grid_rows))
    per_core_N = math.ceil(case.n / (TILE_SIZE * grid_cols))
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=_largest_divisor(case.k // (TILE_SIZE * grid_rows)),
        out_subblock_h=1,
        out_subblock_w=get_out_subblock_w(per_core_N, 1),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    active_m_rows = min(grid_rows, math.ceil(math.ceil(case.m / TILE_SIZE) / per_core_M))
    active_n_cols = min(grid_cols, math.ceil(math.ceil(case.n / TILE_SIZE) / per_core_N))
    return BuiltStrategy(
        input_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_mem_config=_dram_width_sharded_weight_config(device, case.weight_shape),
        output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        num_cores=active_m_rows * active_n_cols,
        layout_summary="in=dram_interleaved wt=dram_width_sharded out=dram_interleaved mm=2d_mcast",
    )


def _build_l1_width_1d(device, case: Gemma3MlpMatmulCase, grid: tuple[int, int]) -> BuiltStrategy:
    _skip_if_grid_exceeds_device(device, grid)
    grid_cols, grid_rows = grid
    num_cores = grid_cols * grid_rows
    if case.k % (TILE_SIZE * num_cores) != 0:
        pytest.skip(f"k={case.k} must divide evenly across {num_cores} width-sharded cores")
    if case.n % (TILE_SIZE * num_cores) != 0:
        pytest.skip(f"n={case.n} must divide evenly across {num_cores} width-sharded cores")
    per_core_N = case.n // (TILE_SIZE * num_cores)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=case.k // (TILE_SIZE * num_cores),
        out_subblock_h=1,
        out_subblock_w=get_out_subblock_w(per_core_N, 1),
        per_core_M=math.ceil(case.flattened_height / TILE_SIZE),
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    return BuiltStrategy(
        input_mem_config=_l1_width_sharded_mem_config(case.input_shape, grid),
        weight_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        output_mem_config=_l1_width_sharded_mem_config(case.output_shape, grid),
        program_config=program_config,
        num_cores=num_cores,
        layout_summary="in=l1_width_sharded wt=dram_interleaved out=l1_width_sharded mm=1d_mcast",
    )


_STRATEGY_BUILDERS: dict[str, Callable[..., BuiltStrategy]] = {
    "dram_2d": lambda device, case, strategy: _build_dram_2d(device, case, strategy.grid),
    "l1_width_1d": lambda device, case, strategy: _build_l1_width_1d(device, case, strategy.grid),
}


def _build_strategy(device, case: Gemma3MlpMatmulCase, strategy: ShardingStrategy) -> BuiltStrategy:
    return _STRATEGY_BUILDERS[strategy.kind](device, case, strategy)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("matmul_case", GEMMA3_MLP_MATMUL_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda strategy: strategy.name)
def test_gemma3_mlp_prefill_matmul_sharding_sweep(device, matmul_case, strategy):
    """Run one Gemma3 prefill MLP matmul under one sharding/program-config strategy."""
    torch.manual_seed(0)

    built = _build_strategy(device, matmul_case, strategy)
    compute_kernel_config = _compute_kernel_config(device, matmul_case)

    print(
        "[GEMMA3_MLP_SHARD_SWEEP] "
        f"case={matmul_case.name} strategy={strategy.name} num_cores={built.num_cores} "
        f"current_profiler_active_cores={matmul_case.profiler_active_cores} "
        f"input_shape={matmul_case.input_shape} weight_shape={matmul_case.weight_shape} "
        f"output_shape={matmul_case.output_shape} {built.layout_summary} "
        f"weight_dtype={matmul_case.weight_dtype} output_dtype={matmul_case.output_dtype} "
        f"math_fidelity={matmul_case.math_fidelity}",
        flush=True,
    )

    torch_input = torch.zeros(matmul_case.input_shape, dtype=torch.bfloat16)
    torch_weight = torch.zeros(matmul_case.weight_shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=built.input_mem_config,
    )
    tt_weight = ttnn.from_torch(
        torch_weight,
        dtype=matmul_case.weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=built.weight_mem_config,
    )

    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        dtype=matmul_case.output_dtype,
        program_config=built.program_config,
        memory_config=built.output_mem_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.synchronize_device(device)

    assert tuple(tt_output.shape) == matmul_case.output_shape

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_input)
