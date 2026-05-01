# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Isolated BGE-M3 B32/S512 MLP matmul program-config sweep.

This intentionally tests the MLP shapes outside the full model before porting
any config into `tt/mlp.py`. The full-model path remains the source of truth,
but this harness makes compile failures and kernel regressions much cheaper to
find.

Examples:
  TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/sweep_mlp_matmul_configs.py \
    -sv -k test_bge_m3_mlp_matmul_config_sweep

  BGE_M3_MLP_SWEEP_WORKLOAD=wi BGE_M3_MLP_SWEEP_LIMIT=0 \
    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/sweep_mlp_matmul_configs.py \
    -sv -k test_bge_m3_mlp_matmul_config_sweep

  BGE_M3_MLP_SWEEP_WORKLOAD=wi BGE_M3_MLP_SWEEP_DTYPES=bfp8,weight_bfp4,bfp4 \
    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/sweep_mlp_matmul_configs.py \
    -sv -k test_bge_m3_mlp_matmul_config_sweep

  TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/sweep_mlp_matmul_configs.py \
    -sv -k test_bge_m3_wi_layout_sweep

  BGE_M3_MLP_LAYOUT_SWEEP_WORKLOAD=wo TT_VISIBLE_DEVICES=0 \
    pytest models/demos/wormhole/bge_m3/tests/perf/sweep_mlp_matmul_configs.py \
    -sv -k test_bge_m3_mlp_layout_sweep

For device-profiler timing, export the usual Tracy profiler vars before pytest:
  export TT_METAL_DEVICE_PROFILER=1
  export TT_METAL_PROFILER_MID_RUN_DUMP=1
  export TT_METAL_PROFILER_CPP_POST_PROCESS=1
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn

BATCH_SIZE = 32
SEQ_LEN = 512
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
TILE_SIZE = 32

DEFAULT_GRIDS = ((11, 10), (8, 8), (10, 8), (8, 10), (7, 7), (5, 10), (5, 8))


@dataclass(frozen=True)
class MLPWorkload:
    name: str
    m: int
    k: int
    n: int
    fused_activation: tuple[ttnn.UnaryOpType, bool] | None
    default_activation: str | None


@dataclass(frozen=True)
class Candidate:
    name: str
    grid: tuple[int, int] | None = None
    in0_block_w: int | None = None
    out_subblock_h: int | None = None
    out_subblock_w: int | None = None
    per_core_m: int | None = None
    per_core_n: int | None = None
    out_block_h: int | None = None
    out_block_w: int | None = None
    m_policy: str = "auto"


@dataclass(frozen=True)
class DTypeCandidate:
    name: str
    input_dtype: Any
    weight_dtype: Any
    output_dtype: Any


@dataclass
class SweepResult:
    workload: str
    candidate: str
    status: str
    host_us: float | None = None
    device_us: float | None = None
    core_count: int | None = None
    error: str = ""

    def to_csv_row(self) -> str:
        host_us = "" if self.host_us is None else f"{self.host_us:.2f}"
        device_us = "" if self.device_us is None else f"{self.device_us:.2f}"
        core_count = "" if self.core_count is None else str(self.core_count)
        escaped_error = self.error.replace('"', "'")
        return f'{self.workload},{self.candidate},{self.status},{host_us},{device_us},{core_count},"{escaped_error}"'


@dataclass(frozen=True)
class LayoutCandidate:
    name: str
    in0_memory_config: Any = None
    in1_memory_config: Any = None
    output_memory_config: Any = None
    program_config: Any = None
    output_dtype: Any = ttnn.bfloat8_b
    construction_error: str = ""


@dataclass
class LayoutSweepResult:
    candidate: str
    status: str
    prepare_us: float | None = None
    host_us: float | None = None
    device_us: float | None = None
    core_count: int | None = None
    error: str = ""

    def to_csv_row(self) -> str:
        prepare_us = "" if self.prepare_us is None else f"{self.prepare_us:.2f}"
        host_us = "" if self.host_us is None else f"{self.host_us:.2f}"
        device_us = "" if self.device_us is None else f"{self.device_us:.2f}"
        core_count = "" if self.core_count is None else str(self.core_count)
        escaped_error = self.error.replace('"', "'")
        return f'{self.candidate},{self.status},{prepare_us},{host_us},{device_us},{core_count},"{escaped_error}"'


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _dtype_candidates() -> tuple[DTypeCandidate, ...]:
    value = os.environ.get("BGE_M3_MLP_SWEEP_DTYPES", "bfp8")
    mapping = {
        "bfp8": DTypeCandidate("bfp8", ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),
        # Most model-compatible BFP4 probe: keep activations/output BFP8, compress weights.
        "weight_bfp4": DTypeCandidate("weight_bfp4", ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat8_b),
        "bfp8_wbfp4": DTypeCandidate("weight_bfp4", ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat8_b),
        # Aggressive probe: both matmul operands BFP4, but keep output BFP8 for downstream compatibility.
        "bfp4": DTypeCandidate("bfp4", ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b),
        "all_bfp4": DTypeCandidate("all_bfp4", ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat4_b),
    }

    candidates: list[DTypeCandidate] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(
                f"Unsupported BGE_M3_MLP_SWEEP_DTYPES entry {item!r}. " f"Supported: {', '.join(sorted(mapping))}"
            )
        candidate = mapping[key]
        if all(existing.name != candidate.name for existing in candidates):
            candidates.append(candidate)
    return tuple(candidates) or (mapping["bfp8"],)


def _parse_grids() -> tuple[tuple[int, int], ...]:
    value = os.environ.get("BGE_M3_MLP_SWEEP_GRIDS")
    if not value:
        return DEFAULT_GRIDS
    grids = []
    for item in value.split(","):
        x, y = item.lower().split("x", maxsplit=1)
        grids.append((int(x), int(y)))
    return tuple(grids)


def _divisors_up_to(n: int, max_value: int) -> list[int]:
    divisors: set[int] = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d != 0:
            continue
        if d <= max_value:
            divisors.add(d)
        comp = n // d
        if comp <= max_value:
            divisors.add(comp)
    return sorted(divisors) or [1]


def _short_error(exc: BaseException, max_len: int = 180) -> str:
    message = str(exc).strip().split("\n")[0] if str(exc).strip() else ""
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _workloads() -> dict[str, MLPWorkload]:
    return {
        "wi": MLPWorkload(
            name="wi",
            m=SEQ_LEN,
            k=HIDDEN_SIZE,
            n=INTERMEDIATE_SIZE,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
            default_activation="gelu",
        ),
        "wo": MLPWorkload(
            name="wo",
            m=SEQ_LEN,
            k=INTERMEDIATE_SIZE,
            n=HIDDEN_SIZE,
            fused_activation=None,
            default_activation=None,
        ),
    }


def _candidate_program_config(workload: MLPWorkload, candidate: Candidate) -> Any:
    if candidate.name == "auto":
        return None

    assert candidate.grid is not None
    assert candidate.in0_block_w is not None
    assert candidate.out_subblock_h is not None
    assert candidate.out_subblock_w is not None
    assert candidate.per_core_m is not None
    assert candidate.per_core_n is not None

    kwargs: dict[str, Any] = {}
    if candidate.out_block_h is not None:
        kwargs["out_block_h"] = candidate.out_block_h
    if candidate.out_block_w is not None:
        kwargs["out_block_w"] = candidate.out_block_w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=candidate.grid,
        in0_block_w=candidate.in0_block_w,
        out_subblock_h=candidate.out_subblock_h,
        out_subblock_w=candidate.out_subblock_w,
        per_core_M=candidate.per_core_m,
        per_core_N=candidate.per_core_n,
        transpose_mcast=False,
        fused_activation=workload.fused_activation,
        fuse_batch=False,
        **kwargs,
    )


def _generate_candidates(workload: MLPWorkload) -> list[Candidate]:
    candidates = [Candidate(name="auto")]
    candidates.extend(_priority_candidates(workload))
    k_tiles = math.ceil(workload.k / TILE_SIZE)
    n_tiles = math.ceil(workload.n / TILE_SIZE)
    seq_tiles = math.ceil(workload.m / TILE_SIZE)
    batch_seq_tiles = math.ceil((BATCH_SIZE * workload.m) / TILE_SIZE)
    in0_limit = _env_int("BGE_M3_MLP_SWEEP_MAX_IN0_BLOCK_W", 8)
    subblock_limit = _env_int("BGE_M3_MLP_SWEEP_MAX_SUBBLOCK", 4)
    include_sequence_policy = os.environ.get("BGE_M3_MLP_SWEEP_INCLUDE_SEQUENCE_POLICY", "1") == "1"

    for grid in _parse_grids():
        grid_x, grid_y = grid
        per_core_n = math.ceil(n_tiles / grid_x)
        in0_candidates = _divisors_up_to(k_tiles, in0_limit)
        out_subblock_w_candidates = [w for w in _divisors_up_to(per_core_n, subblock_limit) if w <= subblock_limit]

        policies: list[tuple[str, int]] = [("batch_seq", math.ceil(batch_seq_tiles / grid_y))]
        if include_sequence_policy:
            # Expected to reject on BGE's current 4D layout for B32, but it is useful
            # to keep as a sanity check against DeepSeek-style sequence-M configs.
            policies.append(("sequence", math.ceil(seq_tiles / grid_y)))

        for m_policy, per_core_m in policies:
            out_subblock_h_candidates = [h for h in _divisors_up_to(per_core_m, subblock_limit) if h <= subblock_limit]
            for in0_block_w in in0_candidates:
                for out_subblock_h in out_subblock_h_candidates:
                    for out_subblock_w in out_subblock_w_candidates:
                        if out_subblock_h * out_subblock_w > subblock_limit:
                            continue
                        base = Candidate(
                            name=(
                                f"{workload.name}_{m_policy}_g{grid_x}x{grid_y}_"
                                f"in0w{in0_block_w}_sub{out_subblock_h}x{out_subblock_w}"
                            ),
                            grid=grid,
                            in0_block_w=in0_block_w,
                            out_subblock_h=out_subblock_h,
                            out_subblock_w=out_subblock_w,
                            per_core_m=per_core_m,
                            per_core_n=per_core_n,
                            m_policy=m_policy,
                        )
                        _append_unique_candidate(candidates, base)

                        # For large B32 per-core M, explicit output block sizing can
                        # lower L1 pressure. Keep this separate so regressions are easy to attribute.
                        if m_policy == "batch_seq":
                            for out_block_h in _divisors_up_to(per_core_m, 8):
                                if out_block_h > per_core_m:
                                    continue
                                if out_block_h % out_subblock_h != 0:
                                    continue
                                _append_unique_candidate(
                                    candidates,
                                    Candidate(
                                        name=f"{base.name}_ob{out_block_h}x{per_core_n}",
                                        grid=grid,
                                        in0_block_w=in0_block_w,
                                        out_subblock_h=out_subblock_h,
                                        out_subblock_w=out_subblock_w,
                                        per_core_m=per_core_m,
                                        per_core_n=per_core_n,
                                        out_block_h=out_block_h,
                                        out_block_w=per_core_n,
                                        m_policy=m_policy,
                                    ),
                                )

    limit = _env_int("BGE_M3_MLP_SWEEP_LIMIT", 24)
    if limit > 0:
        return candidates[:limit]
    return candidates


def _append_unique_candidate(candidates: list[Candidate], candidate: Candidate) -> None:
    if any(existing.name == candidate.name for existing in candidates):
        return
    candidates.append(candidate)


def _priority_candidates(workload: MLPWorkload) -> list[Candidate]:
    """Put the current full-model neighborhood before the broad generated grid search."""
    seq_tiles = SEQ_LEN // TILE_SIZE
    n_tiles = workload.n // TILE_SIZE

    def sequence_candidate(
        grid: tuple[int, int],
        in0_block_w: int,
        out_subblock_w: int,
        name_suffix: str = "",
    ) -> Candidate:
        grid_x, grid_y = grid
        per_core_m = math.ceil(seq_tiles / grid_y)
        per_core_n = math.ceil(n_tiles / grid_x)
        suffix = f"_{name_suffix}" if name_suffix else ""
        return Candidate(
            name=(
                f"{workload.name}_priority_sequence_g{grid_x}x{grid_y}_"
                f"in0w{in0_block_w}_sub1x{out_subblock_w}{suffix}"
            ),
            grid=grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_m=per_core_m,
            per_core_n=per_core_n,
            m_policy="sequence",
        )

    if workload.name == "wi":
        return [
            # Current full-model winner plus nearby lower/higher in0 blocking.
            sequence_candidate((11, 10), 8, 3, "current"),
            sequence_candidate((11, 10), 4, 3),
            sequence_candidate((11, 10), 8, 2),
            sequence_candidate((10, 10), 8, 4),
            sequence_candidate((8, 10), 8, 4),
        ]

    return [
        # Wo stayed on TTNN auto in the full model; these are local sequence-policy probes.
        sequence_candidate((11, 10), 2, 3),
        sequence_candidate((11, 10), 4, 3),
        sequence_candidate((11, 10), 8, 3),
        sequence_candidate((10, 10), 2, 4),
        sequence_candidate((8, 10), 2, 4),
    ]


def _compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _wi_sequence_program_config() -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=2,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
        fuse_batch=False,
    )


def _mlp_sequence_program_config(workload: MLPWorkload) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    if workload.name == "wi":
        return _wi_sequence_program_config()

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=2,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _core_range_grid(x: int, y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(x - 1, y - 1),
            )
        }
    )


def _manual_sharded_memory_config(
    *,
    layout: ttnn.TensorMemoryLayout,
    buffer_type: ttnn.BufferType,
    shard_grid: tuple[int, int],
    shard_shape: list[int],
) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        layout,
        buffer_type,
        ttnn.ShardSpec(_core_range_grid(*shard_grid), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dram_width_sharded_weight_config(k: int, n: int, dram_grid_x: int) -> ttnn.MemoryConfig:
    shard_n = math.ceil(n / dram_grid_x)
    shard_n = math.ceil(shard_n / TILE_SIZE) * TILE_SIZE
    return _manual_sharded_memory_config(
        layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_grid=(dram_grid_x, 1),
        shard_shape=[k, shard_n],
    )


def _latest_device_timing_us(device_id: int = 0) -> tuple[float | None, int | None]:
    try:
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return None, None
    if not latest or device_id not in latest:
        return None, None

    duration_ns = None
    core_count = None
    for program in latest[device_id]:
        core_count = max(core_count or 0, getattr(program, "core_count", 0) or 0)
        for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
            if key not in program.program_analyses_results:
                continue
            duration = program.program_analyses_results[key].duration
            if duration is not None:
                duration_ns = max(duration_ns, duration) if duration_ns is not None else duration
            break

    return (None if duration_ns is None else duration_ns / 1000.0, core_count)


def _wi_layout_candidates(mesh_device) -> list[LayoutCandidate]:
    in0_shape = [BATCH_SIZE, 1, SEQ_LEN, HIDDEN_SIZE]
    out_shape = [BATCH_SIZE, 1, SEQ_LEN, INTERMEDIATE_SIZE]
    pc = _wi_sequence_program_config()

    height_grid = (11, 10)
    width_grid = (8, 4)
    out_width_grid = (8, 8)
    dram_grid_x = mesh_device.dram_grid_size().x

    def height_l1(shape: list[int]) -> ttnn.MemoryConfig:
        return ttnn.create_sharded_memory_config(
            shape,
            core_grid=ttnn.CoreGrid(x=height_grid[0], y=height_grid[1]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def width_l1(shape: list[int], grid: tuple[int, int]) -> ttnn.MemoryConfig:
        return ttnn.create_sharded_memory_config(
            shape,
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def dram_sharded_pc(per_core_m: int) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=per_core_m,
            per_core_N=INTERMEDIATE_SIZE // (TILE_SIZE * dram_grid_x),
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        )

    candidates: list[LayoutCandidate] = []

    def add_candidate(
        name: str,
        in0_factory,
        in1_factory,
        output_factory,
        program_factory,
        output_dtype: Any = ttnn.bfloat8_b,
    ) -> None:
        try:
            candidates.append(
                LayoutCandidate(
                    name=name,
                    in0_memory_config=in0_factory(),
                    in1_memory_config=in1_factory(),
                    output_memory_config=output_factory(),
                    program_config=program_factory(),
                    output_dtype=output_dtype,
                )
            )
        except Exception as exc:
            candidates.append(LayoutCandidate(name=name, construction_error=_short_error(exc)))

    add_candidate(
        "seq_pc_in0_dram_in1_dram_out_dram",
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_l1_in1_dram_out_dram",
        lambda: ttnn.L1_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_dram_in1_dram_out_l1",
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.L1_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_l1_in1_dram_out_l1",
        lambda: ttnn.L1_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.L1_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_height_l1_in1_dram_out_dram",
        lambda: height_l1(in0_shape),
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_height_l1_in1_dram_out_height_l1",
        lambda: height_l1(in0_shape),
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: height_l1(out_shape),
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_width_l1_in1_dram_out_dram",
        lambda: width_l1(in0_shape, width_grid),
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: pc,
    )
    add_candidate(
        "seq_pc_in0_width_l1_in1_dram_out_width_l1",
        lambda: width_l1(in0_shape, width_grid),
        lambda: ttnn.DRAM_MEMORY_CONFIG,
        lambda: width_l1(out_shape, out_width_grid),
        lambda: pc,
    )
    add_candidate(
        "dram_sharded_w_in0_width_l1_out_width_l1_mseq",
        lambda: width_l1(in0_shape, width_grid),
        lambda: _dram_width_sharded_weight_config(HIDDEN_SIZE, INTERMEDIATE_SIZE, dram_grid_x),
        lambda: ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        lambda: dram_sharded_pc(SEQ_LEN // TILE_SIZE),
    )
    add_candidate(
        "dram_sharded_w_in0_width_l1_out_width_l1_m1",
        lambda: width_l1(in0_shape, width_grid),
        lambda: _dram_width_sharded_weight_config(HIDDEN_SIZE, INTERMEDIATE_SIZE, dram_grid_x),
        lambda: ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        lambda: dram_sharded_pc(1),
    )

    return candidates


def _wo_layout_candidates(mesh_device) -> list[LayoutCandidate]:
    workload = _workloads()["wo"]
    in0_shape = [BATCH_SIZE, 1, SEQ_LEN, INTERMEDIATE_SIZE]
    out_shape = [BATCH_SIZE, 1, SEQ_LEN, HIDDEN_SIZE]
    seq_pc = _mlp_sequence_program_config(workload)

    width_grid = (8, 8)
    dram_grid_x = mesh_device.dram_grid_size().x

    def width_l1(shape: list[int], grid: tuple[int, int] = width_grid) -> ttnn.MemoryConfig:
        return ttnn.create_sharded_memory_config(
            shape,
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def dram_sharded_pc(per_core_m: int) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=per_core_m,
            per_core_N=HIDDEN_SIZE // (TILE_SIZE * dram_grid_x),
            fused_activation=None,
        )

    candidates: list[LayoutCandidate] = []

    def add_candidate(
        name: str,
        in0_factory,
        in1_factory,
        output_factory,
        program_factory,
        output_dtype: Any = ttnn.bfloat8_b,
    ) -> None:
        try:
            candidates.append(
                LayoutCandidate(
                    name=name,
                    in0_memory_config=in0_factory(),
                    in1_memory_config=in1_factory(),
                    output_memory_config=output_factory(),
                    program_config=program_factory(),
                    output_dtype=output_dtype,
                )
            )
        except Exception as exc:
            candidates.append(LayoutCandidate(name=name, construction_error=_short_error(exc)))

    for prefix, program_factory in (
        ("auto", lambda: None),
        ("seq_pc", lambda: seq_pc),
    ):
        add_candidate(
            f"{prefix}_in0_dram_in1_dram_out_dram",
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            program_factory,
        )
        add_candidate(
            f"{prefix}_in0_l1_in1_dram_out_dram",
            lambda: ttnn.L1_MEMORY_CONFIG,
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            program_factory,
        )
        add_candidate(
            f"{prefix}_in0_l1_in1_dram_out_l1",
            lambda: ttnn.L1_MEMORY_CONFIG,
            lambda: ttnn.DRAM_MEMORY_CONFIG,
            lambda: ttnn.L1_MEMORY_CONFIG,
            program_factory,
        )

    add_candidate(
        "dram_sharded_w_in0_width_l1_out_width_l1_mseq",
        lambda: width_l1(in0_shape),
        lambda: _dram_width_sharded_weight_config(INTERMEDIATE_SIZE, HIDDEN_SIZE, dram_grid_x),
        lambda: width_l1(out_shape),
        lambda: dram_sharded_pc(SEQ_LEN // TILE_SIZE),
    )
    add_candidate(
        "dram_sharded_w_in0_width_l1_out_width_l1_m1",
        lambda: width_l1(in0_shape),
        lambda: _dram_width_sharded_weight_config(INTERMEDIATE_SIZE, HIDDEN_SIZE, dram_grid_x),
        lambda: width_l1(out_shape),
        lambda: dram_sharded_pc(1),
    )

    return candidates


def _mlp_layout_candidates(mesh_device, workload: MLPWorkload) -> list[LayoutCandidate]:
    if workload.name == "wi":
        return _wi_layout_candidates(mesh_device)
    if workload.name == "wo":
        return _wo_layout_candidates(mesh_device)
    raise ValueError(f"Unsupported MLP layout workload: {workload.name}")


def _run_wi_layout_candidate(mesh_device, candidate: LayoutCandidate, iterations: int) -> LayoutSweepResult:
    return _run_mlp_layout_candidate(mesh_device, _workloads()["wi"], candidate, iterations)


def _run_mlp_layout_candidate(
    mesh_device,
    workload: MLPWorkload,
    candidate: LayoutCandidate,
    iterations: int,
) -> LayoutSweepResult:
    if candidate.construction_error:
        return LayoutSweepResult(candidate=candidate.name, status="reject", error=candidate.construction_error)

    torch.manual_seed(1234)
    activation = torch.randn((BATCH_SIZE, 1, workload.m, workload.k), dtype=torch.bfloat16)
    weight = torch.randn((1, 1, workload.k, workload.n), dtype=torch.bfloat16)

    try:
        prepare_start = time.perf_counter()
        input_tensor = ttnn.from_torch(
            activation,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=candidate.in0_memory_config,
        )
        weight_tensor = ttnn.from_torch(
            weight,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=candidate.in1_memory_config,
        )
        ttnn.synchronize_device(mesh_device)
        prepare_us = (time.perf_counter() - prepare_start) * 1e6

        # Compile once outside the measured loop.
        output = ttnn.linear(
            input_tensor,
            weight_tensor,
            memory_config=candidate.output_memory_config,
            dtype=candidate.output_dtype,
            program_config=candidate.program_config,
            compute_kernel_config=_compute_kernel_config(),
            activation=workload.default_activation if candidate.program_config is None else None,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        start = time.perf_counter()
        for _ in range(iterations):
            output = ttnn.linear(
                input_tensor,
                weight_tensor,
                memory_config=candidate.output_memory_config,
                dtype=candidate.output_dtype,
                program_config=candidate.program_config,
                compute_kernel_config=_compute_kernel_config(),
                activation=workload.default_activation if candidate.program_config is None else None,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)
        host_us = ((time.perf_counter() - start) / iterations) * 1e6

        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass
        device_us, core_count = _latest_device_timing_us()
        return LayoutSweepResult(
            candidate=candidate.name,
            status="pass",
            prepare_us=prepare_us,
            host_us=host_us,
            device_us=device_us,
            core_count=core_count,
        )
    except Exception as exc:
        return LayoutSweepResult(candidate=candidate.name, status="reject", error=_short_error(exc))


def _run_candidate(
    mesh_device,
    workload: MLPWorkload,
    candidate: Candidate,
    dtype_candidate: DTypeCandidate,
    iterations: int,
) -> SweepResult:
    torch.manual_seed(1234)
    activation = torch.randn((BATCH_SIZE, 1, workload.m, workload.k), dtype=torch.bfloat16)
    weight = torch.randn((1, 1, workload.k, workload.n), dtype=torch.bfloat16)
    candidate_name = f"{candidate.name}_{dtype_candidate.name}"

    try:
        input_tensor = ttnn.from_torch(
            activation,
            device=mesh_device,
            dtype=dtype_candidate.input_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weight_tensor = ttnn.from_torch(
            weight,
            device=mesh_device,
            dtype=dtype_candidate.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        program_config = _candidate_program_config(workload, candidate)

        # Compile once outside the measured loop.
        output = ttnn.linear(
            input_tensor,
            weight_tensor,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype_candidate.output_dtype,
            program_config=program_config,
            compute_kernel_config=_compute_kernel_config(),
            activation=workload.default_activation if program_config is None else None,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        start = time.perf_counter()
        for _ in range(iterations):
            output = ttnn.linear(
                input_tensor,
                weight_tensor,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype_candidate.output_dtype,
                program_config=program_config,
                compute_kernel_config=_compute_kernel_config(),
                activation=workload.default_activation if program_config is None else None,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)
        host_us = ((time.perf_counter() - start) / iterations) * 1e6

        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass
        device_us, core_count = _latest_device_timing_us()

        return SweepResult(
            workload=workload.name,
            candidate=candidate_name,
            status="pass",
            host_us=host_us,
            device_us=device_us,
            core_count=core_count,
        )
    except Exception as exc:
        return SweepResult(
            workload=workload.name,
            candidate=candidate_name,
            status="reject",
            error=_short_error(exc),
        )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bge_m3_mlp_matmul_config_sweep(mesh_device):
    workload_filter = os.environ.get("BGE_M3_MLP_SWEEP_WORKLOAD", "all")
    iterations = _env_int("BGE_M3_MLP_SWEEP_ITERS", 5)
    dtype_candidates = _dtype_candidates()
    workloads = _workloads()
    selected = workloads.values() if workload_filter == "all" else [workloads[workload_filter]]

    print("workload,candidate,status,host_us,device_us,core_count,error", flush=True)
    best_by_workload: dict[str, SweepResult] = {}
    pass_count = 0

    for workload in selected:
        candidates = _generate_candidates(workload)
        logger.info(
            f"Sweeping {workload.name}: {len(candidates)} candidates, "
            f"dtypes={[candidate.name for candidate in dtype_candidates]}, iterations={iterations}"
        )
        for candidate in candidates:
            for dtype_candidate in dtype_candidates:
                result = _run_candidate(mesh_device, workload, candidate, dtype_candidate, iterations)
                print(result.to_csv_row(), flush=True)
                if result.status != "pass":
                    continue
                pass_count += 1
                best = best_by_workload.get(workload.name)
                if best is None or (
                    result.host_us is not None and best.host_us is not None and result.host_us < best.host_us
                ):
                    best_by_workload[workload.name] = result

    print("\n# Best host-time config per workload:", flush=True)
    for workload_name, result in sorted(best_by_workload.items()):
        print(
            f"# {workload_name}: {result.candidate} host_us={result.host_us:.2f} "
            f"device_us={'' if result.device_us is None else f'{result.device_us:.2f}'} cores={result.core_count}",
            flush=True,
        )

    assert pass_count > 0, "No MLP matmul sweep candidates compiled and ran"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bge_m3_wi_layout_sweep(mesh_device):
    iterations = _env_int("BGE_M3_MLP_LAYOUT_SWEEP_ITERS", 5)
    candidates = _wi_layout_candidates(mesh_device)
    filter_text = os.environ.get("BGE_M3_MLP_LAYOUT_SWEEP_FILTER")
    if filter_text:
        candidates = [candidate for candidate in candidates if filter_text in candidate.name]

    print("candidate,status,prepare_us,host_us,device_us,core_count,error", flush=True)
    best: LayoutSweepResult | None = None
    pass_count = 0
    for candidate in candidates:
        result = _run_wi_layout_candidate(mesh_device, candidate, iterations)
        print(result.to_csv_row(), flush=True)
        if result.status != "pass":
            continue
        pass_count += 1
        if best is None or (result.host_us is not None and best.host_us is not None and result.host_us < best.host_us):
            best = result

    if best is not None:
        print(
            f"\n# best: {best.candidate} host_us={best.host_us:.2f} "
            f"device_us={'' if best.device_us is None else f'{best.device_us:.2f}'} cores={best.core_count}",
            flush=True,
        )

    assert pass_count > 0, "No Wi layout sweep candidates compiled and ran"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bge_m3_mlp_layout_sweep(mesh_device):
    workload_filter = os.environ.get("BGE_M3_MLP_LAYOUT_SWEEP_WORKLOAD", "all")
    iterations = _env_int("BGE_M3_MLP_LAYOUT_SWEEP_ITERS", 5)
    workloads = _workloads()
    selected = workloads.values() if workload_filter == "all" else [workloads[workload_filter]]
    filter_text = os.environ.get("BGE_M3_MLP_LAYOUT_SWEEP_FILTER")

    print("workload,candidate,status,prepare_us,host_us,device_us,core_count,error", flush=True)
    best_by_workload: dict[str, LayoutSweepResult] = {}
    pass_count = 0

    for workload in selected:
        candidates = _mlp_layout_candidates(mesh_device, workload)
        if filter_text:
            candidates = [candidate for candidate in candidates if filter_text in candidate.name]
        logger.info(f"Sweeping {workload.name} layouts: {len(candidates)} candidates, iterations={iterations}")

        for candidate in candidates:
            result = _run_mlp_layout_candidate(mesh_device, workload, candidate, iterations)
            print(f"{workload.name},{result.to_csv_row()}", flush=True)
            if result.status != "pass":
                continue
            pass_count += 1
            best = best_by_workload.get(workload.name)
            if best is None or (
                result.host_us is not None and best.host_us is not None and result.host_us < best.host_us
            ):
                best_by_workload[workload.name] = result

    print("\n# Best host-time layout per workload:", flush=True)
    for workload_name, result in sorted(best_by_workload.items()):
        print(
            f"# {workload_name}: {result.candidate} host_us={result.host_us:.2f} "
            f"device_us={'' if result.device_us is None else f'{result.device_us:.2f}'} cores={result.core_count}",
            flush=True,
        )

    assert pass_count > 0, "No MLP layout sweep candidates compiled and ran"
