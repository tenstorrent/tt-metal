# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Vision QKV matmul config sweep (``12288 x 1536 x 4608``, BF16 x BFP8 -> BFP8, LoFi).

Sweep families:

  qkv_2d_dram_*    — production 2D mcast (DRAM operands + DRAM output)
  qkv_oom_*        — L1 activation/output stress (weights stay DRAM); ``status=OOM``
  qkv_shard_*      — BLOCK-sharded activations + ``fuse_batch=True`` 2D mcast PC (TTNN requirement)
  qkv_mcast1d_*    — 1D mcast sub-grids (comparison / exploration)

Result statuses (all recorded in the TSV/CSV; pytest uses ``skip`` so the sweep keeps going):

  **PASS** — matmul ran, PCC met target, timing recorded.

  **SKIP** — config invalid for this shape/device, or non-OOM runtime error, or PCC miss.
    We skip (not ``xfail``/fail) so one pytest invocation finishes every param and emits
    one combined table. Common reasons:

    - ``config_build``: grid does not divide M/K/N for BLOCK shard, ``in0_block_w`` does not
      divide K/core_x tiles, vision 2D PC unavailable on harvested grid, etc.
    - ``tensor_alloc``: host/device could not place tensors (non-OOM layout/dtype error).
    - ``runtime``: TT_FATAL / unsupported layout combo (e.g. sharded in0 + wrong factory path).
    - ``pcc``: numerics below ``PCC_TARGET`` for that program config.

  **OOM** — allocator or kernel L1 budget exceeded (``Out of Memory``, CB clash).
    Often expected for ``qkv_oom_*`` (L1 activation/output stress). Production QKV keeps
    weights in DRAM; see ``dots_ocr_vision.py`` (~1432) and ``docs/dots_ocr_l1_math.md``.

One pytest run collects **all** families into a **single** TSV/CSV at the end.

Env:
  MATMUL_VERIFY_ITERS=1          — timed iterations (default 1)
  MATMUL_VERIFY_WARMUP=0         — warmup matmuls (default 0)
  MATMUL_VERIFY_TABLE_DIR=...    — output dir (default generated/matmul_verify)
  MATMUL_VERIFY_TABLE_BASENAME=  — optional filename stem

Run everything (2d + oom + shard + mcast1d) in one shot::

    export MATMUL_VERIFY_TABLE_BASENAME=vision_qkv_full_sweep
    pytest models/experimental/tt_symbiote/tests/test_attn_o_proj_matmul_configs.py -s -v

Subset via pytest ``-k`` (optional)::

    pytest ... -k 'qkv_2d_dram'      # production 2D only
    pytest ... -k 'qkv_oom'          # OOM probes only
    pytest ... -k 'qkv_shard'        # sharding only
    pytest ... -k 'qkv_mcast1d'      # 1D mcast only
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32
PEAK_LOFI_TFLOPS = 130.0
QKV_WEIGHT_BYTES = (1536 * 4608) // 2
DRAM_BW_GB_S = 300.0
MAX_MCAST_IN0_BLOCK_W = 8
MAX_L1_WEIGHT_BYTES = 256 * 1024

NUM_WARMUP = int(os.environ.get("MATMUL_VERIFY_WARMUP", "0"))
NUM_ITERS = int(os.environ.get("MATMUL_VERIFY_ITERS", "1"))
PCC_TARGET = 0.985


def _is_oom_message(msg: str) -> bool:
    lower = msg.lower()
    return any(
        token in lower
        for token in (
            "out of memory",
            "oom",
            "can't fit",
            "cannot fit",
            "not enough space",
            "no available",
            "exceeds",
            "l1_bank",
            "allocator",
        )
    )


_VERIFY_ROWS: list["VerifyRow"] = []
_SKIP_LOG: list[str] = []


@dataclass(frozen=True)
class MatmulShape:
    name: str
    m: int
    k: int
    n: int
    in0_dtype: object
    in1_dtype: object
    out_dtype: object
    in0_kind: str
    in1_kind: str
    out_kind: str

    @property
    def m_tiles(self) -> int:
        return self.m // TILE

    @property
    def k_tiles(self) -> int:
        return self.k // TILE

    @property
    def n_tiles(self) -> int:
        return self.n // TILE

    @property
    def shape_str(self) -> str:
        return f"{self.m} x {self.k} x {self.n}"

    @property
    def dtype_str(self) -> str:
        return f"{self.in0_kind} / {self.in1_kind} -> {self.out_kind}"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.k * self.n


QKV = MatmulShape(
    name="vision_qkv",
    m=12288,
    k=1536,
    n=4608,
    in0_dtype=ttnn.bfloat16,
    in1_dtype=ttnn.bfloat8_b,
    out_dtype=ttnn.bfloat8_b,
    in0_kind="BF16",
    in1_kind="BFP8",
    out_kind="BFP8",
)


VERIFY_COLUMNS: tuple[str, ...] = (
    "status",
    "test_file",
    "op",
    "config",
    "shape",
    "dtypes",
    "fidelity",
    "program_config",
    "in0_mem",
    "in1_mem",
    "out_mem",
    "dram_banks",
    "num_cores",
    "core_grid",
    "in0_block_w",
    "per_core_M",
    "per_core_N",
    "out_subblock_w",
    "num_iters",
    "avg_us",
    "tflops",
    "pct_peak_flops",
    "pct_dram_roofline",
    "pct_vs_best",
    "pcc",
    "notes",
)


@dataclass
class VerifyRow:
    status: str
    test_file: str
    op: str
    config: str
    shape: str
    dtypes: str
    fidelity: str
    program_config: str
    in0_mem: str
    in1_mem: str
    out_mem: str
    dram_banks: int
    num_cores: int
    core_grid: str
    in0_block_w: str
    per_core_m: str
    per_core_n: str
    out_subblock_w: str
    num_iters: int
    avg_us: str
    tflops: str
    pct_peak_flops: str
    pct_dram_roofline: str
    pct_vs_best: str
    pcc: str
    notes: str

    def as_dict(self) -> dict[str, str | int]:
        """Export dict keyed by VERIFY_COLUMNS (spreadsheet column names)."""
        return {
            "status": self.status,
            "test_file": self.test_file,
            "op": self.op,
            "config": self.config,
            "shape": self.shape,
            "dtypes": self.dtypes,
            "fidelity": self.fidelity,
            "program_config": self.program_config,
            "in0_mem": self.in0_mem,
            "in1_mem": self.in1_mem,
            "out_mem": self.out_mem,
            "dram_banks": self.dram_banks,
            "num_cores": self.num_cores,
            "core_grid": self.core_grid,
            "in0_block_w": self.in0_block_w,
            "per_core_M": self.per_core_m,
            "per_core_N": self.per_core_n,
            "out_subblock_w": self.out_subblock_w,
            "num_iters": self.num_iters,
            "avg_us": self.avg_us,
            "tflops": self.tflops,
            "pct_peak_flops": self.pct_peak_flops,
            "pct_dram_roofline": self.pct_dram_roofline,
            "pct_vs_best": self.pct_vs_best,
            "pcc": self.pcc,
            "notes": self.notes,
        }


def _compute_kernel():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _largest_divisor_le(value: int, limit: int) -> int:
    for d in range(min(value, limit), 0, -1):
        if value % d == 0:
            return d
    return 1


def _largest_divisor_of(per_core_n: int, limit: int) -> int:
    return _largest_divisor_le(per_core_n, limit)


def _grid_xy(grid) -> tuple[int, int]:
    """(x, y) from ttnn.CoreCoord or a (x, y) tuple passed at PC construction."""
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return int(grid.x), int(grid.y)
    gx, gy = grid
    return int(gx), int(gy)


def _iter_core_grids(
    shape: MatmulShape,
    device_grid_x: int,
    device_grid_y: int,
    *,
    min_cores: int = 4,
    max_cores: int = 64,
) -> list[tuple[int, int, int]]:
    """Return [(grid_x, grid_y, num_cores), ...] where N_tiles and K_tiles divide num_cores."""
    grids: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    n_tiles = shape.n_tiles
    k_tiles = shape.k_tiles

    for num_cores in range(min_cores, min(max_cores, device_grid_x * device_grid_y) + 1):
        if n_tiles % num_cores != 0 or k_tiles % num_cores != 0:
            continue
        for grid_x in range(1, device_grid_x + 1):
            if num_cores % grid_x != 0:
                continue
            grid_y = num_cores // grid_x
            if grid_y < 1 or grid_y > device_grid_y:
                continue
            key = (grid_x, grid_y)
            if key in seen:
                continue
            seen.add(key)
            grids.append((grid_x, grid_y, num_cores))

    # Prefer wider grids first (8x* before 4x*), then fewer cores last for readability.
    grids.sort(key=lambda t: (-t[0], t[2]))
    return grids


@dataclass(frozen=True)
class CfgResult:
    in0_mem: ttnn.MemoryConfig
    in1_mem: ttnn.MemoryConfig
    out_mem: ttnn.MemoryConfig
    prog: object
    meta: dict


def _fail_row(
    config_name: str,
    shape: MatmulShape,
    reason: str,
    *,
    status: str = "SKIP",
    meta: dict | None = None,
) -> None:
    """Append a non-PASS row to the verification table, then ``pytest.skip``.

    Pytest marks the param as skipped in the junit log, but we still collect the row
    here so the end-of-module TSV/CSV is a complete sweep matrix (not only successes).
    """
    m = meta or {}
    _SKIP_LOG.append(f"{config_name}: [{status}] {reason}")
    _VERIFY_ROWS.append(
        VerifyRow(
            status=status,
            test_file="test_attn_o_proj_matmul_configs.py",
            op=shape.name,
            config=config_name,
            shape=shape.shape_str,
            dtypes=shape.dtype_str,
            fidelity="LoFi",
            program_config=str(m.get("program_config", "-")),
            in0_mem=m.get("in0_mem", "-"),
            in1_mem=m.get("in1_mem", "-"),
            out_mem=m.get("out_mem", "-"),
            dram_banks=m.get("dram_banks", 12),
            num_cores=m.get("num_cores", 0),
            core_grid=m.get("core_grid", "-"),
            in0_block_w=str(m.get("in0_block_w", "-")),
            per_core_m=str(m.get("per_core_m", "-")),
            per_core_n=str(m.get("per_core_n", "-")),
            out_subblock_w=str(m.get("out_subblock_w", "-")),
            num_iters=NUM_ITERS,
            avg_us="-",
            tflops="-",
            pct_peak_flops="-",
            pct_dram_roofline="-",
            pct_vs_best="-",
            pcc="-",
            notes=reason[:500],
        )
    )
    pytest.skip(reason)


def _skip_row(
    config_name: str,
    shape: MatmulShape,
    reason: str,
    meta: dict | None = None,
) -> None:
    """Invalid config, alloc error, runtime fatal, or PCC miss — not an OOM budget issue."""
    _fail_row(config_name, shape, reason, status="SKIP", meta=meta)


def _oom_row(
    config_name: str,
    shape: MatmulShape,
    reason: str,
    meta: dict | None = None,
) -> None:
    """L1/allocator OOM — separate status so the spreadsheet can filter expected failures."""
    m = meta or {}
    if m.get("expected_oom") or config_name.startswith("qkv_oom_"):
        reason = f"expected L1 stress: {reason}"
    _fail_row(config_name, shape, reason, status="OOM", meta=meta)


def _validate_dram_sharded(shape: MatmulShape, grid_x: int, grid_y: int) -> str | None:
    """``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` is decode-only (M=1 tile)."""
    num_cores = grid_x * grid_y
    if shape.m_tiles != 1:
        # Vision QKV M=12288 → 384 tiles; this PC is not in the QKV sweep but kept for reference.
        return f"DRAM-sharded PC requires M=1 tile (got M_tiles={shape.m_tiles})"
    if shape.n_tiles % num_cores != 0:
        return f"N_tiles={shape.n_tiles} not divisible by {num_cores} cores"
    if shape.k_tiles % num_cores != 0:
        return f"K_tiles={shape.k_tiles} not divisible by {num_cores} cores"
    if shape.k % num_cores != 0:
        return f"K={shape.k} not divisible by {num_cores} cores (width shard)"
    return None


def _validate_mcast1d(shape: MatmulShape, grid_x: int, grid_y: int, in0_block_w: int) -> str | None:
    """1D mcast needs even N/K split across cores and a conservative L1 weight block estimate."""
    num_cores = grid_x * grid_y
    if shape.n_tiles % num_cores != 0:
        return f"N_tiles={shape.n_tiles} not divisible by {num_cores} cores"
    per_core_n = shape.n_tiles // num_cores
    if shape.k_tiles % in0_block_w != 0:
        return f"K_tiles={shape.k_tiles} not divisible by in0_block_w={in0_block_w}"
    # 1D mcast streams weight tiles into L1; skip before launch if block would exceed ~256 KiB/core.
    est_l1 = in0_block_w * per_core_n * 1024
    if est_l1 > MAX_L1_WEIGHT_BYTES:
        return f"est in1 L1 {est_l1 // 1024} KB > {MAX_L1_WEIGHT_BYTES // 1024} KB cap"
    return None


def _cfg_dram_sharded(
    device, shape: MatmulShape, grid_x: int, grid_y: int, in0_block_w: int | None = None
) -> CfgResult:
    err = _validate_dram_sharded(shape, grid_x, grid_y)
    if err:
        raise ValueError(err)

    dram_grid = device.dram_grid_size()
    num_banks = int(dram_grid.x) * int(dram_grid.y)
    num_cores = grid_x * grid_y

    n_padded = math.ceil(shape.n / (TILE * num_banks)) * TILE * num_banks
    per_core_n = shape.n_tiles // num_cores
    k_tiles_per_core = shape.k_tiles // num_cores
    if in0_block_w is None:
        in0_block_w = _largest_divisor_le(k_tiles_per_core, MAX_MCAST_IN0_BLOCK_W)
    if k_tiles_per_core % in0_block_w != 0:
        raise ValueError(f"in0_block_w={in0_block_w} does not divide K/core tiles={k_tiles_per_core}")

    compute_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, shape.m, shape.k),
        core_grid=compute_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(dram_grid.x) - 1, int(dram_grid.y) - 1),
            )
        }
    )
    in1_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(in1_grid, [shape.k, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem = ttnn.create_sharded_memory_config(
        (1, 1, shape.m, shape.n),
        core_grid=compute_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=shape.m_tiles,
        per_core_N=per_core_n,
        fused_activation=None,
    )
    return CfgResult(
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog=prog,
        meta={
            "program_config": type(prog).__name__,
            "in0_mem": "L1 width_sharded",
            "in1_mem": "DRAM width_sharded",
            "out_mem": "L1 width_sharded",
            "dram_banks": num_banks,
            "num_cores": num_cores,
            "core_grid": f"{grid_x}x{grid_y}",
            "in0_block_w": in0_block_w,
            "per_core_m": shape.m_tiles,
            "per_core_n": per_core_n,
            "out_subblock_w": "-",
        },
    )


def _iter_block_shard_grids(
    shape: MatmulShape,
    device_grid_x: int,
    device_grid_y: int,
    *,
    require_n_divisible: bool = False,
) -> list[tuple[int, int, int]]:
    """Grids where BLOCK shard geometry divides M/K (and optionally N) across (grid_x, grid_y)."""
    grids: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for grid_x in range(1, device_grid_x + 1):
        for grid_y in range(1, device_grid_y + 1):
            num_cores = grid_x * grid_y
            if num_cores < 4 or num_cores > 64:
                continue
            if shape.m % grid_y != 0 or shape.k % grid_x != 0:
                continue
            if require_n_divisible and shape.n % grid_x != 0:
                continue
            key = (grid_x, grid_y)
            if key in seen:
                continue
            seen.add(key)
            grids.append((grid_x, grid_y, num_cores))
    grids.sort(key=lambda t: (-t[0], t[2]))
    return grids


def _block_shard_ibw_candidates(shape: MatmulShape, grid_x: int) -> list[int]:
    """``in0_block_w`` must divide K-tiles assigned to each column of the grid."""
    k_tiles_per_core_x = shape.k_tiles // grid_x
    return [ibw for ibw in (8, 6, 4, 3, 2, 1) if k_tiles_per_core_x % ibw == 0]


def _best_subblock(block_h: int, block_w: int, limit: int = 8) -> tuple[int, int]:
    for h in range(min(block_h, limit), 0, -1):
        if block_h % h != 0:
            continue
        for w in range(min(block_w, limit // h), 0, -1):
            if block_w % w != 0:
                continue
            return h, w
    return 1, 1


def _block_shard_mem_config(
    grid_x: int,
    grid_y: int,
    tensor_shape: tuple[int, ...],
) -> ttnn.MemoryConfig:
    """BLOCK sharding for 2D mcast matmul (not WIDTH — WIDTH breaks on [1,1,M,K] activations)."""
    return ttnn.create_sharded_memory_config(
        tensor_shape,
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _dram_width_sharded_weight_mem(device, shape: MatmulShape) -> ttnn.MemoryConfig:
    dram_grid = device.dram_grid_size()
    num_banks = int(dram_grid.x) * int(dram_grid.y)
    n_padded = math.ceil(shape.n / (TILE * num_banks)) * TILE * num_banks
    in1_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(dram_grid.x) - 1, int(dram_grid.y) - 1),
            )
        }
    )
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(in1_grid, [shape.k, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _cfg_qkv_block_sharded(
    device,
    shape: MatmulShape,
    grid_x: int,
    grid_y: int,
    ibw: int,
    *,
    block_out: bool = False,
    in1_dram_width: bool = False,
) -> CfgResult:
    """BLOCK-sharded QKV matmul aligned to grid (see ``tests/ttnn/.../test_matmul.py`` 2D sharded).

    TTNN rules for ``MatmulMultiCoreReuseMultiCastProgramConfig`` + sharded in0:
      - ``fuse_batch=True`` (batch dims are ``[1,1]``).
      - in0 must be BLOCK_SHARDED or HEIGHT_SHARDED (not WIDTH — WIDTH expects shard
        height == full M and produced ``1536 vs 12288`` errors on QKV activations).
      - if out is sharded, in0/out must share layout + buffer type.
    """
    if shape.m % grid_y != 0 or shape.k % grid_x != 0:
        raise ValueError(f"M={shape.m} K={shape.k} not divisible by grid {grid_x}x{grid_y}")
    if block_out and shape.n % grid_x != 0:
        raise ValueError(f"N={shape.n} not divisible by grid_x={grid_x} for block-sharded output")

    k_tiles_per_core_x = shape.k_tiles // grid_x
    if k_tiles_per_core_x % ibw != 0:
        raise ValueError(f"in0_block_w={ibw} does not divide K/core_x tiles={k_tiles_per_core_x} (grid_x={grid_x})")

    per_core_m = (shape.m // grid_y) // TILE
    per_core_n = (shape.n // grid_x) // TILE
    out_block_h = per_core_m
    out_block_w = per_core_n
    out_subblock_h, out_subblock_w = _best_subblock(out_block_h, out_block_w)

    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=ibw,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )

    in0_shape = (1, 1, shape.m, shape.k)
    out_shape = (1, 1, shape.m, shape.n)
    in0_mem = _block_shard_mem_config(grid_x, grid_y, in0_shape)
    in0_label = f"L1 block_sharded {grid_x}x{grid_y}"

    if in1_dram_width:
        in1_mem = _dram_width_sharded_weight_mem(device, shape)
        in1_label = "DRAM width_sharded (12 banks)"
    else:
        in1_mem = ttnn.DRAM_MEMORY_CONFIG
        in1_label = "DRAM interleaved"

    if block_out:
        out_mem = _block_shard_mem_config(grid_x, grid_y, out_shape)
        out_label = f"L1 block_sharded {grid_x}x{grid_y}"
    else:
        out_mem = ttnn.DRAM_MEMORY_CONFIG
        out_label = "DRAM interleaved"

    return CfgResult(
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog=pc,
        meta={
            "program_config": type(pc).__name__,
            "in0_mem": in0_label,
            "in1_mem": in1_label,
            "out_mem": out_label,
            "dram_banks": 12,
            "num_cores": grid_x * grid_y,
            "core_grid": f"{grid_x}x{grid_y}",
            "in0_block_w": ibw,
            "per_core_m": per_core_m,
            "per_core_n": per_core_n,
            "out_subblock_w": str(out_subblock_w),
            "fuse_batch": True,
        },
    )


def _cfg_mcast1d(
    shape: MatmulShape,
    grid_x: int,
    grid_y: int,
    in0_block_w: int = 8,
    out_subblock_w: int | None = None,
) -> CfgResult:
    err = _validate_mcast1d(shape, grid_x, grid_y, in0_block_w)
    if err:
        raise ValueError(err)

    num_cores = grid_x * grid_y
    per_core_n = shape.n_tiles // num_cores
    if out_subblock_w is None:
        out_subblock_w = _largest_divisor_of(per_core_n, 8)
    if per_core_n % out_subblock_w != 0:
        raise ValueError(f"out_subblock_w={out_subblock_w} does not divide per_core_N={per_core_n}")

    prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    return CfgResult(
        in0_mem=ttnn.L1_MEMORY_CONFIG,
        in1_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        prog=prog,
        meta={
            "program_config": type(prog).__name__,
            "in0_mem": "L1 interleaved",
            "in1_mem": "DRAM interleaved",
            "out_mem": "L1 interleaved",
            "dram_banks": 12,
            "num_cores": num_cores,
            "core_grid": f"{grid_x}x{grid_y}",
            "in0_block_w": in0_block_w,
            "per_core_m": 1,
            "per_core_n": per_core_n,
            "out_subblock_w": str(out_subblock_w),
        },
    )


def _cfg_qkv_2d(
    device,
    shape: MatmulShape,
    ibw: int | None,
    *,
    in0_l1: bool = False,
    out_l1: bool = False,
) -> CfgResult:
    from models.experimental.tt_symbiote.modules.dots_ocr_vision import _vision_matmul_program_config

    # Weights (in1) always DRAM — matches production vision QKV and avoids multi-MB L1 alloc.
    in1_l1 = False

    pc = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)
    if pc is None:
        raise ValueError("no vision 2D program config on this device")
    if ibw is not None:
        if shape.k_tiles % ibw != 0:
            raise ValueError(f"K_tiles={shape.k_tiles} not divisible by in0_block_w={ibw}")
        pc.in0_block_w = ibw
    gx, gy = _grid_xy(pc.compute_with_storage_grid_size)

    def _mem_cfg(l1: bool) -> ttnn.MemoryConfig:
        return ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG

    def _mem_label(l1: bool) -> str:
        return "L1 interleaved" if l1 else "DRAM interleaved"

    return CfgResult(
        in0_mem=_mem_cfg(in0_l1),
        in1_mem=_mem_cfg(in1_l1),
        out_mem=_mem_cfg(out_l1),
        prog=pc,
        meta={
            "program_config": type(pc).__name__,
            "in0_mem": _mem_label(in0_l1),
            "in1_mem": _mem_label(in1_l1),
            "out_mem": _mem_label(out_l1),
            "dram_banks": 12,
            "num_cores": int(gx) * int(gy),
            "core_grid": f"{gx}x{gy}",
            "in0_block_w": pc.in0_block_w,
            "per_core_m": pc.per_core_M,
            "per_core_n": pc.per_core_N,
            "out_subblock_w": str(pc.out_subblock_w),
            "expected_oom": in0_l1 or out_l1,
        },
    )


def _build_qkv_configs(device_grid_x: int, device_grid_y: int) -> list[tuple[str, Callable]]:
    """All QKV sweep families in one list (2d + oom + shard + mcast1d)."""
    shape = QKV
    configs: list[tuple[str, Callable]] = []

    # --- 2D DRAM (production) ---
    for ibw in (8, 6, 4, 3, 2, 1):
        if shape.k_tiles % ibw != 0:
            continue
        configs.append((f"qkv_2d_dram_ibw{ibw}", lambda d, w=ibw: _cfg_qkv_2d(d, shape, w)))

    # --- OOM probes: deliberate L1 activation/output pressure; weights stay DRAM ---
    # Production path is DRAM interleaved (dots_ocr_vision QKV). These rows document
    # why L1 out / L1 activations fail the ~1.5 MiB/core budget at S=12288.
    for ibw in (8, 6, 4, 3, 2, 1):
        if shape.k_tiles % ibw != 0:
            continue
        configs.append((f"qkv_oom_l1out_ibw{ibw}", lambda d, w=ibw: _cfg_qkv_2d(d, shape, w, out_l1=True)))
    for ibw in (8, 6, 4):
        if shape.k_tiles % ibw != 0:
            continue
        configs.append((f"qkv_oom_l1_in0_ibw{ibw}", lambda d, w=ibw: _cfg_qkv_2d(d, shape, w, in0_l1=True)))
    configs.append(("qkv_oom_l1_in0_out_ibw8", lambda d: _cfg_qkv_2d(d, shape, 8, in0_l1=True, out_l1=True)))

    # --- BLOCK sharding + fuse_batch (TTNN 2D mcast rules; not WIDTH / not vision 8×8 PC) ---
    for grid_x, grid_y, _ in _iter_block_shard_grids(shape, device_grid_x, device_grid_y):
        gx, gy = grid_x, grid_y
        for ibw in _block_shard_ibw_candidates(shape, gx):
            configs.append(
                (
                    f"qkv_shard_block_in0_{gx}x{gy}_ibw{ibw}",
                    lambda d, x=gx, y=gy, w=ibw: _cfg_qkv_block_sharded(d, shape, x, y, w),
                )
            )
            configs.append(
                (
                    f"qkv_shard_dramwt_block_in0_{gx}x{gy}_ibw{ibw}",
                    lambda d, x=gx, y=gy, w=ibw: _cfg_qkv_block_sharded(d, shape, x, y, w, in1_dram_width=True),
                )
            )
    for grid_x, grid_y, _ in _iter_block_shard_grids(shape, device_grid_x, device_grid_y, require_n_divisible=True):
        gx, gy = grid_x, grid_y
        for ibw in _block_shard_ibw_candidates(shape, gx):
            configs.append(
                (
                    f"qkv_shard_block_in0_out_{gx}x{gy}_ibw{ibw}",
                    lambda d, x=gx, y=gy, w=ibw: _cfg_qkv_block_sharded(d, shape, x, y, w, block_out=True),
                )
            )

    # --- 1D mcast sub-grids ---
    for grid_x, grid_y, num_cores in _iter_core_grids(shape, device_grid_x, device_grid_y):
        name = f"qkv_mcast1d_{grid_x}x{grid_y}_pcn{shape.n_tiles // num_cores}"
        configs.append(
            (
                name,
                lambda d, gx=grid_x, gy=grid_y, s=shape: _cfg_mcast1d(s, gx, gy),
            )
        )

    return configs


def _roofline_us(shape: MatmulShape) -> float:
    return QKV_WEIGHT_BYTES / (DRAM_BW_GB_S * 1e3)


def _record_pass_row(
    config_name: str,
    shape: MatmulShape,
    cfg: CfgResult,
    avg_us: float,
    pcc_val: float,
) -> None:
    tflops = shape.flops / max(avg_us, 1e-6) / 1e6
    pct_peak = 100.0 * tflops / PEAK_LOFI_TFLOPS
    roof_us = _roofline_us(shape)
    pct_dram = 100.0 * roof_us / max(avg_us, 1e-6)
    _VERIFY_ROWS.append(
        VerifyRow(
            status="PASS",
            test_file="test_attn_o_proj_matmul_configs.py",
            op=shape.name,
            config=config_name,
            shape=shape.shape_str,
            dtypes=shape.dtype_str,
            fidelity="LoFi",
            program_config=str(cfg.meta.get("program_config", type(cfg.prog).__name__)),
            in0_mem=cfg.meta["in0_mem"],
            in1_mem=cfg.meta["in1_mem"],
            out_mem=cfg.meta["out_mem"],
            dram_banks=cfg.meta["dram_banks"],
            num_cores=cfg.meta["num_cores"],
            core_grid=cfg.meta["core_grid"],
            in0_block_w=str(cfg.meta["in0_block_w"]),
            per_core_m=str(cfg.meta["per_core_m"]),
            per_core_n=str(cfg.meta["per_core_n"]),
            out_subblock_w=str(cfg.meta["out_subblock_w"]),
            num_iters=NUM_ITERS,
            avg_us=f"{avg_us:.1f}",
            tflops=f"{tflops:.1f}",
            pct_peak_flops=f"{pct_peak:.1f}",
            pct_dram_roofline=f"{pct_dram:.1f}",
            pct_vs_best="0.0",
            pcc=f"{pcc_val:.4f}",
            notes="",
        )
    )


def _finalize_pct_vs_best(rows: list[VerifyRow]) -> list[VerifyRow]:
    pass_rows = [r for r in rows if r.status == "PASS" and r.avg_us != "-"]
    if not pass_rows:
        return rows
    best_us = min(float(r.avg_us) for r in pass_rows)
    out: list[VerifyRow] = []
    for r in rows:
        if r.status == "PASS" and r.avg_us != "-":
            pct_vs = 100.0 * (float(r.avg_us) / best_us - 1.0)
            out.append(replace(r, pct_vs_best=f"{pct_vs:.1f}"))
        else:
            out.append(r)
    return out


def _row_field_values(row: VerifyRow) -> list[str | int]:
    return [str(row.as_dict()[col]) for col in VERIFY_COLUMNS]


def _write_verify_table_files(rows: list[VerifyRow], shape: MatmulShape) -> tuple[Path, Path]:
    table_dir = Path(os.environ.get("MATMUL_VERIFY_TABLE_DIR", "generated/matmul_verify"))
    table_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_stem = f"{shape.name}_{shape.m}x{shape.k}x{shape.n}_{stamp}"
    stem = os.environ.get("MATMUL_VERIFY_TABLE_BASENAME", default_stem)
    tsv_path = table_dir / f"{stem}.tsv"
    csv_path = table_dir / f"{stem}.csv"

    header_line = "\t".join(VERIFY_COLUMNS)
    body_lines = ["\t".join(_row_field_values(r)) for r in rows]
    tsv_path.write_text(header_line + "\n" + "\n".join(body_lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(VERIFY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) for k, v in row.as_dict().items()})

    return tsv_path, csv_path


def _emit_verify_report(rows: list[VerifyRow], shape: MatmulShape) -> None:
    if not rows:
        return

    rows = _finalize_pct_vs_best(rows)
    tsv_path, csv_path = _write_verify_table_files(rows, shape)

    print("\n" + "=" * 120)
    print("MATMUL VERIFICATION TABLE (paste into spreadsheet)")
    print("\t".join(VERIFY_COLUMNS))
    for r in rows:
        print("\t".join(_row_field_values(r)))
    n_pass = sum(1 for r in rows if r.status == "PASS")
    n_oom = sum(1 for r in rows if r.status == "OOM")
    n_skip = sum(1 for r in rows if r.status == "SKIP")
    print(f"--- summary: {n_pass} passed, {n_oom} oom, {n_skip} skipped, {len(rows)} total ---")
    print(f"--- saved: {tsv_path} ---")
    print(f"--- saved: {csv_path} ---")
    if _SKIP_LOG:
        print("skip log (last 20):")
        for line in _SKIP_LOG[-20:]:
            print(f"  {line}")
    print("=" * 120 + "\n")


SHAPE = QKV
# All families in one parametrized sweep (Wormhole 8x8); use pytest -k to subset.
CONFIGS: list[tuple[str, Callable]] = _build_qkv_configs(8, 8)
CONFIG_IDS = [name for name, _ in CONFIGS]
print(f"[matmul_verify] collected {len(CONFIGS)} QKV configs (2d + oom + shard + mcast1d)")


def test_qkv_shard_config_preflight():
    """Host-only: BLOCK shard builders must not use invalid WIDTH layouts."""
    shape = QKV
    # 8×8: K/core_x = 6 tiles → ibw ∈ {6,3,2,1}, not 8
    cfg = _cfg_qkv_block_sharded(None, shape, 8, 8, 6)
    assert cfg.meta["fuse_batch"] is True
    assert "block_sharded" in cfg.meta["in0_mem"]
    assert cfg.prog.fuse_batch is True
    with pytest.raises(ValueError, match="in0_block_w=8"):
        _cfg_qkv_block_sharded(None, shape, 8, 8, 8)
    cfg_out = _cfg_qkv_block_sharded(None, shape, 8, 8, 6, block_out=True)
    assert "block_sharded" in cfg_out.meta["out_mem"]
    # 7×8 grid: M not divisible by 8
    with pytest.raises(ValueError, match="not divisible"):
        _cfg_qkv_block_sharded(None, shape, 7, 8, 1)


@pytest.fixture(scope="module", autouse=True)
def _print_verify_table_at_end():
    yield
    _emit_verify_report(_VERIFY_ROWS, SHAPE)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_attn_o_proj_matmul_configs(device, config_name: str, cfg_builder: Callable):
    """One row per program-config variant; failures become SKIP/OOM rows, not hard fails."""
    shape = SHAPE
    torch.manual_seed(0)

    torch_a = torch.randn((1, 1, shape.m, shape.k), dtype=torch.bfloat16) * 0.1
    torch_b = torch.randn((shape.k, shape.n), dtype=torch.bfloat16) * 0.1
    torch_ref = torch.matmul(torch_a, torch_b.unsqueeze(0).unsqueeze(0))

    # --- config_build SKIP: pre-flight validation (grid divisibility, PC build, etc.) ---
    try:
        cfg = cfg_builder(device)
    except (ValueError, RuntimeError, TypeError) as exc:
        _skip_row(config_name, shape, f"config_build: {exc}")

    compute_cfg = _compute_kernel()

    # --- tensor_alloc SKIP/OOM: cannot place activations/weights on device ---
    # in1 (weights) is always DRAM; OOM here is usually huge L1 activation buffers.
    try:
        input_a = ttnn.from_torch(
            torch_a,
            dtype=shape.in0_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=cfg.in0_mem,
            device=device,
        )
        input_b = ttnn.from_torch(
            torch_b,
            dtype=shape.in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=cfg.in1_mem,
            device=device,
        )
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if _is_oom_message(msg):
            _oom_row(config_name, shape, f"tensor_alloc: {msg[:200]}", cfg.meta)
        _skip_row(config_name, shape, f"tensor_alloc: {msg[:200]}", cfg.meta)

    def _run():
        return ttnn.matmul(
            input_a,
            input_b,
            program_config=cfg.prog,
            memory_config=cfg.out_mem,
            dtype=shape.out_dtype,
            compute_kernel_config=compute_cfg,
        )

    last_out = None
    pcc_passed = False
    pcc_val = 0.0
    avg_us = 0.0
    # --- runtime OOM/SKIP: kernel L1 CB budget or TT_FATAL (sharded path, bad grid, etc.) ---
    try:
        for _ in range(NUM_WARMUP):
            out = _run()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        start = time.perf_counter()
        for _ in range(NUM_ITERS):
            if last_out is not None:
                ttnn.deallocate(last_out)
            last_out = _run()
        ttnn.synchronize_device(device)
        avg_us = (time.perf_counter() - start) * 1e6 / max(NUM_ITERS, 1)

        result = ttnn.to_torch(last_out)
        pcc_passed, pcc_val = assert_with_pcc(torch_ref, result, pcc=PCC_TARGET)
        pcc_val = float(pcc_val)
    except (ValueError, RuntimeError) as exc:
        msg = str(exc)
        if _is_oom_message(msg):
            _oom_row(config_name, shape, f"runtime_oom: {msg[:200]}", cfg.meta)
        _skip_row(config_name, shape, f"runtime: {msg[:200]}", cfg.meta)
    finally:
        if last_out is not None:
            ttnn.deallocate(last_out)
        ttnn.deallocate(input_a)
        ttnn.deallocate(input_b)

    # --- pcc SKIP: ran but LoFi matmul config does not match torch reference tightly enough ---
    if not pcc_passed:
        _skip_row(config_name, shape, f"pcc {pcc_val:.4f} < {PCC_TARGET}", cfg.meta)

    _record_pass_row(config_name, shape, cfg, avg_us, pcc_val)
    print(
        f"\n  PASS {config_name}: {avg_us:.1f} us | grid={cfg.meta['core_grid']} | "
        f"pcn={cfg.meta['per_core_n']} ibw={cfg.meta['in0_block_w']} | pcc={pcc_val:.4f}"
    )
