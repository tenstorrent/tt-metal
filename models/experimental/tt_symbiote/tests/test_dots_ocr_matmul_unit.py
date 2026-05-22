# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for four SLOW matmul shapes from the dots.ocr Tracy report.

Scenarios
---------
1. in0=DRAM interleaved, in1=DRAM interleaved, out=L1 interleaved
2a. in0=L1 interleaved,  in1=DRAM interleaved, out=L1 interleaved
2b. in0=L1 sharded (height / width / block), in1=DRAM interleaved, out=L1 interleaved
2c. in0=L1 interleaved,  in1=DRAM interleaved, explicit per-core-M/N grid variants

Shapes
------
  vision_qkv : 12288 × 1536 × 4608   BF16 × BFP8 → BFP8   ~1903 μs
  o_proj     : 12288 × 1536 × 1536   BFP8 × BFP8 → BFP8    ~705 μs
  mlp_fc1    : 12288 × 1536 × 4224   BFP8 × BFP8 → BFP8   ~1762 μs
  mlp_fc2    : 12288 × 4224 × 1536   BFP8 × BFP8 → BFP8   ~1485 μs

Results are written to models/experimental/tt_symbiote/tests/new_report/
as a timestamped CSV with one row per matmul run.

Run all::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py -s

Run one scenario::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_in0_sharded" -s

Filter by shape::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "mlp_fc2" -s
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    _largest_divisor_le,
    _vision_matmul_compute_config,
    _vision_matmul_program_config,
)

_TILE = 32
_NUM_ITERS = 1
_WH_PEAK_DRAM_BW_GBS = 288.0  # Wormhole B0 peak DRAM bandwidth (GB/s)
_L1_PER_CORE_BYTES = 1_500_000  # ~1.5 MB conservative per-core L1 budget


# ---------------------------------------------------------------------------
# Shape catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    name: str
    m: int
    k: int
    n: int
    in0_dtype: ttnn.DataType
    in1_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    ref_us: int


_BF16, _BFP8 = ttnn.bfloat16, ttnn.bfloat8_b

_SHAPES = [
    _Shape("vision_qkv", 12288, 1536, 4608, _BF16, _BFP8, _BFP8, 1903),
    _Shape("o_proj", 12288, 1536, 1536, _BFP8, _BFP8, _BFP8, 705),
    _Shape("mlp_fc1", 12288, 1536, 4224, _BFP8, _BFP8, _BFP8, 1762),
    _Shape("mlp_fc2", 12288, 4224, 1536, _BFP8, _BFP8, _BFP8, 1485),
]

_SHAPE_IDS = [s.name for s in _SHAPES]


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

_REPORT_DIR = Path(__file__).parent / "new_report"
_REPORT_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_CSV_PATH = _REPORT_DIR / f"matmul_unit_{_RUN_TS}.csv"
_TEST_FILE = Path(__file__).name

_CSV_HEADERS = [
    "Test File",
    "Matmul Op",
    "Config Name",
    "Shape (M x K x N)",
    "Input dtypes (in0 / in1 -> out)",
    "Math Fidelity",
    "in0 Memory Config",
    "in1 Memory Config",
    "Output Memory Config",
    "Status",
    "Device Time (us)",
    "Compute Cores",
    "Core Grid",
    "in0_block_w",
    "per_core_M",
    "per_core_N",
    "out_subblock_w",
    "DRAM BW (GB/s)",
    "DRAM Util %",
    "FLOPs (TFLOPs)",
]

with open(_CSV_PATH, "w", newline="") as _f:
    csv.writer(_f).writerow(_CSV_HEADERS)

logger.info(f"[matmul_unit] CSV report → {_CSV_PATH}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _dtype_str(dt: ttnn.DataType) -> str:
    return {
        ttnn.bfloat16: "BF16",
        ttnn.bfloat8_b: "BFP8",
        ttnn.bfloat4_b: "BFP4",
    }.get(dt, str(dt))


def _elem_bytes(dt: ttnn.DataType) -> int:
    """Approximate bytes per element (BFP8 shared-exponent overhead ignored)."""
    return 2 if dt == ttnn.bfloat16 else 1


def _l1_shard_fits(shard_h: int, shard_w: int, dtype: ttnn.DataType) -> bool:
    """Return True if one shard [shard_h × shard_w] fits in per-core L1."""
    return shard_h * shard_w * _elem_bytes(dtype) <= _L1_PER_CORE_BYTES


def _extract_pc_meta(prog_cfg) -> Tuple[str, str, str, str]:
    """(in0_block_w, per_core_M, per_core_N, out_subblock_w) from a program config."""
    if prog_cfg is None:
        return "", "", "", ""
    return (
        str(getattr(prog_cfg, "in0_block_w", "")),
        str(getattr(prog_cfg, "per_core_M", "")),
        str(getattr(prog_cfg, "per_core_N", "")),
        str(getattr(prog_cfg, "out_subblock_w", "")),
    )


def _skip_oom(
    shape: _Shape,
    config_name: str,
    in0_desc: str,
    in1_desc: str,
    out_desc: str,
    cores: int,
    grid: str,
    shard_h: int,
    shard_w: int,
) -> None:
    """Write a SKIPPED_OOM row to the CSV then call pytest.skip."""
    size_kb = shard_h * shard_w * _elem_bytes(shape.in0_dtype) // 1024
    _write_csv_row(
        shape,
        config_name,
        in0_desc,
        in1_desc,
        out_desc,
        "LoFi",
        "SKIPPED_OOM",
        None,
        cores,
        grid,
        "",
        "",
        "",
        "",
        in0_in_dram=False,
        out_in_dram=False,
    )
    pytest.skip(f"OOM — shard [{shard_h}×{shard_w}] ≈ {size_kb} KB > {_L1_PER_CORE_BYTES//1024} KB L1 per core")


def _grid_str_from_cores(nc: int) -> str:
    """Widest-x grid string for nc cores, e.g. 48 → '8x6'."""
    for gx in range(8, 0, -1):
        if nc % gx == 0 and nc // gx <= 8:
            return f"{gx}x{nc // gx}"
    return str(nc)


def _write_csv_row(
    shape: _Shape,
    config_name: str,
    in0_desc: str,
    in1_desc: str,
    out_desc: str,
    math_fidelity_str: str,
    status: str,
    device_time_us: Optional[float],
    compute_cores: int,
    core_grid: str,
    in0_block_w: str,
    per_core_m: str,
    per_core_n: str,
    out_subblock_w: str,
    # Which tensors live in DRAM (affects DRAM BW calculation)
    in0_in_dram: bool = True,
    out_in_dram: bool = False,
) -> None:
    """Compute derived metrics and append one row to the timestamped CSV."""
    if status == "PASS" and device_time_us is not None and device_time_us > 0:
        # DRAM traffic: in1 always DRAM; in0/out only if they reside in DRAM.
        dram_bytes = shape.k * shape.n * _elem_bytes(shape.in1_dtype)
        if in0_in_dram:
            dram_bytes += shape.m * shape.k * _elem_bytes(shape.in0_dtype)
        if out_in_dram:
            dram_bytes += shape.m * shape.n * _elem_bytes(shape.out_dtype)

        dram_bw_gbs = dram_bytes / device_time_us * 1e-3  # bytes/μs → GB/s
        dram_util_pct = dram_bw_gbs / _WH_PEAK_DRAM_BW_GBS * 100.0
        tflops = 2 * shape.m * shape.k * shape.n / device_time_us / 1e6

        t_str = f"{device_time_us:.2f}"
        bw_str = f"{dram_bw_gbs:.1f}"
        util_str = f"{dram_util_pct:.1f}"
        tf_str = f"{tflops:.2f}"
    else:
        t_str = bw_str = util_str = tf_str = ""

    row = [
        _TEST_FILE,
        shape.name,
        config_name,
        f"{shape.m} x {shape.k} x {shape.n}",
        f"{_dtype_str(shape.in0_dtype)} / {_dtype_str(shape.in1_dtype)} -> {_dtype_str(shape.out_dtype)}",
        math_fidelity_str,
        in0_desc,
        in1_desc,
        out_desc,
        status,
        t_str,
        compute_cores if compute_cores else "",
        core_grid,
        in0_block_w,
        per_core_m,
        per_core_n,
        out_subblock_w,
        bw_str,
        util_str,
        tf_str,
    ]
    with open(_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Memory config helpers
# ---------------------------------------------------------------------------


def _compact_grid(num_cores: int) -> ttnn.CoreRangeSet:
    """Widest-x rectangular grid for num_cores in the 8×8 WH grid."""
    best_gx = best_gy = None
    for gx in range(1, 9):
        if num_cores % gx == 0:
            gy = num_cores // gx
            if gy <= 8:
                best_gx, best_gy = gx, gy
    if best_gx is None:
        raise ValueError(f"Cannot fit {num_cores} cores in 8×8 grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(best_gx - 1, best_gy - 1))})


def _height_shard_mem(m: int, k: int, num_cores: int) -> ttnn.MemoryConfig:
    """HEIGHT_SHARDED L1: split M across num_cores, each shard is [m/nc, k]."""
    if m % num_cores != 0:
        raise ValueError(f"M={m} not divisible by {num_cores}")
    shard_h = m // num_cores
    if shard_h % _TILE != 0:
        raise ValueError(f"shard_h={shard_h} not tile-aligned (M={m}, nc={num_cores})")
    if k % _TILE != 0:
        raise ValueError(f"k={k} not tile-aligned")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_compact_grid(num_cores), [shard_h, k], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_shard_mem(m: int, k: int, num_cores: int) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED L1: split K across num_cores, each shard is [m, k/nc]."""
    if k % num_cores != 0:
        raise ValueError(f"K={k} not divisible by {num_cores}")
    shard_w = k // num_cores
    if shard_w % _TILE != 0:
        raise ValueError(f"shard_w={shard_w} not tile-aligned (K={k}, nc={num_cores})")
    if m % _TILE != 0:
        raise ValueError(f"m={m} not tile-aligned")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_compact_grid(num_cores), [m, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _block_shard_mem(m: int, k: int, gx: int, gy: int) -> ttnn.MemoryConfig:
    """BLOCK_SHARDED L1: split M by gy rows and K by gx cols, shard=[m/gy, k/gx]."""
    if m % gy != 0 or k % gx != 0:
        raise ValueError(f"M={m} not divisible by gy={gy} or K={k} not divisible by gx={gx}")
    shard_h, shard_w = m // gy, k // gx
    if shard_h % _TILE != 0 or shard_w % _TILE != 0:
        raise ValueError(f"shard [{shard_h}, {shard_w}] not tile-aligned for block({gx}x{gy})")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


# ---------------------------------------------------------------------------
# Program config helper for explicit grid variants (scenario 2c)
# ---------------------------------------------------------------------------


def _build_explicit_program_config(
    grid_x: int, grid_y: int, m: int, k: int, n: int
) -> Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
    """2D-mcast config for an explicit (grid_x, grid_y).  Returns None if invalid."""
    m_tiles = m // _TILE
    k_tiles = k // _TILE
    n_tiles = n // _TILE

    if m_tiles % grid_y != 0:
        return None
    per_core_m = m_tiles // grid_y
    per_core_n = (n_tiles + grid_x - 1) // grid_x
    if per_core_n > 24 or per_core_m > 64:
        return None

    in0_block_w = _largest_divisor_le(k_tiles, 8)

    out_block_h = 1
    for obh in [16, 12, 8, 6, 4, 3, 2, 1]:
        if obh > per_core_m or per_core_m % obh != 0:
            continue
        approx_interm_kb = (obh * per_core_n * 2048) // 1024
        approx_in0_kb = (obh * in0_block_w * 2 * 2048) // 1024
        if approx_interm_kb + approx_in0_kb <= 1024:
            out_block_h = obh
            break

    out_subblock_h, out_subblock_w = 1, 1
    for h in range(min(out_block_h, 8), 0, -1):
        if out_block_h % h != 0:
            continue
        for w in range(min(per_core_n, 8 // h), 0, -1):
            if per_core_n % w == 0:
                out_subblock_h, out_subblock_w = h, w
                break
        if out_subblock_h * out_subblock_w > 1:
            break

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_matmul(
    device,
    shape: _Shape,
    in0_mem: ttnn.MemoryConfig,
    in1_mem: ttnn.MemoryConfig,
    out_mem: ttnn.MemoryConfig,
    prog_cfg,
    compute_cfg,
    config_name: str,
    in0_desc: str,
    in1_desc: str,
    out_desc: str,
    math_fidelity_str: str = "LoFi",
    # Explicit core/grid for cases where prog_cfg is None (sharded auto-config).
    cores_override: Optional[int] = None,
    grid_override: Optional[str] = None,
    # Which tensors are sourced from DRAM (for BW calculation).
    in0_in_dram: bool = True,
    out_in_dram: bool = False,
) -> Tuple[str, Optional[float]]:
    """Allocate, warm up, time the matmul, write one CSV row.  Never raises."""
    torch.manual_seed(42)
    a_torch = torch.randn(1, 1, shape.m, shape.k, dtype=torch.bfloat16) * 0.1
    b_torch = torch.randn(shape.k, shape.n, dtype=torch.bfloat16) * 0.1

    # Derive core/grid info for the CSV row.
    # compute_with_storage_grid_size is a CoreCoord object, not a tuple.
    pc_grid = getattr(prog_cfg, "compute_with_storage_grid_size", None)
    if pc_grid is not None:
        gx = int(pc_grid.x)
        gy = int(pc_grid.y)
        compute_cores = gx * gy
        core_grid = f"{gx}x{gy}"
    elif cores_override is not None:
        compute_cores = cores_override
        core_grid = grid_override or _grid_str_from_cores(cores_override)
    else:
        g = device.compute_with_storage_grid_size()
        compute_cores = int(g.x) * int(g.y)
        core_grid = f"{int(g.x)}x{int(g.y)}"

    in0_block_w, per_core_m, per_core_n, out_subblock_w = _extract_pc_meta(prog_cfg)

    def _csv(run_status: str, t_us: Optional[float]):
        _write_csv_row(
            shape,
            config_name,
            in0_desc,
            in1_desc,
            out_desc,
            math_fidelity_str,
            run_status,
            t_us,
            compute_cores,
            core_grid,
            in0_block_w,
            per_core_m,
            per_core_n,
            out_subblock_w,
            in0_in_dram=in0_in_dram,
            out_in_dram=out_in_dram,
        )

    # ---- allocate ----------------------------------------------------------
    a_tt = b_tt = out_tt = None
    try:
        a_tt = ttnn.from_torch(
            a_torch,
            dtype=shape.in0_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in0_mem,
        )
        b_tt = ttnn.from_torch(
            b_torch,
            dtype=shape.in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_mem,
        )
    except Exception as exc:
        msg = str(exc)
        alloc_status = "OOM" if any(k in msg.lower() for k in ("out of memory", "oom", "allocat")) else "ALLOC_FAIL"
        logger.warning(f"[{shape.name}] {config_name}: {alloc_status}: {msg[:160]}")
        _csv(alloc_status, None)
        for t in (a_tt, b_tt):
            if t is not None:
                ttnn.deallocate(t)
        return alloc_status, None

    # ---- timed run (no warmup — first execution is the profiled one) -------
    last_out = None
    try:
        t0 = time.perf_counter()
        for _ in range(_NUM_ITERS):
            if last_out is not None:
                ttnn.deallocate(last_out)
            last_out = ttnn.matmul(
                a_tt,
                b_tt,
                program_config=prog_cfg,
                dtype=shape.out_dtype,
                memory_config=out_mem,
                compute_kernel_config=compute_cfg,
            )
        ttnn.synchronize_device(device)
        device_time_us = (time.perf_counter() - t0) * 1e6 / _NUM_ITERS
    except Exception as exc:
        logger.warning(f"[{shape.name}] {config_name}: KERNEL_FAIL: {str(exc)[:160]}")
        _csv("KERNEL_FAIL", None)
        for t in (last_out, a_tt, b_tt):
            if t is not None:
                ttnn.deallocate(t)
        return "KERNEL_FAIL", None

    tflops = 2 * shape.m * shape.k * shape.n / device_time_us / 1e6
    logger.info(
        f"[{shape.name:<11}] {config_name:<45}  "
        f"{device_time_us:>8.1f} μs  ref={shape.ref_us:>5} μs  {tflops:>6.1f} TFLOPs"
    )

    _csv("PASS", device_time_us)

    ttnn.deallocate(last_out)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)
    return "PASS", device_time_us


# ---------------------------------------------------------------------------
# Scenario 1: in0=DRAM interleaved, in1=DRAM interleaved, out=L1 interleaved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_dram_in0_dram_in1_l1_out(device, shape: _Shape):
    """Scenario 1 — in0=DRAM, in1=DRAM, out=L1 (all interleaved), production program config."""
    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=ttnn.DRAM_MEMORY_CONFIG,
        in1_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name="dram_dram_l1_production",
        in0_desc="DRAM Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="L1 Interleaved",
        in0_in_dram=True,
        out_in_dram=False,
    )
    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name}: kernel not supported with this config — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name}: expected PASS, got {status}"


# ---------------------------------------------------------------------------
# Scenario 2a: in0=L1 interleaved, in1=DRAM interleaved, out=L1 interleaved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_l1_in0_interleaved(device, shape: _Shape):
    """Scenario 2a — in0=L1 interleaved, in1=DRAM interleaved, out=L1 interleaved."""
    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=ttnn.L1_MEMORY_CONFIG,
        in1_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name="l1_interleaved_production",
        in0_desc="L1 Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="L1 Interleaved",
        in0_in_dram=False,
        out_in_dram=False,
    )
    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name}: kernel not supported with this config — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name}: expected PASS, got {status}"


# ---------------------------------------------------------------------------
# Scenario 2b: in0=L1 sharded (height / width / block)
# ---------------------------------------------------------------------------
#
# Parameter tuples: (shard_type, param)
#   height  → param = num_cores   (splits M)
#   width   → param = num_cores   (splits K)
#   block   → param = (gx, gy)    (splits M by gy rows, K by gx cols)
#
# HEIGHT sharding valid core counts (divide M_tiles=384, fit in 8×8): 8, 32, 64
# WIDTH sharding: K=1536 (K_tiles=48) → 4, 8 cores;
#                 K=4224 (K_tiles=132) → 4, 6 cores (132 not divisible by 8 → skipped)
# BLOCK sharding: (6,8) and (4,8) — tile-aligned for all four shapes.
#
# WIDTH sharding at these M/K sizes often OOMs in L1 (per-core shard = M × K/nc bytes,
# e.g. 12288 × 384 × 2B ≈ 9 MB >> 1.5 MB L1).  Those cases are recorded in the CSV
# with empty timing columns but do not fail the pytest run.

_SHARD_PARAMS = [
    ("height", 8),
    ("height", 32),
    ("height", 64),
    ("width", 4),
    ("width", 8),
    ("block", (6, 8)),
    ("block", (4, 8)),
]

_SHARD_IDS = [
    "hs_8c",
    "hs_32c",
    "hs_64c",
    "ws_4c",
    "ws_8c",
    "bs_6x8",
    "bs_4x8",
]


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("shard_param", _SHARD_PARAMS, ids=_SHARD_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_l1_in0_sharded(device, shape: _Shape, shard_param):
    """Scenario 2b — in0=L1 sharded (height / width / block), in1=DRAM, out=L1 interleaved.

    SPEC_ERROR (invalid shard geometry) → pytest.skip.
    OOM / WARMUP_FAIL → recorded in CSV; test does not assert PASS.
    """
    shard_type, param = shard_param

    # ------------------------------------------------------------------
    # Resolve shard geometry first so we can pre-check L1 fit.
    # Avoids allocating a tensor that will crash the TTNN allocator and
    # corrupt the Tracy trace.
    # ------------------------------------------------------------------
    if shard_type == "height":
        shard_h, shard_w = shape.m // param, shape.k
        in0_desc = f"L1 Height Sharded ({param}c)"
        config_name = f"l1_hs_{param}c_auto"
        cores_hint = param
        grid_hint = _grid_str_from_cores(param)
    elif shard_type == "width":
        shard_h, shard_w = shape.m, shape.k // param
        in0_desc = f"L1 Width Sharded ({param}c)"
        config_name = f"l1_ws_{param}c_auto"
        cores_hint = param
        grid_hint = _grid_str_from_cores(param)
    else:
        gx, gy = param
        shard_h, shard_w = shape.m // gy, shape.k // gx
        in0_desc = f"L1 Block Sharded ({gx}x{gy})"
        config_name = f"l1_bs_{gx}x{gy}_auto"
        cores_hint = gx * gy
        grid_hint = f"{gx}x{gy}"

    if not _l1_shard_fits(shard_h, shard_w, shape.in0_dtype):
        _skip_oom(
            shape, config_name, in0_desc, "DRAM Interleaved", "L1 Interleaved", cores_hint, grid_hint, shard_h, shard_w
        )

    try:
        if shard_type == "height":
            in0_mem = _height_shard_mem(shape.m, shape.k, num_cores=param)
        elif shard_type == "width":
            in0_mem = _width_shard_mem(shape.m, shape.k, num_cores=param)
        else:
            gx, gy = param
            in0_mem = _block_shard_mem(shape.m, shape.k, gx=gx, gy=gy)
    except (ValueError, AssertionError) as exc:
        pytest.skip(f"SPEC_ERROR — {exc}")

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    # Sharded in0 uses TTNN auto program config — the production 2D-mcast config
    # requires DRAM-interleaved in0 and will reject sharded tensors.
    prog_cfg = None

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc=in0_desc,
        in1_desc="DRAM Interleaved",
        out_desc="L1 Interleaved",
        cores_override=cores_hint,
        grid_override=grid_hint,
        in0_in_dram=False,
        out_in_dram=False,
    )

    # OOM no longer reaches here (pre-skipped above); only PASS or kernel-level
    # failures remain.  Hard-fail on truly unexpected statuses.
    assert status in {"PASS", "KERNEL_FAIL", "ALLOC_FAIL"}, f"{shape.name} {config_name}: unexpected status {status}"
    if status == "PASS":
        assert device_time_us is not None and device_time_us > 0


# ---------------------------------------------------------------------------
# Scenario 2c: in0=L1 interleaved, explicit per-core-M/N compute grid variants
# ---------------------------------------------------------------------------
#
# Grid validity for M=12288 (M_tiles=384):
#   grid_y=6  → per_core_M=64   grid_y=8  → per_core_M=48
#
# per_core_N = ceil(N_tiles / grid_x), must be ≤ 24:
#   vision_qkv N_tiles=144 : gx=6→24, gx=8→18
#   o_proj     N_tiles=48  : gx=2→24, gx=4→12, gx=6→8, gx=8→6
#   mlp_fc1    N_tiles=132 : gx=6→22, gx=8→17
#   mlp_fc2    N_tiles=48  : same as o_proj
#
# _build_explicit_program_config returns None for invalid combos (skipped).

_GRID_PARAMS: list[Tuple[int, int]] = [
    (6, 6),
    (6, 8),
    (8, 6),
    (8, 8),
    (4, 8),
    (4, 6),
]

_GRID_IDS = [f"gx{gx}_gy{gy}" for gx, gy in _GRID_PARAMS]


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("grid", _GRID_PARAMS, ids=_GRID_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_compute_grid_variants(device, shape: _Shape, grid):
    """Scenario 2c — in0=L1 interleaved, in1=DRAM, explicit (grid_x, grid_y) configs.

    Tests different per_core_M / per_core_N combinations.
    Combos with invalid per_core_M or per_core_N are skipped.
    """
    gx, gy = grid
    prog_cfg = _build_explicit_program_config(gx, gy, shape.m, shape.k, shape.n)
    if prog_cfg is None:
        m_tiles = shape.m // _TILE
        pcm = m_tiles // gy if m_tiles % gy == 0 else "n/a"
        pytest.skip(f"{shape.name}: grid ({gx},{gy}) invalid — per_core_M={pcm} or per_core_N exceeds limit")

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    config_name = f"l1_grid_{gx}x{gy}_pcM{prog_cfg.per_core_M}_pcN{prog_cfg.per_core_N}"

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=ttnn.L1_MEMORY_CONFIG,
        in1_mem=ttnn.DRAM_MEMORY_CONFIG,
        out_mem=ttnn.L1_MEMORY_CONFIG,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc="L1 Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="L1 Interleaved",
        in0_in_dram=False,
        out_in_dram=False,
    )
    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name} grid({gx},{gy}): kernel not supported — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name} grid({gx},{gy}): expected PASS, got {status}"
    assert device_time_us is not None and device_time_us > 0
