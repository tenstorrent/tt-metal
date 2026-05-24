# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for four SLOW matmul shapes from the dots.ocr Tracy report.

Scenarios
---------
1. in0=DRAM interleaved, in1=DRAM interleaved, out=L1 interleaved
1b. all four shapes — in0/in1/out all DRAM interleaved (``test_matmul_dram_all_interleaved``)
1c. all four shapes — in0=L1 interleaved, in1/out=DRAM interleaved (``test_matmul_l1_dram_dram_interleaved``)
2a. in0=L1 interleaved,  in1=DRAM interleaved, out=L1 interleaved
2b. in0=L1 sharded (height / width / block), in1=DRAM interleaved, out=L1 interleaved
2b-o. o_proj / mlp_fc1 (32c/64c) / mlp_fc2 (64c only) — L1 height-sharded in0 + DRAM out
     (``test_matmul_o_proj_l1_hs_dram_out``, ``test_matmul_mlp_fc1_l1_hs_dram_out``,
      ``test_matmul_mlp_fc2_l1_hs_dram_out``)
2c. in0=L1 interleaved,  in1=DRAM interleaved, explicit per-core-M/N grid variants
2d. in0=L1 width sharded, in1=DRAM sharded (height / width / block), out=L1 width sharded
    (``test_matmul_l1_ws_dram_in1_sharded_l1_ws_out``)

Shapes
------
  vision_qkv : 12288 × 1536 × 4608   BF16 × BFP8 → BFP8   ~1903 μs
  o_proj     : 12288 × 1536 × 1536   BFP8 × BFP8 → BFP8    ~705 μs
  mlp_fc1    : 12288 × 1536 × 4224   BFP8 × BFP8 → BFP8   ~1762 μs
  mlp_fc2    : 12288 × 4224 × 1536   BFP8 × BFP8 → BFP8   ~1485 μs

Results are written to models/experimental/tt_symbiote/tests/new_report/
as a timestamped CSV with one row per matmul run (includes program-config fields,
shard details, timing, and status for every attempt).

Memory combo sweep (``test_matmul_memory_combo_sweep``):
  L1 in0 × DRAM interleaved in1 × L1 out × {interleaved, height, width, block} × core grids.
  ~100 combos per shape (~400 tests total). Filter with ``-k test_matmul_memory_combo``.

Run all::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py -s

Run one scenario::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_in0_sharded" -s

Filter by shape::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "mlp_fc2" -s

Run one vision_qkv config at a time (use ``-s`` to see program-config prints)::

    # Scenario 1 — DRAM in0 / DRAM in1 / L1 out (production prog config)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_dram_in0_dram_in1_l1_out and vision_qkv" -s

    # Scenario 1d — DRAM/DRAM/L1 with explicit ibw=2/3/4 (8×8 pcM=48 pcN=18)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_dram_dram_l1_explicit_ibw and vision_qkv" -s

    # Scenario 1b — all shapes: DRAM in0 / DRAM in1 / DRAM out (all interleaved)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_dram_all_interleaved" -s

    # Scenario 1c — all shapes: L1 in0 / DRAM in1 / DRAM out (all interleaved)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_dram_dram_interleaved" -s

    # Scenario 2d — L1 width in0/out, DRAM-sharded in1 (height/width/block × 4/8c)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_ws_dram_in1_sharded_l1_ws_out and vision_qkv" -s

    # Scenario 2a — L1 interleaved in0
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_in0_interleaved and vision_qkv" -s

    # Scenario 2b — one shard variant (pick hs_8c, hs_32c, ws_4c, bs_6x8, ...)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_l1_in0_sharded and vision_qkv and hs_8c" -s

    # Scenario 2b-o — o_proj / mlp_fc1 L1 hs32/64c in0 + DRAM out
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_o_proj_l1_hs_dram_out" -s
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_mlp_fc1_l1_hs_dram_out" -s
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_mlp_fc2_l1_hs_dram_out" -s

    # Scenario 2c — one explicit grid (pick gx6_gy6, gx8_gy8, ...)
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py \
        -k "test_matmul_compute_grid_variants and vision_qkv and gx6_gy6" -s

    # Or target an exact parametrized node (no -k ambiguity):
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py::test_matmul_dram_in0_dram_in1_l1_out[vision_qkv-device_params0] -s
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py::test_matmul_l1_in0_sharded[vision_qkv-hs_8c-device_params0] -s
    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_unit.py::test_matmul_compute_grid_variants[vision_qkv-gx8_gy8-device_params0] -s

    # List all vision_qkv nodes: pytest ... --collect-only -q | grep vision_qkv

vision_qkv nodes (what each one tests)
--------------------------------------
All nodes use shape 12288×1536×4608 (BF16 activations × BFP8 weights → BFP8 out),
LoFi math, in1=DRAM interleaved. Tracy ref ≈ 1903 μs for the production path.

Program-config fields printed with ``-s``:

  in0_block_w   — K-direction tile blocks loaded into L1 per iteration (larger → fewer DRAM reads)
  per_core_M    — M tiles each core owns (M_tiles=384; grid_y=8 → 48)
  per_core_N    — N tiles each core owns (N_tiles=144; grid_x=8 → 18 for vision_qkv)
  out_subblock_w — output sub-block width in the matmul kernel (affects DST register use)

Scenario 1 — ``test_matmul_dram_in0_dram_in1_l1_out[vision_qkv-*]``
  Matches dots.ocr production: activations and weights in DRAM, output in L1.
  Uses ``_vision_matmul_program_config`` (tuned 8×8 grid, typically ibw=8 pcM=48 pcN=18).

Scenario 2a — ``test_matmul_l1_in0_interleaved[vision_qkv-*]``
  Activations pre-staged in L1 interleaved (no DRAM read for in0). Same production prog config.
  Isolates DRAM BW savings when in0 is already on-chip.

Scenario 2b — ``test_matmul_l1_in0_sharded[vision_qkv-<shard>-*]``
  Activations in L1 with sharding; TTNN auto program config (prog_cfg=None → prints ``auto``).

  hs_8c / hs_32c / hs_64c — HEIGHT sharded in0: split M across 8 / 32 / 64 cores.
  ws_4c / ws_8c           — WIDTH sharded in0: split K across 4 / 8 cores (often OOM-skipped).
  bs_6x8 / bs_4x8         — BLOCK sharded in0: 6×8 or 4×8 core grid over M and K.

Scenario 2c — ``test_matmul_compute_grid_variants[vision_qkv-<grid>-*]``
  L1 interleaved in0 with explicit ``MatmulMultiCoreReuseMultiCastProgramConfig`` per grid.
  Sweeps compute grid (grid_x × grid_y) and derived per_core_M / per_core_N:

  gx6_gy6 → 36 cores, pcM=64, pcN=24   |  gx6_gy8 → 48 cores, pcM=48, pcN=24
  gx8_gy6 → 48 cores, pcM=64, pcN=18   |  gx8_gy8 → 64 cores, pcM=48, pcN=18  (production grid)
  gx4_gy8 → 32 cores, pcM=48, pcN=36   |  gx4_gy6 → 24 cores, pcM=64, pcN=36
"""

from __future__ import annotations

import csv
import itertools
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    "in0 Shard Detail",
    "in1 Shard Detail",
    "out Shard Detail",
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


_LAYOUT_NAMES = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "INTERLEAVED",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "HEIGHT_SHARDED",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "WIDTH_SHARDED",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "BLOCK_SHARDED",
}


def _describe_mem_config(mem: ttnn.MemoryConfig) -> str:
    """One-line buffer + layout + shard summary for in0 / in1 / out."""
    buf = "DRAM" if mem.buffer_type == ttnn.BufferType.DRAM else "L1"
    layout = _LAYOUT_NAMES.get(mem.memory_layout, str(mem.memory_layout).split(".")[-1])
    parts = [f"{buf} {layout}"]

    if mem.shard_spec is not None:
        shard_h, shard_w = mem.shard_spec.shape
        nc = mem.shard_spec.grid.num_cores()
        ranges = list(mem.shard_spec.grid.ranges())
        if ranges:
            gs = ranges[0].grid_size()
            parts.append(f"grid={int(gs.x)}x{int(gs.y)}")
        parts.append(f"cores={nc}")
        parts.append(f"shard=[{shard_h}x{shard_w}]")

    return " ".join(parts)


def _div_up(n: int, d: int) -> int:
    return (n + d - 1) // d


def _infer_out_subblock_w(per_core_n: int) -> int:
    """Best-effort out_subblock_w (mirrors TTNN subblock divisibility heuristic)."""
    for w in range(min(8, per_core_n), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _infer_sharded_matmul_pc(shape: _Shape, in0_mem: ttnn.MemoryConfig) -> Optional[Tuple[int, int, int, int]]:
    """Infer TTNN auto matmul program config from sharded in0 (see get_matmul_program_config)."""
    if in0_mem.shard_spec is None:
        return None

    shard_h, shard_w = in0_mem.shard_spec.shape
    nc = in0_mem.shard_spec.grid.num_cores()
    ranges = list(in0_mem.shard_spec.grid.ranges())
    grid_x = int(ranges[0].grid_size().x) if ranges else 1
    grid_y = int(ranges[0].grid_size().y) if ranges else 1

    m_tiles = shape.m // _TILE
    k_tiles = shape.k // _TILE
    n_tiles = shape.n // _TILE
    layout = in0_mem.memory_layout

    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        per_core_m = shard_h // _TILE
        per_core_n = n_tiles
        in0_block_w = k_tiles
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        per_core_m = m_tiles
        per_core_n = _div_up(n_tiles, nc)
        in0_block_w = math.gcd(shard_w // _TILE, k_tiles)
    elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        per_core_m = _div_up(m_tiles, grid_y)
        per_core_n = _div_up(n_tiles, grid_x)
        k_per_shard = shard_w // _TILE
        cores_along_k = grid_x == k_tiles // k_per_shard if k_per_shard else False
        in0_block_w = math.gcd(k_per_shard, k_tiles) if cores_along_k else 1
    else:
        return None

    out_subblock_w = _infer_out_subblock_w(per_core_n)
    return in0_block_w, per_core_m, per_core_n, out_subblock_w


def _resolve_pc_meta(
    prog_cfg,
    shape: Optional[_Shape] = None,
    in0_mem: Optional[ttnn.MemoryConfig] = None,
) -> Tuple[str, str, str, str]:
    """(in0_block_w, per_core_M, per_core_N, out_subblock_w) from explicit or sharded-auto config."""
    if prog_cfg is not None:
        return _extract_pc_meta(prog_cfg)

    if shape is not None and in0_mem is not None:
        inferred = _infer_sharded_matmul_pc(shape, in0_mem)
        if inferred is not None:
            ibw, pcm, pcn, osw = inferred
            return str(ibw), str(pcm), str(pcn), str(osw)

    return "", "", "", ""


def _extract_pc_meta(prog_cfg) -> Tuple[str, str, str, str]:
    """(in0_block_w, per_core_M, per_core_N, out_subblock_w) from an explicit program config."""
    return (
        str(getattr(prog_cfg, "in0_block_w", "")),
        str(getattr(prog_cfg, "per_core_M", "")),
        str(getattr(prog_cfg, "per_core_N", "")),
        str(getattr(prog_cfg, "out_subblock_w", "")),
    )


def _print_program_config(
    shape: _Shape,
    config_name: str,
    prog_cfg,
    *,
    in0_mem: Optional[ttnn.MemoryConfig] = None,
    in1_mem: Optional[ttnn.MemoryConfig] = None,
    out_mem: Optional[ttnn.MemoryConfig] = None,
    core_grid: str = "",
    compute_cores: Optional[int] = None,
) -> None:
    """Print matmul program-config knobs and in0/in1/out sharding (pytest -s)."""
    inferred_auto = prog_cfg is None and in0_mem is not None and in0_mem.shard_spec is not None
    in0_block_w, per_core_m, per_core_n, out_subblock_w = _resolve_pc_meta(prog_cfg, shape, in0_mem)
    pc_grid = getattr(prog_cfg, "compute_with_storage_grid_size", None) if prog_cfg is not None else None
    if pc_grid is not None:
        gx, gy = int(pc_grid.x), int(pc_grid.y)
        grid = core_grid or f"{gx}x{gy}"
        cores = compute_cores if compute_cores is not None else gx * gy
    else:
        grid = core_grid or "auto"
        cores = compute_cores if compute_cores is not None else "auto"

    pc_tag = " (TTNN auto, inferred from in0 shard)" if inferred_auto else ""
    lines = [
        f"[{shape.name}] {config_name}{pc_tag}: "
        f"in0_block_w={in0_block_w or 'auto'}  "
        f"per_core_M={per_core_m or 'auto'}  "
        f"per_core_N={per_core_n or 'auto'}  "
        f"out_subblock_w={out_subblock_w or 'auto'}  "
        f"matmul_grid={grid}  matmul_cores={cores}",
    ]
    if in0_mem is not None and in1_mem is not None and out_mem is not None:
        lines.append(f"  in0: {_describe_mem_config(in0_mem)}")
        lines.append(f"  in1: {_describe_mem_config(in1_mem)}")
        lines.append(f"  out: {_describe_mem_config(out_mem)}")

    for line in lines:
        print(line)
        logger.info(line)


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
    # Best-effort PC inference for skip logging (no MemoryConfig object yet).
    try:
        if "Height" in in0_desc:
            tmp_mem = _height_shard_mem(shape.m, shape.k, num_cores=cores)
        elif "Width" in in0_desc:
            tmp_mem = _width_shard_mem(shape.m, shape.k, num_cores=cores)
        elif "Block" in in0_desc:
            gx, gy = (int(grid.split("x")[0]), int(grid.split("x")[1]))
            tmp_mem = _block_shard_mem(shape.m, shape.k, gx=gx, gy=gy)
        else:
            tmp_mem = None
        ibw, pcm, pcn, osw = _resolve_pc_meta(None, shape, tmp_mem) if tmp_mem else ("", "", "", "")
    except (ValueError, AssertionError):
        ibw, pcm, pcn, osw = "", "", "", ""
    print(
        f"[{shape.name}] {config_name}: SKIPPED_OOM\n"
        f"  in0_block_w={ibw or 'n/a'}  per_core_M={pcm or 'n/a'}  "
        f"per_core_N={pcn or 'n/a'}  out_subblock_w={osw or 'n/a'}  (inferred)\n"
        f"  in0: {in0_desc}  shard=[{shard_h}x{shard_w}]  (~{size_kb} KB/core)\n"
        f"  in1: {in1_desc}\n"
        f"  out: {out_desc}"
    )
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
    in0_shard_detail: str = "",
    in1_shard_detail: str = "",
    out_shard_detail: str = "",
    # Which tensors live in DRAM (affects DRAM BW calculation)
    in0_in_dram: bool = True,
    in1_in_dram: bool = True,
    out_in_dram: bool = False,
) -> None:
    """Compute derived metrics and append one row to the timestamped CSV."""
    if status == "PASS" and device_time_us is not None and device_time_us > 0:
        # DRAM traffic: in1 always DRAM; in0/out only if they reside in DRAM.
        dram_bytes = 0
        if in1_in_dram:
            dram_bytes += shape.k * shape.n * _elem_bytes(shape.in1_dtype)
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
        in0_shard_detail,
        in1_shard_detail,
        out_shard_detail,
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


def _height_shard_mem(m: int, dim: int, num_cores: int, buf: ttnn.BufferType = ttnn.BufferType.L1) -> ttnn.MemoryConfig:
    """HEIGHT_SHARDED: split m across num_cores, each shard is [m/nc, dim]."""
    if m % num_cores != 0:
        raise ValueError(f"M={m} not divisible by {num_cores}")
    shard_h = m // num_cores
    if shard_h % _TILE != 0:
        raise ValueError(f"shard_h={shard_h} not tile-aligned (M={m}, nc={num_cores})")
    if dim % _TILE != 0:
        raise ValueError(f"dim={dim} not tile-aligned")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buf,
        ttnn.ShardSpec(_compact_grid(num_cores), [shard_h, dim], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_shard_mem(m: int, dim: int, num_cores: int, buf: ttnn.BufferType = ttnn.BufferType.L1) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED: split dim across num_cores, each shard is [m, dim/nc]."""
    if dim % num_cores != 0:
        raise ValueError(f"dim={dim} not divisible by {num_cores}")
    shard_w = dim // num_cores
    if shard_w % _TILE != 0:
        raise ValueError(f"shard_w={shard_w} not tile-aligned (dim={dim}, nc={num_cores})")
    if m % _TILE != 0:
        raise ValueError(f"m={m} not tile-aligned")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buf,
        ttnn.ShardSpec(_compact_grid(num_cores), [m, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _block_shard_mem(
    m: int, dim: int, gx: int, gy: int, buf: ttnn.BufferType = ttnn.BufferType.L1
) -> ttnn.MemoryConfig:
    """BLOCK_SHARDED: split m by gy rows and dim by gx cols, shard=[m/gy, dim/gx]."""
    if m % gy != 0 or dim % gx != 0:
        raise ValueError(f"M={m} not divisible by gy={gy} or dim={dim} not divisible by gx={gx}")
    shard_h, shard_w = m // gy, dim // gx
    if shard_h % _TILE != 0 or shard_w % _TILE != 0:
        raise ValueError(f"shard [{shard_h}, {shard_w}] not tile-aligned for block({gx}x{gy})")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buf,
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
# Program config for DRAM/DRAM/L1 with explicit in0_block_w (8×8, pcM=48, pcN=18)
# ---------------------------------------------------------------------------

_DRAM_DRAM_L1_IBW_GRID = (8, 8)
_DRAM_DRAM_L1_IBW_PER_CORE_M = 48
_DRAM_DRAM_L1_IBW_PER_CORE_N = 18
_DRAM_DRAM_L1_IBW_MAX_OUT_BLOCK_H = 6  # L1-out headroom (see test_vision_qkv_matmul_tracy)


def _build_dram_dram_l1_ibw_program_config(
    in0_block_w: int,
    m: int,
    k: int,
    n: int,
    *,
    grid_x: int = _DRAM_DRAM_L1_IBW_GRID[0],
    grid_y: int = _DRAM_DRAM_L1_IBW_GRID[1],
    max_out_block_h: int | None = _DRAM_DRAM_L1_IBW_MAX_OUT_BLOCK_H,
) -> Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
    """8×8 2D-mcast config matching dram_dram_l1_production layout with explicit ibw.

    Uses the same per_core_M/N as production vision QKV (48/18) and caps
    ``out_block_h`` so matmul CBs leave room for the large L1 interleaved output.
    """
    if m % _TILE != 0 or k % _TILE != 0 or n % _TILE != 0:
        return None

    m_tiles = m // _TILE
    k_tiles = k // _TILE
    n_tiles = n // _TILE

    if m_tiles % grid_y != 0:
        return None
    per_core_m = m_tiles // grid_y
    per_core_n = (n_tiles + grid_x - 1) // grid_x

    if per_core_m != _DRAM_DRAM_L1_IBW_PER_CORE_M or per_core_n != _DRAM_DRAM_L1_IBW_PER_CORE_N:
        return None
    if per_core_n > 24 or per_core_m > 64:
        return None
    if in0_block_w < 1 or k_tiles % in0_block_w != 0:
        return None

    dst_tiles_budget = 8
    candidate_out_block_h = [16, 12, 8, 6, 4, 3, 2, 1]
    if max_out_block_h is not None:
        candidate_out_block_h = [h for h in candidate_out_block_h if h <= max_out_block_h]

    best_area = 0
    best_out_block_h = 1
    best_subblock_h = 1
    best_subblock_w = 1
    for ob_h in candidate_out_block_h:
        if ob_h > per_core_m or per_core_m % ob_h != 0:
            continue

        approx_interm_kb = (ob_h * per_core_n * 2048) // 1024
        approx_in0_kb = (ob_h * in0_block_w * 2 * 2048) // 1024
        if approx_interm_kb + approx_in0_kb > 1024:
            continue

        cand_area = 0
        cand_h = 1
        cand_w = 1
        for h in range(min(ob_h, dst_tiles_budget), 0, -1):
            if ob_h % h != 0:
                continue
            for w in range(min(per_core_n, dst_tiles_budget // h), 0, -1):
                if per_core_n % w != 0:
                    continue
                area = h * w
                if area > cand_area:
                    cand_area = area
                    cand_h = h
                    cand_w = w
                    break

        if (cand_area > best_area) or (cand_area == best_area and ob_h > best_out_block_h):
            best_area = cand_area
            best_out_block_h = ob_h
            best_subblock_h = cand_h
            best_subblock_w = cand_w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=best_subblock_h,
        out_subblock_w=best_subblock_w,
        out_block_h=best_out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


_DRAM_DRAM_L1_IBW_PARAMS = (2, 3, 4)
_DRAM_DRAM_L1_IBW_IDS = [f"ibw{ibw}" for ibw in _DRAM_DRAM_L1_IBW_PARAMS]


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
    in1_in_dram: bool = True,
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

    in0_block_w, per_core_m, per_core_n, out_subblock_w = _resolve_pc_meta(prog_cfg, shape, in0_mem)
    in0_shard_detail = _describe_mem_config(in0_mem)
    in1_shard_detail = _describe_mem_config(in1_mem)
    out_shard_detail = _describe_mem_config(out_mem)

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
            in0_shard_detail=in0_shard_detail,
            in1_shard_detail=in1_shard_detail,
            out_shard_detail=out_shard_detail,
            in0_in_dram=in0_in_dram,
            in1_in_dram=in1_in_dram,
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
    auto_note = "  (matmul PC inferred from in0 shard)" if prog_cfg is None and in0_mem.shard_spec is not None else ""
    print(
        f"[{shape.name}] {config_name}: PASS  "
        f"in0_block_w={in0_block_w}  per_core_M={per_core_m}  per_core_N={per_core_n}  "
        f"out_subblock_w={out_subblock_w}{auto_note}  "
        f"device_time_us={device_time_us:.1f}  ref_us={shape.ref_us}  tflops={tflops:.1f}"
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
    in0_mem = ttnn.DRAM_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG
    _print_program_config(shape, "dram_dram_l1_production", prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem)

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
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


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("in0_block_w", _DRAM_DRAM_L1_IBW_PARAMS, ids=_DRAM_DRAM_L1_IBW_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_dram_dram_l1_explicit_ibw(device, shape: _Shape, in0_block_w: int):
    """DRAM in0 / DRAM in1 / L1 out with explicit 8×8 pcM=48 pcN=18 and fixed in0_block_w.

    Sweeps ibw=2/3/4 (12/16/24 K iterations for K=1536) to compare L1 CB footprint vs
    production ibw=8 while keeping the same memory layout as ``dram_dram_l1_production``.
    """
    prog_cfg = _build_dram_dram_l1_ibw_program_config(in0_block_w, shape.m, shape.k, shape.n)
    if prog_cfg is None:
        pytest.skip(
            f"{shape.name}: shape does not match 8×8 pcM={_DRAM_DRAM_L1_IBW_PER_CORE_M} "
            f"pcN={_DRAM_DRAM_L1_IBW_PER_CORE_N} or ibw={in0_block_w} does not divide K tiles"
        )

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    config_name = (
        f"l1_grid_{_DRAM_DRAM_L1_IBW_GRID[0]}x{_DRAM_DRAM_L1_IBW_GRID[1]}"
        f"_pcM{_DRAM_DRAM_L1_IBW_PER_CORE_M}_pcN{_DRAM_DRAM_L1_IBW_PER_CORE_N}_ibw{in0_block_w}"
    )
    in0_mem = ttnn.DRAM_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG
    _print_program_config(shape, config_name, prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem)

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc="DRAM Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="L1 Interleaved",
        in0_in_dram=True,
        out_in_dram=False,
    )
    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name} ibw={in0_block_w}: kernel not supported — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name} ibw={in0_block_w}: expected PASS, got {status}"
    assert device_time_us is not None and device_time_us > 0


# ---------------------------------------------------------------------------
# Scenario 1b: in0/in1/out all DRAM interleaved (all four shapes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_dram_all_interleaved(device, shape: _Shape):
    """Scenario 1b — in0=DRAM, in1=DRAM, out=DRAM (all interleaved), production program config."""
    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)
    in0_mem = ttnn.DRAM_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.DRAM_MEMORY_CONFIG
    _print_program_config(
        shape, "dram_dram_dram_production", prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem
    )

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name="dram_dram_dram_production",
        in0_desc="DRAM Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="DRAM Interleaved",
        in0_in_dram=True,
        in1_in_dram=True,
        out_in_dram=True,
    )
    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name}: kernel not supported with this config — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name}: expected PASS, got {status}"


# ---------------------------------------------------------------------------
# Scenario 1c: in0=L1 interleaved, in1/out=DRAM interleaved (all four shapes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_l1_dram_dram_interleaved(device, shape: _Shape):
    """Scenario 1c — in0=L1 interleaved, in1=DRAM interleaved, out=DRAM interleaved."""
    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)
    in0_mem = ttnn.L1_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.DRAM_MEMORY_CONFIG
    _print_program_config(shape, "l1_dram_dram_production", prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem)

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name="l1_dram_dram_production",
        in0_desc="L1 Interleaved",
        in1_desc="DRAM Interleaved",
        out_desc="DRAM Interleaved",
        in0_in_dram=False,
        in1_in_dram=True,
        out_in_dram=True,
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
    in0_mem = ttnn.L1_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG
    _print_program_config(
        shape, "l1_interleaved_production", prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem
    )

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
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
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG
    _print_program_config(
        shape,
        config_name,
        prog_cfg,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        core_grid=grid_hint,
        compute_cores=cores_hint,
    )

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
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
# Scenario 2b-o: L1 height-sharded in0 + DRAM interleaved out
# ---------------------------------------------------------------------------
#
# o_proj: ``l1_hs_{32,64}c_auto`` (2b) PASSes with L1 out (~2189 / ~868 μs). DRAM
# out avoids the per-core L1 output stripe.
#
# mlp_fc1: N=4224 is not divisible by 8×32=256, so ``l1_hs_{32,64}c_auto`` with
# L1 interleaved out hits KERNEL_FAIL. DRAM out bypasses that tiling constraint
# (only dram_dram_dram_production and l1_dram_dram pass in the original sweep).
#
# mlp_fc2: ``l1_hs_64c_auto`` (~5.2 ms) already PASSes with L1 out; DRAM out
# drops the ~288 KB/core interleaved out stripe. Only 64c — hs_32c OOMs on in0
# (full K=4224 in each core’s shard ≈ 3.2 MB >> 1.5 MB L1).

_L1_HS_DRAM_OUT_CORES = (32, 64)
_L1_HS_DRAM_OUT_IDS = tuple(f"hs_{n}c" for n in _L1_HS_DRAM_OUT_CORES)
_O_PROJ_SHAPE = next(s for s in _SHAPES if s.name == "o_proj")
_MLP_FC1_SHAPE = next(s for s in _SHAPES if s.name == "mlp_fc1")
_MLP_FC2_SHAPE = next(s for s in _SHAPES if s.name == "mlp_fc2")


def _run_l1_hs_dram_out_matmul(device, shape: _Shape, num_cores: int) -> None:
    """in0=L1 height-sharded, in1=DRAM, out=DRAM (TTNN auto program config)."""
    shard_h, shard_w = shape.m // num_cores, shape.k
    in0_desc = f"L1 Height Sharded ({num_cores}c)"
    config_name = f"l1_hs_{num_cores}c_dram_out"
    cores_hint = num_cores
    grid_hint = _grid_str_from_cores(num_cores)

    if not _l1_shard_fits(shard_h, shard_w, shape.in0_dtype):
        _skip_oom(
            shape,
            config_name,
            in0_desc,
            "DRAM Interleaved",
            "DRAM Interleaved",
            cores_hint,
            grid_hint,
            shard_h,
            shard_w,
        )

    try:
        in0_mem = _height_shard_mem(shape.m, shape.k, num_cores=num_cores)
    except (ValueError, AssertionError) as exc:
        pytest.skip(f"SPEC_ERROR — {exc}")

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = None
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.DRAM_MEMORY_CONFIG
    _print_program_config(
        shape,
        config_name,
        prog_cfg,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        core_grid=grid_hint,
        compute_cores=cores_hint,
    )

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc=in0_desc,
        in1_desc="DRAM Interleaved",
        out_desc="DRAM Interleaved",
        cores_override=cores_hint,
        grid_override=grid_hint,
        in0_in_dram=False,
        in1_in_dram=True,
        out_in_dram=True,
    )

    if status == "KERNEL_FAIL":
        pytest.skip(f"{shape.name} {config_name}: kernel not supported — KERNEL_FAIL")
    assert status == "PASS", f"{shape.name} {config_name}: expected PASS, got {status}"
    assert device_time_us is not None and device_time_us > 0


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("num_cores", _L1_HS_DRAM_OUT_CORES, ids=_L1_HS_DRAM_OUT_IDS)
def test_matmul_o_proj_l1_hs_dram_out(device, num_cores: int):
    """o_proj — L1 height-sharded in0 (32c/64c), DRAM out."""
    _run_l1_hs_dram_out_matmul(device, _O_PROJ_SHAPE, num_cores)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("num_cores", _L1_HS_DRAM_OUT_CORES, ids=_L1_HS_DRAM_OUT_IDS)
def test_matmul_mlp_fc1_l1_hs_dram_out(device, num_cores: int):
    """mlp_fc1 — L1 height-sharded in0 (32c/64c), DRAM out (N=4224 L1-out tiling bypass)."""
    _run_l1_hs_dram_out_matmul(device, _MLP_FC1_SHAPE, num_cores)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
def test_matmul_mlp_fc2_l1_hs_dram_out(device):
    """mlp_fc2 — L1 height-sharded 64c in0, DRAM out (hs64c+L1 already passes)."""
    _run_l1_hs_dram_out_matmul(device, _MLP_FC2_SHAPE, 64)


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
    in0_mem = ttnn.L1_MEMORY_CONFIG
    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    out_mem = ttnn.L1_MEMORY_CONFIG
    _print_program_config(shape, config_name, prog_cfg, in0_mem=in0_mem, in1_mem=in1_mem, out_mem=out_mem)

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
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


# ---------------------------------------------------------------------------
# Scenario 2d: in0=L1 width, in1=DRAM sharded, out=L1 width
# ---------------------------------------------------------------------------
#
# in0/out split K / N across 4 or 8 cores (WIDTH sharding).  in1 sweeps DRAM
# height (8/32/64c), width (4/8c), and block (4×8, 6×8, 4×6, 8×8) on [K, N].
# Large per-core L1 shards often hit SKIPPED_OOM (same as scenario 2b width).

_IN1_DRAM_SHARD_PARAMS: list[Tuple[str, Union[int, Tuple[int, int]]]] = [
    ("height", 8),
    ("height", 32),
    ("height", 64),
    ("width", 4),
    ("width", 8),
    ("block", (4, 8)),
    ("block", (6, 8)),
    ("block", (4, 6)),
    ("block", (8, 8)),
]

_L1_WS_CORES = (4, 8)


def _in1_dram_shard_label(shard_type: str, param: Union[int, Tuple[int, int]]) -> str:
    if shard_type == "height":
        return f"hs_{param}c"
    if shard_type == "width":
        return f"ws_{param}c"
    gx, gy = param
    return f"bs_{gx}x{gy}"


def _build_dram_in1_shard_mem(shape: _Shape, shard_type: str, param: Union[int, Tuple[int, int]]) -> ttnn.MemoryConfig:
    if shard_type == "height":
        return _height_shard_mem(shape.k, shape.n, int(param), ttnn.BufferType.DRAM)
    if shard_type == "width":
        return _width_shard_mem(shape.k, shape.n, int(param), ttnn.BufferType.DRAM)
    gx, gy = param
    return _block_shard_mem(shape.k, shape.n, gx, gy, ttnn.BufferType.DRAM)


def _dram_in1_shard_desc(shard_type: str, param: Union[int, Tuple[int, int]]) -> str:
    if shard_type == "height":
        return f"DRAM Height Sharded ({param}c)"
    if shard_type == "width":
        return f"DRAM Width Sharded ({param}c)"
    gx, gy = param
    return f"DRAM Block Sharded ({gx}x{gy})"


def _generate_l1_ws_dram_in1_cases() -> List[Tuple[str, Union[int, Tuple[int, int]], int]]:
    cases: List[Tuple[str, Union[int, Tuple[int, int]], int]] = []
    for shard_type, param in _IN1_DRAM_SHARD_PARAMS:
        for ws_nc in _L1_WS_CORES:
            cases.append((shard_type, param, ws_nc))
    return cases


_L1_WS_DRAM_IN1_CASES = _generate_l1_ws_dram_in1_cases()
_L1_WS_DRAM_IN1_IDS = [
    f"l1ws_{ws_nc}c_dram_{_in1_dram_shard_label(st, pr)}_l1ws_{ws_nc}c" for st, pr, ws_nc in _L1_WS_DRAM_IN1_CASES
]

logger.info(f"[matmul_unit] L1-WS / DRAM-in1-shard / L1-WS sweep: {len(_L1_WS_DRAM_IN1_CASES)} cases per shape")


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("case", _L1_WS_DRAM_IN1_CASES, ids=_L1_WS_DRAM_IN1_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_l1_ws_dram_in1_sharded_l1_ws_out(device, shape: _Shape, case):
    """Scenario 2d — in0/out L1 width-sharded; in1 DRAM height/width/block-sharded."""
    shard_type, in1_param, ws_nc = case
    in0_desc = f"L1 Width Sharded ({ws_nc}c)"
    out_desc = f"L1 Width Sharded ({ws_nc}c)"
    in1_desc = _dram_in1_shard_desc(shard_type, in1_param)
    in1_id = _in1_dram_shard_label(shard_type, in1_param)
    config_name = f"l1_ws_{ws_nc}c_dram_{in1_id}_l1_ws_{ws_nc}c"
    cores_hint = ws_nc
    grid_hint = _grid_str_from_cores(ws_nc)

    in0_shard_h, in0_shard_w = shape.m, shape.k // ws_nc
    out_shard_h, out_shard_w = shape.m, shape.n // ws_nc

    if not _l1_shard_fits(in0_shard_h, in0_shard_w, shape.in0_dtype):
        _skip_oom(shape, config_name, in0_desc, in1_desc, out_desc, cores_hint, grid_hint, in0_shard_h, in0_shard_w)
    if not _l1_shard_fits(out_shard_h, out_shard_w, shape.out_dtype):
        _skip_oom(shape, config_name, in0_desc, in1_desc, out_desc, cores_hint, grid_hint, out_shard_h, out_shard_w)

    try:
        in0_mem = _width_shard_mem(shape.m, shape.k, ws_nc, ttnn.BufferType.L1)
        out_mem = _width_shard_mem(shape.m, shape.n, ws_nc, ttnn.BufferType.L1)
        in1_mem = _build_dram_in1_shard_mem(shape, shard_type, in1_param)
    except (ValueError, AssertionError) as exc:
        print(f"[{shape.name}] {config_name}: SPEC_ERROR — {exc}")
        _write_csv_row(
            shape,
            config_name,
            in0_desc,
            in1_desc,
            out_desc,
            "LoFi",
            "SPEC_ERROR",
            None,
            0,
            "",
            "",
            "",
            "",
            "",
        )
        pytest.skip(f"SPEC_ERROR — {exc}")

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    prog_cfg = None
    _print_program_config(
        shape,
        config_name,
        prog_cfg,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        core_grid=grid_hint,
        compute_cores=cores_hint,
    )

    status, device_time_us = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc=in0_desc,
        in1_desc=in1_desc,
        out_desc=out_desc,
        cores_override=cores_hint,
        grid_override=grid_hint,
        in0_in_dram=False,
        in1_in_dram=True,
        out_in_dram=False,
    )
    assert status in {"PASS", "KERNEL_FAIL", "ALLOC_FAIL"}, f"{shape.name} {config_name}: unexpected status {status}"
    if status == "PASS":
        assert device_time_us is not None and device_time_us > 0


# ---------------------------------------------------------------------------
# Memory layout combination sweep (in0 / out × core grids; in1 = DRAM interleaved)
# ---------------------------------------------------------------------------
#
# Sweeps L1 in0 × DRAM interleaved in1 × L1 out across interleaved / height /
# width / block layouts and valid core-grid sizes (production-style weights).
#
# Every run is recorded in the same matmul_unit_*.csv (including OOM / kernel fail).
# Use -k to filter, e.g.:
#   pytest ... -k "test_matmul_memory_combo and vision_qkv" -s
#   pytest ... -k "combo_l1_height_8c_dram_interleaved_l1_block_6x8" -s

_LAYOUTS = ("interleaved", "height", "width", "block")
_HS_CORES = (8, 32, 64)
_WS_CORES = (4, 8)
_BLOCK_GRIDS = ((4, 8), (6, 8), (4, 6), (8, 8))
_IN0_BUF = ttnn.BufferType.L1
_IN1_BUF = ttnn.BufferType.DRAM
_OUT_BUF = ttnn.BufferType.L1

GridParam = Union[None, int, Tuple[int, int]]


@dataclass(frozen=True)
class _MemComboSpec:
    """One tensor memory choice (buffer + layout + shard grid)."""

    role: str
    layout: str
    buf: ttnn.BufferType
    grid_param: GridParam = None

    @property
    def label(self) -> str:
        buf_s = "dram" if self.buf == ttnn.BufferType.DRAM else "l1"
        if self.layout == "interleaved":
            return f"{buf_s}_interleaved"
        if self.layout == "height":
            return f"{buf_s}_height_{self.grid_param}c"
        if self.layout == "width":
            return f"{buf_s}_width_{self.grid_param}c"
        gx, gy = self.grid_param
        return f"{buf_s}_block_{gx}x{gy}"

    def human_desc(self) -> str:
        buf_s = "DRAM" if self.buf == ttnn.BufferType.DRAM else "L1"
        if self.layout == "interleaved":
            return f"{buf_s} interleaved"
        if self.layout == "height":
            return f"{buf_s} height_sharded ({self.grid_param}c)"
        if self.layout == "width":
            return f"{buf_s} width_sharded ({self.grid_param}c)"
        gx, gy = self.grid_param
        return f"{buf_s} block_sharded ({gx}x{gy})"

    def matmul_grid_hint(self) -> Tuple[str, int]:
        if self.layout == "interleaved":
            return "interleaved", 0
        if self.layout == "height":
            nc = int(self.grid_param)
            return _grid_str_from_cores(nc), nc
        if self.layout == "width":
            nc = int(self.grid_param)
            return _grid_str_from_cores(nc), nc
        gx, gy = self.grid_param
        return f"{gx}x{gy}", gx * gy

    def build_in0(self, shape: _Shape) -> ttnn.MemoryConfig:
        if self.layout == "interleaved":
            return ttnn.L1_MEMORY_CONFIG if self.buf == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG
        if self.layout == "height":
            return _height_shard_mem(shape.m, shape.k, int(self.grid_param), self.buf)
        if self.layout == "width":
            return _width_shard_mem(shape.m, shape.k, int(self.grid_param), self.buf)
        gx, gy = self.grid_param
        return _block_shard_mem(shape.m, shape.k, gx, gy, self.buf)

    def build_in1(self, shape: _Shape) -> ttnn.MemoryConfig:
        if self.layout == "interleaved":
            return ttnn.DRAM_MEMORY_CONFIG if self.buf == ttnn.BufferType.DRAM else ttnn.L1_MEMORY_CONFIG
        if self.layout == "height":
            return _height_shard_mem(shape.k, shape.n, int(self.grid_param), self.buf)
        if self.layout == "width":
            return _width_shard_mem(shape.k, shape.n, int(self.grid_param), self.buf)
        gx, gy = self.grid_param
        return _block_shard_mem(shape.k, shape.n, gx, gy, self.buf)

    def build_out(self, shape: _Shape) -> ttnn.MemoryConfig:
        if self.layout == "interleaved":
            return ttnn.L1_MEMORY_CONFIG if self.buf == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG
        if self.layout == "height":
            return _height_shard_mem(shape.m, shape.n, int(self.grid_param), self.buf)
        if self.layout == "width":
            return _width_shard_mem(shape.m, shape.n, int(self.grid_param), self.buf)
        gx, gy = self.grid_param
        return _block_shard_mem(shape.m, shape.n, gx, gy, self.buf)


@dataclass(frozen=True)
class _MemComboCase:
    in0: _MemComboSpec
    in1: _MemComboSpec
    out: _MemComboSpec

    @property
    def config_name(self) -> str:
        return f"combo_{self.in0.label}_{self.in1.label}_{self.out.label}"

    @property
    def case_id(self) -> str:
        return self.config_name


# Weights stay DRAM interleaved (matches model path); only in0/out are swept.
_IN1_INTERLEAVED = _MemComboSpec("in1", "interleaved", _IN1_BUF, None)


def _grid_options(layout: str) -> List[GridParam]:
    if layout == "interleaved":
        return [None]
    if layout == "height":
        return list(_HS_CORES)
    if layout == "width":
        return list(_WS_CORES)
    return list(_BLOCK_GRIDS)


def _generate_mem_combo_cases() -> List[_MemComboCase]:
    """in0/out layout × core grids; in1 fixed to DRAM interleaved."""
    cases: List[_MemComboCase] = []
    for i0_l, o_l in itertools.product(_LAYOUTS, repeat=2):
        for i0_g, o_g in itertools.product(_grid_options(i0_l), _grid_options(o_l)):
            cases.append(
                _MemComboCase(
                    in0=_MemComboSpec("in0", i0_l, _IN0_BUF, i0_g),
                    in1=_IN1_INTERLEAVED,
                    out=_MemComboSpec("out", o_l, _OUT_BUF, o_g),
                )
            )
    return cases


_MEM_COMBO_CASES = _generate_mem_combo_cases()
_MEM_COMBO_IDS = [c.case_id for c in _MEM_COMBO_CASES]

logger.info(f"[matmul_unit] memory combo sweep: {len(_MEM_COMBO_CASES)} cases per shape")


def _l1_shard_bytes(shape: _Shape, spec: _MemComboSpec) -> Optional[int]:
    """Per-core shard bytes for L1 tensors (None if interleaved)."""
    if spec.buf != ttnn.BufferType.L1 or spec.layout == "interleaved":
        return None
    dt = {"in0": shape.in0_dtype, "in1": shape.in1_dtype, "out": shape.out_dtype}[spec.role]
    m_dim, dim = {
        "in0": (shape.m, shape.k),
        "in1": (shape.k, shape.n),
        "out": (shape.m, shape.n),
    }[spec.role]
    if spec.layout == "height":
        nc = int(spec.grid_param)
        return (m_dim // nc) * dim * _elem_bytes(dt)
    if spec.layout == "width":
        nc = int(spec.grid_param)
        return m_dim * (dim // nc) * _elem_bytes(dt)
    gx, gy = spec.grid_param
    return (m_dim // gy) * (dim // gx) * _elem_bytes(dt)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("combo", _MEM_COMBO_CASES, ids=_MEM_COMBO_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_matmul_memory_combo_sweep(device, shape: _Shape, combo: _MemComboCase):
    """Sweep in0/in1/out memory layouts (L1/DRAM × interleaved/height/width/block) × core grids."""
    in0_desc = combo.in0.human_desc()
    in1_desc = combo.in1.human_desc()
    out_desc = combo.out.human_desc()
    config_name = combo.config_name

    for spec, desc in ((combo.in0, in0_desc), (combo.out, out_desc)):
        shard_b = _l1_shard_bytes(shape, spec)
        if shard_b is not None and shard_b > _L1_PER_CORE_BYTES:
            size_kb = shard_b // 1024
            print(
                f"[{shape.name}] {config_name}: SKIPPED_OOM\n"
                f"  {spec.role}: {desc}  (~{size_kb} KB/core > L1 budget)\n"
            )
            _write_csv_row(
                shape,
                config_name,
                in0_desc,
                in1_desc,
                out_desc,
                "LoFi",
                "SKIPPED_OOM",
                None,
                0,
                "",
                "",
                "",
                "",
                "",
            )
            pytest.skip(f"L1 OOM — {spec.role} shard ~{size_kb} KB/core")

    try:
        in0_mem = combo.in0.build_in0(shape)
        in1_mem = combo.in1.build_in1(shape)
        out_mem = combo.out.build_out(shape)
    except (ValueError, AssertionError) as exc:
        print(f"[{shape.name}] {config_name}: SPEC_ERROR — {exc}")
        _write_csv_row(
            shape,
            config_name,
            in0_desc,
            in1_desc,
            out_desc,
            "LoFi",
            "SPEC_ERROR",
            None,
            0,
            "",
            "",
            "",
            "",
            "",
        )
        pytest.skip(f"SPEC_ERROR — {exc}")

    if combo.in0.layout == "interleaved":
        prog_cfg = _vision_matmul_program_config(device, shape.m, shape.k, shape.n)
    else:
        prog_cfg = None

    compute_cfg = _vision_matmul_compute_config(device, math_fidelity=ttnn.MathFidelity.LoFi)
    _, matmul_cores = combo.in0.matmul_grid_hint()
    matmul_grid = combo.in0.matmul_grid_hint()[0] if prog_cfg is None else ""

    _print_program_config(
        shape,
        config_name,
        prog_cfg,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        core_grid=matmul_grid or combo.in0.matmul_grid_hint()[0],
        compute_cores=matmul_cores or None,
    )

    status, _ = _run_matmul(
        device,
        shape,
        in0_mem=in0_mem,
        in1_mem=in1_mem,
        out_mem=out_mem,
        prog_cfg=prog_cfg,
        compute_cfg=compute_cfg,
        config_name=config_name,
        in0_desc=in0_desc,
        in1_desc=in1_desc,
        out_desc=out_desc,
        cores_override=matmul_cores if matmul_cores > 0 else None,
        grid_override=combo.in0.matmul_grid_hint()[0] if prog_cfg is None and matmul_cores > 0 else None,
        in0_in_dram=combo.in0.buf == ttnn.BufferType.DRAM,
        in1_in_dram=combo.in1.buf == ttnn.BufferType.DRAM,
        out_in_dram=combo.out.buf == ttnn.BufferType.DRAM,
    )
    assert status in {"PASS", "KERNEL_FAIL", "ALLOC_FAIL", "OOM", "SKIPPED_OOM"}
