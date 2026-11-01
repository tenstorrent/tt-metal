# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 scoreboard probe (not collected by a directory run — no `test_` prefix).

    P1_CELLS=hgt_tile_g,blk_perf_7168 scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py

Every cell is built exactly as the golden harness builds it (eval/golden_tests/rms_norm/helpers.run_rms_norm:
same torch seed, auto_shard_config / pinned shard geometry, gamma tensor, compute config), one dispatch per cell,
so the profiler CSV rows map 1:1 to the cells (plus the harness's own readback programs).
Env:
  P1_CELLS    comma list of cell ids to keep (default: all)
  P1_ABLATE   comma list of ablation defines (RMS_ABLATE_EMPTY, RMS_ABLATE_DM, CKL_ELTWISE_CHAIN_SKIP_COMPUTE)
  P1_NOGAMMA=1  drop gamma (gamma-cost ablation)
"""

import os

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import run_rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

ML = ttnn.TensorMemoryLayout
HIFI2 = ttnn.MathFidelity.HiFi2


def _cell(
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    ml=ML.INTERLEAVED,
    fp32=True,
    gdtype=ttnn.bfloat16,
    glayout=ttnn.TILE_LAYOUT,
    extras=None,
):
    return dict(
        inputs=(shape,),
        dtype=dtype,
        layout=layout,
        memory_layout=ml,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma" if gdtype is not None else "no_gamma",
        gamma_dtype=gdtype or "none",
        gamma_layout=glayout if gdtype is not None else "none",
        extras=extras or {},
    )


def _perf(shape, ml=ML.INTERLEAVED, shard=None):
    ex = {"math_fidelity": HIFI2, "pcc_threshold": 0.9995}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    return _cell(shape, ml=ml, fp32=False, extras=ex)


RM = ttnn.ROW_MAJOR_LAYOUT
CELLS = {
    # interleaved TILE: perf profiles (bf16, HiFi2, fp32 DEST off)
    "int_prefill_7168": _perf((1, 1, 8192, 7168)),
    "int_prefill_1024": _perf((1, 1, 8192, 1024)),
    "int_mid_7168": _perf((1, 1, 640, 7168)),
    "int_decode_32768": _cell((1, 1, 32, 32768)),  # _WIDE loose case (HiFi4, fp32 DEST)
    # sharded perf profiles (pinned measured-fastest geometries)
    "blk_perf_7168x1024": _perf((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    "blk_perf_8192x1024": _perf((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    "blk_perf_7168x2048": _perf((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
    "blk_perf_640x6144": _perf((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    "wid_perf_32x7168": _perf((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    "wid_perf_32x5120": _perf((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),
    # golden cartesian cells (HiFi4, auto_shard_config geometry)
    "hgt_tile_g": _cell((128, 8192), ml=ML.HEIGHT_SHARDED),
    "hgt_tile_grm": _cell((128, 8192), ml=ML.HEIGHT_SHARDED, glayout=RM),
    "hgt_tile_gbf8": _cell((128, 8192), ml=ML.HEIGHT_SHARDED, gdtype=ttnn.bfloat8_b),
    "hgt_tile_4096_g": _cell((1, 1, 128, 4096), ml=ML.HEIGHT_SHARDED),
    "hgt_rm_grm": _cell((128, 8192), layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM),
    "wid_tile_2048x256": _cell((1, 1, 2048, 256), ml=ML.WIDTH_SHARDED),
    "wid_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    "blk_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
    "int_rm_128x4096_gf32": _cell((1, 1, 128, 4096), layout=RM, glayout=RM, gdtype=ttnn.float32),
    "int_tile_f32_128x8192": _cell((128, 8192), dtype=ttnn.float32, gdtype=ttnn.float32),
    "int_tile_bf8_128x8192": _cell((128, 8192), dtype=ttnn.bfloat8_b, gdtype=ttnn.bfloat16),
    "int_tile_f32_2x4x128x512": _cell((2, 4, 128, 512), dtype=ttnn.float32, gdtype=ttnn.float32),
    # the test_regression cell flagged slower in the R5 snapshot (fp32, no gamma, defaults)
    "reg_f32_2x1x128x256": _cell((2, 1, 128, 256), dtype=ttnn.float32, gdtype=None),
}
_sel = os.environ.get("P1_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]


@pytest.mark.parametrize("cell", IDS)
def test_perf(device, cell, monkeypatch):
    ablate = tuple(a for a in os.environ.get("P1_ABLATE", "").split(",") if a)
    monkeypatch.setattr(pd, "PERF_ABLATE", ablate)
    axes = dict(CELLS[cell])
    if os.environ.get("P1_NOGAMMA") == "1":
        axes.update(gamma_mode="no_gamma", gamma_dtype="none", gamma_layout="none")
    try:
        run_rms_norm(device=device, **axes)
    except Exception as e:  # ablated kernels are wrong by design; the run is for timing only
        if not ablate:
            raise
        print(f"P1 {cell}: ablated output mismatch (expected): {type(e).__name__}")
