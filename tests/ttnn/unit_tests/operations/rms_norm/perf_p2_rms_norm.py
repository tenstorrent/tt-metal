# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 scoreboard probe (not collected by a directory run — no `test_` prefix).

    P2_CELLS=hgt_tile_g,blk_perf_7168 scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/rms_norm/perf_p2_rms_norm.py

Every cell is built exactly as the golden harness builds it (eval/golden_tests/rms_norm/helpers.run_rms_norm:
same torch seed, auto_shard_config / pinned shard geometry, gamma tensor, compute config), one dispatch per cell,
so the profiler CSV rows map 1:1 to the cells (plus the harness's own readback programs).
Env:
  P2_CELLS    comma list of cell ids to keep (default: all)
  P2_ABLATE   comma list of ablation defines (RMS_ABLATE_EMPTY, RMS_ABLATE_DM, CKL_ELTWISE_CHAIN_SKIP_COMPUTE)
  P2_NOGAMMA=1  drop gamma (gamma-cost ablation)
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
F32, BF16, BF8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b


def _res(shape, ml, layout=ttnn.TILE_LAYOUT):
    """golden resilience loose case: bf16, fp32 DEST off, HiFi4, gamma bf16 in the input's Layout."""
    return _cell(shape, layout=layout, ml=ml, fp32=False, glayout=layout)


CELLS = {
    # ---- INTERLEAVED (DRAM-bound) ----
    "int_prefill_7168": _perf((1, 1, 8192, 7168)),
    "int_prefill_5120": _perf((1, 1, 8192, 5120)),
    "int_prefill_1024": _perf((1, 1, 8192, 1024)),
    "int_rm_99991x64": _res((99991, 64), ML.INTERLEAVED, RM),
    "int_tile_13x777x1023": _res((13, 777, 1023), ML.INTERLEAVED),
    "int_tile_f32_128x8192": _cell((128, 8192), dtype=F32, gdtype=F32),
    "int_tile_bf8_128x8192": _cell((128, 8192), dtype=BF8, gdtype=BF8),
    # ---- WIDTH_SHARDED TILE: large-k owner combine (auto geometry) + perf decode ----
    "wid_tile_128x4096": _cell((1, 1, 128, 4096), ml=ML.WIDTH_SHARDED),
    "wid_tile_bf8_128x4096_nog": _cell((1, 1, 128, 4096), dtype=BF8, ml=ML.WIDTH_SHARDED, gdtype=None),
    "wid_tile_224x3072": _res((1, 1, 224, 3072), ML.WIDTH_SHARDED),
    "wid_tile_160x11008": _res((1, 1, 160, 11008), ML.WIDTH_SHARDED),
    "wid_tile_13x777x1023": _res((13, 777, 1023), ML.WIDTH_SHARDED),
    "wid_perf_32x7168": _perf((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    # ---- WIDTH_SHARDED ROW_MAJOR: 16 B shard sticks, k = W / 8 ----
    "wid_rm_5x3x928x544": _res((5, 3, 928, 544), ML.WIDTH_SHARDED, RM),
    "wid_rm_13x777x1023": _res((13, 777, 1023), ML.WIDTH_SHARDED, RM),
    "wid_rm_1024x1024": _cell((1024, 1024), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM, gdtype=F32),
    "wid_rm_f32_4x1x512x512": _cell((4, 1, 512, 512), dtype=F32, layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    # ---- HEIGHT_SHARDED ----
    "hgt_rm_f32_128x4096": _cell((1, 1, 128, 4096), dtype=F32, layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM, gdtype=F32),
    "hgt_rm_224x11008": _res((1, 224, 11008), ML.HEIGHT_SHARDED, RM),
    "hgt_tile_128x8192": _cell((128, 8192), ml=ML.HEIGHT_SHARDED),
    "hgt_tile_res_128x8192": _cell((128, 8192), ml=ML.HEIGHT_SHARDED, fp32=False, glayout=RM, gdtype=F32),
    "hgt_tile_bf8_2048x256": _cell((1, 1, 2048, 256), dtype=BF8, ml=ML.HEIGHT_SHARDED, fp32=False, gdtype=BF8),
    # ---- BLOCK_SHARDED ----
    "blk_perf_7168x1024": _perf((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    "blk_perf_7168x2048": _perf((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
    "blk_tile_128x4096": _cell((1, 1, 128, 4096), ml=ML.BLOCK_SHARDED, glayout=RM, gdtype=F32),
    "blk_rm_99991x64": _res((99991, 64), ML.BLOCK_SHARDED, RM),
    "blk_rm_128x4096": _cell((1, 1, 128, 4096), layout=RM, ml=ML.BLOCK_SHARDED, fp32=False, glayout=RM, gdtype=F32),
}
_sel = os.environ.get("P2_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]


@pytest.mark.parametrize("cell", IDS)
def test_perf(device, cell, monkeypatch):
    ablate = tuple(a for a in os.environ.get("P2_ABLATE", "").split(",") if a)
    monkeypatch.setattr(pd, "PERF_ABLATE", ablate)
    axes = dict(CELLS[cell])
    if os.environ.get("P2_NOGAMMA") == "1":
        axes.update(gamma_mode="no_gamma", gamma_dtype="none", gamma_layout="none")
    try:
        run_rms_norm(device=device, **axes)
    except Exception as e:  # ablated kernels are wrong by design; the run is for timing only
        if not ablate:
            raise
        print(f"P2 {cell}: ablated output mismatch (expected): {type(e).__name__}")
