# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 before/after probe (not collected by a directory run — no `test_` prefix).

Runs each cell twice in one session: `base` = the op's ProgramDescriptor + kernels from a checkout of the round-start
commit (P1AB_BASE_DIR, e.g. a `git worktree` of 8216b2fa1e6), `head` = the current op. Cells are the golden-harness
cells of perf_p1_rms_norm.CELLS (P1AB_CELLS selects).

    git worktree add /tmp/perf1/base_wt 8216b2fa1e6
    P1AB_BASE_DIR=/tmp/perf1/base_wt P1AB_CELLS=... eval/eval_test_runner.sh --no-precompile <this file> <out>
"""

import importlib.util
import os
import sys

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import run_rms_norm

sys.path.insert(0, os.path.dirname(__file__))
from perf_p1_rms_norm import CELLS, _cell  # noqa: E402

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
EXTRA = {
    "int_2047x2047": _cell((2047, 2047), fp32=False),
    "int_rm_7136x736": _cell((7136, 736), layout=RM, glayout=RM, fp32=False),
    "int_3x3104x544": _cell((3, 3104, 544), fp32=False),
    "int_13x777x1023": _cell((13, 777, 1023), fp32=False),
    "int_rm_99991x64": _cell((99991, 64), layout=RM, glayout=RM, fp32=False),
    "int_f32_256x256": _cell((256, 256), dtype=ttnn.float32, gdtype=ttnn.float32),
}
_PERF = {"math_fidelity": ttnn.MathFidelity.HiFi2, "pcc_threshold": 0.9995}
for _w in (256, 544, 736, 1024, 1536):  # k = 1 W sweep at 2 tile-rows per Tensix core (NoC-swap domain check)
    EXTRA[f"int_w_8192x{_w}"] = _cell((1, 1, 8192, _w), fp32=False, extras=dict(_PERF))
EXTRA["int_w_9312x544"] = _cell((1, 1, 9312, 544), fp32=False, extras=dict(_PERF))  # 291 tile-rows, 2-3 per core
ALL = {**CELLS, **EXTRA}
_sel = os.environ.get("P1AB_CELLS")
IDS = [c for c in ALL if not _sel or c in _sel.split(",")]
_BASE = os.environ.get("P1AB_BASE_DIR", "/tmp/perf1/base_wt")


def _base_create():
    path = os.path.join(_BASE, "ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py")
    spec = importlib.util.spec_from_file_location("rms_norm_pd_round_start", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod.create_program_descriptor


VARIANTS = os.environ.get("P1AB_VARIANTS", "base,head").split(",")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cell", IDS)
def test_ab(device, cell, variant, monkeypatch):
    if variant == "base":
        monkeypatch.setattr(
            sys.modules["ttnn.operations.rms_norm.rms_norm"], "create_program_descriptor", _base_create()
        )
    elif variant == "head_noswap":  # head with the DRAM-adjacent NoC swap off
        from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

        monkeypatch.setattr(pd, "NOC_SWAP_COLUMNS", 0)
    run_rms_norm(device=device, **ALL[cell])
