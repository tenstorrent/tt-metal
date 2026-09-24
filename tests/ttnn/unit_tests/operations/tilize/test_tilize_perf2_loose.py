# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 harness: run LOOSE_CASES (by index) against kernel-dir variants, under --profile.

TILIZE_P2_CASES="0,7"            LOOSE_CASES indices (default "0", the perf focus); an entry "s1x1x32x4096"
                                 is an extra shape with LOOSE_CASES[0]'s placement (DRAM interleaved, bf16)
TILIZE_P2_VARIANTS="head,<dir>"  kernel dirs: "head" = the op's kernels/, else a path relative to
                                 ttnn/ttnn/operations/tilize/perf_experiments/ (or absolute)
                                 A variant may carry descriptor-knob overrides after "@", joined
                                 by "+": "head@CO_READ_SPLIT='positional'+ONEPOS_SUB_BLOCK_TILES=0"
TILIZE_P2_CHECK=0                skip the golden contract check (ablated variants)
Each (case, variant) runs the op once; the golden helper checks the contract unless disabled.
Opt-in: TILIZE_PERF_EXPERIMENTS=1.
"""
import ast
import os
from pathlib import Path

import pytest

import ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.feature_matrix import cartesian
from eval.golden_tests.tilize import helpers
from eval.golden_tests.tilize.feature_spec import LOOSE_CASES, TARGET
from ttnn.operations.tilize import INPUT_TAGGERS  # type: ignore

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments"
CASES = os.environ.get("TILIZE_P2_CASES", "0").split(",")
VARIANTS = os.environ.get("TILIZE_P2_VARIANTS", "head").split(",")
CHECK = os.environ.get("TILIZE_P2_CHECK", "1") != "0"


def _axes(case):
    inputs = case["inputs"]
    dt = case.get("dtype", ttnn.bfloat16)
    odt = case.get("output_dtype", dt)
    return next(a for a in cartesian(TARGET, INPUT_TAGGERS, inputs) if a["dtype"] == dt and a["output_dtype"] == odt)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("idx", CASES)
def test_loose_variant(device, monkeypatch, idx, variant):
    kernels, _, knobs = variant.partition("@")
    for knob in filter(None, knobs.split("+")):
        name, _, value = knob.partition("=")
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, ast.literal_eval(value))
    if kernels != "head":
        d = Path(kernels)
        monkeypatch.setattr(pd, "KERNEL_DIR", d if d.is_absolute() else EXP / d)
    if idx.startswith("s"):
        base = LOOSE_CASES[0]
        scenario = dict(base["inputs"][0], input_shape=[int(d) for d in idx[1:].split("x")])
        case = dict(base, inputs=(scenario,))
    else:
        case = LOOSE_CASES[int(idx)]
    axes = _axes(case)
    if not CHECK:
        monkeypatch.setattr(helpers, "check_output", lambda *a, **k: None)
    helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **axes)
    ttnn.synchronize_device(device)
    print(f"P2 case={idx} variant={variant} done")
