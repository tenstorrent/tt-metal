# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench: one call per LOOSE_CASES perf row (run under scripts/run_safe_pytest.sh --profile).

LN_KNOBS="NAME=int,..." overrides layer_norm_tile_program_descriptor knob constants for A/B runs.
"""

import os

import importlib

import pytest

# By module name: `import a.b.c as d` resolves through package attributes, which can bind a second copy
# of the module (sys.modules also holds `layer_norm_tile.layer_norm_tile_program_descriptor`), and
# then the overrides never reach select_blocking.
descriptor = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

PERF_CASES = [case for case in LOOSE_CASES if "achievable_ns" in case.get("extras", {})]


@pytest.fixture(autouse=True)
def _knob_overrides(monkeypatch):
    for item in filter(None, os.environ.get("LN_KNOBS", "").split(",")):
        knob, value = item.split("=")
        cast = bool if isinstance(getattr(descriptor, knob), bool) else int
        monkeypatch.setattr(descriptor, knob, cast(int(value)))


IDS = [
    "focus1_inL1_resDRAM_outDRAM",
    "focus2_allDRAM_res",
    "focus3_allDRAM_nores",
    "allL1_res",
    "L1_nores",
    "8x384_allDRAM_res",
    "32x32_allDRAM_res",
    "affineTILE_DRAM",
]


@pytest.mark.parametrize("case", PERF_CASES, ids=IDS[: len(PERF_CASES)])
def test_perf_case(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


# Guard set: one representative per remaining path (DEST mode, h_non_aligned, sub-tile S, few cores).
_FOCUS1 = PERF_CASES[0]
GUARD_CASES = {
    "focus1_bf16dest": dict(_FOCUS1, fp32_dest_acc_en=False),
    "3x500_inL1_resDRAM": next(c for c in LOOSE_CASES if c["inputs"] == ((3, 1, 500, 768),)),
    "64x17_allDRAM_nores_affTILE": dict(
        _FOCUS1,
        inputs=((64, 1, 17, 768),),
        residual="no_residual",
        buffer_type=PERF_CASES[2]["buffer_type"],
        affine_layout=PERF_CASES[-1]["affine_layout"],
        extras={"output_buffer_type": PERF_CASES[2]["buffer_type"]},
    ),
    "2x512_allL1_res": dict(PERF_CASES[3], inputs=((2, 1, 512, 768),)),
}


@pytest.mark.parametrize("case", list(GUARD_CASES.values()), ids=list(GUARD_CASES))
def test_guard_case(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)
