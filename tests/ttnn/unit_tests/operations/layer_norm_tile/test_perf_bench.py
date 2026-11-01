# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench: one call per LOOSE_CASES perf row (run under scripts/run_safe_pytest.sh --profile)."""

import pytest

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

PERF_CASES = [case for case in LOOSE_CASES if "achievable_ns" in case.get("extras", {})]
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
