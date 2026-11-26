# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block-knob sweep for gen_qkv_causal_conv1d_silu (perf measurement only — DO NOT DELETE).

Run with:  scripts/run_safe_pytest.sh --profile --no-py-zones --run-all <this file>
One GenericOpDeviceOperation row per case, in parametrize order. Every variant is checked for
correctness (the knobs never change the result). The PERF FOCUS targets 001..004 at the default
plan come first; 004 / 020 are also run at the L1-bound block_rows (the design's Overlap lamp).
"""

import importlib

import pytest

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")

KNOBS = [
    pytest.param(0, None, id="001-default"),
    pytest.param(1, None, id="002-default"),
    pytest.param(2, None, id="003-default"),
    pytest.param(3, None, id="004-default"),
    pytest.param(3, 3, id="004-block_rows3_l1_bound"),
    pytest.param(14, None, id="020-default"),
    pytest.param(14, 5, id="020-block_rows5_l1_bound"),
    pytest.param(4, None, id="005-tile-default"),
]


@pytest.mark.parametrize("case_index,max_block_rows", KNOBS)
def test_knob(device, case_index, max_block_rows):
    case = PHASE0_CASES[case_index].values[0]
    with op_module.plan_limits(max_block_rows=max_block_rows):
        outputs, host, device_inputs = _run(device, case)
        plan = op_module.get_last_plan()
    print(f"PLAN {case_index} {max_block_rows}: {plan}")
    _check_case(case, outputs, host, device_inputs)
