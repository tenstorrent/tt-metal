# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Auto-parameterized golden tests for groupnorm_sc_N_1_HW_C (registry model).

Both parametrize loops live in eval/golden_harness.py — this file is
just per-op wiring: imports + the two `@pytest.mark.parametrize`
decorators around the op's run_groupnorm_sc_N_1_HW_C call.

INPUTS entries are `(shape, num_groups)` flat tuples. The op file's
INPUT_TAGGERS includes tag_num_groups (reads inputs[1]) and tag_alignment.
"""

from __future__ import annotations

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.groupnorm_sc_N_1_HW_C.feature_spec import (
    INPUTS,
    INVALID,
    LOOSE_CASES,
    TARGET,
)
from eval.golden_tests.groupnorm_sc_N_1_HW_C.helpers import run_groupnorm_sc_N_1_HW_C
from eval.golden_tests.groupnorm_sc_N_1_HW_C.registry import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
)

# 7000+ test invocations × ~50 ms fixture overhead is the dominant cost
# in the function-scoped baseline. See .claude/eval/profiling/.
pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "inputs,axes",
    parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID),
)
def test_op(inputs, axes, device):
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    parametrize_loose_cases(
        LOOSE_CASES,
        INPUT_TAGGERS,
        SUPPORTED,
        EXCLUSIONS,
        INVALID,
    ),
)
def test_op_loose(inputs, axes, extras, device):
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, extras=extras, **axes)
