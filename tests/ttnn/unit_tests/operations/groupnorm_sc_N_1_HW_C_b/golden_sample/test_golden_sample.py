# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-seed sample of test_golden.py's cases.

The full golden grid (~47k cases) takes days on one card, too long for a gate that reruns after every change. This
file runs a reproducible 300-case sample of test_op plus every test_op_loose case. Coverage of every SUPPORTED axis
value and of the extreme shapes is the job of the golden cut in the op's unit tests, not of this sample.
"""

from __future__ import annotations

import random

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.feature_spec import (
    INPUTS,
    INVALID,
    LOOSE_CASES,
    TARGET,
)
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.test_golden import shape_id
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import (  # type: ignore
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
)

pytestmark = pytest.mark.use_module_device

SAMPLE_SIZE = 300
SEED = 0

_ALL = parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID, shape_id=shape_id)
_SAMPLE = sorted(random.Random(SEED).sample(range(len(_ALL)), min(SAMPLE_SIZE, len(_ALL))))


@pytest.mark.parametrize("inputs,axes", [_ALL[i] for i in _SAMPLE])
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
        shape_id=shape_id,
    ),
)
def test_op_loose(inputs, axes, extras, device):
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, extras=extras, **axes)
