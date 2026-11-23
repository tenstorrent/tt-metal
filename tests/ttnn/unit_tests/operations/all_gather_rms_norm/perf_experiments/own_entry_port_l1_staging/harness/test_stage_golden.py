# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT own_entry_port_l1_staging: the golden suite's cells (GOLDEN_TIER selects the subset) run on the staging
copy of the op (correctness only). Knobs: AGRN_STAGE_* (agrn_stage/all_gather_rms_norm_program_descriptor.py)."""

import os
import re

import pytest

from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases, _parametrize_cases
from ttnn.operations.all_gather_rms_norm.perf_experiments.own_entry_port_l1_staging.agrn_stage import (
    all_gather_rms_norm as stage_op,
)


def _cells():
    """AGRN_CELLS: regexes (';'-separated, all must match) on the cell id."""
    pats = [re.compile(p) for p in os.environ.get("AGRN_CELLS", "").split(";") if p]
    return [c for c in _parametrize_cases() if all(p.search(c.id) for p in pats)]


@pytest.mark.parametrize("inputs,axes", _cells())
def test_stage_op(inputs, axes, mesh_device):
    call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh_device, op=stage_op, **axes)
    check(call())


@pytest.mark.parametrize("inputs,axes,extras", _loose_cases())
def test_stage_op_loose(inputs, axes, extras, mesh_device):
    call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh_device, extras=extras, op=stage_op, **axes)
    check(call())
