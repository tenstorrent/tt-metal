# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 production cases the acceptance test leaves out, picked for the kernel branches they
reach: N row and column masks with a boundary remainder >= 16 (case 69), N row-only masks
(cases 68, 71), and E segments starting in faces 2/3 with a non-zero c (stored, not NoC-zeroed).
Same checks as test_g6_pad.py (poisoned input padding, bit-exact, one program, storage)."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import g6_pad

from .test_g6_pad import _check_call, _make_input, _raw_rows, call_one_program

CASES = [
    ((1, 185, 128), ((0, 71), (0, 0)), 0),  # case 68
    ((1, 1, 185, 185), ((0, 71), (0, 71)), -1.7014118346046923e38),  # case 69: remainders 25, 25
    ((1, 207, 128), ((0, 49), (0, 0)), 0),  # case 71
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.1),  # case 247 shape, rows 4..31, stored c
    ((1, 1, 20, 62080), ((0, 0), (0, 0), (0, 7), (0, 0)), -3.5),  # rows 20..26: faces 2/3 only
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{'x'.join(map(str, c[0]))}-v{c[2]}")
def test_g6_pad_extra_cases(device, case):
    shape, padding, value = case
    torch_input, tt_input = _make_input(shape, device, poison=1000.0)
    raw_before = _raw_rows(tt_input).clone()
    tt_output = call_one_program(lambda: g6_pad(tt_input, padding, value=value))
    _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before)
