# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 production cases the acceptance test leaves out, picked for the kernel branches they
reach: new_storage row and column patches with a remainder >= 16 (case 69), row-only patches
(cases 68, 71), and extension band segments that start in faces 2/3 with a non-zero c.
Same checks as test_g6_pad.py (poisoned input padding, bit-exact, one program, storage)."""

import pytest
import torch

from .test_g6_pad import is_extension, run_case

CASES = [
    ((1, 185, 128), ((0, 71), (0, 0)), 0),  # case 68
    ((1, 1, 185, 185), ((0, 71), (0, 71)), -1.7014118346046923e38),  # case 69: remainders 25, 25
    ((1, 207, 128), ((0, 49), (0, 0)), 0),  # case 71
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.1),  # case 247 shape, rows 4..31, inexact c
    ((1, 1, 20, 62080), ((0, 0), (0, 0), (0, 7), (0, 0)), -3.5),  # rows 20..26: faces 2/3 only
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{'x'.join(map(str, c[0]))}-v{c[2]}")
def test_g6_pad_extra_cases(device, case):
    torch.manual_seed(42)
    shape, padding, value = case
    run_case(device, shape, padding, value, poison=1000.0, untouched_rows=is_extension(shape, padding))
