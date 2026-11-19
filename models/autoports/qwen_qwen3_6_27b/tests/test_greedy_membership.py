# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The numerical probe accepts exact tied maxima and rejects lower candidates."""

import ast
from pathlib import Path

import torch


def test_exact_tie_membership_has_no_tolerance():
    path = Path(__file__).with_name("probe_tile_continuation.py")
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "greedy_membership"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    reference = torch.tensor([[6.03125, 6.03125], [6.03125, 6.0], [6.03125, 6.03124]])
    actual = torch.tensor([[6.0, 6.0625]]).repeat(3, 1)
    assert scope["greedy_membership"](reference, actual).tolist() == [True, False, False]
