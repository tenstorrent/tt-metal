# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ablation-peel subset of test_perf_bench (flagged rows + 8x384 + 32x32); one call each under --profile."""

import pytest

from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import IDS, PERF_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

PEEL = {name: case for name, case in zip(IDS, PERF_CASES) if name.startswith(("focus", "8x384", "32x32"))}


@pytest.mark.parametrize("case", list(PEEL.values()), ids=list(PEEL))
def test_peel_case(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


def test_repeat(device):
    """One row N times in one process (LN_ROW=<id prefix>, LN_REPEAT=N) for the in-process spread."""
    import os

    import ttnn

    row = os.environ.get("LN_ROW", "focus2")
    case = dict(next(c for name, c in PEEL.items() if name.startswith(row)))
    inputs = case.pop("inputs")
    for _ in range(int(os.environ.get("LN_REPEAT", "8"))):
        run_layer_norm(inputs, device=device, **case)
        ttnn.ReadDeviceProfiler(device)
