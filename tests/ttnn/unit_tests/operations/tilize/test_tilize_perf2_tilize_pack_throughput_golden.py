# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-run the tilize golden suite (test_op / test_op_loose) against a perf_experiments kernel dir.

TPT_GOLDEN_VARIANT=tilize_pack_throughput/kernels_graduate   (unset = the op's own kernels/)
Opt-in: TILIZE_PERF_EXPERIMENTS=1. The op's files are not touched: KERNEL_DIR is monkeypatched.
"""
import os
from pathlib import Path

import pytest

import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.golden_tests.tilize.test_golden import test_op, test_op_loose  # noqa: F401  (re-collected here)

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments"


@pytest.fixture(autouse=True)
def _variant_kernels(monkeypatch):
    v = os.environ.get("TPT_GOLDEN_VARIANT")
    if v:
        monkeypatch.setattr(pd, "KERNEL_DIR", EXP / v)
