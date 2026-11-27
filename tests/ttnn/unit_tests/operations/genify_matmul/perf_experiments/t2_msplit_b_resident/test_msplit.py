# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shim: the t2_msplit_b_resident bench lives in ttnn/ttnn/operations/genify_matmul/perf_experiments/
t2_msplit_b_resident/bench_msplit.py (collected from here so the ttnn package is not imported twice)."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[7] / (
    "ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_msplit_b_resident/bench_msplit.py"
)
_spec = importlib.util.spec_from_file_location("ms_bench", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
pytestmark = _m.pytestmark
test_msplit = _m.test_msplit
