# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pytest entry point for the `fused_sumsq` isolated bake-off.

The experiment itself lives entirely in
`ttnn/ttnn/operations/rms_norm/perf_experiments/fused_sumsq/` — this file only
exists because a test module placed under `ttnn/ttnn/...` gets a
`ttnn.ttnn.<...>` module name under pytest's importlib mode, which re-execs
`ttnn/__init__.py` and double-registers every C++ op.  Nothing but the collection
hook belongs here.

    scripts/run_safe_pytest.sh --run-all <this file> -k correctness -s
    scripts/run_safe_pytest.sh --run-all <this file> -k bias -s
    scripts/run_safe_pytest.sh --profile  <this file> -k perf -s
"""

import importlib.util
from pathlib import Path

_LAB = (
    Path(__file__).resolve().parents[5]
    / "ttnn"
    / "ttnn"
    / "operations"
    / "rms_norm"
    / "perf_experiments"
    / "fused_sumsq"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_fused_sumsq_{name}", _LAB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_cases = _load("fs_cases")

test_correctness = _cases.test_correctness
test_forced_b_correctness = _cases.test_forced_b_correctness
test_bias = _cases.test_bias
test_default_corner_correctness = _cases.test_default_corner_correctness
test_perf = _cases.test_perf
test_l1_accounting = _cases.test_l1_accounting
