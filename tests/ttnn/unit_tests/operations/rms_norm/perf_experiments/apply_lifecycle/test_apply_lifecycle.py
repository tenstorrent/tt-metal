# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the apply-lifecycle bake-off (idea I11).

Run under the device profiler:
    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/apply_lifecycle/test_apply_lifecycle.py

Correctness is the only assertion; every timing is printed, never asserted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bakeoff import OPTIONS, main  # noqa: E402

SHAPES = [(1, 3), (32, 4), (16, 4), (1, 112)]

# Sweep set overridable from the environment so a follow-up run can narrow.
_ENV_SHAPES = os.environ.get("APPLY_LIFECYCLE_SHAPES")
if _ENV_SHAPES:
    SHAPES = [tuple(int(v) for v in s.split("x")) for s in _ENV_SHAPES.split(",")]
_ENV_OPTS = os.environ.get("APPLY_LIFECYCLE_OPTIONS")
OPTS = _ENV_OPTS.split(",") if _ENV_OPTS else list(OPTIONS)


def test_apply_lifecycle_bakeoff(device):
    results = main(device, SHAPES, OPTS, iters=(1, 21))
    broken = [k for k, v in results.items() if not v[2]]
    assert not broken, f"options failed the correctness gate: {broken}"
