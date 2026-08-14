# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A device run is budgeted from runs of ITS OWN size, not from whatever shares its label.

Every device subprocess filed its duration under `observe_op="profile"`, and adaptive_timer budgets
an op at 6x the p95 of that op's own history. So a 25 s capped probe and a 2144 s uncapped full-model
measurement shared one bucket.

Measured on Voxtral-Mini-3B, 2026-08-14 -- the bucket held

    [25.0, 55.0, 85.1, 100.2, 150.5, 281.1, 1734.2, 2143.9]
     └────────── capped, 2 layers ──────────┘  └─ uncapped ─┘

Before the last two existed the p95 was 281.1, so the budget was 6 x 281.1 = 1686 s. The uncapped run
needs 1734 s and was killed about fifty seconds short, printing "likely a device wedge / leaked mesh"
while also reporting "killed holders none" -- it was neither wedged nor over a fixed wall, just
budgeted from a run six times smaller. The pollution runs both ways: with 2143.9 in the bucket, the
next capped probe is budgeted ~12800 s, so a genuinely hung 25 s probe would sit for three hours.

The distinguishing fact was already in hand. A capped run carries TT_PERF_LAYERS in the environment
being launched; an uncapped one expresses "all layers" by its ABSENCE.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("_run_timing", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_capped_run_is_timed_as_a_profile():
    run = _run()
    assert run.timed_op_for({"TT_PERF_LAYERS": "2"}) == "profile"


def test_an_uncapped_run_gets_its_own_bucket():
    """ALL LAYERS is expressed by ABSENCE -- never by a sentinel, because "0" arrives truthy."""
    run = _run()
    assert run.timed_op_for({}) == "fullpipe"
    assert run.timed_op_for({"TT_PERF_OSL_TOKENS": "1"}) == "fullpipe"


def test_a_non_numeric_cap_is_not_a_cap():
    """The dict-into-TT_PERF_LAYERS defect wrote "{'stack3': 2}", which every model reads as no cap."""
    run = _run()
    assert run.timed_op_for({"TT_PERF_LAYERS": "{'stack3': 2, 'stack2': 2}"}) == "fullpipe"
    assert run.timed_op_for({"TT_PERF_LAYERS": ""}) == "fullpipe"


def test_the_callers_label_cannot_override_the_run():
    """The bug was a typed string. A caller naming "profile" for an uncapped run must not win."""
    run = _run()
    assert run.timed_op_for({}, "profile") == "fullpipe"


def test_the_full_model_measurement_budgets_from_its_own_bucket():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index('f"full-pipeline ({label})"')
    window = src[max(0, i - 900) : i + 400]
    assert "_fp_op = timed_op_for(env" in window, "the full-model run still takes a hardcoded op"
    assert "adaptive_timer(repo_root, _fp_op" in window, "its budget is not drawn from its own bucket"
    assert 'observe_op="profile"' not in window, "it still files its duration under the capped bucket"


def test_the_new_bucket_has_a_cold_start():
    """With no history adaptive_timer falls back to 600 s -- far under a 2144 s run, so the FIRST
    uncapped run on any model would die exactly as this one did and teach the bucket nothing."""
    run = _run()
    assert run._OP_IN_BASE_UNITS.get("fullpipe", 0) >= 7.0, run._OP_IN_BASE_UNITS
    assert "fullpipe" in run._OP_MULT


def test_the_runner_routes_by_environment_not_by_argument():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("if observe_op and observe_root is not None:")
    assert "observe_op = timed_op_for(env, observe_op)" in src[i : i + 300], "the runner trusts the caller's label"
