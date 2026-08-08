# SPDX-License-Identifier: Apache-2.0
"""Band-driven exit rule (plan §3.3): opt-in, above the manual-target rule, no-op when off."""
import importlib.util
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "exit_policy_ut", str(Path(__file__).resolve().parents[1] / "agent" / "exit_policy.py"))
ep = importlib.util.module_from_spec(_S)
_S.loader.exec_module(ep)


def test_band_in_band_done():
    assert ep.check_exit({"target_band": True, "perf_status": "IN_BAND"}) == "DONE"


def test_band_below_continues():
    assert ep.check_exit({"target_band": True, "perf_status": "BELOW_BAND"}) == "continue"


def test_band_above_continues_not_stop():
    assert ep.check_exit({"target_band": True, "perf_status": "ABOVE_BAND"}) == "continue"


def test_band_unknown_falls_through_to_metric():
    # UNKNOWN + a met metric -> existing DONE path still fires
    st = {"target_band": True, "perf_status": "UNKNOWN",
          "metric": {"direction": "min", "current": 1.0, "target": 2.0}}
    assert ep.check_exit(st) == "DONE"


def test_band_off_is_byte_identical():
    # no target_band -> band rule is a no-op; behaves exactly as before
    assert ep.check_exit({"perf_status": "IN_BAND"}) == "continue"  # ignored, no metric met
    assert ep.check_exit({"metric": {"direction": "max", "current": 5, "target": 3}}) == "DONE"
    assert ep.check_exit({"max_iter": 3, "iteration": 3}) == "STOPPED"
