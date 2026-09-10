"""RED tests for BUG 4 of PERF_AUTOMATION_FIXES_PLAN.md — adaptive timers.

Measured on 2026-07-25: all four adaptive paths compute min(ceil, max(floor, 3*base))
with base = the tracy BASELINE PROFILE duration, so `3*base` loses to the absolute
floors (2400 / 3600) for every model whose baseline profile is under ~800 s. Result:
adaptivity is inert for everything actually run, and the floors are the de-facto policy.

Two failure directions, both reproduced:
  small models  -> 3600 s budget for 6 s of work (600x), so a hang idles for an hour
  big models    -> llama's round needed >2400 s and was killed 4x on 2026-07-25

Hermetic: synthetic manifest/events fixtures, no device.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = "models/experimental/perf_automation"


def _load_run():
    spec = importlib.util.spec_from_file_location("ccrun_under_test", _ROOT / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(root: Path, base_s: float, timeout: int = 10800) -> Path:
    run = root / PERF_DIR / "runs" / "2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"config": {"timeout": timeout}}))
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": base_s}) + "\n"
    )
    return run / "manifest.json"


@pytest.fixture(autouse=True)
def _no_env_leak():
    """Timer helpers read env at call time; never leak a stale manifest path to other tests."""
    keys = (
        "PERF_MCP_MANIFEST",
        "PERF_MCP_MEASURE_BACKSTOP",
        "PERF_MCP_ROUND_MAX_SEC",
        "PERF_MCP_ROUND_STALL_SEC",
    )
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# observed anchors: ACE-Step modules 3.16-42.6 s, llama 8B full pipeline 146.7 s
SMALL = [("micro", 0.8, 6), ("tiny_ACE_encoder_layer", 3.16, 13), ("mid_ACE_dit_layer", 13.06, 52)]
BIG = [("llama_8B_full_pipeline", 146.72, 1400)]


@pytest.mark.parametrize("name,base,work", SMALL)
def test_small_model_budget_is_proportional_not_an_hour(tmp_path, name, base, work):
    """A 3 s module must not be granted a 3600 s budget: a real hang would idle for an
    hour before detection. ACE survived only because the FROZEN progress detector fired."""
    m = _load_run()
    os.environ["PERF_MCP_MANIFEST"] = str(_fixture(tmp_path / name, base))
    os.environ.pop("PERF_MCP_MEASURE_BACKSTOP", None)
    os.environ.pop("PERF_MCP_ROUND_MAX_SEC", None)
    root = tmp_path / name
    cap = m._round_hard_cap(root, 600)
    assert cap <= 40 * work, (
        f"{name}: round cap {cap}s is {cap / work:.0f}x the real work ({work}s) — "
        "a hang here goes undetected for far too long"
    )


@pytest.mark.parametrize("name,base,work", BIG)
def test_big_model_round_cap_exceeds_its_real_cycle(tmp_path, name, base, work):
    """llama's check_pcc alone runs ~1400 s (full 8B demo). A round must be allowed to
    complete edit -> check_pcc -> measure -> commit; on 2026-07-25 it was killed 4x at
    the 2400 s floor with `killed holders none` and nothing wrong."""
    m = _load_run()
    os.environ["PERF_MCP_MANIFEST"] = str(_fixture(tmp_path / name, base))
    os.environ.pop("PERF_MCP_MEASURE_BACKSTOP", None)
    os.environ.pop("PERF_MCP_ROUND_MAX_SEC", None)
    root = tmp_path / name
    cap = m._round_hard_cap(root, 600)
    assert cap > 2 * work, (
        f"{name}: round cap {cap}s cannot fit one cycle whose gate alone is {work}s — "
        "this is the 4x premature kill observed on 2026-07-25"
    )


def test_adaptivity_actually_engages_below_800s_baseline(tmp_path):
    """`3*base` must not be dominated by an absolute floor for ordinary models: two
    different baselines must yield two different budgets."""
    m = _load_run()
    caps = []
    for base in (3.16, 146.72):
        os.environ["PERF_MCP_MANIFEST"] = str(_fixture(tmp_path / f"b{base}", base))
        os.environ.pop("PERF_MCP_ROUND_MAX_SEC", None)
        caps.append(m._round_hard_cap(tmp_path / f"b{base}", 600))
    assert caps[0] != caps[1], (
        f"a 3 s module and an 8B pipeline both got {caps[0]}s — the floor is the policy, " "adaptivity is inert"
    )


def test_operator_override_still_pins_and_is_visible(tmp_path):
    """An explicit override must win (it is a budget decision), but it disables adaptivity
    and must therefore be observable rather than silent."""
    m = _load_run()
    os.environ["PERF_MCP_MANIFEST"] = str(_fixture(tmp_path / "ov", 146.72))
    os.environ["PERF_MCP_ROUND_MAX_SEC"] = "999"
    assert m._round_hard_cap(tmp_path / "ov", 600) == 999
    assert hasattr(m, "_timer_overrides_active"), "no way to report that adaptivity was pinned off"
    assert "PERF_MCP_ROUND_MAX_SEC" in m._timer_overrides_active()
