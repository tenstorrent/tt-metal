"""The agent times a candidate before it checks its correctness, and a win still needs both.

The end-to-end correctness run is the longest step of an attempt (~25 min on a 50-step diffusion
model), and every rung told the agent to run it FIRST -- so a candidate that then measured slower had
already paid for it. The prompt now runs it last, only for a candidate the timing calls faster. The
banking gate is unchanged: a commit still needs check_pcc ok.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("cc_run_order_ut", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module", params=["_PROMPT", "_HITL_PROMPT"])
def prompt(run, request):
    return getattr(run, request.param).format(model="M", task="T", metric="device_ms")


def test_no_rung_runs_correctness_first(prompt):
    for first in ("check_pcc; measure_candidate", "check_pcc + measure_candidate", "check_pcc; measure the"):
        assert first not in prompt, first


def test_the_order_is_stated_once_and_keeps_correctness_mandatory(run):
    p = run._PROMPT.format(model="M", task="T", metric="device_ms")
    assert p.count("ORDER (time before correctness)") == 1
    assert "every commit still requires check_pcc ok" in p
    assert "IRON RULE: a real win = check_pcc ok AND check_full_pipeline_latency" in p


def test_every_rung_still_names_the_correctness_check(run):
    src = (_PA / "cc_optimize" / "run.py").read_text()
    rungs = [ln for ln in src.splitlines() if ln.strip().startswith("knob:") or "-> author a" in ln]
    assert rungs
    for ln in rungs:
        assert "check_pcc only if" in ln, ln.strip()[:60]


def test_banking_still_refuses_without_an_ok_correctness_verdict():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def gates_allow_banking(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert 'return False, "check_pcc has not run since the last commit"' in body
    assert 'if str(pcc.get("status")) != "ok":' in body
