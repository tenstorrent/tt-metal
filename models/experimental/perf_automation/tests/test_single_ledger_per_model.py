# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One model, one run, ONE ledger.

The gemma-3-12b-it run of 2026-07-31 wrote to two:

    perf_measurements_gemma3_main.jsonl   active_bytes, modeled_floor, fullpipe_e2e
    perf_measurements_model_main.jsonl    eager_per_op 240.8588  <- source: before_loop

`model_main` is the FALLBACK name ledger_path() invents when nobody passes a key. Two writers land
there:

  * run.py's fullpipe bookend calls first()/record() with NO model= at all
  * before_loop's eager anchor reads PERF_MCP_MODEL_NAME, which run.py does not set until AFTER
    discover() has already taken the baseline -- so at anchor time it is empty

The damage is not a stray file. The BEFORE/AFTER decision is `first(...) -> is there a before yet?`,
and it was being asked per FILE: the genuine fullpipe before went to model_main, so when the
committed-best 40.13 ms was later recorded against gemma3_main it found no before there and CLAIMED
the before slot. The report then printed

    trace+1CQ full-pipeline e2e (all layers):  40.13 ms  ->  (after not measured yet)

which reads as "the model runs at 40 ms BEFORE optimization" -- when 40.13 IS the optimized result.
Same for the eager line: 240.86 ms was measured and written, just to the other file, so the report
said "not measured (no ledger reading for this run)".

Two fixes, both here:
  1. ledger_path() falls back to PERF_MCP_MODEL_ROOT's basename before the literal "model", so a
     caller that runs before PERF_MCP_MODEL_NAME is set still keys correctly.
  2. an UNKEYED call is loud: PERF_MCP_STRICT_LEDGER_KEY=1 makes it raise, so a new call site that
     forgets the key is caught in test rather than discovered in a report months later.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _led():
    spec = importlib.util.spec_from_file_location("meas_single_ledger", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _clean(monkeypatch):
    for k in (
        "PERF_MCP_LEDGER",
        "PERF_MCP_MODEL_NAME",
        "PERF_MCP_MODEL_ROOT",
        "PERF_MCP_TASK",
        "PERF_MCP_STRICT_LEDGER_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- MODEL_ROOT fallback
def test_model_root_keys_the_ledger_when_name_is_unset(monkeypatch):
    """THE gemma3 bug: before_loop runs before run.py sets PERF_MCP_MODEL_NAME, but MODEL_ROOT is
    already there. Without this the anchor lands in model_main.jsonl."""
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/home/ttuser/tt-metal-gemma3/models/demos/multimodal/gemma3")
    p = led.ledger_path()
    assert "gemma3" in p.name, f"MODEL_ROOT did not key the ledger: {p.name}"
    assert p.name != "perf_measurements_model_main.jsonl"


def test_explicit_model_still_wins(monkeypatch):
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/somewhere/else")
    assert "gemma3" in led.ledger_path("gemma3", "main").name


def test_model_name_beats_model_root(monkeypatch):
    """NAME is the more specific signal; ROOT is only the fallback for callers that run early."""
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "chosen")
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/models/demos/other")
    assert "chosen" in led.ledger_path().name


# --------------------------------------------------------------------------- strict mode
def test_strict_mode_rejects_a_fully_unkeyed_call(monkeypatch):
    """A missing key must be loud. Silently inventing model_main.jsonl is what let two writers
    disagree for a whole run without anyone noticing."""
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", "1")
    with pytest.raises(ValueError):  # allow-pytest.raises: repo-root conftest bypassed
        led.ledger_path()


def test_strict_mode_allows_any_real_key(monkeypatch):
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", "1")
    assert led.ledger_path("gemma3", "main")
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/models/demos/multimodal/gemma3")
    assert led.ledger_path()


def test_strict_mode_is_off_by_default(monkeypatch):
    """Production keeps degrading rather than crashing a long run over a ledger name."""
    led = _led()
    _clean(monkeypatch)
    assert led.ledger_path()


def test_explicit_ledger_override_bypasses_everything(monkeypatch, tmp_path):
    led = _led()
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", "1")
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "x.jsonl"))
    assert led.ledger_path() == tmp_path / "x.jsonl"


# --------------------------------------------------------------------------- the run, replayed
def test_the_gemma3_run_would_now_use_one_ledger(monkeypatch):
    """Replay the two writers that disagreed, in the order the run ran them."""
    led = _led()
    _clean(monkeypatch)
    root = "/home/ttuser/tt-metal-gemma3/models/demos/multimodal/gemma3"
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", root)

    # 1. before_loop's eager anchor -- PERF_MCP_MODEL_NAME not set yet
    early = led.ledger_path()
    # 2. perf_mcp, after run.py has set the name
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "gemma3")
    late = led.ledger_path()

    assert early == late, f"one run, two ledgers: {early.name} vs {late.name}"


def test_run_py_fullpipe_writer_passes_a_key():
    """run.py's fullpipe bookend called first()/record() with no model= at all -- the other half of
    the split. Wiring check: it must key like every other writer."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("KIND_FULLPIPE, led.PHASE_BEFORE")
    call = src[i - 200 : i + 600]
    assert "model=" in call, "run.py's fullpipe bookend still reads/writes an unkeyed ledger"
