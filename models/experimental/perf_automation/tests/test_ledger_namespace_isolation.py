# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No test may write into the shared ledger namespace.

The ledger is durable by design and never truncated, so anything a test leaves in the real temp dir
outlives it. The autouse fixture redirected PERF_MCP_LEDGER, which names ONE file -- and the tests
that deliberately delenv it to exercise unkeyed behaviour then made KEYED calls, which resolved
straight back to the shared temp dir.

That is not hypothetical. Reviewing a stopped gemma-3-12b run, /tmp held:

    perf_measurements_gemma3_main.jsonl        the run's real ledger
    perf_measurements_named_model_main.jsonl   modeled_floor 537.23, depth=16, source=""

and the second was read as the run having split its anchors across two files -- a bug that had been
fixed. It was written by test_floor_anchor_writeonce.py:106, whose delenv is correct and whose keyed
call had nowhere safe to land. Wrong diagnosis of a live run, caused by test residue.

PERF_MCP_LEDGER_DIR redirects the whole namespace, so a delenv of the file variable cannot re-expose
the shared directory.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _meas():
    spec = importlib.util.spec_from_file_location("meas_ns", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_keyed_path_honours_the_dir_override(tmp_path, monkeypatch):
    led = _meas()
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    p = led.ledger_path(model="gemma3", task="main")
    assert p.parent == tmp_path
    assert p.name == "perf_measurements_gemma3_main.jsonl"


def test_the_exact_leak_stays_inside_the_box(tmp_path, monkeypatch):
    """THE case: delenv the file var (as the unkeyed tests must), then make a keyed write."""
    led = _meas()
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_NAME", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    assert led.anchor(led.KIND_FLOOR, 537.23, depth="16", model="named_model") == 537.23
    written = list(tmp_path.glob("perf_measurements_*.jsonl"))
    assert [p.name for p in written] == ["perf_measurements_named_model_main.jsonl"]
    assert not (Path(tempfile.gettempdir()) / "perf_measurements_named_model_main.jsonl").exists() or (
        os.environ.get("PERF_MCP_LEDGER_DIR") == tempfile.gettempdir()
    ), "the keyed write escaped into the shared temp dir"


def test_without_the_override_it_still_uses_the_temp_dir(tmp_path, monkeypatch):
    """Production behaviour is unchanged when the variable is unset."""
    led = _meas()
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    assert led.ledger_path(model="m", task="t").parent == Path(tempfile.gettempdir())


def test_the_autouse_fixture_sets_both(tmp_path):
    """Whatever a test does to PERF_MCP_LEDGER, the directory redirect is already in place."""
    box = os.environ.get("PERF_MCP_LEDGER_DIR")
    assert box, "the autouse fixture did not redirect the ledger namespace"
    assert Path(box) != Path(tempfile.gettempdir())


def test_file_override_still_wins_when_present(tmp_path, monkeypatch):
    """PERF_MCP_LEDGER remains the most specific override; the dir does not displace it."""
    led = _meas()
    explicit = tmp_path / "explicit.jsonl"
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("PERF_MCP_LEDGER", str(explicit))
    assert led.ledger_path(model="gemma3", task="main") == explicit
