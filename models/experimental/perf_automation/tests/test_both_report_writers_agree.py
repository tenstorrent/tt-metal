"""One run, two writers, one answer.

The live report is rendered by the perf-mcp SERVER and the final one by the RUN itself. They read the
same files and must therefore say the same thing. They did not: the server was launched with
PERF_MCP_MODEL_ROOT in its env and the run was not, so the run resolved the model as `Path(".").name`
-- "" -- which falls through to the literal "model", and every (model, task)-keyed lookup asked for a
file no run writes.

Measured on voxtral_mini_3b_2507 (2026-09-05/06). Two reports of one run, side by side:

    live   batch: 8    ENCODE -- per request   compute binds   134.6 TFLOPS   ladder rendered
    final  batch: not reported   every stage per-token   memory binds   0.07 TFLOPS   ladder empty

and the wrong one was the final. A first fix set the variable at the point the per-pipeline root is
derived -- which runs AFTER perf_mcp is first imported, and perf_mcp freezes _MODEL_ROOT at import
across 69 read sites. So the fix changed nothing and the next run's final report was still wrong.
Verified directly: importing with no root and setting it afterwards leaves _MODEL_ROOT.name == "" and
read_stage_batch() == 0.

It is stamped at the top of the run now, before the first import. These cases hold the invariant
rather than the mechanism, because the mechanism has been wrong twice.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _own(monkeypatch, *names):
    """Make monkeypatch OWN these variables, absent or not, so teardown really removes them.

    delenv(raising=False) does not record a key that was already absent -- pytest has nothing to
    restore, so it tracks nothing. Importing perf_mcp then setdefaults PERF_MCP_MODEL_NAME into the
    real environment and it survives the test, which is how this file made
    test_stage_ms_belongs_to_a_run fail two files later. setenv first, then delenv, and the key is
    tracked either way.
    """
    for _n in names:
        monkeypatch.setenv(_n, "_owned_by_test")
        monkeypatch.delenv(_n, raising=False)


def _state(tmp_path):
    """A run's state dir, as the server would have written it."""
    root = tmp_path / "some_model"
    root.mkdir(exist_ok=True)
    (tmp_path / "perf_mcp_stage_ms_some_model_main.json").write_text(
        json.dumps({"run": "a-run", "stages": {"s": 1.0}, "batch": 8})
    )
    return root


def _load_perf_mcp(tag):
    spec = importlib.util.spec_from_file_location("pmcp_two_writers_%s" % tag, str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _writer(monkeypatch, tmp_path, tag, *, stamp_first: bool):
    """One writer's process: the run stamps the root first, the server is handed it in its env."""
    root = _state(tmp_path)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    _own(monkeypatch, "PERF_MCP_MODEL_ROOT", "PERF_MCP_MODEL_NAME", "PERF_MCP_RUN_ID")
    if stamp_first:
        import run as _run

        _run._stamp_model_root(root)
    else:
        monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    return _load_perf_mcp(tag)


def test_the_two_writers_read_the_same_batch(monkeypatch, tmp_path):
    """The number that decides every stage's unit and every per-item ceiling."""
    server = _writer(monkeypatch, tmp_path, "server", stamp_first=False)
    run = _writer(monkeypatch, tmp_path, "run", stamp_first=True)
    assert server.read_stage_batch() == run.read_stage_batch() == 8


def test_the_two_writers_read_the_same_stage_timings(monkeypatch, tmp_path):
    server = _writer(monkeypatch, tmp_path, "server2", stamp_first=False)
    run = _writer(monkeypatch, tmp_path, "run2", stamp_first=True)
    assert server.read_stage_ms() == run.read_stage_ms() == {"s": 1.0}


def test_the_run_resolves_the_model_before_anything_reads_it(monkeypatch, tmp_path):
    """Setting it after the import cannot reach a value frozen at import -- that was the first fix."""
    m = _writer(monkeypatch, tmp_path, "run3", stamp_first=True)
    assert m._MODEL_ROOT.name == "some_model"


def test_a_root_that_arrives_after_the_import_is_still_picked_up(monkeypatch, tmp_path):
    """BEHAVIOUR CHANGE (2026-09-06), deliberate. This case pinned the old failure -- a late root was
    ignored because _MODEL_ROOT is frozen at import and eleven paths read it directly. Those paths now
    resolve the key at CALL time, so the module recovers instead of naming a file no run writes. The
    frozen attribute is still frozen; nothing reads it for a path any more."""
    root = _state(tmp_path)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    _own(monkeypatch, "PERF_MCP_MODEL_ROOT", "PERF_MCP_MODEL_NAME")
    m = _load_perf_mcp("late")
    assert m._MODEL_ROOT.name == "" and m.read_stage_batch() == 0
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    assert m._MODEL_ROOT.name == "", "the import-time attribute is still frozen, by design"
    assert m.read_stage_batch() == 8, "but the PATH is resolved at call time and recovers"


def test_an_operator_who_named_the_root_outranks_the_inference(monkeypatch, tmp_path):
    """setdefault, not assignment: a supervising process that already stated it must win."""
    import run as _run

    other = tmp_path / "stated_elsewhere"
    other.mkdir()
    _own(monkeypatch, "PERF_MCP_MODEL_NAME")
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(other))
    _run._stamp_model_root(tmp_path / "inferred")
    import os as _os

    assert _os.environ["PERF_MCP_MODEL_ROOT"] == str(other)


def test_an_unusable_path_states_nothing(monkeypatch):
    import run as _run

    _own(monkeypatch, "PERF_MCP_MODEL_ROOT", "PERF_MCP_MODEL_NAME")
    assert _run._stamp_model_root(None) == ""


def test_the_name_is_kept_beside_the_root_that_produced_it(monkeypatch, tmp_path):
    """Two keys for one fact drift; perf_mcp exports the same pair at import for this reason."""
    import os as _os

    import run as _run

    _own(monkeypatch, "PERF_MCP_MODEL_ROOT", "PERF_MCP_MODEL_NAME")
    root = tmp_path / "named_model"
    root.mkdir()
    _run._stamp_model_root(root)
    assert _os.environ["PERF_MCP_MODEL_NAME"] == "named_model"
    assert Path(_os.environ["PERF_MCP_MODEL_ROOT"]).name == _os.environ["PERF_MCP_MODEL_NAME"]


def test_no_model_name_is_typed_into_the_stamp():
    src = (_CC / "run.py").read_text(encoding="utf-8")
    i = src.index("def _stamp_model_root")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    code = code[code.index('"""', code.index('"""') + 3) + 3 :]
    for typed in ("voxtral", "gemma", "llama", "nemotron"):
        assert typed not in code.lower(), typed
