"""A per-run path must name the model this process is working on, not the one it knew at import.

Eleven paths spelled `_MODEL_ROOT.name if _MODEL_ROOT else "model"`. _MODEL_ROOT is resolved at
IMPORT from PERF_MCP_MODEL_ROOT, so a process that learns the root afterwards keeps "" for its whole
life, and "" falls through to the literal "model" -- naming a file no run writes.

Three bugs from this one expression, none theoretical:

  - the final report read a stage doc that was never written under that name, and printed a roofline
    with no batch, per-token labels on every stage and an empty fidelity ladder
  - the round gate wrote 185 refusals to perf_mcp_round_finish_model_main.json while the loop read
    the model's own file, last touched two days earlier by a different run, and reported five clean
    finishes on a run that finished nothing
  - a matmul sweep's entire pre-pass was invisible, for the same reason, weeks earlier

The import-time export was part of it: it published the literal "model" into PERF_MCP_MODEL_NAME,
which model_key prefers, so a blind import poisoned every later reader in that process. It now
exports only a real name -- an unset variable lets the root be consulted, a set one never does.
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

_SRC = (_CC / "perf_mcp.py").read_text(encoding="utf-8")


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


def _code_only(src: str) -> str:
    """Source with comments AND docstrings removed.

    Stripping only comments is not enough: the paragraph explaining why an expression is wrong
    contains that expression, so the check failed on its own prose. Twice now, in two files.
    """
    import re

    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    return "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))


def _blind(monkeypatch, tmp_path, tag):
    """A module imported BEFORE the root is known -- the state every failing process was in."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    _own(monkeypatch, "PERF_MCP_MODEL_ROOT", "PERF_MCP_MODEL_NAME", "PERF_MCP_RUN_ID")
    spec = importlib.util.spec_from_file_location("pmcp_calltime_%s" % tag, str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_path_picks_up_a_root_that_arrives_after_the_import(monkeypatch, tmp_path):
    """The whole defect: the root arrives late and every path kept the import-time answer."""
    m = _blind(monkeypatch, tmp_path, "paths")
    root = tmp_path / "late_model"
    root.mkdir()
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    assert m._round_finish_path().name == "perf_mcp_round_finish_late_model_main.json"
    assert m._stage_ms_path().name == "perf_mcp_stage_ms_late_model_main.json"
    assert m._throughput_path().name == "perf_mcp_throughput_late_model_main.json"


def test_the_round_verdict_and_its_reader_name_the_same_file(monkeypatch, tmp_path):
    """The gate refused 185 times into one filename while the loop read another."""
    m = _blind(monkeypatch, tmp_path, "verdict")
    root = tmp_path / "late_model"
    root.mkdir()
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    m._record_round_finish({"finished": False, "why": "still short"})
    assert (m.read_round_finish() or {}).get("finished") is False
    assert m._round_finish_path().exists()


def test_the_stage_doc_is_found_once_the_root_is_known(monkeypatch, tmp_path):
    m = _blind(monkeypatch, tmp_path, "stage")
    root = tmp_path / "late_model"
    root.mkdir()
    (tmp_path / "perf_mcp_stage_ms_late_model_main.json").write_text(
        json.dumps({"run": "a-run", "stages": {"s": 1.0}, "batch": 8})
    )
    assert m.read_stage_batch() == 0, "before the root is known there is nothing to find"
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    assert m.read_stage_batch() == 8


def test_the_import_never_publishes_a_placeholder_name(monkeypatch, tmp_path):
    """Exporting "model" makes model_key answer "model" for the life of the process."""
    import os as _os

    _blind(monkeypatch, tmp_path, "publish")
    assert _os.environ.get("PERF_MCP_MODEL_NAME") in (None, ""), "a placeholder was published"


def test_a_real_name_is_still_published(monkeypatch, tmp_path):
    import os as _os

    root = tmp_path / "known_model"
    root.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    _own(monkeypatch, "PERF_MCP_MODEL_NAME")
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    spec = importlib.util.spec_from_file_location("pmcp_pub", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert _os.environ.get("PERF_MCP_MODEL_NAME") == "known_model"
    assert m._model_key() == "known_model"


def test_the_expression_is_written_once():
    """Eleven copies of one frozen rule is what made this recur in three separate places."""
    code = _code_only(_SRC)
    assert code.count('_MODEL_ROOT.name if _MODEL_ROOT else "model"') == 0
    assert _SRC.count("def _model_key") == 1


def test_the_accessor_delegates_to_the_one_owner():
    i = _SRC.index("def _model_key")
    code = _code_only(_SRC[i : _SRC.index("\ndef ", i + 10)])
    assert "model_key()" in code, "it is not asking the ledger, which owns this fact"


def test_no_model_name_is_typed_into_the_accessor():
    i = _SRC.index("def _model_key")
    code = _code_only(_SRC[i : _SRC.index("\ndef ", i + 10)])
    for typed in ("voxtral", "gemma", "llama", "nemotron"):
        assert typed not in code.lower(), typed
