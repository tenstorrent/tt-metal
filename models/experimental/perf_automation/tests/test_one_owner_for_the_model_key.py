"""One owner for "what is this model called".

Three places worked it out independently -- perf_mcp at import, ledger_path in measurements, and the
run when it stamps its own environment -- each spelling "explicit name, else the root's basename",
each free to drift. That single fact has already produced two bugs: a final report keyed on the
literal "model" reading a file no run writes, and a matmul sweep whose whole pre-pass was invisible
for the same reason.

model_key returns "" when nothing states a name, rather than a placeholder. Unknown is unknown, and
the callers that must not guess refuse on it; the one legacy path that still needs a filename spells
its fallback where it accepts it, visibly.

perf_mcp's import-time export deliberately does NOT route through it: the ledger is loaded lazily
there and does not exist yet at import. That line is the SOURCE model_key reads, so the two agree by
construction instead of by repeating the rule -- which is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _led():
    from cc_optimize import measurements

    return measurements


def test_an_explicit_name_from_the_caller_wins(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "from_env")
    assert _led().model_key("from_caller") == "from_caller"


def test_the_exported_name_beats_the_root(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "from_env")
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/a/b/from_root")
    assert _led().model_key() == "from_env"


def test_the_root_supplies_it_when_no_name_is_exported(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MODEL_NAME", raising=False)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", "/a/b/from_root")
    assert _led().model_key() == "from_root"


def test_nothing_stated_is_unknown_not_a_placeholder(monkeypatch):
    """The whole reason this exists: "model" is a filename no run writes."""
    for _k in ("PERF_MCP_MODEL_NAME", "PERF_MCP_MODEL_ROOT"):
        monkeypatch.delenv(_k, raising=False)
    assert _led().model_key() == ""


def test_the_legacy_fallback_is_spelled_where_it_is_accepted():
    """ledger_path still needs a filename; it says so itself rather than hiding it in the resolver."""
    src = (_CC / "measurements.py").read_text(encoding="utf-8")
    i = src.index("def ledger_path")
    body = src[i : src.index("\ndef ", i + 10)]
    assert 'model_key(model) or "model"' in body


def test_the_rule_is_written_once():
    """The basename-of-root rule must not reappear beside the resolver that owns it."""
    src = (_CC / "measurements.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    assert code.count('Path(os.environ.get("PERF_MCP_MODEL_ROOT"') == 0, "a second copy of the rule is back"
    assert code.count("def model_key") == 1


def test_the_run_stamp_uses_the_owner():
    src = (_CC / "run.py").read_text(encoding="utf-8")
    i = src.index("def _stamp_model_root")
    body = src[i : src.index("\ndef ", i + 10)]
    assert "model_key()" in body, "the run is spelling the rule out again"


def test_perf_mcp_states_why_it_cannot_use_the_owner():
    """A deliberate exception must say so, or the next reader deletes it and breaks the import."""
    src = (_CC / "perf_mcp.py").read_text(encoding="utf-8")
    i = src.index('os.environ.setdefault("PERF_MCP_MODEL_NAME"')
    seg = src[max(0, i - 700) : i]
    assert "lazily" in seg and "import" in seg


def test_perf_mcp_still_imports():
    """The reason for that exception, as a test: routing it through the ledger raises NameError."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pmcp_model_key_import", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m is not None
