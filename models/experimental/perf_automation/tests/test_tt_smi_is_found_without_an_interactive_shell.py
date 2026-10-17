"""tt-smi is resolved, never assumed to be on PATH.

Nine sites needed the binary. Eight spelled `shutil.which("tt-smi") or "<a home>/.tenstorrent-venv/
bin/tt-smi"`; the ninth, tt_smi_probe, ran the bare name and trusted PATH. That one fails from any
launch that did not inherit an interactive shell -- the venv reaches PATH from a .bashrc line sitting
AFTER the `case $- in *i*) ;; *) return;;` guard, so a service, a cron entry or a CI job gets
FileNotFoundError at Step 1/10 of optimize.

Verified rather than reasoned: `env -i bash -lc 'command -v tt-smi'` finds nothing, with or without
-l. A run started from a terminal is fine only because it inherits that terminal's PATH, which is not
a property of the tool.

The literal path was also one machine's user directory sitting in source. PERF_MCP_TT_SMI states it
for a host that puts the binary elsewhere, PATH answers where PATH knows, and the historical location
-- now written ~-relative -- is the last resort so nothing that works today stops working.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (str(PERF), str(PERF.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PROBES = (PERF / "agent" / "probes.py").read_text(encoding="utf-8")


def _bin():
    from agent.probes import tt_smi_bin

    return tt_smi_bin


def test_an_explicit_location_wins(monkeypatch):
    """A host that puts the binary elsewhere states it, rather than being told where it must live."""
    monkeypatch.setenv("PERF_MCP_TT_SMI", "/opt/somewhere/tt-smi")
    assert _bin()() == "/opt/somewhere/tt-smi"


def test_path_answers_when_it_knows(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TT_SMI", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/local/bin/tt-smi")
    assert _bin()() == "/usr/local/bin/tt-smi"


def test_an_empty_path_still_yields_a_real_location(monkeypatch):
    """THE DEFECT. A non-interactive launch has no tt-smi on PATH; a bare name is FileNotFoundError."""
    monkeypatch.delenv("PERF_MCP_TT_SMI", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    got = _bin()()
    assert got and got != "tt-smi", "fell back to a bare name, which PATH cannot resolve"
    assert Path(got).is_absolute()


def test_the_probe_no_longer_trusts_a_bare_name():
    """tt_smi_probe was the one site that did, and the one that failed."""
    i = _PROBES.index("def tt_smi_probe(")
    body = _PROBES[i : _PROBES.index("\ndef ", i + 10)]
    assert '["tt-smi"' not in body, "the bare name is back"
    assert "tt_smi_bin()" in body


def test_the_location_is_resolved_in_one_place():
    """Eight copies of one rule is how the ninth site came to disagree with all of them."""
    import re

    for _rel in (
        "agent/probes.py",
        "agent/device_recovery.py",
        "cc_optimize/perf_mcp.py",
        "cc_optimize/run.py",
    ):
        src = (PERF / _rel).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert not re.search(r'which\("tt-smi"\)\s*or\s*"/home/', code), _rel
    assert _PROBES.count("def tt_smi_bin") == 1


def test_no_user_directory_is_written_into_source():
    """A path naming one machine's user cannot be the answer for another machine."""
    code = "\n".join(ln for ln in _PROBES.splitlines() if not ln.strip().startswith("#"))
    assert "/home/ttuser" not in code, "a specific user's home is back in source"


def test_the_fallback_is_expanded_not_left_literal():
    """~ is not a path any subprocess can execute."""
    from agent.probes import _TT_SMI_FALLBACK

    assert _TT_SMI_FALLBACK.startswith("~"), "the fallback should be written host-independently"
    i = _PROBES.index("def tt_smi_bin")
    assert "expanduser()" in _PROBES[i : i + 600], "the ~ reaches a subprocess unexpanded"
