"""The attempt log lives where the rest of the tool's state lives.

PERF_MCP_STATE_DIR moves the ledger, the gate verdicts and the full-pipeline baseline onto real disk.
The attempt log ignored it:

    kernel_log = f"/tmp/cc_kernlog_{model_name}_{task}.json"

Two hardcoded sites, so a crash took the history and left the anchors. On 2026-08-02 the host went
down mid-run at 14:11; /tmp was cleared at boot and run 20's 98 attempts -- every lever tried, every
measurement, every recorded reason -- went with it, while the ledger (already redirectable) survived
wherever it had been pointed. Rebuilding the ladder took a hand-transcription of the report text.

The path now comes from state_dir(), the same source as everything else. Unset, state_dir() is
tempfile.gettempdir(), so the default is byte-identical to today; set, the attempt log lands beside
the ledger and survives a reboot.

Not a merge: the two files keep their separate jobs. The ledger holds measurement anchors, the
attempt log holds what was tried. Only the directory is shared.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def run_mod(monkeypatch):
    import models.experimental.perf_automation.cc_optimize.run as R

    importlib.reload(R)
    return R


def test_the_path_follows_the_state_dir(run_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    p = run_mod._kernel_log_path("gemma3", "main")
    assert Path(p).parent == tmp_path, p
    assert Path(p).name == "cc_kernlog_gemma3_main.json"


def test_the_default_is_unchanged(run_mod, monkeypatch):
    """Unset -> tempfile.gettempdir(), i.e. exactly the old hardcoded location."""
    import tempfile

    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    p = run_mod._kernel_log_path("gemma3", "main")
    assert Path(p).parent == Path(tempfile.gettempdir())
    assert Path(p).name == "cc_kernlog_gemma3_main.json"


def test_the_derived_files_follow_too(run_mod, tmp_path, monkeypatch):
    """.cumulative / .target / .agent.log are built as str(kernel_log) + suffix, so redirecting the
    log has to carry the ladder history with it -- that history is the whole point."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    p = run_mod._kernel_log_path("gemma3", "main")
    for suffix in (".cumulative", ".target", ".agent.log"):
        assert Path(p + suffix).parent == tmp_path


def test_each_task_keeps_its_own_log(run_mod, tmp_path, monkeypatch):
    """Per-(model, task) isolation is why S2TT's ladder never leaks into T2T. It must survive."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    a = run_mod._kernel_log_path("seamless", "s2tt")
    b = run_mod._kernel_log_path("seamless", "t2t")
    assert a != b and Path(a).parent == Path(b).parent == tmp_path


def test_no_site_still_hardcodes_tmp(run_mod):
    """Two call sites built this path by hand; both must go through the helper, or a crash keeps
    eating the history at whichever one was missed."""
    src = Path(run_mod.__file__).read_text()
    assert 'f"/tmp/cc_kernlog_' not in src, "a hardcoded /tmp kernel-log path is back"
