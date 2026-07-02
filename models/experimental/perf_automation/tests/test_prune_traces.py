import importlib.util
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_under_test",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
reap = perf_mcp._reap_measurement_dir


def test_reaps_our_measurement_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    d = tmp_path / "perf_mcp_abc123"
    (d / "tracy_out" / ".logs").mkdir(parents=True)
    (d / "tracy_out" / ".logs" / "profile_log_device.csv").write_text("x" * 1000)
    assert reap(d) is True
    assert not d.exists()


def test_refuses_non_perf_mcp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    d = tmp_path / "something_important"
    d.mkdir()
    assert reap(d) is False
    assert d.exists()


def test_refuses_perf_mcp_dir_outside_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "realtmp"))
    (tmp_path / "realtmp").mkdir()
    d = tmp_path / "elsewhere" / "perf_mcp_abc"
    d.mkdir(parents=True)
    assert reap(d) is False
    assert d.exists()


def test_idempotent_on_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    d = tmp_path / "perf_mcp_gone"
    assert reap(d) is True
