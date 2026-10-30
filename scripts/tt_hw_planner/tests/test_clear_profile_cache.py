import importlib.util
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "module_optimize_pc",
    str(Path(__file__).resolve().parents[1] / "commands" / "module_optimize.py"),
)
mo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mo)


def test_clear_profile_cache_removes_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    cache = tmp_path / "perf_mcp_profile_cache"
    cache.mkdir()
    (cache / "deadbeef.json").write_text('{"device_ms": 42.65}')
    mo._clear_profile_cache()
    assert not cache.exists()


def test_clear_profile_cache_noop_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    mo._clear_profile_cache()
    assert not (tmp_path / "perf_mcp_profile_cache").exists()
