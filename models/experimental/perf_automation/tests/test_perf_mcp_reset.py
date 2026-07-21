import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_reset_ut",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)


def test_board_reset_delegates_to_run_reset_devices(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(_reset_devices=lambda d: calls.append(d) or "tt-smi -r 0,1 rc=0")
    monkeypatch.setattr(perf_mcp, "_run_module", lambda: fake)
    monkeypatch.setenv("PERF_MCP_DEVICES", "0")
    perf_mcp._board_reset("where", "device recovered")
    assert calls == ["0"]


def test_board_reset_passes_all_when_devices_unset(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(_reset_devices=lambda d: calls.append(d) or "ok")
    monkeypatch.setattr(perf_mcp, "_run_module", lambda: fake)
    monkeypatch.delenv("PERF_MCP_DEVICES", raising=False)
    perf_mcp._board_reset("where", "note")
    assert calls == ["all"]


def test_board_reset_falls_back_to_single_chip_when_run_unavailable(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_run_module", lambda: None)
    seen = []

    class _R:
        returncode = 0

    monkeypatch.setattr(perf_mcp._sp, "run", lambda cmd, **k: seen.append(cmd) or _R())
    perf_mcp._board_reset("where", "note")
    assert seen and seen[0][1:] == ["-r", "0"]


def test_device_recover_and_reclaim_use_board_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(perf_mcp, "_board_reset", lambda where, note: calls.append(note))
    perf_mcp._device_recover("w")
    perf_mcp._reclaim_mesh("w")
    assert calls == ["device recovered", "full-mesh reset (L1 overflow)"]
