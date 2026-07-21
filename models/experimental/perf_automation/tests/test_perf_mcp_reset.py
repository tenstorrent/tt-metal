import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_reset_ut",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)


def test_reset_cmds_uses_full_enumerated_from_reset_arg_sets(monkeypatch):
    import agent.probes as P

    monkeypatch.setattr(P, "_reset_arg_sets", lambda: [["-r", "0,1,2,3"]])
    assert perf_mcp._reset_cmds([perf_mcp._TT_SMI, "-r", "0"]) == [[perf_mcp._TT_SMI, "-r", "0,1,2,3"]]


def test_reset_cmds_falls_back_when_source_unavailable(monkeypatch):
    import agent.probes as P

    def _boom():
        raise RuntimeError("no probe")

    monkeypatch.setattr(P, "_reset_arg_sets", _boom)
    fb = [perf_mcp._TT_SMI, "-r", "0"]
    assert perf_mcp._reset_cmds(fb) == [fb]


def test_reset_cmds_preserves_galaxy_arg_sets(monkeypatch):
    import agent.probes as P

    monkeypatch.setattr(P, "_reset_arg_sets", lambda: [["-glx_reset_auto"], ["-glx_reset"], ["-r", "0,1"]])
    assert perf_mcp._reset_cmds([perf_mcp._TT_SMI, "-r"]) == [
        [perf_mcp._TT_SMI, "-glx_reset_auto"],
        [perf_mcp._TT_SMI, "-glx_reset"],
        [perf_mcp._TT_SMI, "-r", "0,1"],
    ]


def test_run_reset_cmds_stops_after_first_success(monkeypatch):
    calls = []

    class _R:
        def __init__(self, rc):
            self.returncode = rc

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _R(0)

    monkeypatch.setattr(perf_mcp._sp, "run", _fake_run)
    perf_mcp._run_reset_cmds([["a", "1"], ["b", "2"]], "where", "note", 10)
    assert calls == [["a", "1"]]
