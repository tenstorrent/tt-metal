"""EXECUTE the device-adjacent code paths with stubs, so a runtime NameError cannot reach a run.

Why this exists: `py_compile` proves SYNTAX and never executes a line, and the unit tests do not
reach these functions because they expect a device. So a missing import inside `_adaptive_run`
passed both, and surfaced only 15 minutes into a device run -- on the full-pipeline gate, which is
the path that produces the AFTER headline.

These tests call each function once with stubs. They assert nothing about the RESULT (that is the
other tests' job); they assert only that every name resolves and the body runs. A NameError or
AttributeError here is a real defect; a domain error (no device, stubbed subprocess) is expected
and treated as a pass.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types


_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bootstrap(tmp_path):
    run = tmp_path / "runs" / "x"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"config": {"timeout": 10800, "metric": "device_ms"}, "perf_test_resolved": {"path": "t.py"}})
    )
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": 146.72}) + "\n"
    )
    os.environ.update(
        {
            "PERF_MCP_MANIFEST": str(run / "manifest.json"),
            "PERF_MCP_KERNEL_LOG": str(tmp_path / "k.json"),
            "TMPDIR": str(tmp_path),
        }
    )
    return run


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _calls(pm, ccr, tmp_path, run):
    """(label, thunk) for every path that only executes on a real run."""
    pm.subprocess = types.SimpleNamespace(
        Popen=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stubbed")), PIPE=-1, STDOUT=-2
    )
    return [
        # the exact path whose missing import crashed the full-pipeline gate
        ("perf_mcp._adaptive_run", lambda: pm._adaptive_run(["true"], str(tmp_path), dict(os.environ), "smoke")),
        ("perf_mcp._win_threshold", lambda: pm._win_threshold(2266.0)),
        ("perf_mcp.classify_failure", lambda: pm.classify_failure("Segmentation fault")),
        ("perf_mcp._is_measurement_failure", lambda: pm._is_measurement_failure("tt-perf-report exited 1")),
        ("perf_mcp._normalise_rung", lambda: pm._normalise_rung("knob:grid")),
        (
            "perf_mcp._fullpipe_verdict_for",
            lambda: pm._fullpipe_verdict_for(90.0, "trace", "trace+1cq", 100.0, "trace+1cq"),
        ),
        ("perf_mcp._promote_fullpipe_if_committed", lambda: pm._promote_fullpipe_if_committed()),
        ("perf_mcp._head_sha_quiet", lambda: pm._head_sha_quiet()),
        ("perf_mcp._authored_source_files", lambda: pm._authored_source_files(pathlib.Path("."))),
        ("perf_mcp._model_source_fingerprint", lambda: pm._model_source_fingerprint(1)),
        ("perf_mcp._dirty_model_files", lambda: pm._dirty_model_files()),
        ("perf_mcp._detect_partial_capture", lambda: pm._detect_partial_capture(str(run / "profiles"))),
        ("perf_mcp._reliable_forward_ms", lambda: pm._reliable_forward_ms(5.0)),
        ("perf_mcp._autorecord_wedge", lambda: pm._autorecord_wedge("device wedge")),
        ("run.adaptive_timer", lambda: ccr.adaptive_timer(tmp_path, "pcc")),
        ("run._round_hard_cap", lambda: ccr._round_hard_cap(tmp_path, 600)),
        ("run._measure_backstop", lambda: ccr._measure_backstop(tmp_path)),
        (
            "run.watchdog_decide",
            lambda: ccr.watchdog_decide(
                {"op": "round", "op_elapsed": 10, "since_commit": 10, "observed": {}}, agent=None
            ),
        ),
        ("run._host_transfer_ops", lambda: ccr._host_transfer_ops({"ttnn.matmul(a)"})),
        ("run._parse_facts", lambda: ccr._parse_facts("TP=4 DP=1", {"ttnn.matmul(a)"})),
        ("run._git", lambda: (ccr._git(tmp_path, "status"), ccr._git_ok())),
        ("run._tail_lines", lambda: ccr._tail_lines(str(tmp_path / "nope.log"))),
        ("run._observed_stats", lambda: ccr._observed_stats(tmp_path, "pcc")),
    ]


def test_every_hot_path_executes(tmp_path):
    """A NameError/AttributeError in any of these is a defect that would only appear mid-run."""
    run = _bootstrap(tmp_path)
    pm = _load("pm_hot", "cc_optimize/perf_mcp.py")
    ccr = _load("ccr_hot", "cc_optimize/run.py")

    broken = []
    for label, thunk in _calls(pm, ccr, tmp_path, run):
        try:
            thunk()
        except (NameError, AttributeError) as exc:
            broken.append("%s -> %s: %s" % (label, type(exc).__name__, exc))
        except Exception:  # noqa: BLE001 -- a domain error means the body RAN, which is the point
            pass
    assert not broken, "unresolved names in paths that only execute on a real run:\n  " + "\n  ".join(broken)
