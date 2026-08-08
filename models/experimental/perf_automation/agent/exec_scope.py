"""Proactive execution-scope: restrict the optimizer's edit targets to the model
files the profiled workload ACTUALLY runs.

Run once at the start of the loop. Re-runs the perf test under
exec_trace_plugin (sys.settrace), collects the model-source files that execute,
and intersects them with the discovered model_files. The result lands in
ctx.state["exec_scoped_files"]; PLAN edits only those. This is the fix for
multi-modal / multi-head pipelines, where a static file list offers the agent
speech/vocoder/duplicate stubs that a given task never executes.

Best-effort: any failure leaves the scope unset (loop falls back to the full
file list + the reactive edit_inert steering), and never crashes the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def ensure_scope(ctx) -> None:
    if ctx.state.get("exec_scope_done"):
        return
    ctx.state["exec_scope_done"] = True  # attempt once; don't retry on failure
    try:
        cfg = ctx.manifest.get("config", {}) or {}
        pr = ctx.manifest.get("perf_test_resolved") or {}
        perf = pr.get("path") or cfg.get("perf_test")
        case = pr.get("case") or cfg.get("case")
        if not perf:
            return

        model_root = ctx.model_root()
        tt_root = model_root.parents[2]  # .../tt-metal/models/demos/<model> -> tt-metal
        pa_dir = Path(__file__).resolve().parent.parent  # the perf_automation dir (for `agent.*`)

        out = tempfile.mktemp(suffix="_execscope.json")
        attr_out = tempfile.mktemp(suffix="_opattr.jsonl")  # op->source attribution sidecar
        env = dict(os.environ)
        env["TT_EXEC_TRACE_ROOT"] = str(model_root)
        env["TT_EXEC_TRACE_OUT"] = out
        env["TT_OP_ATTR_ROOT"] = str(model_root)  # attribute ttnn ops to model-source lines
        env["TT_OP_ATTR_OUT"] = attr_out
        env["PYTHONPATH"] = f"{tt_root}:{pa_dir}:" + env.get("PYTHONPATH", "")
        vd = cfg.get("visible_devices")
        if vd:
            env["TT_VISIBLE_DEVICES"] = str(vd)

        # one perf-test run, BOTH plugins: exec-trace (which files run) + op-attribution
        # (which source line emits each matmul) -> the automated "where is the hot op" deep-dive.
        cmd = [
            "python",
            "-m",
            "pytest",
            perf,
            "-p",
            "agent.exec_trace_plugin",
            "-p",
            "agent.op_attribution_plugin",
            "-q",
            "-s",
        ]
        if case:
            cmd += ["-k", case]
        subprocess.run(cmd, cwd=str(tt_root), env=env, timeout=int(cfg.get("timeout", 900) or 900), capture_output=True)

        executed: set = set()
        try:
            executed = set(json.load(open(out)))
        except Exception:
            executed = set()
        finally:
            try:
                os.remove(out)
            except Exception:
                pass
        if not executed:
            return

        exec_base = {os.path.basename(e) for e in executed}
        mf = ctx.model_files()
        scoped = [str(f) for f in mf if os.path.basename(str(f)) in exec_base]
        if scoped:
            ctx.state["exec_scoped_files"] = scoped
            ctx.log_event(
                "ROUTE",
                "info",
                f"exec-scope: {len(scoped)}/{len(mf)} model files on the executed path "
                f"-> {[os.path.basename(s) for s in scoped]}",
            )

        # op->source attribution: rank the source lines that emit the most matmul work,
        # so the agent aims at where the hot matmul ACTUALLY executes (the automated
        # deep-dive that removes the human). Best-effort; absence falls back to blind.
        try:
            from . import op_attribution

            hot = op_attribution.aggregate(attr_out)
            if hot:
                ctx.state["hot_sources"] = hot
                ctx.log_event("ROUTE", "info", f"op-attribution: hottest source = {hot[0]['src']} ({hot[0]['ops']})")
        except Exception:
            pass
        finally:
            try:
                os.remove(attr_out)
            except Exception:
                pass
    except Exception as exc:  # best-effort: never break the loop
        ctx.log_event("ROUTE", "warn", f"exec-scope skipped: {exc}")
