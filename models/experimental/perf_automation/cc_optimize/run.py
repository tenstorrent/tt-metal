# SPDX-License-Identifier: Apache-2.0
"""Claude-Code-native (cc) optimize engine, driven in-process from tt_hw_planner.

Ports the scratch bash driver into Python. For EACH discovered pipeline it drives `claude -p`
against the perf-mcp deterministic gate, re-invoking until the gate's OWN termination_check returns
can_stop (the agent's self-declared "done" is never trusted — only the deterministic gate stops it).

Single-key: authenticates off the ambient ANTHROPIC_API_KEY exactly like every other planner command
(auto-up / up / promote / emit-e2e). It does NOT read .env.agent and does NOT mint a proxy — so a
local LiteLLM mapping can't clobber the key the user exported.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import sys
from pathlib import Path

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py. Loaded by path
# because cc_optimize is not a package: these modules run both as scripts and as plain imports.
import importlib.util as _ilu_ts

_ts_spec = _ilu_ts.spec_from_file_location("_tmpstate", str(Path(__file__).resolve().parent / "tmpstate.py"))
_tmpstate = _ilu_ts.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_tmpstate)
state_dir = _tmpstate.state_dir


PERF_DIR = "models/experimental/perf_automation"
CC_DIR = PERF_DIR + "/cc_optimize"
DEFAULT_MAX_ROUNDS = 3

# A REFUSAL IS NOT A CRASH, and the supervisor could not tell them apart because both exited 1.
#
# The auto-restart supervisor exists for a native tt-metal SIGSEGV: a crash kills the whole Python
# process, no in-process handler can catch it, and relaunching is the right answer. It reads any
# non-zero exit as that case -- "likely native crash / device wedge" -- resets the board and runs
# again, up to PERF_MCP_MAX_RESTARTS times.
#
# But a refusal is a DECISION the tool already made on evidence: the preflight suite is red, the
# model tree is dirty under PERF_MCP_REQUIRE_CLEAN. Relaunching re-derives the same decision from
# the same evidence and gets the same answer, so the only effect is to spend three device resets and
# ten minutes before reporting a verdict that was available immediately -- and to bury the reason
# under three "likely native crash" lines that misdescribe it.
#
# So a deliberate refusal exits with its own code and the supervisor returns it untouched. Anything
# that is genuinely unexpected keeps exit 1 and keeps being retried.
EXIT_REFUSED = 3
_LAST_SCORECARD: dict = {}


def _resolve_claude_bin() -> str:
    """Resolve the `claude` CLI to an absolute path so the orchestrator spawn is
    PATH-independent (fixes-plan Point 9). Inlined (stdlib only) because run.py is
    loaded standalone and cannot import the agent-package helper. Falls back to
    bare "claude" so the spawn always gets a string."""
    local = os.path.expanduser("~/.local/bin/claude")
    return (
        os.environ.get("TT_PLANNER_AGENT_BIN")
        or os.environ.get("CLAUDE_BIN")
        or shutil.which("claude")
        or (local if os.path.exists(local) else None)
        or "claude"
    )


_ALLOWED_TOOLS = [
    "mcp__perf-mcp__profile_model",
    "mcp__perf-mcp__measure_candidate",
    "mcp__perf-mcp__check_pcc",
    "mcp__perf-mcp__check_full_pipeline_latency",
    "mcp__perf-mcp__recall_knobs",
    "mcp__perf-mcp__distill_knob",
    "mcp__perf-mcp__git_head",
    "mcp__perf-mcp__git_commit",
    "mcp__perf-mcp__git_revert",
    "mcp__perf-mcp__termination_check",
    "mcp__perf-mcp__record_kernel_attempt",
    "mcp__perf-mcp__check_lever_coverage",
    "mcp__perf-mcp__tp_pick_degree",
    "mcp__perf-mcp__verify_tp_fracture",
    "Read",
    "Edit",
    "Write",
    "Bash",
    "Grep",
    "Glob",
]

_PROMPT = """You are optimizing the TTNN model {model} ({task} pipeline) for {metric} via the perf-mcp tools. Drive {metric} toward the roofline floor. Run CONTINUOUSLY.

HANDS OFF THE HARDWARE — device and process recovery is NOT your job. NEVER run kill, pkill, tt-smi, fuser, or any command that kills a process or resets the device, and NEVER open or close a mesh device yourself. Device wedges, hangs, and leaked device handles are recovered AUTOMATICALLY by the harness (watchdog + supervisor + device reclaim) between rounds. If a perf-mcp tool returns a device error or a measurement appears stuck, do NOT try to fix the device: if you have a measurement, record the attempt; otherwise just note it and move on — the harness will reclaim, reset, and restart as needed. Killing processes or resetting the device yourself WILL BREAK THE RUN (the agent has killed its own orchestrator this way). Your ONLY job is to choose and apply optimizations via the perf-mcp tools and source edits.

termination_check() is the SOLE authority on whether more optimization is needed. It returns a DETERMINISTIC per-op CHECKLIST and a single next_target = {{op, op_class, grid, bound_by, rung}} you MUST work next. The per-op ladder ORDER is: knob:grid -> knob:fidelity -> knob:dtype -> knob:shard -> structural -> tt-lang -> cpp. structural (an ALGORITHMIC restructure) is tried BEFORE the kernel rungs because it is usually a bigger, cheaper win than a hand kernel and a long ladder must not exhaust the iteration budget on tt-lang/C++ before ever reaching it. There are DISTINCT structural levers and one does NOT substitute for another: trace fixes a DISPATCH-bound chain (removes host gaps); KV-cache fixes a RECOMPUTE decode (a repeat_prefill loop with no cache); gather fixes a sparse/MoE matmul that loads experts it never fires. In particular, having trace applied does NOT satisfy a kv-cache target — they address different costs. An op is "nothing left" ONLY when every rung is ticked; you may STOP ONLY when can_stop=true.

LOOP:
  git_head -> termination_check -> read next_target.
  REUSE-FIRST: call recall_knobs(next_target.op_class, next_target.grid, next_target.bound_by) and APPLY/ADAPT any matching catalogued knob (heed its negative knowledge) BEFORE improvising one.
  WARM-START (matmul fidelity/dtype): a matmul_sweep.json may exist in the model directory (Glob for it once) — a pre-pass table of PCC-verified best (fidelity, dtype) per matmul shape, measured EAGER so treat each entry as a STARTING GUESS, not a verdict. When next_target is a matmul on the knob:fidelity or knob:dtype rung, look up next_target's shape (m,k,n) in that table and APPLY its recommended fidelity/dtype FIRST, then check_pcc + measure_candidate + check_lever_coverage and commit/revert AS USUAL (the eager guess still must pass the trace-mode verify). If the file is missing or the shape is absent, proceed normally.
  Do EXACTLY next_target.rung on next_target.op:
    knob:grid  -> full-grid program_config. check_pcc; measure_candidate; commit a real win else revert. record_kernel_attempt(op,'grid',measured_ms,beat_baseline).
    knob:fidelity -> lower math fidelity (HiFi4->HiFi2->LoFi) on this compute-bound op. check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'fidelity',measured_ms,beat_baseline) EVEN IF pcc forced a revert (that marks the knob tried).
    knob:dtype -> lower that op's WEIGHT dtype (bf16->bf8_b->bf4_b). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'dtype',measured_ms,beat_baseline) EVEN IF pcc forced a revert (that marks the knob tried).
    knob:shard -> shard the op's weights/activations into L1 (height/width shard) to cut DRAM reads. check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'shard',measured_ms,beat_baseline) EVEN IF no gain (that marks the knob tried).
    structural-decode / kv-cache -> the decode loop is repeat_prefill: it re-runs the FULL prefill every token because there is NO KV-cache (use_cache=False). This is a DISTINCT lever from the trace dispatch lever — trace removes DISPATCH gaps, it does NOT remove the REDUNDANT RECOMPUTE, so 'trace is already applied' does NOT satisfy this and 'irreducible' is NOT an acceptable answer. You MUST ADD a KV-cache + a single-token decode_step (each token attends to CACHED K/V and computes seq_len=1, not a re-prefill); Read the recipe via recall_knobs(op_class='decode'). check_pcc; measure the per-token ms; commit only a real win. record_kernel_attempt(op='generation_loop','kv-cache',measured_ms,beat_baseline). This target clears ONLY on a MEASURED per-token reduction from the cache (bounded retries; do NOT record 'none/irreducible' for it).
    tt-lang    -> author a tt-lang (ttl) kernel (Read GUIDELINES/11). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'tt-lang',measured_ms,beat_baseline).
    cpp        -> author a C++ Metalium kernel via ttnn.generic_op (Read GUIDELINES/12). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'cpp',measured_ms,beat_baseline).
  COVERAGE — the profiled slice is a REPRESENTATIVE set of layers, not all of them, so after a dtype knob or a kernel swap call check_lever_coverage(op_match, stale_dtype, new_dtype) to CONFIRM the lever reached EVERY layer instance. A repeated block is ONE class instantiated N times, so editing the SHARED block definition/config propagates to all N; editing an instance-specific path (e.g. layers[0], a per-layer override) changes only that one and silently misses the rest. If fully_applied is false, REAPPLY on the shared definition (target the reported missed_blocks) and re-check until fully_applied — a partial application is NOT a real win even if the slice looks faster.
  ALWAYS pass note= to record_kernel_attempt: ONE line stating (a) WHY you tried this lever on this op (the hypothesis — e.g. 'op is DRAM-bw bound, bf8_b weights halve reads') and (b) WHY it won or failed (the outcome reason — e.g. 'kept: 4.1->3.6ms', 'reverted: PCC 0.71<0.95', 'no gain: 4.1->4.1ms bw-bound', 'OOM under trace'). This note is streamed LIVE into the model's RUN_REPORT.md the instant the attempt resolves (win OR fail), so it must explain the reasoning, not just restate the numbers. ALSO pass stages_json to record_kernel_attempt whenever you have per-stage trace timings (the SAME JSON list of {{"name","ms","dominant?"}} you'd pass hitl_gate, e.g. from check_full_pipeline_latency's stage breakdown) — this renders the block-level timing table in RUN_REPORT.md so BOTH hitl and non-hitl runs show where device time went per stage/block.
  TWO measurements are fed back to you each step — use BOTH: (1) measure_candidate returns the per-op tracy device_ms (the fast steering signal that tells you WHICH op moved); (2) check_full_pipeline_latency returns the robust whole-pipeline trace+1cq per-token ms (its `mode` field = trace+1cq, `full_pipeline_ms` + `delta_pct` vs best) — this is the per-iteration VERDICT you bank a compute win on. This is the ONLY production metric — the run is trace+1cq end to end: trace+1cq always engages (a single command queue), so a dtype/grid/fusion/kernel win it confirms is real. The BEFORE number is one 1cq bookend run and the AFTER number is simply your last committed 1cq verdict (no second full-model run at the end).
  (IRON RULE: a real win = check_pcc ok AND check_full_pipeline_latency status 'ok' (moved TOWARD the target / not diverged, at its trace+1cq mode) AND measure_candidate verdict 'valid' AND is_real_gain AND (for a dtype/kernel lever) check_lever_coverage fully_applied (reached every layer, not just the profiled slice). REJECTED, pcc-fail, or a DIVERGED full-pipeline latency is never a win — revert. Note: check_full_pipeline_latency never fails for missing the target, only for getting SLOWER than the trace+1cq best-so-far.)
  WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs had no match), call distill_knob to persist the general technique; if the win RE-USED a provisional lever learned on another model, pass its id to distill_knob to graduate it.
  Re-run termination_check. Repeat. NEVER stop while can_stop=false. NEVER reason a lever "won't help" — prove it by measuring + recording the attempt. For a structural-decode/kv-cache target you may NOT record 'none'/'irreducible' and you may NOT point at an already-applied trace lever as the resolution — you MUST add the KV-cache and prove it by a measured per-token reduction; the gate will keep returning this target until a kv-cache attempt actually lowers the number.

LEAVE CLEAN (commit wins, revert in-progress edits); end with git_head. Report start->final {metric}, committed wins, and per blocking op which rungs were done + measured ms."""


_HITL_PROMPT = (
    _PROMPT
    + """

HITL MODE (human-in-the-loop): you do NOT have git_commit / git_revert. After you apply ONE lever and measure it (check_pcc; measure_candidate; check_full_pipeline_latency; check_lever_coverage for a dtype/kernel lever), call hitl_gate(tried_op, tried_lever, why_tried, is_win, why_not, next_target, next_why, before_ms, after_ms, stages_json) INSTEAD of committing. stages_json = the per-stage trace timings you just measured, a JSON list of {{"name","ms"}} (add "dominant" if known). hitl_gate returns {{action}}: on 'commit' or 'revert' the operator's git action is ALREADY DONE for you — move to the next target; on 'try', apply the operator's returned knob next. Exactly ONE lever per hitl_gate call; never batch. record_kernel_attempt as usual so RUN_REPORT stays live."""
)


def _apply_scope(env, config):
    """Set the visibility+descriptor pair from a manifest config. Delegates to the one owner."""
    try:
        from agent.mesh_descriptor import apply_scope

        return apply_scope(env, config or {})
    except Exception:  # noqa: BLE001 -- a scope that cannot be applied must not stop the run
        return env


def cc_env(repo_root: Path, devices: str) -> dict:
    """Subprocess env for cc. SINGLE-KEY: inherits the ambient environment verbatim (so the user's
    exported ANTHROPIC_API_KEY flows to `claude`), and adds only the repo paths + visible devices.
    Deliberately does NOT read .env.agent — nothing may clobber the exported key."""
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = f"{repo_root / PERF_DIR}{os.pathsep}{repo_root}"
    # MANDATORY from-scratch perf tests: discovery (_enumerate_pipelines) regenerates each pipeline's
    # WHOLE-forward perf test from its demo every run and never reuses a prior/partial one.
    env["PERF_REGEN_PERF_TEST"] = "1"
    pybin = repo_root / "python_env" / "bin"
    if pybin.is_dir():
        env["PATH"] = str(pybin) + os.pathsep + env.get("PATH", "")
    # THE SCOPE IS INHERITED, NOT RE-DECIDED. This used to pop both variables unconditionally --
    # on the reasoning that pinning a chip subset
    # crashes fabric init. It does, but only WITHOUT a mesh-graph descriptor, and the pair is now
    # resolved once after discovery and lives in os.environ. Popping it here re-widened every
    # cc_env-built subprocess back to the whole host: verified on a --devices 0 run where the
    # full-pipeline measurement ran on four chips at 85W each while the rest of the run used one.
    return env


def _python_bin(repo_root: Path) -> str:
    cand = repo_root / "python_env" / "bin" / "python"
    return str(cand) if cand.is_file() else "python"


def _latest_manifest(perf_dir: Path) -> Path | None:
    cands = sorted((perf_dir / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def discover(
    demo_dir: Path, repo_root: Path, devices: str, metric: str, perf_test=None, case=None, pcc_test=None
) -> dict | None:
    """Run before_loop (discovery + per-pipeline perf-test auto-gen) and return the manifest dict."""
    perf_dir = repo_root / PERF_DIR
    cmd = [
        _python_bin(repo_root),
        "-m",
        "agent.before_loop",
        str(demo_dir),
        "--metric",
        metric,
        "--devices",
        devices,
        "--cc-discovery",
    ]
    if perf_test:
        cmd += ["--perf-test", perf_test]
    if pcc_test:
        cmd += ["--pcc-test", pcc_test]
    if case:
        cmd += ["-k", case]
    launch_ts = time.time()
    rc, _ = _run_device_step(
        cmd,
        perf_dir,
        cc_env(repo_root, devices),
        devices,
        int(os.environ.get("PERF_MCP_DISCOVER_TIMEOUT", "10800") or "10800"),
        "discovery",
        capture=False,
        stall_s=adaptive_timer(repo_root, "build", env_key="PERF_MCP_DISCOVER_STALL_SEC"),
    )
    if rc == EXIT_REFUSED:
        # Discovery REFUSED — the lead agent rejected the plan (e.g. the correctness gate does not
        # cover the perf surface). That is a verdict, not a failure to be worked around: neither the
        # complete-manifest fallback below nor a supervisor restart may override it. Propagating the
        # same code keeps the refusal intact all the way out of the process tree.
        print("  [optimize/cc] discovery refused the run — not continuing, not restarting.", flush=True)
        raise SystemExit(EXIT_REFUSED)
    mani = _latest_manifest(perf_dir)
    if mani is None or rc is None:
        return None
    if rc != 0:
        _fresh = mani.stat().st_mtime >= launch_ts
        _m = {}
        try:
            _m = json.loads(mani.read_text())
        except Exception:  # noqa: BLE001
            _m = {}
        _pm = (_m.get("pathmap") or {}) if isinstance(_m, dict) else {}
        _complete = bool(_pm.get("pipelines") or _pm.get("perf_test")) and bool(_pm.get("pcc"))
        if not (_fresh and _complete):
            print(
                "  [optimize/cc] discovery exited %s and its manifest is %s — refusing to run on a "
                "partial manifest." % (rc, "incomplete" if _fresh else "stale"),
                flush=True,
            )
            return None
        print("  [optimize/cc] discovery exited %s but the manifest is complete; continuing." % rc, flush=True)
    return json.loads(mani.read_text())


def pipelines_from_manifest(manifest: dict, model_rel: str) -> list[dict]:
    """Normalize the discovered pipeline list. Multi-modal -> one entry per task; else a single
    'main' pipeline from the top-level perf_test. Paths are made model-root-relative for the mcp env."""
    pm = manifest.get("pathmap", {})
    resolved_case = (manifest.get("perf_test_resolved") or {}).get("case")
    out = []
    for p in pm.get("pipelines", []) or []:
        if not p.get("perf_test"):
            continue
        out.append(
            {
                "task": p.get("task", "main"),
                "perf_test": f"{model_rel}/{p['perf_test']}",
                "pcc_test": f"{model_rel}/{p['pcc_test']}" if p.get("pcc_test") else "",
                "case": p.get("case") or resolved_case,
            }
        )
    if not out and pm.get("perf_test", {}).get("path"):
        out.append(
            {
                "task": "main",
                "perf_test": f"{model_rel}/{pm['perf_test']['path']}",
                "pcc_test": "",
                "case": pm["perf_test"].get("case") or resolved_case,
            }
        )
    return out


def _model_rel_from_perf_test(perf_test) -> str:
    """The model directory, from a perf-test node id.

    pipelines_from_manifest builds every node as "<model_rel>/<path>::<case>", and model_rel is a
    PARAMETER of that function rather than a key on the pipe -- so the model dir has to be recovered
    from the node. A demo's tests live under the model dir, so the model root is everything above
    the first "tests/" segment:

        models/demos/multimodal/gemma3/tests/e2e/test_main_perf.py::test_main_perf
        -> models/demos/multimodal/gemma3
    """
    s = str(perf_test or "").split("::")[0].strip()
    if not s:
        return ""
    parts = Path(s).parts
    if "tests" in parts:
        return str(Path(*parts[: parts.index("tests")]))
    # No tests/ segment: fall back to the file's own directory, which is still inside the model.
    return str(Path(s).parent)


def _mcp_config(repo_root: Path, manifest_path: str, pipe: dict, devices: str, kernel_log: str) -> dict:
    env = {
        "PERF_MCP_MANIFEST": manifest_path,
        "PERF_MCP_PERF_TEST": pipe["perf_test"],
        "PERF_MCP_PCC_TEST": pipe["pcc_test"],
        "PERF_MCP_KERNEL_LOG": kernel_log,
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{os.pathsep}/usr/bin:/bin",
    }
    if pipe.get("case"):
        env["PERF_MCP_PERF_CASE"] = pipe["case"]
    # THE STATE DIRECTORY MUST CROSS THE PROCESS BOUNDARY. This env dict is explicit -- the server does
    # NOT inherit os.environ -- so a redirect set here would apply to the orchestrator only. The
    # orchestrator READS what the server WRITES (summary.py reads the 1cq full-pipeline baseline that
    # perf_mcp writes; the ledger is shared the same way), so a one-sided redirect points them at two
    # different directories and the report silently finds nothing. Unset on both sides they agree via
    # gettempdir(), which is why this is latent rather than broken -- forward it so it stays that way.
    # PERF_MCP_RUN_ID joins these because the recovery counters are scoped to a RUN, and perf_mcp
    # counts in its OWN process -- an unforwarded stamp puts the two sides in different runs, so each
    # would read the other's failures as zero and the backstop would never trigger.
    for _k in ("PERF_MCP_STATE_DIR", "PERF_MCP_LEDGER_DIR", "PERF_MCP_RUN_ID"):
        if os.environ.get(_k):
            env[_k] = os.environ[_k]
    # TELL THE SERVER WHERE THE RUN IS. perf_mcp is a SEPARATE PROCESS and resolves its model dir as
    # `PERF_MCP_MODEL_ROOT or manifest.config.model_root or "."` -- and nothing ever set that
    # variable, so it landed on "." (the server's own cwd), which has no relationship to the run.
    # On gemma-3-12b-it the matmul sweep wrote 14 PCC-gated shapes into the worktree's demo dir
    # while _warm_start_for looked for ./matmul_sweep.json: every lookup returned None and the whole
    # pre-pass was invisible to the deterministic path. Derived from the pipeline's own model_rel so
    # it points INSIDE the isolated worktree -- the copy the sweep actually writes to.
    # Derived from perf_test, which pipelines_from_manifest always builds as
    # "<model_rel>/<path>::<case>" -- model_rel itself is a PARAMETER of that function, not a key on
    # the pipe, so reading pipe["model_rel"] would silently yield None and no-op (the same
    # missing-value-becomes-wrong-answer shape this whole fix is about).
    _mrel = pipe.get("model_rel") or _model_rel_from_perf_test(pipe.get("perf_test"))
    if _mrel:
        env["PERF_MCP_MODEL_ROOT"] = str((Path(repo_root) / _mrel).resolve())
    # The chip scope is inherited from os.environ, resolved once after discovery -- see _apply_scope.
    _seq = os.environ.get("TT_PERF_SEQ_LEN")
    if _seq:
        env["TT_PERF_SEQ_LEN"] = _seq
    for _k in (
        "PERF_MCP_TARGET_BAND",
        "TT_PERF_MODULE_LEVEL",
        "TT_PERF_MESH_ROWS",
        "TT_PERF_MESH_COLS",
        "TT_PERF_SHARD_DEGREE",
    ):
        _v = os.environ.get(_k)
        if _v:
            env[_k] = _v
    return {
        "mcpServers": {
            "perf-mcp": {
                "command": _python_bin(repo_root),
                "args": [str(repo_root / CC_DIR / "perf_mcp.py")],
                "env": env,
            }
        }
    }


def _gate_status(repo_root: Path, mcp_env: dict, devices: str) -> dict:
    """Ask the gate ITSELF (not the agent): can_stop, and whether the run must HALT (e.g. a material
    op needs the tt-lang rung but the ttl toolchain is not installed). Deterministic stop authority."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import perf_mcp as P; "
        "t=P.termination_check\n"
        "for a in ('fn','func','_fn','__wrapped__'):\n"
        "    if hasattr(t,a): t=getattr(t,a); break\n"
        "r=t()\n"
        "print('CANSTOP=' + str(bool(r.get('can_stop'))))\n"
        "print('HALT=' + str(bool(r.get('halt'))))\n"
        # WHICH halt, not just that one happened. `halt` is a truthy STRING naming the kind
        # (needs_host_reboot / device_unrecoverable / True for the tt-lang rung), and the supervisor
        # used to print one hardcoded remedy for all of them.
        "print('HALTKIND=' + ('' if r.get('halt') is True else str(r.get('halt') or '')))\n"
        "print('HALTREASON=' + str(r.get('halt_reason') or r.get('error') or ''))"
    )
    env = cc_env(repo_root, devices)
    env.update(mcp_env)  # PERF_MCP_* so the gate targets this pipeline
    rc, out = _run_device_step(
        [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
        repo_root / PERF_DIR,
        env,
        devices,
        _measure_backstop(repo_root),
        "termination_check",
        stall_s=adaptive_timer(repo_root, "profile", env_key="PERF_MCP_MEASURE_STALL_SEC"),
        observe_op="profile",
        observe_root=repo_root,
    )
    if rc is None:
        return {"can_stop": False, "halt": False, "reason": "", "kind": ""}
    out = out or ""
    reason = kind = ""
    for line in out.splitlines():
        if line.startswith("HALTREASON="):
            reason = line[len("HALTREASON=") :]
        elif line.startswith("HALTKIND="):
            kind = line[len("HALTKIND=") :]
    return {
        "can_stop": "CANSTOP=True" in out,
        "halt": "HALT=True" in out,
        "reason": reason,
        "kind": kind,
    }


# What the operator must DO, per halt kind. Keyed off the gate's own name for the condition, so a
# new halt cannot silently inherit another one's remedy -- which is what "install tt-lang first"
# did for a board that needed a host reboot.
_HALT_REMEDY = {
    "needs_host_reboot": "reboot the host, then re-run",
    "device_unrecoverable": "the device could not be recovered — check the board, then re-run",
    "": "install tt-lang first, then re-run",
}


def _reset_fullpipe_baselines() -> None:
    """Drop the full-pipeline (trace+1cq) bar ONLY when there is no usable one for this (model, task).

    This used to unlink the file unconditionally at task start, and that single line defeated every
    protection built around the bar. The sequence each run was:

        1. delete the bar
        2. measure the BEFORE bookend
        3. whatever that reading happened to be BECAME the bar

    So the ratchet had nothing to ratchet against, and "is the bar readable?" was moot because the
    file was genuinely absent -- establishing from the new reading is the correct behaviour when no
    bar exists. On gemma-3-12b-it that is how a thermally clamped 68.3241 ms reading (14.6 tok/s/u,
    against a true ~34) became the anchor for a whole run, and how a later run replaced a committed
    33.981 with 35.9253.

    The fear the delete was written for -- inheriting a stale best from a DIFFERENT model or module
    -- is already handled: _fullpipe_1cq_name() keys the file by (model, task). The delete was
    vestigial protection from when the file was global, and it was costing the thing it protected.

    So: a usable bar for THIS (model, task) is kept and reused. Anything else -- no file, an
    unparseable one, a non-positive value -- is cleared so the run establishes a fresh one.
    PERF_MCP_FORCE_REBASELINE=1 forces the old unconditional behaviour."""
    p = state_dir() / _fullpipe_1cq_name()
    if str(os.environ.get("PERF_MCP_FORCE_REBASELINE", "")).lower() not in ("1", "true", "yes"):
        try:
            if float(json.loads(p.read_text()).get("full_pipeline_ms") or 0.0) > 0:
                return
        except Exception:  # noqa: BLE001
            pass
    try:
        p.unlink()
    except Exception:
        pass


def _fullpipe_1cq_name() -> str:
    """Filename of the full-pipeline scoreboard, keyed by (model, task) to match perf_mcp. An unkeyed
    global file let a stray process overwrite a live run's AFTER number."""
    model = os.environ.get("PERF_MCP_MODEL_NAME") or Path(os.environ.get("PERF_MCP_MODEL_ROOT", "model")).name
    task = os.environ.get("PERF_MCP_TASK", "main")
    return "perf_mcp_full_pipeline_baseline_1cq_%s_%s.json" % (model, task)


def _read_fullpipe_best_1cq():
    """The best-so-far full-pipeline latency banked by the per-lever gate, as (ms, mode).

    The MODE matters as much as the number. _establish_fullpipe_baseline RE-BASELINES this file when
    the measurement mode changes, because the stored value then means a different thing (an eager
    wall-clock over the whole forward vs a trace+1cq per-token step). The BEFORE bookend is captured
    once and never re-taken, so if the mode moves underneath it the two are different units --
    subtracting them produced "before 47.10 ms -> after 100.00 ms (-112.3% SLOWER)" on
    llama3_1_8b_p150, a regression that is not one. Return the mode so the caller can refuse.
    """
    try:
        p = state_dir() / _fullpipe_1cq_name()
        d = json.loads(p.read_text())
        ms = float(d.get("full_pipeline_ms") or 0.0)
        mode = str(d.get("mode") or d.get("method") or "")
        return (ms, mode) if ms > 0 else (None, "")
    except Exception:  # noqa: BLE001
        return (None, "")


def _fullpipe_e2e(repo_root: Path, mcp_env: dict, devices: str, label: str) -> float | None:
    _fp_t0 = time.monotonic()
    try:
        return _fullpipe_e2e_inner(repo_root, mcp_env, devices, label)
    finally:
        # BUG 4 (#3): the full-pipeline gate is the dominant cost of a round on a big
        # model (llama's is ~1400 s); record it so the pcc/round budgets learn from it.
        try:
            record_observed(repo_root, "pcc", time.monotonic() - _fp_t0)
        except Exception:  # noqa: BLE001
            pass


def _fullpipe_e2e_inner(repo_root: Path, mcp_env: dict, devices: str, label: str) -> float | None:
    """Measure the FULL-model end-to-end (ALL 52 layers, no tracy, prefill + 1 decode) ONCE at trace+1cq
    and print it with `label` (typically the BEFORE bookend). Returns end_to_end_ms or None. This is the
    whole-model SCOREBOARD; the tool is trace+1cq end to end, so the one BEFORE run here establishes the
    same 1cq baseline the per-lever check_full_pipeline_latency then reuses (no separate bookend run).
    The device_ms loop metric is the fast 2-layer STEERING signal; this is the verdict. Disable via
    PERF_MCP_FULLPIPE_E2E=0."""
    # Stale-across-pipelines guard: the dict is only cleared when a PERF_SCORECARD line is parsed,
    # so a pipeline whose run emits none used to display the PREVIOUS pipeline's throughput.
    _LAST_SCORECARD.clear()
    if os.environ.get("PERF_MCP_FULLPIPE_E2E", "1") != "1":
        return None
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import perf_mcp as P; "
        "g=P.check_full_pipeline_latency\n"
        "for a in ('fn','func','_fn','__wrapped__'):\n"
        "    if hasattr(g,a): g=getattr(g,a); break\n"
        "r=g()\n"
        "print('FULLPIPE_MS=' + str(r.get('full_pipeline_ms')))\n"
        "print('FULLPIPE_MODE=' + str(r.get('mode') or r.get('method') or ''))"
    )
    env = cc_env(repo_root, devices)
    env.update(mcp_env)
    env.setdefault("PERF_MCP_FULLPIPE_SAMPLES", "3")
    print(
        f"  [optimize/cc] measuring FULL-model end-to-end ({label}) — ALL layers (uncapped), no tracy (one slow run, minutes)..."
    )
    ms = None
    # UNCAPPED RUN, UNCAPPED BUDGET. This builds every layer and is the one measurement whose number
    # describes the whole model; budgeting it from the capped profile's history killed it at 1686 s
    # (6 x a 281 s capped baseline) when it needed 1734 s. `fullpipe` has its own history and its own
    # cold start, expressed in baseline-profile units like every other op.
    _fp_op = "fullpipe"
    rc, out = _run_device_step(
        [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
        repo_root / PERF_DIR,
        env,
        devices,
        adaptive_timer(repo_root, _fp_op, env_key="PERF_MCP_FULLPIPE_BACKSTOP"),
        f"full-pipeline ({label})",
        stall_s=adaptive_timer(repo_root, _fp_op, env_key="PERF_MCP_FULLPIPE_STALL_SEC"),
        observe_op=_fp_op,
        observe_root=repo_root,
    )
    if rc is None:
        return (None, "")
    mode = ""
    for line in (out or "").splitlines():
        if line.startswith("FULLPIPE_MS="):
            try:
                ms = float(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                ms = None
        if line.startswith("FULLPIPE_MODE="):
            mode = line.split("=", 1)[1].strip()
        if "[full-pipeline-gate]" in line:
            print("  [optimize/cc] " + line.strip())
            if "PERF_SCORECARD" in line:
                _LAST_SCORECARD.clear()
                for tok in line.split("PERF_SCORECARD", 1)[1].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        _LAST_SCORECARD[k] = v
    if ms is not None:
        print(
            f"  [optimize/cc] FULL-model end-to-end ({label}) = {ms:.1f} ms"
            # WHAT ONE PASS OF THIS MODEL IS. "prefill + 1 decode" was printed for every model
            # measured, so a classifier's one forward pass and a diffusion step were both announced
            # as an LLM's two phases. The run already records the unit trace_replay derived from the
            # pipeline's own structure; unknown prints nothing rather than a borrowed description.
            f"  (ALL layers{_e2e_shape()}{', ' + mode if mode else ''})"
        )
        _ledger_fullpipe(ms, mode, label)
    return ms, mode


def _ledger():
    """The measurement ledger (cc_optimize/measurements.py), loaded by path."""
    global _LEDGER_MOD
    try:
        return _LEDGER_MOD
    except NameError:
        pass
    import importlib.util as _ilu

    _p = Path(__file__).with_name("measurements.py")
    _spec = _ilu.spec_from_file_location("tt_measurements", str(_p))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    globals()["_LEDGER_MOD"] = _m
    return _m


def _e2e_shape() -> str:
    """How one measured pass is composed, from the unit the run recorded. "" when nothing said.

    trace_replay derives the unit from the decode_step CONTRACT, not from any stage name, and prints
    it as TRACE_HEADLINE_UNIT; perf_mcp keeps the last one. token -> an LLM request, step -> a
    diffusion denoise, inference -> one forward pass.
    """
    _u = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT", "") or "").strip().lower()
    return {"token": ", prefill + 1 decode", "step": ", 1 step", "inference": ", 1 forward pass"}.get(_u, "")


def _ledger_fullpipe(ms: float, mode: str, label: str) -> None:
    """Record the whole-model gate reading AT THE MOMENT IT IS TAKEN, with the mode it was taken in.

    The mode matters more than the number: the BEFORE bookend is captured once and never re-taken,
    so an eager BEFORE could sit next to a trace+1cq AFTER and be subtracted, printing
    `before 47.10 ms -> after 100.00 ms (-112.3% SLOWER)`. Stored side by side with their modes,
    that pair is refused instead of computed. The first such reading ever taken for this
    (model, task) is the BEFORE and survives every rerun."""
    try:
        led = _ledger()
        # KEY BOTH THE READ AND THE WRITE. Unkeyed, these used the fallback
        # perf_measurements_model_main.jsonl while perf_mcp's writers used the model's own ledger --
        # so `first()` asked "is there a BEFORE yet?" of a DIFFERENT FILE. On gemma-3-12b-it the
        # genuine before landed in the fallback, the later committed-best found nothing in the real
        # ledger and claimed the BEFORE slot, and the report printed
        # "40.13 ms -> (after not measured yet)" -- with 40.13 actually being the OPTIMIZED result.
        _model = (
            os.environ.get("PERF_MCP_MODEL_NAME") or Path(os.environ.get("PERF_MCP_MODEL_ROOT", "") or "model").name
        )
        _task = os.environ.get("PERF_MCP_TASK", "main")
        seen = led.first(led.KIND_FULLPIPE, led.PHASE_BEFORE, model=_model, task=_task)
        # A RERUN'S OWN BASELINE IS NOT A RESULT. Write-once BEFORE is right for RESULTS -- it keeps the
        # original anchor alive across reruns so the headline reads 84.05 -> x instead of resetting.
        # Applied to the next run's BEFORE bookend it reclassifies a starting measurement as an
        # outcome, and readers take the last AFTER as the current state. gemma-3-12b-it: run 21's cold
        # 36.2548 landed as an AFTER behind run 20's committed 34.9909, so the report showed 36.25 as
        # where the model stood and the next run's bar became 36.25 -- then the run measured warm and
        # "beat" its own cold start three times. The source string on that row still read BEFORE:
        # correctly labelled, filed in the wrong phase.
        #
        # So drop it. The anchor already exists, the run's starting number lives in the gate's own
        # per-run baseline file, and the ledger keeps only readings that describe progress. An
        # UNLABELLED bookend still records: unknown provenance must fail toward keeping the reading.
        if seen and "before" in (label or "").strip().lower():
            return
        phase = led.PHASE_AFTER if seen else led.PHASE_BEFORE
        led.record(
            led.KIND_FULLPIPE,
            phase,
            ms,
            depth="all",
            mode=mode or "unknown",
            source="fullpipe-gate:%s" % (label or ""),
            model=_model,
            task=_task,
        )
    except Exception:  # noqa: BLE001
        pass


# Public alias: this is the ledger's whole-model bookend recorder.
_record_fullpipe_bookend = _ledger_fullpipe


_HOST_XFER_OPS = (
    "from_torch",
    "to_torch",
    "from_device",
    "to_device",
    "as_tensor",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
    "to_torch_tensor",
    ".cpu",
    "concatmeshtotensor",
    "shardtensortomesh",
    "shardtensor2dmesh",
    "replicatetensortomesh",
    "concat_mesh_to_tensor",
    "read_tensor",
    "write_tensor",
    "dump_tensor",
    "load_tensor",
    "numpy",
    "tolist",
    "item(",
)


def _parse_facts(raw: str, sigs: set | None) -> dict:
    """Extract the UNIVERSAL scorecard facts from an op-sig probe run: TP/DP + shard state (from the
    pipeline's MeshDevice line) and whether the step round-trips to host (host-transfer ops in the op
    set) — the latter is the fully-on-device (host-free) gate. Model-agnostic: reads the op stream, not a per-model map.
    """
    # `parallelism_known` distinguishes MEASURED from ASSUMED. The old code defaulted to
    # TP=1 x DP=1 and printed it as a fact, and its regex required "DP=... TP=..." while the only
    # producer (perf_mcp's PERF_SCORECARD line) emits TP before DP -- so it never once matched and
    # even an 8-chip TP=8 run reported single-chip/replicated. Match each field independently.
    facts = {
        "dp": 1,
        "tp": 1,
        "shard_active": False,
        "host_ops": [],
        "n_op_types": len(sigs or ()),
        "parallelism_known": False,
    }
    _raw = raw or ""
    _tp = re.search(r"\bTP=(\d+)", _raw)
    _dp = re.search(r"\bDP=(\d+)", _raw)
    if _tp:
        facts["tp"] = int(_tp.group(1))
    if _dp:
        facts["dp"] = int(_dp.group(1))
    facts["parallelism_known"] = bool(_tp or _dp)
    if re.search(r"shard(?:_active)?\s*=\s*(?:True|true|1|yes)", _raw):
        facts["shard_active"] = True
    facts["host_ops"] = sorted(_host_transfer_ops(sigs or set()))
    return facts


def _host_transfer_ops(sigs) -> set:
    """Which ops in this stream round-trip through the host.

    This drives the "fully on device" verdict, i.e. whether the pipeline is trace-capable. A literal
    name list is the wrong shape for it: the list started at four names, missed `.cpu()`, `as_tensor`
    and the mesh composers, and reported "fully on device: YES" for a pipeline that round-trips.
    Extending the list just moves the miss to the next op. So the KNOWN names resolve locally (free)
    and anything unrecognised is judged once by the agent, from the op name itself, and cached.
    """
    names = {s.split("(")[0].strip() for s in sigs if s}
    known = {n for n in names if any(h in n.lower() for h in _HOST_XFER_OPS)}
    unknown = names - known
    if not unknown:
        return known
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from agent import integrity as _integrity

        for n in sorted(unknown):
            verdict = _integrity.classify(
                n,
                {"host_transfer", "device_op"},
                what="ttnn operation",
                evidence=(
                    "host_transfer = the call moves tensor data between host and device, or converts "
                    "to/from a torch/numpy object (so it breaks a device-side trace). device_op = it "
                    "executes on the accelerator. Operation name: " + n
                ),
            )
            if verdict == "host_transfer":
                known.add(n)
    except Exception:  # noqa: BLE001
        pass
    return known


def _run_op_sigs(repo_root: Path, mcp_env: dict, devices: str, node: str, case, k: int):
    """Run the perf test forward at TT_PERF_LAYERS=k (no tracy, 1 decode token) through the generic
    _op_sig_probe. Returns (sigs_set_or_None, raw_stdout_stderr, sequence_list) -- ALWAYS a 3-tuple.
    The device-timeout path used to return a 2-tuple while all four callers unpack three, so a
    timeout raised ValueError and _print_optimize_stop then blamed "a build/env/version mismatch"
    while the pipeline was simply never optimized."""
    env = cc_env(repo_root, devices)
    env.update(mcp_env)
    # k<=0 means ALL LAYERS and is expressed by REMOVING the cap, never by sending "0": that value
    # arrives as a truthy string and was read by model builders as "build zero layers".
    from agent.layer_depth import set_depth as _set_depth

    _set_depth(env, k)
    env["TT_PERF_OSL_TOKENS"] = "1"
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    cmd = [_python_bin(repo_root), str(repo_root / CC_DIR / "_op_sig_probe.py"), node]
    if case:
        cmd.append(case)
    rc, raw = _run_device_step(
        cmd,
        repo_root,
        env,
        devices,
        _measure_backstop(repo_root),
        "coverage probe",
        stall_s=adaptive_timer(repo_root, "profile", env_key="PERF_MCP_MEASURE_STALL_SEC"),
        observe_op="profile",
        observe_root=repo_root,
    )
    if rc is None:
        return None, "", []
    raw = raw or ""
    sigs = None
    seq = []
    for line in raw.splitlines():
        if line.startswith("PERF_OP_SIGS="):
            try:
                sigs = set(json.loads(line.split("=", 1)[1]))
            except (ValueError, TypeError):
                sigs = None
        elif line.startswith("PERF_OP_SIG_SEQUENCE="):
            try:
                seq = json.loads(line.split("=", 1)[1])
            except (ValueError, TypeError):
                seq = []
    if not sigs:
        return None, raw, []
    return sigs, raw, seq


_LAYER_PATTERN_ATTRS = ("hybrid_override_pattern", "layer_types", "layers_block_type", "block_types")


def _config_layer_kinds(model_name: str):
    """Enumerate distinct layer KINDS from the model's HF-config-declared per-layer pattern, WITHOUT
    building or running the model. Returns (k, n_kinds) where the first k layers already include one of
    EVERY kind (so profiling that slice is representative), or (None, 0) when the config declares no
    per-layer pattern (a homogeneous model, or one that doesn't expose it) so the caller falls back to
    the observed climb. Reading the DECLARED pattern catches a kind that first appears DEEP in the stack
    (past any shallow-probe ceiling) — the exact case an observation-only climb silently misses."""
    if not model_name:
        return None, 0
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:  # noqa: BLE001
        return None, 0
    pat = None
    for attr in _LAYER_PATTERN_ATTRS:
        v = getattr(cfg, attr, None)
        if v:
            pat = v
            break
    seq = list(pat) if pat else []
    if not seq:
        return None, 0
    first: dict = {}
    for i, sym in enumerate(seq):
        first.setdefault(sym, i)
    # ONE reader for "how deep is this model". A flat getattr returns 0 for every multimodal config
    # (gemma3 declares it under text_config, an object), so this silently fell through to len(seq) --
    # the sequence length, which is not a layer count.
    from agent.layer_depth import _depth_from_mapping as _dfm

    n_layers = int(_dfm(cfg) or len(seq)) or len(seq)
    return min(max(first.values()) + 1, n_layers), len(first)


def _coverage_cache_path(repo_root: Path) -> Path:
    return repo_root / CC_DIR / ".coverage_cache.json"


def _coverage_fingerprint(node, repo_root=None) -> str:
    """Newest .py mtime beside the node, or "" when it cannot be established.

    RESOLVED AGAINST THE REPO, AND "NOTHING FOUND" IS NOT A VALUE. `node` is repo-relative, and this
    globbed it against the CURRENT DIRECTORY -- so a caller whose cwd was elsewhere matched no files
    and `default=0.0` handed back the string "0", which is a perfectly valid-looking cache key.

    Two callers then fingerprinted the same node differently and each wrote its own slot:

        depth|...test_main_perf.py::...  {'env': {'TT_PERF_LAYERS': '2', ...}, 'fp': '0'}
        depth|...test_main_perf.py::...  {'env': {},                          'fp': '1786856611'}

    The first is the working depth knob, the second is the verdict that discarded it. The cache
    exists precisely so the second caller reuses the first's answer instead of re-probing, and a
    bogus fingerprint split it in two -- so the expensive probe ran twice AND the second one got the
    wrong answer.

    "" now means unfingerprintable, and the cache refuses to read or write under it rather than
    storing an entry nothing can match.
    """
    try:
        base = Path(str(node).split("::", 1)[0])
        if not base.is_absolute() and repo_root:
            base = Path(repo_root) / base
        mts = [f.stat().st_mtime for f in base.parent.rglob("*.py")]
        return str(int(max(mts))) if mts else ""
    except Exception:  # noqa: BLE001
        return ""


def _coverage_cache_get(repo_root: Path, node, case):
    try:
        _fp = _coverage_fingerprint(node, repo_root)
        if not _fp:
            return None  # unfingerprintable: a cache that cannot be invalidated must not be read
        entry = json.loads(_coverage_cache_path(repo_root).read_text()).get(f"{node}|{case}")
        if entry and entry.get("fp") == _fp:
            return int(entry["k"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _coverage_cache_put(repo_root: Path, node, case, k: int) -> None:
    try:
        path = _coverage_cache_path(repo_root)
        data = json.loads(path.read_text()) if path.is_file() else {}
        _fp = _coverage_fingerprint(node, repo_root)
        if not _fp:
            return
        data[f"{node}|{case}"] = {"k": int(k), "fp": _fp}
        path.write_text(json.dumps(data, indent=1))
    except Exception:  # noqa: BLE001
        pass


def _depth_cache_get(repo_root: Path, node):
    try:
        _fp = _coverage_fingerprint(node, repo_root)
        if not _fp:
            return None
        entry = json.loads(_coverage_cache_path(repo_root).read_text()).get(f"depth|{node}")
        if entry and entry.get("fp") == _fp:
            return dict(entry["env"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _depth_cache_put(repo_root: Path, node, env) -> None:
    try:
        path = _coverage_cache_path(repo_root)
        data = json.loads(path.read_text()) if path.is_file() else {}
        _fp = _coverage_fingerprint(node, repo_root)
        if not _fp:
            return
        data[f"depth|{node}"] = {"env": dict(env), "fp": _fp}
        path.write_text(json.dumps(data, indent=1))
    except Exception:  # noqa: BLE001
        pass


def _claude_text(prompt: str, timeout_s: int = 300):
    env = dict(os.environ)
    for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(_k, None)
    _nat = env.pop("PERF_NATIVE_ANTHROPIC_API_KEY", "")
    if _nat:
        env["ANTHROPIC_API_KEY"] = _nat
    else:
        env.pop("ANTHROPIC_API_KEY", None)
    try:
        r = subprocess.run(
            [_resolve_claude_bin(), "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def _blocks_ran(seq) -> int:
    """How deep the deepest block the model actually ran was.

    TWO SIGNPOST FORMATS, AND THIS READ ONLY ONE. A single-stack model emits
    `PERF_BLOCK_SIGNPOST:7`; the moment a SECOND stack becomes visible the emitter switches to
    `PERF_BLOCK_SIGNPOST:stack2:7` so each block can be attributed to its own stack. Splitting on the
    first colon then parsed "stack2:7" as an integer, raised, was swallowed, and returned 0 -- read
    downstream as the model having no discoverable block stacks at all, which REFUSES the run.

    Measured on Voxtral 2026-08-12: the visibility repair made lm_layers discoverable, the walk went
    from 1 device stack to 2 exactly as intended, the probe emitted 155 signposts -- and the run
    refused, because succeeding is what changed the token format. A latent bug no single-stack model
    could ever reach.
    """
    m = -1
    for tok in seq or []:
        if not isinstance(tok, str) or not tok.startswith(_SIGNPOST_TOKEN):
            continue
        parsed = _parse_signpost_payload(tok[len(_SIGNPOST_TOKEN) :])
        if parsed is not None:
            m = max(m, parsed[1])
    return m + 1


def _model_root_from_node(repo_root: Path, node):
    p = (node or "").split("::", 1)[0]
    if "/tests/" in p:
        p = p.split("/tests/", 1)[0]
    root = repo_root / p
    return root if root.is_dir() else None


def _knob_cache_file():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_knob_cache.json"


def _knob_key(model_root) -> str:
    p = str(model_root).replace("\\", "/")
    i = p.find("models/")
    return p[i:] if i >= 0 else p


def _knob_fingerprint(model_root) -> str:
    try:
        import hashlib

        tt = Path(model_root) / "tt"
        h = hashlib.sha256()
        for f in sorted(tt.rglob("*.py")):
            try:
                h.update(f.read_bytes())
            except Exception:  # noqa: BLE001
                pass
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


def _knob_cache_get(model_root):
    try:
        entry = json.loads(_knob_cache_file().read_text()).get(_knob_key(model_root))
        if entry and entry.get("fp") == _knob_fingerprint(model_root):
            return dict(entry["env"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _knob_cache_put(model_root, env) -> None:
    try:
        data = json.loads(_knob_cache_file().read_text()) if _knob_cache_file().is_file() else {}
        data[_knob_key(model_root)] = {"env": dict(env), "fp": _knob_fingerprint(model_root)}
        _knob_cache_file().write_text(json.dumps(data))
    except Exception:  # noqa: BLE001
        pass


def _stage_layers_var(stage) -> str:
    """layer_depth owns the spelling; lazy import for the same reason _set_depth is."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.layer_depth import stage_layers_var

    return stage_layers_var(stage)


def _stack_layers_var(i) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.layer_depth import stack_layers_var

    return stack_layers_var(i)


def _depth_in_force() -> str:
    """The one answer to "what depth is this". See layer_depth.depth_in_force."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.layer_depth import depth_in_force

        return depth_in_force()
    except Exception:  # noqa: BLE001
        return "all"


def _active_depth_caps(env=None) -> dict:
    """Every depth cap in force in `env`. See layer_depth.active_depth_caps."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.layer_depth import active_depth_caps

        return active_depth_caps(env)
    except Exception:  # noqa: BLE001
        return {}


def _set_depth(env, depth, key=None):
    """Module-level shim for layer_depth.set_depth (imported lazily; the package pulls in device
    deps that must not load at import time)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.layer_depth import set_depth

    return set_depth(env, depth, key=key)


def _knob_at(env, cov):
    # Route through set_depth so the depth variable and PERF_MCP_FORCE_ALL_LAYERS move together. A
    # bare `env[numkey] = str(cov)` left a stale FORCE_ALL armed, and the depth guard then stripped
    # the cap that had just been written -- so a "2-layer" rung profiled the whole model, which is
    # the "every rung profiled the SAME full model" symptom. It also wrote a non-positive cov as
    # the literal "0", a truthy string that builders read as "build zero layers".
    env = dict(env)
    numkey = next((k for k, v in env.items() if str(v).isdigit()), None)
    if numkey:
        _set_depth(env, cov, key=numkey)
    return env


_ENV_READ_RE = re.compile(
    r"""(?:os\.environ\.get|os\.getenv|os\.environ)\s*[\(\[]\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)


_SHALLOWEST_RUNG = 2
_SKIP_DIRS = {"__pycache__", ".git", "build", ".pytest_cache", "node_modules", ".venv"}


def _env_reads(model_root: Path) -> list:
    """Every environment variable the model's own sources READ, as (var, file, line).

    Returns LINES, not files. The prompt used to concatenate whole ``tt/*.py`` files in alphabetical
    order and stop once it passed 60,000 chars -- appending first and checking after -- so on
    llama3_1_8b_p150 attention.py (56 KB) and ccl.py (18 KB) consumed the budget and pipeline.py,
    whose line 70 is the ONLY place TT_PERF_LAYERS is read, was never sent. A prompt built from
    env-read lines cannot be truncated past the answer, because the answer IS one of the lines, and
    the whole set is a few hundred bytes even for a large model.

    Scans the WHOLE model directory, naming no subfolder. Looking only in ``tt/`` found the knob on
    llama, whose pipeline.py lives there, and missed it on gemma3, where the only reader is the
    generated ``tests/e2e/test_main_perf.py`` -- so discovery reported "no knob" for a variable this
    tool had itself injected two steps earlier. Any folder name is a guess about a layout the next
    model is free to differ on; a recursive walk of the model's own directory is not.
    """
    out = []
    if model_root is None or not Path(model_root).is_dir():
        return out
    for py in sorted(Path(model_root).rglob("*.py")):
        if _SKIP_DIRS & set(py.parts):
            continue
        try:
            txt = py.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for line in txt.splitlines():
            for var in _ENV_READ_RE.findall(line):
                out.append((var, py.name, line.strip()))
    return out


def _known_depth_env(model_root: Path) -> dict:
    """The depth knob when the tree already answers, WITHOUT asking an agent.

    The perf test this tool profiles is generated by this tool, and the template hardcodes
    ``os.environ.get("TT_PERF_LAYERS")`` (agent/perf_test_gen.py). Asking an agent which variable
    caps the layer count is therefore asking about a variable THIS TOOL injects -- a question with a
    known answer, put to a model whose source may not even mention it. If the tool's own convention
    is read anywhere in the tree, that is the knob; no agent call is warranted.
    """
    from agent.layer_depth import ENV as _DEPTH_ENV

    if any(var == _DEPTH_ENV for var, _f, _l in _env_reads(model_root)):
        return {_DEPTH_ENV: "1"}
    return {}


def _llm_depth_env(model_root: Path, cov: int) -> dict:
    if model_root is None:
        return {}
    cached = _knob_cache_get(model_root)
    if cached:
        return _knob_at(cached, cov)
    known = _known_depth_env(model_root)
    if known:
        print(f"  [optimize/cc] depth knob is the tool's own convention ({', '.join(known)}); not asking an agent")
        _knob_cache_put(model_root, known)
        return _knob_at(known, cov)
    reads = _env_reads(model_root)
    if not reads:
        return {}
    srcs = [f"{f}: {line}" for _v, f, line in reads]
    prompt = (
        f"This TTNN model runs a stack of transformer layers/blocks. A profiler must execute only {cov} "
        f"layers (a representative slice), not all of them, to keep profiling fast. Below is EVERY line in "
        f"the model's source that reads an environment variable. Find the variable(s) this model reads to "
        f"LIMIT how many layers/blocks it runs, plus any flag it requires to permit a partial/truncated run. "
        f"Respond with ONLY a JSON object mapping env-var name to the string value that makes it run exactly "
        f"{cov} layers; respond with {{}} if the model exposes no such control.\n\n" + "\n".join(srcs)
    )
    attempts = max(1, int(os.environ.get("PERF_MCP_KNOB_RETRIES", "8")))
    for i in range(attempts):
        out = _claude_text(prompt) or ""
        m = re.search(r"\{[^{}]*\}", out, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
            except (ValueError, TypeError):
                d = None
            if isinstance(d, dict) and d:
                env = {str(k): str(v) for k, v in d.items() if str(k)}
                # VALIDATE BEFORE CACHING. Any non-empty dict used to be accepted and written
                # straight to the knob cache, so an invented variable was believed, persisted, and
                # -- because the cache is read first -- reused by every later run. The coverage
                # ladder then set a variable the model never reads, and each rung profiled the
                # identical full model while the run reported that it was slicing. The model's own
                # source is the ground truth and checking it is a grep. All-or-nothing: a knob is a
                # SET applied together, so a half-real answer would still not slice anything.
                _readable = {v for v, _f, _l in reads}
                _unknown = sorted(k for k in env if k not in _readable)
                if _unknown:
                    print(
                        f"  [optimize/cc] depth-knob discovery: rejecting {_unknown} -- not read anywhere in "
                        f"{model_root.name}'s source (model reads: {sorted(_readable) or 'nothing'})"
                    )
                    continue
                _knob_cache_put(model_root, env)
                return _knob_at(env, cov)
        print(f"  [optimize/cc] depth-knob discovery: empty answer, retrying ({i + 1}/{attempts})")
    print(f"  [optimize/cc] depth-knob discovery: no knob after {attempts} attempts (model may not support slicing)")
    return {}


def _is_control(tok) -> bool:
    """A bookkeeping token the probe wrote into the sequence, not an op the device ran.

    The op filter used to name ONE prefix, so every control token invented later was silently
    counted as work. That matters more than it sounds: the work signal is what decides INERT, so
    markers would inflate the very number used to prove a depth cap did nothing -- an uncappable
    model could look cappable purely from its own instrumentation. Caller markers are emitted per
    block ENTRY and EXIT, which on a 32-layer stack is 64 phantom ops.
    """
    return isinstance(tok, str) and tok.startswith(
        ("PERF_BLOCK_SIGNPOST:", "PERF_BLOCK_SIGNPOST_END:", "PERF_STAGE_SIGNPOST:", "PERF_CALLER:")
    )


def _work_signal(seq) -> int:
    return sum(1 for t in seq or [] if isinstance(t, str) and not _is_control(t))


def _op_block_count(seq) -> int:
    from collections import Counter

    ops = [t for t in seq or [] if isinstance(t, str) and not _is_control(t)]
    vals = [c for c in Counter(ops).values() if c > 1]
    if not vals:
        return 0
    return Counter(vals).most_common(1)[0][0]


_SIGNPOST_TOKEN = "PERF_BLOCK_SIGNPOST:"
_SIGNPOST_END_TOKEN = "PERF_BLOCK_SIGNPOST_END:"


def _signpost_entries(seq) -> int:
    """How many times a tagged block was ENTERED -- not how many distinct blocks exist."""
    return sum(1 for t in seq or [] if isinstance(t, str) and t.startswith(_SIGNPOST_TOKEN))


def _signposts_usable(seq) -> bool:
    """Are there signposts to read? That is the ONLY question worth asking here.

    This used to cross-check them against _op_block_count -- "the most common repeat count among the
    ops" -- and discard them when the two differed by more than 20%. Backwards twice:

      * _op_block_count counts op EXECUTIONS, so it reports layers x passes. A perf test that
        prefills and then decodes over 48 layers reports 96 against 48 distinct blocks, a ratio of
        0.5, and the signposts were thrown out on every two-pass model.
      * it is a histogram auditing a direct measurement. _tag_stack attaches an index to each block
        by identity; the histogram infers structure from symbol frequency. The weaker estimate does
        not get to overrule the stronger one.

    _op_block_count remains what runs when there are no signposts -- a fallback, never an auditor.
    Two distinct blocks is the floor: one tagged block has no second boundary to delimit anything.
    """
    idx = [i for i, t in enumerate(seq or []) if isinstance(t, str) and t.startswith(_SIGNPOST_TOKEN)]
    if len({seq[i] for i in idx}) <= 1:
        return False
    # DECOUPLED SIGNPOSTS ARE NOT SIGNPOSTS. A stack whose markers all land in a clump -- typically
    # trailing the ops entirely -- delimits nothing: every op would attribute to block 0. Presence is
    # necessary, interleaving is what makes them usable.
    return any(isinstance(t, str) and not _is_control(t) for t in (seq or [])[idx[0] :])


def _block_start_positions(seq):
    if _signposts_usable(seq):
        pos = [i for i, t in enumerate(seq or []) if isinstance(t, str) and t.startswith(_SIGNPOST_TOKEN)]
        return pos, "signposts"
    n = _op_block_count(seq)
    if n <= 1:
        return [], "none"
    from collections import Counter

    ops = [t for t in seq or [] if isinstance(t, str) and not _is_control(t)]
    counts = Counter(ops)
    anchor = next((t for t in ops if counts.get(t) == n), None)
    if anchor is None:
        return [], "none"
    return [i for i, t in enumerate(seq or []) if t == anchor], "inferred"


def _parse_signpost_payload(raw: str):
    """Parse the payload after 'PERF_BLOCK_SIGNPOST:'.

    Supports two formats:
      - Old (single-stack): '5'               -> ("stack0", 5)
      - New (multi-stack):  'stack0:5'         -> ("stack0", 5)
                            'stack1:12'        -> ("stack1", 12)

    Returns (stack_id: str, block_idx: int) or None on malformed input.
    """
    raw = raw.strip()
    if raw.startswith("stack"):
        colon = raw.find(":", 5)  # find ':' after 'stack'
        if colon == -1:
            return None
        stack_id = raw[:colon]
        idx_str = raw[colon + 1 :]
        if not idx_str.isdigit():
            return None
        return stack_id, int(idx_str)
    # Old format: plain integer
    if raw.isdigit():
        return "stack0", int(raw)
    return None


def _stack_ids_from_seq(seq) -> list:
    """Return the ordered list of distinct stack IDs seen in signpost tokens.

    Scans the op sequence for PERF_BLOCK_SIGNPOST tokens and parses their stack
    prefixes.  The list preserves first-appearance order and always contains at
    least "stack0" (the single-stack default) even when no signposts are present,
    so callers can always build a per-stack dict without a separate None-check.

    Used by the unverified-floor and measured-ladder paths to build a full-coverage
    dict that addresses every known stack rather than just "stack0".
    """
    seen: list = []
    for tok in seq or []:
        if not isinstance(tok, str) or not tok.startswith(_SIGNPOST_TOKEN):
            continue
        raw = tok[len(_SIGNPOST_TOKEN) :]
        parsed = _parse_signpost_payload(raw)
        if parsed is not None:
            sid = parsed[0]
            if sid not in seen:
                seen.append(sid)
    return seen if seen else ["stack0"]


def _first_block_map(seq):
    """Per-stack map of {stack_id: {op: the block it FIRST appears in}}, plus source.

    READ THE SIGNPOST'S OWN INDEX. _tag_stack stamps each block with its position in the stack and
    the probe emits it as "PERF_BLOCK_SIGNPOST:<i>" (single-stack) or
    "PERF_BLOCK_SIGNPOST:stack{si}:<i>" (multi-stack), which is a layer index -- the same unit as
    TT_PERF_LAYERS, the knob this feeds. Counting signposts instead gives an ORDINAL, and the two
    agree only when the model is entered exactly once: a perf test that prefills and then decodes
    enters all 48 layers twice, so the ordinal runs 0..95 and decode's layer 0 reads as block 48. On
    gemma-3-12b-it that produced a coverage window of 96 for a 48-layer model, which was handed to
    the layer knob, stamped into the ledger, and printed as "96 layers".

    Ordering is not assumed either: a model may enter a shared block early, so the index is
    authoritative and the position in the sequence is not.

    The ordinal path stays for sequences with no signposts (source "inferred"), where counting block
    starts is the only information there is.  In that case the result is wrapped under "stack0" for
    consistency: {"stack0": {op: block}}.
    """
    starts, source = _block_start_positions(seq)
    if source == "signposts":
        # Per-stack maps: {stack_id: {op: first_block_idx}}
        # ATTRIBUTE ONLY WHAT RAN INSIDE A BLOCK. The start marker says where a block begins; the END
        # marker says where it stops. Without the closing edge every op after a stack's last block was
        # credited to that block -- and "after the last block" is the entire rest of the model.
        #
        # Measured on Voxtral 2026-08-13: a normal encoder block dispatches 20 ops, and the last one
        # was credited with 12573 -- 67% of the whole run -- including embedding, rms_norm, silu,
        # scaled_dot_product_attention_decode and argmax, which are language-model decode ops not
        # present in the encoder at all. Coverage therefore reported that the 32-layer encoder needed
        # all 32 layers, when every one of its blocks is the same class emitting identical ops and 1-2
        # would do. max() took that 32 as the window, capping to 32 changed no work, and the run
        # profiled the entire model.
        #
        # An op emitted outside every block belongs to no stack: it is prologue, epilogue or another
        # stage, and it cannot say anything about how deep a stack must be profiled.
        # A LIFO, NOT A SINGLE SLOT. A stack can sit INSIDE a block -- experts within a layer, a
        # decoder nested in a wrapper -- and closing the inner block must return to the enclosing one,
        # not to "no block". Clearing instead of popping drops every op the outer block runs after the
        # nested call, which under-sizes it: the ops that would have demanded a deeper window are
        # simply not counted. Flat stacks behave identically either way, which is why this is easy to
        # get wrong and never notice.
        per_stack: dict = {}
        open_blocks: list = []
        for tok in seq or []:
            if not isinstance(tok, str):
                continue
            if tok.startswith(_SIGNPOST_END_TOKEN):
                parsed = _parse_signpost_payload(tok[len(_SIGNPOST_END_TOKEN) :])
                # Close the matching frame, not blindly the innermost: an unbalanced end (a block that
                # raised before its start was recorded) must not unwind a frame it does not own.
                for j in range(len(open_blocks) - 1, -1, -1):
                    if parsed is None or open_blocks[j] == parsed:
                        del open_blocks[j:]
                        break
                continue
            if tok.startswith(_SIGNPOST_TOKEN):
                raw = tok[len(_SIGNPOST_TOKEN) :]
                parsed = _parse_signpost_payload(raw)
                # A malformed payload must not reset the walk to block 0 -- that would attribute
                # the whole rest of the stack to the shallowest window and shrink coverage to nothing.
                if parsed is not None:
                    open_blocks.append(parsed)
                continue
            if not open_blocks or _is_control(tok):
                continue
            sid, idx = open_blocks[-1]
            per_stack.setdefault(sid, {}).setdefault(tok, idx)
        return per_stack, source

    import bisect

    fb: dict = {}
    for i, tok in enumerate(seq or []):
        if not isinstance(tok, str) or _is_control(tok):
            continue
        b = bisect.bisect_right(starts, i) - 1 if starts else 0
        fb.setdefault(tok, max(b, 0))
    # Wrap in single-stack dict for consistent return type
    return {"stack0": fb} if fb else {}, source


def _bridge_depth_env(
    repo_root: Path,
    mcp_env: dict,
    devices: str,
    node,
    case,
    cov,
    full_hint: int = 0,
    full_blocks: int = 0,
    knob=None,
    stage_depths=None,
) -> dict:
    """Verify that the depth cap(s) actually reduce work, and return the env vars to enforce them.

    `cov` may be a plain int (single-stack, backward compat) or a dict mapping stack_id to depth
    (multi-stack).  Either way the probe is run once with ALL caps applied simultaneously and the
    combined op-count is checked against the uncapped baseline.
    """
    if not node or os.environ.get("PERF_MCP_DEPTH_BRIDGE", "1") != "1":
        return {}
    cached = _depth_cache_get(repo_root, node)
    if cached is not None:
        if cached:
            print(f"  [optimize/cc] depth-knob bridge (cached): {cached}")
        return cached
    model_root = _model_root_from_node(repo_root, node)
    if model_root is None:
        return {}
    full_op = int(full_hint)
    full_sp = int(full_blocks)
    _cov_int = next(iter(cov.values())) if isinstance(cov, dict) else int(cov)
    if full_op <= 0:
        # ALL LAYERS FOR THE BASELINE. This measured the "uncapped" probe at _cov_int -- the CAP --
        # so both halves of the comparison ran at the same depth and were identical by construction:
        # the knob could never be shown to reduce work, and every model was profiled at full depth
        # with "did not reduce work ... ignoring" in the log. Run 10, 2026-08-19: op-count
        # 3572 -> 3572, on a pipeline that implements the cap correctly for BOTH towers.
        #
        # IT SURVIVED BECAUSE THE OTHER CALLER WORKS. before_loop passes full_hint, so its bridge
        # gets a real baseline and enforces the cap; only this path improvises one. Both outcomes
        # appear in the SAME run's log, on the same model and the same variables:
        #
        #   enforcing {'TT_PERF_LAYERS': '2', ...}  (op-count 25034->3612)     <- before_loop
        #   ... did not reduce work (op-count 3612->3612); ignoring            <- here
        #
        # 3612 is the CAPPED count from the line above, arriving as this path's "full" baseline. An
        # earlier note here blamed the model -- claiming this only bites pipelines whose knob is the
        # tool's own env convention -- and that is disproved by the first line: those are exactly
        # those variables, and they were enforced. The discriminator is which caller asked.
        #
        # set_depth(env, 0) is the established "no cap" form: it clears the depth variable and arms
        # PERF_MCP_FORCE_ALL_LAYERS, rather than writing a literal 0 that a builder reads as "build
        # zero layers".
        _, _, seq = _run_op_sigs(repo_root, mcp_env, devices, node, case, 0)
        full_op = _work_signal(seq)
        full_sp = _blocks_ran(seq)
    if full_op <= 0:
        print("  [optimize/cc] depth-knob bridge: full-model work-signal is 0 (probe empty); skipping")
        _depth_cache_put(repo_root, node, {})
        return {}
    env = dict(knob) if knob else _llm_depth_env(model_root, _cov_int)
    if not env:
        print(f"  [optimize/cc] depth-knob bridge: no depth knob found (work-signal {full_op})")
        _depth_cache_put(repo_root, node, {})
        return {}
    _numkey = next((k for k, v in env.items() if str(v).isdigit()), None)
    if _numkey:
        _set_depth(env, _cov_int, key=_numkey)  # see _knob_at: never write the cap without clearing FORCE_ALL
    # PER-STAGE FIRST, BY THE NAME THE MODEL DECLARES. The generated perf test reads
    # TT_PERF_<STAGE>_LAYERS and forwards it as the builder's `<stage>_layers`, and the knob repair
    # creates exactly those parameters -- all three derived from PIPELINE_STAGES, which the model
    # states in its own source. Setting a variable nothing reads is how a cap silently caps nothing:
    # measured 18729 -> 18729 when this set TT_PERF_STACK{i}_LAYERS while the test read names an LLM
    # had invented from stack paths.
    #
    # The positional form stays for models that declare no stages, where there is no shared
    # vocabulary to derive from -- it is no worse than what those models had.
    _per_stage = dict(stage_depths or {})
    if _per_stage:
        for _stage, _depth in sorted(_per_stage.items()):
            env[_stage_layers_var(_stage)] = str(_depth)
    elif isinstance(cov, dict) and len(cov) > 1:
        for _i, (_sid, _depth) in enumerate(sorted(cov.items())):
            _stack_key = _stack_layers_var(_i)
            env[_stack_key] = str(_depth)
    probe_env = dict(mcp_env)
    probe_env.update(env)
    _, _, seq2 = _run_op_sigs(repo_root, probe_env, devices, node, case, _cov_int)
    cap_op = _work_signal(seq2)
    cap_sp = _blocks_ran(seq2)
    if full_sp > 1 and cap_sp >= 1 and cap_sp <= full_sp * 0.7:
        full, capped, metric = full_sp, cap_sp, "block-signpost"
    else:
        full, capped, metric = full_op, cap_op, "op-count"
    if capped <= 0 or capped >= full * 0.7:
        print(f"  [optimize/cc] depth-knob bridge: {env} did not reduce work ({metric} {full}->{capped}); ignoring")
        _depth_cache_put(repo_root, node, {})
        return {}
    print(f"  [optimize/cc] depth-knob bridge: enforcing {env} ({metric} {full}->{capped})")
    _depth_cache_put(repo_root, node, env)
    return env


def _env_int(key: str, default: int, allow_zero: bool = False) -> int:
    """A positive int from the environment, or `default` for anything unusable. allow_zero because
    max_shapes uses 0 to mean 'no cap', which is a legitimate value rather than a bad one."""
    try:
        v = int(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default
    if v > 0 or (allow_zero and v == 0):
        return v
    return default


def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _invoke_matmul_sweep(*, node, case, out_path, pcc_threshold, iters, max_shapes, repo_root):
    """Run the sweep as a SUBPROCESS via matmul_sweep.py's CLI, NOT in-process.

    The sweep opens the device mesh to benchmark matmuls. A mesh opened in THIS engine process is not
    released by close_mesh_device -- the UMD device cluster stays held until the process exits -- so
    the op-sig probe CHILD the engine spawns next would deadlock opening the same chips it still holds
    (reproduced: parent-opens-in-process then child-opens hangs at open_mesh_device; two subprocess
    opens do not). A subprocess releases the mesh when it exits, so the probe and every round after
    open cleanly. Reads the summary back from out_path, which the subprocess writes."""
    sweep_py = Path(__file__).resolve().parent / "matmul_sweep.py"
    pa_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = os.pathsep.join([str(pa_root), str(repo_root)])
    cmd = [
        sys.executable,
        str(sweep_py),
        node,
        "--out",
        str(out_path),
        "--pcc",
        str(pcc_threshold),
        "--iters",
        str(iters),
        "--max-shapes",
        str(max_shapes),
        "--repo-root",
        str(repo_root),
    ]
    if case:
        cmd += ["--case", case]
    subprocess.run(
        cmd, cwd=str(repo_root), env=env, timeout=int(os.environ.get("PERF_MCP_MATMUL_SWEEP_TIMEOUT") or 3600)
    )
    try:
        return json.loads(Path(out_path).read_text())
    except Exception:  # noqa: BLE001
        return None


def _matmul_sweep_after_discovery(demo_dir, repo_root, pipes, devices: str = "0") -> None:
    """Run the matmul fidelity x dtype sweep AFTER discovery, using the perf test just generated.

    The sweep used to be a literal pre-pass, called before the engine ran, so the only node it could
    possibly use was an operator-supplied --perf-test: `--matmul-sweep` on its own printed "no node
    to sweep" and silently did nothing, and anyone who wanted the sweep had to hand over a perf test
    the tool was about to generate anyway.

    Nothing required that ordering. The output is a warm-start table consumed much later, when
    next_target is a matmul on the knob:fidelity/knob:dtype rung. Running it here means the node
    exists, no second test is asked for, and the shapes swept are the ones from the SAME node
    optimize goes on to measure rather than a possibly-different hand-passed one.

    Strictly opt-in (PERF_MCP_MATMUL_SWEEP=1) and strictly best-effort: the sweep is an
    optimisation, not a prerequisite, so any failure is reported and the run continues.
    """
    if os.environ.get("PERF_MCP_MATMUL_SWEEP") != "1":
        return
    pipe = next((p for p in (pipes or []) if p.get("perf_test")), None)
    if not pipe:
        print("  [optimize/cc] matmul-sweep: no pipeline with a perf test to sweep; skipping")
        return
    out = str(Path(demo_dir) / "matmul_sweep.json")
    node = pipe["perf_test"]
    case = pipe.get("case")
    print(f"  [optimize/cc] matmul-sweep: {node}{' -k ' + case if case else ''} -> {out}")
    try:
        s = _invoke_matmul_sweep(
            node=node,
            case=case,
            out_path=out,
            pcc_threshold=_env_float("PERF_MCP_MATMUL_SWEEP_PCC", 0.99),
            iters=_env_int("PERF_MCP_MATMUL_SWEEP_ITERS", 5),
            max_shapes=_env_int("PERF_MCP_MATMUL_SWEEP_MAX_SHAPES", 0, allow_zero=True),
            repo_root=repo_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] matmul-sweep failed ({type(exc).__name__}: {str(exc)[-300:]}); optimize continues")
        return
    if isinstance(s, dict):
        print(
            "  [optimize/cc] matmul-sweep: %s shape(s), %s seeded -> %s" % (s.get("shapes", 0), s.get("seeded", 0), out)
        )


def _declared_depth(model_root, model_id: str = ""):
    """The block count the model itself declares, or None when nothing declares one."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.layer_depth import full_depth_from_config

        n = full_depth_from_config(model_id=model_id, model_dir=model_root)
    except Exception:  # noqa: BLE001
        return None
    return n if isinstance(n, int) and n > 0 else None


def _is_emitted_model(model_root) -> bool:
    """Did emit-e2e write this model, i.e. is it bound by emit-e2e's spec?

    Structural, not a name or a flag: an emitted demo carries a _stubs/ directory of graduated
    modules and the e2e_plan.json that routed them. A hand-written tt-metal model has neither.

    This is what separates "must comply" from "measured through the ladder by design". gemma3 and
    llama3_1_8b_p150 legitimately expose no discoverable stacks and are optimized through the
    coverage ladder; refusing them would refuse the entire direct path. A model the tool GENERATED
    has no such excuse -- its spec requires every repeated stack to be discoverable, and a violation
    is a defect in what the tool just produced.
    """
    try:
        root = Path(model_root)
        return (root / "_stubs").is_dir() and (root / "e2e_plan.json").is_file()
    except Exception:  # noqa: BLE001
        return False


_STAGE_TOKEN = "PERF_STAGE_SIGNPOST:"


def stacks_by_stage(seq) -> dict:
    """{stage: [stack ids that ran in it]}, from execution order alone.

    THE LINK BETWEEN A MEASUREMENT AND A KNOB. Coverage is sized per stack -- one saturates at 2,
    another may need 8 -- and a model can accept a depth per stage. Between them nothing said which
    stack IS the encoder: the walk labels stacks by traversal position, the knobs are named for
    stages. So the tool sent max() to every stack and the shallow ones were profiled several times
    deeper than they needed.

    Execution order answers it without HF, without names and without a convention: whichever blocks
    run between the encode and prefill boundaries belong to encode.

    A STACK IN TWO WINDOWS IS NOT AMBIGUOUS. A text decoder runs in prefill AND decode, and it is one
    physical stack, so it appears under both and the caller takes the max of their depths -- deep
    enough for either. Ambiguity would be assigning it ONE stage; listing both is the fact.

    Returns {} when no stage boundaries were emitted, which is the signal to fall back to a single
    uniform depth rather than guess.
    """
    out, cur = {}, None
    for tok in seq or []:
        if not isinstance(tok, str):
            continue
        if tok.startswith(_STAGE_TOKEN):
            cur = tok[len(_STAGE_TOKEN) :]
            out.setdefault(cur, [])
        elif tok.startswith(_SIGNPOST_TOKEN) and cur is not None:
            body = tok[len(_SIGNPOST_TOKEN) :]
            sid = body.split(":")[0] if ":" in body else "stack0"
            if sid not in out[cur]:
                out[cur].append(sid)
    return {k: v for k, v in out.items() if v}


def _model_id_for_facts(model_root) -> str:
    """The model's HF id, read from its own source. "" when there is none.

    THE SECOND SITE OF THE SAME DEFECT, and it had never returned an id. This called
    _hf_repo_ids(Path(model_root)) -- that function takes a parsed Source and does
    `for _path, tree in src.trees.items()`, which raises AttributeError on a Path. The bare except
    turned that into "", so every caller believed the model had no hub id.

    _section_bytes_cached had the identical line and was fixed on 2026-08-17; this one was not,
    because nothing pointed at it until stage_roots went looking. The cost: declared_sections needs
    the id to find the checkpoint in the shared HF cache (the demo directory holds no weights), so it
    returned {} -- and stage_roots bails on an empty section map before it ever reaches its fallback.
    Every fix made to that fallback was therefore unreachable, across four runs.

    model_id_from_source answers the same question and takes a path, which is what the callers have.
    """
    try:
        from agent.stack_survey import model_id_from_source

        return str(model_id_from_source(model_root) or "")
    except Exception:  # noqa: BLE001
        return ""


def _merge_model_facts(model_root, extra: dict) -> None:
    """Merge keys into perf_target_inputs.json, leaving everything else in it untouched.

    Merged rather than rewritten for the same reason the census does it: a hand-tuned per-tensor list
    beside these keys has to survive. Best-effort -- a fact that cannot be written costs a ceiling,
    never a run.
    """
    try:
        p = Path(model_root) / "perf_target_inputs.json"
        doc = {}
        if p.is_file():
            try:
                doc = json.loads(p.read_text()) or {}
            except Exception:  # noqa: BLE001
                doc = {}
        if not isinstance(doc, dict):
            return
        if all(doc.get(k) == v for k, v in (extra or {}).items()):
            return
        doc.update(extra or {})
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(str(tmp), str(p))
    except Exception:  # noqa: BLE001
        pass


def stage_roots(seq, model_root, model_id: str = "", perf_test=None) -> dict:
    """{stage: the top-level module its blocks live under}, or {} when it cannot be established.

    THE LINK THE ROOFLINE NEEDED. A stage streams the weights of the subtree it runs and nothing
    else, so pricing every stage from one whole-model byte count overcharges the backbone stages and
    leaves a third tower unpriced entirely -- an audio encoder measured at 12.8 ms with no ceiling
    beside it. weight_census declines to guess at the split; this establishes it instead.

    BY BLOCK COUNT, which is evidence rather than naming. The probe reports stacks positionally
    (stack2, stack3) and the checkpoint reports them by path (audio_tower.layers: 32,
    language_model.model.layers: 30), and nothing connects the two vocabularies -- but a stack of 32
    blocks and a section of 32 blocks are the same stack. No string matching, no per-model table, and
    it works for a tower whose name nobody has seen.

    AMBIGUITY IS REFUSED, NOT RESOLVED. Two sections of equal depth cannot be told apart this way, and
    a stage spanning two roots has no single answer, so both are left out: the roofline then falls
    back to the whole-model figure for backbone stages and withholds the ceiling for the rest, which
    is what it did before this existed. A wrong divisor is worse than a missing one.
    """
    try:
        from agent.checkpoint_sections import declared_sections
    except Exception:  # noqa: BLE001
        return {}
    secs = declared_sections(model_root, model_id) or {}
    if not secs:
        return {}
    by_count: dict = {}
    for path, n in secs.items():
        try:
            by_count.setdefault(int(n), []).append(str(path))
        except (TypeError, ValueError):
            continue
    counts = {sid: n for sid, n, _kind in _stack_paths(seq)}
    out: dict = {}
    for stage, sids in (stacks_by_stage(seq) or {}).items():
        roots = set()
        for sid in sids or []:
            cands = by_count.get(int(counts.get(sid) or 0)) or []
            if len(cands) == 1:
                roots.add(cands[0].split(".", 1)[0])
        if len(roots) == 1:
            out[str(stage)] = roots.pop()
    # MERGED, NOT SHORT-CIRCUITED. This was `out or _stage_roots_from_generated(...)`, so the
    # fallback ran only when the count join found NOTHING. A PARTIAL count join therefore suppressed
    # a complete one: run 10, 2026-08-19, published stage_roots={'encode': 'audio_tower'} -- one
    # stage of three -- while the generated test named all three unambiguously. prefill and decode
    # were then unmapped on a two-tower model, which is refused rather than guessed, so the two
    # heaviest stages lost their memory ceiling entirely.
    #
    # The two joins answer per STAGE, not per model, and they cannot disagree here: a stage the count
    # join established keeps that answer, and every stage it could not reach asks the other source.
    _gen = _stage_roots_from_generated(secs, perf_test, model_root)
    for _st, _root in (_gen or {}).items():
        out.setdefault(str(_st), _root)
    return out


def _stage_roots_from_generated(secs: dict, perf_test, model_root=None) -> dict:
    """The same mapping, from the perf test the TOOL generated, when the probe's counts cannot give it.

    THE COUNT JOIN CANNOT FIRE DURING A REAL RUN, and had never fired. It compares the block count
    the PROBE observed against the depths the checkpoint declares -- but the probe runs
    depth-capped, by design, so it reports the coverage depth and not the model's:

        stack survey:  audio_tower.layers(32), language_model.model.layers(30)
        the probe:     TT_PERF_LAYERS={'stack2': 2, 'stack3': 2}

    2 matches no section, and both stacks report the same 2, so even a section of depth 2 would be
    ambiguous. Measured on voxtral run 5, 2026-08-16: stage_roots returned {}, every stage fell back
    to the whole-model byte count, and encode -- whose regime the byte model cannot price -- printed
    "not modelled" beside a 12.80 ms measurement.

    The binding exists elsewhere, written by this tool and not inferred. perf_test_gen indexes the
    survey's stacks deepest-first and emits `PERF_STACK{i}_LAYERS -> {path}`; the generated test then
    binds each stage to one of those indices:

        PERF_ENCODE_LAYERS  = _env_layers("TT_PERF_ENCODE_LAYERS",  "TT_PERF_STACK0_LAYERS")
        PERF_PREFILL_LAYERS = _env_layers("TT_PERF_PREFILL_LAYERS", "TT_PERF_STACK1_LAYERS")
        PERF_DECODE_LAYERS  = _env_layers("TT_PERF_DECODE_LAYERS",  "TT_PERF_STACK1_LAYERS")

    Ordering the declared sections the same way -- deepest first -- turns index into path: stack0 is
    the 32-block audio tower, stack1 the 30-block language backbone. Both halves are artifacts this
    tool produced, so nothing here is a guess about the model.

    TIES ARE REFUSED. Two sections of equal depth cannot be ordered by depth, and an order chosen by
    name would be a coin toss that silently prices one tower at another's bytes.
    """
    if not secs:
        return {}
    ordered = sorted(secs.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    depths = [int(c) for _p, c in ordered]
    if len(set(depths)) != len(depths):
        return {}
    # THE BINDING IS IN THE GENERATED TEST, WHEREVER THAT IS -- not in whichever node the caller
    # happened to be given. `pipe["perf_test"]` is null on a run that falls back to the pcc gate
    # ("no_perf_test: perf_test null; falling back to pcc.end_to_end", run 7), and the pcc test
    # carries no stage->stack bindings at all, so keying on it returned {} while three generated
    # perf tests sat on disk beside it holding exactly the mapping wanted.
    #
    # So the node is tried first, and then every generated perf test under the model root. They are
    # written by the same generator from the same survey, so they agree; a file that disagrees is
    # dropped rather than allowed to win a coin toss.
    _srcs = []
    for _cand in [
        str(perf_test or "").split("::", 1)[0],
        *sorted(str(x) for x in Path(model_root).glob("tests/e2e/*_perf.py")),
    ]:
        if not _cand:
            continue
        try:
            _srcs.append(Path(_cand).read_text(errors="ignore"))
        except OSError:
            continue
    if not _srcs:
        return {}
    out: dict = {}
    _seen: dict = {}
    for src in _srcs:
        for m in re.finditer(r"PERF_([A-Z0-9_]+)_LAYERS\s*=.*?TT_PERF_STACK(\d+)_LAYERS", src):
            stage, idx = m.group(1).lower(), int(m.group(2))
            if not (0 <= idx < len(ordered)):
                continue
            _root_for = str(ordered[idx][0]).split(".", 1)[0]
            if _seen.setdefault(stage, _root_for) != _root_for:
                out.pop(stage, None)  # two generated tests disagree: neither is evidence
                continue
            out[stage] = _root_for
    return out


def _last_baseline_profile() -> dict:
    """The most recent baseline profile this run wrote, or {}. Best-effort.

    observed_gathered_numels needs a profile, and _perf_target_inputs is not handed one -- it runs
    from the checkpoint. The baseline is written BEFORE the facts (run 17: profiled at line 117,
    facts emitted at line 144), so by the time this is asked there normally IS one on disk. Read it
    rather than thread a parameter through three layers that have no other use for it.

    Absent is the ordinary pre-baseline case, not an error: the caller falls back to the name rule.
    """
    try:
        _root = Path(__file__).resolve().parent.parent / "runs"
        _cands = sorted(_root.glob("*/profiles/baseline_profile.json"), key=lambda q: q.stat().st_mtime)
        if not _cands:
            return {}
        return json.loads(_cands[-1].read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def observed_gathered_numels(profile) -> list:
    """Element counts of the tensors the DEVICE actually gathered, from the profile. [] when unknown.

    WHAT THE HARDWARE DID, NOT WHAT A NAME SUGGESTS. The compute floor must exclude tensors that are
    INDEXED rather than multiplied, and the rule for finding them was a name list --
    embed_tokens|wte|word_embeddings|token_embedding -- which fails silently on a model that calls
    its table something else, over-charging that tower exactly as voxtral's prefill was.

    The profile already answers it by observation: ops classified `embedding` are gathers, and each
    carries the operand shape it read. Voxtral's baseline shows EmbeddingsDeviceOperation on
    131072x3072 and on 640x128 -- 402,653,184 and 81,920 elements, gathered, measured.

    SHAPE CANNOT SAY *WHICH* TENSOR, AND DOES NOT NEED TO. embed_tokens and lm_head are both
    131072x3072 on this model, so the shape is ambiguous by construction. But the question is how
    many parameters are multiplied, not which object was read: ONE gather was observed, so ONE
    tensor of that size is excluded and the head stays counted. Each distinct observed size is
    subtracted once, never per matching tensor.

    Returns element counts, so the caller can match them against the checkpoint's own numels without
    depending on how a shape happens to be formatted.
    """
    out = []
    try:
        for b in (profile or {}).get("buckets") or []:
            if str((b or {}).get("id") or "").lower() != "embedding":
                continue
            for o in b.get("top_ops") or []:
                _sh = str(o.get("shape") or "")
                # "1x1 @ 131072x3072" -- the operand is the side after the @, which is the table
                _rhs = _sh.split("@")[-1].strip() if "@" in _sh else ""
                dims = [int(x) for x in _rhs.split("x") if x.strip().isdigit()]
                if len(dims) >= 2:
                    n = 1
                    for d in dims:
                        n *= d
                    if n > 0 and n not in out:
                        out.append(n)
    except Exception:  # noqa: BLE001 -- no observation is the name rule's cue, not a failure
        return []
    return out


def _model_block_facts(model_root, model_id: str = "", cfg: dict | None = None, profile=None) -> dict:
    """{root: geometry} for every block the model declares. {} when nothing can be established.

    THE JOIN IS DEPTH, so nothing is recognised by name. Three independent sources each report a
    block's depth -- the checkpoint's sections, the config's sub-dicts, and the probe's stacks -- and
    that shared number is what lets a stage reach its own geometry:

        stage -> root (stage_roots) -> depth (declared_sections) -> geometry (tower_geometry)

    Voxtral: audio_tower.layers 32 -> 32x1280x5120, language_model.model.layers 30 -> 30x3072x8192.
    A vocoder, a denoiser or a second vision stack lands here the same way, with no code change --
    which the "vision"/"audio" name blacklist this replaces could never do.

    AMBIGUITY IS REFUSED. Two sections of equal depth cannot be told apart by depth, so neither gets
    geometry: a stage priced with its neighbour's widths is worse than a stage with no ceiling, and
    the report already knows how to print "not modelled".
    """
    try:
        from agent.checkpoint_sections import declared_sections, hf_cache_dir, tower_geometry
    except Exception:  # noqa: BLE001
        return {}
    try:
        secs = declared_sections(model_root, model_id) or {}
        if not secs:
            # NO SECTIONS TO NAME, which is the ordinary single-tower case: one geometry, and the
            # model IS that block. Published under the empty root, so `len(blocks) == 1` still holds
            # and the flat keys are emitted exactly as before -- a model with one tower loses
            # nothing by the facts becoming per-block.
            _snap1 = (hf_cache_dir(model_id) if model_id else None) or model_root
            _geo1 = tower_geometry(_snap1) or tower_geometry(cfg or {}) or {}
            return {"": dict(next(iter(_geo1.values())))} if len(_geo1) == 1 else {}
        snap = hf_cache_dir(model_id) if model_id else None
        geo_by_depth = tower_geometry(snap or model_root) or tower_geometry(cfg or {}) or {}
        if not geo_by_depth:
            return {}
        depths = [int(d) for d in secs.values()]
        out: dict = {}
        for path, depth in secs.items():
            depth = int(depth)
            if depths.count(depth) != 1:
                continue  # two towers of the same depth: the join cannot separate them
            geo = geo_by_depth.get(depth) or geo_by_depth.get(str(depth))
            if not geo:
                continue
            out[str(path).split(".", 1)[0]] = dict(geo)
        # PARAMS PER BLOCK, from the checkpoint's own tensors. The compute floor is 2 x params x
        # items, and `params` was the WHOLE model for every stage -- so the audio encoder was charged
        # 3.611e9 parameters when its tower has 0.637e9, on top of being charged the wrong geometry.
        # Summed per section from the same header read the numel join already uses; a block the
        # checkpoint does not account for keeps geometry and simply has no param count, which the
        # caller reads as "cannot price the compute term" rather than as zero work.
        try:
            from agent.weight_census import _checkpoint_tensor_sections

            # TWO COUNTS PER TOWER, because two consumers ask different questions of it.
            #
            # `params` is the tower's SIZE -- what it holds -- and the memory floor divides by it.
            # `matmul_params` is what a matmul MULTIPLIES, which excludes a lookup table: an
            # embedding is read by INDEX, one row per token, never multiplied. The compute floor is
            # 2 x params x tokens, so handing it the size charges work that never happens -- on
            # voxtral prefill, 2 x 0.403B x 4096 = 3.30 TFLOP, 18.8 ms of a 222.61 ms floor.
            #
            # The rule is model_bytes._LOOKUP_ONLY, which total_params has always applied; blocks[]
            # arrived later for multi-tower models and recorded only the size. Same regex, so the
            # two definitions cannot drift.
            try:
                from agent.model_bytes import _LOOKUP_ONLY as _LO
            except Exception:  # noqa: BLE001
                _LO = None
            # OBSERVED GATHERS FIRST, THE NAME LIST ONLY AS A FALLBACK.
            #
            # A name list cannot recognise a table a model calls something new, and fails SILENTLY --
            # that tower is then over-charged exactly as voxtral's prefill was. A gather is an op the
            # device RAN, so the profile answers it by observation whatever the tensor is called.
            #
            # ONE OBSERVED SIZE EXCLUDES ONE TENSOR. embed_tokens and lm_head are both 131072x3072
            # here, so a shape cannot say which was gathered -- and does not need to. The question is
            # how many parameters are MULTIPLIED: one gather was seen, so one tensor of that size is
            # excluded and the head stays counted. Matching every tensor of that size would silently
            # drop lm_head too, which is an error in the dangerous direction.
            _obs = list(observed_gathered_numels(profile) or [])
            _pp: dict = {}
            _lk: dict = {}
            _rows = list(_checkpoint_tensor_sections(snap or model_root))
            for _t in _rows:
                _pp[str(_t[1])] = _pp.get(str(_t[1]), 0) + int(_t[0])
            for _n in _obs:
                for _t in _rows:
                    if int(_t[0]) == int(_n):
                        _lk[str(_t[1])] = _lk.get(str(_t[1]), 0) + int(_n)
                        break
            if not _lk and _LO is not None:
                # No observation: the pre-baseline emitter call, or a window that ran no gather.
                # The name rule keeps a ceiling alive from the first second.
                for _t in _rows:
                    _nm = str(_t[2]) if len(_t) > 2 else ""
                    if _nm and _LO.search(_nm):
                        _lk[str(_t[1])] = _lk.get(str(_t[1]), 0) + int(_t[0])
            for _root, _geo in out.items():
                if _pp.get(_root):
                    _geo["params"] = int(_pp[_root])
                    if _lk.get(_root):
                        _geo["lookup_params"] = int(_lk[_root])
                        _mm = max(0, int(_pp[_root]) - int(_lk[_root]))
                        # PIN IT, for the reason the bytes and the peak are pinned: this is the
                        # compute roof's numerator, and the arch mirror it otherwise lives in is a
                        # last-write-wins cache. matmul_params subtracts the gathers the profile
                        # OBSERVED, so a later run that observes a different set recomputes it and
                        # the compute ceiling moves under a measurement that did not. The ceiling is
                        # pinned or it is not; whether this stage happens to be compute-bound is not
                        # the question. Write-once, keyed by section so prefill and decode cannot
                        # disagree about the subtree they share.
                        try:
                            _pinned_mm = _ledger_anchor_matmul_params(model_root, str(_root), _mm)
                            if _pinned_mm:
                                _mm = int(_pinned_mm)
                        except Exception:  # noqa: BLE001
                            pass
                        _geo["matmul_params"] = _mm
        except Exception:  # noqa: BLE001 -- geometry without params still prices the memory term
            pass
        return out
    except Exception:  # noqa: BLE001 -- no blocks is the refused-ceiling path, not a failure
        return {}


def _ledger_anchor_matmul_params(model_root, section: str, value: int):
    """Pin one section's multiplied-parameter count, returning whatever is pinned. 0 when unavailable.

    Best-effort in both directions: a ledger that cannot be reached leaves the freshly computed value
    in place (the behaviour before this pin existed), and a pin that already holds outranks the new
    computation, which is the whole point.
    """
    try:
        from . import measurements as _led
    except Exception:  # noqa: BLE001
        try:
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location("tt_meas_mm", str(Path(__file__).with_name("measurements.py")))
            _led = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_led)
        except Exception:  # noqa: BLE001
            return 0
    try:
        held = _led.anchor(
            _led.KIND_MATMUL_PARAMS,
            float(value),
            depth=str(section).strip().lower(),
            mode="params",
            source="checkpoint sections minus observed gathers",
            model=Path(model_root).name if model_root else "",
        )
        return int(held) if held else 0
    except Exception:  # noqa: BLE001
        return 0


def _publish_stage_roots(seq, model_root, node) -> dict:
    """Establish {stage: tower} and write it into the model's facts. Best-effort; never raises.

    Split out so it can be called from one place, unconditionally. Inline inside the signpost branch
    it inherited that branch's precondition -- a tracy signpost sequence -- which it does not use:
    the count join needs `seq`, the generated-test join needs only the model root and the perf test.
    A model without signposts got no mapping at all, and its towers were priced from one whole-model
    byte count.
    """
    try:
        # NARROW THE ROUTER TO THIS MODEL'S STAGES. The lever catalogue tags each lever with the
        # stage it applies to, and that axis was a fixed {prefill, decode, na}: a lever for an audio
        # encoder could not be tagged at all, and a model with no decode was still routed by a
        # vocabulary that only knows decode. Declared here because this is where the model's stages
        # are already in hand, and it is called unconditionally and early.
        try:
            from agent.model_contract import declared_stage_names
            from agent.router import declare_stages

            declare_stages(declared_stage_names(model_root))
        except Exception:  # noqa: BLE001 -- an undeclared axis stays open, which is the safe default
            pass
        _roots = stage_roots(seq, model_root, _model_id_for_facts(model_root), node)
        if _roots:
            _merge_model_facts(model_root, {"stage_roots": _roots})
            # discovery is the ONLY writer of stage_roots; mirror it here or it is lost on the
            # first revert with no path back (the emitter never produces it).
            _mirror_arch_facts({"stage_roots": _roots})
            print("  [optimize/cc] stage subtrees: %s" % _roots, flush=True)
        return _roots or {}
    except Exception:  # noqa: BLE001 -- a missing mapping is the old behaviour, not a failure
        return {}


def depth_per_stage(per_stack_cov: dict, seq) -> dict:
    """{stage: depth} -- each stage deep enough for every stack that runs in it.

    Conservative by construction: a stage takes the MAX over its stacks, and a stack shared by two
    stages makes both at least that deep. Nothing ends up shallower than the single-number behaviour
    it replaces, so the worst case is what the tool does today and the good case is cheaper.

    Empty when the stage boundaries are missing -- no signposts, an uncalled stack, interleaved
    stages -- and an empty answer means "use one uniform depth", which is the existing path.
    """
    by_stage = stacks_by_stage(seq)
    if not by_stage or not per_stack_cov:
        return {}
    out = {}
    for stage, sids in by_stage.items():
        depths = [int(per_stack_cov[s]) for s in sids if s in per_stack_cov]
        if depths:
            out[stage] = max(depths)
    return out


def _facts_from(raw, sigs, seq) -> dict:
    """The discovery facts a probe's output yields. One place, because a re-walk must produce the
    same shape as the first walk -- recomputing three of the four fields by hand is how a repaired
    model gets described by its unrepaired numbers."""
    facts = _parse_facts(raw, sigs)
    facts["all_ops"] = sorted(sigs)
    facts["full_signal"] = _work_signal(seq)
    facts["full_blocks"] = _blocks_ran(seq)
    return facts


def _stack_census(raw) -> list:
    """What the probe's walk saw, both device and reference stacks. [] if it emitted no census."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.stack_visibility import parse_census

        return parse_census(raw)
    except Exception:  # noqa: BLE001
        return []


def _stack_evidence(model_root, seq) -> dict:
    """What the model's structure is, from witnesses that need no HF reference.

    THE REFERENCE CENSUS ONLY SERVES MODELS THAT CARRY A REFERENCE, and a model trained in-house or
    shipped as a bare checkpoint carries none -- leaving the walk as the only statement of how many
    stacks exist, which is the thing being checked. Two witnesses replace it:

      checkpoint   Weight keys are paths and a repeated block prints its index into every one, so
                   grouping keys gives a section count and a depth per section. No config, no
                   transformers, no torch, no device -- and it reads keys, never values.

      observed     Each plausible container's elements were bracketed during the probe, so a
                   container whose elements all ran and all emitted the same op subsequence IS a
                   stack. This is the only witness that names the exact PATH of a hidden stack, which
                   turns the repair from a search into an edit.
    """
    out = {"checkpoint": {}, "observed": []}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.checkpoint_sections import declared_sections as _ckpt

        out["checkpoint"] = _ckpt(model_root)
    except Exception:  # noqa: BLE001
        pass
    try:
        from agent.caller_stacks import stacks_that_ran

        out["observed"] = stacks_that_ran(seq)
    except Exception:  # noqa: BLE001
        pass
    return out


def make_stacks_visible(model_root, stacks, rewalk=None, attempts: int = 2, evidence=None) -> dict:
    """A stack the model runs and the walk cannot see -- ask for it to be held in a readable shape.

    THE DISCREPANCY IS THE EVIDENCE. A pipeline built from HF weights carries the reference model,
    whose stacks torch holds as ModuleLists of one class, so the walk always sees those. Two
    reference stacks against one device stack is a measured fact: the model has more sections than
    the device side exposes. It does not say what the missing stack looks like, and it does not need
    to -- an agent reading the source can see which list is built from the reference's layers and run
    in sequence by the forward.

    WIDENING THE WALK'S RULE INSTEAD DOES NOT WORK. Two attempts on 2026-08-12: comparing attribute
    sets scored every pair of torch modules as identical (all carry _parameters, _modules, training),
    so three unrelated submodules registered as a stack and shadowed the real ones -- the walk went
    from 5 stacks to 3 and lost an encoder. Comparing child-module names with framework internals
    excluded still could not separate three wrappers around one layer kind from three submodules of a
    model. Both made it worse than leaving it alone.

    Verified by RE-WALKING, not by reading the diff: the caller supplies `rewalk`, the count either
    falls or it did not work, and the feedback for the next round is the stack list as it now reads.
    """
    out = {"hidden": 0, "fixed": False, "rounds": 0}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.stack_visibility import (
            _expected,
            hidden_stack_count,
            repair as _vis_repair,
            retry_prompt,
            stack_counts,
        )

        out["hidden"] = hidden_stack_count(stacks, _expected(evidence))
        if out["hidden"] <= 0:
            return out
        _dev, _ref = stack_counts(stacks)
        _decl = max(_ref, _expected(evidence))
        print(
            "  [optimize/cc] %d block stack(s) are hidden from the walk: %d exist, the device side "
            "exposes %d. A hidden stack cannot be sized, capped or attributed, so it profiles at "
            "FULL depth." % (out["hidden"], _decl, _dev),
            flush=True,
        )
        for _o in (evidence or {}).get("observed") or []:
            print(
                "  [optimize/cc]   observed running: %s (%d blocks)" % (_o.get("path", "?"), _o.get("depth", 0)),
                flush=True,
            )
        cur = stacks
        for i in range(max(1, int(attempts))):
            _vis_repair(model_root, cur, feedback=i > 0, evidence=evidence)
            out["rounds"] = i + 1
            if rewalk is None:
                break
            cur = rewalk() or cur
            if hidden_stack_count(cur, _expected(evidence)) <= 0:
                out["fixed"] = True
                break
            print(
                "  [optimize/cc] still hidden after round %d; %s" % (i + 1, retry_prompt(cur).splitlines()[0]),
                flush=True,
            )
        _d2, _r2 = stack_counts(cur)
        print(
            "  [optimize/cc] stack visibility after %d round(s): device stacks %d -> %d (reference %d)"
            % (out["rounds"], _dev, _d2, _r2),
            flush=True,
        )
    except Exception:  # noqa: BLE001 -- a repair that cannot run leaves the model as it was
        pass
    return out


def make_model_cappable(model_root, seq=None, per_stack_cov=None, n_stacks=1) -> dict:
    """The depth went nowhere -- get the model a knob that receives it.

    CALLED ON THE INERT VERDICT, which is the only MEASURED proof that a model cannot be capped:
    the cap was applied, the work signal did not move, so nothing consumed it. Everything else is
    inference. A model can have five discoverable stacks and a factory that drops `layers` silently
    (Voxtral: build_pipeline(device, model=None, **kwargs)), and it can equally have one stack and no
    knob at all -- both land here, and both mean every profile builds the whole model. Measured on
    Voxtral with the variable unset: n_layers=30, enc_a=32, enc_b=32, bulk=27.

    The base knob is what meets the goal: ONE depth that reaches every repeated stack. Per-stage
    overrides are added only when the run knows which stage each stack ran in, because a name taken
    by position is worse than no name -- slicing PIPELINE_STAGES by stack count asked Voxtral for
    `prefill_layers` when both of its visible stacks run in encode.

    Opt-in, and verified by the same check that triggered it: after the edit the caller caps and
    re-measures, so a repair that adds parameters without wiring them reports INERT exactly as an
    unrepaired model does. Nothing here is trusted on the agent's word.
    """
    out = {"needed": [], "added": [], "attempted": False}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.stack_knob_repair import missing_knobs, repair as _knob_repair

        stage_map = stacks_by_stage(seq) if seq else None
        needed = missing_knobs(model_root, n_stacks, stage_map)
        out["needed"] = needed
        if not needed:
            return out
        # NOT BEHIND A FLAG. Capping is how this tool profiles at all: without a depth the builder
        # can receive, every profile builds the whole model, and on a 3B multimodal pipeline that is
        # 36.8M tracy zones, tracy's 32K source-location limit exceeded, a test process that never
        # exits, and a run killed at its budget having measured nothing. An operator cannot opt in to
        # the tool working -- either it makes the model measurable or it cannot do its job.
        #
        # This is not the same as editing a model for PERFORMANCE, which stays a decision. The depth
        # argument changes no numerics at full depth (None means every layer, exactly as before); it
        # exists so the profiler can look at a slice. It is instrumentation, and the run verifies it
        # by re-measuring rather than trusting the edit.
        print("  [optimize/cc] making the model cappable: adding %s" % ", ".join(needed), flush=True)
        res = _knob_repair(model_root, _stack_paths(seq or []), needed)
        out["attempted"] = True
        out["added"] = res.get("added") or []
        # SAY HOW MANY ROUNDS AND WHAT THE SIGNATURE ENDED UP AS. The first live run printed only
        # "added nothing", which cannot distinguish "the agent was asked once and missed" from "it
        # was asked three times and missed every time" -- and those want opposite fixes. The
        # parameter list is the fact that decides it.
        print(
            "  [optimize/cc] knob repair after %d round(s): added %s; build_pipeline now takes (%s)"
            % (res.get("rounds", 0), ", ".join(out["added"]) or "nothing", ", ".join(res.get("params") or [])),
            flush=True,
        )
    except Exception:  # noqa: BLE001 -- a repair that cannot run leaves the model as it was
        pass
    return out


def _stack_paths(seq) -> list:
    """(stack id, block count) per stack, as the probe reported them -- the repair's targets."""
    seen = {}
    for tok in seq or []:
        if not isinstance(tok, str) or not tok.startswith(_SIGNPOST_TOKEN):
            continue
        parsed = _parse_signpost_payload(tok[len(_SIGNPOST_TOKEN) :])
        if parsed is None:
            continue
        sid, idx = parsed
        seen[sid] = max(seen.get(sid, -1), idx)
    return [(sid, top + 1, "block") for sid, top in sorted(seen.items())]


def _declared_sections(model_root, model_id: str = "") -> list:
    """Every block depth the config declares, per section, deepest first.

    THE CONFIG IS THE AUTHORITY ON STRUCTURE, and it costs nothing: transformers has already parsed
    it, so "how many repeated sections does this model have, and how deep is each" is answerable
    before the device is touched -- no markers, no walk, no naming convention, no per-model code.

    _walk_depths always collected this and _depth_from_mapping reduced it with max() one line later,
    because both callers wanted a single ceiling. Voxtral-Mini-3B declares 32 for the audio tower and
    32 for the text decoder; the tool saw one number, sized one depth, capped the text decoder and
    left both encoders whole, and nothing could notice because the question was never asked.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.layer_depth import declared_section_depths

        found = [int(d) for d in declared_section_depths(model_id=model_id, model_dir=model_root) if int(d) > 0]
    except Exception:  # noqa: BLE001
        found = []
    if found:
        return found
    # NO CONFIG IS NOT NO STRUCTURE. declared_section_depths reads a transformers config, so every
    # model that did not come from HF -- trained in-house, exported from a research repo, shipped as a
    # bare checkpoint -- answered "no sections" and lost the one independent witness that says how
    # many stacks there should be. The weights themselves declare it: a repeated block prints its
    # index into every key it owns, so grouping keys gives a section count and a depth per section
    # with no config, no transformers, no torch and no device.
    try:
        from agent.checkpoint_sections import declared_sections as _ckpt_sections

        return sorted((int(d) for d in _ckpt_sections(model_root).values() if int(d) > 0), reverse=True)
    except Exception:  # noqa: BLE001
        return []


def _cov_ladder(model_root, model_id: str = "") -> list:
    """The coverage-search rungs, BOUNDED BY THE MODEL'S DECLARED DEPTH.

    The ladder used to be the literal default "2,4,8,16" with no idea how deep the model actually
    is. On llama3_1_8b_p150 (32 layers) that meant the search topped out at 16 with 2 op types still
    uncovered, returned 16 anyway as "measured", and the run wore a false "(16 layers)" label from
    then on. The rung that would have covered those 2 op types is the FULL depth, where coverage is
    total by construction -- and the model declares that number in its own config.

    layer_depth.full_depth_from_config() reads it without building or running anything, but it had
    zero production callers; this is that caller. When nothing declares a depth it returns None and
    we keep the plain ladder rather than inventing a bound.

    Also drops rungs deeper than the model: probing 16 on a 6-layer model burns device time and
    yields two rungs that measure the identical thing.
    """
    raw = os.environ.get("PERF_MCP_COV_LADDER", "2,4,8,16")
    rungs = [int(x) for x in raw.split(",") if x.strip().isdigit()]
    full = _declared_depth(model_root, model_id)
    if full is None:
        return rungs
    out = [d for d in rungs if d < full]
    out.append(full)
    return out


def _validate_signpost_window(window: int, stack_len: int, declared) -> tuple:
    """(ok, why) for a window the signpost path derived, cross-checked against the model's config.

    Two free checks, and neither may raise -- they run inside a live optimize where a wrong window is
    recoverable by falling back and a crash is not.

      window <= declared        A window deeper than the model is a UNIT bug: something counted
                                blocks, or executions, and handed the result to a layer knob.
                                gemma-3-12b-it produced 96 for 48 layers and nothing noticed.
      stack_len == declared     Confirms the walker tagged the DECODER stack. It picks the largest
                                list of same-typed repeated objects, which can just as easily be the
                                vision tower (27) or a sub-block list inside one layer (9) -- the
                                9-element case really happened, and only a live run caught it. The
                                indices look perfectly reasonable; they are about the wrong stack.

    No declared depth is MISSING INFORMATION, not a failure: a model that ships no config cannot be
    cross-checked, and the window still stands on the signposts alone.
    """
    try:
        d = int(declared or 0)
    except (TypeError, ValueError):
        d = 0
    if d <= 0:
        return True, "no declared depth to check against"
    try:
        w, s = int(window), int(stack_len)
    except (TypeError, ValueError):
        return True, "window or stack length not numeric"
    if w > d:
        return False, "window %d exceeds the declared depth %d -- the index is not in layer units" % (w, d)
    if s and s != d:
        return False, "tagged stack has %d blocks but the model declares %d -- wrong stack tagged" % (s, d)
    return True, "window %d within declared depth %d" % (w, d)


def _cap_took_effect(capped_signal, full_signal):
    """Did capping to the window actually make the model smaller? True / False / None for unknown.

    The signal is the work the probe observed. Identical work at 2 layers and at full depth means the
    cap never reached the builder -- a knob is a NAME the tool exports, and whether anything acts on
    it is a property of the model. gemma3 reads TT_PERF_LAYERS in its perf test and never forwards it
    to build_pipeline, which has no depth parameter, so every rung profiles the same 48 layers.

    NONE IS NOT TRUE. A missing measurement must not read as "the cap worked" -- claiming a window
    that was never verified is the failure this exists to stop, and it is how a full-model profile
    ends up labelled as an N-layer one.
    """
    if not capped_signal or not full_signal:
        return None
    return capped_signal != full_signal


def _knob_is_inert(seq_at_depth, full_signal, depth, model_root) -> bool:
    """Did capping to `depth` actually make the model smaller?

    A knob is a NAME the tool exports; whether anything acts on it is a property of the model, not
    of the name. gemma3 shows the failure: its generated perf test READS TT_PERF_LAYERS but its
    build_pipeline takes no depth argument and reads no env, so the value is computed and dropped.
    Every rung then profiles the identical full model, returns the full op set, satisfies the
    coverage test on the FIRST rung, and the run is labelled "measured" against a window that was
    never applied -- strictly worse than the honest "unverified-floor" it replaces.

    The existing absent-op-types guard cannot catch this: with the cap ignored, nothing is missing.
    Work signal can: slicing a model to 2 of N layers must reduce the op count, so an identical
    signal means the cap did not reach the builder.
    """
    if not full_signal or not seq_at_depth:
        return False
    declared = _declared_depth(model_root)
    if declared is None:
        # An UNDECLARED depth means we cannot tell "the cap was ignored" from "this rung IS the whole
        # model" -- a 4-layer model probed at rung 4 legitimately does full-model work. Only the
        # shallowest rung is safe to judge: a model with <= 2 blocks has nothing to slice anyway.
        return depth <= _SHALLOWEST_RUNG and _work_signal(seq_at_depth) == full_signal
    if depth >= declared:
        return False
    return _work_signal(seq_at_depth) == full_signal


def _measure_cov(
    repo_root: Path, mcp_env: dict, devices: str, node, case, full_types, model_root, base_knob=None, full_signal=None
):
    base = dict(base_knob) if base_knob else (_llm_depth_env(model_root, 2) if model_root is not None else {})
    if not base:
        print("  [optimize/cc] coverage measurement skipped: no depth knob")
        return None
    numkey = next((k for k, v in base.items() if str(v).isdigit()), None)
    want = set(full_types or [])
    if not want:
        return None
    ladder = _cov_ladder(model_root)
    got = set()
    for d in ladder:
        env = dict(base)
        if numkey:
            _set_depth(env, d, key=numkey)  # see _knob_at: cap and FORCE_ALL must move together
        penv = dict(mcp_env)
        penv.update(env)
        sigs_d, _, seq_d = _run_op_sigs(repo_root, penv, devices, node, case, d)
        if not sigs_d:
            print(f"  [optimize/cc] coverage measurement inconclusive: depth-{d} probe returned no ops")
            return None
        if _knob_is_inert(seq_d, full_signal, d, model_root):
            print(
                f"  [optimize/cc] depth knob is INERT: capping to {d} produced the SAME work signal "
                f"({full_signal}) as the full model, so the cap never reached the builder. Refusing to "
                f"report a coverage window measured against an uncapped model."
            )
            # THE MEASURED MOMENT: the cap was applied and nothing consumed it. Repair here, then
            # re-measure -- if the knob now bites, the coverage search continues on a model that can
            # actually be capped instead of abandoning the window.
            _fix = make_model_cappable(model_root, seq=seq_d, n_stacks=1)
            if _fix.get("added"):
                sigs_r, _, seq_r = _run_op_sigs(repo_root, penv, devices, node, case, d)
                if sigs_r and not _knob_is_inert(seq_r, full_signal, d, model_root):
                    print(
                        "  [optimize/cc] the knob now caps: work signal moved after the repair",
                        flush=True,
                    )
                    sigs_d, seq_d = sigs_r, seq_r
                else:
                    print(
                        "  [optimize/cc] still INERT after the repair -- the arguments were added "
                        "but nothing consumed them; profiling FULL depth.",
                        flush=True,
                    )
                    return None
            else:
                return None
        got = set(sigs_d)
        if want <= got:
            return d, [], "measured"
    # THE SEARCH FAILED. This used to return "measured" -- the same label a real hit gets -- so
    # "16 covers every op type" and "16 was just the last rung and 2 op types were never seen" were
    # indistinguishable, and the run proceeded on a window known not to cover the model.
    missing = sorted(want - got)
    last = ladder[-1] if ladder else 16
    # The DECLARED depth, not the ladder's last rung -- the two are equal whenever a depth was
    # declared, so comparing against the rung would be circular and would also mislabel an
    # undeclared-depth model (whose ladder merely stops at 16) as having an inert knob.
    _full = _declared_depth(model_root)
    if missing and _full is not None and last >= _full:
        # An op type CANNOT be absent at full depth -- full depth is the whole model. So the cap is
        # not taking effect: the knob names something the model never acts on, and every rung just
        # profiled the identical full model. Reporting a coverage number here would launder a
        # broken knob into a measurement.
        print(
            f"  [optimize/cc] coverage: {len(missing)} op-type(s) still absent at the model's FULL depth "
            f"({last}) -- impossible unless the depth knob is not slicing. Treating the knob as INERT "
            f"rather than reporting a coverage window. Missing: {missing}"
        )
        return None
    print(
        f"  [optimize/cc] coverage: ladder exhausted at {last} with {len(missing)} op-type(s) never seen "
        f"{missing}; this window does NOT cover the model"
    )
    return last, missing, "measured-incomplete"


def _cap_cov_depth(depth: int, model_id: str = "") -> int:
    """The profiling window, bounded only by things that are real.

    UNCAPPED BY DEFAULT. The old ceiling of 16 was the last rung of the 2/4/8/16 ladder this path
    replaced; no code derives it from any profiler capacity. A marker overflow is handled where it
    happens: profiler_heal patches the TT_FATAL into a warning so the run yields a PARTIAL report,
    and _detect_partial_capture flags that capture as partial so it is never read as complete.
    Capping below what the ops need does not make the profile safe, it makes it BLIND --
    gemma-3-12b-it was handed a 16-layer window with 54 op types "present in full model, un-timed",
    any of which could hold the next bottleneck.

    Two bounds remain, both meaningful:
      * the model's DECLARED depth -- profiling deeper than the model is nonsense, not caution;
      * PERF_MCP_COV_MAX_DEPTH -- an explicit opt-in ceiling for a box that really does overflow,
        so the limit is a decision someone made rather than a constant nobody remembers choosing.
    """
    d = max(int(depth), 2)
    full = _declared_depth(None, model_id) if model_id else None
    if full:
        d = min(d, int(full))
    raw = (os.environ.get("PERF_MCP_COV_MAX_DEPTH") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        d = min(d, int(raw))
    return d


def _coverage_layers(
    repo_root: Path,
    mcp_env: dict,
    devices: str,
    node,
    case,
    n_layers: int = 52,
    model_name: str = "",
    config_ref: str = "",
    depth_knob=None,
):
    """MODEL-AGNOSTIC profiling-window sizing. One all-layers probe (TT_PERF_LAYERS=0, no tracy)
    enumerates EVERY distinct op across all layers (overflow-safe: host-side op wrapping, no marker
    buffer) and, via its per-block signposts, the block each op first appears in. The tracy timing window
    is the smallest depth that still holds a fresh instance of every op, capped at 16 (the marker limit);
    ops that first appear past 16 are reported as present-but-un-timed. Falls back to the config-declared
    layer pattern when the k=0 probe yields nothing (a model that reads TT_PERF_LAYERS=0 as an empty
    stack). Cached per model. Disable via PERF_MCP_COVERAGE_SIZING=0."""
    # WHY THERE IS NO WINDOW, not merely that there is none. None is the answer for three unrelated
    # situations -- sizing switched off, the knob proved inert, the probe found nothing -- and a caller
    # handed a bare None cannot tell "profile everything, deliberately" from "something broke". That
    # ambiguity is what let before_loop read a deliberate None as a failure and invent a depth of 4,
    # export it as TT_PERF_LAYERS, and announce a 4-layer profile on a run that profiled 48. The
    # reason was known HERE and thrown away on the way out; facts already travels, so it carries it.
    facts: dict = {}
    if os.environ.get("PERF_MCP_COVERAGE_SIZING", "1") != "1" or not node:
        facts["no_window"] = "sizing_disabled" if node else "no_node"
        return None, facts
    cached = _coverage_cache_get(repo_root, node, case)
    if cached is not None:
        print(f"  [optimize/cc] coverage (cached): TT_PERF_LAYERS={cached}")
        return cached, facts
    sigs, raw, seq = _run_op_sigs(repo_root, mcp_env, devices, node, case, 0)
    if sigs:
        facts = _facts_from(raw, sigs, seq)
        # A STACK THE MODEL RUNS AND THE WALK CANNOT SEE -- REPAIRED HERE, BEFORE ANYTHING IS SIZED.
        #
        # This runs ahead of every check below because all of them read the walk's output: the
        # empty-walk refusal, the declared-sections comparison, the depth repair and the coverage
        # sizing. A model whose sections are half-visible passes each of them on the visible part
        # alone -- markers emitted, coverage measured, a depth sized for whichever section happened
        # to be readable, and the rest built at FULL depth with no error anywhere. Repairing first
        # means those checks see the model as it actually is.
        #
        # The probe's census carries both kinds of stack, so the discrepancy needs no naming rule and
        # no per-model code; the re-walk below is a real probe, which is what makes the verdict
        # evidence rather than the agent's word.
        _root = _model_root_from_node(repo_root, node)
        _last = {}

        def _rewalk():
            s2, r2, q2 = _run_op_sigs(repo_root, mcp_env, devices, node, case, 0)
            if s2:
                _last.update(sigs=s2, raw=r2, seq=q2)
            return _stack_census(r2)

        make_stacks_visible(_root, _stack_census(raw), rewalk=_rewalk, evidence=_stack_evidence(_root, seq))
        if _last:
            # Every number below is read off the walk, so they come from the REPAIRED model or the
            # repair would be invisible to the run that asked for it.
            sigs, raw, seq = _last["sigs"], _last["raw"], _last["seq"]
            facts = _facts_from(raw, sigs, seq)
        # SAY IT WHEN THE MODEL'S BLOCKS COULD NOT BE FOUND. The probe walks the object the factory
        # returned and tags every repeated stack the device runs; zero blocks means that walk found
        # nothing, and everything downstream degrades quietly -- the ladder is climbed instead of
        # read (four extra device probes, each reloading the weights), one depth is inferred for a
        # model that may have several sections, and per-block attribution is unavailable for the
        # whole run. Voxtral-Mini-3B ran that way for a full day without the reason ever being
        # printed: full_blocks=0 appeared only inside a debug line nobody reads unless they already
        # suspect it.
        #
        # Structural, not a name check: the walk accepts a list of same-typed callables or a hybrid
        # sharing a base, so this fires for any model whose blocks are held in a shape it cannot see,
        # whatever those classes are called.
        if not facts["full_blocks"]:
            # REFUSE, DO NOT RECOMMEND. Printing a suggestion here is what let Voxtral run for a day
            # with full_blocks=0 buried in a debug line: the ladder was climbed instead of read (four
            # device probes, each reloading weights), ONE depth was inferred for a three-section
            # model, and per-block attribution was unavailable for every round. Every downstream
            # number was still produced, which is what makes it dangerous -- the run looks like it
            # worked.
            _sections = _declared_sections(_model_root_from_node(repo_root, node), model_name or config_ref)
            _msg = (
                "the built model exposes NO discoverable block stacks: depth can only be inferred "
                "and per-block attribution is impossible"
            )
            if len(_sections) > 1:
                _msg += "; the config declares %d sections (depths %s), so %s" % (
                    len(_sections),
                    ", ".join(str(d) for d in _sections[:4]),
                    "each needs to be reachable as its own stack",
                )
            _root = _model_root_from_node(repo_root, node)
            _emitted = _is_emitted_model(_root)
            print("  [optimize/cc] %s: %s." % ("REFUSING" if _emitted else "WARNING", _msg), flush=True)
            print(
                "  [optimize/cc] Hold each repeated stack as a list of same-typed blocks, or give "
                "differing per-layer wrappers a common base, so the walk can see them.",
                flush=True,
            )
            # ENFORCED FOR MODELS THE TOOL WROTE, reported for the ones it did not.
            #
            # emit-e2e's spec requires every repeated stack to be discoverable, and the HF config
            # says how many sections there are to find -- so for an emitted model the two can be
            # compared and a mismatch is a defect in the tool's own output, caught before a run
            # spends hours on it. Voxtral shipped with none discoverable and nobody knew for a day.
            #
            # Hand-written tt-metal models legitimately
            # have no discoverable stacks and are measured through the ladder -- gemma3 and
            # llama3_1_8b_p150 both are, and blocking this path refuses them outright (proved by
            # test_coverage_source_order, which exercises exactly that shape). The contract's own
            # rule applies: only a COMPATIBILITY defect may block, and it must block BEFORE the
            # device is touched. So enforcement lives in the depth-knob clause, which fails a
            # factory that cannot accept a depth at all; what is left here is a loud, specific
            # report at the moment the walk comes back empty.
            if _emitted and os.environ.get("PERF_MCP_ALLOW_NO_STACKS") != "1":
                raise SystemExit(EXIT_REFUSED)
        else:
            # FOUND SOME -- BUT THE CONFIG SAYS HOW MANY THERE SHOULD BE. A model can expose one
            # discoverable stack and hide the rest, which reads as success everywhere: markers are
            # emitted, coverage is measured, and the depth is sized for the section that happened to
            # be visible. Voxtral did exactly this before its wrappers shared a base -- one encoder
            # stack reporting, the other silent, one depth applied to both.
            #
            # The HF config is the independent witness: it declares a depth per section, already
            # parsed by transformers, needing no device, no markers and no naming convention. Fewer
            # stacks than declared sections means structure is hidden, and for a model the tool
            # itself generated that is a defect in its output rather than a property to work around.
            _root = _model_root_from_node(repo_root, node)
            _sections = _declared_sections(_root, model_name or config_ref)
            _seen = len(_stack_ids_from_seq(seq))
            if len(_sections) > 1 and _seen < len(_sections):
                print(
                    "  [optimize/cc] %s: the config declares %d sections (depths %s) but only %d "
                    "block stack(s) are discoverable -- the rest cannot be sized, capped or "
                    "attributed."
                    % (
                        "REFUSING" if _is_emitted_model(_root) else "WARNING",
                        len(_sections),
                        ", ".join(str(d) for d in _sections[:4]),
                        _seen,
                    ),
                    flush=True,
                )
                if _is_emitted_model(_root) and os.environ.get("PERF_MCP_ALLOW_NO_STACKS") != "1":
                    raise SystemExit(EXIT_REFUSED)
            # ONE KNOB PER STACK, ADDED BY THE AGENT THAT IS ALREADY HERE.
            #
            # A factory with a single depth argument forces every stack to one value -- or worse, the
            # value reaches ONE stack and the rest build at FULL depth, which is not a tidiness issue
            # but the entire cost. Voxtral-Mini-3B profiled 18729 ops and 35.2M tracy zones that way
            # against 2471 once every stack was capped, and its baseline was killed at the budget
            # leaving the run with no BEFORE number at all.
            #
            # Detection alone would only move the manual work earlier, so the run repairs it. The
            # optimize loop already edits model source for every lever it tries, under a PCC gate
            # that reverts anything breaking correctness, and the walk has just produced the stack
            # paths so nothing is guessed. Whether the new knobs actually CAP is settled immediately
            # below by the depth bridge, which caps and re-measures the work signal and reports INERT
            # when the op count does not move -- the one check an edit cannot talk its way past.
            # ONE PLACE DECIDES THIS, and it is the INERT verdict. An earlier version repaired
            # here, off the walk's result -- which never fired, because a run that cannot cap reports
            # INERT and leaves discovery before reaching this branch. What survives here is the
            # per-stage refinement: the walk knows how many stacks there are, and make_model_cappable
            # uses the execution-order mapping to name the overrides that go with them.
            try:
                _fix = make_model_cappable(_root, seq=seq, n_stacks=_seen)
                if _fix.get("needed") and not _fix.get("added"):
                    print(
                        "  [optimize/cc] %d stacks; still missing %s after the repair attempt"
                        % (_seen, ", ".join(_fix["needed"])),
                        flush=True,
                    )
            except Exception:  # noqa: BLE001 -- a repair that cannot run leaves the model as it was
                pass
        # SIGNPOSTS BEFORE THE LADDER. Both answer the same question -- what depth still contains
        # every op type -- but at very different prices. Signposts are already paid for: the k=0
        # probe above emitted them, so reading them costs nothing. The ladder REBUILDS the model at
        # 2, 4, 8 and 16, up to four extra device probes, each reloading the weights. Running the
        # expensive one first and the free one only as its fallback was backwards.
        # WHICH SUBTREE EACH STAGE RUNS -- published HERE, not inside the signpost branch below.
        #
        # It was written there, beside the per-stage depths, because both are "things learned from
        # the probe". They are not: the depths need the signpost sequence, and this needs only the
        # model root and the generated test. Voxtral has no tracy signposts -- "WARN signpost: no
        # tracy signposts in .../tests -- using default 'start'/'stop'" -- so _signposts_usable(seq)
        # is False, the whole branch is skipped, and stage_roots was never even attempted. Measured
        # run 6, 2026-08-17: stage_roots absent from the facts, encode priced at "not modelled".
        #
        # Called unconditionally and early, so a mapping that does not depend on the probe does not
        # inherit the probe's preconditions.
        _publish_stage_roots(seq, _root, node)
        _signpost = None
        if _signposts_usable(seq):
            per_stack_map, _ = _first_block_map(seq)
            # Compute per-stack coverage depth. per_stack_map is {stack_id: {op: block_idx}}.
            # For single-stack models this is {"stack0": {...}}.
            _per_stack_cov: dict = {}
            _per_stack_deep: dict = {}
            _declared = _declared_depth(_model_root_from_node(repo_root, node), model_name or config_ref)
            _signpost_ok = True
            for _sid, _fb in per_stack_map.items():
                _deepest = max(_fb.values()) if _fb else 0
                _cov_s = _cap_cov_depth(max(_deepest + 1, 2), model_name or config_ref)
                _ok, _why = _validate_signpost_window(_cov_s, facts.get("full_blocks") or 0, _declared)
                if not _ok:
                    print(f"  [optimize/cc] signpost window REJECTED ({_sid}): {_why}; falling back to the ladder")
                    _signpost_ok = False
                    break
                _per_stack_cov[_sid] = _cov_s
                _per_stack_deep[_sid] = sorted(op for op, b in _fb.items() if b >= _cov_s)
            if _signpost_ok and _per_stack_cov:
                # PRINT THE PER-STACK NUMBERS, not just the value that survives.
                #
                # These are the inputs to every depth decision the run makes, and only max() reached
                # the log -- so a run that profiled full depth said "capping to 32 left the work
                # signal unchanged" without ever showing WHICH stack asked for 32. On Voxtral that
                # hid the whole story: the encoders may saturate in a couple of blocks while the
                # decoder's graduated stubs sit at 28..31 and drag the maximum to the full model.
                # Deciding what to fix required a number nobody could see.
                _per_stage = depth_per_stage(_per_stack_cov, seq)
                # PUBLISHED, NOT JUST PRINTED. This is the mapping that lets the bridge set the
                # variables the generated test actually reads (TT_PERF_<STAGE>_LAYERS) and the repair
                # actually creates (`<stage>_layers`). It used to reach the log and nothing else.
                facts["per_stage"] = dict(_per_stage)
                # AND WHICH SUBTREE EACH STAGE RUNS. Same publication, same reason: the mapping was
                # derivable and reached nobody, so the roofline priced every stage from one
                # whole-model byte count -- overcharging the backbone for a tower it never reads and
                # leaving that tower with a measurement and no ceiling. Written into the model's own
                # facts file, which is where the report reads facts from.
                print(
                    "  [optimize/cc] coverage per stack: %s%s"
                    % (
                        ", ".join("%s=%s" % (k, v) for k, v in sorted(_per_stack_cov.items())),
                        (" | per stage: " + ", ".join("%s=%s" % (k, v) for k, v in sorted(_per_stage.items())))
                        if _per_stage
                        else " | no stage boundaries -> one uniform depth",
                    ),
                    flush=True,
                )
                # NO SECOND VERIFICATION HERE. The depth bridge already applies the caps and
                # measures: "did not reduce work ... ignoring" declines to enforce and profiles full
                # depth, which is the same protection this used to claim -- against gemma3, whose
                # perf test reads TT_PERF_LAYERS and whose builder drops it.
                #
                # This copy was worse in three ways. It probed at max(per-stack depths), which on a
                # model whose deepest stack IS the model (Voxtral: encoder 32 of 32) asks for FULL
                # depth, so the work signal could not move and the knob was declared dead -- a false
                # INERT that discarded a correct window (stack0=2, stack2=32, stack3=3) and refused
                # the run. It also threw away the WHOLE window rather than just declining to enforce
                # it, and it spent an extra device probe to do so. One decision, measured once, in
                # the place that applies it.
                _signpost = (_per_stack_cov, _per_stack_deep)
        if _signpost is not None:
            _cov_dict, _deep_dict = _signpost
            blk_source = "signposts"
            # Flatten deep ops across stacks for facts/reporting
            deep = sorted({op for ops in _deep_dict.values() for op in ops})
        else:
            measured = _measure_cov(
                repo_root,
                mcp_env,
                devices,
                node,
                case,
                sigs,
                _model_root_from_node(repo_root, node),
                base_knob=depth_knob,
                full_signal=facts["full_signal"],
            )
            # Discover all stack IDs present in the k=0 probe sequence so that every
            # stack receives a depth cap -- not just stack0. For single-stack models
            # this returns ["stack0"] and the behaviour is identical to before.
            _all_stacks = _stack_ids_from_seq(seq)
            if measured is not None:
                _cov_scalar, deep, blk_source = measured
            else:
                deep = []
                _cov_scalar = 2
                blk_source = "unverified-floor"
            _cov_dict = {sid: _cov_scalar for sid in _all_stacks}
        facts["deep_ops"] = deep
        tail = f"; {len(deep)} op-type(s) still absent at max depth (present in full model, un-timed)" if deep else ""
        _cov_repr = _cov_dict if len(_cov_dict) > 1 else next(iter(_cov_dict.values()))
        print(
            f"  [optimize/cc] coverage ({blk_source}): {len(sigs)} distinct op(s) -> TT_PERF_LAYERS={_cov_repr}{tail}"
        )
        # Cache stores the maximum depth across stacks (an int); the full per-stack dict is
        # reconstructed on the live path and not preserved in the on-disk cache.
        _cache_val = max(_cov_dict.values()) if isinstance(_cov_repr, dict) else _cov_repr
        _coverage_cache_put(repo_root, node, case, _cache_val)
        return _cov_dict, facts
    k, n_kinds = _config_layer_kinds(config_ref or model_name)
    if k is None:
        # THE CHECKPOINT COUNTS THE KINDS THE CONFIG WAS GUESSED FOR. _config_layer_kinds reads a
        # per-layer pattern out of one of four attribute names and needs AutoConfig to load the
        # model at all -- which it cannot for voxtral, whose model type this transformers does not
        # know. Both fell through and the run ended here with "no_window: probe_failed".
        #
        # Two blocks are the same kind when they hold the same set of parameter names. That is
        # visible in the checkpoint without a vocabulary, an import, or a device.
        try:
            from agent.checkpoint_sections import layer_kinds as _ck_kinds

            k, n_kinds = _ck_kinds(_model_root_from_node(repo_root, node), model_name or config_ref)
            if k is not None:
                print(
                    "  [optimize/cc] coverage (checkpoint fallback; k=0 probe empty, config silent): "
                    "%d kind(s) -> TT_PERF_LAYERS=%d" % (n_kinds, min(k, 16))
                )
        except Exception:  # noqa: BLE001 -- no checkpoint is the old "probe_failed" path
            k, n_kinds = None, 0
    if k is not None:
        _cov = min(k, 16)
        print(
            f"  [optimize/cc] coverage (config fallback; k=0 probe empty): {n_kinds} kind(s), deepest first "
            f"appears at layer {k - 1} -> TT_PERF_LAYERS={_cov}"
        )
        _coverage_cache_put(repo_root, node, case, _cov)
        return {"stack0": _cov}, facts
    facts["no_window"] = "probe_failed"
    return None, facts


def _print_scorecard(
    devices: str, manifest: dict, pipe: dict, facts: dict, before_ms, after_ms, model_name: str = ""
) -> None:
    """End-of-run scorecard. UNIVERSAL fields (hardware, TP/DP, fully-on-device, batch, users) print for
    ANY model; token-throughput fields (TTFT / T/S/U / T/S / ISL / OSL) are class-specific and print only
    when the model is autoregressive AND fully on-device, else N/A with the reason. Best-effort, never fails."""
    try:
        env = (manifest or {}).get("env", {}) or {}
        arch = env.get("arch") or "?"
        chips = env.get("device_count") or env.get("mesh_chips") or _chip_count(devices)
        dp, tp = facts.get("dp", 1), facts.get("tp", 1)
        host_ops = facts.get("host_ops", [])
        probed = bool(facts) and facts.get("n_op_types", 0) > 0
        on_device = probed and not host_ops
        batch = int(os.environ.get("TT_PERF_BATCH", "1") or "1")
        isl = os.environ.get("TT_PERF_SEQ_LEN") or "(default)"
        # Same fallback the skeleton uses, so the scorecard reports the OSL that RAN. "4" here printed
        # OSL=4 on a run measuring 128 whenever the variable was unset.
        osl = os.environ.get("TT_PERF_OSL_TOKENS", "128")
        L = ["  ┌─ optimize scorecard — pipeline: %s" % pipe.get("task", "?")]
        L.append("  │ hardware          : %s  x%s chip(s)" % (arch, chips))
        if facts.get("parallelism_known"):
            L.append(
                "  │ parallelism       : TP=%s x DP=%s  (%s)"
                % (tp, dp, "sharded mesh" if facts.get("shard_active") else "single-chip / replicated")
            )
        else:
            L.append("  │ parallelism       : UNKNOWN  (no TP/DP line in the probe output — not assumed 1x1)")
        if not probed:
            L.append("  │ fully on device   : UNKNOWN  (op-coverage probe did not run)")
        elif on_device:
            L.append(
                "  │ fully on device   : YES  (no host-transfer op among the %d known signatures)" % len(_HOST_XFER_OPS)
            )
        else:
            L.append(
                "  │ fully on device   : NO   -> full-device trace blocked; host round-trips: %s" % ", ".join(host_ops)
            )
        L.append("  │ batch / users     : %s" % batch)
        reason = (
            "probe did not run"
            if not probed
            else ("not fully on-device" if not on_device else "needs a trace-capturable decode step")
        )
        for name in ("TTFT", "T/S/U", "T/S"):
            L.append("  │ %-16s : N/A  (%s)" % (name, reason))
        L.append("  │ ISL / OSL         : %s / %s  (tokens; N/A for non-token models)" % (isl, osl))
        # The condition a step/vision measurement is meaningless without. Printed only when the model
        # HAS one -- a resolution line on a text model would state a condition that does not exist.
        _res = os.environ.get("TT_PERF_RESOLUTION") or (facts or {}).get("resolution")
        if _res:
            L.append("  │ resolution        : %sx%s  (px per unit of work)" % (_res, _res))
        if before_ms and after_ms:
            d = (before_ms - after_ms) / before_ms * 100.0
            L.append("  │ full-model e2e    : %.1f -> %.1f ms  (%+.1f%%)" % (before_ms, after_ms, d))
        L.append("  └─")
        print("\n".join(L))
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] scorecard skipped ({exc})")
    try:
        # _LAST_SCORECARD is filled by the BEFORE bookend only (_fullpipe_e2e runs once), so these
        # are pre-optimization numbers. Label them rather than presenting them as the run's result.
        if _LAST_SCORECARD.get("TTFT_ms") or _LAST_SCORECARD.get("TSU"):
            import scorecard_profiles as _sp

            _env = (manifest or {}).get("env", {}) or {}
            _arch = _env.get("arch") or "?"
            _chips = _env.get("device_count") or _env.get("mesh_chips") or _chip_count(devices)
            _meas = {}
            for _k in ("TTFT_ms", "TSU", "TS", "ISL", "OSL", "batch"):
                _v = _LAST_SCORECARD.get(_k)
                if _v is None:
                    continue
                try:
                    _meas[_k] = float(_v) if _k in ("TTFT_ms", "TSU", "TS") else int(float(_v))
                except Exception:  # noqa: BLE001
                    _meas[_k] = _v
            _mid = (manifest or {}).get("model_id") or model_name or pipe.get("task", "")
            print("  (throughput card below is the BASELINE bookend — pre-optimization, not the run result)")
            print(_sp.render(_mid, _arch, _chips, _meas))
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] model_targets card skipped ({exc})")


_GIT_LAST_ERROR: dict = {}


def _git(repo_root: Path, *args: str) -> str:
    """stdout of a git command, or "" on failure.

    "" used to be indistinguishable from a real empty result AND from a non-zero exit, so a git
    failure looked like "no progress": _progress_token could not advance and a productive round was
    killed as unproductive, while the commit/revert paths still reported success. The failure is now
    recorded so callers can tell the two apart (see _git_ok).
    """
    try:
        r = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            _GIT_LAST_ERROR["err"] = "git %s -> rc=%d: %s" % (
                " ".join(args),
                r.returncode,
                (r.stderr or "").strip()[:200],
            )
            return ""
        _GIT_LAST_ERROR.pop("err", None)
        return (r.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        _GIT_LAST_ERROR["err"] = "git %s raised: %s" % (" ".join(args), str(exc)[:200])
        return ""


def _git_ok() -> bool:
    """Did the most recent _git call succeed?"""
    return "err" not in _GIT_LAST_ERROR


def _git_last_error() -> str:
    return _GIT_LAST_ERROR.get("err", "")


# chip-index -> its board's PCI-resettable local chip, snapshotted while healthy. RESET PATH ONLY --
# nothing about mesh-open / parallelism / scorecard reads any of this; it exists solely to pick
# `tt-smi -r` targets so a whole n300 board resets (never half a board, never a non-PCIe remote chip).
def _board_map_file():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_board_topology.json"


def _read_board_topology() -> dict | None:
    """Delegates to the shared primitive so the reset map has exactly one implementation."""
    return _dr().read_board_topology()


def _capture_board_topology() -> None:
    """Persist the reset map while the board is HEALTHY (startup) so the reset path has a trustworthy
    map even if the board later wedges. Best-effort; reset falls back to a live read if the file's gone.
    RESET PATH ONLY -- captured here, consumed only by _board_reset_targets."""
    m = _read_board_topology()
    if m:
        try:
            _board_map_file().write_text(json.dumps(m))
        except Exception:  # noqa: BLE001
            pass


def _board_reset_targets(chip_ids: list[int]) -> str | None:
    """Map logical chip ids -> the PCI-resettable chips of each board they live on. Delegates to the
    shared primitive; kept as a name because the reset paths and their tests refer to it."""
    return _dr().expand_to_boards(chip_ids)


def _reset_chip_list(devices: str) -> str:
    """BOARD-AWARE reset target derived from --devices: explicit ids / 'single' -> the WHOLE board(s)
    they live on (every PCI-resettable chip of those boards, so both p300c ASICs reset, never half a
    board, never other boards). '' when a per-board target can't be determined (all/empty devices, or
    topology unavailable) so the caller falls back to its full-enumerated reset -- never a partial
    subset (which wedges device-open)."""
    d = (devices or "").strip().lower()
    if d in ("all", ""):
        return ""
    if d == "single":
        req = [0]
    else:
        req = [int(x) for x in d.split(",") if x.strip().isdigit()]
    if not req:
        return ""
    return _board_reset_targets(req) or ""


def _reset_devices(devices: str) -> str:
    """tt-smi reset the visible chips to recover a wedged fabric. Best-effort; returns a status string.

    GALAXY-AWARE and UNIFIED with the profiler-layer reset (agent.probes._reset_arg_sets): a Galaxy host
    uses -glx_reset (a plain `-r` does NOT reset a Galaxy), a plain board uses `-r`, and the
    TT_HW_PLANNER_RESET_ARGS / TT_HW_PLANNER_GALAXY overrides are honored -- previously this path
    hard-coded `-r` and ignored all of that. For 'all'/'' the plain reset stays BARE `tt-smi -r` (resets
    EVERY chip): the enumerated count comes from tt-smi -s / ttnn and a stale value would reset only
    chip 0, leaving a multi-chip ETH fabric half-reset (heartbeat-stuck wedge). Explicit/single ids
    target exactly those chips."""
    d = (devices or "").strip().lower()
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    if not Path(tt_smi).is_file():
        return "device reset SKIPPED (tt-smi not found)"
    try:
        import agent.probes as _pr  # galaxy-aware reset invocations (single source of truth)

        if _pr._GALAXY_HOST is None and not os.environ.get("TT_HW_PLANNER_GALAXY"):
            try:
                _pr.note_board(tt_smi=tt_smi)  # one-time galaxy capability probe (cheap on plain boards)
            except Exception:  # noqa: BLE001
                pass
        arg_sets = _pr._reset_arg_sets()
    except Exception:  # noqa: BLE001
        arg_sets = [["-r"]]
    chips = _reset_chip_list(devices) if d not in ("all", "") else ""
    last = "no reset ran"
    for args in arg_sets:
        cmd = [tt_smi, "-r", chips] if (chips and args and args[0] == "-r") else [tt_smi, *args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
            last = "tt-smi %s rc=%d" % (" ".join(cmd[1:]), r.returncode)
            if r.returncode == 0:
                return last
        except Exception as exc:  # noqa: BLE001
            last = "tt-smi %s FAILED (%s)" % (" ".join(cmd[1:]), exc)
    return "device reset (%s)" % last


def _dr():
    """The shared device-recovery primitive (agent/device_recovery.py), imported lazily and by path
    because run.py is itself loaded by path from perf_mcp/optimize with a bare sys.path."""
    global _DR_MOD
    try:
        return _DR_MOD
    except NameError:
        pass
    try:
        from agent import device_recovery as _m
    except Exception:  # noqa: BLE001
        import importlib.util as _ilu

        _p = Path(__file__).resolve().parents[1] / "agent" / "device_recovery.py"
        _spec = _ilu.spec_from_file_location("tt_device_recovery", str(_p))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
    globals()["_DR_MOD"] = _m
    return _m


def _reclaim_device(devices: str, error_text: str = "", after_kill: bool = False) -> str:
    """UNIVERSAL device reclaim used at EVERY recovery point: kill every process holding
    /dev/tenstorrent (except this process + its ancestors, so the supervisor/self is never killed),
    then tt-smi -r the chips. A wedge is cleared no matter WHO holds the device -- a stray child, a
    hung profiler, a busy pytest, or a leaked resident mesh. The one holder this cannot kill is the
    caller's own tree; an orchestrator self-hold is handled by exiting to the supervisor, which then
    reclaims from outside.

    ``error_text`` is the failure output, and it is the TARGET EVIDENCE: the runtime names the chip
    that died ("Read 0xffffffff over PCIe ID 3"). This used to reset whatever ``devices`` implied --
    `single` -> chip 0 -> board 0,1 -- which is INTENT, not PLACEMENT, so a mesh placed on chip 3
    was never reset while the healthy board was reset repeatedly for eleven hours. The reset is now
    routed through the shared primitive, so it picks the target from evidence, VERIFIES the device
    came back, and spends the same escalation budget as every other reset in the tool."""
    _dr_mod = _dr()
    # The reap lives in the recovery primitive now, so EVERY reset gets it rather than the three
    # sites that happened to call this function. Kept here too because this caller reports what it
    # killed, and `recover` below is a no-op reap by then -- the holders are already gone.
    killed = _dr_mod.reap_device_holders()
    _status = {"last": ""}

    def _issue(target):
        _status["last"] = _reset_devices(target)
        return True

    # A KILLED PROCESS HELD EVERY DEVICE, NOT THE ONE IT WAS ASKED TO USE. `devices` is what the run
    # was CONFIGURED with; a ttnn process maps all enumerated chips regardless -- a single probe was
    # observed holding /dev/tenstorrent/0,1,2,3 on a --devices 0 run. So SIGKILL can leave any of them
    # half-initialised, and scoping the reset to the configured board resets the wrong one.
    #
    # Measured 2026-08-14: run 10's full-pipeline measurement was killed at its budget, the reclaim
    # issued `tt-smi -r 0,1` (the --devices 0 board), the health check passed for THAT board and the
    # ladder never escalated to `all`. Device 2 stayed wedged -- "tenstorrent!2: Failed to set initial
    # power state: -22", repeating -- and the next run died at Step 1 because `tt-smi -s` blocked on
    # it. By then `tt-smi -r` hung too, so only a host reboot cleared it.
    #
    # After a kill the scope is UNKNOWN, and an unverifiable target must widen rather than narrow
    # (expand_spec's own rule): a reset covering too much is recoverable, one covering too little
    # leaves a chip nobody touches.
    ok = _dr_mod.recover(
        "reclaim",
        _issue,
        error_text=error_text,
        # NOT named `killed`: that local already holds reap_device_holders()'s list, and a
        # parameter by the same name is overwritten before it is read -- silently, because the
        # list is empty after a SIGKILL and an empty string is falsy.
        config_target="all" if after_kill else devices,
    )
    return "reclaimed device (killed holders %s) + %s%s" % (
        killed or "none",
        _status["last"] or "no reset issued",
        "" if ok else " [DEVICE STILL UNHEALTHY]",
    )


def _pg_cpu_jiffies(pgid: int) -> int:
    # PROBES OWNS THE /proc WALK. This was a verbatim copy of probes._pgroup_cpu_jiffies.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.probes import _pgroup_cpu_jiffies

        return _pgroup_cpu_jiffies(pgid)
    except Exception:  # noqa: BLE001
        return 0


def _hard_ceiling_mult() -> int:
    """probes owns this too -- one number, one home. 0 means "no ceiling" if probes is unreachable,
    and the caller treats 0 as disabled rather than as an instant kill."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.probes import _HARD_CEILING_MULT

        return int(_HARD_CEILING_MULT)
    except Exception:  # noqa: BLE001
        return 0


def _progress_watch(pgid, log_path=None, stall_s=0.0):
    """probes owns the arithmetic; see ProgressWatch there. Lazy import, as _set_depth is.

    The fallback answers "moved" for every poll, which is the safe direction: an unreadable
    signature must not be mistaken for a wedge and kill working code. The ceiling still bounds the
    run if the signature never becomes readable.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.probes import ProgressWatch

        return ProgressWatch(pgid, log_path, stall_s)
    except Exception:  # noqa: BLE001

        class _Blind:
            def moved(self, *_a, **_k):
                return True

        return _Blind()


def _tree_cpu_jiffies(root_pid: int) -> int:
    """Sum utime+stime over the WHOLE process TREE rooted at root_pid (every descendant, ACROSS
    process groups / sessions). The build's on-device validation is spawned with start_new_session
    (its own pgrp), so _pg_cpu_jiffies(pgid) cannot see its CPU -- and the no-output watchdog would
    then false-kill a validation that is actually pegging the device (observed on XTTS: a ~10 min
    perf-test validation killed as a wedge). Walking the tree counts that busy child as progress."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from agent.probes import _proc_stat_fields
    except Exception:  # noqa: BLE001
        return 0
    kids: dict = {}
    jiff: dict = {}
    for pid, f in _proc_stat_fields():
        if len(f) > 12:
            kids.setdefault(int(f[1]), []).append(pid)
            try:
                jiff[pid] = int(f[11]) + int(f[12])
            except ValueError:
                jiff[pid] = 0
    total, stack = jiff.get(int(root_pid), 0), [int(root_pid)]
    while stack:
        for kid in kids.get(stack.pop(), ()):
            total += jiff.get(kid, 0)
            stack.append(kid)
    return total


# Defined by perf_mcp, which emits them; named here so the watchdog can recognise a cooling child.
_COOL_BEGIN = "PERF_MCP_COOLING_BEGIN"
_COOL_END = "PERF_MCP_COOLING_END"
# A cooling child re-asserts itself every poll (perf_mcp._COOLDOWN_POLL_S, 20 s). Three missed beats
# means it is no longer cooling, whatever it last claimed, and the clock starts again.
_COOL_HEARTBEAT_S = float(os.environ.get("PERF_MCP_COOL_HEARTBEAT_S", "90"))


_LIVENESS_PROBE_S = float(os.environ.get("PERF_MCP_LIVENESS_PROBE_S", "20"))


def _device_answers() -> bool:
    """Is the board still talking? Used to decide whether a timeout deserves a reset.

    Delegates to agent.probes.device_is_responsive, which asks only whether tt-smi came back naming
    any device -- NOT the startup probe, which raises on an unrecognised board_type and would call
    this host's own `p300c` dead.

    Failure to answer returns False, which resets exactly as before -- this only ever ADDS a reason
    not to reset, never a reason to.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.probes import device_is_responsive

        return bool(device_is_responsive(_LIVENESS_PROBE_S))
    except Exception:  # noqa: BLE001 -- no answer, a slow answer, or no probe at all: reset as before
        return False


_SIBLINGS: list = []


def _siblings():
    """Load the sibling resolver itself -- the one import that cannot use the resolver.

    Four lines by path, because this module may have no package and no sys.path entry; everything
    after this point goes through siblings.load(). See cc_optimize/siblings.py.
    """
    if _SIBLINGS:
        return _SIBLINGS[0]
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("cc_optimize_siblings", str(Path(__file__).resolve().parent / "siblings.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _SIBLINGS.append(_mod)
    return _mod


def _perf_mcp():
    """perf_mcp, reachable under every load style. Delegates to cc_optimize/siblings.py."""
    return _siblings().load("perf_mcp")


_THERMAL_GATE_BROKEN = [False]


def _wait_for_thermal_headroom_before_device_work(label: str = "") -> None:
    """Let the board cool BEFORE any device subprocess, not just before a measurement.

    THE GAP THIS CLOSES, measured on a liquid-cooled p300c. The gate existed and worked, but its
    only caller was _measure_full_pipeline_guarded -- so it covered readings and nothing else. A
    Voxtral perf-test build is 30-60 minutes of continuous device work (HF weights + 17 stub
    uploads, repeated once per generator attempt) with no gate on it at all. The chips went 57C ->
    103C and the AICLK fell 1350 -> 800, and then the FIRST measurement started on a board already
    pinned to the clamp. gemma never showed this because its device time is mostly gated
    measurements, which cool between readings; the build phase is where the heat actually comes
    from.

    So the gate belongs at the point where a device process is LAUNCHED -- which is here, the one
    function every device-touching subprocess goes through -- rather than at the point where a
    number is taken. One call site, no second copy of the policy.

    BEST-EFFORT BY DESIGN. It never raises: a board whose temperature cannot be read, or one that
    stays hot past PERF_MCP_THERMAL_WAIT_S, still runs. The clamp check downstream decides whether
    the resulting reading counts. Refusing to launch would turn a hot board into a failed run,
    which is worse than a reading the ledger already knows how to reject.

    It also does NOT fix a NOC hang. Measured: the same soak died at 5.5 min at full rate and 36
    min at a quarter of it. Cooler and slower delays the hang; it does not prevent it.
    """
    try:
        # ONE TRIGGER, AND IT IS THE SAFETY CEILING. This also ran the 65C measurement gate here,
        # before every device process, waiting up to 900s for a board that idles in the sixties.
        # The operator's call: hold work only when the board is genuinely dangerous, then cool it
        # properly. A clamped reading is still caught afterwards by detect_overheat, which discards
        # it and re-measures -- measurement quality keeps a guard, and the board gains a real one.
        _perf_mcp().cool_if_over_safety_ceiling(label or "device work")
    except Exception as exc:  # noqa: BLE001 -- a gate that cannot run must not stop the work
        _warn_gate_broken(exc)


def _warn_gate_broken(exc: BaseException) -> None:
    """Say ONCE that temperature protection is not running, then let the work continue.

    "A gate that cannot run must not stop the work" was implemented as a bare `except: pass`, so a
    gate that could not run also told nobody. On 2026-08-29 that silence let the board hold 99-103C
    for an hour with every gate inert, and two chips stopped answering.
    """
    if _THERMAL_GATE_BROKEN[0]:
        return
    _THERMAL_GATE_BROKEN[0] = True
    print(
        "  [thermal-gate] WARNING: THE THERMAL GATE CANNOT RUN (%s: %s). Device work will proceed "
        "with NO temperature protection for the rest of this run." % (type(exc).__name__, str(exc)[:120]),
        file=sys.stderr,
        flush=True,
    )


_THERMAL_WATCH_REPORT_S = 300.0
_THERMAL_ABORTED = [False]
_THERMAL_ABORT_RETRIES = 2


def _thermal_watch_new() -> dict:
    """Fresh state for _thermal_watch_sample: when it last reported a crossing."""
    return {"last_report": 0.0}


def _thermal_watch_sample(state: dict, label: str = "") -> None:
    """Have the thermal owner record a board running hot WHILE a device subprocess is in flight.

    THE GAP THIS CLOSES. _wait_for_thermal_headroom_before_device_work samples once, at LAUNCH, and
    nothing samples again until the next launch. A build, a coverage probe or a long agent round holds
    the device for tens of minutes, so a board that was cool when the process started can bake for the
    whole run with nobody reading the thermometer. On 2026-08-28 the gate last fired at 20:53, all four
    chips then sat at 98-103C for fifty minutes inside one call, and chip 2 died mid-run at 21:43.

    This cannot cool -- the work is in flight and freezing a process that holds the device risks a
    wedge, which is worse than the heat. It records, so the next failure is not silent. Cooling that
    can actually happen is offered between units of work by probes.thermal_yield, and after the
    subprocess exits by the post-run call at the end of _run_device_proc.
    """
    now = time.monotonic()
    if now - float(state.get("last_report") or 0.0) < _THERMAL_WATCH_REPORT_S:
        return
    try:
        if _perf_mcp().report_board_over_clamp(label):
            state["last_report"] = now
    except Exception:  # noqa: BLE001 -- a watcher that cannot run must not stop the work
        return


def timed_op_for(env, fallback: str = "profile") -> str:
    """DEPRECATED -- kept only so an external caller does not break. Do not route by this.

    "Uncapped" is NOT the same question as "which operation". The coverage probe removes the cap on
    purpose (set_depth(env, 0)) to see every layer, and it is CHEAP -- one forward at OSL=1, 80-135 s.
    Routing on the cap therefore filed those probes in the same bucket as the full-pipeline
    measurement: [135.3, 80.1, 453.9], p95 135.3, budget 3 x 135 = 406 s for a run that needs ~1700 s.
    That is worse than the capped bucket it replaced, which at least gave 1686 s.

    An operation is named by what it IS, at the one call site that performs it.


    THE BUCKET WAS A TYPED STRING AND THE RUNS WERE NOT THE SAME SIZE. Every device subprocess filed
    its duration under `observe_op="profile"`, so a 25 s capped probe and a 2144 s uncapped
    full-model measurement shared one history -- and adaptive_timer budgets an op at 6x the p95 of its
    own bucket.

    Measured on Voxtral 2026-08-14, the bucket held
    [25.0, 55.0, 85.1, 100.2, 150.5, 281.1, 1734.2, 2143.9]: the first six are capped 2-layer runs,
    the last two uncapped. Before those last two existed the p95 was 281.1, so the budget was
    6 x 281.1 = 1686 s -- and the uncapped run, which needs 1734 s, was killed about fifty seconds
    short. It pollutes the other direction too: with 2143.9 now in the bucket, the next capped probe
    is budgeted ~12800 s, so a genuinely hung 25 s probe would sit for three hours.

    The distinguishing fact is already in hand: a capped run carries TT_PERF_LAYERS in the environment
    being launched, and an uncapped one expresses "all layers" by its ABSENCE. Reading that removes
    the hardcode instead of adding another, and no call site can mislabel a run again.
    """
    try:
        capped = bool(_active_depth_caps(env or {}))
    except Exception:  # noqa: BLE001
        capped = True
    return fallback if capped else "fullpipe"


def _run_device_step(*args, **kwargs):
    """_run_device_proc, retried when the board got too hot to keep going.

    THE STEP IS RE-RUN, NOT THE RUN. Killing the whole optimize run over a temperature is a heavy
    answer to a recoverable problem: the round's ledger, its baseline and its best-so-far are all
    still good, and the board only needs a few minutes. Ending the CHILD releases the device (the
    driver cleans up, observed every time a run was killed today), the launch gate then cools to
    _COOL_BACK_TO_C before the relaunch, and the caller sees one slower step instead of a failure.

    Bounded, because a board that reaches the abort line three times running is not going to be
    fixed by a fourth attempt -- at that point the last result is returned and the ordinary failure
    handling upstream takes it from there.
    """
    for attempt in range(1 + _THERMAL_ABORT_RETRIES):
        _THERMAL_ABORTED[0] = False
        result = _run_device_proc(*args, **kwargs)
        if not _THERMAL_ABORTED[0]:
            return result
        if attempt >= _THERMAL_ABORT_RETRIES:
            print(
                "  [thermal-abort] the board reached the abort limit on every attempt (%d); "
                "returning the last result rather than retrying forever" % (attempt + 1),
                file=sys.stderr,
                flush=True,
            )
            return result
        print(
            "  [thermal-abort] retrying that step from a cool board (attempt %d of %d)"
            % (attempt + 2, _THERMAL_ABORT_RETRIES + 1),
            file=sys.stderr,
            flush=True,
        )
    return result


def _run_device_proc(
    cmd,
    cwd,
    env,
    devices: str,
    timeout_s: int,
    label: str = "",
    reset_on_timeout: bool = True,
    capture: bool = True,
    stall_s: int = 0,
    observe_op: str = "",
    observe_root: Path | None = None,
):
    """Run a DEVICE-touching subprocess so a device wedge can never hang the tool forever. Own session +
    hard timeout; on timeout SIGKILL the WHOLE process group + _reclaim_device (kill any holder + tt-smi
    -r); AND reap the group on every exit so no stale holder survives to wedge the next op. Returns (rc,
    combined stdout+stderr); rc is None when it timed out / was killed.

    Recovery-timeout tiers (all env-overridable, stall-detector on no-progress + absolute backstop):
      BUILD   discover                                          -> PERF_MCP_DISCOVER_STALL_SEC (1200s), backstop PERF_MCP_DISCOVER_TIMEOUT (10800s)
      MEASURE gate / coverage / op-sig / full-pipeline runs     -> PERF_MCP_MEASURE_STALL_SEC  (600s),  backstop PERF_MCP_MEASURE_BACKSTOP (3600s)
      ROUND   agent round                                       -> PERF_MCP_ROUND_STALL_SEC   (600s)"""
    _wait_for_thermal_headroom_before_device_work(label)
    _obs_t0 = time.monotonic()
    _therm = _thermal_watch_new()
    _piped = bool(capture or stall_s)
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE if _piped else None,
        stderr=subprocess.STDOUT if _piped else None,
        text=True if _piped else None,
        start_new_session=True,
    )
    rc, out = None, ""
    try:
        if stall_s:
            import sys as _sys
            import threading as _th

            _buf: list = []
            _act = [time.monotonic()]
            # COOLING IS NOT WORK, AND IT IS NOT A HANG. A thermal wait runs inside this subprocess,
            # so on 2026-08-14 the tool cooled the board, the cooling ate the wall-clock budget, and
            # this watchdog killed it at 1716 s as "likely a device wedge" -- then reset the device,
            # which is what actually broke it. The child brackets every thermal wait with these
            # markers; time spent between them is credited back, so a board that takes an hour to
            # cool costs an hour of waiting and none of the op's budget.
            # CREDIT IS EARNED PER HEARTBEAT, NEVER EXTRAPOLATED. The first version of this trusted
            # a single BEGIN and credited every second until END arrived, which handed back the one
            # protection the absolute cap exists to provide: a child that printed BEGIN and then
            # busy-wait deadlocked would accrue credit as fast as wall clock, so the cap could never
            # fire -- and the stall detector was told to ignore it too. Voxtral produced exactly that
            # shape once before: 85 minutes, 91 minutes of CPU, no output after the first second.
            # The child now re-asserts cooling every poll, and a gap longer than _COOL_HEARTBEAT_S
            # earns nothing, so credit stops the moment the child does.
            _cool = {"in": False, "last": None, "total": 0.0}

            def _cool_beat():
                now = time.monotonic()
                prev = _cool["last"]
                if prev is not None and now - prev <= _COOL_HEARTBEAT_S:
                    _cool["total"] += now - prev
                _cool["last"], _cool["in"] = now, True

            def _cool_total():
                return _cool["total"]

            def _cooling_now():
                last = _cool["last"]
                return bool(_cool["in"] and last is not None and time.monotonic() - last <= _COOL_HEARTBEAT_S)

            def _pump():
                try:
                    for _ln in proc.stdout:
                        _buf.append(_ln)
                        if _COOL_BEGIN in _ln:
                            _cool_beat()
                        elif _COOL_END in _ln:
                            _cool["in"] = False
                            _cool["last"] = None  # no open claim survives the end of the wait
                        if not capture:
                            _sys.stdout.write(_ln)
                            _sys.stdout.flush()
                        _act[0] = time.monotonic()
                except Exception:  # noqa: BLE001
                    pass

            _pt = _th.Thread(target=_pump, daemon=True)
            _pt.start()
            pgid = proc.pid
            start = time.monotonic()
            last_progress = start
            # PROGRESS, NOT ACTIVITY -- probes.progress_signature. Two things counted as life here
            # that a hang has in abundance: CPU, and the mere EXISTENCE of a child process
            # (_llm_child_alive), which no hung run can fail. Cooling stays: it is a deliberate
            # pause this tool asked for.
            _watch = _progress_watch(pgid, None, stall_s)
            max_gap = 0.0
            _over_budget = [False]
            _ceiling_mult = _hard_ceiling_mult()
            while proc.poll() is None:
                time.sleep(5)
                _thermal_watch_sample(_therm, label)
                # ABOVE THE ABORT LINE, END THE CHILD RATHER THAN WATCH IT COOK. The ceiling holds
                # work at a boundary; inside one long job there is no boundary to hold at, and the
                # board did 75C -> 95C with none available on 2026-08-29. Raising the same exception
                # the stall detector raises reuses the kill + _reclaim_device path below verbatim --
                # the child dies, the device is released, and _run_device_proc's caller retries it
                # from cool rather than the whole run starting over.
                _hot, _hot_c = _perf_mcp().board_over_abort_limit()
                if _hot:
                    _THERMAL_ABORTED[0] = True
                    print(
                        "  [thermal-abort] %s: board at %.1fC -- ending this step and re-running it "
                        "once the board is cool, rather than holding the device at this temperature"
                        % (label or "device subprocess", _hot_c if _hot_c is not None else -1.0),
                        file=sys.stderr,
                        flush=True,
                    )
                    raise subprocess.TimeoutExpired(cmd, int(time.monotonic() - start))
                now = time.monotonic()
                # A cooling child is idle ON PURPOSE: it is sleeping against a thermometer, so it
                # burns no CPU and prints only when the temperature moves. Both of this loop's
                # liveness signals read that as a wedge, which is exactly wrong.
                moved = _watch.moved(now, last_progress, proc.pid) or _act[0] > last_progress or _cooling_now()
                if moved:
                    max_gap = max(max_gap, now - last_progress)
                    last_progress = now
                limit = max(stall_s, int(3 * max_gap))
                idle = now - last_progress
                if idle >= limit:
                    print(
                        f"  [optimize/cc] {label or 'device subprocess'} STALLED (no output, syscalls, "
                        f"bytes or stack movement for "
                        f"{int(idle)}s > adaptive limit {limit}s) -- treating as wedge",
                        flush=True,
                    )
                    raise subprocess.TimeoutExpired(cmd, limit)
                # NO WALL-CLOCK KILL WHILE THE WORK IS REAL -- BUT THERE IS A CEILING.
                #
                # This raised the moment `timeout_s` elapsed, regardless of what the process was
                # doing. On 2026-08-17 that killed a full-depth measurement after three hours while
                # its tree was burning CPU and the board sat at 97-102C -- work in progress, ended
                # on a number. The number was not even a judgement about this measurement: it is the
                # CEILING, taken because --fresh had wiped the observed durations meant to size it.
                #
                # The loop already knows the difference. `moved` is log bytes, syscalls, io bytes,
                # stack movement or an in-progress cooldown, and the stall clock above kills the
                # moment all of them go quiet. A process that is demonstrably working is not a
                # wedge, and a clock cannot make it one.
                #
                # WHAT THIS COMMENT USED TO SAY, AND WHY IT WAS WRONG. It said `moved` was "tree CPU"
                # and concluded the budget should be a REPORT, not a sentence -- said once, run
                # continues, "bounded by death, not duration". Run 12 is what that costs. At 03:09,
                # exactly 3h in, this loop printed "over its 10800s budget and STILL WORKING (tree
                # CPU is moving) -- not killing it", and then sat silent for NINE HOURS holding the
                # board until it was killed by hand. CPU was moving because the process was spinning;
                # the thing being trusted as proof of work was the symptom of the wedge.
                #
                # So: the budget is still not a sentence, because the stall check above is now the
                # real judge and it watches work rather than CPU. But there IS a ceiling behind it.
                # A run that somehow keeps its signature twitching without ever finishing stops at
                # _HARD_CEILING_MULT x its budget, and stops by RAISING -- so the caller books a
                # failed attempt and its retries take over, rather than the run hanging forever.
                _worked = now - start - _cool_total()
                if not _over_budget[0] and _worked >= timeout_s:
                    _over_budget[0] = True
                    print(
                        f"  [optimize/cc] {label or 'device subprocess'} is over its {int(timeout_s)}s budget "
                        f"and STILL WORKING (output, syscalls, bytes or stack are moving) -- not killing "
                        f"it; the stall check decides, and a hard ceiling at "
                        f"{int(timeout_s * (_ceiling_mult or 0))}s is behind that",
                        flush=True,
                    )
                if _ceiling_mult and timeout_s and _worked >= timeout_s * _ceiling_mult:
                    print(
                        f"  [optimize/cc] {label or 'device subprocess'} exceeded "
                        f"{int(timeout_s * _ceiling_mult)}s -- {_ceiling_mult}x its budget -- while still "
                        f"looking busy. Nothing legitimate takes that long: killing it and failing the "
                        f"attempt so the retry can run.",
                        flush=True,
                    )
                    raise subprocess.TimeoutExpired(cmd, int(timeout_s * _ceiling_mult))
            rc = proc.returncode
            _pt.join(timeout=30)
            out = "".join(_buf)
        elif capture:
            out, _ = proc.communicate(timeout=timeout_s)
            out = out or ""
            rc = proc.returncode
        else:
            proc.wait(timeout=timeout_s)
            rc = proc.returncode
    except subprocess.TimeoutExpired as _te:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        # ASK THE BOARD BEFORE RESETTING IT. This used to reset unconditionally on every timeout, and
        # the accompanying "likely a device wedge" was a fixed string, not a finding -- nothing looked
        # at the device. On 2026-08-15 that reset four HEALTHY chips because an op ran long, and the
        # reset is what produced `Failed to set initial power state: -22`, a fault no PCIe reset
        # clears; the host needed a reboot. Runs 13, 17 and 20 all died that way.
        #
        # A timeout means SLOW. It does not mean wedged, and the difference is cheap to establish:
        # measured on this host, a live board answers tt_smi_probe() in 0.24 s and a wedged one does
        # not answer at all. Resetting a working board is not a neutral act, so it needs evidence.
        if reset_on_timeout and _device_answers():
            tail = "process group killed; device answered a liveness probe, so it was NOT reset"
        else:
            tail = (
                _reclaim_device(devices, error_text=out, after_kill=True)
                if reset_on_timeout
                else "process group killed"
            )
        _lim = int(getattr(_te, "timeout", None) or timeout_s)
        _why = "no-progress stall" if _lim < timeout_s else "hard limit"
        print(
            f"  [optimize/cc] {label or 'device subprocess'} KILLED after {_lim}s ({_why}) "
            f"(likely a device wedge / leaked mesh) -- killed the whole process group + {tail}"
        )
        return None, ""
    finally:
        # TEACH THE TIMER WHAT THIS OP COSTS -- INCLUDING WHEN IT WAS KILLED.
        #
        # `_measure_backstop` derives the hard wall from "observed PROFILE durations", and nothing
        # ever recorded one: record_observed had exactly two callers, "pcc" and "round". So
        # _op_cost("profile") stayed 0, adaptive_timer took its `cost <= 0` cold-start branch every
        # time, and every measurement on every model got the same 600 s guess forever.
        #
        # Measured on Voxtral, 2026-08-11: the baseline profile was killed at 900 s ("hard limit")
        # while still printing decode_trace_step #96. The run then optimized for hours with no
        # BEFORE number. A second attempt would have used the same 600 s, because a kill taught the
        # timer nothing -- and a kill is the single strongest piece of evidence that the budget was
        # too small.
        #
        # So it records in `finally`, on the timeout path as much as the success path, exactly as
        # _fullpipe_e2e already does for "pcc". A hard-limit kill at 900 s becomes 900 s of observed
        # cost, and the next budget is 6x that instead of a constant.
        if observe_op and observe_root is not None:
            try:
                record_observed(observe_root, observe_op, time.monotonic() - _obs_t0)
            except Exception:  # noqa: BLE001
                pass
        # Reap any lingering group member on EVERY exit. A daemon child (profiler, not-fully-closed mesh)
        # can outlive the main subprocess and keep holding the device -- a stale holder that wedges the
        # NEXT device op (observed: a completed baseline measurement leaked a holder that blocked the
        # coverage probe). Killing the whole process group here guarantees no leftover survives.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    _wait_for_thermal_headroom_before_device_work("%s (post-run cooldown)" % (label or "device work"))
    return rc, out


def _progress_token(repo_root: Path, kernel_log: str):
    """Forward-progress signal for the round watchdog: (committed HEAD, kernel-attempt-log mtime). A live
    agent advances one of these; a device-wedged agent (blocked in a measurement) advances neither."""
    try:
        mt = os.path.getmtime(kernel_log)
    except OSError:
        mt = 0.0
    return (_git(repo_root, "rev-parse", "HEAD"), mt)


def _record_wedge_to_log(kernel_log: str, reason: str) -> None:
    try:
        target = {}
        try:
            target = json.loads(Path(str(kernel_log) + ".target").read_text())
        except Exception:  # noqa: BLE001
            target = {}
        if not isinstance(target, dict):
            target = {}
        try:
            attempts = json.loads(Path(kernel_log).read_text())
        except Exception:  # noqa: BLE001
            attempts = []
        if not isinstance(attempts, list):
            attempts = []
        op = target.get("op") or "candidate config"
        kind = str(target.get("rung") or "knob").split(":")[-1] or "knob"
        attempts = [
            a
            for a in attempts
            if not (
                isinstance(a, dict) and a.get("wedged") and a.get("op_signature") == op and a.get("kernel_kind") == kind
            )
        ]
        attempts.append(
            {
                "op_signature": op,
                "kernel_kind": kind,
                "measured_ms": None,
                "beat_baseline": False,
                "note": reason,
                "stages": [],
                "kernel_detected_in_source": False,
                "wedged": True,
                "evidence": {},
                "diff": "",
            }
        )
        Path(kernel_log).write_text(json.dumps(attempts))
    except Exception:  # noqa: BLE001
        pass


def _kernel_log_path(model_name: str, task: str) -> str:
    """Where this (model, task)'s attempt log lives -- in the state dir, not hardcoded /tmp.

    PERF_MCP_STATE_DIR already moves the ledger, the gate verdicts and the full-pipeline baseline onto
    real disk; this path ignored it, so a crash took the history and left the anchors. The host went
    down mid-run on 2026-08-02, /tmp was cleared at boot, and run 20's 98 attempts -- every lever
    tried, every measurement, every recorded reason -- went with it. Rebuilding the ladder meant
    hand-transcribing the report text.

    Unset, state_dir() is tempfile.gettempdir(), so the default is exactly the old location. The
    derived .cumulative / .target / .agent.log are built as str(path) + suffix and follow it, which
    is the point: the ladder history is what a resumed run reads.
    """
    return str(state_dir() / ("cc_kernlog_%s_%s.json" % (model_name, task)))


def _fold_cumulative(kernel_log: str) -> None:
    cum = str(kernel_log) + ".cumulative"

    def _ld(p):
        try:
            v = json.loads(Path(p).read_text())
            return v if isinstance(v, list) else []
        except Exception:  # noqa: BLE001
            return []

    seen, merged = set(), []
    for a in _ld(cum) + _ld(kernel_log):
        if not isinstance(a, dict):
            continue
        k = (
            a.get("op_signature") or a.get("op_code") or "",
            a.get("kernel_kind") or "",
            (a.get("note") or "")[:200],
            bool(a.get("wedged")),
        )
        if k in seen:
            continue
        seen.add(k)
        merged.append(a)
    try:
        Path(cum).write_text(json.dumps(merged))
    except Exception:  # noqa: BLE001
        pass


def _baseline_ceiling(repo_root: Path) -> tuple[float, int]:
    ceil = 10800
    base = 0.0
    mani = _latest_manifest(repo_root / PERF_DIR)
    if mani is not None:
        try:
            cfg = json.loads(mani.read_text()).get("config", {}) or {}
            ceil = int(cfg.get("timeout", ceil) or ceil)
        except Exception:  # noqa: BLE001
            pass
        # probes owns this parse; it was written out identically in three places.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from agent.probes import observed_tracy_baseline_seconds

            base = observed_tracy_baseline_seconds(mani) or base
        except Exception:  # noqa: BLE001
            pass
    return base, ceil


# --- BUG 4 (2026-07-25): timers derive from OBSERVED durations, not absolute floors ----
# The previous form was min(ceil, max(floor, 3*base)) with base = the tracy BASELINE
# PROFILE duration. `3*base` lost to the 2400/3600 floors for every model whose baseline
# profile is under ~800 s, so the floors were the de-facto policy and adaptivity was inert.
# Measured consequences on 2026-07-25: a 3 ms ACE module was granted 3600 s (1139x its
# work, so a hang idled for an hour), while llama's round -- whose check_pcc gate alone
# runs ~1400 s -- was killed 4x at the 2400 s floor with `killed holders none`.
#
# Now: every bound is a MULTIPLE OF THE OBSERVED COST OF THE OPERATION IT GOVERNS, clamped
# by a small absolute minimum (measurement noise) and the operator ceiling (manifest
# config.timeout) -- the only operator-supplied number.
_OBS_LOG = "observed_durations.json"
_MIN_TIMER_S = float(os.environ.get("PERF_MCP_MIN_TIMER_S", "30") or "30")
# how many times an operation's own observed cost it may take before we stop waiting
# tolerance multiplier per op. "round" is 2.0 because _OP_IN_BASE_UNITS["round"]
# already estimates the WHOLE cycle (pcc + measure + commit); multiplying a
# full-cycle estimate again would compound two proxies.
_OP_MULT = {"profile": 6.0, "pcc": 6.0, "build": 6.0, "round": 2.0, "agent": 8.0, "fullpipe": 3.0}
# fallback cost of one operation, expressed in baseline-profile units, used until the
# operation has been observed in its own right
# fullpipe measured at 1734-2144 s against a 281 s capped baseline on Voxtral (6.2-7.6x);
# 8.0 rounds up so the FIRST uncapped run on a model is not killed by a hair, before the
# bucket has any history of its own.
_OP_IN_BASE_UNITS = {"profile": 1.0, "pcc": 9.0, "build": 6.0, "round": 12.0, "agent": 2.0, "fullpipe": 8.0}


def _timer_overrides_active() -> list:
    """Env overrides that PIN a timer and therefore disable adaptivity. Reported so a
    pinned timer is never silent (PERF_MCP_MEASURE_BACKSTOP=900 once hard-capped every
    llama PCC call at 900 s and produced `timed out after 900 seconds` on every attempt)."""
    keys = (
        "PERF_MCP_MEASURE_BACKSTOP",
        "PERF_MCP_ROUND_MAX_SEC",
        "PERF_MCP_ROUND_STALL_SEC",
        "PERF_MCP_MEASURE_STALL_SEC",
        "PERF_MCP_DISCOVER_STALL_SEC",
    )
    return [k for k in keys if os.environ.get(k)]


def _observed(repo_root: Path, op: str) -> list:
    """Durations already logged for this operation on this model, newest last."""
    try:
        mani = _latest_manifest(repo_root / PERF_DIR)
        if mani is None:
            return []
        data = json.loads((mani.parent / _OBS_LOG).read_text())
        vals = [float(x) for x in (data.get(op) or []) if float(x) > 0]
        return vals[-32:]
    except Exception:  # noqa: BLE001
        return []


def record_observed(repo_root: Path, op: str, seconds: float) -> None:
    """Append a real duration so later timers scale off measured cost, not a proxy."""
    try:
        mani = _latest_manifest(repo_root / PERF_DIR)
        if mani is None or not seconds or seconds <= 0:
            return
        f = mani.parent / _OBS_LOG
        data = {}
        if f.is_file():
            data = json.loads(f.read_text()) or {}
        data.setdefault(op, []).append(round(float(seconds), 3))
        data[op] = data[op][-64:]
        f.write_text(json.dumps(data))
    except Exception:  # noqa: BLE001
        pass


def _op_cost(repo_root: Path, op: str) -> float:
    """Best estimate of what ONE `op` costs on this model: p95 of its own observations,
    else the baseline-profile duration scaled into that operation's units."""
    obs = _observed(repo_root, op)
    if obs:
        s = sorted(obs)
        return s[min(len(s) - 1, int(0.95 * len(s)))]
    # COLD START: ask the agent to size this op from the model's own evidence instead of applying a
    # frozen per-op multiplier table (the table is what capped llama's 872 s build at 240 s).
    base, ceil = _baseline_ceiling(repo_root)
    try:
        from agent.probes import _agent_seconds

        est = _agent_seconds(op, base, ceil)
        if est > 0:
            return est / max(1.0, _OP_MULT.get(op, 4.0))
    except Exception:  # noqa: BLE001
        pass
    if base > 0:
        return base * _OP_IN_BASE_UNITS.get(op, 1.0)
    return 0.0


def adaptive_timer(repo_root: Path, op: str, *, env_key: str = "", mult: float = 0.0) -> int:
    """Budget for one `op`, proportional to that op's observed cost.

    clamp(mult * cost, _MIN_TIMER_S, operator_ceiling). No absolute floor, so a 3 s module
    gets tens of seconds and an 8B pipeline gets what its own cycle costs.
    """
    if env_key:
        ov = os.environ.get(env_key)
        if ov:
            try:
                return int(ov)
            except ValueError:
                pass
    _, ceil = _baseline_ceiling(repo_root)
    cost = _op_cost(repo_root, op)
    m = mult or _OP_MULT.get(op, 4.0)
    if cost <= 0:
        return int(min(ceil, max(_MIN_TIMER_S, 600)))  # cold start: no history, no proxy yet
    return int(min(ceil, max(_MIN_TIMER_S, m * cost)))


def _adaptive_cap(repo_root: Path, floor: int, mult: int = 3) -> int:
    """Retained for callers that pass an explicit floor; the floor is now a CEILING-side
    hint only and no longer allowed to dominate a measured budget."""
    base, ceil = _baseline_ceiling(repo_root)
    if ceil < floor:
        ceil = floor
    scaled = int(mult * base)
    return min(ceil, max(int(_MIN_TIMER_S), scaled or floor))


# How many times the agent watchdog may answer "wait" and re-arm a round that has shown no real
# progress. Each reprieve is worth one max_no_progress window. Finite, because an LLM verdict is a
# judgement and not a bound -- see _run_round_with_watchdog.
_MAX_WATCHDOG_REPRIEVES = 3


def _round_hard_cap(repo_root: Path, stall_sec: int) -> int:
    """UNPRODUCTIVE bound for one agent round, derived from the observed ROUND cycle."""
    return adaptive_timer(repo_root, "round", env_key="PERF_MCP_ROUND_MAX_SEC")


def _measure_backstop(repo_root: Path) -> int:
    """Hard wall for one on-device measurement, derived from observed PROFILE durations."""
    return adaptive_timer(repo_root, "profile", env_key="PERF_MCP_MEASURE_BACKSTOP")


# One refusal is worth a retry; two identical ones are not a blip. Env-tunable like the wedge
# counter beside it, so an operator on a flaky link can allow more without editing source.
_MAX_AUTH_STRIKES = max(1, int(os.environ.get("PERF_MCP_MAX_AUTH_STRIKES", "2") or "2"))


# A renewal that keeps succeeding while rounds keep being refused must not spin: past this many
# recoveries the refusal is treated as permanent and the round is spent, so the loop always advances.
_MAX_AUTH_RECOVERIES = max(1, int(os.environ.get("PERF_MCP_MAX_AUTH_RECOVERIES", "3") or "3"))


def _recover_agent_auth() -> bool:
    """Ask the agent binary to prove it can authenticate, which is also what renews it.

    The client holds a refresh token of its own and renews an expired access token when something
    asks it to work -- verified by expiring a COPY of the credential and watching one call both
    succeed and rewrite the expiry. So the recovery is simply to ask, with a prompt small enough to
    cost nothing: if the answer comes back the credential is usable and the round can be re-run,
    and if it does not, no renewal this tool could perform would have helped either.

    Deliberately does not touch the credential file. Rewriting it by hand would mean re-implementing
    the client's own refresh flow, and getting that wrong locks the box out of the agent entirely --
    a far worse failure than the lost round this is recovering from.
    """
    try:
        r = subprocess.run(
            [_resolve_claude_bin(), "-p", "ok", "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("PERF_MCP_AUTH_PROBE_TIMEOUT_S", "180") or "180"),
        )
    except Exception:  # noqa: BLE001 -- a probe that cannot run is a probe that proves nothing
        return False
    if r.returncode != 0:
        return False
    from agent.probes import detect_auth_failure, detect_quota_exhausted

    # A VALID CREDENTIAL WITH NO BUDGET LEFT ANSWERS THIS PROBE. It is not refused for credentials,
    # so the auth check alone reports a successful recovery, the round is retried, and the retry is
    # refused again -- draining the budget over a problem renewing cannot touch.
    _out = (r.stdout or "") + (r.stderr or "")
    return detect_auth_failure(_out) is None and detect_quota_exhausted(_out) is None


def _round_refusal(kernel_log, detector: str) -> str | None:
    """The phrase this round's transcript used to refuse, per `detector`, or None if it was let in.

    Reads the round's own transcript rather than re-deriving a cause: the client writes its refusal
    there verbatim, and quoting it is what tells an operator which of several unrelated remedies
    applies -- re-login, a key, an org allowlist, or waiting for a limit to reset.
    """
    try:
        from agent import probes
    except Exception:  # noqa: BLE001 -- a check that cannot load must not stop the run
        return None
    fn = getattr(probes, detector, None)
    if fn is None:
        return None
    return fn("\n".join(_tail_lines(str(kernel_log) + ".agent.log", 60)))


def _agent_auth_failure(kernel_log) -> str | None:
    """The round was refused for want of CREDENTIALS -- renewing may fix it."""
    return _round_refusal(kernel_log, "detect_auth_failure")


def _agent_quota_exhausted(kernel_log) -> str | None:
    """The round was refused for want of BUDGET -- renewing cannot fix it."""
    return _round_refusal(kernel_log, "detect_quota_exhausted")


def _tail_lines(path, n: int = 6) -> list:
    """Last n lines of a log, for watchdog evidence. Raw text matters: a repeated error line
    is what distinguishes a spin loop from progress."""
    try:
        with open(path, errors="ignore") as fh:
            return [ln.strip()[:200] for ln in fh.readlines()[-n:]]
    except Exception:  # noqa: BLE001
        return []


def _observed_stats(repo_root: Path, op: str) -> dict:
    """p50/p95/p99 of this operation's own logged durations, for watchdog evidence."""
    vals = sorted(_observed(repo_root, op))
    if not vals:
        return {}
    q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]  # noqa: E731
    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


# --- BUG 4 agreed design: agent watchdog on FULL evidence -----------------------
# Benchmarked 2026-07-25 on 84 held-out scenarios: fixed timers 59/84 (70%), agent given
# only summary stats 71/84, agent + derived bounds 82/84, agent on FULL RAW EVIDENCE 84/84
# (100%, zero false kills). The gains are entirely in cases a clock cannot judge:
#   host-bound quiet -> compile / weight load / thermal cooldown / device reset / git op /
#                       API backoff / JIT use no device CPU and may emit no log, yet healthy
#   zombie           -> a constant tiny CPU trickle with zero log growth is not progress
#   spin             -> log grows but the SAME action repeats; needs the novelty signal
# Summary statistics destroy the signal that separates working from stuck (repetition), so
# the agent is handed the raw action sequence and log tail, not aggregates.
HOST_BOUND_OPS = {
    "kernel_compile",
    "weight_load",
    "thermal_cool",
    "device_reset",
    "git_op",
    "api_backoff",
    "jit_compile",
}

_WATCHDOG_PROMPT = """Watchdog decision for a model-optimization round: KEEP WAITING or KILL.

model/pipeline: {model}
operation in flight: {op}   running for: {op_elapsed}s
time since last commit/kernel attempt: {since_commit}s
OBSERVED history for this operation on this model: p50={p50}s p95={p95}s p99={p99}s
device CPU per window (oldest->newest): {cpu_hist}
transcript bytes per window (oldest->newest): {txt_hist}
actions in window: {actions}   DISTINCT actions: {distinct_actions}
action sequence: {action_seq}
last log lines: {log_tail}
absolute operator ceiling: {ceiling}s

Host-bound work ({host_bound}) consumes NO device CPU and may emit almost no log, yet is
healthy. A constant tiny CPU trickle with zero log growth can be a zombie. Many actions but
only 1-2 DISTINCT means it is repeating itself: a spin/retry loop, not progress. Judge against
the observed history, never a fixed number.

Reply with ONLY: {{"decision":"wait"|"kill"}}"""


def _watchdog_bounds(ev: dict) -> tuple:
    """Derived fallback net: grace = p95 of the op, flat = p99 scaled by its own spread."""
    o = ev.get("observed") or {}
    p50 = float(o.get("p50") or 0.0)
    p95 = float(o.get("p95") or 0.0)
    p99 = float(o.get("p99") or p95)
    ceiling = float(ev.get("ceiling") or 10800.0)
    spread = (p95 / p50) if p50 > 0 else 2.0
    grace = min(ceiling, p95) if p95 > 0 else float(_MIN_TIMER_S)
    flat = min(ceiling, p99 * spread) if p99 > 0 else ceiling
    return grace, flat, ceiling


def _watchdog_ask_agent(ev: dict) -> str:
    """Ask the Claude Code agent. Returns "wait"/"kill", or "" when unavailable."""
    claude = shutil.which(_resolve_claude_bin()) or shutil.which("claude")
    if not claude:
        return ""
    o = ev.get("observed") or {}
    prompt = _WATCHDOG_PROMPT.format(
        model=ev.get("model", "?"),
        op=ev.get("op", "?"),
        op_elapsed=round(float(ev.get("op_elapsed") or 0), 1),
        since_commit=round(float(ev.get("since_commit") or 0), 1),
        p50=o.get("p50"),
        p95=o.get("p95"),
        p99=o.get("p99"),
        cpu_hist=ev.get("cpu_hist"),
        txt_hist=ev.get("txt_hist"),
        actions=ev.get("actions"),
        distinct_actions=ev.get("distinct_actions"),
        action_seq=(ev.get("action_seq") or [])[:14],
        log_tail=(ev.get("log_tail") or [])[-6:],
        ceiling=int(float(ev.get("ceiling") or 10800)),
        host_bound=", ".join(sorted(HOST_BOUND_OPS)),
    )
    try:
        r = subprocess.run(
            [claude, "-p", prompt, "--output-format", "text"], capture_output=True, text=True, timeout=180
        )
        out = (r.stdout or "").strip()
        i, j = out.find("{"), out.rfind("}")
        d = json.loads(out[i : j + 1]).get("decision", "")
        return d if d in ("wait", "kill") else ""
    except Exception:  # noqa: BLE001
        return ""


def watchdog_decide(ev: dict, agent=_watchdog_ask_agent) -> str:
    """Continue or kill this round, judged from evidence rather than elapsed wall clock.

    The agent decides; derived bounds are only a net. Guardrails (validated in the same
    benchmark) fix the agent's two observed failure modes: it killed active work early, and
    it once waited forever on an all-flat round.
    """
    grace, flat, ceiling = _watchdog_bounds(ev)
    since = float(ev.get("since_commit") or 0.0)
    elapsed = float(ev.get("op_elapsed") or 0.0)
    alive = bool((ev.get("cpu_hist") or [0])[-1] or (ev.get("txt_hist") or [0])[-1])
    acts = int(ev.get("actions") or 0)
    novel = int(ev.get("distinct_actions") or 0) > 1 or acts <= 1
    host_bound = (ev.get("op") or "") in HOST_BOUND_OPS

    decision = ""
    if agent is not None:
        try:
            decision = agent(ev) or ""
        except Exception:  # noqa: BLE001
            decision = ""

    if decision == "kill":
        if elapsed < grace and (alive or host_bound) and novel:
            return "wait"  # grace: never kill inside the op's own normal duration
        return "kill"
    if decision == "wait":
        if since > ceiling:
            return "kill"  # operator ceiling: a confused agent cannot wait forever
        return "wait"

    # No agent available: derived net only.
    if since > ceiling:
        return "kill"
    if host_bound and since <= flat:
        return "wait"  # legitimately quiet
    if not alive and since > flat:
        return "kill"
    if not novel and acts > 1 and since > flat:
        return "kill"  # spin loop
    return "wait"


def _run_round_with_watchdog(cmd: list, repo_root: Path, devices: str, kernel_log: str, stall_sec: int) -> bool:
    """Run one `claude -p` round under a forward-progress watchdog. If neither a commit nor a kernel
    attempt is recorded for stall_sec while the round is alive, treat it as a device wedge: SIGKILL the
    whole process group (claude + its mcp server + any hung profiler) and reset the device. Returns True
    if the round was killed as wedged, False if it exited on its own. The NEXT round re-spawns a fresh
    mcp server + runs on the reset mesh, so a stale cached-mesh handle can't persist across the wedge."""
    agent_log = str(kernel_log) + ".agent.log"
    try:
        _lf = open(agent_log, "a", buffering=1, errors="ignore")
    except Exception:  # noqa: BLE001
        _lf = subprocess.DEVNULL
    # CLEAN screen: the agent's raw stream-json transcript goes to agent_log, not the terminal —
    # the terminal shows only a periodic heartbeat. Full detail stays in the log file.
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=cc_env(repo_root, devices),
        start_new_session=True,
        stdout=_lf,
        stderr=subprocess.STDOUT,
    )
    try:
        _pgid = os.getpgid(proc.pid)
    except Exception:  # noqa: BLE001
        _pgid = None

    def _liveness():
        # A slow-but-WORKING round advances one of these even before it commits: the agent transcript
        # grows (agent thinking / choosing tools) or the process group accrues CPU (a long tracy profile
        # compiling kernels + running device ops -- GLM's 8-chip mesh profile alone is ~6 min). Only a
        # TRULY FROZEN round (wedged device: everything blocked on I/O) advances NEITHER -- that is the
        # real wedge. Watching only git/kernel-log killed legit multi-minute profiles as false wedges.
        try:
            amt = os.path.getmtime(agent_log)
        except OSError:
            amt = 0.0
        cpu = 0
        if _pgid is not None:
            try:
                from agent.probes import _pgroup_cpu_jiffies

                cpu = _pgroup_cpu_jiffies(_pgid)
            except Exception:  # noqa: BLE001
                cpu = 0
        return (amt, cpu)

    # Two independent kill bounds so a round can NEVER run unbounded:
    #   stall_sec        - FROZEN: no sign of life at all (fast kill of a true device wedge).
    #   max_no_progress  - HARD CAP: alive but produced NO real progress (commit/kernel attempt) for
    #                      this long -> kill anyway (default 4x stall / >=40min, comfortably above one
    #                      legit slow measure cycle, so a productive round always records well within it).
    max_no_progress = _round_hard_cap(repo_root, stall_sec)
    last_tok = _progress_token(repo_root, kernel_log)
    last_live = _liveness()
    _now0 = time.monotonic()
    last_active = _now0  # last sign of life (CPU / transcript / real progress)
    last_real = _now0  # last REAL progress (commit / recorded kernel attempt)
    _reprieves = [0]  # how many times the watchdog has re-armed this round; see below
    _stuck_since = [None]  # when real progress was last seen; NOT rewound by a reprieve
    _t0 = _now0
    wedge_reason = ""
    try:
        while True:
            try:
                proc.wait(timeout=60)
                return False
            except subprocess.TimeoutExpired:
                _now = time.monotonic()
                print(f"  · optimizing… {int(_now - _t0)}s (agent transcript → {agent_log})", flush=True)
                tok = _progress_token(repo_root, kernel_log)
                live = _liveness()
                if tok != last_tok:  # real progress resets BOTH clocks
                    last_tok, last_live, last_active, last_real = tok, live, _now, _now
                    _stuck_since[0] = None  # and only real progress clears the stuck clock
                elif live[0] != last_live[0] or (live[1] - last_live[1]) > 200:  # alive: transcript/CPU
                    last_live, last_active = live, _now
                if _now - last_active > stall_sec:
                    wedge_reason = "FROZEN %ds — no commit, no device CPU, no agent activity (real wedge)" % stall_sec
                    break
                if _now - last_real > max_no_progress:
                    if _stuck_since[0] is None:
                        _stuck_since[0] = last_real
                    # BUG 4: elapsed wall clock alone cannot tell slow-but-working from stuck --
                    # it killed a healthy llama round 4x on 2026-07-25. Ask the agent watchdog,
                    # which reads the actual evidence; the derived net still bounds it.
                    _ev = {
                        "model": str(repo_root.name),
                        "op": "round",
                        "op_elapsed": _now - _t0,
                        # NOT `_now - last_real`. A reprieve rewinds last_real, so reporting from
                        # it reset the watchdog's own operator ceiling ("a confused agent cannot wait
                        # forever") to zero on every reprieve -- the bound was being cleared by the
                        # thing it existed to bound. This clock is only cleared by REAL progress.
                        "since_commit": _now - (_stuck_since[0] or _now),
                        "cpu_hist": [1 if live else 0 for _ in range(5)],
                        "txt_hist": [1 if live else 0 for _ in range(5)],
                        "actions": 1,
                        "distinct_actions": 1,
                        "action_seq": [],
                        "log_tail": _tail_lines(agent_log, 6),
                        "observed": _observed_stats(repo_root, "round"),
                        "ceiling": _baseline_ceiling(repo_root)[1],
                    }
                    # AN LLM MAY NOT RE-ARM THIS FOREVER.
                    #
                    # "wait" resets last_real, so a watchdog that keeps answering "wait" keeps a
                    # stuck round alive with no bound at all -- the round equivalent of the budget
                    # that was demoted to a warning and let run 12 spin for nine hours. A judgement
                    # is worth having; an unlimited number of them is not a bound.
                    #
                    # So the verdict is honoured a fixed number of times and then stops being asked.
                    # Each reprieve is worth max_no_progress, so the round still gets
                    # _MAX_WATCHDOG_REPRIEVES x that before it is called stuck -- generous for slow
                    # work, finite for stuck work.
                    _verdict = watchdog_decide(_ev)
                    if _verdict == "wait":
                        if _reprieves[0] < _MAX_WATCHDOG_REPRIEVES:
                            _reprieves[0] += 1
                            print(
                                "  [optimize/cc] watchdog judged the round healthy — reprieve %d/%d "
                                "(%ds each, %ds stuck)"
                                % (
                                    _reprieves[0],
                                    _MAX_WATCHDOG_REPRIEVES,
                                    max_no_progress,
                                    int(_now - (_stuck_since[0] or _now)),
                                ),
                                flush=True,
                            )
                            last_real = _now  # judged healthy: re-arm and keep going
                            continue
                        wedge_reason = (
                            "UNPRODUCTIVE %ds — the watchdog re-armed this round %d times and it "
                            "still has no real progress; a verdict is not a bound" % (max_no_progress, _reprieves[0])
                        )
                        break
                    wedge_reason = (
                        "UNPRODUCTIVE %ds — agent watchdog judged the round stuck (no real progress)" % max_no_progress
                    )
                    break
    finally:
        # BUG 4 (#3): feed the real round duration back so later budgets scale off OBSERVED
        # cost instead of a baseline-derived proxy. Without this the adaptive path never
        # learns and every timer stays on its estimate.
        try:
            record_observed(repo_root, "round", time.monotonic() - _now0)
        except Exception:  # noqa: BLE001
            pass
        try:
            if _lf not in (None, subprocess.DEVNULL):
                _lf.close()
        except Exception:  # noqa: BLE001
            pass
    _record_wedge_to_log(kernel_log, f"wedged: round killed ({wedge_reason})")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    rst = _reclaim_device(devices, error_text=_tail_lines(agent_log, 40))
    print(
        "  [optimize/cc] WATCHDOG: round %s — killed the round + %s; next round starts a FRESH mcp "
        "server on the reset mesh." % (wedge_reason, rst)
    )
    return True


def _baseline_name() -> str:
    """Filename of the rolling baseline, KEYED by (model, task) to match perf_mcp._baseline_path().

    This read used to hardcode the unkeyed "perf_mcp_baseline.json". perf_mcp writes the keyed file,
    so the two never referred to the same run: whatever model last profiled anywhere on the box
    supplied the "before" number. llama3_1_8b_p150 reported `eager per-op (all layers): 0.06 ms ->
    648.17 ms (-1062476.1%)` -- a sub-millisecond anchor from an unrelated run against a real
    648 ms reading, while this run's own baseline sat in the keyed file at 2464.18 ms. Same defect
    as the full-pipeline scoreboard: a file any other process can write is not this run's baseline.
    """
    model = os.environ.get("PERF_MCP_MODEL_NAME") or Path(os.environ.get("PERF_MCP_MODEL_ROOT", "") or "model").name
    task = os.environ.get("PERF_MCP_TASK", "main")
    return "perf_mcp_baseline_%s_%s.json" % (model, task)


def _baseline_ms() -> float | None:
    try:
        pass

        d = json.loads((state_dir() / _baseline_name()).read_text())
        return float(d["device_ms"]) if d.get("device_ms") is not None else None
    except Exception:  # noqa: BLE001
        return None


def _last_committed_ms(kernel_log_path) -> float | None:
    """device_ms of the most recently banked win (last beat_baseline record carrying a measured ms)
    = the current committed state. The report's 'final' comes from this, not from the rolling
    perf_mcp_baseline.json — that file is only refreshed by profile_model, so on a module where
    profile_model was not re-run after the wins it stays at the stale initial baseline and the
    headline reads +0.0% despite real committed wins."""
    try:
        rows = json.loads(Path(kernel_log_path).read_text())
    except Exception:  # noqa: BLE001
        return None
    ms = None
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and r.get("beat_baseline") and r.get("measured_ms") is not None:
            ms = float(r["measured_ms"])
    return ms


def _comparable(value, stored_depth: str, want_depth: str):
    """Verdict on whether a stored ms value may anchor a comparison at `want_depth`.

    Delegates to integrity.Reading so DEPTH means the same thing here as in the report renderer. An
    UNSTAMPED value is UNKNOWN, never assumed to match: the file predates the stamp, so it could be
    from any window -- that is how a 2-layer number anchored a 16-layer run.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.integrity import Reading, Verdict
    except Exception:  # noqa: BLE001

        class _V:  # minimal stand-in so a missing import never silently accepts
            is_pass = False
            reason = "integrity unavailable"

        return _V()
    if not stored_depth:
        return Verdict.unknown("no depth stamp, so the window it was profiled at is unknown")
    return Reading(value, depth=stored_depth).comparable_to(Reading(value, depth=want_depth))


def _depth_scoped_throughput(snap, want_depth: str):
    """Drop the depth-SENSITIVE part of a roofline snapshot, keep the depth-invariant part.

    The snapshot mixes two kinds of quantity and only one of them cares about the profiled window:

      * ``modeled_floor_ms`` is a SUM OVER THE PROFILED OPS, so a 2-layer floor rendered against a
        16-layer measurement is meaningless -- the 832.93-vs-1088.15 headline.
      * ``theoretical_rate`` / ``band`` / ``active_bytes`` / ``bw_fraction`` / ``unit`` are per-unit
        model physics (bytes per token over bandwidth). The window never entered the arithmetic.

    This used to be inline in _emit_summary and set the whole snapshot to None, so a depth mismatch
    also deleted the ceiling. llama3_1_8b_p150 wrote its snapshot at TT_PERF_LAYERS=16 and finalized
    the report at `all`, so the computed 54.577 tok/s/u ceiling (band 32.75-43.66) was discarded and
    the report printed NO_BAND. The log line already said "omitting the FLOOR" -- the code just did
    more than the message claimed.

    An UNSTAMPED snapshot also loses its floor: per _comparable, no stamp means the window is
    unknown, and assuming it matches is exactly how a 2-layer number once anchored a 16-layer run.
    """
    if not isinstance(snap, dict):
        return None
    stored = str(snap.get("perf_layers", "")).strip()
    if stored and _comparable(1.0, stored, want_depth).is_pass:
        return snap
    scoped = dict(snap)  # copy: the caller still holds the original
    scoped["modeled_floor_ms"] = None
    return scoped


def _prune_legacy_reports(demo_dir: Path) -> None:
    for legacy in ("E2E_REPORT.md", "summary.md"):
        try:
            (Path(demo_dir) / legacy).unlink()
        except OSError:
            pass


def _emit_summary(
    repo_root: Path,
    kernel_log: str,
    model_name: str,
    task: str,
    metric: str,
    start_sha: str,
    perf_test: str = "",
    before_ms=None,
    after_ms=None,
    before_mode: str = "",
    after_mode: str = "",
) -> None:
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("cc_summary", str(Path(__file__).parent / "summary.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] summary unavailable: {exc}")
        return
    wins = None
    if start_sha:
        c = _git(repo_root, "rev-list", f"{start_sha}..HEAD", "--count")
        wins = int(c) if c.isdigit() else None
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    report_csv = ""
    residual = None
    try:
        _runs = repo_root / PERF_DIR / "runs"
        _rc = sorted(_runs.rglob("*report*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if _rc:
            report_csv = str(_rc[0])
        # DERIVE THE RESIDUAL, DO NOT LOAD IT. This read the newest residual_report.json by MTIME across
        # every run directory -- a file NOTHING in the tree writes, so the line it fed was dead, and had
        # it existed the mtime search could have taken another model's run. Computing it from the profile
        # this report is already rendering makes it current by construction, so it cannot go stale.
        try:
            from agent import roofline as _rl

            _prof = _read_baseline_profile_for_report(repo_root)
            if _prof:
                residual = _rl.residual_report(_prof, (manifest or {}).get("env", {}) or {})
        except Exception:  # noqa: BLE001
            residual = None
    except Exception:  # noqa: BLE001
        pass
    render_kernel = kernel_log
    _cum = str(kernel_log) + ".cumulative"

    def _load_list(_p):
        try:
            _v = json.loads(Path(_p).read_text())
            return _v if isinstance(_v, list) else []
        except Exception:
            return []

    _seen, _merged = set(), []
    for _a in _load_list(_cum) + _load_list(kernel_log):
        if not isinstance(_a, dict):
            continue
        _k = (
            _a.get("op_signature") or _a.get("op_code") or "",
            _a.get("kernel_kind") or "",
            (_a.get("note") or "")[:200],
            bool(_a.get("wedged")),
        )
        if _k in _seen:
            continue
        _seen.add(_k)
        _merged.append(_a)
    try:
        Path(_cum).write_text(json.dumps(_merged))
        render_kernel = _cum
    except Exception:
        render_kernel = kernel_log
    # THIS run's log, never the cumulative one: the cumulative view is for rendering attempt
    # HISTORY, but the current/final number must come from what this run actually committed.
    _lc = _last_committed_ms(kernel_log)
    _cur_ms = _lc if _lc is not None else _baseline_ms()
    _throughput = None
    try:
        pass

        _tp = state_dir() / ("perf_mcp_throughput_%s_%s.json" % (model_name, task))
        if _tp.exists():
            _throughput = json.loads(_tp.read_text())
            # The roofline floor sums per-op floors over the PROFILED window, so it is only
            # comparable to a measurement taken at the same depth. This snapshot survives between
            # runs; drop THE FLOOR when the window moved -- but keep the ceiling, which is per-unit
            # physics and never depended on the window.
            _wl = _depth_in_force()
            _sl = str((_throughput or {}).get("perf_layers", "")).strip()
            _throughput = _depth_scoped_throughput(_throughput, _wl)
            if (_throughput or {}).get("modeled_floor_ms") is None:
                print(
                    "  [optimize/cc] roofline snapshot was computed at TT_PERF_LAYERS=%s but this run "
                    "uses %s; omitting the floor rather than comparing across depths (the ceiling is "
                    "per-unit and is kept)" % (_sl or "<unstamped>", _wl)
                )
    except Exception:  # noqa: BLE001
        _throughput = None
    text = mod.render_summary(
        render_kernel,
        _cur_ms,
        model=model_name,
        task=task,
        metric=metric,
        committed_wins=wins,
        opt_branch=branch,
        perf_test=perf_test,
        report_csv=report_csv,
        residual=residual,
        # THE PROFILE IS READ FROM WHERE IT IS WRITTEN. This looked beside `report_csv`; the profile
        # is written to runs/<ts>/profiles/, so it resolved to None on every real run -- and with no
        # profile the report silently loses its Op breakdown, its Dispatch row, its Fidelity ladder
        # and every Utilization bar, because each of those renders nothing rather than complaining.
        # _read_baseline_profile_for_report already resolves it correctly and was in use a few lines
        # above for the residual; the CSV-adjacent path stays as a fallback.
        baseline_profile=(
            _read_baseline_profile_for_report(repo_root)
            or (
                json.loads(Path(report_csv).parent.joinpath("baseline_profile.json").read_text())
                if report_csv and Path(report_csv).parent.joinpath("baseline_profile.json").is_file()
                else None
            )
        ),
        finalized=True,
        final_override_ms=_cur_ms,
        throughput=_throughput,
        # summary cannot find the model dir on its own under the by-path load it gets, and the
        # compute roofs need perf_target_inputs.json that lives there.
        model_root=str(_model_root_for_report(repo_root) or ""),
    )
    print("\n" + text + "\n")
    md = _latest_manifest(repo_root / PERF_DIR)
    if md:
        try:
            _demo = Path(json.loads(md.read_text()).get("config", {}).get("model_root") or "")
        except Exception:  # noqa: BLE001
            _demo = None
        if _demo and str(_demo):
            when = f"Final end-of-run summary: {time.strftime('%Y-%m-%d %H:%M:%S %Z')} (adds committed wins, full-pipeline e2e, roofline residual)"
            # BRING-UP IS NOT PART OF OPTIMIZE. This refreshed tt_hw_planner's bring-up section into the
            # same RUN_REPORT.md on every optimize finalize, so a perf run rewrote a section it does not
            # own and cannot vouch for. Bring-up has its own report path; optimize owns the optimize
            # section only.
            _key = os.environ.get("PERF_MCP_REPORT_KEY", "optimize")
            _module = os.environ.get("PERF_MCP_REPORT_MODULE")
            if _module:
                _block = mod.module_optimize_block(
                    _demo,
                    0,
                    text,
                    when,
                    module=_module,
                    index=os.environ.get("PERF_MCP_REPORT_INDEX", ""),
                    pcc_gate=os.environ.get("PERF_MCP_REPORT_PCC", ""),
                    outcome="optimizing…",
                )
            else:
                _block = mod.optimize_block(_demo, 0, text, when)
            _written = mod.upsert_report_section(_demo, _key, _block)
            # The RESOLVED path, not an assumed one: the report now lands in the git-ignored run
            # directory (see summary.report_path), so printing model_root/RUN_REPORT.md would send
            # a reader to a file that no longer updates -- exactly the confusion this change fixes.
            print(f"  [optimize/cc] report updated: {_written or (_demo / 'RUN_REPORT.md')} ({_key} section)")
            _prune_legacy_reports(_demo)
        try:
            (md.parent / "summary.md").unlink()
        except OSError:
            pass


def _hitl_watch(repo_root, hitl_dir, stop_event):
    """Orchestrator-side HITL loop (own thread): watch for the agent's lever proposal, render the pause
    screen, read the operator's commit/revert/try, perform the git action, and answer the blocked agent."""
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("cc_hitl", str(Path(__file__).parent / "hitl.py"))
    _h = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_h)

    while not stop_event.is_set():
        prop = _h.read_proposal(hitl_dir)
        if prop is None:
            stop_event.wait(0.4)
            continue
        print("\n" + _h.render_pause_screen(prop) + "\n", flush=True)
        try:
            ans = input("  choice [c=commit / r=revert / t=try other]: ").strip().lower()
        except (EOFError, OSError):
            ans = "r"
        if ans.startswith("t"):
            try:
                knob = input("  knob / instruction to try next: ").strip()
            except (EOFError, OSError):
                knob = ""
            _h.post_decision(hitl_dir, "try", knob=knob)
        elif ans.startswith("c"):
            _git(repo_root, "add", "-A")
            _git(repo_root, "commit", "-m", "hitl: %s" % (prop.get("tried", {}).get("lever", "lever")))
            # Report what actually happened: this used to print "committed." unconditionally, so a
            # failed git silently lost the win while the operator and the agent were told it landed.
            if _git_ok():
                _h.post_decision(hitl_dir, "commit")
                print("  [hitl] committed.", flush=True)
            else:
                print("  [hitl] COMMIT FAILED — the win is NOT banked: %s" % _git_last_error(), flush=True)
        else:
            _git(repo_root, "checkout", "--", ".")
            if _git_ok():
                _h.post_decision(hitl_dir, "revert")
                print("  [hitl] reverted.", flush=True)
            else:
                print(
                    "  [hitl] REVERT FAILED — the rejected edit is STILL IN THE TREE: %s" % _git_last_error(),
                    flush=True,
                )


def _report_baseline_for_seed(model_name: str, task: str):
    """The ledger's baseline anchor for this (model, task), or None. Read-only."""
    try:
        led = _ledger()
        row = led.first(led.KIND_EAGER, led.PHASE_BEFORE, model_name, task)
        v = float((row or {}).get("value_ms"))
        return v if v > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _model_root_for_report(repo_root):
    """The model directory this run is optimizing, from the run manifest. None if unknown.

    summary needs it for perf_target_inputs.json (params -> the compute roofs). It cannot resolve it
    itself: loaded by path, perf_mcp's _MODEL_ROOT falls back to "." and the lookup silently misses.

    _latest_manifest returns a PATH, not the parsed document. This called .get() on that Path, threw
    AttributeError, and the bare `except` turned it into None -- on every run, silently, since the day
    it was written. So the report never received the model directory it exists to supply, fell through
    to perf_mcp's "." fallback, and read whatever perf_target_inputs.json was in the working directory:
    on gemma-3 a 31 MB, 32-layer file the tool had itself written there, giving a prefill memory
    ceiling of 0.061 ms against a 100 ms measurement, no param count, and therefore no compute roof
    and no fidelity ladder.

    The except stays -- a missing or malformed manifest must not take the report down -- but it now
    guards a read that can actually succeed, and it no longer hides a bug in this function's own logic.
    """
    try:
        md = _latest_manifest(repo_root / PERF_DIR)
        if not md:
            return None
        cfg = json.loads(Path(md).read_text()).get("config", {}) or {}
        root = str(cfg.get("model_root") or "").strip()
        return Path(root) if root else None
    except Exception:  # noqa: BLE001
        return None


def _read_baseline_profile_for_report(repo_root):
    """The newest baseline profile JSON for THIS run, or None. Used to derive the roofline residual at
    render time rather than reading a stale artifact."""
    try:
        _runs = repo_root / PERF_DIR / "runs"
        cands = sorted(_runs.rglob("baseline_profile.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return json.loads(cands[0].read_text()) if cands else None
    except Exception:  # noqa: BLE001
        return None


def _warn_dirty_model_tree(demo_dir, repo_root) -> list:
    """Uncommitted edits inside the MODEL dir at start-up. Returns the paths (empty when clean).

    A killed run leaves its half-applied edit in the working tree, and nothing detected that: the
    restart profiled the partial state and treated it as the starting point, while the ledger's baseline
    still described the last COMMITTED state. Every later "gain vs base" was then measured against work
    that was never the baseline.

    This reports rather than reverts. Discarding an edit could throw away a real win the previous run had
    not yet committed, which is not a decision to make silently -- so the run is told what it inherited
    and PERF_MCP_REQUIRE_CLEAN=1 turns it into a hard stop for unattended use.
    """
    try:
        from agent import gitio

        dirty = [f for f in gitio.changed_files(repo_root, "HEAD", pathspec=demo_dir) if f.strip()]
    except Exception:  # noqa: BLE001
        return []
    if not dirty:
        return []
    print(
        "  [optimize/cc] WARNING: %d uncommitted file(s) in the model dir at start-up — a previous run "
        "may have been interrupted mid-attempt:" % len(dirty)
    )
    for f in dirty[:10]:
        print("      %s" % f)
    if len(dirty) > 10:
        print("      … and %d more" % (len(dirty) - 10))
    print(
        "      The ledger baseline describes the last COMMITTED state, so measurements taken now are "
        "against different work. Commit or revert before trusting a gain."
    )
    if os.environ.get("PERF_MCP_REQUIRE_CLEAN") == "1":
        print("  [optimize/cc] PERF_MCP_REQUIRE_CLEAN=1 and the model tree is dirty — refusing to start.", flush=True)
        raise SystemExit(EXIT_REFUSED)
    return dirty


def _preflight_tool(repo_root: Path) -> bool:
    """Run the TOOL'S OWN test suite against the copy this run will execute. Returns True to proceed.

    THE COPY THAT RUNS IS NOT THE COPY THAT WAS EDITED. The tool is developed in one checkout and
    synced into the repo the run uses, and nothing checked that the sync landed or that what landed
    imports. Three distinct failures came from that gap, each costing a run:

      * an edit that applied to the WRONG PLACE -- DEFAULT_ISL_TOKENS landed inside a template
        STRING rather than at module scope, so the module imported cleanly and the symbol did not
        exist. Found hours later, from an unrelated traceback;
      * a module reachable by package name but not by path -- the report loader uses
        spec_from_file_location, which supplies no package context and no sys.path entry for the
        module's own directory, so both relative and absolute imports raise. The report rendered
        with three blank sections and every failure was silent;
      * a `git stash` during a debugging detour that never popped, so two committed fixes were
        absent from the tree that then ran.

    Every one of them is caught by importing the tree and running its tests, which takes ~90 seconds
    against a run measured in hours. The suite is the tool's own; a failing one means the thing about
    to spend the night on a board is not the thing that was verified.

    PERF_MCP_SKIP_PREFLIGHT=1 skips it, for a deliberate run on a knowingly-red tree.
    """
    if os.environ.get("PERF_MCP_SKIP_PREFLIGHT") == "1":
        print("  [optimize/cc] preflight SKIPPED (PERF_MCP_SKIP_PREFLIGHT=1)")
        return True
    tests = Path(repo_root) / PERF_DIR / "tests"
    if not tests.is_dir():
        print(f"  [optimize/cc] preflight: no test suite at {tests} — cannot verify the tool that is about to run")
        return os.environ.get("PERF_MCP_REQUIRE_PREFLIGHT") != "1"
    print(f"  [optimize/cc] preflight: running the tool's own suite against {tests}")
    # THE SUITE TESTS THE CODE, NOT THIS RUN'S STATE, so it runs with this run's configuration
    # STRIPPED. Inheriting it is both wrong and dangerous:
    #
    #   WRONG -- the same test gives different answers in different shells. Under a run's env,
    #   test_all_boards_is_the_last_resort read the ambient PERF_MCP_DEVICES, so the config target
    #   won and the "all" fallback it exists to check was never reached. Green in a terminal, red
    #   here, and neither result was about the code.
    #
    #   DANGEROUS -- _KERNEL_LOG_PATH is resolved AT IMPORT from PERF_MCP_KERNEL_LOG. With the run's
    #   value inherited, a suite that calls record_kernel_attempt writes into the LIVE ladder: a
    #   check meant to protect the run would corrupt the state the run resumes from. It happened not
    #   to fire only because the tests that write were the ones failing.
    #
    # Stripped by prefix rather than by list, because the failure mode of a list is that the next
    # variable someone adds is not on it. TT_METAL_HOME and PYTHONPATH stay: those locate the code,
    # they do not configure a run.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PERF_MCP_", "TT_PERF_"))}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests), "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            env=env,
            timeout=int(os.environ.get("PERF_MCP_PREFLIGHT_TIMEOUT_S", "900")),
        )
    except Exception as exc:  # noqa: BLE001 -- a preflight that cannot RUN has cleared nothing
        print(f"  [optimize/cc] preflight could not run ({str(exc)[:160]}) — treating as UNKNOWN, not as passed")
        return os.environ.get("PERF_MCP_REQUIRE_PREFLIGHT") != "1"
    tail = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()][-1:] or ["(no output)"]
    if proc.returncode == 0:
        print(f"  [optimize/cc] preflight OK — {tail[0]}")
        return True
    print(f"  [optimize/cc] preflight FAILED — {tail[0]}")
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("FAILED") or ln.startswith("ERROR"):
            print("      %s" % ln)
    print(
        "      This is the tool that is about to run, in the repo it will run from. Fix it, re-sync, "
        "or set PERF_MCP_SKIP_PREFLIGHT=1 to start anyway."
    )
    return False


def _seed_ladder_from_cumulative(kernel_log: str, current_baseline_ms=None) -> int:
    """Start the ladder from what this (model, task) has ALREADY conclusively tried. Returns the count.

    The live log is deleted per pipeline so S2TT's ladder never leaks into T2T -- that isolation is the
    point, and it is preserved here because the cumulative file is itself keyed per (model, task). What
    it should NOT mean is amnesia across a RESTART of the same pipeline: a killed run left its verdicts
    in the cumulative log, and re-walking them costs hours of re-measurement for outcomes already known.

    Only CONCLUSIVE rows seed, and only when they still describe the same work:
      * `wedged` rows never seed -- a crash is exactly what an interrupted run leaves behind, and the
        state that caused it is gone;
      * a row whose `baseline_at_record` differs from the baseline now never seeds, because its verdict
        was reached against different work (a changed input, seq len or layer depth);
      * a row with no stamp at all never seeds -- logs written before the stamp cannot be judged, so they
        re-walk exactly as they do today.

    PERF_MCP_FRESH_LADDER=1 forces the old behaviour.
    """
    if os.environ.get("PERF_MCP_FRESH_LADDER") == "1":
        return 0
    cum = Path(str(kernel_log) + ".cumulative")
    try:
        rows = json.loads(cum.read_text())
        if not isinstance(rows, list):
            return 0
    except Exception:  # noqa: BLE001
        return 0
    try:
        base = round(float(current_baseline_ms), 4) if current_baseline_ms is not None else None
    except (TypeError, ValueError):
        base = None
    keep = []
    for r in rows:
        if not isinstance(r, dict) or r.get("wedged"):
            continue
        stamp = r.get("baseline_at_record")
        if stamp is None or base is None or round(float(stamp), 4) != base:
            continue
        keep.append(r)
    if not keep:
        return 0
    try:
        Path(kernel_log).write_text(json.dumps(keep))
    except OSError:
        return 0
    _rungs = sorted({"%s/%s" % (r.get("op_signature"), r.get("kernel_kind")) for r in keep})
    print(
        "  [optimize/cc] resuming ladder: %d attempt(s) already conclusive for this baseline, skipping %s"
        % (len(keep), ", ".join(_rungs[:8]) + (" …" if len(_rungs) > 8 else ""))
    )
    return len(keep)


def optimize_pipeline(
    repo_root: Path,
    manifest_path: str,
    pipe: dict,
    devices: str,
    metric: str,
    model_name: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    hitl: bool = False,
    config_ref: str = "",
) -> dict:
    """Drive one pipeline: claude -p re-invoked until the gate's can_stop, bounded by max_rounds.
    hitl=True runs the human-in-the-loop gate: the agent proposes one lever at a time via hitl_gate and
    a watcher thread renders the pause screen + performs the operator's commit/revert."""
    task = pipe["task"]
    kernel_log = _kernel_log_path(model_name, task)
    try:
        _fold_cumulative(kernel_log)
        os.path.exists(kernel_log) and os.remove(kernel_log)  # fresh ladder state per pipeline
        # ...then RESUME from what this pipeline already concluded, if anything still applies.
        _seed_ladder_from_cumulative(kernel_log, _report_baseline_for_seed(model_name, task))
    except OSError:
        pass
    _capture_board_topology()  # snapshot chip->board reset map while the device is healthy (reset-only)
    cfg = _mcp_config(repo_root, manifest_path, pipe, devices, kernel_log)
    _cov_env = cfg["mcpServers"]["perf-mcp"]["env"]
    # Discover the model's depth knob BEFORE the first probe, not inside it. The probe asks for ALL
    # layers by REMOVING the cap, and a perf test can fill that straight back in at import time
    # (os.environ.setdefault("TT_PERF_LAYERS", "2")). The depth guard undoes that, but only if it knows
    # WHICH variable to drop, and an existing demo may read any name. before_loop.py:569 already
    # discovers first and passes it down; doing the same here makes both entry points agree and means
    # even the first probe is guarded. Agent call only -- no device time.
    _depth_knob = {}
    try:
        _mr = _model_root_from_node(repo_root, pipe.get("perf_test"))
        if _mr is not None:
            _depth_knob = _llm_depth_env(_mr, 2) or {}
    except Exception:  # noqa: BLE001
        _depth_knob = {}
    if _depth_knob:
        _cov_env["PERF_MCP_DEPTH_VARS"] = ",".join(sorted(_depth_knob))
        print(f"  [optimize/cc] depth knob(s): {', '.join(sorted(_depth_knob))}")
    _cov, _cov_facts = _coverage_layers(
        repo_root,
        _cov_env,
        devices,
        pipe.get("perf_test"),
        pipe.get("case"),
        model_name=model_name,
        config_ref=config_ref,
        depth_knob=_depth_knob or None,
    )
    # _coverage_layers returns a dict {stack_id: depth} or None/0.
    # Wire the coverage depths into per-stack env vars.
    if _cov:
        from agent.layer_depth import set_depth as _set_depth

        if isinstance(_cov, dict) and len(_cov) > 1:
            # Multi-stack: set TT_PERF_STACK{N}_LAYERS for each stack in sorted order.
            _profile_extra: dict = {}
            for _i, (_sid, _depth) in enumerate(sorted(_cov.items())):
                _stack_key = _stack_layers_var(_i)
                _cov_env[_stack_key] = str(_depth)
                _profile_extra[_stack_key] = str(_depth)
            # Merge per-stack vars into PERF_MCP_PROFILE_ENV so the tracy subprocess sees them.
            try:
                _existing_prof = json.loads(_cov_env.get("PERF_MCP_PROFILE_ENV") or "{}")
            except (ValueError, TypeError):
                _existing_prof = {}
            _existing_prof.update(_profile_extra)
            _cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(_existing_prof)
            _cov_repr = ", ".join(
                f"TT_PERF_STACK{_i}_LAYERS={_d}" for _i, (_sid, _d) in enumerate(sorted(_cov.items()))
            )
            print(f"  [optimize/cc] coverage-sized profiling window (multi-stack): {_cov_repr}")
            # THE FULL-MODEL SIGNAL THE COVERAGE PROBE ALREADY MEASURED. Without it the bridge has
            # no baseline to compare its capped run against, and falls back to probing for one --
            # which 2026-07-19 removed for the other caller as "a fragile 2nd detection probe",
            # wiring before_loop to pass full_hint from exactly these facts. This call site predates
            # that by a day and was never brought along, so it has been running the bridge blind
            # ever since.
            _depth_env = _bridge_depth_env(
                repo_root,
                _cov_env,
                devices,
                pipe.get("perf_test"),
                pipe.get("case"),
                _cov,
                full_hint=int((_cov_facts or {}).get("full_signal") or 0),
                full_blocks=int((_cov_facts or {}).get("full_blocks") or 0),
            )
            if _depth_env:
                try:
                    _ep2 = json.loads(_cov_env.get("PERF_MCP_PROFILE_ENV") or "{}")
                except (ValueError, TypeError):
                    _ep2 = {}
                _ep2.update(_depth_env)
                _cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(_ep2)
        else:
            # Single-stack (dict with 1 entry, or plain int): use existing TT_PERF_LAYERS convention.
            _cov_single = next(iter(_cov.values())) if isinstance(_cov, dict) else _cov
            _set_depth(_cov_env, _cov_single)
            print(
                f"  [optimize/cc] coverage-sized profiling window: TT_PERF_LAYERS={_cov_single} (covers all block types)"
            )
            _depth_env = _bridge_depth_env(
                repo_root,
                _cov_env,
                devices,
                pipe.get("perf_test"),
                pipe.get("case"),
                _cov_single,
                full_hint=int((_cov_facts or {}).get("full_signal") or 0),
                full_blocks=int((_cov_facts or {}).get("full_blocks") or 0),
            )
            if _depth_env:
                _cov_env["PERF_MCP_PROFILE_ENV"] = json.dumps(_depth_env)
    tools = list(_ALLOWED_TOOLS)
    hitl_dir = None
    if hitl:
        hitl_dir = tempfile.mkdtemp(prefix=f"hitl_{model_name}_{task}_")
        _cov_env["PERF_MCP_HITL_DIR"] = hitl_dir
        tools = [t for t in _ALLOWED_TOOLS if not (t.endswith("git_commit") or t.endswith("git_revert"))]
        tools.append("mcp__perf-mcp__hitl_gate")
    cfg_path = repo_root / CC_DIR / f".mcp_config_{model_name}_{task}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    prompt = (_HITL_PROMPT if hitl else _PROMPT).format(model=model_name, task=task, metric=metric)
    start_sha = _git(repo_root, "rev-parse", "HEAD")
    mcp_env = cfg["mcpServers"]["perf-mcp"]["env"]
    _reset_fullpipe_baselines()
    # The BEFORE bookend is a full-model run of several minutes AND it defines the bar every win is
    # graded against. If this (model, task) already has one, re-measuring it can only move the bar to
    # whatever the board felt like doing today -- which is exactly how a clamped 68.3241 ms replaced a
    # true ~34. Reuse it and skip the run.
    before_ms, before_mode = _read_fullpipe_best_1cq()
    if before_ms and before_ms > 0:
        print(
            "  [optimize/cc] FULL-model end-to-end (BEFORE) = %.4f ms REUSED from the established "
            "baseline (not re-measured; PERF_MCP_FORCE_REBASELINE=1 to re-take it)" % before_ms
        )
    else:
        before_ms, before_mode = _fullpipe_e2e(repo_root, mcp_env, devices, "BEFORE")
    rounds, can_stop, halted = 0, False, False
    stall_sec = adaptive_timer(repo_root, "round", env_key="PERF_MCP_ROUND_STALL_SEC", mult=0.5)
    max_wedge = int(os.environ.get("PERF_MCP_MAX_WEDGE_STRIKES", "2") or "2")
    wedge_strikes = 0
    auth_strikes = 0
    auth_recoveries = 0
    round_cmd = [
        _resolve_claude_bin(),
        "-p",
        prompt,
        "--mcp-config",
        str(cfg_path),
        "--strict-mcp-config",
        "--allowedTools",
        *tools,
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    _stop_watcher = threading.Event()
    _wt = None
    if hitl:
        _wt = threading.Thread(target=_hitl_watch, args=(repo_root, hitl_dir, _stop_watcher), daemon=True)
        _wt.start()
        print(f"  [optimize/cc] HITL on — pausing at each lever for your commit/revert/try (handshake {hitl_dir})")
    while rounds < max_rounds:
        st = _gate_status(repo_root, mcp_env, devices)
        if st.get("halt"):
            _remedy = _HALT_REMEDY.get(st.get("kind") or "") or _HALT_REMEDY[""]
            print(f"  [optimize/cc] HALT — {_remedy}: {st.get('reason') or '(no reason reported)'}")
            halted = True
            break
        if st.get("can_stop"):
            can_stop = True
            break
        wedged = _run_round_with_watchdog(round_cmd, repo_root, devices, kernel_log, stall_sec)
        # A ROUND THAT WAS NEVER LET IN IS NOT A ROUND THAT FOUND NOTHING. A refused credential
        # produces a round that runs, writes a transcript and exits cleanly having done nothing,
        # which the loop cannot tell from an agent that looked and found no win -- so it spent all
        # ten rounds of a 7h37m run on it and reported "no kernel attempts recorded", which reads as
        # "already optimal" rather than "nobody was allowed in".
        #
        # RECOVER, THEN CARRY ON -- the same shape as the device reclaim below, because the run's
        # measured baseline and ladder state are just as expensive to rebuild after a credential
        # blip as after a wedge. The client renews an expired token from its own refresh token when
        # something asks it to, so asking is the whole recovery; a round is only lost when even that
        # is refused, and then the run stops rather than spending its remaining rounds being told no.
        # BUDGET BEFORE CREDENTIALS, because the remedies are opposites. Renewing an account that is
        # merely out of budget "succeeds" -- the credential is valid -- so the round is retried, the
        # retry is refused again, and the run drains its recovery budget before reporting that it
        # could not authenticate. Sending an operator to re-login over a spent quota is the wrong
        # answer to the wrong question. Waiting is not this tool's call either: a reset can be hours
        # away and the run holds the device throughout, so it stops and quotes the limit.
        _spent = _agent_quota_exhausted(kernel_log)
        if _spent:
            print(
                "  [optimize/cc] STOPPING: the agent is out of budget (%s), not out of credentials — "
                "renewing cannot help and retrying only spends more. The remaining rounds are not run; "
                "the measured baseline is kept, so restarting after the limit resets costs nothing." % _spent,
                flush=True,
            )
            halted = True
            break
        _auth = _agent_auth_failure(kernel_log)
        if _auth:
            print("  [optimize/cc] round %d was refused (%s) — recovering" % (rounds + 1, _auth), flush=True)
            # BOUNDED, because a retry that does not consume a round cannot be unbounded: if renewal
            # keeps succeeding while the round keeps being refused, an unbounded `continue` spins
            # forever without ever finishing the run. Past the budget the refusal is treated as
            # permanent and the round is spent, so the loop always makes progress.
            if _recover_agent_auth() and auth_recoveries < _MAX_AUTH_RECOVERIES:
                auth_recoveries += 1
                print(
                    "  [optimize/cc] credential renewed (%d/%d) — re-running this round"
                    % (auth_recoveries, _MAX_AUTH_RECOVERIES),
                    flush=True,
                )
                auth_strikes = 0
                continue
            auth_strikes += 1
            if auth_strikes >= _MAX_AUTH_STRIKES:
                print(
                    "  [optimize/cc] STOPPING: the agent cannot authenticate (%s) and renewing did not "
                    "help. Nothing was tried and nothing can be, so the remaining rounds are not run. "
                    "Sign in again (`claude /login`) and start this run over." % _auth,
                    flush=True,
                )
                halted = True
                break
            print("  [optimize/cc] still refused — one more attempt before stopping", flush=True)
        else:
            auth_strikes = 0
        if wedged:
            wedge_strikes += 1
            if wedge_strikes >= max_wedge:
                print(
                    "  [optimize/cc] WATCHDOG: %d consecutive wedged rounds — reset device + continue in-process "
                    "(no restart; process healthy, chips reset); ladder state is preserved." % wedge_strikes,
                    flush=True,
                )
                print(
                    "  [optimize/cc] "
                    + _reclaim_device(devices, error_text=_tail_lines(str(kernel_log) + ".agent.log", 40)),
                    flush=True,
                )
                wedge_strikes = 0
        else:
            wedge_strikes = 0
        rounds += 1
    _stop_watcher.set()
    # AFTER bookend: reuse the best committed trace+1cq verdict (the per-lever gate already banked
    # the final number in the 1cq baseline file) — do NOT run the full pipeline again at the end.
    after_ms, after_mode = _read_fullpipe_best_1cq()
    if after_ms is not None:
        print(
            f"  [optimize/cc] FULL-model end-to-end (AFTER) = {after_ms:.1f} ms  "
            "(best committed trace+1cq; reused, no extra device run)"
        )
    if before_ms and after_ms:
        d = (before_ms - after_ms) / before_ms * 100.0
        print(
            f"  [optimize/cc] FULL-model end-to-end (ALL 52 layers): BEFORE {before_ms:.1f} ms -> "
            f"AFTER {after_ms:.1f} ms  ({d:+.1f}% {'faster' if d >= 0 else 'SLOWER'})"
        )
    try:
        _mf = json.loads(Path(manifest_path).read_text())
    except Exception:  # noqa: BLE001
        _mf = {}
    _print_scorecard(devices, _mf, pipe, _cov_facts, before_ms, after_ms, model_name)
    _emit_summary(
        repo_root,
        kernel_log,
        model_name,
        task,
        metric,
        start_sha,
        perf_test=(pipe or {}).get("perf_test", ""),
    )
    return {"task": task, "rounds": rounds, "can_stop": can_stop, "halted": halted}


_GL_REL = PERF_DIR + "/GUIDELINES"


def catalog_pull(repo_root: Path, remote: str, branch: str) -> None:
    """Best-effort: fetch the shared catalog branch and bring its GRADUATED_* knobs into the live
    GUIDELINES dir, so this run recalls the latest cross-model-proven knobs. Never raises."""
    try:
        f = subprocess.run(
            ["git", "fetch", remote, branch], cwd=str(repo_root), capture_output=True, text=True, timeout=180
        )
        if f.returncode != 0:
            print(f"  [catalog] pull skipped (no {remote}/{branch} yet)")
            return
        ls = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD", _GL_REL],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        n = 0
        for path in (ls.stdout or "").splitlines():
            if "/GRADUATED_" in path and path.endswith(".md"):
                blob = subprocess.run(
                    ["git", "show", f"FETCH_HEAD:{path}"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if blob.returncode == 0:
                    (repo_root / path).parent.mkdir(parents=True, exist_ok=True)
                    (repo_root / path).write_text(blob.stdout)
                    n += 1
        print(f"  [catalog] pulled {n} GRADUATED knob(s) from {remote}/{branch}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [catalog] pull error (ignored): {str(exc)[-120:]}")


def catalog_push(repo_root: Path, remote: str, branch: str) -> None:
    """Best-effort: commit the local GRADUATED_* knobs onto a TEMP WORKTREE of <branch> and push —
    isolated from the working branch so NO model-optimization commits are included. Never raises."""
    import shutil
    import tempfile

    try:
        gl = repo_root / _GL_REL
        grads = sorted(gl.glob("GRADUATED_*.md"))
        if not grads:
            print("  [catalog] nothing to push (no GRADUATED knobs).")
            return
        subprocess.run(
            ["git", "fetch", remote, branch], cwd=str(repo_root), capture_output=True, text=True, timeout=180
        )
        has_remote = (
            subprocess.run(
                ["git", "rev-parse", "--verify", "FETCH_HEAD"], cwd=str(repo_root), capture_output=True, text=True
            ).returncode
            == 0
        )
        wt = tempfile.mkdtemp(prefix="cc_catalog_")
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", wt, *(["FETCH_HEAD"] if has_remote else [])],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if has_remote:
                subprocess.run(["git", "checkout", "-B", branch], cwd=wt, capture_output=True, text=True, timeout=300)
            else:
                subprocess.run(
                    ["git", "checkout", "--orphan", branch], cwd=wt, capture_output=True, text=True, timeout=300
                )
                subprocess.run(["git", "rm", "-rf", "."], cwd=wt, capture_output=True, text=True, timeout=300)
            dest = Path(wt) / _GL_REL
            dest.mkdir(parents=True, exist_ok=True)
            for g in grads:
                shutil.copy2(g, dest / g.name)
            subprocess.run(["git", "add", _GL_REL], cwd=wt, capture_output=True, text=True, timeout=300)
            c = subprocess.run(
                ["git", "commit", "-m", f"[perf-catalog] graduated knobs ({len(grads)})"],
                cwd=wt,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if c.returncode != 0:
                print("  [catalog] no new graduated knobs to push.")
            else:
                p = subprocess.run(
                    ["git", "push", remote, f"HEAD:{branch}"], cwd=wt, capture_output=True, text=True, timeout=240
                )
                print(
                    f"  [catalog] push {'ok' if p.returncode == 0 else 'FAILED'}: "
                    f"{(p.stderr or p.stdout).strip()[-140:]}"
                )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [catalog] push error (ignored): {str(exc)[-140:]}")


_HF_ID_RE = __import__("re").compile(r"['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")


def _hf_hub_root() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"


def _is_cached_model_id(cand) -> bool:
    if not cand or "/" not in str(cand):
        return False
    org, _, name = str(cand).partition("/")
    return (_hf_hub_root() / f"models--{org}--{name}").is_dir()


def _run_test_files(demo_dir, manifest=None) -> tuple:
    """The test files THIS RUN executes, absolute. Empty when the manifest cannot name them.

    These are what make _resolve_model_id a fact rather than a scan: the model pinned in the PCC test
    and the perf test is the model being run, whatever else the directory happens to mention."""
    out = []
    try:
        # No manifest means the caller genuinely has none (a probe, a unit test). Returning ()
        # sends _resolve_model_id to its tree scan, which is the documented last resort -- it does
        # NOT go hunting for a manifest, because a function that finds its own inputs is a function
        # nobody can reason about from the call site.
        cfg = ((manifest or {}).get("config") or {}) if isinstance(manifest, dict) else {}
        pm = ((manifest or {}).get("pathmap") or {}) if isinstance(manifest, dict) else {}
        cands = [cfg.get("pcc_test"), cfg.get("perf_test")]
        for key in ("pcc", "perf_test"):
            v = pm.get(key)
            if isinstance(v, str):
                cands.append(v)
            elif isinstance(v, dict):
                cands.append(v.get("path"))
            elif isinstance(v, (list, tuple)):
                cands.extend(x for x in v if isinstance(x, str))
        for c in cands:
            if not c:
                continue
            p = Path(str(c).partition("::")[0])
            for base in (
                Path(demo_dir),
                Path(demo_dir).parents[2] if len(Path(demo_dir).parents) > 2 else Path(demo_dir),
            ):
                q = p if p.is_absolute() else (base / p)
                if q.is_file():
                    out.append(str(q))
                    break
    except Exception:  # noqa: BLE001
        return ()
    return tuple(dict.fromkeys(out))


def _resolve_model_id(demo_dir, hint=None, prefer_files=()) -> str | None:
    """Which HF model this run is optimizing: hint, then HF_MODEL, then the directory scan.

    The middle tier was missing. optimize.py passes `model_id_hint=(None if model_dir else
    args.target)`, so pointing at a DEMO DIRECTORY -- which is how every brought-up model is
    optimized -- nulls the hint and drops straight to the scan. The scan returns the first cached id
    found in any .py under the model dir, and gemma3's tree names three variants: conftest.py pins
    google/gemma-3-12b-it, test_ci_dispatch.py mentions the 4b and the 27b. It returned the 4b, so
    every derived figure inherited a 4B model -- ceiling 102.4 tok/s/u instead of 34.1, band
    61.4-81.9 instead of 20.5-27.3, utilisation 28% for a run that was actually at 84%.

    Nothing was mismeasured: the tool runs a pytest node and the MODEL resolves its own identity from
    HF_MODEL, so execution never needs this id. Only the roofline arithmetic does -- and HF_MODEL was
    sitting in this process's own environment, correct, unread. before_loop.py:262 already reads it
    (`config.get("model_id") or os.environ.get("HF_MODEL")`); this gives the cc engine the same tier.
    """
    if _is_cached_model_id(hint):
        return hint
    env_id = (os.environ.get("HF_MODEL") or "").strip()
    if _is_cached_model_id(env_id):
        return env_id
    # THE RUN'S OWN TESTS, BEFORE THE TREE. This dropped straight to "first cached id in the first
    # .py rglob yields", and a model tree names more than one: gemma3's conftest, perf test, PCC test
    # and host-split test all pin google/gemma-3-12b-it, while test_ci_dispatch.py lists the 4b and
    # the 27b as a CI matrix. It returned the 4b -- because rglob reached that file first -- and every
    # derived figure then described a 4B model.
    #
    # Counting which id appears in the most files was the first attempt and was rejected: a vote is a
    # heuristic, it has no reason to be right on a tree nobody has seen, and the stress suite says so.
    # The run already KNOWS which files it executes -- the PCC test and the perf test are named in its
    # own config -- and the model those files pin is the model being run. That is a fact about this
    # run, not a property of the directory layout.
    for _f in prefer_files or ():
        try:
            _txt = Path(str(_f).partition("::")[0]).read_text(errors="ignore")
        except OSError:
            continue
        for cand in _HF_ID_RE.findall(_txt):
            if _is_cached_model_id(cand):
                return cand
    # Last resort: the tree. Unchanged -- first cached id wins -- because with nothing stating which
    # model this is, one answer from the family beats no answer, and the callers above are what make
    # it rare rather than what it falls back to.
    try:
        for p in sorted(Path(demo_dir).rglob("*.py")):
            try:
                txt = p.read_text(errors="ignore")
            except OSError:
                continue
            for cand in _HF_ID_RE.findall(txt):
                if _is_cached_model_id(cand):
                    return cand
    except Exception:  # noqa: BLE001
        return None


def _chip_count(devices) -> int:
    d = (devices or "").strip().lower()
    if d and d not in ("all", "single"):
        return max(1, len([x for x in d.split(",") if x.strip()]))
    if d == "single":
        return 1
    try:
        import ttnn

        return max(1, int(ttnn.GetNumAvailableDevices()))
    except Exception:
        return 1


def _hf_snapshots(model_id: str) -> list:
    org, _, name = model_id.partition("/")
    snaps = _hf_hub_root() / f"models--{org}--{name}" / "snapshots"
    try:
        return sorted([d for d in snaps.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    except Exception:
        return []


def _hf_cache_weight_bytes(model_id: str) -> int:
    best = 0
    for snap in _hf_snapshots(model_id):
        total = 0
        for p in snap.iterdir():
            if p.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth"):
                try:
                    total += os.path.getsize(os.path.realpath(p))
                except OSError:
                    pass
        best = max(best, total)
    return best


def _hf_cache_dims(model_id: str) -> dict:
    for snap in _hf_snapshots(model_id):
        cfg = snap / "config.json"
        if cfg.is_file():
            try:
                return json.loads(cfg.read_text())
            except Exception:
                continue
    return {}


def _model_weight_bytes(demo_dir, hint=None, manifest=None) -> int:
    # _captured/ (golden PCC input/output tensors) and _stubs/ (graduated-stub .last_good snapshots)
    # are TEST FIXTURES, not model weights. Summing them made a tiny non-weight total short-circuit
    # the real checkpoint lookup -- XTTS's 102 captured *.pt files (133 MB) masked the true 1.868 GB
    # model.pth, so the roofline ceiling divided by a 14x-too-small size. Skip fixture dirs; when the
    # demo ships no real local weights the total is 0 and we fall through to the HF-cache checkpoint.
    total = 0
    _skip = {"_captured", "_stubs"}
    try:
        for p in Path(demo_dir).rglob("*"):
            if p.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth") and p.is_file():
                if _skip & set(p.parts):
                    continue
                total += p.stat().st_size
    except Exception:
        total = 0
    if total:
        return total
    mid = _resolve_model_id(demo_dir, hint, _run_test_files(demo_dir, manifest))
    return _hf_cache_weight_bytes(mid) if mid else 0


def _decide_parallelism_route(
    demo_dir, manifest, repo_root=None, metric="device_ms", devices="all", model_id_hint=None
) -> None:
    """Decide single-chip vs tensor-parallel from model size + detected hardware, print the route, and
    (when the model does not fit on one chip) export TT_PERF_TP_REGIME=1 to the loop automatically.
    Fully fail-safe: any missing input leaves the regime OFF, so a run is byte-identical to today
    unless TP is positively selected."""
    try:
        import sys

        _perf = str(Path(repo_root) / PERF_DIR) if repo_root else str(Path(__file__).resolve().parent.parent)
        if _perf not in sys.path:
            sys.path.insert(0, _perf)
        from agent.environment import ARCH_FACTS
        from agent.tp import decide_parallelism

        env = manifest.get("env", {}) or {}
        arch = (env.get("arch") or "").lower()
        facts = ARCH_FACTS.get(arch, {})
        cap = int(os.environ.get("TT_PERF_DRAM_CAPACITY_BYTES") or facts.get("dram_capacity_bytes") or 0)
        chips = int(env.get("device_count") or env.get("mesh_chips") or env.get("num_devices") or 0) or _chip_count(
            devices
        )
        weight_bytes = _model_weight_bytes(demo_dir, model_id_hint, manifest)
        if not (cap and weight_bytes):
            return
        cfg = manifest.get("model_config") or {}
        if not cfg.get("hidden_size"):
            mid = _resolve_model_id(demo_dir, model_id_hint, _run_test_files(demo_dir, manifest))
            if mid:
                cfg = {**_hf_cache_dims(mid), **cfg}
        heads = int(cfg.get("num_attention_heads") or cfg.get("num_heads") or 1)
        hidden = int(cfg.get("hidden_size") or cfg.get("d_model") or 1)
        route = decide_parallelism(weight_bytes, cap, chips, heads, hidden, metric)
        print(f"  [optimize/cc] parallelism route: {route['route']} — {route['reason']}")
        if route.get("tp_regime"):
            os.environ["TT_PERF_TP_REGIME"] = "1"
            os.environ["TT_PERF_TP_FLOOR"] = str(route.get("floor", 1))
            print("  [optimize/cc] tensor-parallel regime ENABLED; propagated to loop")
    except Exception as exc:  # never fail the run on the route decision
        print(f"  [optimize/cc] parallelism route decision skipped ({exc})")


def _env_params(var: str) -> int:
    """Param count from an env override, accepting a bare count or a "3B"/"800M" shorthand."""
    raw = (os.environ.get(var) or "").strip()
    if not raw:
        return 0
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([BbMm])?", raw)
    if not m:
        return 0
    scale = {"b": 1e9, "m": 1e6}.get((m.group(2) or "").lower(), 1.0)
    return int(float(m.group(1)) * scale)


def _params_from_model_id(model_id: str) -> tuple[int, int]:
    """(total_params, active_params) as published by the model NAME, or (0, 0).

    Model ids carry the size the card advertises: "Llama-3.1-8B" -> 8B total; a MoE id carries the
    ACTIVE count after an A, "NVIDIA-Nemotron-3-Nano-30B-A3B" -> 30B total / 3B active. That naming is
    the number a team quotes when saying what a model's ceiling should be, so it is the right fallback
    when the checkpoint headers cannot be read.
    """
    s = str(model_id or "")
    active = 0
    m_act = re.search(r"[-_]A([0-9]*\.?[0-9]+)\s*([BbMm])", s)
    if m_act:
        active = int(float(m_act.group(1)) * (1e9 if m_act.group(2).lower() == "b" else 1e6))
    total = 0
    # Last size-looking token wins ("Llama-3.2-11B-Vision" -> 11B); the A-suffix match is skipped so a
    # MoE id never reports its ACTIVE count as the total.
    for m in re.finditer(r"(?<![A-Za-z0-9.])([0-9]*\.?[0-9]+)\s*([BbMm])(?![A-Za-z0-9])", s):
        if m_act and m.start() == m_act.start() + 1:
            continue
        total = int(float(m.group(1)) * (1e9 if m.group(2).lower() == "b" else 1e6))
    return total, active


def _perf_target_inputs(demo_dir, model_id_hint, manifest) -> dict | None:
    """The weight-bytes-per-token facts the DECODE roofline needs, or None when they cannot be known.

    perf_target.compute_target implements the standard decode bound -- peak DRAM bandwidth divided by
    the bytes that must be streamed per token, with 60-80% of peak as the achievable band. For
    Llama-3.1-8B on a 512 GB/s part: 512/8 = 64 tok/s/u ceiling, 38-51 tok/s/u achievable.

    It reads those facts from perf_target_inputs.json, and NOTHING in the tool wrote that file, so
    active_bytes was always 0 and every report fell back to the Sigma-per-op ms floor -- a far weaker
    statement, and one that moves whenever the op mix changes. This produces it, from the same
    checkpoint size and HF config the parallelism route already reads.

    MoE is deliberately excluded: the reachable read set is shared + top_k x per-expert bytes, and the
    per-expert split cannot be taken from config alone without guessing the FFN shapes. A guessed
    ceiling is worse than the floor fallback, so those models keep the floor.
    """
    # THE CEILING NEEDS A PARAM COUNT AND NOTHING ELSE. These two gates rejected models over inputs the
    # formula does not consult: `weight_bytes` is only a fallback divisor now that params drive it, and
    # `cfg` supplies the KV terms, which are unused unless a seq_len is given. So a model with no HF
    # config, or whose weights are in a format the byte-sizer cannot read, got NO ceiling at all and its
    # report fell to the band-less ms floor -- three unrelated-looking symptoms with one cause.
    wb = _model_weight_bytes(demo_dir, model_id_hint, manifest) or 0
    cfg = dict(manifest.get("model_config") or {})
    mid = _resolve_model_id(demo_dir, model_id_hint, _run_test_files(demo_dir, manifest))
    if mid:
        cfg = {**_hf_cache_dims(mid), **cfg}
    experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts")
    src = "checkpoint bytes + HF config"
    analytic_params = 0
    _unit = ""  # bound before the try below, which can raise before assigning it (params_basis reads it)
    # ANALYTIC FIRST: every tensor's shape and dtype from the safetensors header, with the on-device
    # widths applied per name pattern. The checkpoint's FILE SIZE counts the stored dtype -- 15.0 GB of
    # bf16 for Llama-3.1-8B, where the device streams 6.09 GB as bfp4/bfp8 -- so it understates the
    # ceiling by 2.4x. Falls through to the file size only when the headers cannot be read.
    try:
        from agent import model_bytes as _mb

        # THE OBSERVED UNIT HERE TOO. This exclusion drops the embedding table from the streamed
        # bytes, because a token unit reads it by INDEX -- one row per token -- and counting the whole
        # table overstates a decode step (gemma3's is 262144 x 3840, about 1 GB against 11 GB). But
        # whether the model IS a token unit was decided by the tag, which is the same defect the
        # ceiling just stopped making: Kokoro-82M is tagged text-to-speech, reads as `token`, and has
        # no token loop at all -- so its tables would be excluded from a byte count they belong in.
        # The tag remains only for the window before the first trace, where nothing is observed yet.
        _unit = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT") or "").strip().lower() or _mb.unit_for_tag(
            cfg.get("pipeline_tag") or (manifest.get("model_meta") or {}).get("pipeline_tag") or ""
        )
        _snap = _hf_snapshots(mid)[0] if mid and _hf_snapshots(mid) else None
        # THE PARAM COUNT DOES NOT DEPEND ON THE UNIT -- only the lookup-only exclusion does -- so the
        # header walk runs whenever a snapshot is readable. Gating it on `_unit` too meant a model whose
        # unit could not be determined (bge-large-en-v1.5) silently fell back to the checkpoint's FILE
        # SIZE as the divisor: 1.34 GB of float32 instead of its param count, i.e. ~4 B/param, so the
        # xB -> xGB rule was bypassed for exactly the models least able to report the error themselves.
        if _snap:
            _an = _mb.weight_bytes(
                _snap,
                # Unknown unit -> count as "token", which EXCLUDES lookup-only tensors. One row of an
                # embedding table crosses the bus per token, never the whole matrix, and that is true of
                # an encoder pass too -- so excluding it is right for any unit, and erring toward a
                # smaller divisor errs toward a HIGHER ceiling, which merely fails to bind rather than
                # stopping a run early.
                unit=_unit or "token",
                overrides=_mb.overrides_from_env(),
                default_device_dtype=os.environ.get("TT_PERF_DEFAULT_WEIGHT_DTYPE", ""),
            )
            if _an.get("bytes") and _unit:
                wb, src = _an["bytes"], "analytic: %d tensors from safetensors headers, unit=%s" % (
                    _an["tensors"],
                    _unit,
                )
            # SAY IT WHEN THE NAME LIST MISSED A TOWER. _TOWER_ONLY keeps an encoder out of the
            # per-token read set by matching names; stage_roots says, from the checkpoint's own key
            # structure, that a non-recurring stage runs out of some section. When the second names a
            # section the first did not exclude, that encoder is being charged to every token: the
            # divisor is inflated, the ceiling lowered, and the run looks nearer the wall than it is.
            #
            # Reported, not corrected. Substituting stage_roots for the name list was tried and is
            # worse -- it drops lm_head on any untied model, which errs the other way and by more.
            # The real answer is an observation of what a token reads, and the profile records no
            # phase per op, so it does not exist yet. Until it does, the choice is between guesses,
            # and a visible wrong number beats a silent one.
            try:
                _sr = (read_arch_mirror() or {}).get("stage_roots") or {}
                if not _sr:
                    _pf = Path(demo_dir) / "perf_target_inputs.json"
                    if _pf.is_file():
                        _sr = (json.loads(_pf.read_text()) or {}).get("stage_roots") or {}
                _missed = _mb.untowered_sections(_snap, _sr) if _sr else []
                if _missed:
                    print(
                        "  [optimize/cc] WARNING: %s runs a separate stage but is counted in the "
                        "per-%s read set -- the tower name list does not know it. The ceiling is "
                        "PESSIMISTIC (divisor too large), so at-floor will read high. Add the name to "
                        "model_bytes._TOWER_ONLY, or read the per-stage rows instead of the "
                        "model-level one." % (", ".join(_missed), _unit or "unit"),
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception:  # noqa: BLE001 -- a warning must never cost a ceiling
                pass
            # EXACT param count, free: the header walk already sums numel. This is the params-based
            # ceiling's input, and it does not depend on the width the device serves.
            if _an.get("params"):
                analytic_params = int(_an["params"])
    except (NameError, AttributeError, TypeError) as _bug:
        # A BUG IS NOT AN UNREADABLE CHECKPOINT, and this block swallowed both identically.
        #
        # `except Exception: pass` here is right for what it was written for: a truncated shard, a
        # dtype with no width, an unreadable header -- environmental failures where falling through
        # to the file size is the documented weaker-but-not-wrong answer. It is wrong for a
        # programming error, because the fall-through then hides it perfectly: a NameError on one
        # line deleted total_params for the WHOLE run and the only symptom was a ceiling computed
        # from the checkpoint's file size instead of its param count. That is a 2.4x error on
        # Llama-3.1-8B, printed with no warning and no traceback. One pre-existing test caught it;
        # nothing in a real run would have.
        #
        # Still not raised -- a ceiling must never cost a run -- but it is now SAID, with the
        # exception named, so the next one is visible in the first second instead of inferred from a
        # number being oddly large.
        print(
            "  [optimize/cc] BUG in the analytic byte walk (%s: %s) -- falling back to the "
            "checkpoint's file size, which counts the STORED dtype and overstates the divisor. "
            "This is a defect in the tool, not in the model." % (type(_bug).__name__, str(_bug)[:200]),
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001 -- an unreadable checkpoint falls through to the file size
        pass
    override = (os.environ.get("TT_PERF_WEIGHT_BYTES") or "").strip()
    if override:
        # THE BYTES THAT ACTUALLY STREAM, when they are known to differ from the checkpoint. The
        # ceiling is peak_BW / bytes-per-token, and the bytes are the ones the DEVICE reads: a bf16
        # checkpoint served as bf8_b weights halves them, which doubles the ceiling. Llama-3.1-8B is
        # exactly this case -- 16.06 GB on disk, 8 GB resident, 512/8 = 64 tok/s/u rather than 31.9 --
        # so a run that quantises weights must be able to say so instead of being judged against the
        # dtype its checkpoint happens to be stored in.
        try:
            _ov = float(override)
            if _ov > 0:
                wb, src = _ov, "TT_PERF_WEIGHT_BYTES (on-device weight bytes)"
        except (TypeError, ValueError):
            pass
    facts = {
        "weight_bytes": int(wb or 0),
        "dominant_dtype": str(cfg.get("torch_dtype") or "bfloat16"),
        "source": src,
    }
    # PARAMS drive the ceiling (xB -> xGB). Exact count from the headers when readable, else the count
    # the model NAME publishes; for MoE the A-suffix ("30B-A3B") is the ACTIVE count, which is the read
    # set a routed token streams.
    name_total, name_active = _params_from_model_id(mid or "")
    total_params = analytic_params or name_total
    if total_params:
        facts["total_params"] = int(total_params)
        # SAY WHICH COUNT THIS IS. The key is named total_params and, when the header walk supplied
        # it, the value is NOT a total: model_bytes counts the READ SET for the observed unit, which
        # for a token unit deliberately drops lookup-only tensors (an embedding table is read one row
        # at a time) and tower-only tensors (an encoder runs per clip, not per token).
        #
        # On Voxtral-Mini-3B-2507 that reads: total_params 3.611B, under two blocks declaring 0.637B
        # and 4.014B. A total smaller than its own parts is alarming to anyone checking the ceiling,
        # and the only way to find out it was correct was to rediscover both exclusions in
        # model_bytes. 3.611B is exactly 4.014B (language_model) minus 0.403B (embedding); the
        # checkpoint holds 4.676B.
        #
        # Nothing is renamed here: `total_params` is read by perf_target.ceiling_params,
        # simple_active_bytes and two places in summary, and by any perf_target_inputs.json already
        # on disk. Renaming buys a clearer word and costs a compatibility break. Stating the basis
        # costs one string and removes the ambiguity outright -- and a reader that does not know the
        # basis cannot tell a read set from a total, which is the actual failure.
        facts["params_basis"] = (
            "read set for unit=%s: lookup-only and tower-only tensors excluded" % (_unit or "unknown")
            if analytic_params
            else "count published by the model name (no readable checkpoint headers)"
        )
    if experts:
        facts["is_moe"] = True
        active = _env_params("TT_PERF_ACTIVE_PARAMS") or name_active
        if active:
            facts["active_params"] = int(active)
        else:
            # No active count = no honest MoE ceiling. Total params would overstate the read set by
            # experts/top_k, making the ceiling far too low and every run read ABOVE_BAND. Returning
            # None keeps the ms-floor fallback: weaker, but not wrong.
            return None
    # ONLY THE OBSERVATION SETS THE UNIT. It used to fall back to a table keyed on the HF
    # pipeline_tag, and a wrong unit does not degrade the ceiling -- it puts it in the wrong currency
    # entirely, then the band, the at-floor verdict and the headline rate all inherit that. A tag
    # names the TASK and cannot state whether a model loops: `text-to-speech` covers XTTS, which emits
    # tokens, and Kokoro, which is StyleTTS2 and produces a whole waveform in one pass, so the table
    # has to pick and is wrong for the other. HunyuanImage-3.0, tagged text-to-image but
    # autoregressive, is the same failure the other way.
    #
    # No observation yet means NO UNIT, which means no unit ceiling and the run lands on the ms-floor
    # form until the first trace reports. That is the rule the rest of this code already follows --
    # _anchored_ceiling_facts: "No recoverable unit means no ceiling, which lands on the floor
    # fallback: weaker, but not wrong." The table contradicted it; now it does not.
    #
    # The tag is still read ABOVE, for default_conditions: ISL/OSL/steps/resolution must be chosen
    # BEFORE anything runs, so nothing observed can supply them, and a wrong guess there is visible in
    # the scorecard rather than silently rescaling the ceiling.
    _observed = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT") or "").strip().lower()
    if _observed:
        facts["unit"] = _observed
    # RESOLUTION IS A MEASUREMENT CONDITION, and for a step or vision unit it IS the work: a denoise
    # step at 1024 is ~4x the step at 512, so two runs of one model differ ~4x. emit-e2e already reads
    # image_size to build its PCC input; the perf side never received it, so a steps/s figure could
    # not say what it described. Recorded here rather than guessed at render time, and left absent for
    # text models, where there is no such condition to state.
    try:
        from agent.model_bytes import resolution_from_config as _rfc

        _res = int(os.environ.get("TT_PERF_RESOLUTION") or 0) or _rfc(cfg)
        if _res:
            facts["resolution"] = int(_res)
    except Exception:  # noqa: BLE001
        pass
    # A CONFIG VALUE MAY BE A LIST. Per-layer configs carry lists where a scalar is expected, and raw
    # int() on one raises TypeError -- which the caller swallows, so the model lost its ENTIRE ceiling
    # over a KV-cache field the ceiling does not even need without a seq_len. perf_target._scalar
    # already coerces exactly this (per-layer top_k), so reuse it instead of a second rule here.

    # FLAT FIRST, then the nested walk. _scalar coerces a per-layer LIST into a scalar, which the
    # walk deliberately will not (int(list) is not a depth), so routing the flat keys through the walk
    # dropped a list-valued num_hidden_layers on the floor. The walk is the fallback that catches the
    # nested case -- gemma3 declares it under text_config and a flat .get() reads 0 for every
    # multimodal config, and 0 layers feeds the roofline facts.

    # GEOMETRY IS A PROPERTY OF A BLOCK, NOT OF A MODEL, so it is no longer collected as loose keys.
    #
    # WHAT THIS REPLACES. Two rules picked from the same config, independently:
    #
    #     layers            = _depth_from_mapping(cfg)   "the DEEPEST depth anywhere"  -> 32
    #     hidden/intermediate = first sub-config whose key says neither "vision" nor "audio" -> 3072/8192
    #
    # On voxtral that is the AUDIO tower's depth welded to the LANGUAGE tower's widths: a 32-layer,
    # 3072-wide model that does not exist. Every stage then divided those numbers, so the audio
    # encoder was priced at 0.041 ms against a 12.80 ms measurement -- 312x -- and prefill's
    # activation term used a width its own tower does not have.
    #
    # The name blacklist was the tell. "vision"/"audio" is a list of towers someone had seen, and it
    # decides geometry for every model that has any. A tower called vocoder, denoiser or projector
    # walks straight through it.
    #
    # Blocks are read WHOLE instead: tower_geometry keys each tower by its DEPTH, which is the one
    # number the checkpoint's sections, the config's sub-dicts and the probe's stacks all agree on,
    # so a stage reaches its own geometry by structure -- stage -> root -> depth -> geometry -- with
    # nothing recognised by name.
    _blocks = _model_block_facts(demo_dir, mid or "", cfg, profile=_last_baseline_profile())
    if _blocks:
        facts["blocks"] = _blocks
    # THE FLAT KEYS SURVIVE FOR EXACTLY ONE SHAPE: a model with a single block, where "the model's
    # geometry" and "that block's geometry" are the same sentence and every existing caller stays
    # correct. With two or more, no flat answer exists -- emitting one is what produced the chimera --
    # so they are omitted and a caller that has not learned about blocks gets nothing rather than a
    # number from the wrong tower. Missing degrades to a refused ceiling; wrong degrades to 312x.
    if len(_blocks or {}) == 1:
        _only = next(iter(_blocks.values()))
        for key in ("layers", "kv_heads", "head_dim", "hidden_size", "intermediate_size"):
            if _only.get(key):
                facts[key] = int(_only[key])
    # A DIVISOR IS THE ONE THING THAT CANNOT BE MISSING. With neither a param count nor a byte count there
    # is nothing to divide by, so returning facts would produce a zero ceiling that renders as a real one.
    if not facts.get("total_params") and not facts.get("active_params") and not facts.get("weight_bytes"):
        return None
    return facts


ARCH_MIRROR_NAME = "model_blocks.json"
ARCH_KEYS = ("blocks", "stage_roots")


def _mirror_arch_facts(facts: dict) -> None:
    """Keep the ARCHITECTURE MAP somewhere git_revert cannot delete.

    WHAT GETS LOST AND WHY. perf_target_inputs.json is untracked and lives in the model directory,
    and git_revert calls gitio.remove_new_untracked -- which deletes untracked files created since
    the checkpoint. That is deliberate and correct: `git checkout <sha> -- <path>` only rewrites
    TRACKED files, so a lever that CREATED a kernel module would otherwise survive every revert. The
    facts file looks exactly like such a file, so every rejected attempt deletes it.

    Most of it heals. The flat keys are re-derived from the checkpoint by the rebuild, and the census
    is re-measured by the next full-pipeline run. Two do NOT: `blocks` needs a resolvable model id,
    and `stage_roots` is never written here at all -- discovery merges it in once, and nothing
    re-runs that. So after the first revert a stage can no longer find its own tower.

    Measured on Voxtral run 16: with both gone, _stage_block("encode") returned None and the compute
    roof fell back to the flat total_params -- 3.611B, the LANGUAGE model's per-token read set --
    for an audio encoder that holds 0.662B. Encode was charged 5.5x its work and read 46% of a
    ceiling it was really at ~8% of. Decode looked fine only by coincidence: total_params IS its
    read set, so the fallback happened to be its right answer.

    SAFE TO CACHE WITHOUT EXPIRY, unlike the census. These are ARCHITECTURE -- which towers exist,
    how deep, how many params -- read from the checkpoint. A dtype or grid knob changes the bytes on
    device; it cannot change how many towers the model has. The census is deliberately NOT mirrored
    here for exactly that reason: it must be re-measured, and a cached one would go stale the first
    time a precision knob lands.

    The same home and the same reasoning as summary._mirror_report: outside git so no revert can
    reach it, outside the worktree so no reboot can take it, keyed per model.
    """
    try:
        import os as _os

        # Same reasoning as read_arch_mirror: no import, the variable IS the contract.
        _sd = (_os.environ.get("PERF_MCP_STATE_DIR") or "").strip()
        if not _sd:
            return  # without --persist there is no durable directory to mirror into
        keep = {k: facts[k] for k in ARCH_KEYS if facts.get(k)}
        if not keep:
            return
        d = Path(_sd)
        d.mkdir(parents=True, exist_ok=True)
        prev = {}
        try:
            prev = json.loads((d / ARCH_MIRROR_NAME).read_text())
        except Exception:  # noqa: BLE001
            prev = {}
        merged = {**(prev if isinstance(prev, dict) else {}), **keep}
        (d / ARCH_MIRROR_NAME).write_text(json.dumps(merged, indent=2) + "\n")
        # PIN EACH STAGE'S PARAM COUNT, because this file is merged newest-wins and the compute
        # ceiling divides by it: 2 x params x tokens. Everything else the THEORETICAL column rests on
        # is anchored -- the peak, the read set, the item count -- and this was the last input still
        # re-derived from whatever the model currently looks like. A ceiling that moves while the run
        # works cannot be compared across rounds, which is the whole reason an anchor exists.
        #
        # The docstring above argues architecture is stable, and it usually is; the anchor is for the
        # case where it is not. Write-once, so the honest first answer wins over any later one.
        try:
            from cc_optimize import measurements as _led

            _blocks = (merged.get("blocks") or {}) if isinstance(merged, dict) else {}
            _roots = (merged.get("stage_roots") or {}) if isinstance(merged, dict) else {}
            for _stage, _root in _roots.items():
                _blk = _blocks.get(_root) or {}
                _mm = int(_blk.get("matmul_params") or 0)
                if not _mm:
                    _lo = int(_blk.get("lookup_params") or 0)
                    _mm = max(0, int(_blk.get("params") or 0) - _lo) if _lo else int(_blk.get("params") or 0)
                if _stage and _mm > 0:
                    _led.anchor(
                        _led.KIND_MATMUL_PARAMS,
                        float(_mm),
                        depth=str(_stage).strip().lower(),
                        mode="params",
                        source="arch mirror: %s" % str(_root)[:40],
                        # The state dir is keyed per model (.state/<model>), which is the same
                        # name every other anchor in this run uses. Taking it from `d` keeps the key
                        # identical without threading the model root into a mirroring helper.
                        model=d.name,
                    )
        except Exception:  # noqa: BLE001 -- a pin that cannot be written must not cost the mirror
            pass
    except Exception:  # noqa: BLE001 -- a mirror that cannot be written must never cost the write
        pass


def read_arch_mirror() -> dict:
    """The mirrored architecture map, or {}. See _mirror_arch_facts."""
    # NO IMPORT. The rebuild loads this module BY PATH -- spec_from_file_location, no package and
    # often no sys.path entry -- so `from cc_optimize.tmpstate import` and `from .tmpstate import`
    # BOTH fail there, and the reader silently returned {} in the one context it exists for. Measured
    # on run 17: the mirror held blocks and stage_roots, the emitter rebuilt without them anyway.
    # The variable is the whole contract, so read it directly and depend on nothing.
    try:
        import os as _os

        _sd = (_os.environ.get("PERF_MCP_STATE_DIR") or "").strip()
        if not _sd:
            return {}
        d = json.loads((Path(_sd) / ARCH_MIRROR_NAME).read_text())
        return {k: v for k, v in d.items() if k in ARCH_KEYS and v} if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _emit_perf_target_inputs(model_root, demo_dir, model_id_hint, manifest) -> None:
    """Write perf_target_inputs.json into the model root so the decode ceiling can be computed.

    Never raises. Two rules about WHERE and WHETHER, both learned the same way:

    WHERE -- into a real, stated model directory or nowhere. With a relative or empty model_root this
    wrote into the WORKING DIRECTORY, and perf_mcp's reader had the identical "." default, so the
    file the run dropped in the repo root was then adopted as the model's facts. On gemma-3-12b that
    meant a report priced against a 32-layer, hidden-1280, 30 MB model: prefill memory ceiling
    0.061 ms against a 100 ms measurement, no param count, hence no compute roof and no fidelity
    ladder. Three broken sections from one file written to the wrong place and read back from it.

    WHETHER -- refresh the tool's OWN output, refuse to clobber a HAND-TUNED one. `never overwrites`
    was written to protect a file someone had filled in with real per-tensor dtypes, which is right,
    but applied to every file it also froze the tool's own first guess forever: the geometry keys
    that prefill's byte model needs could never reach a model that already had a file, so the roof
    silently degraded for exactly the models that had been run before. The file records who wrote it
    in `source`, so the two cases are distinguishable rather than assumed.
    """
    try:
        root = Path(model_root)
        if not root.is_absolute() or not root.is_dir():
            print(
                "  [optimize/cc] not writing perf_target_inputs.json: model root %r is not a stated "
                "directory (a model fact written to the working directory gets read back as some "
                "other model's)" % str(model_root),
                flush=True,
            )
            return
        out = root / "perf_target_inputs.json"
        facts = _perf_target_inputs(demo_dir, model_id_hint, manifest)
        if not facts:
            return
        # THE CARRY-FORWARD BELOW READS THE PREVIOUS FILE, AND A REVERT DELETES IT.
        # gitio.remove_new_untracked removes untracked files created since the checkpoint, and this
        # file is one -- so the common case is not "overwritten with less" but "gone entirely", and
        # then there is no _prev to carry anything forward from. The mirror survives that, because it
        # lives outside the model directory the revert scans. Restored BEFORE the guards below so a
        # recovered value counts as present.
        # Best-effort, like the mirror write: a mirror that cannot be READ must never cost the
        # facts write it exists to protect. Without this guard a raising read aborted the emit
        # entirely -- worse than the loss it repairs.
        try:
            for _k, _v in (read_arch_mirror() or {}).items():
                if _v and not facts.get(_k):
                    facts[_k] = _v
        except Exception:  # noqa: BLE001
            pass
        if out.exists():
            try:
                _prev = json.loads(out.read_text())
            except Exception:  # noqa: BLE001
                _prev = {}
            if not isinstance(_prev, dict):
                return  # unreadable: strictly better than what is derived here
            # ADD WHAT IS MISSING, NEVER REPLACE WHAT IS THERE. This refused outright whenever the
            # file's `source` was not this producer's, to protect a hand-tuned file. It also refused
            # every file the tool's OWN device census writes, because that writer stamps no source:
            # the census creates perf_target_inputs.json early carrying device_weight_bytes and
            # bytes_per_param and nothing else, the producer then reads a source it does not
            # recognise, treats the tool's own output as someone's careful manual work, and declines
            # -- so the param count, the per-block geometry and the layer counts never arrive for the
            # rest of the run. Measured on voxtral run 37: a census file written at 23:46 left every
            # per-stage compute ceiling, and the entire fidelity ladder, unrenderable to the end of
            # the run for want of a param count this producer already had.
            #
            # Merging key-by-key protects a hand-tuned value exactly as well -- it is present, so it
            # is kept -- while letting a fact nobody has recorded reach the file. Copying first also
            # means the divisor guard below can no longer fire on a key the previous file owned.
            # A HAND-TUNED FILE NAMES ITS AUTHOR; THE CENSUS'S DOES NOT. That is the whole
            # distinction, and it is recorded in the file rather than inferred: `source` present and
            # not this producer's means someone stated where those facts came from, and they stay
            # untouched exactly as before. `source` absent means no producer has ever written here,
            # which is the census's file, and the keys it lacks are merged in.
            _mine = str(_prev.get("source") or "") == str(facts.get("source") or "")
            _unclaimed = not str(_prev.get("source") or "").strip()
            if not (_mine or _unclaimed):
                return  # hand-tuned: strictly better than what is derived here
            # The gap-fill itself runs AFTER the divisor guard below: filling a hole first would
            # hide the very loss that guard exists to catch -- a regeneration that dropped
            # weight_bytes would silently inherit the old one and be written anyway.
            if _prev == facts:
                return
            # A REFRESH MUST NEVER DOWNGRADE. `never overwrites` was crude but it was SAFE: it could
            # not replace good facts with worse ones. Refreshing on the strength of "the file is
            # mine" checks who WROTE it and never whether what is about to be written is any good --
            # and the deriver can be wrong. It was: with four gemma-3 variants in the HF cache it
            # resolved gemma-3-4b, so a run overwrote gemma-3-12b's facts (24.37 GB, 11.18B params,
            # 48 layers) with `weight_bytes: 0`, 4B params and 34 layers, and every roofline number
            # after that described a model that was not running.
            #
            # The divisor is the thing that cannot be lost: with no byte count and no param count
            # there is nothing to divide by, so a file that HAS one must not be replaced by one that
            # does not. Refuse the write and say so -- silently keeping the old file would leave the
            # geometry keys unable to land, which is the problem the refresh exists to solve.
            _lost = [
                k
                for k in ("weight_bytes", "total_params", "active_params")
                if (_prev.get(k) or 0) > 0 and not (facts.get(k) or 0) > 0
            ]
            # THE STRUCTURAL FACTS ARE CARRIED FORWARD, NOT GUARDED BY REFUSAL.
            #
            # `blocks` and `stage_roots` say WHICH TOWER each stage runs, and neither is produced
            # here: blocks needs a resolvable model id, and stage_roots is merged in from discovery
            # (_merge_model_facts) by a different code path entirely. So every successful write of
            # this file dropped them, and the guard above could not notice -- it checks the three
            # divisor keys, which the rebuild DOES produce, so `_lost` came back empty and the write
            # went through.
            #
            # Measured on run 13, 2026-08-21: a git_revert after a no-gain attempt deleted this file,
            # perf_mcp rebuilt it, and the multi-tower shape became a flat one -- layers 32 from the
            # audio tower beside hidden_size 3072 from the language model. Every stage then fell back
            # to that flat geometry and to total_params, so the audio encoder was priced with the
            # language model's 3.611B instead of its own 0.637B: 5.7x the real work, and the report
            # showed encode at 321% of a 702 TFLOPS peak, which is not a thing that can happen.
            #
            # Refusing the write would be wrong here -- the refresh exists so geometry keys can land,
            # and a legitimately updated weight_bytes must not be blocked by a key this producer was
            # never going to emit. Carrying the old value forward keeps both. Timeline for the
            # record: the guard is from 2026-08-09 and has never changed; blocks and stage_roots were
            # added on 2026-08-17 without widening it.
            for _k in ARCH_KEYS:
                if _prev.get(_k) and not facts.get(_k):
                    facts[_k] = _prev[_k]
            if _lost:
                print(
                    "  [optimize/cc] NOT refreshing perf_target_inputs.json: the new facts drop %s "
                    "(the ceiling's divisor). Keeping the existing file -- a regenerated file that "
                    "lost the divisor describes no model." % ", ".join(_lost),
                    flush=True,
                )
                return
            # NOW fill the gaps: the new facts are known not to have dropped a divisor, so every key
            # the previous file carries and these do not is added rather than lost to the rewrite.
            for _k, _v in _prev.items():
                # `_v not in (None, "", 0, ...)` was wrong for a BOOLEAN: False == 0 in Python, so a
                # recorded False -- device_census_complete among them -- tested as "no value" and was
                # dropped, and since this write REPLACES the file the key disappeared entirely. A
                # bool is a value; only None and the empty string are not.
                if _v is None or _v == "":
                    continue
                _cur = facts.get(_k, None)
                _held = not (
                    _k not in facts
                    or _cur is None
                    or _cur == ""
                    or (isinstance(_cur, (int, float)) and not isinstance(_cur, bool) and _cur == 0)
                )
                if not _held:
                    facts[_k] = _v
            if _prev == facts:
                return
        out.write_text(json.dumps(facts, indent=2) + "\n")
        _mirror_arch_facts(facts)
        # ANCHOR IT IN THE LEDGER TOO. The file lives in the model directory, which the optimize loop
        # reverts between attempts -- it was rolled back twice in one run, each time restoring a
        # different vintage. The ledger is keyed, append-only and outside that directory, so the
        # ceiling the report divides by cannot change underneath it.
        try:
            led = _ledger()
            # ANCHOR THE DIVISOR THE CEILING ACTUALLY USES, not the checkpoint's byte count. The
            # ceiling divides by PARAMS (xB -> xGB); pinning weight_bytes pinned a different quantity,
            # so the report -- which recomputes from this anchor -- and the stop gate -- which computes
            # from the facts -- divided by different numbers for the same run. On a bf16 checkpoint
            # (16.06 GB stored, 7.5B params) that is 25.5 vs 54.6 tok/s/u. perf_target owns the choice,
            # so the two cannot drift apart again.
            from agent import perf_target as _pt_div

            _divisor = _pt_div.simple_active_bytes(facts) or float(facts["weight_bytes"])
            # NO UNIT, NO ANCHOR. This ran `depth=str(facts.get("unit") or "unit")`, so a run with no
            # unit yet filed the anchor under the literal string "unit" -- a key nothing looks up.
            #
            # The unit is missing here BY DESIGN. It is set only from an observation
            # (PERF_MCP_LAST_HEADLINE_UNIT), because an HF tag names the TASK and cannot say whether a
            # model loops -- `text-to-speech` covers XTTS, which emits tokens, and Kokoro, which
            # produces a whole waveform in one pass. And this function runs ONCE AT SETUP, before any
            # trace exists. So the one moment the anchor is written is the one moment the thing it
            # must be keyed by does not yet exist.
            #
            # What that cost, measured on voxtral run 18: the report looks up depth="token" and MISSES,
            # falling back to the snapshot (4.777 GB); the gate scans rows depth-agnostically and HITS
            # the placeholder row (7.223 GB). One run, two divisors, 1.51x apart -- the exact failure
            # the anchor was introduced to prevent, and it passed test_the_gate_and_the_report_divide_
            # by_the_same_bytes because both sides agreed on a key that was wrong.
            #
            # Declining is what the rest of this chain already does: "No recoverable unit means no
            # ceiling, which lands on the floor fallback: weaker, but not wrong." Defaulting to
            # "token" would be worse than the placeholder -- it is the bug that once labelled every
            # diffusion and classifier model per-token. Nothing is lost by waiting: before the first
            # trace there is no measurement, so there is no ceiling for the two readers to disagree
            # about, and once a trace reports, the rebuild path anchors with the observed unit.
            _unit = str(facts.get("unit") or "").strip().lower()
            if _unit:
                led.anchor(
                    led.KIND_ACTIVE_BYTES,
                    float(_divisor) / 1e6,
                    depth=_unit,
                    mode="bytes_mb",
                    source=facts["source"][:120],
                    model=Path(model_root).name,
                )
        except Exception:  # noqa: BLE001
            pass
        print(
            "  [optimize/cc] decode roofline inputs: %.2f GB of weights @ %s -> perf_target_inputs.json"
            % (facts["weight_bytes"] / 1e9, facts["dominant_dtype"])
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] decode roofline inputs skipped ({exc})")


def _tt_lang_available() -> bool:
    try:
        import importlib.util

        return any(importlib.util.find_spec(m) is not None for m in ("ttl", "tt_lang", "ttlang"))
    except Exception:
        return False


def _print_optimize_stop(pipe, exc) -> None:
    """On any per-pipeline crash, tell the user — in plain language — why optimize stopped, the exact
    next step to fix it, and where to see what was accomplished. Never raises."""
    import re as _re
    import sys as _sys

    err = f"{type(exc).__name__}: {exc}"
    low = str(exc).lower()
    bar = "=" * 78
    steps = []
    if isinstance(exc, ModuleNotFoundError) or "no module named" in low:
        _m = _re.search(r"no module named ['\"]([\w.]+)['\"]", low)
        pkg = (_m.group(1).split(".")[0] if _m else "") or "<the-missing-package>"
        steps.append(f"a Python dependency ('{pkg}') is missing — install it, then re-run:")
        steps.append(f"    {_sys.executable} -m pip install {pkg}")
    elif "_ttnncpp" in low or "cannot open shared object" in low or ("ttnn" in low and "shared object" in low):
        steps.append("ttnn is not built for this checkout (its compiled .so is missing) — build it, then re-run:")
        steps.append("    ./build_metal.sh")
    elif "transformers" in low and ("flash_attn" in low or "unrecognized" in low or "attn_implementation" in low):
        steps.append('the model needs a different transformers version — e.g.  pip install "transformers<5"  (in a')
        steps.append("    dedicated venv if it would conflict with other models), then re-run.")
    else:
        steps.append("this is usually a build/env/version mismatch — read the CAUSE above, fix it, and re-run.")
    try:
        print("\n" + bar)
        print(f"  OPTIMIZE STOPPED — pipeline '{(pipe or {}).get('task', '?')}'")
        print(f"  CAUSE: {err}")
        print("  NEXT STEPS to make it run:")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}" if not s.startswith("    ") else s)
        print("  What was accomplished so far is preserved — committed speedups are already in git, and")
        print("  the per-op ledger is at models/experimental/perf_automation/runs/<timestamp>/ledger.jsonl.")
        print(bar)
    except Exception:
        pass


def _stamp_run_id() -> str:
    """One id for this optimize run, set once and inherited by every child.

    The recovery counters are scoped to it: "resets have stopped working" is a fact about THIS run
    against THIS board, and carrying it into the next run is what turned a limit into a latch (run 39
    left reset_fails=34 in a (model, task)-keyed file that survived the board being fixed and a host
    reboot). Set here rather than in the CLI so every entry point -- supervisor restarts included --
    lands in the same run, and never overwritten, so a restart does not silently get a fresh budget.
    """
    cur = str(os.environ.get("PERF_MCP_RUN_ID") or "").strip()
    if not cur:
        cur = "%d_%d" % (int(time.time()), os.getpid())
        os.environ["PERF_MCP_RUN_ID"] = cur
    return cur


def run_cc_optimize(
    demo_dir: Path,
    repo_root: Path,
    devices: str = "0",
    metric: str = "device_ms",
    perf_test=None,
    case=None,
    pcc_test=None,
    baseline_only: bool = False,
    e2e_only: bool = False,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    sync_catalog: bool = False,
    catalog_remote: str = "origin",
    catalog_branch: str = "perf-catalog",
    model_id_hint=None,
    hitl: bool = False,
) -> dict | None:
    """Top-level cc engine: discover pipeline(s), then optimize EVERY one to the gate's can_stop.

    If sync_catalog: pull the shared GRADUATED_* knob catalog from catalog_remote/catalog_branch BEFORE
    discovery (so this run recalls the latest cross-model-proven knobs), and push any GRADUATED_* back
    at the end. Off by default — learning stays local unless opted in. Both steps are best-effort and
    never fail the run; the remote/branch is fully configurable (nothing hard-coded)."""
    _stamp_run_id()
    # BEFORE the device is touched and before discovery spends an agent call: verify the tool this
    # run will execute is the tool that was verified. See _preflight_tool.
    if not _preflight_tool(repo_root):
        print("  [optimize/cc] refusing to start against a tool whose own tests fail.", flush=True)
        raise SystemExit(EXIT_REFUSED)
    # ONCE, AT THE START, AND NOWHERE ELSE. The 65C gate used to run before EVERY device process,
    # which meant waiting on a board that idles in the sixties over and over for a threshold that
    # barely separates clamped runs from clean ones (its own data: medians 72.5C vs 70.8C, ranges
    # overlapping). Begin the run from a known-cool board, and from then on the only thing that
    # holds work is the safety ceiling -- which is about the hardware, not the reading.
    try:
        _ok, _start_c = _perf_mcp()._wait_for_thermal_headroom()
        if _start_c is not None:
            print("  [optimize/cc] starting with the board at %.1fC" % _start_c, flush=True)
    except Exception as exc:  # noqa: BLE001 -- a gate that cannot run must not stop the work
        _warn_gate_broken(exc)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # No exported key is FINE: `claude` may be authenticated via `claude /login` (README §5.2
        # Option A). Every claude subprocess uses those stored creds; claude surfaces its own error
        # if truly unauthenticated. So we don't hard-block here.
        print("  [optimize/cc] no ANTHROPIC_API_KEY in env — using `claude` login credentials.")
    if sync_catalog:
        catalog_pull(repo_root, catalog_remote, catalog_branch)
    manifest = discover(demo_dir, repo_root, devices, metric, perf_test, case, pcc_test)
    if not manifest:
        print("  [optimize/cc] discovery failed (before_loop produced no manifest).")
        return None
    # ADOPT THE CHIP SCOPE DISCOVERY WORKED OUT, FOR THIS PROCESS AND EVERY CHILD OF IT.
    #
    # before_loop runs as a SUBPROCESS, so the TT_VISIBLE_DEVICES + descriptor pair it sets reaches
    # only its own children -- this process never saw it, and cc_env below then built each device
    # subprocess from a fresh os.environ with those variables explicitly popped. Measured on a
    # 4-chip p300c: most of a --devices 0 run correctly used one chip, while the five cc_env paths
    # (the full-pipeline measurement among them, the hottest step in the run) used all four and took
    # the board to 89C. Applying it here puts it in os.environ once, so cc_env inherits it like any
    # other setting and all five paths agree with the rest of the run.
    _apply_scope(os.environ, (manifest.get("config") or {}))
    perf_dir = repo_root / PERF_DIR
    manifest_path = str(_latest_manifest(perf_dir))
    _seqf = Path(manifest_path).parent / "perf_seq_len"
    if _seqf.is_file():
        _seq = _seqf.read_text().strip()
        if _seq:
            os.environ["TT_PERF_SEQ_LEN"] = _seq
            print(f"  [optimize/cc] perf workload seq pinned to {_seq} (baseline shape-retry); propagated to loop")
    _warn_dirty_model_tree(demo_dir, repo_root)
    _decide_parallelism_route(demo_dir, manifest, repo_root, metric, devices, model_id_hint)
    _emit_perf_target_inputs(demo_dir, demo_dir, model_id_hint, manifest)
    model_rel = os.path.relpath(demo_dir, repo_root)
    model_name = Path(demo_dir).name
    os.environ.setdefault("PERF_MCP_MODEL_NAME", model_name or "model")
    _cfg_ref = _resolve_model_id(demo_dir, model_id_hint, _run_test_files(demo_dir, manifest)) or str(demo_dir)
    pipes = pipelines_from_manifest(manifest, model_rel)
    is_mm = manifest.get("pathmap", {}).get("is_multimodal")
    print(f"  [optimize/cc] discovered pipelines: {[p['task'] for p in pipes]} (multimodal={is_mm})")
    # AFTER discovery, so the sweep can use the perf test that was just generated instead of
    # demanding a second one from the operator. No-op unless --matmul-sweep was passed.
    _matmul_sweep_after_discovery(demo_dir, repo_root, pipes, devices)
    if e2e_only:
        os.environ["PERF_MCP_FULLPIPE_E2E"] = "1"
        for pipe in pipes:
            kernel_log = _kernel_log_path(model_name, pipe["task"])
            mcp_env = _mcp_config(repo_root, manifest_path, pipe, devices, kernel_log)["mcpServers"]["perf-mcp"]["env"]
            print(f"  [optimize/cc] === full-model end-to-end MEASURE (no optimization): {pipe['task']} ===")
            _fullpipe_e2e(repo_root, mcp_env, devices, "MEASURE")
        return {"pipelines": pipes, "is_multimodal": is_mm, "results": [], "e2e_only": True}
    if baseline_only or not pipes:
        return {"pipelines": pipes, "is_multimodal": is_mm, "results": []}
    results = []
    _ttl_ok = _tt_lang_available()
    for pipe in pipes:
        print(f"  [optimize/cc] === optimizing pipeline: {pipe['task']} ===")
        try:
            results.append(
                optimize_pipeline(
                    repo_root,
                    manifest_path,
                    pipe,
                    devices,
                    metric,
                    model_name,
                    max_rounds,
                    hitl=hitl,
                    config_ref=_cfg_ref,
                )
            )
        except Exception as exc:  # noqa: BLE001 — never let one pipeline's crash kill the whole run silently
            _print_optimize_stop(pipe, exc)
            results.append(None)
    if not _ttl_ok:
        import sys as _sys

        print(
            "\n  ⚠ tt-lang was NOT used this run — the ttl toolchain is not installed in this environment\n"
            "    (commonly a Python-version mismatch). The knob / dtype / C++ / structural levers still ran;\n"
            "    only the tt-lang kernel rung was skipped. To enable it next time:\n"
            f'    {_sys.executable} -m pip install "tt-lang==1.0.1" --no-deps   (must match your ttnn)'
        )
    if sync_catalog:
        catalog_push(repo_root, catalog_remote, catalog_branch)
    return {"pipelines": pipes, "is_multimodal": is_mm, "results": results, "tt_lang_used": _ttl_ok}
