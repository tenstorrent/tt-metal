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
from pathlib import Path

PERF_DIR = "models/experimental/perf_automation"
CC_DIR = PERF_DIR + "/cc_optimize"
DEFAULT_MAX_ROUNDS = 3
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

termination_check() is the SOLE authority on whether more optimization is needed. It returns a DETERMINISTIC per-op CHECKLIST and a single next_target = {{op, op_class, grid, bound_by, rung}} you MUST work next. The per-op ladder ORDER is: knob:grid -> knob:dtype -> tt-lang -> cpp. An op is "nothing left" ONLY when every rung is ticked; you may STOP ONLY when can_stop=true.

LOOP:
  git_head -> termination_check -> read next_target.
  REUSE-FIRST: call recall_knobs(next_target.op_class, next_target.grid, next_target.bound_by) and APPLY/ADAPT any matching catalogued knob (heed its negative knowledge) BEFORE improvising one.
  Do EXACTLY next_target.rung on next_target.op:
    knob:grid  -> full-grid program_config. check_pcc; measure_candidate; commit a real win else revert. record_kernel_attempt(op,'grid',measured_ms,beat_baseline).
    knob:dtype -> lower that op's WEIGHT dtype (bf16->bf8_b->bf4_b). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'dtype',measured_ms,beat_baseline) EVEN IF pcc forced a revert (that marks the knob tried).
    tt-lang    -> author a tt-lang (ttl) kernel (Read GUIDELINES/11). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'tt-lang',measured_ms,beat_baseline).
    cpp        -> author a C++ Metalium kernel via ttnn.generic_op (Read GUIDELINES/12). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'cpp',measured_ms,beat_baseline).
  COVERAGE — the profiled slice is a REPRESENTATIVE set of layers, not all of them, so after a dtype knob or a kernel swap call check_lever_coverage(op_match, stale_dtype, new_dtype) to CONFIRM the lever reached EVERY layer instance. A repeated block is ONE class instantiated N times, so editing the SHARED block definition/config propagates to all N; editing an instance-specific path (e.g. layers[0], a per-layer override) changes only that one and silently misses the rest. If fully_applied is false, REAPPLY on the shared definition (target the reported missed_blocks) and re-check until fully_applied — a partial application is NOT a real win even if the slice looks faster.
  ALWAYS pass note= to record_kernel_attempt: ONE line stating (a) WHY you tried this lever on this op (the hypothesis — e.g. 'op is DRAM-bw bound, bf8_b weights halve reads') and (b) WHY it won or failed (the outcome reason — e.g. 'kept: 4.1->3.6ms', 'reverted: PCC 0.71<0.95', 'no gain: 4.1->4.1ms bw-bound', 'OOM under 2CQ'). This note is streamed LIVE into the model's RUN_REPORT.md the instant the attempt resolves (win OR fail), so it must explain the reasoning, not just restate the numbers. ALSO pass stages_json to record_kernel_attempt whenever you have per-stage trace timings (the SAME JSON list of {{"name","ms","dominant?"}} you'd pass hitl_gate, e.g. from check_full_pipeline_latency's stage breakdown) — this renders the block-level timing table in RUN_REPORT.md so BOTH hitl and non-hitl runs show where device time went per stage/block.
  TWO measurements are fed back to you each step — use BOTH: (1) measure_candidate returns the per-op tracy device_ms (the fast steering signal that tells you WHICH op moved); (2) check_full_pipeline_latency returns the robust whole-pipeline trace+1cq per-token ms (its `mode` field = trace+1cq, `full_pipeline_ms` + `delta_pct` vs best) — this is the per-iteration VERDICT you bank a compute win on. trace+1cq always engages (no 2-CQ reservation), so a dtype/grid/fusion/kernel win it confirms is real and will NOT spuriously fail the way a 2-CQ run can; the trace+2cq production number is measured only at the start/end bookend, so DO NOT treat a mid-loop 1cq result as a downgrade. Only levers whose whole value is the 2-CQ input/compute overlap or L1 headroom need the 2cq bookend to judge.
  (IRON RULE: a real win = check_pcc ok AND check_full_pipeline_latency status 'ok' (moved TOWARD the target / not diverged, at its trace+1cq mode) AND measure_candidate verdict 'valid' AND is_real_gain AND (for a dtype/kernel lever) check_lever_coverage fully_applied (reached every layer, not just the profiled slice). REJECTED, pcc-fail, or a DIVERGED full-pipeline latency is never a win — revert. Note: check_full_pipeline_latency never fails for missing the target, only for getting SLOWER than best-so-far in its CQ track.)
  WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs had no match), call distill_knob to persist the general technique; if the win RE-USED a provisional lever learned on another model, pass its id to distill_knob to graduate it.
  Re-run termination_check. Repeat. NEVER stop while can_stop=false. NEVER reason a lever "won't help" — prove it by measuring + recording the attempt.

LEAVE CLEAN (commit wins, revert in-progress edits); end with git_head. Report start->final {metric}, committed wins, and per blocking op which rungs were done + measured ms."""


_HITL_PROMPT = (
    _PROMPT
    + """

HITL MODE (human-in-the-loop): you do NOT have git_commit / git_revert. After you apply ONE lever and measure it (check_pcc; measure_candidate; check_full_pipeline_latency; check_lever_coverage for a dtype/kernel lever), call hitl_gate(tried_op, tried_lever, why_tried, is_win, why_not, next_target, next_why, before_ms, after_ms, stages_json) INSTEAD of committing. stages_json = the per-stage trace timings you just measured, a JSON list of {{"name","ms"}} (add "dominant" if known). hitl_gate returns {{action}}: on 'commit' or 'revert' the operator's git action is ALREADY DONE for you — move to the next target; on 'try', apply the operator's returned knob next. Exactly ONE lever per hitl_gate call; never batch. record_kernel_attempt as usual so RUN_REPORT stays live."""
)


def _visible_devices(devices: str) -> str | None:
    """DEVICE VISIBILITY IS INTENTIONALLY NEVER RESTRICTED — this is what makes the tool chip-count /
    hardware agnostic. Pinning TT_VISIBLE_DEVICES to a chip SUBSET ('single' -> '0', '0,1', ...) makes
    tt-metal's fabric auto-discovery classify the board as a CUSTOM cluster and fatally demand a
    mesh-graph-descriptor path that we don't provide — so a subset spec crashes device/fabric init
    (before any forward) on any multi-chip board. The full physical topology must stay visible so
    auto-discovery works for single | all | explicit-ids alike. HOW MANY chips a run actually uses is
    controlled by the mesh SHAPE (TT_PERF_MESH_ROWS/COLS via _derive_topology_env + resolve_mesh_shape),
    which is the chip-count-agnostic lever. Hence: always None (leave TT_VISIBLE_DEVICES unset)."""
    _ = devices  # spec informs the mesh shape, not OS-level visibility (which would break fabric)
    return None


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
    vis = _visible_devices(devices)
    if vis is not None:
        env["TT_VISIBLE_DEVICES"] = vis
        env["TT_METAL_VISIBLE_DEVICES"] = vis
    else:
        env.pop("TT_VISIBLE_DEVICES", None)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
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
    rc, _ = _run_device_proc(
        cmd,
        perf_dir,
        cc_env(repo_root, devices),
        devices,
        int(os.environ.get("PERF_MCP_DISCOVER_TIMEOUT", "10800") or "10800"),
        "discovery",
        capture=False,
        stall_s=int(os.environ.get("PERF_MCP_DISCOVER_STALL_SEC", "1200") or "1200"),
    )
    mani = _latest_manifest(perf_dir)
    if mani is None or rc is None:
        return None
    if rc != 0 and mani.stat().st_mtime < launch_ts:
        return None
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


def _mcp_config(repo_root: Path, manifest_path: str, pipe: dict, devices: str, kernel_log: str) -> dict:
    env = {
        "PERF_MCP_MANIFEST": manifest_path,
        "PERF_MCP_PERF_TEST": pipe["perf_test"],
        "PERF_MCP_PCC_TEST": pipe["pcc_test"],
        "PERF_MCP_KERNEL_LOG": kernel_log,
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{os.pathsep}/usr/bin:/bin",
        "PERF_MCP_FULLPIPE_CQ": "1",
    }
    if pipe.get("case"):
        env["PERF_MCP_PERF_CASE"] = pipe["case"]
    vis = _visible_devices(devices)
    if vis is not None:
        env["TT_VISIBLE_DEVICES"] = vis
        env["TT_METAL_VISIBLE_DEVICES"] = vis
    _seq = os.environ.get("TT_PERF_SEQ_LEN")
    if _seq:
        env["TT_PERF_SEQ_LEN"] = _seq
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
        "print('HALTREASON=' + str(r.get('halt_reason') or ''))"
    )
    env = cc_env(repo_root, devices)
    env.update(mcp_env)  # PERF_MCP_* so the gate targets this pipeline
    rc, out = _run_device_proc(
        [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
        repo_root / PERF_DIR,
        env,
        devices,
        int(os.environ.get("PERF_MCP_MEASURE_TIMEOUT", "1200") or "1200"),
        "termination_check",
    )
    if rc is None:
        return {"can_stop": False, "halt": False, "reason": ""}
    out = out or ""
    reason = ""
    for line in out.splitlines():
        if line.startswith("HALTREASON="):
            reason = line[len("HALTREASON=") :]
    return {"can_stop": "CANSTOP=True" in out, "halt": "HALT=True" in out, "reason": reason}


def _fullpipe_e2e(repo_root: Path, mcp_env: dict, devices: str, label: str) -> float | None:
    """Measure the FULL-model end-to-end (ALL 52 layers, no tracy, prefill + 1 decode) ONCE and print it
    with `label` (BEFORE / AFTER). Returns end_to_end_ms or None. This is the whole-model SCOREBOARD, run
    only at the two BOOKENDS of a pipeline's optimization (start + right before stop) — never per iteration
    — so a real before/after full-model speedup is reported without the per-step cost. The device_ms loop
    metric is the fast 2-layer STEERING signal; this is the verdict. Disable via PERF_MCP_FULLPIPE_E2E=0."""
    if os.environ.get("PERF_MCP_FULLPIPE_E2E", "1") != "1":
        return None
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import perf_mcp as P; "
        "g=P.check_full_pipeline_latency\n"
        "for a in ('fn','func','_fn','__wrapped__'):\n"
        "    if hasattr(g,a): g=getattr(g,a); break\n"
        "r=g()\n"
        "print('FULLPIPE_MS=' + str(r.get('full_pipeline_ms')))"
    )
    env = cc_env(repo_root, devices)
    env.update(mcp_env)
    env["PERF_MCP_FULLPIPE_CQ"] = "2"
    env.setdefault("PERF_MCP_FULLPIPE_SAMPLES", "3")
    print(
        f"  [optimize/cc] measuring FULL-model end-to-end ({label}) — ALL 52 layers, no tracy (one slow run, minutes)..."
    )
    ms = None
    timeout_s = int(os.environ.get("PERF_MCP_MEASURE_TIMEOUT", "1200") or "1200")
    proc = subprocess.Popen(
        [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
        cwd=str(repo_root / PERF_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        print(
            f"  [optimize/cc] FULL-model end-to-end ({label}) TIMED OUT after {timeout_s}s (likely a "
            f"device wedge / leaked mesh) — killed the whole process group + {_reclaim_device(devices)}"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] FULL-model end-to-end ({label}) skipped ({exc})")
        return None
    for line in (out or "").splitlines():
        if line.startswith("FULLPIPE_MS="):
            try:
                ms = float(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                ms = None
        if "[full-pipeline-gate]" in line:
            print("  [optimize/cc] " + line.strip())
            if "PERF_SCORECARD" in line:
                _LAST_SCORECARD.clear()
                for tok in line.split("PERF_SCORECARD", 1)[1].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        _LAST_SCORECARD[k] = v
    if ms is not None:
        print(f"  [optimize/cc] FULL-model end-to-end ({label}) = {ms:.1f} ms  (ALL 52 layers, prefill + 1 decode)")
    return ms


_HOST_XFER_OPS = ("from_torch", "to_torch", "from_device", "to_device")


def _parse_facts(raw: str, sigs: set | None) -> dict:
    """Extract the UNIVERSAL scorecard facts from an op-sig probe run: TP/DP + shard state (from the
    pipeline's MeshDevice line) and whether the step round-trips to host (host-transfer ops in the op
    set) — the latter is the trace+2CQ gate. Model-agnostic: reads the op stream, not a per-model map."""
    facts = {"dp": 1, "tp": 1, "shard_active": False, "host_ops": [], "n_op_types": len(sigs or ())}
    m = re.search(r"DP=(\d+)\s+TP=(\d+)", raw or "")
    if m:
        facts["dp"], facts["tp"] = int(m.group(1)), int(m.group(2))
    if "shard_active=True" in (raw or ""):
        facts["shard_active"] = True
    facts["host_ops"] = sorted({s.split("(")[0] for s in (sigs or set()) if any(h in s for h in _HOST_XFER_OPS)})
    return facts


def _run_op_sigs(repo_root: Path, mcp_env: dict, devices: str, node: str, case, k: int):
    """Run the perf test forward at TT_PERF_LAYERS=k (no tracy, 1 decode token) through the generic
    _op_sig_probe. Returns (sigs_set_or_None, raw_stdout_stderr)."""
    env = cc_env(repo_root, devices)
    env.update(mcp_env)
    env["TT_PERF_LAYERS"] = str(k)
    env["TT_PERF_MAX_NEW_TOKENS"] = "1"
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    cmd = [_python_bin(repo_root), str(repo_root / CC_DIR / "_op_sig_probe.py"), node]
    if case:
        cmd.append(case)
    rc, raw = _run_device_proc(
        cmd,
        repo_root,
        env,
        devices,
        int(os.environ.get("PERF_MCP_MEASURE_TIMEOUT", "1200") or "1200"),
        "coverage probe",
    )
    if rc is None:
        return None, ""
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
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or len(seq)) or len(seq)
    return min(max(first.values()) + 1, n_layers), len(first)


def _coverage_cache_path(repo_root: Path) -> Path:
    return repo_root / CC_DIR / ".coverage_cache.json"


def _coverage_fingerprint(node) -> str:
    try:
        base = Path(str(node).split("::", 1)[0])
        mt = max((f.stat().st_mtime for f in base.parent.rglob("*.py")), default=0.0)
        return str(int(mt))
    except Exception:  # noqa: BLE001
        return ""


def _coverage_cache_get(repo_root: Path, node, case):
    try:
        entry = json.loads(_coverage_cache_path(repo_root).read_text()).get(f"{node}|{case}")
        if entry and entry.get("fp") == _coverage_fingerprint(node):
            return int(entry["k"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _coverage_cache_put(repo_root: Path, node, case, k: int) -> None:
    try:
        path = _coverage_cache_path(repo_root)
        data = json.loads(path.read_text()) if path.is_file() else {}
        data[f"{node}|{case}"] = {"k": int(k), "fp": _coverage_fingerprint(node)}
        path.write_text(json.dumps(data, indent=1))
    except Exception:  # noqa: BLE001
        pass


def _depth_cache_get(repo_root: Path, node):
    try:
        entry = json.loads(_coverage_cache_path(repo_root).read_text()).get(f"depth|{node}")
        if entry and entry.get("fp") == _coverage_fingerprint(node):
            return dict(entry["env"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _depth_cache_put(repo_root: Path, node, env) -> None:
    try:
        path = _coverage_cache_path(repo_root)
        data = json.loads(path.read_text()) if path.is_file() else {}
        data[f"depth|{node}"] = {"env": dict(env), "fp": _coverage_fingerprint(node)}
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
    m = -1
    for tok in seq or []:
        if isinstance(tok, str) and tok.startswith("PERF_BLOCK_SIGNPOST:"):
            try:
                m = max(m, int(tok.split(":", 1)[1]))
            except (ValueError, IndexError):
                pass
    return m + 1


def _model_root_from_node(repo_root: Path, node):
    p = (node or "").split("::", 1)[0]
    if "/tests/" in p:
        p = p.split("/tests/", 1)[0]
    root = repo_root / p
    return root if root.is_dir() else None


def _llm_depth_env(model_root: Path, cov: int) -> dict:
    tt_dir = model_root / "tt"
    srcs, total = [], 0
    for py in sorted(tt_dir.glob("*.py")) if tt_dir.is_dir() else []:
        try:
            txt = py.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        srcs.append(f"### {py.name}\n{txt}")
        total += len(txt)
        if total > 60000:
            break
    if not srcs:
        return {}
    prompt = (
        f"This TTNN model runs a stack of transformer layers/blocks. A profiler must execute only {cov} "
        f"layers (a representative slice), not all of them, to keep profiling fast. From the source below, "
        f"find the environment variable(s) this model reads to LIMIT how many layers/blocks it runs, plus "
        f"any flag it requires to permit a partial/truncated run. Respond with ONLY a JSON object mapping "
        f"env-var name to the string value that makes it run exactly {cov} layers; respond with {{}} if the "
        f"model exposes no such control.\n\n" + "\n\n".join(srcs)
    )
    out = _claude_text(prompt) or ""
    m = re.search(r"\{[^{}]*\}", out, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {str(k): str(v) for k, v in d.items() if str(k)}


_ANCHOR_KEYS = ("flash", "sdpa", "attention", "attn", "mla")


def _layer_signal(seq) -> int:
    ops = [t for t in seq or [] if isinstance(t, str) and not t.startswith("PERF_BLOCK_SIGNPOST:")]
    if not ops:
        return 0
    from collections import Counter

    names = Counter(o.split("(", 1)[0].strip().lower() for o in ops)
    anchor = sum(n for name, n in names.items() if any(k in name for k in _ANCHOR_KEYS))
    if anchor > 0:
        return anchor
    blk = _blocks_ran(seq)
    if blk > 0:
        return blk
    return len(ops)


def _bridge_depth_env(repo_root: Path, mcp_env: dict, devices: str, node, case, cov: int) -> dict:
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
    _, _, seq = _run_op_sigs(repo_root, mcp_env, devices, node, case, cov)
    full = _layer_signal(seq)
    if full <= 0:
        _depth_cache_put(repo_root, node, {})
        return {}
    env = _llm_depth_env(model_root, cov)
    if not env:
        print(f"  [optimize/cc] depth-knob bridge: no depth knob found (layer-signal {full})")
        _depth_cache_put(repo_root, node, {})
        return {}
    probe_env = dict(mcp_env)
    probe_env.update(env)
    _, _, seq2 = _run_op_sigs(repo_root, probe_env, devices, node, case, cov)
    capped = _layer_signal(seq2)
    if capped <= 0 or capped >= full * 0.7:
        print(f"  [optimize/cc] depth-knob bridge: {env} did not reduce layers (signal {full}->{capped}); ignoring")
        _depth_cache_put(repo_root, node, {})
        return {}
    print(f"  [optimize/cc] depth-knob bridge: enforcing {env} (layer-signal {full}->{capped})")
    _depth_cache_put(repo_root, node, env)
    return env


def _coverage_layers(
    repo_root: Path,
    mcp_env: dict,
    devices: str,
    node,
    case,
    n_layers: int = 52,
    model_name: str = "",
    config_ref: str = "",
):
    """MODEL-AGNOSTIC profiling-window sizing. One all-layers probe (TT_PERF_LAYERS=0, no tracy)
    enumerates EVERY distinct op across all layers (overflow-safe: host-side op wrapping, no marker
    buffer) and, via its per-block signposts, the block each op first appears in. The tracy timing window
    is the smallest depth that still holds a fresh instance of every op, capped at 16 (the marker limit);
    ops that first appear past 16 are reported as present-but-un-timed. Falls back to the config-declared
    layer pattern when the k=0 probe yields nothing (a model that reads TT_PERF_LAYERS=0 as an empty
    stack). Cached per model. Disable via PERF_MCP_COVERAGE_SIZING=0."""
    facts: dict = {}
    if os.environ.get("PERF_MCP_COVERAGE_SIZING", "1") != "1" or not node:
        return None, facts
    cached = _coverage_cache_get(repo_root, node, case)
    if cached is not None:
        print(f"  [optimize/cc] coverage (cached): TT_PERF_LAYERS={cached}")
        return cached, facts
    sigs, raw, seq = _run_op_sigs(repo_root, mcp_env, devices, node, case, 0)
    if sigs:
        facts = _parse_facts(raw, sigs)
        facts["all_ops"] = sorted(sigs)
        first_block: dict = {}
        cur = 0
        for tok in seq or []:
            if isinstance(tok, str) and tok.startswith("PERF_BLOCK_SIGNPOST:"):
                try:
                    cur = int(tok.split(":", 1)[1])
                except (ValueError, IndexError):
                    pass
            else:
                first_block.setdefault(tok, cur)
        if first_block:
            deepest = max(first_block.values())
            deep = sorted(op for op, b in first_block.items() if b >= 16)
        else:
            _kc, _ = _config_layer_kinds(config_ref or model_name)
            deepest = (_kc - 1) if _kc else 1
            deep = []
        _cov = min(max(deepest + 1, 2), 16)
        facts["deep_ops"] = deep
        tail = f"; {len(deep)} op(s) appear only past layer 16 (present, un-timed)" if deep else ""
        print(
            f"  [optimize/cc] coverage (all-layers probe): {len(sigs)} distinct op(s); deepest new op at "
            f"block {deepest} -> TT_PERF_LAYERS={_cov}{tail}"
        )
        _coverage_cache_put(repo_root, node, case, _cov)
        return _cov, facts
    k, n_kinds = _config_layer_kinds(config_ref or model_name)
    if k is not None:
        _cov = min(k, 16)
        print(
            f"  [optimize/cc] coverage (config fallback; k=0 probe empty): {n_kinds} kind(s), deepest first "
            f"appears at layer {k - 1} -> TT_PERF_LAYERS={_cov}"
        )
        _coverage_cache_put(repo_root, node, case, _cov)
        return _cov, facts
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
        osl = os.environ.get("TT_PERF_MAX_NEW_TOKENS") or "4"
        L = ["  ┌─ optimize scorecard — pipeline: %s" % pipe.get("task", "?")]
        L.append("  │ hardware          : %s  x%s chip(s)" % (arch, chips))
        L.append(
            "  │ parallelism       : TP=%s x DP=%s  (%s)"
            % (tp, dp, "sharded mesh" if facts.get("shard_active") else "single-chip / replicated")
        )
        if not probed:
            L.append("  │ fully on device   : UNKNOWN  (op-coverage probe did not run)")
        elif on_device:
            L.append("  │ fully on device   : YES  (trace + 2CQ possible)")
        else:
            L.append("  │ fully on device   : NO   -> trace + 2CQ blocked; host round-trips: %s" % ", ".join(host_ops))
        L.append("  │ batch / users     : %s" % batch)
        reason = (
            "probe did not run"
            if not probed
            else ("not fully on-device" if not on_device else "needs a trace-capturable decode step")
        )
        for name in ("TTFT", "T/S/U", "T/S"):
            L.append("  │ %-16s : N/A  (%s)" % (name, reason))
        L.append("  │ ISL / OSL         : %s / %s  (tokens; N/A for non-token models)" % (isl, osl))
        if before_ms and after_ms:
            d = (before_ms - after_ms) / before_ms * 100.0
            L.append("  │ full-model e2e    : %.1f -> %.1f ms  (%+.1f%%)" % (before_ms, after_ms, d))
        L.append("  └─")
        print("\n".join(L))
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] scorecard skipped ({exc})")
    try:
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
            print(_sp.render(_mid, _arch, _chips, _meas))
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] model_targets card skipped ({exc})")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=300
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# chip-index -> its board's PCI-resettable local chip, snapshotted while healthy. RESET PATH ONLY --
# nothing about mesh-open / parallelism / scorecard reads any of this; it exists solely to pick
# `tt-smi -r` targets so a whole n300 board resets (never half a board, never a non-PCIe remote chip).
_BOARD_MAP_FILE = Path(tempfile.gettempdir()) / "perf_mcp_board_topology.json"


def _read_board_topology() -> dict | None:
    """Live-read chip-index -> board-local-chip from tt-smi -s. Two ASICs of an n300 share a board_id;
    only the one with a real PCI bus_id is resettable, and resetting it resets its remote partner too.
    Returns {str(chip): local_chip_index} or None. Static per host (board_ids / BDFs don't change)."""
    try:
        tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
        r = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=120)
        di = (json.loads(r.stdout) or {}).get("device_info") or []
    except Exception:  # noqa: BLE001
        return None
    board_of: dict[int, str] = {}
    local_of_board: dict[str, int] = {}
    for i, dev in enumerate(di):
        bi = dev.get("board_info") or {}
        bid = bi.get("board_id")
        board_of[i] = bid
        bus = bi.get("bus_id")
        if bid is not None and bus and bus != "N/A":
            local_of_board.setdefault(bid, i)
    m = {str(i): local_of_board.get(board_of.get(i)) for i in board_of}
    m = {k: v for k, v in m.items() if v is not None}
    return m or None


def _capture_board_topology() -> None:
    """Persist the reset map while the board is HEALTHY (startup) so the reset path has a trustworthy
    map even if the board later wedges. Best-effort; reset falls back to a live read if the file's gone.
    RESET PATH ONLY -- captured here, consumed only by _board_reset_targets."""
    m = _read_board_topology()
    if m:
        try:
            _BOARD_MAP_FILE.write_text(json.dumps(m))
        except Exception:  # noqa: BLE001
            pass


def _board_reset_targets(chip_ids: list[int]) -> str | None:
    """Map logical chip ids -> the PCI-resettable LOCAL chip of each board they live on, using the map
    captured at healthy startup (live-read fallback). Ensures a reset hits whole n300 boards -- never
    half a board, never a non-PCIe remote chip (the half-reset fabric that caused the ETH wedge).
    Returns a sorted comma list, or None if no topology is available. RESET PATH ONLY."""
    m = None
    try:
        m = json.loads(_BOARD_MAP_FILE.read_text())
    except Exception:  # noqa: BLE001
        m = None
    if not m:
        m = _read_board_topology()
    if not m:
        return None
    targets = {m[str(c)] for c in chip_ids if str(c) in m and m[str(c)] is not None}
    return ",".join(str(x) for x in sorted(targets)) if targets else None


def _reset_chip_list(devices: str) -> str:
    """BOARD-AWARE reset target derived from --devices. Explicit ids / 'single' are translated to the
    PCI-resettable LOCAL chip of each board they live on (so a whole n300 board resets, never half of
    one). 'all'/'' returns '' so the caller uses a bare `tt-smi -r` (resets every board)."""
    d = (devices or "").strip().lower()
    if d in ("all", ""):
        return ""
    if d == "single":
        req = [0]
    else:
        req = [int(x) for x in d.split(",") if x.strip().isdigit()]
    if not req:
        return ""
    board = _board_reset_targets(req)
    if board is not None:
        return board
    return ",".join(str(x) for x in req)  # fallback: raw ids if topology probe failed


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
        cmd = [tt_smi, "-r", chips] if (chips and args == ["-r"]) else [tt_smi, *args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
            last = "tt-smi %s rc=%d" % (" ".join(cmd[1:]), r.returncode)
            if r.returncode == 0:
                return last
        except Exception as exc:  # noqa: BLE001
            last = "tt-smi %s FAILED (%s)" % (" ".join(cmd[1:]), exc)
    return "device reset (%s)" % last


def _reclaim_device(devices: str) -> str:
    """UNIVERSAL device reclaim used at EVERY recovery point: kill every process holding
    /dev/tenstorrent (except this process + its ancestors, so the supervisor/self is never killed),
    then tt-smi -r the chips. A wedge is cleared no matter WHO holds the device -- a stray child, a
    hung profiler, a busy pytest, or a leaked resident mesh. The one holder this cannot kill is the
    caller's own tree; an orchestrator self-hold is handled by exiting to the supervisor, which then
    reclaims from outside."""
    import glob as _glob

    protected = set()
    _p = os.getpid()
    for _ in range(64):
        if _p <= 1:
            break
        protected.add(_p)
        try:
            _p = int(open("/proc/%d/stat" % _p).read().split()[3])
        except Exception:  # noqa: BLE001
            break
    holders = set()
    for _n in _glob.glob("/dev/tenstorrent/*"):
        try:
            _r = subprocess.run(["fuser", _n], capture_output=True, text=True, timeout=30)
            holders.update(int(_t) for _t in (_r.stdout + " " + _r.stderr).split() if _t.strip().isdigit())
        except Exception:  # noqa: BLE001
            pass
    killed = []
    for _pid in holders - protected:
        try:
            os.kill(_pid, signal.SIGKILL)
            killed.append(_pid)
        except Exception:  # noqa: BLE001
            pass
    return "reclaimed device (killed holders %s) + %s" % (killed or "none", _reset_devices(devices))


def _pg_cpu_jiffies(pgid: int) -> int:
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    target = str(pgid)
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                data = fh.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        rp = data.rfind(")")
        if rp == -1:
            continue
        fields = data[rp + 2 :].split()
        if len(fields) > 12 and fields[2] == target:
            try:
                total += int(fields[11]) + int(fields[12])
            except ValueError:
                pass
    return total


def _llm_child_alive(pgid: int) -> bool:
    target = str(pgid)
    try:
        entries = os.listdir("/proc")
    except OSError:
        return False
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                data = fh.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        rp = data.rfind(")")
        if rp == -1:
            continue
        fields = data[rp + 2 :].split()
        if len(fields) <= 2 or fields[2] != target:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore").lower()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        if "claude" in cmd:
            return True
    return False


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
):
    """Run a DEVICE-touching subprocess so a device wedge can never hang the tool forever. Own session +
    hard timeout; on timeout SIGKILL the WHOLE process group + _reclaim_device (kill any holder + tt-smi
    -r); AND reap the group on every exit so no stale holder survives to wedge the next op. Returns (rc,
    combined stdout+stderr); rc is None when it timed out / was killed.

    Recovery-timeout tiers (all env-overridable, one knob each):
      BUILD   discover (stall-detector on no CPU progress)      -> PERF_MCP_DISCOVER_STALL_SEC (1200s), backstop PERF_MCP_DISCOVER_TIMEOUT (10800s)
      MEASURE gate / coverage / full-pipeline device runs       -> PERF_MCP_MEASURE_TIMEOUT  (1200s)
      ROUND   agent round (stall-detector on no-progress)       -> PERF_MCP_ROUND_STALL_SEC  (600s)"""
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
        if capture:
            out, _ = proc.communicate(timeout=timeout_s)
            out = out or ""
            rc = proc.returncode
        elif stall_s:
            import sys as _sys
            import threading as _th

            _act = [time.monotonic()]

            def _pump():
                try:
                    for _ln in proc.stdout:
                        _sys.stdout.write(_ln)
                        _sys.stdout.flush()
                        _act[0] = time.monotonic()
                except Exception:  # noqa: BLE001
                    pass

            _th.Thread(target=_pump, daemon=True).start()
            pgid = proc.pid
            start = time.monotonic()
            last_progress = start
            last_cpu = _pg_cpu_jiffies(pgid)
            max_gap = 0.0
            while proc.poll() is None:
                time.sleep(5)
                now = time.monotonic()
                cpu = _pg_cpu_jiffies(pgid)
                moved = cpu > last_cpu + 10 or _act[0] > last_progress or _llm_child_alive(pgid)
                last_cpu = cpu
                if moved:
                    max_gap = max(max_gap, now - last_progress)
                    last_progress = now
                limit = max(stall_s, int(3 * max_gap))
                idle = now - last_progress
                if idle >= limit:
                    print(
                        f"  [optimize/cc] {label or 'device subprocess'} STALLED (no output/CPU for "
                        f"{int(idle)}s > adaptive limit {limit}s) -- treating as wedge",
                        flush=True,
                    )
                    raise subprocess.TimeoutExpired(cmd, limit)
                if now - start >= timeout_s:
                    raise subprocess.TimeoutExpired(cmd, timeout_s)
            rc = proc.returncode
        else:
            proc.wait(timeout=timeout_s)
            rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        tail = _reclaim_device(devices) if reset_on_timeout else "process group killed"
        print(
            f"  [optimize/cc] {label or 'device subprocess'} TIMED OUT after {timeout_s}s "
            f"(likely a device wedge / leaked mesh) -- killed the whole process group + {tail}"
        )
        return None, ""
    finally:
        # Reap any lingering group member on EVERY exit. A daemon child (profiler, not-fully-closed mesh)
        # can outlive the main subprocess and keep holding the device -- a stale holder that wedges the
        # NEXT device op (observed: a completed baseline measurement leaked a holder that blocked the
        # coverage probe). Killing the whole process group here guarantees no leftover survives.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    return rc, out


def _progress_token(repo_root: Path, kernel_log: str):
    """Forward-progress signal for the round watchdog: (committed HEAD, kernel-attempt-log mtime). A live
    agent advances one of these; a device-wedged agent (blocked in a measurement) advances neither."""
    try:
        mt = os.path.getmtime(kernel_log)
    except OSError:
        mt = 0.0
    return (_git(repo_root, "rev-parse", "HEAD"), mt)


def _apply_adaptive_round_timers(baseline_wall_s) -> None:
    """Scale the module-level round hard-cap and full-model measure timeout to this
    module's observed baseline measurement wall, so a small module stops waiting the
    fixed 2400s/1200s before the watchdog acts while a large one keeps proportional
    headroom. The FROZEN stall check is untouched (it adapts by liveness signal, not
    clock). Only applied under module-level optimize."""
    w = max(1.0, float(baseline_wall_s or 0.0))
    max_no_progress = int(min(2400, max(300, 8.0 * w)))
    measure_timeout = int(min(1200, max(300, 6.0 * w)))
    os.environ["PERF_MCP_ROUND_MAX_SEC"] = str(max_no_progress)
    os.environ["PERF_MCP_MEASURE_TIMEOUT"] = str(measure_timeout)
    print(
        "  [optimize/cc] adaptive per-module timers (baseline wall %.0fs): round hard-cap %ds, "
        "measure timeout %ds" % (w, max_no_progress, measure_timeout),
        flush=True,
    )


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
    max_no_progress = int(os.environ.get("PERF_MCP_ROUND_MAX_SEC", str(max(stall_sec * 4, 2400))) or 2400)
    last_tok = _progress_token(repo_root, kernel_log)
    last_live = _liveness()
    _now0 = time.monotonic()
    last_active = _now0  # last sign of life (CPU / transcript / real progress)
    last_real = _now0  # last REAL progress (commit / recorded kernel attempt)
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
                elif live[0] != last_live[0] or (live[1] - last_live[1]) > 200:  # alive: transcript/CPU
                    last_live, last_active = live, _now
                if _now - last_active > stall_sec:
                    wedge_reason = "FROZEN %ds — no commit, no device CPU, no agent activity (real wedge)" % stall_sec
                    break
                if _now - last_real > max_no_progress:
                    wedge_reason = (
                        "UNPRODUCTIVE %ds — alive but no commit/kernel attempt in that time (hard cap)"
                        % max_no_progress
                    )
                    break
    finally:
        try:
            if _lf not in (None, subprocess.DEVNULL):
                _lf.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    rst = _reclaim_device(devices)
    print(
        "  [optimize/cc] WATCHDOG: round %s — killed the round + %s; next round starts a FRESH mcp "
        "server on the reset mesh." % (wedge_reason, rst)
    )
    return True


def _baseline_ms() -> float | None:
    try:
        import tempfile

        d = json.loads((Path(tempfile.gettempdir()) / "perf_mcp_baseline.json").read_text())
        return float(d["device_ms"]) if d.get("device_ms") is not None else None
    except Exception:  # noqa: BLE001
        return None


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
        _rr = sorted(_runs.rglob("residual_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if _rr:
            residual = json.loads(_rr[0].read_text())
    except Exception:  # noqa: BLE001
        pass
    text = mod.render_summary(
        kernel_log,
        _baseline_ms(),
        model=model_name,
        task=task,
        metric=metric,
        committed_wins=wins,
        opt_branch=branch,
        perf_test=perf_test,
        report_csv=report_csv,
        residual=residual,
        before_ms=before_ms,
        after_ms=after_ms,
        baseline_profile=(
            json.loads(Path(report_csv).parent.joinpath("baseline_profile.json").read_text())
            if report_csv and Path(report_csv).parent.joinpath("baseline_profile.json").is_file()
            else None
        ),
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
            try:
                from scripts.tt_hw_planner.run_report import refresh_bringup_section

                refresh_bringup_section(_demo)
            except Exception:
                pass
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
            mod.upsert_report_section(_demo, _key, _block)
            print(f"  [optimize/cc] report updated: {_demo / 'RUN_REPORT.md'} ({_key} section)")
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
            _h.post_decision(hitl_dir, "commit")
            print("  [hitl] committed.", flush=True)
        else:
            _git(repo_root, "checkout", "--", ".")
            _h.post_decision(hitl_dir, "revert")
            print("  [hitl] reverted.", flush=True)


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
    kernel_log = f"/tmp/cc_kernlog_{model_name}_{task}.json"
    try:
        os.path.exists(kernel_log) and os.remove(kernel_log)  # fresh ladder state per pipeline
    except OSError:
        pass
    _capture_board_topology()  # snapshot chip->board reset map while the device is healthy (reset-only)
    cfg = _mcp_config(repo_root, manifest_path, pipe, devices, kernel_log)
    _cov_env = cfg["mcpServers"]["perf-mcp"]["env"]
    _cov, _cov_facts = _coverage_layers(
        repo_root,
        _cov_env,
        devices,
        pipe.get("perf_test"),
        pipe.get("case"),
        model_name=model_name,
        config_ref=config_ref,
    )
    if _cov:
        _cov_env["TT_PERF_LAYERS"] = str(_cov)
        print(f"  [optimize/cc] coverage-sized profiling window: TT_PERF_LAYERS={_cov} (covers all block types)")
        _depth_env = _bridge_depth_env(repo_root, _cov_env, devices, pipe.get("perf_test"), pipe.get("case"), _cov)
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
    try:
        (Path(tempfile.gettempdir()) / "perf_mcp_full_pipeline_baseline.json").unlink()
    except Exception:
        pass
    _bl_t0 = time.monotonic()
    before_ms = _fullpipe_e2e(repo_root, mcp_env, devices, "BEFORE")
    if os.environ.get("TT_PERF_MODULE_LEVEL") == "1":
        _apply_adaptive_round_timers(time.monotonic() - _bl_t0)
    rounds, can_stop, halted = 0, False, False
    stall_sec = int(os.environ.get("PERF_MCP_ROUND_STALL_SEC", "600") or "600")
    max_wedge = int(os.environ.get("PERF_MCP_MAX_WEDGE_STRIKES", "2") or "2")
    wedge_strikes = 0
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
            print(f"  [optimize/cc] HALT — install tt-lang first, then re-run: {st.get('reason')}")
            halted = True
            break
        if st.get("can_stop"):
            can_stop = True
            break
        wedged = _run_round_with_watchdog(round_cmd, repo_root, devices, kernel_log, stall_sec)
        if wedged:
            wedge_strikes += 1
            if wedge_strikes >= max_wedge:
                if os.environ.get("PERF_MCP_SUPERVISED") == "1":
                    print(
                        "  [optimize/cc] WATCHDOG: %d consecutive wedged rounds — exiting so the supervisor "
                        "reclaims the device (kills holders + reset) and restarts; ladder state is preserved."
                        % wedge_strikes,
                        flush=True,
                    )
                    _reclaim_device(devices)
                    os._exit(75)
                print(
                    "  [optimize/cc] WATCHDOG: %d consecutive wedged rounds — aborting this pipeline "
                    "(all committed wins are safe)." % wedge_strikes
                )
                break
        else:
            wedge_strikes = 0
        rounds += 1
    _stop_watcher.set()
    after_ms = _fullpipe_e2e(repo_root, mcp_env, devices, "AFTER")
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
        before_ms=before_ms,
        after_ms=after_ms,
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


def _resolve_model_id(demo_dir, hint=None) -> str | None:
    if _is_cached_model_id(hint):
        return hint
    try:
        for p in Path(demo_dir).rglob("*.py"):
            try:
                txt = p.read_text(errors="ignore")
            except OSError:
                continue
            for cand in _HF_ID_RE.findall(txt):
                if _is_cached_model_id(cand):
                    return cand
    except Exception:
        return None
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


def _model_weight_bytes(demo_dir, hint=None) -> int:
    total = 0
    try:
        for p in Path(demo_dir).rglob("*"):
            if p.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth") and p.is_file():
                total += p.stat().st_size
    except Exception:
        total = 0
    if total:
        return total
    mid = _resolve_model_id(demo_dir, hint)
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
        weight_bytes = _model_weight_bytes(demo_dir, model_id_hint)
        if not (cap and weight_bytes):
            return
        cfg = manifest.get("model_config") or {}
        if not cfg.get("hidden_size"):
            mid = _resolve_model_id(demo_dir, model_id_hint)
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
    perf_dir = repo_root / PERF_DIR
    manifest_path = str(_latest_manifest(perf_dir))
    _seqf = Path(manifest_path).parent / "perf_seq_len"
    if _seqf.is_file():
        _seq = _seqf.read_text().strip()
        if _seq:
            os.environ["TT_PERF_SEQ_LEN"] = _seq
            print(f"  [optimize/cc] perf workload seq pinned to {_seq} (baseline shape-retry); propagated to loop")
    _decide_parallelism_route(demo_dir, manifest, repo_root, metric, devices, model_id_hint)
    model_rel = os.path.relpath(demo_dir, repo_root)
    model_name = Path(demo_dir).name
    _cfg_ref = _resolve_model_id(demo_dir, model_id_hint) or str(demo_dir)
    pipes = pipelines_from_manifest(manifest, model_rel)
    is_mm = manifest.get("pathmap", {}).get("is_multimodal")
    print(f"  [optimize/cc] discovered pipelines: {[p['task'] for p in pipes]} (multimodal={is_mm})")
    if e2e_only:
        os.environ["PERF_MCP_FULLPIPE_E2E"] = "1"
        for pipe in pipes:
            kernel_log = f"/tmp/cc_kernlog_{model_name}_{pipe['task']}.json"
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
