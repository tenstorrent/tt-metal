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
import subprocess
from pathlib import Path

PERF_DIR = "models/experimental/perf_automation"
CC_DIR = PERF_DIR + "/cc_optimize"
DEFAULT_MAX_ROUNDS = 20

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

termination_check() is the SOLE authority on whether more optimization is needed. It returns a DETERMINISTIC per-op CHECKLIST and a single next_target = {{op, op_class, grid, bound_by, rung}} you MUST work next. The per-op ladder ORDER is: knob:grid -> knob:dtype -> tt-lang -> cpp. An op is "nothing left" ONLY when every rung is ticked; you may STOP ONLY when can_stop=true.

LOOP:
  git_head -> termination_check -> read next_target.
  REUSE-FIRST: call recall_knobs(next_target.op_class, next_target.grid, next_target.bound_by) and APPLY/ADAPT any matching catalogued knob (heed its negative knowledge) BEFORE improvising one.
  Do EXACTLY next_target.rung on next_target.op:
    knob:grid  -> full-grid program_config. check_pcc; measure_candidate; commit a real win else revert. record_kernel_attempt(op,'grid',measured_ms,beat_baseline).
    knob:dtype -> lower that op's WEIGHT dtype (bf16->bf8_b->bf4_b). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'dtype',measured_ms,beat_baseline) EVEN IF pcc forced a revert (that marks the knob tried).
    tt-lang    -> author a tt-lang (ttl) kernel (Read GUIDELINES/11). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'tt-lang',measured_ms,beat_baseline).
    cpp        -> author a C++ Metalium kernel via ttnn.generic_op (Read GUIDELINES/12). check_pcc; measure_candidate; commit a win else revert. record_kernel_attempt(op,'cpp',measured_ms,beat_baseline).
  (IRON RULE: a real win = check_pcc ok AND check_full_pipeline_latency ok (moved TOWARD the target / not diverged) AND verdict 'valid' AND is_real_gain. REJECTED, pcc-fail, or a DIVERGED full-pipeline latency is never a win — revert. Note: check_full_pipeline_latency never fails for missing the target, only for getting SLOWER than best-so-far.)
  WRITE-BACK: after you COMMIT a win you IMPROVISED (recall_knobs had no match), call distill_knob to persist the general technique; if the win RE-USED a provisional lever learned on another model, pass its id to distill_knob to graduate it.
  Re-run termination_check. Repeat. NEVER stop while can_stop=false. NEVER reason a lever "won't help" — prove it by measuring + recording the attempt.

LEAVE CLEAN (commit wins, revert in-progress edits); end with git_head. Report start->final {metric}, committed wins, and per blocking op which rungs were done + measured ms."""


def _visible_devices(devices: str) -> str | None:
    """Translate the --devices spec to a TT_VISIBLE_DEVICES value the same way before_loop does (UMD
    wants explicit ids or the var UNSET): 'single' -> '0', 'all'/'' -> None (all chips visible, var
    left unset), explicit ids ('0,1,2,3') pass through. Setting the literal 'all'/'single' breaks UMD."""
    if not devices or devices == "all":
        return None
    if devices == "single":
        return "0"
    return devices


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
    rc = subprocess.run(cmd, cwd=str(perf_dir), env=cc_env(repo_root, devices)).returncode
    if rc != 0:
        return None
    mani = _latest_manifest(perf_dir)
    return json.loads(mani.read_text()) if mani else None


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
    try:
        r = subprocess.run(
            [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
            cwd=str(repo_root / PERF_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except Exception:  # noqa: BLE001 — gate crashed/timed out -> treat as not-done, loop retries
        return {"can_stop": False, "halt": False, "reason": ""}
    out = r.stdout or ""
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
    print(
        f"  [optimize/cc] measuring FULL-model end-to-end ({label}) — ALL 52 layers, no tracy (one slow run, minutes)..."
    )
    ms = None
    try:
        r = subprocess.run(
            [_python_bin(repo_root), "-c", code, str(repo_root / CC_DIR)],
            cwd=str(repo_root / PERF_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=5400,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimize/cc] FULL-model end-to-end ({label}) skipped ({exc})")
        return None
    for line in ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines():
        if line.startswith("FULLPIPE_MS="):
            try:
                ms = float(line.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                ms = None
        if "[full-pipeline-gate]" in line:
            print("  [optimize/cc] " + line.strip())
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
    try:
        r = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=1800)
    except Exception:  # noqa: BLE001
        return None, ""
    raw = (r.stdout or "") + "\n" + (r.stderr or "")
    sigs = None
    for line in raw.splitlines():
        if line.startswith("PERF_OP_SIGS="):
            try:
                sigs = set(json.loads(line.split("=", 1)[1]))
            except Exception:  # noqa: BLE001
                sigs = None
    if not sigs:
        return None, raw
    return sigs, raw


def _coverage_layers(repo_root: Path, mcp_env: dict, devices: str, node, case, n_layers: int = 52):
    """MODEL-AGNOSTIC profiling-window sizing: grow TT_PERF_LAYERS until the set of distinct ttnn op
    signatures SATURATES (a deeper window adds no new op type) — so the tracy 2-layer-style slice actually
    covers EVERY block type, not just whatever falls in the first N layers. Homogeneous models saturate at
    1-2; heterogeneous ones (mamba/attention/MoE interleaved) grow until all types appear. No per-model
    layer maps. Returns (layer_count_or_None, facts) — facts from the deepest probe feed the scorecard.
    Disable via PERF_MCP_COVERAGE_SIZING=0."""
    facts: dict = {}
    if os.environ.get("PERF_MCP_COVERAGE_SIZING", "1") != "1" or not node:
        return None, facts
    results: list = []
    for k in (2, 4, 8, 16):
        k = min(k, n_layers)
        sigs, raw = _run_op_sigs(repo_root, mcp_env, devices, node, case, k)
        if sigs is None:
            break
        facts = _parse_facts(raw, sigs)
        print(f"  [optimize/cc] coverage probe: {k} layer(s) -> {len(sigs)} distinct op signatures")
        results.append((k, sigs))
        if k >= n_layers:
            break
    if not results:
        return None, facts
    max_sigs = max((s for _, s in results), key=len)
    if results[-1][1] == max_sigs and len(results) >= 2 and results[-2][1] != max_sigs and results[-1][0] >= n_layers:
        print("  [optimize/cc] coverage still growing at the depth cap — op coverage may be incomplete")
    for k, s in results:
        if s == max_sigs:
            return k, facts
    return results[-1][0], facts


def _print_scorecard(devices: str, manifest: dict, pipe: dict, facts: dict, before_ms, after_ms) -> None:
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


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _baseline_ms() -> float | None:
    try:
        import tempfile

        d = json.loads((Path(tempfile.gettempdir()) / "perf_mcp_baseline.json").read_text())
        return float(d["device_ms"]) if d.get("device_ms") is not None else None
    except Exception:  # noqa: BLE001
        return None


def _emit_summary(repo_root: Path, kernel_log: str, model_name: str, task: str, metric: str, start_sha: str) -> None:
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
    text = mod.render_summary(
        kernel_log, _baseline_ms(), model=model_name, task=task, metric=metric, committed_wins=wins, opt_branch=branch
    )
    print("\n" + text + "\n")
    md = _latest_manifest(repo_root / PERF_DIR)
    if md:
        try:
            (md.parent / "summary.md").write_text(text)
            print(f"  [optimize/cc] summary saved: {md.parent / 'summary.md'}")
        except OSError:
            pass


def optimize_pipeline(
    repo_root: Path,
    manifest_path: str,
    pipe: dict,
    devices: str,
    metric: str,
    model_name: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict:
    """Drive one pipeline: claude -p re-invoked until the gate's can_stop, bounded by max_rounds."""
    task = pipe["task"]
    kernel_log = f"/tmp/cc_kernlog_{model_name}_{task}.json"
    try:
        os.path.exists(kernel_log) and os.remove(kernel_log)  # fresh ladder state per pipeline
    except OSError:
        pass
    cfg = _mcp_config(repo_root, manifest_path, pipe, devices, kernel_log)
    _cov_env = cfg["mcpServers"]["perf-mcp"]["env"]
    _cov, _cov_facts = _coverage_layers(repo_root, _cov_env, devices, pipe.get("perf_test"), pipe.get("case"))
    if _cov:
        _cov_env["TT_PERF_LAYERS"] = str(_cov)
        print(f"  [optimize/cc] coverage-sized profiling window: TT_PERF_LAYERS={_cov} (covers all block types)")
    cfg_path = repo_root / CC_DIR / f".mcp_config_{model_name}_{task}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    prompt = _PROMPT.format(model=model_name, task=task, metric=metric)
    start_sha = _git(repo_root, "rev-parse", "HEAD")
    mcp_env = cfg["mcpServers"]["perf-mcp"]["env"]
    before_ms = _fullpipe_e2e(repo_root, mcp_env, devices, "BEFORE")
    rounds, can_stop, halted = 0, False, False
    while rounds < max_rounds:
        st = _gate_status(repo_root, mcp_env, devices)
        if st.get("halt"):
            print(f"  [optimize/cc] HALT — install tt-lang first, then re-run: {st.get('reason')}")
            halted = True
            break
        if st.get("can_stop"):
            can_stop = True
            break
        subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--mcp-config",
                str(cfg_path),
                "--strict-mcp-config",
                "--allowedTools",
                *_ALLOWED_TOOLS,
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            cwd=str(repo_root),
            env=cc_env(repo_root, devices),
        )
        rounds += 1
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
    _print_scorecard(devices, _mf, pipe, _cov_facts, before_ms, after_ms)
    _emit_summary(repo_root, kernel_log, model_name, task, metric, start_sha)
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
                subprocess.run(["git", "checkout", "-B", branch], cwd=wt, capture_output=True, text=True)
            else:
                subprocess.run(["git", "checkout", "--orphan", branch], cwd=wt, capture_output=True, text=True)
                subprocess.run(["git", "rm", "-rf", "."], cwd=wt, capture_output=True, text=True)
            dest = Path(wt) / _GL_REL
            dest.mkdir(parents=True, exist_ok=True)
            for g in grads:
                shutil.copy2(g, dest / g.name)
            subprocess.run(["git", "add", _GL_REL], cwd=wt, capture_output=True, text=True)
            c = subprocess.run(
                ["git", "commit", "-m", f"[perf-catalog] graduated knobs ({len(grads)})"],
                cwd=wt,
                capture_output=True,
                text=True,
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
                ["git", "worktree", "remove", "--force", wt], cwd=str(repo_root), capture_output=True, text=True
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
    for pipe in pipes:
        print(f"  [optimize/cc] === optimizing pipeline: {pipe['task']} ===")
        results.append(optimize_pipeline(repo_root, manifest_path, pipe, devices, metric, model_name, max_rounds))
    if sync_catalog:
        catalog_push(repo_root, catalog_remote, catalog_branch)
    return {"pipelines": pipes, "is_multimodal": is_mm, "results": results}
