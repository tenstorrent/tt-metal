# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e — LLM-driven end-to-end pipeline builder (build agent + grader agent)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _verbose() -> bool:
    """Screen-verbosity gate (matches the cli.py TT_HW_PLANNER_VERBOSE convention).
    Off by default: keep the terminal clean; the full agent stream always lands
    in the per-phase log file regardless."""
    return os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")


def _md_to_terminal(text: str) -> str:
    """Strip the markdown markup (** , `, leading #) the agent emits so a
    fallback summary reads cleanly on a terminal instead of as raw .md source."""
    out = []
    for ln in (text or "").splitlines():
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", ln)
        s = re.sub(r"`([^`]+)`", r"\1", s)
        s = s.replace("**", "")
        s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s)
        out.append("  " + s)
    return "\n".join(out)


def _render_grader_report(demo_dir: Path) -> bool:
    """Render the structured grader_report.json as a clean, aligned terminal
    block (no markdown). Returns True if rendered, False if unavailable —
    callers fall back to a stripped version of the agent's prose."""
    try:
        rep = json.loads((demo_dir / "grader_report.json").read_text())
    except Exception:
        return False

    rule = "  " + "─" * 74
    lines = [rule, f"  GRADER REPORT — {demo_dir.name}", rule]

    calls = rep.get("calls") or []
    if calls:
        lines.append(f"  {'Call':<6} {'Re-run':<7} {'Final PCC':<38} Audit")
        for c in calls:
            pccs = c.get("final_pcc") or []
            try:
                pcc_s = " / ".join(f"{float(x):.6f}" for x in pccs)
            except Exception:
                pcc_s = ", ".join(str(x) for x in pccs)
            lines.append(
                f"  {str(c.get('call', '?')):<6} {str(c.get('rerun', '?')):<7} "
                f"{pcc_s:<38} {c.get('source_audit', '')}"
            )
        lines.append(rule)

    def _ok(d):
        return "pass" if d.get("ok") else "FAIL"

    struct = rep.get("structure") or {}
    nw = rep.get("no_waste") or {}
    holes = rep.get("holes") or []
    lines.append(f"  {'Structure':<11} {_ok(struct)}")
    nw_extra = ""
    if nw:
        nw_extra = f" — {nw.get('names_present', '?')}/{nw.get('graduated_total', '?')} graduated invoked"
        missing = nw.get("missing") or []
        if missing:
            nw_extra += f", missing: {', '.join(map(str, missing))}"
    lines.append(f"  {'No-waste':<11} {_ok(nw)}{nw_extra}")
    if holes:
        lines.append(f"  {'Holes':<11} {len(holes)}")
        for h in holes[:8]:
            lines.append(
                f"    - [{h.get('severity', '?')}] {h.get('id', '?')} " f"@ {h.get('file', '?')}:{h.get('lines', '?')}"
            )
    else:
        lines.append(f"  {'Holes':<11} none")
    lines.append(f"  {'Verdict':<11} {rep.get('verdict', '?')}")
    lines.append(rule)
    print("\n" + "\n".join(lines))
    return True


def _render_compute_split(model_id: str) -> None:
    """Show how much of the pipeline runs natively on the TT device vs torch on
    CPU — reusing the exact split the auto-iterate loop prints (component-level
    + op-level), read from bringup_status.json + the op-synth manifests."""
    try:
        from ..cli import _format_compute_split, _format_op_split
    except Exception:
        return
    lines = []
    try:
        lines += _format_compute_split(model_id, label="compute split (TT device vs CPU)")
    except Exception:
        pass
    try:
        lines += _format_op_split(model_id, label="operations")
    except Exception:
        pass
    if lines:
        print()
        for ln in lines:
            print(ln)


_G1_TORCH_DELEGATION = (
    r"self\._torch_module\s*\(",
    r"self\.torch_module\s*\(",
    r"_get_torch_submodule\s*\(",
)

_G5_HOST_SAMPLING = (
    r"torch\.argmax\s*\(",
    r"torch\.multinomial\s*\(",
    r"torch\.topk\s*\(",
)
_G5_HOST_XFER = ("from_torch", "to_torch", "from_device", "to_device")


def _run_deterministic_gates(demo_dir: Path, pcc: float, timeout_s: int):
    """Model-agnostic gate runner: G1 native, G2/G3 (run tests/e2e), G4 demo/ structure. Returns (ok, reasons)."""
    reasons = []
    e2e_dir = demo_dir / "tests" / "e2e"
    test_files = sorted(e2e_dir.glob("test_*.py")) if e2e_dir.is_dir() else []
    if not test_files:
        return False, ["G2/G3: no tests/e2e/test_*.py to run"]

    demo_subdir = demo_dir / "demo"
    demo_entrypoints = sorted(demo_subdir.glob("demo_*.py")) if demo_subdir.is_dir() else []
    if not demo_entrypoints:
        reasons.append("G4 structure: no runnable demo/demo_*.py entrypoint (standard layout requires per-Call demos)")
    else:
        no_main = [p.name for p in demo_entrypoints if "__main__" not in p.read_text(errors="ignore")]
        if no_main:
            reasons.append(f"G4 structure: demo entrypoint(s) missing `__main__` (not runnable): {', '.join(no_main)}")
    if not (demo_dir / "tt").is_dir():
        reasons.append("G4 structure: no tt/ package (standard demo layout)")
    if not (demo_dir / "README.md").is_file():
        reasons.append("G4 structure: no README.md (standard demo layout)")

    stub_dir = demo_dir / "_stubs"
    nonnative = []
    for p in sorted(stub_dir.glob("*.py")) if stub_dir.is_dir() else []:
        if p.name.startswith("_"):
            continue
        try:
            src = p.read_text(errors="ignore")
        except Exception:
            continue
        if any(re.search(pat, src) for pat in _G1_TORCH_DELEGATION):
            nonnative.append(p.stem)
    if nonnative:
        reasons.append("G1: stub(s) delegate to the torch reference (not native ttnn): " + ", ".join(nonnative[:8]))

    if os.environ.get("E2E_REQUIRE_ON_DEVICE") == "1" and os.environ.get("E2E_ALLOW_HOST_DECODE") != "1":
        tt_dir = demo_dir / "tt"
        host_hits = []
        for p in sorted(tt_dir.glob("*.py")) if tt_dir.is_dir() else []:
            try:
                src = p.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if any(re.search(pat, src) for pat in _G5_HOST_SAMPLING):
                host_hits.append(p.stem)
        if host_hits:
            reasons.append(
                "G5 on-device: pipeline samples on the HOST (torch.argmax/topk/multinomial) — decode is not "
                "fully on-device, so trace + 2CQ is blocked: " + ", ".join(host_hits[:8]) + " (move sampling "
                "on-device with ttnn; set E2E_ALLOW_HOST_DECODE=1 to waive for a genuinely host-bound model)"
            )
        else:
            repo = demo_dir
            for parent in demo_dir.parents:
                if (parent / "models").is_dir():
                    repo = parent
                    break
            probe = repo / "models" / "experimental" / "perf_automation" / "cc_optimize" / "_op_sig_probe.py"
            if probe.is_file() and test_files:
                cap = int(os.environ.get("E2E_HOST_XFER_MAX", "6"))
                penv = dict(os.environ)
                penv["TT_METAL_HOME"] = str(repo)
                penv["PYTHONPATH"] = str(repo) + os.pathsep + penv.get("PYTHONPATH", "")
                penv["TT_PERF_MAX_NEW_TOKENS"] = "2"
                penv.pop("TT_METAL_DEVICE_PROFILER", None)
                _pb = repo / "python_env" / "bin" / "python"
                _pbin = str(_pb) if _pb.exists() else sys.executable
                try:
                    pr = subprocess.run(
                        [_pbin, str(probe), str(test_files[0].relative_to(repo))],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                        cwd=str(repo),
                        env=penv,
                    )
                    xfer = []
                    for line in ((pr.stdout or "") + "\n" + (pr.stderr or "")).splitlines():
                        if line.startswith("PERF_OP_SIGS="):
                            import json as _json

                            try:
                                sigs = _json.loads(line.split("=", 1)[1])
                            except Exception:  # noqa: BLE001
                                sigs = []
                            xfer = sorted({s.split("(")[0] for s in sigs if any(h in s for h in _G5_HOST_XFER)})
                    if len(xfer) > cap:
                        reasons.append(
                            f"G5 on-device: {len(xfer)} host round-trip op types in the forward (weight "
                            f"streaming / host readback) exceeds {cap} — not fully on-device, trace+2CQ blocked: "
                            + ", ".join(xfer[:8])
                            + " (keep weights resident + sample on-device; "
                            "E2E_ALLOW_HOST_DECODE=1 or raise E2E_HOST_XFER_MAX to waive)"
                        )
                except Exception:  # noqa: BLE001 — probe failure is not a gate failure (best-effort)
                    pass

    py = sys.executable
    for parent in [Path.cwd(), *demo_dir.parents]:
        cand = parent / "python_env" / "bin" / "python"
        if cand.exists():
            py = str(cand)
            break
    demo_repo_root = demo_dir
    for parent in demo_dir.parents:
        if (parent / "models").is_dir():
            demo_repo_root = parent
            break
    gate_env = dict(os.environ)
    gate_env["PYTHONPATH"] = str(demo_repo_root) + os.pathsep + gate_env.get("PYTHONPATH", "")
    gate_env["TT_METAL_HOME"] = str(demo_repo_root)
    pytest_out = ""
    try:
        proc = subprocess.run(
            [py, "-m", "pytest", str(e2e_dir), "-p", "no:cacheprovider", "-rA", "-s"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(demo_repo_root),
            env=gate_env,
        )
        pytest_out = proc.stdout or ""
        if proc.returncode != 0:
            tail = "\n".join(pytest_out.splitlines()[-15:])
            reasons.append(f"G2/G3: tests/e2e did not pass (pytest rc={proc.returncode}); tail:\n{tail}")
    except subprocess.TimeoutExpired:
        reasons.append(f"G2/G3: tests/e2e exceeded {timeout_s}s with no verdict")

    for cnt, kind in re.findall(r"(\d+)\s+(xfailed|xpassed|skipped|errors?)\b", pytest_out):
        if int(cnt) > 0:
            reasons.append(
                f"G2/G3: tests/e2e reported {cnt} {kind} — a gate test may only PASS; "
                f"xfail/skip/error is not an accepted outcome (fix it or it stays a gate failure)"
            )

    for _val in re.findall(r"PCC[^=\n]*=\s*(-?\d+(?:\.\d+)?)", pytest_out):
        if float(_val) < pcc:
            reasons.append(f"G3: measured PCC {_val} < required {pcc} (tool-enforced threshold)")

    for p in test_files:
        try:
            src = p.read_text(errors="ignore")
        except Exception:
            continue
        if re.search(r"pytest\.xfail|mark\.xfail|pytest\.skip|assert\s+True\b", src):
            reasons.append(f"honesty: {p.name} contains pytest.xfail / pytest.skip / assert True")

    return (len(reasons) == 0), reasons


def _build_cc_fix_prompt(*, model_id, demo_dir, pcc) -> str:
    """Per-round prompt for the emit-e2e cc engine. The gate is the sole authority; the agent works
    exactly the failing gates it names and never weakens them."""
    return (
        f"You are finishing the end-to-end TTNN pipeline for {model_id} in {demo_dir} (required e2e "
        f"PCC >= {pcc}).\n"
        "LOOP every iteration: call mcp__e2e-mcp__termination_check FIRST. It is the SOLE authority on "
        "whether you are done (can_stop=true) and returns next_target = the FAILING gates: G1 (stubs "
        "must be native ttnn, not torch-delegating), G2/G3 (tests/e2e must PASS on device with measured "
        "PCC >= threshold; xfail/skip/error is NOT acceptable), G4 (demo/ + tt/ + README structure).\n"
        "Fix EXACTLY the failing gates by editing tests/e2e/, _stubs/, demo/, tt/ as needed. NEVER "
        "weaken, xfail, skip, or assert-True a gate. Re-run termination_check after each fix. STOP only "
        "when can_stop=true; if it is already true, do nothing."
    )


def _run_emit_e2e_cc(*, model_id, demo_dir, pcc, timeout_s, agent_bin, max_rounds) -> int:
    """emit-e2e cc engine: after the builder runs, drive the fix loop through the shared cc harness
    against the e2e_mcp deterministic gate (which REUSES the same G1–G4 `_run_deterministic_gates` the
    legacy loop uses). The gate is the sole stop authority. Returns 0 iff the gate reports can_stop."""
    import json as _json
    import os as _os

    from .. import cc_harness

    repo_root = Path(__file__).resolve().parents[3]
    thp_dir = repo_root / "scripts" / "tt_hw_planner"
    server_path = thp_dir / "e2e_mcp.py"
    pybin = str(repo_root / "python_env" / "bin" / "python")
    if not Path(pybin).is_file():
        pybin = sys.executable
    mcp_env = {
        "E2E_MCP_DEMO_DIR": str(demo_dir),
        "E2E_MCP_PCC": str(pcc),
        "E2E_MCP_TIMEOUT": str(timeout_s),
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{_os.pathsep}/usr/bin:/bin",
    }
    cfg = cc_harness.build_mcp_config(pybin, server_path, mcp_env, "e2e-mcp")
    cfg_path = thp_dir / f".e2e_mcp_config_{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}.json"
    cfg_path.write_text(_json.dumps(cfg, indent=2))
    env = dict(_os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = str(repo_root)

    def gate_fn():
        return cc_harness.gate_status(pybin, thp_dir, "e2e_mcp", mcp_env, repo_root)

    prompt = _build_cc_fix_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
    allowed = ["mcp__e2e-mcp__termination_check", "Read", "Edit", "Write", "Bash", "Grep", "Glob"]
    print("\n  ===== PHASE 3 (cc engine): harness fix-loop on the e2e gate =====\n")
    res = cc_harness.run_cc_loop(
        prompt=prompt,
        mcp_config_path=cfg_path,
        allowed_tools=allowed,
        cwd=repo_root,
        env=env,
        gate_fn=gate_fn,
        max_rounds=max_rounds,
        claude_bin=agent_bin,
    )
    final = gate_fn()
    sep = "=" * 78
    print("\n" + sep)
    print(f"  cc engine: rounds={res['rounds']} can_stop={final.get('can_stop')} halted={res['halted']}")
    print(sep)
    return 0 if final.get("can_stop") else 1


def _source_phase2_shard_stubs(demo_dir: Path) -> list:
    stub_dir = demo_dir / "_stubs"
    if not stub_dir.is_dir():
        return []
    sourced = []
    for snap in sorted(stub_dir.glob("*.py.last_good_sharded")):
        live = snap.with_suffix("")
        if live.suffix != ".py":
            continue
        try:
            live.write_bytes(snap.read_bytes())
            sourced.append(live.stem)
        except OSError:
            pass
    return sourced


def cmd_emit_e2e(args) -> int:
    try:
        from ..cli import _quiet_framework_logging

        _quiet_framework_logging()
    except Exception:
        pass
    model_id = args.model_id
    demo_dir = _resolve_demo_dir(args)
    pcc = float(getattr(args, "pcc_target", 0.9) or 0.9)
    agent_model = getattr(args, "model", None) or "opus"
    agent_bin = getattr(args, "agent_bin", "claude") or "claude"
    timeout_s = int(getattr(args, "agent_timeout_s", 0) or 0) or 14400
    skip_grade = bool(getattr(args, "no_grade", False))
    max_grade_rounds = int(getattr(args, "max_grade_rounds", 0) or 0) or 3

    # One consolidated full log for the whole run (builder + grader + fix
    # appended in order). Clean screen, complete log, no per-phase scatter.
    import re as _re

    _safe = _re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    full_log = Path("generated") / f"emit_e2e_{_safe}_full.log"
    try:
        full_log.parent.mkdir(parents=True, exist_ok=True)
        full_log.write_text("")  # start fresh each run
    except Exception:
        full_log = None

    sep = "=" * 78
    print(sep)
    print(f"  EMIT-E2E (LLM agent)  {model_id}")
    print(f"  demo_dir={demo_dir}  pcc>={pcc}  model={agent_model}")
    if full_log is not None:
        print(f"  full log (complete transcript) → {full_log}")
    print(sep)

    _pc = _planned_parallelism(model_id, args)
    _parallel_note = _parallelism_prompt_block(_pc)
    if _pc is not None and _pc.chips > 1:
        print(f"  chip placement: {_pc.chips}-chip mesh → TP={_pc.tp} x DP={_pc.dp} (kernel-viability selected)")
        print("  builder will open the mesh at this split; tt-metal auto-discovers the fabric topology.")
        if _pc.tp > 1:
            _sharded = _source_phase2_shard_stubs(demo_dir)
            if _sharded:
                print(
                    f"  [shard] sourced Phase-2 TP-sharded stubs (compose as-is, do NOT replicate): {', '.join(_sharded)}"
                )
                _parallel_note += (
                    f"\n\nPHASE-2 SHARD STUBS ({', '.join(_sharded)}): these _stubs ALREADY implement the "
                    f"proven TP={_pc.tp} split (ShardTensorToMesh + all_reduce/cluster_axis). Compose them "
                    f"AS-IS on the mesh — do NOT rewrite their sharding to replication. Only components with "
                    f"NO shard implementation may be replicated. The final pipeline MUST contain "
                    f"ShardTensorToMesh + a collective (all_reduce/all_gather); a pure-replication pipeline is "
                    f"NOT an acceptable TP={_pc.tp} result."
                )
            else:
                print(
                    "  [shard] no Phase-2 (.last_good_sharded) stubs present — components will be REPLICATED "
                    "(TP=1). Run the shard bring-up (promote TT_HW_PLANNER_SHARD=1) to produce them for real TP."
                )
        print(sep)

    print("\n  ===== PHASE 1+2: BUILDER agent (plan → build → iterate) =====\n")
    build_prompt = _build_agent_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc, parallel_note=_parallel_note)
    rc_build, build_final = _run_agent(
        prompt=build_prompt,
        agent_bin=agent_bin,
        agent_model=agent_model,
        timeout_s=timeout_s,
        label="builder",
        log_path=full_log,
    )
    if rc_build != 0:
        print(f"\n  ✗ builder agent exited rc={rc_build}; skipping grade")
        return 1
    print("  ✓ builder finished (exit 0)")

    if (getattr(args, "engine", "cc") or "cc") == "cc":
        return _run_emit_e2e_cc(
            model_id=model_id,
            demo_dir=demo_dir,
            pcc=pcc,
            timeout_s=timeout_s,
            agent_bin=agent_bin,
            max_rounds=max_grade_rounds,
        )

    if skip_grade:
        print("\n  (--no-grade) skipping independent grader phase.\n")
        # No grader report to render; show a clean (markdown-stripped) build summary.
        if (build_final or "").strip():
            print(_md_to_terminal(build_final))
        _render_compute_split(model_id)
        return 0

    rule = "  " + "─" * 74
    for rnd in range(1, max_grade_rounds + 1):
        print(f"\n  ===== PHASE 3: DETERMINISTIC GATES (tool-run, round {rnd}/{max_grade_rounds}) =====\n")
        ok, reasons = _run_deterministic_gates(demo_dir, pcc, timeout_s)
        print(rule)
        print(f"  GATES (tool-enforced, not agent-reported): {'PASS' if ok else 'FAIL'}")
        for r in reasons:
            print(f"    - {r}")
        print(rule)
        _render_compute_split(model_id)

        if ok:
            print("\n" + sep)
            print(f"  ✓ TOOL-ENFORCED GATES: PASS (round {rnd}) — verdict computed by the tool")
            print(sep)
            return 0
        if rnd == max_grade_rounds:
            break
        print(f"\n  ===== FIX agent (round {rnd}/{max_grade_rounds - 1}) — addressing gate failures =====\n")
        fix_prompt = _build_fix_prompt(
            model_id=model_id,
            demo_dir=demo_dir,
            pcc=pcc,
            grader_findings="TOOL-ENFORCED GATES FAILED (fix these, do not weaken them):\n"
            + "\n".join(f"  - {r}" for r in reasons),
        )
        _run_agent(
            prompt=fix_prompt,
            agent_bin=agent_bin,
            agent_model=agent_model,
            timeout_s=timeout_s,
            label="fix",
            log_path=full_log,
        )

    print("\n" + sep)
    print(f"  ✗ TOOL-ENFORCED GATES: did NOT pass within {max_grade_rounds} round(s)")
    print(sep)
    return 1


def _run_agent(*, prompt: str, agent_bin: str, agent_model: str, timeout_s: int, label="agent", log_path: Path = None):
    """Run one agent. The SCREEN always stays clean — only a throttled
    `· <label> working…` heartbeat — while the COMPLETE agent stream (narration,
    tool calls, results) is appended to ``log_path`` (one consolidated file for
    the whole emit-e2e run). The structured grader report is rendered by the
    caller. This is how emit-e2e gets a clean screen + one full log without a
    regex filter (the agent's free-form narration can't be pattern-matched)."""
    cmd = [
        agent_bin,
        "-p",
        prompt,
        "--model",
        agent_model,
        "--dangerously-skip-permissions",
        "--add-dir",
        str(Path.cwd()),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    log_fh = None
    if log_path is not None:
        try:
            log_fh = open(log_path, "a", buffering=1, errors="ignore")
        except Exception:
            log_fh = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(f"  ✗ agent binary not found: {agent_bin!r}")
        if log_fh:
            log_fh.close()
        return 2, ""

    final_text = ""
    start = time.monotonic()
    last_hb = start
    tool_calls = 0
    HB_EVERY_S = 45
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_fh is not None:  # COMPLETE stream → one consolidated log file
                try:
                    log_fh.write(line)
                except Exception:
                    pass
            _rendered, final, _atext, n_tool = _render_stream_event(line)
            if final:
                final_text = final
            tool_calls += n_tool
            now = time.monotonic()  # CLEAN screen: heartbeat only, never the transcript
            if now - last_hb >= HB_EVERY_S:
                sys.stdout.write(f"  · {label} working… {int(now - start)}s, {tool_calls} tool calls\n")
                sys.stdout.flush()
                last_hb = now
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n  ✗ agent exceeded {timeout_s}s; killed")
        if log_fh:
            log_fh.close()
        return 1, final_text
    finally:
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass
    return (0 if rc == 0 else 1), final_text


def _render_stream_event(line: str):
    """Render one stream-json event to a screen line.

    Returns ``(rendered, final, assistant_text, n_tool_use)``: ``rendered`` is
    what to print under verbose (or ``None``), ``final`` is the agent's terminal
    ``result`` text, ``assistant_text`` is the raw text of an assistant turn
    (used to dedup the verbose final summary), and ``n_tool_use`` is how many
    tool calls this event carried (for the non-verbose progress heartbeat)."""
    line = line.rstrip("\n")
    if not line.strip():
        return None, None, None, 0
    try:
        ev = json.loads(line)
    except Exception:
        # Non-JSON lines (framework log spill) are noise on screen; the full
        # raw stream is in the log file. Show only under verbose.
        return (("  · " + line) if (_verbose() and line.strip()) else None), None, None, 0

    etype = ev.get("type")
    if etype == "system":
        # init / thinking_tokens / task_started / task_notification / task_updated
        # carry no signal for the watcher and arrive dozens of times — drop them.
        return None, None, None, 0

    if etype == "assistant":
        out = []
        text_parts = []
        n_tool = 0
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            t = c.get("type")
            if t == "text":
                txt = (c.get("text") or "").strip()
                if txt:
                    out.append("  " + txt.replace("\n", "\n  "))
                    text_parts.append(txt)
            elif t == "tool_use":
                n_tool += 1
                out.append("  → " + _fmt_tool(c.get("name", "?"), c.get("input", {}) or {}))
        return (
            ("\n".join(out) if out else None),
            None,
            ("\n".join(text_parts) if text_parts else None),
            n_tool,
        )

    if etype == "user":
        # Tool-result previews (`↳`) are the bulk of the on-screen clutter: file
        # headers, ttnn DEBUG dumps leaking through Read/Bash output, and the
        # agent's own `<tool_use_error>` retries. The preceding `→` action line
        # already says what the agent did, and the full result is in the log.
        # Keep these only under verbose.
        if not _verbose():
            return None, None, None, 0
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            if c.get("type") == "tool_result":
                content = c.get("content")
                txt = content if isinstance(content, str) else json.dumps(content)
                first = (txt or "").strip().splitlines()[0] if (txt or "").strip() else ""
                if first:
                    return "      ↳ " + first[:160], None, None, 0
        return None, None, None, 0

    if etype == "result":
        return None, ev.get("result") or "", None, 0

    return None, None, None, 0


def _fmt_tool(name: str, inp: dict) -> str:
    try:
        if name == "Bash":
            return "Bash: " + str(inp.get("command", ""))[:150]
        if name in ("Read", "Edit", "Write", "NotebookEdit"):
            return f"{name} {inp.get('file_path', inp.get('path', ''))}"
        if name in ("Grep", "Glob"):
            return f"{name} {inp.get('pattern', '')} {inp.get('path', '')}".rstrip()
        if name in ("Task", "Agent"):
            return f"{name}: {str(inp.get('description', inp.get('prompt', '')))[:120]}"
        return f"{name} {json.dumps(inp)[:120]}"
    except Exception:
        return name


def _build_fix_prompt(*, model_id: str, demo_dir: Path, pcc: float, grader_findings: str) -> str:
    return f"""The TOOL-ENFORCED end-to-end gates FAILED for `{model_id}` at {demo_dir}.
The verdict is computed by the tool itself (it runs tests/e2e on the device and
reads pass/fail) — you CANNOT pass by editing a report; you must make the gates
genuinely pass when the tool re-runs them.

Gate failures to fix:
{grader_findings}

Fix rules:
  - Failures here are WIRING/ASSEMBLY, not component math. The components are
    already graduated and PCC-verified in isolation — do NOT change stub math or
    re-run per-component PCC. Make every flagged module genuinely on the real
    compute path: fed by the previous TT stage's real output, its output flowing
    into the FINAL output. NO off-path side-runs, NO matched/reference tensor
    injected at a joint, NO counter bumped while the real compute bypasses the
    stub.
  - GENERATIVE heads (reference is `model.generate()`): reproduce generate()'s
    real chain and compare to it. Keep the gate fast by capping BOTH sides to the
    same small N (e.g. 40): pass `max_new_tokens=N` to `model.generate()` AND stop
    the TT decode loop at N, then compare the first-N sequence (+ per-step PCC).
    Do NOT run full-length generation (the gate times out). Do NOT cap only the
    TT side while HF runs full (lengths won't match → false fail, and HF stays
    slow).
  - Do NOT weaken any gate, lower a threshold, add pytest.skip/assert True, or
    relax no-waste. Keep input from the real HF processor/tokenizer and the
    golden from the real HF reference (`model.generate()`).
  - STRUCTURE/demo failures: emit-e2e's deliverable is a runnable demo. Ensure
    `demo/demo_<task>.py` entrypoints exist (each with `__main__` + argparse),
    plus `tt/` and `README.md`. The demo MUST call the SAME shared pipeline in
    `tt/` that the e2e test calls — do NOT give the demo its own copy of the
    wiring (it will drift and ship broken while the test stays green). Fix the
    demo by routing it through the test's pipeline, not by rewriting wiring.
  - Keep the demo package structure (demo/ + tt/ + tests/ + README) intact.
  - Re-run tests/e2e on the device yourself and confirm they pass before finishing.

Report what you changed and the device re-run result."""


def _build_grader_prompt(*, model_id: str, demo_dir: Path, pcc: float) -> str:
    return f"""You are the independent GRADER for the end-to-end TTNN pipeline of
`{model_id}` at {demo_dir}. You did NOT build this pipeline. Your job is NOT to
fix or improve it — it is to ADVERSARIALLY VERIFY it and try to find a hole.
Assume the builder may have gamed the gates; prove it didn't.

Do all of this with your own tools (Read/Bash), then report a verdict.

1. RE-RUN (independent execution). For each {demo_dir}/tests/e2e/test_*.py, run
   it FRESH on the device yourself:
     ./python_env/bin/python -m pytest <file> -s
   Record pass/fail and every FINAL_PCC printed. A pass you reproduce counts;
   a result you only read in a summary does NOT.

2. AUDIT THE TEST SOURCE (is the test honest?). Read each test file and CONFIRM:
   - the PCC is computed between the TT pipeline output and the HUGGINGFACE
     reference output — NOT TT-vs-TT, not against a constant or a trivially
     equal tensor;
   - the input is built from the real HF processor/tokenizer/feature_extractor
     (Source A), not random tensors;
   - the INVOKED counters wrap stubs that are actually ON the compute path
     (the wrapped object's output is really used downstream) — not a counter
     that is bumped while the real compute bypasses it;
   - there is NO `pytest.skip`, no `assert True`, no early return that dodges
     the PCC assertion;
   - every PCC assertion threshold is >= {pcc}.

3. STRUCTURE (standard demo layout). Confirm the emitted package exists and is real:
   {demo_dir}/demo/ (runnable per-Call entrypoints), {demo_dir}/tt/,
   {demo_dir}/tests/e2e/, and a README.md. Flag missing/placeholder pieces.

4. NO-WASTE. From {demo_dir}/bringup_status.json, take the GRADUATED set (NEW
   components with a `_stubs/<name>.py.last_good_native` OR `.py.last_good_sharded`
   snapshot — bring-up is single-phase, so a TP>1 mesh run graduates shardable
   modules DIRECTLY sharded). Confirm the UNION of INVOKED stubs across all task
   heads' runs == that graduated set. Name any graduated module never invoked.

WRITE the structured machine-readable report to {demo_dir}/grader_report.json
so the fix agent gets precise targets. Use EXACTLY this schema:
  {{
    "verdict": "PASS" | "FAIL",
    "calls": [
      {{"call": "<id>", "rerun": "pass|fail", "final_pcc": [<num>, ...],
        "source_audit": "clean|ISSUE"}}
    ],
    "structure": {{"ok": true|false, "detail": "<...>"}},
    "no_waste": {{"ok": true|false, "graduated_total": <N>, "on_path": <N>,
                  "names_present": <N>, "missing": [<name>, ...]}},
    "holes": [
      {{"id": "<short-slug>",
        "call": "<id>",
        "modules": ["<graduated module name>", ...],
        "file": "<path relative to {demo_dir}>",
        "lines": "<start-end>",
        "mechanism": "<exactly how the gate is gamed / what is wrong>",
        "fix_hint": "<concrete action that would make it genuinely pass>",
        "severity": "blocker" | "minor"}}
    ]
  }}
A clean call contributes no holes. Every FAIL reason MUST appear as a hole with
file+lines+mechanism+fix_hint filled in (no vague entries).

Then ALSO print this verdict block to stdout (one row per Call):

  GRADER_REPORT
  | Call | re-run | final_pcc | source-audit | holes_found |
  | ...  | pass/fail | <num> | clean/ISSUE | <what, or none> |
  STRUCTURE: pass/fail (<detail>)
  NO_WASTE: pass/fail (<N>/<total> graduated invoked; missing: <list>)
  GRADER_VERDICT: PASS    <-- only if EVERY call re-ran-pass + source-audit clean
                              + STRUCTURE pass + NO_WASTE pass. Otherwise:
  GRADER_VERDICT: FAIL

Do not write or edit any pipeline/stub/test files — you are read-only except for
{demo_dir}/grader_report.json (the structured report above) and an optional
{demo_dir}/grader_report.md prose summary. Be skeptical; if anything is
ambiguous, it is a FAIL with a hole describing the ambiguity."""


def _mesh_chip_count(mesh_arg) -> int:
    if not mesh_arg:
        return 1
    try:
        prod = 1
        for tok in str(mesh_arg).lower().split("x"):
            prod *= int(tok)
        return max(prod, 1)
    except Exception:
        return 1


def _planned_parallelism(model_id: str, args):
    chips = _mesh_chip_count(getattr(args, "mesh", None))
    if chips <= 1:
        return None
    try:
        from ..cli import evaluate_kernels, probe_model
        from ..parallelism import select_parallelism

        probe = probe_model(model_id)
        if not getattr(probe, "raw_config", None):
            return None
        kr = evaluate_kernels(probe.raw_config, tp_grid=None)
        pc = select_parallelism(chips, kr)
    except Exception:
        return None
    return pc


def _parallelism_prompt_block(pc) -> str:
    if pc is None or pc.chips <= 1:
        return ""
    return f"""

================ CHIP PLACEMENT — {pc.chips}-CHIP MESH (TP={pc.tp} x DP={pc.dp}) ================
The tool has selected this parallelism split for `{pc.chips}` chips by checking per-TP kernel
viability (largest kernel-viable TP degree that divides the mesh; the remaining chips become
data-parallel replicas). Place the pipeline on the mesh accordingly:

  - BEFORE opening the mesh, enable the inter-chip fabric: `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)`.
    Without this, any CCL (all_gather / all_reduce) raises `TT_FATAL ... fabric_context_ != nullptr`.
    tt-metal AUTO-DISCOVERS the cluster topology, so do NOT set TT_MESH_GRAPH_DESC_PATH for any mesh size.
  - Open a mesh device of {pc.chips} chips via `ttnn.open_mesh_device(ttnn.MeshShape({pc.dp}, {pc.tp}))`
    (rows = DP={pc.dp}, cols = TP={pc.tp}); close it at the end. If only a single device is available
    at runtime, fall back to it and note that in the run output.
  - DATA-PARALLEL axis (DP={pc.dp}): replicate the model across the {pc.dp} replica rows using
    `mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)` when moving tensors on-device, and compose
    back with the matching composer. DP replicates weights (no per-chip memory saving) and runs
    independent data copies.
  - TENSOR-PARALLEL axis (TP={pc.tp}): where the reused ttnn modules already support column/row
    sharding, shard the sharded weights along the TP axis with `ttnn.ShardTensorToMesh(mesh_device, dim=<shard_dim>)`
    on the {pc.tp} TP columns; keep embeddings / norms / lm_head replicated. If a module does not
    expose a shard dim, keep it replicated rather than guessing a split.
  - The e2e PCC gate is unchanged: parity is still measured against the same HF golden. Placing the
    pipeline on more chips must NOT change the numerical result — only where it runs.
"""


def _build_agent_prompt(*, model_id: str, demo_dir: Path, pcc: float, parallel_note: str = "") -> str:
    return f"""You are bringing up a REAL end-to-end TTNN pipeline for the model
`{model_id}`. Work in this repository with your tools (Read/Edit/Write/Bash).

There are exactly TWO information sources. Use ONLY these — do NOT read any
sibling model under models/demos/<other-model>/:

  SOURCE A — HuggingFace hub for `{model_id}`:
    config.json, tokenizer/processor/feature_extractor, the AutoModel
    registry (which task heads this model supports), and the reference
    model + model.generate() as the golden output for parity.

  SOURCE B — the bring-up tool output for this model at:
    {demo_dir}
      - bringup_status.json   (components + status; GRADUATED = NEW with a
        `_stubs/<name>.py.last_good_native` OR `.py.last_good_sharded` snapshot.
        Bring-up is single-phase: TP=1 graduates a native single-device body,
        TP>1 graduates the shardable modules DIRECTLY sharded (the .last_good_sharded
        body already does ShardTensorToMesh + all_reduce). The LIVE `_stubs/<name>.py`
        IS the graduated body — compose it as-is. REUSE entries have no stub and are
        NOT graduated work products)
      - _stubs/*.py           (the graduated TTNN stubs; each exposes
                               build(device, torch_module) and a callable)
      - _captured/<name>/{{args,kwargs,output}}.pt   (HF golden tensors)
      - tests/pcc/            (per-component PCC tests)

================ COMMAND 1 — ACT AS PLANNER ================
Based on Group A and Group B information ONLY, act as a planner and create a
sketch plan (mental model) that produces a task_heads JSON with: what "pass"
means, which graduated stubs go where, the validation metric, behavioral
proof, and a self-validation plan. Make sure the pipeline uses ALL graduated
modules from Source B and does not leave any graduated module out. Correctly
VERIFY that the graduated modules are listed correctly so none are wasted.
Write the plan to {demo_dir}/e2e_plan.json.

================ COMMAND 2 — ORCHESTRATE THE BUILD ================
Based on that plan and only information from the plan, fire parallel agents
working on Call 1, Call 2, … Call N (the task heads) separately if there is no
dependency between them; if two calls share a graduated module, use only ONE
agent for them. Iterate using Gate 1, Gate 2, and Gate 3 until you have an
end-to-end pipeline ready:

  Gate 1 — every routed graduated stub is still real ttnn (not torch fallback);
           a sharded (TP>1) body counts as native — do NOT rewrite it to replication.
  Gate 2 — every graduated module is actually INVOKED in the pipeline run
           (no graduated module left out — this is critical).
  Gate 3 — the pipeline's FINAL output PCC vs the HF golden (Source A) is
           >= {pcc}.

CRITICAL REQUIREMENTS:
  - The pipeline must NOT be a smoke test. It must be a REAL pipeline that
    takes input exactly as collected from Sources A+B and emits output exactly
    as defined in Sources A+B (e.g. audio->text, text->text, text->audio).
    Input is constructed via the HF processor/tokenizer/feature_extractor;
    output is the real task output, compared to the HF reference (Source A).
  - It must chain the graduated stubs into the actual forward pass and produce
    real task output — not just pass tensors around. Each stage must be fed the
    previous TT stage's real output; NEVER inject a matched/reference tensor at
    a joint (that hides wiring bugs the e2e test exists to catch).
  - ALL graduated modules/components must be used in the pipeline.
  - The end-to-end pipeline must pass PCC >= {pcc}.
  - ALWAYS print the achieved end-to-end PCC on EVERY run, pass OR fail — e.g.
    `print(f"e2e PCC={{achieved_pcc}}")` on its own line immediately BEFORE the
    final assert — so the measured number is visible in the test output
    regardless of the verdict (not only surfaced in the assert message on fail).
  - GENERATIVE heads (reference is `model.generate()`): reproduce generate()'s
    real chain and compare the TT-generated output to it. To keep the on-device
    gate fast, CAP BOTH SIDES to the same small horizon N (e.g. 40): pass
    `max_new_tokens=N` to `model.generate()` AND stop the TT decode loop at N,
    then compare the first-N sequence (+ per-step PCC). Do NOT run full-length
    generation (too slow — the gate times out). Do NOT cap only the TT side
    while HF runs full length (lengths won't match → false fail, and HF is still
    slow). Both sides capped to the same N → fast, faithful, no false mismatch.

STRUCTURE — emit a complete, runnable package in the standard demo layout
(the same demo/ + tt/ + tests/ package style used by demos under models/demos/).
For ANY model, emit a complete, runnable package — not a lone test file:
  {demo_dir}/
    demo/         per-task runnable demo entrypoint(s) (one per Call) that load
                  real input, run the chained TTNN pipeline, emit real output.
                  EACH must have a `__main__` + argparse and be runnable as
                  `python -m ...demo.demo_<task>`.
    tt/           the ONE shared chained pipeline (the real forward pass over the
                  graduated stubs) that BOTH demo/ and tests/e2e/ import and call.
    tests/e2e/    the e2e pipeline test(s): real input -> chained stubs ->
                  real output, asserting Gate 1/2/3 (all stubs INVOKED + final
                  PCC >= {pcc} vs HF golden).
    README.md     what each Call does, how to run it, the PCC numbers.

  CRITICAL — DEMO AND TEST MUST SHARE ONE PIPELINE: the chained forward pass (the
  exact wiring of the graduated stubs) lives in `tt/` as a single function, and
  BOTH the demo entrypoint AND the e2e test import and call it. Do NOT write two
  separate copies of the wiring — if the demo has its own copy it WILL drift from
  the test and ship a broken pipeline while the test stays green. A passing test
  must GUARANTEE a working demo because they run identical code. emit-e2e's
  deliverable is a runnable demo; a green test with no/working demo is NOT done.
Match the conventions of existing demos under models/demos/ rather than
inventing a new layout. Keep iterating (fix the stub/wiring, re-run on the TT device) until the
gates pass. Use `./python_env/bin/python -m pytest <file> -s` to run on device.
Report a final summary: which calls are READY, the FINAL_PCC per call, and
confirm all graduated modules were invoked.
{parallel_note}"""


def _resolve_demo_dir(args) -> Path:
    raw = getattr(args, "output", None)
    if raw:
        p = Path(raw)
        return p.parent if p.suffix == ".py" else p
    slug = args.model_id.split("/")[-1].replace("-", "_").lower()
    demos_root = Path.cwd() / "models" / "demos"
    if demos_root.is_dir():
        for cand in demos_root.rglob(slug):
            if cand.is_dir() and (cand / "bringup_status.json").is_file():
                return cand
    return Path(f"models/demos/{slug}")
