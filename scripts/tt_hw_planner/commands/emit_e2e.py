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

    print("\n  ===== PHASE 1+2: BUILDER agent (plan → build → iterate) =====\n")
    build_prompt = _build_agent_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
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

    if skip_grade:
        print("\n  (--no-grade) skipping independent grader phase.\n")
        # No grader report to render; show a clean (markdown-stripped) build summary.
        if (build_final or "").strip():
            print(_md_to_terminal(build_final))
        _render_compute_split(model_id)
        return 0

    grade_prompt = _build_grader_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
    for rnd in range(1, max_grade_rounds + 1):
        print(f"\n  ===== PHASE 3: GRADER agent (round {rnd}/{max_grade_rounds}) =====\n")
        rc_grade, grade_final = _run_agent(
            prompt=grade_prompt,
            agent_bin=agent_bin,
            agent_model=agent_model,
            timeout_s=timeout_s,
            label="grader",
            log_path=full_log,
        )
        # Clean, professional terminal summary: render the structured
        # grader_report.json; fall back to a markdown-stripped prose summary.
        if not _render_grader_report(demo_dir) and (grade_final or "").strip():
            print("\n" + _md_to_terminal(grade_final))
        _render_compute_split(model_id)

        if rc_grade == 0 and "GRADER_VERDICT: PASS" in (grade_final or ""):
            print("\n" + sep)
            print(f"  ✓ INDEPENDENT GRADER: PASS (round {rnd}) — verified by a separate agent")
            print(sep)
            return 0
        if rnd == max_grade_rounds:
            break
        print(f"\n  ===== FIX agent (round {rnd}/{max_grade_rounds - 1}) — addressing grader findings =====\n")
        fix_prompt = _build_fix_prompt(
            model_id=model_id,
            demo_dir=demo_dir,
            pcc=pcc,
            grader_findings=grade_final or "",
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
    print(f"  ✗ INDEPENDENT GRADER: did NOT pass within {max_grade_rounds} round(s) — see grader report")
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
    return f"""The independent GRADER FAILED the end-to-end TTNN pipeline of
`{model_id}` at {demo_dir}. Your job is to fix EXACTLY the holes it found —
nothing else — and leave a result that an adversarial grader will pass.

FIRST read {demo_dir}/grader_report.json — it lists every hole as a structured
record: {{id, call, modules, file, lines, mechanism, fix_hint, severity}}. Work
through the `holes` array and close EACH one at its file:lines using its
fix_hint. (grader_report.md and the verdict text below are supporting context.)

Grader verdict text:
{grader_findings}

Fix rules:
  - Make every flagged module GENUINELY on the real compute path: its real
    input must come from the actual pipeline (the previous stage's output),
    and its output must flow downstream into the FINAL output that the PCC
    asserts on. NO off-path side-runs, NO random/synthetic inputs whose output
    is discarded, NO counter that is bumped while the real compute bypasses
    the stub. If a monolithic top-level stub inlines work instead of delegating
    to its graduated children, either route the pipeline THROUGH the children
    so they are the real compute path, or replace the inlined section with real
    calls to those child stubs whose outputs are used.
  - Do NOT weaken any gate, lower any PCC threshold, add skips, or relax the
    no-waste requirement. Keep input from the real HF processor/tokenizer and
    the golden from the real HF reference.
  - Keep the yito package structure intact.
  - Re-run the affected tests/e2e on the TT device yourself and confirm they
    still pass with PCC >= {pcc} AND the flagged modules now fire on-path.

Edit only what is needed to close the grader's holes. Report what you changed
and the device re-run result."""


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

3. STRUCTURE (yito layout). Confirm the emitted package exists and is real:
   {demo_dir}/demo/ (runnable per-Call entrypoints), {demo_dir}/tt/,
   {demo_dir}/tests/e2e/, and a README.md. Flag missing/placeholder pieces.

4. NO-WASTE. From {demo_dir}/bringup_status.json, take the GRADUATED set (NEW
   components with a `_stubs/<name>.py.last_good_native` snapshot). Confirm the
   UNION of INVOKED stubs across all task heads' runs == that graduated set.
   Name any graduated module that is never invoked.

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


def _build_agent_prompt(*, model_id: str, demo_dir: Path, pcc: float) -> str:
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
        `_stubs/<name>.py.last_good_native` snapshot and a native ttnn body;
        REUSE entries have no stub and are NOT graduated work products)
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

  Gate 1 — every routed graduated stub is still native (not torch fallback).
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
    real task output — not just pass tensors around.
  - ALL graduated modules/components must be used in the pipeline.
  - The end-to-end pipeline must pass PCC >= {pcc}.

STRUCTURE — follow the "yito" demo layout (the same package style used by the
hand-authored demos under models/demos/, and the existing demo/ files in this
model's dir). For ANY model, emit a complete, runnable package — not a lone
test file:
  {demo_dir}/
    demo/         per-task runnable demo entrypoint(s) (one per Call) that load
                  real input, run the chained TTNN pipeline, emit real output.
    tt/           thin re-exports / wiring of the graduated stubs used.
    tests/e2e/    the e2e pipeline test(s): real input -> chained stubs ->
                  real output, asserting Gate 1/2/3 (all stubs INVOKED + final
                  PCC >= {pcc} vs HF golden).
    README.md     what each Call does, how to run it, the PCC numbers.
Match the conventions of the existing yito demos rather than inventing a new
layout. Keep iterating (fix the stub/wiring, re-run on the TT device) until the
gates pass. Use `./python_env/bin/python -m pytest <file> -s` to run on device.
Report a final summary: which calls are READY, the FINAL_PCC per call, and
confirm all graduated modules were invoked.
"""


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
