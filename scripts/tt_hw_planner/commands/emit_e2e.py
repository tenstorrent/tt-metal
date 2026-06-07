# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e — LLM-driven end-to-end pipeline builder (build agent + grader agent)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def cmd_emit_e2e(args) -> int:
    model_id = args.model_id
    demo_dir = _resolve_demo_dir(args)
    pcc = float(getattr(args, "pcc_target", 0.9) or 0.9)
    agent_model = getattr(args, "model", None) or "opus"
    agent_bin = getattr(args, "agent_bin", "claude") or "claude"
    timeout_s = int(getattr(args, "agent_timeout_s", 0) or 0) or 14400
    skip_grade = bool(getattr(args, "no_grade", False))

    sep = "=" * 78
    print(sep)
    print(f"  EMIT-E2E (LLM agent)  {model_id}")
    print(f"  demo_dir={demo_dir}  pcc>={pcc}  model={agent_model}")
    print(sep)

    print("\n  ===== PHASE 1+2: BUILDER agent (plan → build → iterate) =====\n")
    build_prompt = _build_agent_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
    rc_build, _ = _run_agent(
        prompt=build_prompt,
        agent_bin=agent_bin,
        agent_model=agent_model,
        timeout_s=timeout_s,
    )
    if rc_build != 0:
        print(f"\n  ✗ builder agent exited rc={rc_build}; skipping grade")
        return 1

    if skip_grade:
        print("\n  (--no-grade) skipping independent grader phase.")
        return 0

    print("\n  ===== PHASE 3: GRADER agent (independent adversarial verify) =====\n")
    grade_prompt = _build_grader_prompt(model_id=model_id, demo_dir=demo_dir, pcc=pcc)
    rc_grade, grade_final = _run_agent(
        prompt=grade_prompt,
        agent_bin=agent_bin,
        agent_model=agent_model,
        timeout_s=timeout_s,
    )

    verdict_pass = rc_grade == 0 and "GRADER_VERDICT: PASS" in (grade_final or "")
    print("\n" + sep)
    if verdict_pass:
        print("  ✓ INDEPENDENT GRADER: PASS — pipeline verified by a separate agent")
    else:
        print("  ✗ INDEPENDENT GRADER: did NOT confirm PASS — see grader report above")
    print(sep)
    return 0 if verdict_pass else 1


def _run_agent(*, prompt: str, agent_bin: str, agent_model: str, timeout_s: int):
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
        return 2, ""

    final_text = ""
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            rendered, final = _render_stream_event(line)
            if final:
                final_text = final
            if rendered:
                sys.stdout.write(rendered + "\n")
                sys.stdout.flush()
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n  ✗ agent exceeded {timeout_s}s; killed")
        return 1, final_text

    if final_text:
        print("\n  ── agent final summary ──")
        for ln in final_text.splitlines():
            print("  " + ln)
    return (0 if rc == 0 else 1), final_text


def _render_stream_event(line: str):
    line = line.rstrip("\n")
    if not line.strip():
        return None, None
    try:
        ev = json.loads(line)
    except Exception:
        return ("  · " + line) if line.strip() else None, None

    etype = ev.get("type")
    if etype == "system":
        sub = ev.get("subtype", "")
        return (f"  [system] {sub}" if sub else None), None

    if etype == "assistant":
        out = []
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            t = c.get("type")
            if t == "text":
                txt = (c.get("text") or "").strip()
                if txt:
                    out.append("  " + txt.replace("\n", "\n  "))
            elif t == "tool_use":
                out.append("  → " + _fmt_tool(c.get("name", "?"), c.get("input", {}) or {}))
        return ("\n".join(out) if out else None), None

    if etype == "user":
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            if c.get("type") == "tool_result":
                content = c.get("content")
                txt = content if isinstance(content, str) else json.dumps(content)
                first = (txt or "").strip().splitlines()[0] if (txt or "").strip() else ""
                if first:
                    return "      ↳ " + first[:160], None
        return None, None

    if etype == "result":
        return None, ev.get("result") or ""

    return None, None


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

Then output a verdict block, EXACTLY in this form (one row per Call):

  GRADER_REPORT
  | Call | re-run | final_pcc | source-audit | holes_found |
  | ...  | pass/fail | <num> | clean/ISSUE | <what, or none> |
  STRUCTURE: pass/fail (<detail>)
  NO_WASTE: pass/fail (<N>/<total> graduated invoked; missing: <list>)
  GRADER_VERDICT: PASS    <-- only if EVERY call re-ran-pass + source-audit clean
                              + STRUCTURE pass + NO_WASTE pass. Otherwise:
  GRADER_VERDICT: FAIL

Do not write or edit any pipeline/stub files — you are read-only except for a
short {demo_dir}/grader_report.md you may write with your findings. Be skeptical;
if anything is ambiguous, it is a FAIL."""


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
