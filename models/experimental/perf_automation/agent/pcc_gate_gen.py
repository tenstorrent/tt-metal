"""Generate an HF-referenced e2e PCC gate when the model is in HF and no usable gate was supplied.

Mirrors the perf-test builder (generate -> run -> judge -> regenerate, bounded by a stall limit),
but the acceptance test is different in a way that matters. A perf test is accepted when it RUNS and
emits a trace marker. A correctness gate accepted on "it ran and passed" is worthless, because a gate
that always passes also passes: llama3_1_8b_p150 shipped a file named test_pcc.py that asserted top-1
TOKEN ACCURACY, ran green forever, and could not have caught a bf4_b lever sitting at PCC 0.513 with
100% token match.

So a generated gate must clear BOTH:

  1. PASSES on the unedited model, printing `PCC: <float>` above its own declared threshold
     -> proves it produces a number the optimize loop can read.
  2. FAILS on a deliberately PERTURBED model
     -> proves the number MEANS something. This is the check that would have rejected the
        token-accuracy gate, and it is the reason this module exists.

The gate itself is written by an agent because the comparison is model-specific (text logits for an
LLM, per-head for a VLM, mel/waveform for TTS). The tool supplies the CONTRACT and the two-part
judgement; it never encodes how any particular model computes its reference.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .layer_depth import set_depth as _set_depth

_DEPTH_GUARD = "models.experimental.perf_automation.agent.depth_guard_plugin"

STALL_LIMIT = int(os.environ.get("PERF_MCP_PCC_GEN_STALL_LIMIT", "3") or "3")

# Emitted by the generated gate; the optimize loop parses exactly this.
PCC_LINE = re.compile(r"(?i)\bpcc\s*[:=]\s*(-?\d+\.\d+)")

VERDICT_OK = "ok"
VERDICT_NO_NUMBER = "no_pcc_number"
VERDICT_CANNOT_FAIL = "cannot_fail"
VERDICT_BROKEN = "broken"


def _contract(model_dir: Path, model_id: str, out_rel: str, threshold: float) -> str:
    """The instructions handed to the authoring agent. Contract only -- no per-model knowledge."""
    return (
        f"Write a pytest END-TO-END CORRECTNESS GATE at `{out_rel}` for the TTNN model in "
        f"`{model_dir}` (HuggingFace id: {model_id}).\n\n"
        "CONTRACT (the optimize loop depends on every point):\n"
        f"1. Compare the TT model's RAW OUTPUT TENSOR against the HuggingFace reference for the same "
        f"fixed input, and print exactly one line `PCC: <float>`.\n"
        f"2. Declare the floor as a module-level constant and assert against it, e.g. "
        f"`PCC_MIN = {threshold}` then `assert pcc >= PCC_MIN`. The tool reads this number out of "
        f"the source, so it must be a literal.\n"
        "3. Capture the tensor BEFORE any argmax/sampling. Argmax discards magnitude, so a gate "
        "built on token ids stays ~100% correct while the underlying values collapse -- that is the "
        "exact failure this gate replaces.\n"
        "4. TEACHER-FORCE both sides from the reference's own outputs for multi-step models, so a "
        "single divergence does not leave the two sides comparing different contexts.\n"
        "5. Cache the HF reference OUTSIDE the repo (e.g. under ~/.cache/) and reuse it. This gate "
        "runs after EVERY edit, so recomputing the reference each time is not acceptable.\n"
        "6. Build the model at FULL DEPTH. If the builder reads a layer-cap env var (TT_PERF_LAYERS "
        "or similar), assert it is unset/0 rather than silently building a truncated model.\n"
        "7. It must import cleanly with no device (`pytest --collect-only`).\n\n"
        "Return ONLY the file content."
    )


def _run_gate(node: str, repo_root: Path, env=None, timeout=None) -> tuple:
    """Run the gate; return (rc, output, parsed_pcc_or_None)."""
    e = dict(os.environ)
    e.update(env or {})
    _set_depth(e, None)  # correctness always runs full depth (cap REMOVED, never sent as 0)
    try:
        r = subprocess.run(
            # -p depth_guard: correctness must run at FULL depth; see agent/depth_guard_plugin.py
            [sys.executable, "-m", "pytest", "-p", _DEPTH_GUARD, "-o", "addopts=", "-o", "timeout=0", node, "-sv"],
            cwd=str(repo_root),
            env=e,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return None, "gate run failed: %s" % (str(exc)[-300:],), None
    out = (r.stdout or "") + (r.stderr or "")
    hits = PCC_LINE.findall(out)
    return r.returncode, out, (min(float(h) for h in hits) if hits else None)


def judge(clean_rc, clean_pcc, perturbed_pcc, threshold: float) -> tuple:
    """Two-part verdict. Returns (verdict, why).

    Split out so the acceptance rule is testable without a device -- it is the part that decides
    whether a generated gate is trustworthy, and it is where the previous gate would have been
    rejected.
    """
    if clean_pcc is None:
        return VERDICT_NO_NUMBER, (
            "the gate produced no `PCC: <float>` line on the unedited model, so the optimize loop "
            "has nothing to read and no edit can ever be judged"
        )
    if clean_rc not in (0, None) or clean_pcc < threshold:
        return VERDICT_BROKEN, (
            "the gate FAILS on the UNEDITED model (pcc=%s, floor=%s): it is measuring the wrong "
            "thing, or comparing misaligned tensors" % (clean_pcc, threshold)
        )
    if perturbed_pcc is None or perturbed_pcc >= threshold:
        return VERDICT_CANNOT_FAIL, (
            "the gate still reports pcc=%s on a DELIBERATELY PERTURBED model, so it cannot detect "
            "damage. A gate that cannot fail is not a correctness gate -- this is exactly how a "
            "token-accuracy check passed as a PCC gate" % (perturbed_pcc,)
        )
    return VERDICT_OK, "passes clean (pcc=%s) and fails perturbed (pcc=%s)" % (clean_pcc, perturbed_pcc)


def generate_pcc_gate(model_dir, model_id, repo_root, runner, threshold: float = 0.95, perturb_env=None):
    """Author + validate an HF-referenced gate. Returns the pytest node id, or None.

    `runner` (prompt -> file content) is the authoring agent, injected so this is testable.
    `perturb_env` is the env that degrades the model for check 2 (e.g. forcing a low-precision
    weight dtype); without one, check 2 cannot run and the gate is NOT accepted.
    """
    model_dir = Path(model_dir)
    out_rel = "tests/e2e/test_pcc_hf_generated.py"
    out_abs = model_dir / out_rel
    node = "%s::%s" % (
        (out_abs.relative_to(repo_root) if out_abs.is_relative_to(repo_root) else out_abs),
        "test_e2e_pcc",
    )

    feedback = ""
    for attempt in range(1, STALL_LIMIT + 1):
        content = runner(_contract(model_dir, model_id, out_rel, threshold) + feedback)
        if not content or not content.strip():
            feedback = "\n\nPREVIOUS ATTEMPT returned nothing. Return the file content only."
            continue
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        out_abs.write_text(content)

        clean_rc, clean_out, clean_pcc = _run_gate(node, repo_root)
        pert_pcc = None
        if perturb_env:
            _, _, pert_pcc = _run_gate(node, repo_root, env=perturb_env)

        verdict, why = judge(clean_rc, clean_pcc, pert_pcc, threshold)
        if verdict == VERDICT_OK:
            print("      ✔ generated PCC gate accepted: %s" % why, file=sys.stderr, flush=True)
            return node
        print(
            "      · PCC-gate regen %d/%d: %s — %s" % (attempt, STALL_LIMIT, verdict, why),
            file=sys.stderr,
            flush=True,
        )
        feedback = "\n\nPREVIOUS ATTEMPT REJECTED (%s): %s\nOutput tail:\n%s" % (verdict, why, clean_out[-1500:])

    try:
        out_abs.unlink()
    except OSError:
        pass
    return None
