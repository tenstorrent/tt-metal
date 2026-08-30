"""PCC measurement for GATE_PCC (PLAN 8.6) — single-stage e2e.

parse_pcc() is deterministic and unit-tested. run_pcc() runs the model's
end-to-end PCC test on hardware and is the injectable default (ctx.deps["pcc_runner"]);
it is exercised live, not in unit tests. TBD(pcc-parse): the regex assumes the
test prints a "PCC: <float>" style number — refine per the real test's output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from . import gitio, probes
from .layer_depth import set_depth as _set_depth

_DEPTH_GUARD = "models.experimental.perf_automation.agent.depth_guard_plugin"

_PCC_RE = re.compile(r"(?i)pcc[^\n]*?[:=]\s*(-?\d+\.\d+)")


def parse_pcc(text: str):
    """WORST 'pcc ... <float>' in the test output, or None.

    Was the LAST occurrence, which banked the wrong number in two real shapes: many tt-metal
    tests print the THRESHOLD as `pcc: 0.99` (if that line came last it was recorded as the
    measured value), and a per-layer sweep prints many values (only the last was judged). For a
    correctness gate the worst observed PCC is the one that has to clear the threshold.
    """
    matches = _PCC_RE.findall(text or "")
    return min(float(m) for m in matches) if matches else None


def _require_pcc() -> bool:
    """Must the correctness gate produce an actual PCC number? Default yes.

    Model-agnostic: the tool does not care HOW the number is produced (logits for a text LLM,
    per-head for a VLM, mel/waveform for TTS) -- only that one exists and clears a floor.
    """
    return (os.environ.get("PERF_MCP_REQUIRE_PCC", "1") or "1") != "0"


def _operator_pcc_floor() -> float:
    """A floor the operator can impose on top of whatever the test declares (stricter wins)."""
    try:
        return float(os.environ.get("PERF_MCP_PCC_MIN", "0") or "0")
    except ValueError:
        return 0.0


def _verdict_from_output(out: str, threshold: float) -> dict:
    """Correctness verdict for a captured pytest run. Split out of run_pcc so the gate's own
    logic is testable without a device -- these branches decide whether an edit is kept."""
    pcc = parse_pcc(out)

    # A SKIPPED e2e test verified NOTHING -- never accept it as correct just because a stale
    # "pcc=..." string happened to be in the log (the seamless SKIP-mislabel pattern).
    _skipped = re.search(r"\b[1-9]\d*\s+skipped\b", out, re.IGNORECASE)
    if _skipped and not re.search(r"\b[1-9]\d*\s+passed\b", out):
        return {"status": "crash", "error": "e2e PCC test SKIPPED (correctness NOT verified): " + _useful_tail(out)}
    # PARTIAL skip: the old guard required NO `passed`, so a file where a trivial test passes and
    # the real e2e case SKIPS printed `1 passed, 1 skipped` and sailed through -- reopening the
    # very SKIP-mislabel class the guard exists for. A skip is only acceptable if some case
    # actually produced a PCC number.
    if _skipped and pcc is None:
        return {
            "status": "crash",
            "error": (
                "e2e PCC test partially SKIPPED and no PCC value was produced (correctness NOT "
                "verified): " + _useful_tail(out)
            ),
        }

    if (
        pcc is None
        and re.search(r"\b[1-9]\d*\s+passed\b", out)
        and not re.search(r"\b[1-9]\d*\s+(failed|errors?)\b", out, re.IGNORECASE)
    ):
        if _require_pcc():
            return {
                "status": "pcc_low",
                "pcc": None,
                "pcc_verified": False,
                "error": (
                    "the correctness gate PASSED but produced NO PCC value, so numerical correctness "
                    "was never checked. A pass on a proxy (e.g. top-1 token accuracy) does NOT bound "
                    "PCC: argmax rarely flips for confident tokens, so a model whose PCC has collapsed "
                    "can still match most tokens. Refusing to bank a win on an unverified gate. Set "
                    "PERF_MCP_REQUIRE_PCC=0 to accept a proxy gate."
                ),
            }
        return {
            "status": "ok",
            "pcc": None,
            "pcc_verified": False,
            "note": "gate passed but NO PCC value was captured -- correctness is UNVERIFIED, not confirmed",
        }

    if pcc is None:
        return {"status": "crash", "error": _useful_tail(out)}

    # PCC IS the correctness signal for a perf edit. A non-zero pytest EXIT with PCC>=threshold
    # is NOT an edit-induced regression: the e2e gate also enforces BRING-UP checks (Gate-2
    # "graduated modules invoked") and the process prints benign nanobind teardown leaks at
    # interpreter shutdown -- BOTH set a non-zero exit while the math is perfect, and BOTH fail
    # on the UNEDITED baseline too (verified: clean nemotron e2e exits 1 on Gate-2 with PCC
    # 0.999). Gating on the raw return code here rejected every edit. So gate on PCC: a genuine
    # device crash already yields pcc=None above; below-threshold PCC is pcc_low (repairable).
    effective = max(float(threshold or 0.0), _operator_pcc_floor())
    return (
        {"status": "ok", "pcc": pcc, "pcc_verified": True, "threshold": effective}
        if pcc >= effective
        else {"status": "pcc_low", "pcc": pcc, "pcc_verified": True, "threshold": effective}
    )


def run_pcc(ctx) -> dict:
    """Run the e2e PCC test, parse the measured PCC, compare the manifest threshold.

    Returns {status: ok|pcc_low|crash, pcc?, error?}. A parsed number below
    threshold is pcc_low (expected pytest non-zero exit); an unparseable result
    or an exception is crash.
    """
    entry = ctx.manifest["pathmap"]["pcc"]["end_to_end"]
    file_part, sep, fn = str(entry["path"]).partition("::")
    repo = gitio.repo_root(ctx.model_root())
    resolved = next(
        (b / file_part for b in (Path(ctx.model_root()), Path(repo)) if (b / file_part).is_file()),
        Path(ctx.model_root()) / file_part,
    )
    test = str(resolved) + (sep + fn)
    threshold = entry["threshold"]
    env = dict(os.environ)
    # FULL DEPTH for correctness, expressed by REMOVING the cap rather than by a sentinel: "0"
    # arrives as a truthy string and was read by model builders as "build zero layers", which PCC'd
    # a model that had done no work. See agent/layer_depth.py.
    _set_depth(env, None)
    from .mesh_descriptor import apply_scope

    apply_scope(env, ctx.manifest.get("config", {}))
    try:
        r = subprocess.run(
            # -p depth_guard: correctness must run at FULL depth; see agent/depth_guard_plugin.py
            [sys.executable, "-m", "pytest", "-p", _DEPTH_GUARD, "-o", "addopts=", "-o", "timeout=0", test, "-sv"],
            cwd=str(gitio.repo_root(ctx.model_root())),
            env=env,
            capture_output=True,
            text=True,
            timeout=probes.adaptive_backstop(3600),
        )
    except Exception as exc:  # timeout, OS error, etc.
        return {"status": "crash", "error": str(exc)}
    out = (r.stdout or "") + (r.stderr or "")
    return _verdict_from_output(out, threshold)


# Lines that pollute the crash excerpt: nanobind dumps ~hundreds of "leaked ..." lines at
# interpreter shutdown, which otherwise BURY the real error in the [-N:] tail fed to repair.
_TEARDOWN_NOISE = re.compile(r"nanobind|leaked (type|function)|reference counting|skipped remainder", re.IGNORECASE)


def _useful_tail(out: str, n: int = 2000) -> str:
    """Last n chars of the output with teardown noise removed, so the real error survives."""
    kept = [ln for ln in (out or "").splitlines() if not _TEARDOWN_NOISE.search(ln)]
    return "\n".join(kept).strip()[-n:]
