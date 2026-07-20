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
from pathlib import Path

from . import gitio, probes

_PCC_RE = re.compile(r"(?i)pcc[^\n]*?[:=]\s*(-?\d+\.\d+)")


def parse_pcc(text: str):
    """Last 'pcc ... <float>' occurrence in the test output, or None."""
    matches = _PCC_RE.findall(text or "")
    return float(matches[-1]) if matches else None


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
    vd = ctx.manifest.get("config", {}).get("visible_devices")
    if vd is not None:
        env["TT_VISIBLE_DEVICES"] = str(vd)
        env["TT_METAL_VISIBLE_DEVICES"] = str(vd)
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "-o", "addopts=", "-o", "timeout=0", test, "-sv"],
            cwd=str(gitio.repo_root(ctx.model_root())),
            env=env,
            capture_output=True,
            text=True,
            timeout=probes.adaptive_backstop(3600),
        )
    except Exception as exc:  # timeout, OS error, etc.
        return {"status": "crash", "error": str(exc)}
    out = (r.stdout or "") + (r.stderr or "")
    pcc = parse_pcc(out)

    # A SKIPPED e2e test verified NOTHING -- never accept it as correct just because a stale
    # "pcc=..." string happened to be in the log (the seamless SKIP-mislabel pattern).
    if re.search(r"\b[1-9]\d*\s+skipped\b", out, re.IGNORECASE) and not re.search(r"\b[1-9]\d*\s+passed\b", out):
        return {"status": "crash", "error": "e2e PCC test SKIPPED (correctness NOT verified): " + _useful_tail(out)}

    if (
        pcc is None
        and re.search(r"\b[1-9]\d*\s+passed\b", out)
        and not re.search(r"\b[1-9]\d*\s+(failed|errors?)\b", out, re.IGNORECASE)
    ):
        return {"status": "ok", "pcc": None, "note": "gate passed; PCC value not in captured output"}

    if pcc is None:
        return {"status": "crash", "error": _useful_tail(out)}

    # PCC IS the correctness signal for a perf edit. A non-zero pytest EXIT with PCC>=threshold
    # is NOT an edit-induced regression: the e2e gate also enforces BRING-UP checks (Gate-2
    # "graduated modules invoked") and the process prints benign nanobind teardown leaks at
    # interpreter shutdown -- BOTH set a non-zero exit while the math is perfect, and BOTH fail
    # on the UNEDITED baseline too (verified: clean nemotron e2e exits 1 on Gate-2 with PCC
    # 0.999). Gating on the raw return code here rejected every edit. So gate on PCC: a genuine
    # device crash already yields pcc=None above; below-threshold PCC is pcc_low (repairable).
    return {"status": "ok", "pcc": pcc} if pcc >= threshold else {"status": "pcc_low", "pcc": pcc}


# Lines that pollute the crash excerpt: nanobind dumps ~hundreds of "leaked ..." lines at
# interpreter shutdown, which otherwise BURY the real error in the [-N:] tail fed to repair.
_TEARDOWN_NOISE = re.compile(r"nanobind|leaked (type|function)|reference counting|skipped remainder", re.IGNORECASE)


def _useful_tail(out: str, n: int = 2000) -> str:
    """Last n chars of the output with teardown noise removed, so the real error survives."""
    kept = [ln for ln in (out or "").splitlines() if not _TEARDOWN_NOISE.search(ln)]
    return "\n".join(kept).strip()[-n:]
