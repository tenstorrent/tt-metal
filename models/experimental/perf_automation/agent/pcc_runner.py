"""PCC measurement for GATE_PCC (PLAN 8.6) — single-stage e2e.

parse_pcc() is deterministic and unit-tested. run_pcc() runs the model's
end-to-end PCC test on hardware and is the injectable default (ctx.deps["pcc_runner"]);
it is exercised live, not in unit tests. TBD(pcc-parse): the regex assumes the
test prints a "PCC: <float>" style number — refine per the real test's output.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import gitio, probes
from .layer_depth import set_depth as _set_depth

_DEPTH_GUARD = "models.experimental.perf_automation.agent.depth_guard_plugin"

_PCC_RE = re.compile(r"(?i)pcc[^\n]*?[:=]\s*(-?\d+\.\d+)")

_TRACEBACK_BANNER = re.compile(r"^=+ (?:FAILURES|ERRORS) =+$", re.MULTILINE)
_REPORTED_ERROR_LINE = re.compile(r"^E\s")


def parse_pcc(text: str):
    """WORST 'pcc ... <float>' in the test output, or None.

    Was the LAST occurrence, which banked the wrong number in two real shapes: many tt-metal
    tests print the THRESHOLD as `pcc: 0.99` (if that line came last it was recorded as the
    measured value), and a per-layer sweep prints many values (only the last was judged). For a
    correctness gate the worst observed PCC is the one that has to clear the threshold.

    Below pytest's FAILURES banner a `pcc = <float>` is an ECHO of the THRESHOLD, not a
    measurement. When a device fatal lands AT or INSIDE the compare (assert_with_pcc calls
    ttnn.to_torch internally, which OOMs), pytest echoes `pcc=0.99` three ways -- the `>`
    failing line, the def signature, and its frame-argument header -- so a run that computed
    NOTHING scraped 0.99, cleared the threshold and banked pcc_verified=True on a crash. The
    `min` fix above only covers the case where a REAL value is also present to be lower.
    Since the runner passes -s, the test's own prints stream BEFORE the banner; below it only
    pytest's `E ` lines carry a value the test actually reported (a real sub-threshold PCC
    raised in the assertion message), which must stay visible so it routes to pcc_low and gets
    repaired rather than being misread as a crash.
    """
    text = text or ""
    banner = _TRACEBACK_BANNER.search(text)
    if banner:
        reported = [ln for ln in text[banner.start() :].splitlines() if _REPORTED_ERROR_LINE.match(ln)]
        text = text[: banner.start()] + "\n".join(reported)
    matches = _PCC_RE.findall(text)
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


# One line per failed or errored case in pytest's short test summary (-rfE, on by default):
#   FAILED models/.../test_e2e.py::test_signal_quality_wer_and_mos - AssertionError: ...
#   ERROR models/.../test_e2e.py::test_x - ...
# The status word leads there, so a `-s` print cannot split it from the name the way it can on the
# per-case `path::name PASSED` lines of a -v transcript.
_SUMMARY_FAIL_LINE = re.compile(r"^(?:FAILED|ERROR)\s+\S+::(\S+)", re.MULTILINE)
# pytest's closing tally: `===== 10 passed, 1 failed, 5 warnings in 900.12s (0:15:00) =====`
_FINAL_TALLY = re.compile(r"^=+ .*\bin\s+[0-9.]+s\b.*=+\s*$", re.MULTILINE)
_FAILED_COUNT = re.compile(r"\b([1-9]\d*)\s+(?:failed|errors?)\b", re.IGNORECASE)
UNNAMED_FAILURE = "<unnamed>"


def parse_failed_tests(out: str) -> list:
    """Names of the cases pytest reported FAILED or ERROR, in output order, without duplicates.

    Read from the short test summary, which names each case on its own line. When no summary
    survived (a transcript cut before it, or a -r switch that hid it) but pytest's closing tally
    still counts failures, the count is kept as UNNAMED_FAILURE entries so a failure never
    disappears just because its name did.
    """
    text = out or ""
    names = list(dict.fromkeys(_SUMMARY_FAIL_LINE.findall(text)))
    if names:
        return names
    tallies = _FINAL_TALLY.findall(text)
    scope = tallies[-1] if tallies else text
    total = sum(int(n) for n in _FAILED_COUNT.findall(scope))
    return [UNNAMED_FAILURE] * total


def _verdict_from_output(out: str, threshold: float, baseline_failed=None) -> dict:
    """Correctness verdict for a captured pytest run. Split out of run_pcc so the gate's own
    logic is testable without a device -- these branches decide whether an edit is kept.

    `baseline_failed`: the cases of the correctness file that already fail on the UNEDITED model
    (recorded by the loop's BEFORE gate, perf_mcp.record_pcc_baseline). Only those are tolerated;
    every other failing case is `tests_failed`, whatever the PCC. None or [] tolerates nothing."""
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

    # PCC is the first correctness signal for a perf edit, but not the only one. This used to gate
    # on PCC alone: the raw pytest EXIT code was useless because the e2e file also enforces
    # BRING-UP checks (Gate-2 "graduated modules invoked") and nanobind prints teardown leaks at
    # interpreter shutdown -- both set a non-zero exit while the math is perfect, and both fail on
    # the UNEDITED model too (a clean nemotron e2e exited 1 on Gate-2 with PCC 0.999), so gating
    # on the exit code rejected every edit. Ignoring failures altogether over-corrected: the file's
    # OTHER cases (rendered-speech WER/MOS, stop rule, batch independence) print no PCC, so a win
    # could be banked while one of them failed (voxtral, 2026-09-30). The rule is now RELATIVE TO
    # THE UNEDITED MODEL: a case that fails there is tolerated, a case that passed there and fails
    # now is the edit's doing and blocks the win. A genuine device crash already yields pcc=None
    # above; below-threshold PCC is pcc_low (repairable).
    effective = max(float(threshold or 0.0), _operator_pcc_floor())
    failed = parse_failed_tests(out)
    verdict = {"pcc": pcc, "pcc_verified": True, "threshold": effective, "failed_tests": failed}
    if pcc < effective:
        return {"status": "pcc_low", **verdict}
    tolerated = set(baseline_failed or ())
    new_failed = [t for t in failed if t not in tolerated]
    if new_failed:
        return {
            "status": "tests_failed",
            **verdict,
            "new_failed_tests": new_failed,
            "error": (
                "PCC %.6f clears %.2f, but %d case(s) of the correctness file fail that pass on the "
                "unedited model: %s. Every case in the gate file must pass -- fix or revert the edit."
                % (pcc, effective, len(new_failed), ", ".join(new_failed))
            ),
        }
    return {"status": "ok", **verdict}


def _inside_repo(resolved: Path, model_root, repo) -> Path:
    """Keep the correctness gate in the SAME tree the perf gate measures.

    THE GATE AND THE MEASUREMENT HAD DIFFERENT TREES. A run started with an ABSOLUTE
    `--pcc-test` keeps that absolute path in the manifest, and when the run is later driven
    from a git WORKTREE the perf test (a model-root-relative path) resolves into the worktree
    while the PCC test still resolves into the ORIGINAL checkout. pytest then imports
    `models.demos.<model>...` from the original checkout -- so every edit was measured in one
    tree and PCC'd in another, and a gate that never saw the edit answered "ok" for all of them.
    Measured on voxtral_4b_tts_2603 2026-09-21: the two trees were on different branches, and a
    residual-add rewrite in the worktree did not execute one line of the code the gate ran.

    A test file that escapes the repo under measurement is therefore relocated to its in-repo
    twin when there is exactly one, and only then. No twin means the caller really did point
    outside on purpose, and the path is left alone rather than guessed at.
    """
    try:
        repo_root = Path(repo).resolve()
        if repo_root in resolved.resolve().parents:
            return resolved
    except OSError:
        return resolved
    twins = [
        cand
        for base in (Path(model_root), repo_root)
        for cand in sorted(Path(base).rglob(resolved.name))
        if cand.is_file()
    ]
    seen, unique = set(), []
    for cand in twins:
        key = str(cand.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique[0] if len(unique) == 1 else resolved


def run_pcc(ctx) -> dict:
    """Run the e2e PCC test, parse the measured PCC, compare the manifest threshold.

    Returns {status: ok|pcc_low|tests_failed|crash, pcc?, failed_tests?, error?}. A parsed
    number below threshold is pcc_low (expected pytest non-zero exit); a case of the file that
    fails here but passes on the unedited model (ctx.baseline_failed_tests(), when the context
    has one) is tests_failed; an unparseable result or an exception is crash.
    """
    entry = ctx.manifest["pathmap"]["pcc"]["end_to_end"]
    file_part, sep, fn = str(entry["path"]).partition("::")
    repo = gitio.repo_root(ctx.model_root())
    resolved = next(
        (b / file_part for b in (Path(ctx.model_root()), Path(repo)) if (b / file_part).is_file()),
        Path(ctx.model_root()) / file_part,
    )
    resolved = _inside_repo(resolved, ctx.model_root(), repo)
    test = str(resolved) + (sep + fn)
    threshold = entry["threshold"]
    env = dict(os.environ)
    # FULL DEPTH for correctness, expressed by REMOVING the cap rather than by a sentinel: "0"
    # arrives as a truthy string and was read by model builders as "build zero layers", which PCC'd
    # a model that had done no work. See agent/layer_depth.py.
    _set_depth(env, None)
    from .mesh_descriptor import apply_scope

    apply_scope(env, ctx.manifest.get("config", {}))
    probes.wait_for_memory_headroom_before_device_work("check_pcc (full-depth)")
    # -p depth_guard: correctness must run at FULL depth; see agent/depth_guard_plugin.py
    argv = [sys.executable, "-m", "pytest", "-p", _DEPTH_GUARD, "-o", "addopts=", *probes.PYTEST_NO_TIMEOUT]
    cmd = [*argv, test, "-sv"]

    def _once():
        # a fresh log per attempt: run_with_low_memory_fallback may call this twice, and the first
        # attempt's log is removed once it has been read
        log = Path(tempfile.mkdtemp(prefix="pcc_run_")) / "run.log"
        # SUPERVISED LIKE EVERY OTHER DEVICE STEP. This was a plain subprocess.run with a wall-clock
        # kill, so a hung check sat until adaptive_backstop ran out: Qwen-Image-Edit, 2026-09-30, the
        # device went quiet 7 min in and the check was killed at 7210 s, then the retry did the same.
        # probes._execute ends a step that makes no forward progress (ProgressWatch) and keeps the
        # budget only as the ceiling behind it -- the same runner emit-e2e's G6 gate runs this test in.
        rc = probes._execute(
            cmd,
            Path(gitio.repo_root(ctx.model_root())),
            env,
            probes.adaptive_backstop(3600),
            log,
            preexec_fn=probes.memory_cap_preexec_fn(),
            label="check_pcc",
        )
        out = log.read_text(errors="ignore") if log.exists() else ""
        # read, so the log has served its purpose; a STALL raises above and keeps it (its path is in
        # the error). Without this every check left a pytest -sv transcript behind in the tempdir.
        shutil.rmtree(log.parent, ignore_errors=True)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")

    try:
        r = probes.run_with_low_memory_fallback(_once, env)
    except Exception as exc:  # a stall (TracyHangError), the ceiling, an OS error, etc.
        return {"status": "crash", "error": str(exc)}
    out = (r.stdout or "") + (r.stderr or "")
    # A context without the record (the kernel_test shim, older callers) tolerates no failure.
    baseline_failed = getattr(ctx, "baseline_failed_tests", lambda: None)()
    return _verdict_from_output(out, threshold, baseline_failed=baseline_failed)


# Lines that pollute the crash excerpt: nanobind dumps ~hundreds of "leaked ..." lines at
# interpreter shutdown, which otherwise BURY the real error in the [-N:] tail fed to repair.
_TEARDOWN_NOISE = re.compile(r"nanobind|leaked (type|function)|reference counting|skipped remainder", re.IGNORECASE)


def _useful_tail(out: str, n: int = 2000) -> str:
    """Last n chars of the output with teardown noise removed, so the real error survives -- led by
    any dead-board evidence the cut would drop (device_recovery.dead_board_evidence). This text is
    what check_pcc hands the device-recovery policy, and a tail of pytest's warnings summary carried
    none of the signatures that justify a reset."""
    kept = [ln for ln in (out or "").splitlines() if not _TEARDOWN_NOISE.search(ln)]
    tail = "\n".join(kept).strip()[-n:]
    try:
        from .device_recovery import dead_board_evidence
    except Exception:  # noqa: BLE001 -- without the policy module there is no evidence to carry
        return tail
    lead = [ln for ln in dead_board_evidence("\n".join(kept)).splitlines() if ln not in tail]
    return "\n".join(lead + [tail]) if lead else tail
