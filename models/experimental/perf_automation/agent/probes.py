"""Real-boundary probes for the Before Loop (PLAN sections 7.1/7.3/7.4).

Production implementations for the three injectable boundaries M3 left mocked:

  tt_smi_probe            environment_check probe — closes TBD(env-script).
  sdk_model_files_runner  read_model_files runner — the SDK sub-agent (the ONLY
                          LLM call in this module; output is validated by
                          model_files._validate, never trusted).
  make_run_profiled       tracy stage-1 RUN per the stage-1 REAL-RUN contract
                          (command from GUIDELINES/09 section 1; tee log;
                          watermark-glob CSV discovery; archive-before-parse;
                          crash on nonzero/timeout/no-CSV).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .environment import EnvironmentError_


def observed_tracy_baseline_seconds(manifest_path) -> float:
    """Seconds the tracy baseline took, from the run's own events.jsonl. 0.0 when not recorded.

    THREE PLACES PARSED THIS, identically: adaptive_backstop and adaptive_op_timeout here, and
    run.py's _baseline_ceiling -- each opening events.jsonl beside the manifest, skipping blank
    lines, json-loading each one and matching stage == "tracy_baseline" with event == "done". Three
    copies of one file format is three places to change when the event is renamed, and nothing to
    keep them agreeing meanwhile.
    """
    try:
        import json as _json

        for ln in (Path(manifest_path).parent / "events.jsonl").read_text().splitlines():
            if not ln.strip():
                continue
            e = _json.loads(ln)
            if e.get("stage") == "tracy_baseline" and e.get("event") == "done" and e.get("seconds"):
                return float(e["seconds"])
    except Exception:  # noqa: BLE001 -- an unreadable log is no observation, never a failure
        return 0.0
    return 0.0


def adaptive_backstop(floor_default: int = 3600, mult: int = 3, env_key: str = "PERF_MCP_MEASURE_BACKSTOP") -> int:
    """Hard backstop for a long device operation.

    Was `max(floor_default, mult*baseline)` capped at the ceiling -- and since 3*baseline only beats
    a 3600 s floor above a ~1200 s baseline, the floor was the de-facto policy for every model
    actually run, including the PCC gate (the longest operation in a round). Same defect the other
    timers were fixed for, so it now uses the same chain: observed p95 for this op, else an
    agent estimate from the model's own evidence, else scaled from its baseline.
    """
    override = os.environ.get(env_key)
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    _chained = adaptive_op_timeout("pcc", mult=float(mult))
    if _chained > 0:
        return _chained
    floor = floor_default
    ceil = 10800
    base = 0.0
    mp = os.environ.get("PERF_MCP_MANIFEST")
    if mp:
        m = Path(mp)
        try:
            cfg = json.loads(m.read_text()).get("config", {}) or {}
            ceil = int(cfg.get("timeout", ceil) or ceil)
        except Exception:  # noqa: BLE001
            pass
        base = observed_tracy_baseline_seconds(m) or base
    if ceil < floor:
        ceil = floor
    return min(ceil, max(floor, int(mult * base)))


# ---------------------------------------------------------------------------
# 7.1 environment probe — `tt-smi -s` (TBD(env-script): CLOSED)
# ---------------------------------------------------------------------------


def adaptive_op_timeout(op: str, *, env_key: str = "", mult: float = 0.0) -> int:
    """Timeout for one operation, scaled from OBSERVED durations for that operation.

    Agent-side counterpart to cc_optimize.run.adaptive_timer (importing run.py from here
    would be circular). Reads the same observed_durations.json the round writes, falling
    back to the tracy baseline scaled into the operation's units. No absolute floors --
    they were the BUG 4 defect: a 240 s build cap vs llama's real 872 s build, and a 300 s
    agent-call cap on a model whose calls take 900 s, while a 3 ms module got 3600 s.
    """
    if env_key:
        ov = os.environ.get(env_key)
        if ov:
            try:
                return int(ov)
            except ValueError:
                pass
    min_s = float(os.environ.get("PERF_MCP_MIN_TIMER_S", "30") or "30")
    ceil = 10800.0
    base = 0.0
    cost = 0.0
    mp = os.environ.get("PERF_MCP_MANIFEST")
    if mp:
        m = Path(mp)
        try:
            cfg = json.loads(m.read_text()).get("config", {}) or {}
            ceil = float(cfg.get("timeout", ceil) or ceil)
        except Exception:  # noqa: BLE001
            pass
        try:
            obs = json.loads((m.parent / "observed_durations.json").read_text()).get(op) or []
            vals = sorted(float(x) for x in obs if float(x) > 0)
            if vals:
                cost = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
        except Exception:  # noqa: BLE001
            pass
        if cost <= 0:
            base = observed_tracy_baseline_seconds(m) or base

    if cost > 0:
        # OBSERVED cost for this very operation on this very model: the only precise input.
        return int(min(ceil, max(min_s, (mult or 4.0) * cost)))

    # COLD START -- no observation yet. Ask the agent to size it from this model's own evidence
    # rather than applying a per-op multiplier table, which is a guess about every future model
    # frozen at authoring time (that table is what put a 240 s cap against llama's 872 s build).
    est = _agent_seconds(op, base, ceil)
    if est > 0:
        return int(min(ceil, max(min_s, est)))
    # No observation and no agent. Do NOT invent a fixed number (300/240 were the defect) and do
    # NOT concede the whole ceiling either -- that would let a frozen call sit for hours. Scale from
    # the model's OWN baseline with one generous factor, so a 3 s module is judged in seconds and an
    # 8B pipeline gets room for its real work. One constant, no per-operation table.
    # With no observation AND no agent there is genuinely nothing to derive an op's cost from. The
    # options are a table of relative op costs (a guess about every future model, frozen at authoring
    # time -- and dropping it made a tiny module's PCC backstop 37 s where the table implied 170 s),
    # or conceding the operator's own ceiling. Concede: a budget that is too TIGHT kills healthy work
    # and wastes the run, while a loose one only delays detection -- and round liveness is judged
    # separately by watchdog_decide from real evidence, not by this clock.
    return int(max(min_s, ceil))


def _agent_seconds(op: str, baseline_s: float, ceiling_s: float) -> float:
    """Agent-estimated budget for `op` on this model, cached per (op, baseline)."""
    try:
        from . import integrity as _integrity
    except Exception:  # noqa: BLE001
        return 0.0
    model = os.environ.get("PERF_MCP_TASK", "") or "the model under test"
    known = (
        "its full profiled baseline run takes %.1f s" % baseline_s
        if baseline_s > 0
        else "its baseline duration is not yet known"
    )
    return _integrity.ask_number(
        "A Tenstorrent TTNN performance-optimization tool needs a timeout for ONE operation of kind "
        "%r on %s, where %s. Operation kinds: 'profile' = one tracy-profiled forward pass; 'pcc' = "
        "the full end-to-end correctness test; 'build' = generating/compiling and running a perf "
        "test; 'round' = a complete edit -> correctness -> measure -> commit cycle; 'agent' = one LLM "
        "call that may use many tool turns. How many seconds should the budget be?" % (op, model, known),
        lo=30.0,
        hi=ceiling_s,
        cache_key="timeout|%s|%s|%d" % (op, model, int(baseline_s)),
    )


def board_to_arch(board_type: str) -> str | None:
    b = (board_type or "").strip().lower()
    if not b:
        return None
    try:
        from scripts.tt_hw_planner.hardware import HARDWARE
    except Exception:
        return None
    for box in HARDWARE:
        for bt in box.board_types:
            if bt and b.startswith(bt.lower()):
                return box.arch.lower()
    return None


def device_is_responsive(timeout_s: float = 20.0) -> bool:
    """Does the board answer at all? The question a timeout must ask before resetting anything.

    DELIBERATELY NOT tt_smi_probe. That one raises on an unrecognised board_type -- measured here,
    it rejects this host's own `p300c` -- and a board that answers with a name the table lacks is
    ALIVE. Conflating "did not answer" with "answered something unfamiliar" would reset healthy
    hardware, which is the exact failure this check exists to prevent.

    So the bar is only: did tt-smi return, and did it name any device. No arch mapping, no schema
    beyond `device_info` being non-empty. Bounded well under tt-smi's own 120 s because what is being
    established is whether the answer comes back PROMPTLY; a board that needs two minutes to say
    hello is not one worth protecting from a reset.

    Any failure returns False, which resets exactly as before: this only ever adds a reason NOT to
    reset, never a reason to.
    """
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    try:
        proc = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=timeout_s)
        return bool((json.loads(proc.stdout) or {}).get("device_info"))
    except Exception:  # noqa: BLE001
        return False


def tt_smi_probe() -> str:
    """Run `tt-smi -s`, normalize to the snapshot shape parse_env_snapshot expects.

    The real snapshot has no `arch` key — it carries board_info.board_type
    (e.g. "n300 L"); we adapt that to the arch token here.
    """
    proc = subprocess.run(["tt-smi", "-s"], check=True, capture_output=True, text=True, timeout=120)
    data = json.loads(proc.stdout)
    devices = data.get("device_info") or []
    if not devices:
        raise EnvironmentError_("tt-smi -s reported no devices")
    board = (devices[0].get("board_info") or {}).get("board_type", "")
    arch = board_to_arch(board)
    if arch is None:
        raise EnvironmentError_(f"unrecognized board_type from tt-smi: {board!r}")
    return json.dumps({"card": board, "arch": arch, "device_count": len(devices)})


# ---------------------------------------------------------------------------
# 7.3 model-files sub-agent runner — SDK query(), read-only tools
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str:
    """Return the last balanced top-level {...} block in `text` (the agent may
    precede its JSON with prose despite instructions — never trust formatting)."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        end = text.rfind("}", 0, end)
    return text  # let model_files raise its own ModelFilesError


def _usage_summary(result_msg) -> dict:
    """Flatten a ResultMessage into {tokens_in, tokens_out, cost_usd, latency_s}."""
    u = getattr(result_msg, "usage", None) or {}
    # input_tokens counts only the UNCACHED slice; the bulk of the prompt is
    # in the cache fields. tokens_in = total tokens the model actually saw.
    tokens_in = (
        (u.get("input_tokens") or 0)
        + (u.get("cache_creation_input_tokens") or 0)
        + (u.get("cache_read_input_tokens") or 0)
    )
    return {
        "tokens_in": tokens_in or None,
        "tokens_cached": u.get("cache_read_input_tokens"),
        "tokens_out": u.get("output_tokens"),
        "cost_usd": getattr(result_msg, "total_cost_usd", None),
        "latency_s": round(getattr(result_msg, "duration_ms", 0) / 1000.0, 2),
    }


def cli_model_files_runner(max_turns: int = 24) -> Callable[[str], str]:
    """CC-native discovery runner: drives the `claude` CLI (login auth, no SDK, no model tier) to map
    the model tree and return the pathmap JSON. The cc engine uses this so its discovery is claude-code
    like the rest of cc; the FSM engine keeps sdk_model_files_runner. Same prompt/tools as the SDK one."""
    _sys = (
        "You map model source trees for performance tooling. Use ONLY the read-only tools provided "
        "(Read, Glob, Grep). Your FINAL message must be exactly one JSON object — no prose, no code fences."
    )

    def runner(prompt: str) -> str:
        env = dict(os.environ)
        for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(_k, None)
        try:
            from .agent_bin import resolve_claude_bin

            r = subprocess.run(
                [
                    resolve_claude_bin(),
                    "-p",
                    prompt,
                    "--output-format",
                    "text",
                    "--system-prompt",
                    _sys,
                    "--allowedTools",
                    "Read,Glob,Grep",
                    "--max-turns",
                    str(max_turns),
                    "--permission-mode",
                    "bypassPermissions",
                ],
                capture_output=True,
                text=True,
                timeout=1200,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cc discovery (claude CLI) failed to run: {exc}") from exc
        if r.returncode != 0:
            raise RuntimeError(f"cc discovery (claude CLI) exit {r.returncode}: {(r.stderr or '')[-200:]}")
        return _extract_json_object(r.stdout or "")

    runner.last_usage = None
    runner.model = "claude-cli"
    return runner


# ---------------------------------------------------------------------------
# 7.4 tracy stage-1 RUN + preflight — per the FINAL stage-1 contract
# ---------------------------------------------------------------------------


class TracyRunError(Exception):
    """Stage-1 crash: nonzero exit, timeout, or no usable CSV produced."""


class PerfRunFailed(TracyRunError):
    """The profiled perf test CRASHED at runtime (a ttnn op TT_FATAL/RuntimeError
    during the forward) — NOT a flaky/partial measurement. `python -m tracy -m pytest`
    exits 0 even when the inner test fails, so the partial CSV would otherwise be
    mistaken for an `op_count_mismatch` measurement. Carries `.error` (the device-op
    error) so REMEASURE can route it to REPAIR_CODE and the agent fixes its own edit."""

    def __init__(self, error: str, log_path=None):
        super().__init__(f"perf test crashed at runtime: {error}")
        self.error = error
        self.log_path = log_path


class ThrottledRunError(TracyRunError):
    """The run completed, but the device throttled its clock while it was being measured.

    NOT a crash and NOT a usable measurement. The driver clamps AICLK from 1350 MHz to 800 when the
    board is too hot, so the numbers are real timings of a slower machine. Comparing one against a
    baseline taken at full clock is comparing two different pieces of hardware: a good edit measured
    hot looks ~40% slower and gets reverted, and a hot BASELINE makes the next candidate look like a
    win worth committing. Both directions corrupt the ledger silently, which is worse than failing.

    Raised only after re-measuring has been tried and the board still cannot hold its clock.
    """


class TracyHangError(TracyRunError):
    """Watchdog killed a run that made no forward progress (stalled/deadlocked,
    e.g. an intermittent multi-chip CCL deadlock) — distinct from an edit-induced
    crash. Retriable: reset the device and re-profile."""


_ERR_RE = re.compile(
    r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Fault):"  # `ExceptionType: message` (not a `raise X(` line)
    r"|Segmentation fault|Aborted|core dumped|TT_FATAL|terminate called|Fatal Python error",
)


def _salient_tail(text: str, n: int = 4) -> str:
    """The human-meaningful last lines of a failed run's log — the actual error/signal, not the Python
    frame stack. Prefers lines that look like an error or a fatal signal (so a terminal shows e.g.
    'Segmentation fault' + 'AssertionError: cpp_device_perf_report.csv not found' instead of 15 lines of
    traceback), de-duped, most recent last. Falls back to non-frame lines if nothing matches. The full
    log path is always printed alongside for the details."""
    hits, seen = [], set()
    for ln in text.splitlines():
        s = ln.strip()
        if s and _ERR_RE.search(s) and s not in seen:
            seen.add(s)
            hits.append(s)
    if hits:
        return "\n".join(hits[-n:])
    keep = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith('File "') and set(ln.strip()) != {"^"}
    ]
    return "\n".join(keep[-n:])


# A device-op runtime crash (the edit broke the model), distinct from a benign
# perf-threshold AssertionError (the model ran fully — valid measurement). TT_FATAL is
# the unambiguous device-op abort; a ttnn-op RuntimeError (decorators.py) is the wrapper.
# Device-op / runtime crash signatures, distinct from a benign perf-threshold AssertionError
# (the model ran fully -> valid measurement). Broadened beyond TT_FATAL to cover C++ aborts,
# segfaults, and TT_ASSERT that surface in the log even though `python -m tracy` exits 0.
_CRASH_RE = re.compile(
    r"(TT_FATAL[^\n]*|TT_THROW[^\n]*|TT_ASSERT[^\n]*|E\s+RuntimeError:[^\n]*"
    r"|Segmentation fault[^\n]*|terminate called[^\n]*|libc\+\+abi[^\n]*|Aborted[^\n]*|core dumped[^\n]*"
    # tt-lang (ttl) kernel authoring: a custom kernel that fails to COMPILE/LOWER must route to
    # REPAIR_CODE (fix the kernel) instead of being misread as a partial/benign capture. Covers ttl
    # Python exceptions, MLIR diagnostics, and compile/lower/build failures. (Refine the exact
    # signatures against a real tt-lang compile error once the kernel lever runs on device.)
    r"|ttl\.[A-Za-z_]*(?:Error|Exception)[^\n]*|tt-lang[^\n]*?[Ee]rror[^\n]*"
    r"|Compil(?:e|ation)Error[^\n]*|LoweringError[^\n]*|failed to (?:compile|lower|build)[^\n]*"
    r"|loc\([^\n]*\):\s*error:[^\n]*|ttmlir[^\n]*?error[^\n]*)"
)
_DEVICE_CRASH_RE = re.compile(r"Segmentation fault|core dumped|Aborted|terminate called|libc\+\+abi")
# pytest end-of-run summary: BOTH "failed" and "error" (collection/fixture errors print as
# "N errors", never "failed") mark a non-passing run.
_TEST_FAILED_RE = re.compile(r"=+\s*(\d+)\s+(?:failed|error)", re.IGNORECASE)


def detect_perf_crash(log_text: str) -> str | None:
    """If the profiled run crashed in a device op, return the error excerpt; else None.
    Requires BOTH a pytest failure/error AND a crash signature, so a model that ran fully but
    failed only a perf-threshold assert is NOT treated as a crash. `tracy -m pytest` exits 0
    even on inner failure, so a non-zero exit can't be relied on -- the log is the evidence."""
    if not log_text:
        return None
    fm = _TEST_FAILED_RE.search(log_text)
    failed = bool(fm and int(fm.group(1)) > 0) or ("FAILED " in log_text and _CRASH_RE.search(log_text) is not None)
    if not failed:
        return None
    cm = _CRASH_RE.search(log_text)
    return cm.group(1).strip() if cm else None


_MARKER_DROP_RE = re.compile(
    r"markers were dropped"
    r"|marker was dropped"
    r"|PERF_AUTOMATION_ORPHAN_SKIP"
    r"|report will be partial"
    r"|DRAM[- ]buffer overflow"
    r"|marker imbalance"
    r"|dropped due to DRAM",
    re.IGNORECASE,
)

_MAX_PROFILER_SUPPORT_COUNT = 2_000_000
_MAX_HEAL_ATTEMPTS = 4
_HEAL_GROWTH = 8


def detect_marker_drop(log_text: str) -> str | None:
    if not log_text:
        return None
    m = _MARKER_DROP_RE.search(log_text)
    return m.group(0) if m else None


class PreflightError(Exception):
    """The discovered perf test selects zero tests (the S512 trap)."""


# THE ENVIRONMENT A TRACY PROFILING RUN EXECUTES UNDER, named once.
#
# Two consumers need to agree on it and used to hold their own copies of the literals: this module,
# which sets them on the subprocess, and stage_marks, which has to work out WHICH branch of the
# generated test runs under them before it can put the stage marks in a reachable place. It guessed
# instead -- "the last bare call is the profiled branch" -- and on a regenerated test that call sat
# inside `if _PERF_TRACE:`, which is false here, so both the bracket and the per-stage pass landed in
# code the profiler never executes and the capture came back with zero signposts.
PROFILING_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    # EAGER-ONLY under tracy: a trace replay runs as one fused program and emits no per-op device
    # data, so profiling it floods the buffer with nothing useful.
    "TT_PERF_TRACE": "0",
}


def build_tracy_command(perf_test: str, case: str | None, out_dir: str | Path) -> list[str]:
    """The raw profile_this command (C++ post-processing default) + -o.

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -o <out> -m pytest ... -sv
    Run directly (never via profile_this.py: it swallows the exit code).
    """
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-v",
        "-r",
        "-p",
        "-o",
        str(out_dir),
        "-m",
        "pytest",
        "-o",
        "timeout=0",
        perf_test,
    ]
    if case:
        cmd += ["-k", case]
    cmd += ["-sv"]
    return cmd


def _proc_stat_fields():
    """(pid, fields) for every live process, with the comm field already stepped over.

    THE SAME EIGHT LINES, FOUR TIMES. Walking /proc, opening <pid>/stat, skipping past the comm --
    which is parenthesised and may itself contain spaces or a ')', so the split must start after the
    LAST one -- and splitting the rest, was written out in _pgroup_cpu_jiffies, _descendant_pids,
    the pids-in-group helper below, and again verbatim as run.py's _pg_cpu_jiffies. Only the fields
    each wanted differed: 11 and 12 for jiffies, 1 for the parent, 2 for the process group.

    A generator, so a caller reads the fields it needs and nothing accumulates a list of every
    process on the machine.
    """
    import os as _os

    try:
        entries = _os.listdir("/proc")
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as fh:
                data = fh.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        rp = data.rfind(")")
        if rp == -1:
            continue
        yield int(entry), data[rp + 2 :].split()


_STACK_EVERY_S = 30.0

# How far past its budget a still-moving step may run before the attempt is failed. A multiple, so
# it scales with what the caller already said the work is worth.
_HARD_CEILING_MULT = 4


def _pgroup_io_counters(pgid) -> tuple:
    """(syscalls, io_bytes) summed over the process group. (0, 0) when /proc cannot be read.

    CPU IS NOT EVIDENCE OF PROGRESS, and this is what is. A process doing real work with a device
    crosses into the kernel constantly -- ioctl to the driver, reads, writes -- so its syscall and
    byte counters move. A process spinning in userspace moves neither.

    Measured on run 12's hang, 2026-08-20: pinned at a full core for ten hours inside one
    ttnn.from_torch call, with syscr and syscw unchanged across a twenty-second window and
    read_bytes/write_bytes flat. Its stall clock never fired because CPU movement reset it on every
    poll.
    """
    calls = 0
    total = 0
    for pid, fields in _proc_stat_fields():
        if len(fields) <= 2 or fields[2] != str(pgid):
            continue
        try:
            with open("/proc/%d/io" % pid) as fh:
                for line in fh:
                    k, _, v = line.partition(":")
                    if k in ("syscr", "syscw"):
                        calls += int(v)
                    elif k in ("read_bytes", "write_bytes"):
                        total += int(v)
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
            continue
    return calls, total


def _stack_fingerprint(pid) -> str:
    """The top of the process's Python stack, or "" when it cannot be sampled.

    THE ONE SIGNAL A KERNEL-SIDE LIVELOCK CANNOT FAKE. A poll loop asking the driver "ready yet?"
    moves the syscall counter forever, so the counters above call it alive; its STACK does not move.
    Sampled with py-spy when it is installed -- optional by design, because the counters already
    cover the common case and a missing profiler must not make the tool refuse to run.
    """
    import shutil as _shutil
    import subprocess as _sp

    exe = _shutil.which("py-spy")
    if not exe:
        return ""
    try:
        r = _sp.run([exe, "dump", "--pid", str(int(pid))], capture_output=True, text=True, timeout=8)
    except Exception:  # noqa: BLE001 -- an unsampleable process contributes nothing, never an error
        return ""
    if r.returncode != 0:
        return ""
    # The frames only -- not the thread header, which carries a state that flickers between samples.
    return "\n".join(ln.strip() for ln in (r.stdout or "").splitlines() if ln.startswith("    "))[:2000]


def progress_signature(pgid, log_path=None, pid=None) -> tuple:
    """Everything that changes when a process tree is getting somewhere, and nothing that changes
    when it merely runs.

    Deliberately EXCLUDES CPU. The three states a supervised step can be in are told apart like so:

        working    log grows, or syscalls/bytes move, or the stack moves
        deadlock   nothing moves at all, CPU included
        livelock   CPU pinned; log, counters and stack all still

    Only the last was invisible before, and it is the one that cost ten hours.
    """
    try:
        size = int(Path(log_path).stat().st_size) if log_path else 0
    except OSError:
        size = -1
    calls, io_bytes = _pgroup_io_counters(pgid)
    return (size, calls, io_bytes, _stack_fingerprint(pid) if pid else "")


_STACK_MIN_QUIET_S = 60.0


class ProgressWatch:
    """Has this process made progress since the last poll? One owner for that question.

    THIS WAS COPY-PASTED INTO THREE SUPERVISED LOOPS on 2026-08-20 -- probes._execute,
    run._run_device_proc and perf_mcp._adaptive_run -- and immediately drifted: three spellings of
    the stack threshold (`_STACK_EVERY_S` in one, a literal 30.0 in the others) and two different
    quiet windows (`stall/2` in one, `max(60, stall/2)` in the others). Same rule, three copies,
    already diverging on the day it was written. So it lives here, once.

    Holds the last signature and the last time a stack was sampled; `moved()` folds a fresh sample
    in and answers. The stack is the expensive field -- it costs a subprocess -- so it is only
    sampled once the cheap counters have been still for half the stall window (and never more often
    than _STACK_EVERY_S), and only when the caller offers a pid.
    """

    def __init__(self, pgid, log_path=None, stall_s=0.0):
        self._pgid = pgid
        self._log = log_path
        self._stall_s = float(stall_s or 0.0)
        self._sig = progress_signature(pgid, log_path)
        self._last_stack_at = 0.0

    def moved(self, now, last_progress, pid=None) -> bool:
        want = (
            pid is not None
            and (now - last_progress) >= max(_STACK_MIN_QUIET_S, self._stall_s / 2)
            and now - self._last_stack_at >= _STACK_EVERY_S
        )
        new = progress_signature(self._pgid, self._log, pid if want else None)
        if want:
            self._last_stack_at = now
        old = self._sig
        # Compare the stack only when BOTH samples carry one; otherwise the cheap fields decide.
        did = new[:3] != old[:3] or (bool(new[3]) and bool(old[3]) and new[3] != old[3])
        # Keep the last stack we managed to read, so a poll that skipped it does not look like a change.
        self._sig = new if (new[3] or not old[3]) else (new[0], new[1], new[2], old[3])
        return did


def _pgroup_cpu_jiffies(pgid: int) -> int:
    """Sum utime+stime (jiffies) over all live PIDs in process group `pgid`, from /proc.
    Liveness signal: a process doing real work (e.g. compiling kernels) keeps accruing CPU;
    a hung/deadlocked one blocked on a lock or I/O accrues ~none. Best-effort; 0 on any error."""
    total = 0
    target = str(pgid)
    for _pid, fields in _proc_stat_fields():
        if len(fields) > 12 and fields[2] == target:
            try:
                total += int(fields[11]) + int(fields[12])
            except ValueError:
                continue
    return total


def _descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, fields in _proc_stat_fields():
        if len(fields) > 1:
            children.setdefault(int(fields[1]), []).append(pid)
    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        for kid in children.get(pid, ()):
            out.append(kid)
            stack.append(kid)
    return out


def _pgroup_members(pgid) -> list:
    """Live PIDs whose process group is `pgid`, from /proc. Best-effort; [] on any error."""
    target = str(pgid)
    return [pid for pid, fields in _proc_stat_fields() if len(fields) > 2 and fields[2] == target]


def _reap_process_group(pgid) -> list:
    """Kill anything still alive in `pgid` after its leader has exited. Returns the PIDs killed.

    _execute deliberately starts the profiled run in its OWN session so the group can be killed --
    the docstring says the group kill exists "so orphaned capture-release daemons die too". But the
    group was only killed on the stall/backstop paths; the normal `return proc.wait(...)` reaped the
    leader and nothing else. tools/tracy/__main__.py launches tracy-capture and serve_wasm.py as
    children, and a daemon outliving its parent is RE-PARENTED, not killed -- so a run that SUCCEEDS
    left them behind (7 tracy-capture + 2 serve_wasm observed on llama3_1_8b_p150), holding the
    device so the next run could not open it.

    Refuses to touch our own group: the profiled run gets a fresh session, so a pgid equal to ours
    means the caller passed something wrong, and killing it would take out the optimize run itself.
    """
    import signal

    try:
        target = int(pgid)
    except (TypeError, ValueError):
        return []
    if target <= 0 or target == os.getpgid(0):
        return []
    victims = [p for p in _pgroup_members(target) if p != os.getpid()]
    if not victims:
        return []
    try:
        os.killpg(target, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        for p in victims:
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return victims


def _kill_tree(root_pid: int, extra=()) -> None:
    """SIGKILL root_pid, every descendant still traceable from it, and every process group involved.

    `extra` IS THE HALF THE /proc WALK CANNOT DO. _descendant_pids reads PPIDs, so it only sees a
    tree whose ancestors are still alive: the moment root_pid exits, its children are reparented to
    init and the link is gone for good. A caller that waits for a process to finish on its OWN --
    rather than killing it -- therefore has nothing left to walk, and its grandchildren survive.
    Snapshot them WHILE the root lives and pass them here.

    Observed 2026-08-16: a perf-test agent (`claude -p ...`) and its parent outlived the run that
    spawned them by 37 and 70 minutes, in their own sessions, holding no device -- invisible to both
    the process-group kill and the device-holder reclaim. A second optimize attempt then started
    alongside the first, and two runs driving one board took its ARC cores down.
    """
    import signal

    # NEVER OURSELVES. The walk could only ever reach our descendants, so this was safe by
    # construction; `extra` is a caller-supplied list and is not. A snapshot pid whose group happens
    # to be ours would take out the run doing the reaping.
    _self, _selfpg = os.getpid(), os.getpgid(0)
    pids = [
        p for p in (_descendant_pids(root_pid) + [root_pid] + [int(x) for x in (extra or ())]) if p != _self and p > 0
    ]
    pgids = set()
    for pid in pids:
        try:
            _pg = os.getpgid(pid)
            if _pg != _selfpg:
                pgids.add(_pg)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


_GALAXY_HOST: bool | None = None


def _galaxy_capability_probe(tt_smi: str) -> bool | None:
    """Ask tt-smi DIRECTLY whether this is a Galaxy host, on the healthy startup board — the only
    signal that survives the mesh rewiring (board_type strings, physical ASIC enumeration, and the
    --box/--mesh chip count are all unreliable now). `-glx_list_tray_to_device` lists galaxy trays and
    succeeds ONLY on a Galaxy; it errors on a plain board. Returns None if the probe itself failed to
    run (tt-smi missing / timed out), so the caller can fall back to hints."""
    try:
        r = subprocess.run([tt_smi, "-glx_list_tray_to_device"], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "tray" in out.lower():
        return True
    if r.returncode != 0:
        return False
    return None


def note_board(card: str = "", device_count: int = 0, box: str = "", tt_smi: str | None = None) -> None:
    """Record, at healthy STARTUP, whether this host is a Galaxy — a Galaxy needs `-glx_reset`, a plain
    board needs `-r`, and a WEDGED board can't be re-probed at reset time so the decision must be made
    now. Order of trust: explicit env override -> tt-smi galaxy-tray capability probe (authoritative,
    survives the mesh rewiring) -> cheap hints (box/board name says 'galaxy', or >=32 chips) as a
    last-ditch fallback when the probe couldn't run."""
    global _GALAXY_HOST
    v = os.environ.get("TT_HW_PLANNER_GALAXY")
    if v is not None:
        _GALAXY_HOST = v.strip().lower() in ("1", "true", "yes")
        return
    text = f"{card} {box}".strip().lower()
    if "galaxy" not in text and 0 < device_count < 32:
        _GALAXY_HOST = False
        return
    smi = tt_smi or shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    probed = _galaxy_capability_probe(smi)
    if probed is not None:
        _GALAXY_HOST = probed
        return
    _GALAXY_HOST = "galaxy" in text or device_count >= 32


def _enumerated_device_count() -> int:
    """How many local Tenstorrent chips the host exposes, counted from /dev/tenstorrent (works even when
    the runtime is wedged and ttnn enumeration itself throws). Drives a box-complete reset device list."""
    try:
        import glob

        return len([p for p in glob.glob("/dev/tenstorrent/*") if os.path.basename(p).isdigit()])
    except Exception:
        return 0


def _reset_arg_sets() -> list[list[str]]:
    """The tt-smi reset invocations to try, in order, for THIS host. An explicit override wins; else a
    Galaxy host uses the galaxy-tray reset (auto-retry first) with the plain reset as a last-ditch
    fallback, and a non-Galaxy host resets the FULL enumerated chip list (`-r 0,1,..,N-1`). A partial
    reset (a subset of a multi-chip board's chips) leaves the untouched chips' clock arbiter inconsistent
    and wedges device-open, so the reset must cover every chip the box exposes, not a fixed subset."""
    override = os.environ.get("TT_HW_PLANNER_RESET_ARGS")
    if override:
        return [override.split()]
    galaxy = _GALAXY_HOST
    if galaxy is None:
        galaxy = os.environ.get("TT_HW_PLANNER_GALAXY", "").strip().lower() in ("1", "true", "yes")
    if galaxy:
        return [["-glx_reset_auto"], ["-glx_reset"], ["-r"]]
    n = _enumerated_device_count()
    if n >= 2:
        return [["-r", ",".join(str(i) for i in range(n))]]
    if n == 1:
        return [["-r", "0"]]
    return [["-r"]]


def _device_reset(error_text: str = "", config_target: str = "") -> bool:
    """Reset the device and report whether it CAME BACK -- not merely whether tt-smi exited 0.

    Routed through the shared recovery primitive so the profiler layer picks its target from the
    same evidence, verifies the same way, and spends the same escalation budget as the orchestrator
    and the MCP server. This used to return the exit code of a reset aimed at whatever
    _reset_arg_sets() decided, with nothing checking the device afterwards.
    """
    from . import device_recovery as _dr

    def _issue(target):
        tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
        arg_sets = [["-r", target]] if target and target != "all" else _reset_arg_sets()
        for args in arg_sets:
            try:
                proc = subprocess.run([tt_smi, *args], capture_output=True, text=True, timeout=300)
                if proc.returncode == 0:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    return _dr.recover("probes", _issue, error_text=error_text, config_target=config_target)


# "AICLK failed to settle" is what UMD emits TODAY (tt_device.cpp:342) and is arch-independent: it
# fires whenever the clock misses what was asked for. The older "Waiting for AICLK value to settle
# failed" and "possible overheating" wordings are no longer anywhere in UMD, and are kept only so
# archived logs still match. Without the current phrasing this caught a clamp ONLY via the "AICLK
# clamped" arbiter tag, which rides on TelemetryTag::AICLK_ARB_MAX behind is_entry_available() --
# so on a part whose telemetry enum lacks that tag, a clamped run read as perfectly healthy.
_DEVICE_OVERHEAT_RE = re.compile(
    r"Waiting for AICLK value to settle failed|possible overheating|AICLK clamped|AICLK failed to settle"
)
_COOL_MARGIN_C = float(os.environ.get("PERF_MCP_COOL_MARGIN_C", "5") or "5")
_COOL_POLL_S = float(os.environ.get("PERF_MCP_COOL_POLL_S", "5") or "5")
_COOL_MAX_S = float(os.environ.get("PERF_MCP_COOL_MAX_S", "120") or "120")
_MAX_THROTTLE_RETRIES = int(os.environ.get("PERF_MCP_THROTTLE_RETRIES", "2") or "2")


_SIBLINGS: list = []
_THERMAL_INERT_WARNED = [False]


def _warn_thermal_inert(where: str, exc: BaseException) -> None:
    """Say ONCE, loudly, that temperature protection is not running.

    "A gate that cannot run must not stop the work" is right, but it was implemented as `except:
    pass`, which also means a gate that cannot run does not TELL anyone. On 2026-08-29 a swallowed
    ImportError left every thermal gate inert; the board held 99-103C for an hour and chips 2 and 3
    stopped answering, and the log looked completely ordinary throughout. The run still continues --
    that part was never wrong -- but silence is what made a one-line bug cost two chips.
    """
    if _THERMAL_INERT_WARNED[0]:
        return
    _THERMAL_INERT_WARNED[0] = True
    print(
        "  [thermal-gate] WARNING: temperature protection is INERT at %s (%s: %s). Device work will "
        "continue with NO thermal gating for the rest of this run." % (where, type(exc).__name__, str(exc)[:110]),
        file=sys.stderr,
        flush=True,
    )


def _cc_optimize(name: str):
    """A cc_optimize module, reachable however THIS process loaded probes.

    probes is imported both as `models.experimental.perf_automation.agent.probes` and, during an
    optimize run, as `agent.probes` with perf_automation on sys.path -- where `from ..cc_optimize.x`
    raises "beyond top-level package". Resolution lives in cc_optimize/siblings.py, the one owner,
    loaded here by path because that is the only route guaranteed under either shape.
    """
    if not _SIBLINGS:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "cc_optimize_siblings",
            str(Path(__file__).resolve().parent.parent / "cc_optimize" / "siblings.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _SIBLINGS.append(_mod)
    mod = _SIBLINGS[0].load(name)
    if mod is None:
        raise ImportError("cc_optimize.%s is not reachable from %r" % (name, __package__))
    return mod


def _cool_before_remeasure():
    """Cool the way the full-pipeline gate does: to an ABSOLUTE target, with no deadline on physics.

    _await_cool below is relative (entry minus 5C) and capped at 120 s, which is fine as a courtesy
    pause between runs and useless after a clamp -- entry-5 on a 96C board asks for 91C, still hot
    enough to clamp again. One rule for "cool enough to measure", owned by perf_mcp, so the two
    measurement paths cannot drift apart again.
    """
    try:
        return _cc_optimize("perf_mcp")._cooldown_after_clamp()
    except Exception:  # noqa: BLE001 -- never let the cooldown import stop a run
        _await_cool()
        return True, None


_THERMAL_YIELD_MIN_GAP_S = 30.0
_thermal_yield_last = [0.0]


def thermal_yield(label: str = "") -> None:
    """Cool BETWEEN units of a long device run, at a point where nothing is in flight.

    THE GAP THIS CLOSES. Every other cooldown in this tool hangs off a boundary controlled from
    OUTSIDE the work: before a device subprocess is launched, after a run's log reports a clamp,
    after a reset. None of them can see inside a single call that holds the device for tens of
    minutes -- and that is exactly where the board cooks. Measured on this liquid-cooled p300c on
    2026-08-28: the launch gate last fired at 20:53, one call then held the device while all four
    chips sat at 98-103C, and at 21:43 chip 2 stopped answering MID-RUN. Its telemetry went straight
    to the 0xffffffff sentinel with no kernel message, no thermal trip and no PCIe error; the hottest
    chip followed. A cooldown that runs after the process exits could never have fired, because the
    card died before the exit.

    Offer this only BETWEEN units of work, never inside a timed region -- a trace replay's timing
    loop must not contain a sleep. Repeated offers inside the minimum gap return immediately, so a
    caller may offer the yield as often as it likes; the read itself is a ~0.3 ms sysfs file read.
    """
    try:
        now = time.monotonic()
        if now - _thermal_yield_last[0] < _THERMAL_YIELD_MIN_GAP_S:
            return
        _thermal_yield_last[0] = now
        _cc_optimize("perf_mcp").cool_if_over_safety_ceiling(label or "next unit")
    except Exception as exc:  # noqa: BLE001 -- a thermometer that cannot be read must never stop the work
        _warn_thermal_inert("thermal_yield", exc)
        return


def detect_overheat(log_text: str) -> str | None:
    """A run's log carrying the device's OWN thermal-distress signal (AICLK failed to settle / clamped /
    possible overheating). Returns the matched phrase, else None. Distinct from a crash: the run may
    complete, but the chip is throttling and the next run should let it cool first."""
    if not log_text:
        return None
    m = _DEVICE_OVERHEAT_RE.search(log_text)
    return m.group(0) if m else None


# The agent's own words when its credentials will not serve it. Plain substrings, not a pattern:
# these are the client's message text, and matching them literally is what keeps a rewording from
# silently turning the check off -- a miss must look like a miss, not like a clean run.
_AGENT_AUTH_FAILURES = (
    "Failed to authenticate",
    "Access Denied",
    "Invalid API key",
    "authentication_error",
    "OAuth token has expired",
)


# Being out of budget is not being unauthenticated: the credential is valid and renewing it changes
# nothing, so the two must not share a remedy. Plain substrings for the same reason as above.
_AGENT_EXHAUSTED = (
    "usage limit reached",
    "rate_limit_error",
    "exceeded your usage",
    "quota",
    "Credit balance is too low",
    "insufficient_quota",
)


def _first_phrase_in(log_text: str, phrases) -> str | None:
    """The first of `phrases` present in `log_text`, or None. Shared by the detectors below."""
    if not log_text:
        return None
    for phrase in phrases:
        if phrase in log_text:
            return phrase
    return None


def detect_quota_exhausted(log_text: str) -> str | None:
    """The agent was refused for want of BUDGET, not credentials.

    This is the failure auth handling gets wrong in the most expensive way. A rate or usage limit
    produces the same shape as a refused credential -- a round that runs, writes a transcript and
    does nothing -- but renewing is not the remedy: the credential is already valid, so a renewal
    "succeeds", the round is retried, and the retry is refused again. The recovery budget drains,
    the run is then reported as unable to authenticate, and the operator is sent to re-login over a
    problem that re-logging in cannot touch.

    Waiting is not this tool's call either: a reset can be hours away and the run holds the device
    the whole time. So this is a hard stop with the reason quoted, which is the one thing that lets
    an operator decide whether to wait or to raise the limit.
    """
    return _first_phrase_in(log_text, _AGENT_EXHAUSTED)


def detect_auth_failure(log_text: str) -> str | None:
    """The agent could not authenticate, so nothing it was asked to do can have happened.

    An expired or rejected credential produces a round that runs, writes a transcript and exits
    cleanly, having done nothing -- indistinguishable, to the loop, from an agent that looked and
    found no win. A credential that lapsed overnight therefore burned all ten rounds of a 7h37m run
    and the report said "no kernel attempts recorded", which reads as "the model is already optimal"
    rather than "nobody was allowed in".

    Returns the matched phrase so the caller can quote the agent verbatim instead of paraphrasing a
    cause it did not observe; None when the text carries no such failure.
    """
    return _first_phrase_in(log_text, _AGENT_AUTH_FAILURES)


def _max_asic_temp(data) -> float | None:
    temps: list[float] = []

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "asic_temperature":
                    try:
                        temps.append(float(v))
                    except (TypeError, ValueError):
                        pass
                else:
                    _walk(v)
        elif isinstance(o, list):
            for x in o:
                _walk(x)

    _walk(data)
    return max(temps) if temps else None


_SYSFS_HWMON = "/sys/class/hwmon"
# WHAT A DIE TEMPERATURE CAN BE. Blackhole throttles around 90C and shuts down well under 125, so
# 150 is far clear of any real reading while excluding the driver's all-ones "no data" value by a
# factor of four hundred. 0 excludes the other sentinel shape -- an all-zeros register reads as a
# perfectly plausible 0C and would drag a max-of-chips DOWN, which is the dangerous direction.
_DIE_TEMP_MIN_C, _DIE_TEMP_MAX_C = 0.0, 150.0
_TT_SMI_TEMP_TIMEOUT_S = float(os.environ.get("PERF_MCP_TT_SMI_TEMP_TIMEOUT_S", "15") or "15")
# How long tt-smi is left alone after it hangs, and when it last did. See _tt_smi_asic_temp.
_TT_SMI_BREAKER_S = 120.0
_TT_SMI_HUNG_AT = 0.0


def _sysfs_asic_temps() -> list:
    """Per-chip die temperature straight from the driver, in degrees C.

    THE KERNEL ALREADY PUBLISHES THIS. tt-smi shells out and OPENS THE DEVICE to answer a question the
    tenstorrent driver exposes as a file; measured on this host, sysfs is 0.0003 s against tt-smi's
    0.26 s, agrees to 0.2 C, and returns while a matmul is running (81.9 vs 82.1 C under load).

    Matched on the DRIVER, not the hwmon `name`. The name is the arch ("blackhole"), so keying on it
    would silently find zero chips on Wormhole and report no temperature at all -- the exact shape of
    failure this function exists to remove.

    A READ THAT SUCCEEDS IS NOT A READING. The temperature comes from each chip's ARC management
    core; when the ARC is not running the driver still publishes the file, filled with all ones:

        cat temp1_input -> 76617      = 76.6C, a temperature
        cat temp1_input -> 65535999   = 65535.999C, "no data" wearing a temperature's units

    Identical file, identical format, identical successful read -- there is no error to catch and no
    validity flag to consult, so the VALUE is the only thing that can tell them apart. Unchecked, one
    such chip decided the whole board: _read_asic_temp takes the hottest chip, 65535 beats every real
    number, and on 2026-08-16 that made every thermal gate wait its full 900s and then measure hot,
    for a board whose live chips were sitting at 80C.

    Implausible chips are DROPPED, not clamped: a chip with no telemetry is a chip whose temperature
    nobody knows, and inventing one would be the same mistake in the other direction. Two dead ARCs
    is also a fault worth acting on in its own right -- it meant a leaked process tree was holding
    the board -- which is why they are counted and reported rather than quietly filtered.
    """
    out: list = []
    try:
        entries = sorted(Path(_SYSFS_HWMON).iterdir())
    except OSError:
        return out
    for h in entries:
        try:
            if (h / "device" / "driver").resolve().name != "tenstorrent":
                continue
            t = int((h / "temp1_input").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if not (_DIE_TEMP_MIN_C < t < _DIE_TEMP_MAX_C):
            _NO_TELEMETRY_CHIPS.add(str(h.name))
            continue
        _NO_TELEMETRY_CHIPS.discard(str(h.name))
        out.append(t)
    return out


# Chips whose sensor answered with something that is not a temperature. Not a cache -- a record, so
# the caller can say "this board has a chip with no telemetry" instead of silently measuring on the
# ones that still work. Populated by _sysfs_asic_temps on every read, so it is always current.
_NO_TELEMETRY_CHIPS: set = set()


def board_telemetry():
    """(live die temperatures, hwmon names that could not report) -- the recovery gate's whole input.

    A named seam, deliberately. The gate must be stubbable by a test WITHOUT stubbing
    _sysfs_asic_temps, because that function is itself under test elsewhere: patching it to silence
    the gate also silenced the tests that check how a sentinel is parsed. One accessor for one
    question keeps those independent.
    """
    live = list(_sysfs_asic_temps() or [])
    return live, chips_without_telemetry()


def chips_without_telemetry() -> list:
    """hwmon names whose last read was not a temperature. Empty when every chip answered.

    A chip in this list has a dead ARC: it is not merely unreadable, it is broken, and the run that
    finds one is running on a board that needs attention rather than a board that is a bit warm."""
    return sorted(_NO_TELEMETRY_CHIPS)


def _tt_smi_asic_temp():
    """The second opinion. Bounded, because tt-smi's failure mode is to HANG, not to answer.

    AND NOT RE-ASKED WHILE IT IS HANGING. A wedged ARC does not make tt-smi slow, it makes tt-smi
    never answer, so every call costs the full timeout and every caller pays it. Measured on a board
    whose ARC had failed to start: `tt-smi -s` never returned, the preflight suite went from 2.2min
    to 5.5min, and the tax lands on every probe in every round.

    So a timeout opens a breaker for _TT_SMI_BREAKER_S. That is only skipping calls that would have
    timed out and returned None anyway -- the reading is not being suppressed, it was never going to
    arrive. Thermal safety is unchanged: sysfs is still read every single time (it is the source that
    still works on a wedged board), the two sources are still max()'d, and a clean answer from tt-smi
    immediately closes the breaker again. The window is short next to how fast a die actually heats.
    """
    global _TT_SMI_HUNG_AT
    if _TT_SMI_HUNG_AT and time.time() - _TT_SMI_HUNG_AT < _TT_SMI_BREAKER_S:
        return None
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    try:
        proc = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=_TT_SMI_TEMP_TIMEOUT_S)
        temp = _max_asic_temp(json.loads(proc.stdout))
    except subprocess.TimeoutExpired:
        _TT_SMI_HUNG_AT = time.time()
        return None
    except Exception:  # noqa: BLE001
        return None
    _TT_SMI_HUNG_AT = 0.0
    return temp


def _read_asic_temp():
    """Hottest die across the chips, from BOTH sources, or None when neither answers.

    TWO SOURCES, AND THE HOTTER WINS. Either one can be unavailable -- sysfs needs a driver that
    registers hwmon, tt-smi needs to not be contending with a running profile -- and they fail
    independently, so asking both is how a reading survives either one being out. Taking the max is
    the safe direction: a source that says hot is evidence, a source that says nothing is not evidence
    of cool.

    BOTH ARE ASKED, ALWAYS. This said "the hotter wins" and did not do it: tt-smi was consulted only
    when sysfs found NO chips, so `max` ran across the four sysfs chips and never between the two
    sources. Measured: sysfs 60C with tt-smi at 90C returned 60C -- the hotter source losing outright,
    which is the exact direction the docstring calls unsafe. The test that guarded this passed only
    because its sysfs value happened to be the higher one, so it would have passed against code that
    ignored tt-smi entirely.

    Fallback was the wrong shape for a reason. A fallback trusts the first source to know when it has
    failed, and sysfs cannot: a dead ARC publishes 65535999 rather than an error, so sysfs "succeeds"
    and the second opinion is never sought -- precisely when it is most needed. Asking both costs one
    bounded subprocess and removes the case where a working source is never heard from.

    Both are bounds-checked. tt-smi reads the same ARC telemetry over a different path, so it can
    return the same sentinel, and a source is only evidence while its answer is possible.

    None means NEITHER produced a usable reading, and the caller must treat that as unknown rather
    than as a cool board -- see _wait_for_thermal_headroom.
    """

    def _usable(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if _DIE_TEMP_MIN_C < v < _DIE_TEMP_MAX_C else None

    # PER CHIP, not on the max. Checking only the maximum lets a single sentinel discard every live
    # chip with it -- the reading would go from "80.5C" to "unknown" because a neighbour is broken.
    # _sysfs_asic_temps already drops them at the parse; this holds for any caller that does not.
    vals = [
        v for v in ([_usable(x) for x in (_sysfs_asic_temps() or [])] + [_usable(_tt_smi_asic_temp())]) if v is not None
    ]
    return max(vals) if vals else None


def _await_cool(read_temp=_read_asic_temp, sleeper=time.sleep) -> None:
    """Idle-wait until the chip sheds heat, keeping the device OPEN (no reset, no close) -- passive
    cooling while it does no work. Target is RELATIVE (entry temp minus a margin) so there is no absolute
    magic threshold; best-effort -- returns immediately if temp is unreadable and never blocks past the
    max wait. Call only at a run boundary (device idle)."""
    entry = read_temp()
    if entry is None:
        return
    target = entry - _COOL_MARGIN_C
    waited = 0.0
    while waited < _COOL_MAX_S:
        sleeper(_COOL_POLL_S)
        waited += _COOL_POLL_S
        t = read_temp()
        if t is None or t <= target:
            return


def _execute(
    cmd: list[str],
    cwd: Path,
    env: dict,
    timeout_s: int,
    log_path: Path,
    stall_timeout_s: int = 600,
) -> int:
    """Run cmd with output streamed to log_path (live-tailable). Hang-proof:
    no pipes (a daemon child inheriting them cannot deadlock us), and the
    whole process GROUP is killed on timeout (so orphaned capture-release
    daemons die too). Returns the exit code. Injectable seam for tests.

    Two-tier watchdog. A fixed wall-clock kill cannot tell 'hung' from 'slow' —
    cold profiler-instrumented kernel compilation for a multi-chip mesh is slow
    but alive (CPU-busy, log still streaming), and a flat 30-min cap killed it
    mid-compile before a single op ran. So the watchdog gates on FORWARD PROGRESS,
    not elapsed time: kill only when the log has not grown AND the process group
    has burned ~no CPU for `stall_timeout_s` (a real stall/deadlock). `timeout_s`
    remains as a generous ABSOLUTE backstop against a pathological busy-spin."""
    _therm_label = "generated-test run"
    try:
        _run = _cc_optimize("run")
        _thermal_watch_new = _run._thermal_watch_new
        _thermal_watch_sample = _run._thermal_watch_sample
        _run._wait_for_thermal_headroom_before_device_work(_therm_label)
    except Exception as exc:  # noqa: BLE001 -- a gate that cannot run must never stop the work
        _warn_thermal_inert("_execute launch gate", exc)

        def _thermal_watch_new():
            return {}

        def _thermal_watch_sample(state, label):
            return None

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group
        )

        def _kill_and_raise(reason: str):
            _kill_tree(proc.pid)
            proc.wait()
            raise TracyHangError(f"tracy run {reason}; log: {log_path}") from None

        pgid = proc.pid
        start = time.monotonic()
        last_progress = start
        # PROGRESS, NOT ACTIVITY. See progress_signature: CPU is excluded, because a livelock has
        # plenty of it. The stack is sampled only every _STACK_EVERY_S -- py-spy costs a fraction of
        # a second and the counters catch the common case on their own.
        _watch = ProgressWatch(pgid, log_path, stall_timeout_s)
        _over_budget = [False]
        poll = 5.0
        _therm = _thermal_watch_new()
        while True:
            try:
                rc = proc.wait(timeout=poll)
                # A CLEAN EXIT MUST LEAVE NOTHING BEHIND. The leader is gone, so anything still in
                # its group is a daemon it spawned and did not reap -- tracy-capture, serve_wasm --
                # and those hold the device against the next run. Does not touch rc.
                _orphans = _reap_process_group(pgid)
                if _orphans:
                    print(
                        f"  [probes] reaped {len(_orphans)} orphaned profiler process(es) after the run "
                        f"exited: {_orphans}",
                        file=sys.stderr,
                        flush=True,
                    )
                return rc
            except subprocess.TimeoutExpired:
                pass
            _thermal_watch_sample(_therm, _therm_label)
            now = time.monotonic()
            try:
                size = log_path.stat().st_size
            except OSError:
                size = last_size
            if _watch.moved(now, last_progress, proc.pid):
                last_progress = now
            if stall_timeout_s and now - last_progress >= stall_timeout_s:
                _kill_and_raise(
                    f"made no forward progress for {stall_timeout_s}s -- no log growth, no syscalls, "
                    f"no bytes and an unchanged stack. CPU alone is not progress; a livelock has "
                    f"plenty of it. Process group killed"
                )
            # THE SAME RULE AS _run_device_proc: a clock does not get to call working code dead.
            #
            # This killed on elapsed time alone, with the stall check directly above it already
            # holding the answer -- log growth and process-group CPU. A profile that is still
            # emitting and still burning CPU is not hung, and ending it at a fixed number throws
            # away the work AND triggers a recovery, which is the part that has damaged this board.
            #
            # The budget is reported once and the run continues; what ends it is going quiet.
            if not _over_budget[0] and now - start >= timeout_s:
                _over_budget[0] = True
                print(
                    f"  [probes] profile is over its {int(timeout_s)}s budget and still making "
                    f"progress -- not killing it; the stall check decides, and the hard ceiling at "
                    f"{int(timeout_s * _HARD_CEILING_MULT)}s is behind that",
                    file=sys.stderr,
                    flush=True,
                )
            # THE CEILING BEHIND THE DETECTOR. The signature above catches a step that stops
            # progressing; it cannot catch one that genuinely re-executes work forever -- fresh
            # syscalls, a moving stack, no end. Nothing in this tool's own loops does that (they are
            # bounded by counters: rounds, restarts, regens, kv attempts), but a model's code can,
            # and this supervises model code.
            #
            # A MULTIPLE OF THE BUDGET, NOT A FIXED CLOCK. The 3-hour timer this replaces was set so
            # high it was useless, because firing wrongly killed the RUN. This raises the same
            # TracyHangError every other detection raises, which every caller already treats as a
            # failed attempt -- the perf-test loop regenerates, the supervisor restarts. Getting it
            # wrong costs one attempt, so it can be set low enough to matter.
            if timeout_s and now - start >= timeout_s * _HARD_CEILING_MULT:
                _kill_and_raise(
                    f"exceeded {int(timeout_s * _HARD_CEILING_MULT)}s -- {_HARD_CEILING_MULT}x its "
                    f"{int(timeout_s)}s budget. It was still moving, so this is the ceiling behind "
                    f"the stall detector, not a stall: the attempt is failed and may be retried"
                )


_CSV_STDOUT_RE = re.compile(r"OPs csv generated at:\s*(\S+ops_perf_results_\S+\.csv)")


def _validate_csv(path: Path, log_path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise TracyRunError(f"ops CSV missing/empty: {path}; log: {log_path}")
    with path.open() as _fh:
        header = _fh.readline()
        if not header.startswith("OP CODE"):
            raise TracyRunError(f"unexpected CSV header in {path}: {header[:60]!r}; log: {log_path}")
        # A valid header with ZERO data rows used to pass, which is exactly the upstream zero-row
        # condition: it becomes device_ms 0.0 and then reads as a 100% speedup. Treat it as a
        # measurement failure (the caller retries) rather than a measurement of nothing.
        if not any(ln.strip() for ln in _fh):
            raise TracyRunError(f"ops CSV has a header but NO op rows: {path}; log: {log_path}")


_NODE_ID_RE = re.compile(r"^[\w./\-]+\.py::[\w:\[\]\-.]+$")


def _dr_is_dead_board(line: str) -> bool:
    """device_recovery's own dead-board test, imported lazily so probes keeps working under a bare
    sys.path (the MCP client launches this package without the repo root on it)."""
    try:
        from .device_recovery import is_dead_board
    except Exception:  # noqa: BLE001
        return "0xffffffff" in line.lower() or "board should be reset" in line.lower()
    return is_dead_board(line)


def collect_cases(
    tt_metal_root: str | os.PathLike[str],
    perf_test: str,
    env: dict | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[str]:
    """List the test node ids pytest would collect for perf_test (no -k).

    Used to pick the DEFAULT case (the FIRST collected) when neither the user
    nor the sub-agent supplied one."""
    # -o addopts= : neutralize pytest.ini verbosity so collect prints FLAT
    # node ids (repo addopts include -v, which turns the listing into a tree).
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", perf_test, "--collect-only", "-q"]
    proc = runner(
        cmd, cwd=Path(tt_metal_root), env=env or dict(os.environ), capture_output=True, text=True, timeout=120
    )
    # A node id is `<file>.py::<test>[...]` and NOTHING else. "any line containing ::" also matched
    # the C++ stack frames pytest prints when COLLECTION ITSELF fails on a wedged device
    # (`E  1. tt::umd::TTDevice::is_pcie_hung(unsigned int, ...)`), and that frame then became the
    # "node id" -- tracy joins its argv into a /bin/sh string, so the parens died as
    # `Syntax error: "(" unexpected` and the whole run mis-reported as "profiler crashed".
    ids = [ln.strip() for ln in (proc.stdout or "").splitlines() if _NODE_ID_RE.match(ln.strip()) and "(" not in ln]
    # HOIST the dead-board line. When collection dies on a wedged board, `Read 0xffffffff over PCIe
    # ID N: the board should be reset` sits ~30 lines up -- above the C++ frames and the pydantic
    # warnings -- so both a positional tail and _salient_tail's last-N dropped the ONE string
    # device_recovery.is_dead_board/dead_chip_from_error look for. Without it recovery cannot see
    # that the card died, nor which chip died, so it guesses the target from `--devices` intent.
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = _salient_tail(out, n=6)
    dead = next((ln.strip() for ln in out.splitlines() if _dr_is_dead_board(ln)), None)
    if dead and dead not in tail:
        tail = dead + "\n" + tail
    return ids, tail


def first_case_param(node_id: str) -> str | None:
    """'path::test_fn[S128]' -> 'S128' (None when unparametrized)."""
    m = re.search(r"\[(.+)\]\s*$", node_id)
    return m.group(1) if m else None


_NODE_ID_CACHE: dict[str, tuple] = {}


def resolve_node_id(
    tt_metal_root: str | os.PathLike[str],
    perf_test: str,
    case: str | None = None,
    env: dict | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Resolve (perf_test, optional case hint) -> ONE exact pytest node id collected from the LIVE
    test file.

    Exact node ids ('path::test_fn[param]') select deterministically. This replaces `pytest -k
    <case>`, whose stored/guessed case string is substring-matched against pytest's auto-generated
    parametrize id — so it silently deselects (0 tests run -> empty capture -> mis-reported as
    'profiler crashed') the moment the guess drifts from the live id. Resolving an exact node id
    from the live test removes that whole failure class: a regenerated/renamed test re-collects
    (cache keyed by mtime), an unmatched/stale case hint self-heals to the first real node id, and a
    genuinely empty test raises a CLEAR PreflightError instead of a misleading crash."""
    key = str(perf_test)
    try:
        mtime = (Path(tt_metal_root) / perf_test).stat().st_mtime
    except OSError:
        mtime = None
    cached = _NODE_ID_CACHE.get(key)
    if cached and cached[0] == mtime and cached[2] == case:
        return cached[1]
    ids, tail = collect_cases(tt_metal_root, perf_test, env=env, runner=runner)
    if not ids:
        raise PreflightError(f"perf test collects no tests: {perf_test}\n{tail}")
    chosen = None
    if case:
        chosen = next((n for n in ids if n.endswith(f"[{case}]") or n.rsplit("::", 1)[-1] == case), None)
        if chosen is None:
            sub = [n for n in ids if case in n]
            chosen = sub[0] if len(sub) == 1 else None
    if chosen is None:
        chosen = ids[0]
    _NODE_ID_CACHE[key] = (mtime, chosen, case)
    return chosen


def preflight_collect(
    tt_metal_root: str | os.PathLike[str],
    perf_test: str,
    case: str | None,
    env: dict | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Verify the discovered perf test selects >=1 case BEFORE a long profile run.

    Catches the zero-selection trap ('5 deselected, 0 selected') in seconds.
    Returns the number of selected tests."""
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", perf_test, "--collect-only", "-q"]
    if case:
        cmd += ["-k", case]
    proc = runner(
        cmd, cwd=Path(tt_metal_root), env=env or dict(os.environ), capture_output=True, text=True, timeout=120
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    # "1/5 tests collected" must win over the bare form ("5 tests collected"
    # is a substring of it and reports the WRONG number when -k deselects).
    m = re.search(r"(\d+)/\d+ tests collected", out) or re.search(r"(\d+)\s+tests? collected", out)
    n = int(m.group(1)) if m else 0
    if proc.returncode != 0 or n == 0:
        tail = "\n".join(out.splitlines()[-8:])
        raise PreflightError(
            f"perf test selects no cases: pytest {perf_test} -k {case!r} "
            f"(exit {proc.returncode}, {n} collected)\n{tail}"
        )
    return n


def make_run_profiled(
    tt_metal_root: str | os.PathLike[str],
    perf_test: str,
    case: str | None = None,
    timeout_s: int = 10800,
    execute: Callable[..., int] = _execute,
    extra_env: dict[str, str] | None = None,  # e.g. TT_METAL_VISIBLE_DEVICES
    collect_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    retries: int = 2,
    device_reset: Callable[[], bool] = _device_reset,
) -> Callable[..., tuple[Path, float]]:
    """Factory for tracy_tool's stage-1 `run_profiled` (real hardware).

    Extraction, three layers: (1) glob OUR -o dir; (2) 'OPs csv generated at:'
    regex from the log as cross-check; (3) watermark glob of the shared
    generated/profiler area as fallback. Winner is validated then archived
    into profiles_dir (generated/ and tracy_out/ are overwritten by later runs).
    wall_ms is the harness clock — interim until TBD(wall-metric-source)."""
    root = Path(tt_metal_root)

    def run_profiled(
        pcc_path: str, batch_size: int, seq_len: int, profiles_dir: str | Path, i: int
    ) -> tuple[Path, float]:
        profiles_dir = Path(profiles_dir)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        out_dir = profiles_dir / "tracy_out"
        log_path = profiles_dir / f"run{i}_tracy.log"
        env = dict(os.environ)
        env["TT_METAL_DEVICE_PROFILER"] = PROFILING_ENV["TT_METAL_DEVICE_PROFILER"]
        # EAGER-ONLY under tracy. tracy profiles per-op device time from eager dispatch; a trace replay
        # runs as one fused program that emits NO per-op device data, so profiling it just floods tracy's
        # post-processor with one "device data missing" warning per traced op (~180k for a whole pipeline)
        # -- slow enough that the no-output watchdog false-kills the run. The end-to-end trace_replay
        # verdict is measured SEPARATELY (check_full_pipeline_latency, profiler off), so the tracy run
        # needs only device_ms. Keep the two apart; the redundant per-token this used to scrape is None-safe.
        env["TT_PERF_TRACE"] = PROFILING_ENV["TT_PERF_TRACE"]
        env.update(extra_env or {})
        _prof = os.environ.get("PERF_MCP_PROFILE_ENV")
        if _prof:
            try:
                env.update(json.loads(_prof))
            except (ValueError, TypeError):
                pass
        try:
            from .profiler_heal import ensure_profiler_patched

            ensure_profiler_patched(root)
        except Exception:
            pass
        # COOL FIRST. This launches the profiled run, and a profiled run BUILDS THE MODEL -- weights
        # plus every graduated stub -- before it takes a single sample. On a 3B multimodal pipeline
        # that is tens of minutes of device work, and doing it on a board still hot from the last
        # attempt is how a profile ends up sampled entirely at the 800 MHz clamp. The gate lives in
        # one place (perf_mcp) and is called from the two points where device work STARTS.
        try:
            _cc_optimize("run")._wait_for_thermal_headroom_before_device_work("profiled run")
        except Exception as exc:  # noqa: BLE001 -- never let the gate stop the run
            _warn_thermal_inert("make_run_profiled", exc)
        node_id = resolve_node_id(root, perf_test, case, env=env, runner=collect_runner)
        cmd = build_tracy_command(node_id, None, out_dir)
        support_count = int(env.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or 0)
        t_start = time.monotonic()
        partial_reason = None
        heal_attempt = 0
        throttle_retry = 0
        while True:
            if support_count > 0:
                env["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = str(support_count)
            for _attempt in range(retries + 1):
                watermark = time.time() - 0.05
                try:
                    code = execute(cmd, root, env, timeout_s, log_path)
                    break
                except TracyHangError:
                    if _attempt >= retries:
                        raise
                    device_reset()
            if code != 0:
                tail = _salient_tail(log_path.read_text()) if log_path.is_file() else ""
                raise TracyRunError(f"tracy run exit {code} (log: {log_path})\n{tail}")
            log_text = log_path.read_text() if log_path.is_file() else ""
            if detect_overheat(log_text):
                # DISCARD AND RE-MEASURE. This used to cool and then KEEP the number, which is the
                # worst of both: it paid for the wait and still banked a reading taken at 800 MHz
                # instead of 1350. The full-pipeline gate has always thrown these away and retried
                # (perf_mcp._measure_full_pipeline_guarded); the path that measures every candidate
                # -- the one whose numbers decide what gets committed -- did not.
                if throttle_retry < _MAX_THROTTLE_RETRIES:
                    throttle_retry += 1
                    _cooled, _at = _cool_before_remeasure()
                    with open(log_path, "a") as fh:
                        fh.write(
                            f"\n[harness] device throttled during this run; reading DISCARDED, "
                            f"re-profiling from {_at if _at is not None else 'an unknown temperature'} "
                            f"(attempt {throttle_retry}/{_MAX_THROTTLE_RETRIES})\n"
                        )
                    if _cooled:
                        continue
                    raise ThrottledRunError(
                        "device throttled and the board stopped cooling at "
                        f"{_at if _at is not None else 'an unknown temperature'}; "
                        "no reading can be taken at full clock right now"
                    )
                raise ThrottledRunError(
                    f"device throttled on all {throttle_retry + 1} attempts; every reading was taken "
                    "at a clamped clock and none is comparable to the baseline"
                )
            # `python -m tracy -m pytest` exits 0 even when the inner test FAILS, so a device-op
            # crash (the edit broke the model) leaves a PARTIAL CSV that would be misread as an
            # op_count_mismatch measurement. Detect the runtime crash here and raise PerfRunFailed
            # (carries the error) so REMEASURE routes it to REPAIR_CODE and the agent fixes its edit.
            crash = detect_perf_crash(log_text)
            if crash:
                if _DEVICE_CRASH_RE.search(log_text) and heal_attempt < _MAX_HEAL_ATTEMPTS:
                    heal_attempt += 1
                    _await_cool()
                    device_reset()
                    with open(log_path, "a") as fh:
                        fh.write(
                            f"\n[harness] device crash ({crash}); reset + re-profile "
                            f"(heal {heal_attempt}/{_MAX_HEAL_ATTEMPTS})\n"
                        )
                    continue
                raise PerfRunFailed(crash, log_path)
            drop = detect_marker_drop(log_text)
            if drop and support_count < _MAX_PROFILER_SUPPORT_COUNT and heal_attempt < _MAX_HEAL_ATTEMPTS:
                heal_attempt += 1
                support_count = min(max(support_count, 1000) * _HEAL_GROWTH, _MAX_PROFILER_SUPPORT_COUNT)
                with open(log_path, "a") as fh:
                    fh.write(
                        f"\n[harness] profiler buffer grew to TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="
                        f"{support_count}; re-profiling (heal {heal_attempt}/{_MAX_HEAL_ATTEMPTS})\n"
                    )
                continue
            partial_reason = drop
            break
        wall_ms = (time.monotonic() - t_start) * 1000.0

        # layer 1: directed output (-o). out_dir PERSISTS across iterations, so a PRIOR
        # run's CSV is still sitting here -- filter to THIS run (mtime > watermark) or the
        # glob can return the stale baseline. That stale-CSV reuse made every REMEASURE
        # re-read the baseline, so real edits measured identical to baseline and were
        # wrongly flagged inert/no-gain and reverted (the "zero gains" root cause).
        found = sorted(
            (p for p in out_dir.glob("**/ops_perf_results_*.csv") if p.stat().st_mtime > watermark),
            key=lambda p: p.stat().st_mtime,
        )
        # layer 2: the stdout path is AUTHORITATIVE -- tracy logs the exact CSV it wrote for
        # THIS run ("OPs csv generated at: <path>"). Trust it over the glob, which can tie or
        # pick a touched older dir. Previously this only WARNED on a mismatch and kept the
        # (stale) glob result; now the reported path wins whenever it exists.
        log_text = log_path.read_text() if log_path.is_file() else ""
        m = _CSV_STDOUT_RE.search(log_text)
        if m:
            reported = Path(m.group(1))
            if reported.is_file():
                if found and reported.resolve() != found[-1].resolve():
                    with open(log_path, "a") as fh:
                        fh.write(f"\n[harness] using authoritative stdout CSV {reported} over glob {found[-1]}\n")
                found = [reported]
        # layer 3: watermark fallback in the shared area
        if not found:
            found = sorted(
                (p for p in root.glob("generated/profiler/**/ops_perf_results_*.csv") if p.stat().st_mtime > watermark),
                key=lambda p: p.stat().st_mtime,
            )
        if not found:
            raise TracyRunError(
                f"no ops_perf_results_*.csv produced (checked {out_dir}, stdout, "
                f"generated/profiler); log: {log_path}"
            )
        newest = found[-1]
        _validate_csv(newest, log_path)
        dest = profiles_dir / f"run{i}_raw.csv"
        shutil.copyfile(newest, dest)
        if partial_reason:
            try:
                (profiles_dir / f"run{i}.partial").write_text(str(partial_reason))
            except Exception:
                pass
        return dest, wall_ms

    return run_profiled


# ---------------------------------------------------------------------------
# Discovery review gate — the LEAD approves what the sub-agent gathered
# ---------------------------------------------------------------------------


class DiscoveryRejected(Exception):
    """The lead agent reviewed the discovery evidence and stopped the run."""


REVIEW_PROMPT = (
    "You are the lead optimization agent. A discovery sub-agent explored a model "
    "directory and returned the findings below (already form-validated: all "
    "paths exist, the perf case selects tests). YOUR decision: is this a sound "
    "basis to start a profiling/optimization run?\n\n"
    "Findings:\n{findings}\n\n"
    "Consider: does the end-to-end entry truly look like a full-model "
    "correctness check (not a unit test)? Is the perf test appropriate to "
    "profile? Is the extracted end-to-end PCC threshold plausible as a "
    "correctness gate (not a loose debug value)? Do the warnings change "
    "anything?\n"
    "The perf test may be AUTO-GENERATED by the harness from the specified PCC "
    "gate when the repository ships no dedicated perf test; this is a supported, "
    "expected mode. When the findings' perf_test entry is populated (even if "
    "marked auto-generated), that is the artifact that will actually be profiled "
    "— judge THAT, on the hardware this harness is currently running on. Warnings "
    "may describe OTHER, pre-existing tests in the repository that this run will "
    "NOT execute (e.g. a reduced-scope micro-benchmark that disables features, or "
    "a test written for a different accelerator). Do NOT stop merely because such "
    "an unused test exists, targets another platform, or disables features, and do "
    "NOT stop merely because no hand-written perf test was present. Base the "
    "decision on the resolved perf_test and pcc entries themselves.\n"
    "Judge ONLY the soundness of the resolved perf_test + PCC gate. Do NOT stop because the component "
    "looks already-optimized, was optimized in a prior run, is 'at terminal / at the floor', or because a "
    "re-run seems redundant or wasteful — WHETHER to (re)optimize is the operator's decision (they launched "
    "this run), NOT yours. IGNORE any memory, notes, prior-run state, or git history about earlier "
    "optimization; a freshly-reset/clean-slate component MUST still be allowed to run. Your ONLY grounds to "
    "stop are a genuine perf-test / PCC-gate SOUNDNESS blocker.\n"
    # THE DECISION ON ITS OWN LINE, not prose inside a JSON string. Asking a model for
    # {{"reasoning": <2-3 sentences>}} puts free text where a raw newline is illegal, and a long
    # enough answer eventually wraps: run 9 lost a sound verdict to "Invalid control character at
    # char 1028". A leading token cannot break that way, and the reasoning can then be any shape.
    # JSON stays accepted by the parser, because a format instruction is a request, not a guarantee.
    "Answer with the DECISION ON ITS OWN LINE, then the reasoning. Line breaks in the reasoning are "
    "fine:\n"
    "DECISION: continue|stop\n"
    "REASON: <2-3 sentences>\n"
    "Stop only for genuine soundness blockers — warnings with a sensible fallback are acceptable. "
    "(A JSON object with the same two fields is also accepted.)"
)


# The DECISION line, decorated as models decorate it, EXCLUDING the spec quoted back
# (`DECISION: continue|stop`). Shared, so the parser and the complaint below can never disagree
# about what counts as an answer -- a retry that scolds the reviewer for something the parser would
# have accepted is worse than no retry at all.
_VERDICT_TOKEN_RE = re.compile(r"^[\s>*_#-]*DECISION[\s*_]*[:=]?[\s*_]*(continue|stop)\b(?!\s*[|/])", re.I | re.M)
_VERDICT_SPEC_RE = re.compile(r"^[\s>*_#-]*DECISION[\s*_]*[:=]?[\s*_]*(?:continue|stop)\s*[|/]", re.I | re.M)


def review_verdict_complaint(raw: str) -> str:
    """Why that reply could not be read as a verdict, addressed TO the reviewer.

    A retry that just repeats the question invites the same answer. This names the specific defect
    and quotes the reply back, so the second attempt is a correction rather than a coin toss -- the
    reviewer is a language model, and a model told exactly what it got wrong generally fixes it.
    """
    text = (raw or "").strip()
    if not text:
        return "Your previous reply was EMPTY -- no text at all came back."
    _hits = {m.group(1).lower() for m in _VERDICT_TOKEN_RE.finditer(text)}
    if len(_hits) > 1:
        return (
            "Your previous reply stated BOTH decisions (%s), so it decided nothing. Pick exactly "
            "one and state it once." % ", ".join(sorted(_hits))
        )
    if _VERDICT_SPEC_RE.search(text):
        return (
            "Your previous reply quoted the FORMAT back -- a literal `DECISION: continue|stop` line "
            "with the bar in it. That is the template, not an answer. Write ONE of the two words."
        )
    if re.search(r"\b(continue|stop|proceed|reject)\b", text, re.I):
        return (
            "Your previous reply discussed the decision in prose but never put it on a `DECISION:` "
            "line, so it could not be read. The verdict must be on its OWN line, as the FIRST thing."
        )
    return "Your previous reply contained no `DECISION:` line and no JSON `decision` field."


def parse_review_verdict(text: str):
    """(decision, reasoning) from a lead-review reply, or (None, raw) when it states neither.

    THE FORMAT IS THE FIX. This asked for `{"decision": ..., "reasoning": <2-3 sentences>}` and then
    parsed it strictly -- free prose inside a JSON string, scraped out of raw stdout. A model writing
    a kilobyte of reasoning eventually presses enter mid-sentence, and a literal newline inside a
    JSON string is invalid JSON. Run 9, 2026-08-17: "Invalid control character at char 1028", a
    perfectly sound verdict discarded, and the run refused for it. The prompt has asked for that
    shape since 2026-06-27; it only needed a long enough answer to break.

    A leading token cannot break that way, because the prose is no longer inside a quoted string:

        DECISION: continue
        REASON: ...any length, any number of lines...

    JSON IS STILL ACCEPTED, with strict=False, because the reviewer is a language model and a format
    instruction is a request rather than a guarantee -- and because an older harness may be talking
    to a newer prompt or the reverse. Both shapes are read; neither is required.
    """
    import re as _re

    raw = (text or "").strip()
    if not raw:
        return None, ""
    # A model asked for a bare format still fences it about half the time; the fence is not content.
    raw = _re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()
    # 1. THE TOKEN FORM, anywhere in the reply -- a model often greets before it answers, and often
    # decorates: `**DECISION:** stop`, `- DECISION: stop`, `DECISION stop`. The separator and the
    # emphasis are noise; the pair of words is the signal.
    #
    # THE SPEC IS NOT A VERDICT. `DECISION: continue|stop` is the instruction quoted back, and a
    # naive match reads its first alternative as an answer -- handing an automatic "continue" to any
    # model that restates the format before answering. It is excluded explicitly.
    #
    # TWO DIFFERENT ANSWERS ARE NO ANSWER. A reply that says continue and later stop has not decided;
    # picking the first (or the last) invents a verdict from ordering. That is UNKNOWN, which the
    # caller handles, rather than a coin toss on whether to stop the run.
    _hits = [(m, m.group(1).lower()) for m in _VERDICT_TOKEN_RE.finditer(raw)]
    if _hits:
        _vals = {v for _m, v in _hits}
        if len(_vals) > 1:
            return None, raw
        m = _hits[0][0]
        why = raw[m.end() :]
        r = _re.search(r"^[\s>*_#-]*REASON(?:ING)?[\s*_]*[:=]?[\s*_]*", why, _re.I | _re.M)
        return _hits[0][1], (why[r.end() :] if r else why).strip(" \n*_#>-")
    # 2. the JSON form, tolerant of the raw newlines that broke it
    try:
        obj = json.loads(_extract_json_object(raw), strict=False)
        d = str(obj.get("decision", "")).strip().lower()
        if d in ("continue", "stop"):
            return d, str(obj.get("reasoning", "")).strip()
    except Exception:  # noqa: BLE001 -- neither shape; the caller decides what silence means
        pass
    return None, raw


def cli_lead_review_gate(
    pathmap: dict[str, Any],
    max_turns: int = 4,
) -> dict[str, Any]:
    """CC-native lead review: same go/no-go decision as lead_review_gate but via the `claude` CLI
    (no SDK, no model tier, no tools). Raises DiscoveryRejected on stop; returns the verdict on continue."""
    findings = json.dumps({k: pathmap[k] for k in ("perf_test", "pcc", "components", "summary", "warnings")}, indent=1)
    prompt = REVIEW_PROMPT.format(findings=findings)
    if os.environ.get("TT_PERF_MODULE_LEVEL", "") not in ("", "0", "false", "False"):
        prompt += (
            "\n\nMODULE-LEVEL RUN (--module-level): this is a SINGLE-COMPONENT optimization. The perf test "
            "times ONE module in isolation and the correctness gate is DELIBERATELY that module's OWN "
            "per-component PCC test (a unit-level PCC >= its target), NOT a full-model end-to-end check. A "
            "whole-pipeline / end-to-end gate is NOT expected or required here — the per-component PCC test "
            "IS the correct and sufficient correctness signal for the single module being optimized. Do NOT "
            "stop for 'the gate is only a per-component/unit test' or 'no correctness signal for the other "
            "stages'; judge ONLY whether the per-component perf test and its per-component PCC gate are sound "
            "for that one module."
        )
    env = dict(os.environ)
    for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(_k, None)
    from .agent_bin import resolve_claude_bin

    # ASK AGAIN BEFORE THROWING THE RUN AWAY.
    #
    # The reviewer was asked ONCE, and an unusable answer failed the whole discovery. The retry then
    # lived only at the run level: regenerate every perf test, re-survey the model -- fifteen to
    # twenty minutes -- and only then ask a second time. Run 9, 2026-08-17, spent exactly that on a
    # reply whose only fault was a line break, and the attempt after it passed because the next
    # answer happened to fit on one line. That is luck, bought at twenty minutes.
    #
    # A reviewer that garbles one answer is not a reviewer that cannot answer, so the cheap question
    # comes first: ask again in place, seconds, nothing regenerated. Only if it will not state a
    # decision across ALL these attempts is the discovery rejected -- and that rejection still
    # regenerates and re-asks at the run level, so an unreadable reviewer is bounded twice over
    # instead of costing a rebuild per question.
    #
    # AND THE RETRY CARRIES THE COMPLAINT. Repeating an identical prompt to a model that just
    # misread it mostly buys an identical answer; naming the defect and quoting the reply back is
    # what makes the second attempt a correction. The complaint comes from the same regexes the
    # parser uses, so it can never scold the reviewer for something the parser would have accepted.
    _tries = max(1, int(os.environ.get("PERF_MCP_REVIEW_TRIES", "5") or "5"))
    _ask, decision, reasoning = prompt, None, ""
    for _attempt in range(1, _tries + 1):
        r = subprocess.run(
            [
                resolve_claude_bin(),
                "-p",
                _ask,
                "--output-format",
                "text",
                "--system-prompt",
                "You make go/no-go calls for an automated perf-optimization harness.",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if r.returncode != 0:
            raise DiscoveryRejected(f"cc lead review (claude CLI) exit {r.returncode}: {(r.stderr or '')[-200:]}")
        # A VERDICT THAT WILL NOT PARSE IS UNKNOWN, NOT A REFUSAL, and not an approval either.
        #
        # strict=False in the parser, because the failure was never about the DECISION. Run 9:
        #
        #     lead review returned unparseable verdict:
        #     Invalid control character at: line 1 column 1029 (char 1028)
        #
        # The answer WAS json; it carried a literal newline inside a text field, which json.loads
        # rejects by default. The review may well have said continue -- nobody knew, because the
        # text went out with the exception, the supervisor restarted the run, and the retry passed
        # only because the agent happened to phrase itself without a stray newline the second time.
        #
        # "I could not read the verdict" and "the plan is rejected" are different states, and
        # conflating them turned a formatting glitch into a stopped run. So an unreadable answer is
        # neither: it is a question worth asking again. A genuine `stop` still stops, and a non-zero
        # exit above still refuses.
        decision, reasoning = parse_review_verdict(r.stdout or "")
        if decision is not None:
            break
        _why = review_verdict_complaint(r.stdout or "")
        print(
            "  [probes] lead review stated no decision (attempt %d/%d): %s%s"
            % (_attempt, _tries, _why, " Asking again." if _attempt < _tries else " Out of attempts."),
            file=sys.stderr,
            flush=True,
        )
        if _attempt == _tries:
            break
        _ask = (
            prompt
            + "\n\nYOUR PREVIOUS ANSWER COULD NOT BE READ, so the question stands unanswered.\n"
            + _why
            + "\n\nThis is what you sent (verbatim, truncated):\n"
            + (r.stdout or "").strip()[:1500]
            + "\n\nAnswer again. The FIRST line must be exactly `DECISION: continue` or `DECISION: stop` "
            "-- one word, no bar, no brackets, no emphasis, nothing before it. Put your reasoning after "
            "it on a `REASON:` line, where it may run to any length. Judge the findings themselves; do "
            "not stop merely because this is a retry."
        )
    if decision is None:
        # NO VERDICT MEANS ASK AGAIN, NOT PROCEED -- and by here, it has been asked again, with the
        # defect named each time. Continuing now would run a plan nobody approved, silently, and make
        # the gate bypassable by any reply that failed to state a decision.
        #
        # The parse is already generous -- token form and JSON, decorated or fenced, newlines and all
        # -- so a reply that survives every attempt stated nothing usable, or stated two different
        # things. Neither is approval.
        #
        # THE COST, STATED: a systematically unreadable reviewer exhausts these attempts, then the
        # supervisor's, and stops the run. That is the right outcome -- never getting a verdict is
        # not "carry on".
        raise DiscoveryRejected(
            "lead review stated no decision in %d attempts: %s (reply began: %s)"
            % (_tries, review_verdict_complaint(reasoning or ""), (reasoning or "")[:200].replace("\n", " "))
        )
    if decision == "stop":
        raise DiscoveryRejected(f"cc lead agent stopped the run: {reasoning}")
    return {"decision": decision, "reasoning": reasoning, "model": "claude-cli", "usage": None}


# ---------------------------------------------------------------------------
# Human-readable input -> test case matching (--input 128 / --input 128x128)
# ---------------------------------------------------------------------------


class InputMatchError(Exception):
    """The requested input matches zero (or several) discovered test cases."""


def match_input_to_case(user_input: str, params: list[str]) -> str:
    """Map a human input spec onto EXACTLY ONE parametrize id, else raise.

    Supported (deliberately just these two for now):
      "128"     -> sequence-length style: matches params whose embedded integer
                   tokens include 128 (S128, seq128, 128 all match; S1024 not).
      "128x128" -> image-size style: matches params containing that exact
                   normalized NxM string.
    Zero matches -> stop (the S512 lesson: never run something the user didn't
    ask for). Multiple matches -> stop and demand the raw pytest id via -k.
    """
    spec = user_input.strip().lower()
    image = re.fullmatch(r"(\d+)x(\d+)", spec)
    matched = []
    for p in params:
        if p is None:
            continue
        if image:
            if spec in p.lower():
                matched.append(p)
        else:
            if spec.isdigit() and int(spec) in [int(n) for n in re.findall(r"\d+", p)]:
                matched.append(p)
    if not matched:
        raise InputMatchError(
            f"input {user_input!r} matches NO test case. Available cases: "
            f"{[p for p in params if p]} — pick one of these dimensions, or use -k."
        )
    if len(matched) > 1:
        raise InputMatchError(
            f"input {user_input!r} is ambiguous — matches {matched}. " f"Use -k with the exact case id."
        )
    return matched[0]


def resolve_signposts(tests_dir, default_start="start", default_end="stop"):
    """Resolve tracy start/end signpost names by scanning <model_root>/tests/.

    Scoped to tests/ ONLY (perf + pcc tests both live there). Captures the first
    arg of signpost(...) and keeps it only if it is a string literal; constant /
    expression args (e.g. signpost(WARMUP_SIGNPOST)) are skipped and surface via
    the warning. Returns {"start_signpost","end_signpost","found":[...],"warning"}.

    Fallback is the conventional "start"/"stop" even when none are found:
    tt-perf-report tolerates absent signpost names (full capture) -- the
    proven-working behavior. NEVER fall back to no-signpost (None truncates).
    """
    import re
    from pathlib import Path

    call = re.compile(r"signpost\(\s*(?:header\s*=\s*)?([^)]*)\)")
    found = []
    tdir = Path(tests_dir)
    if tdir.is_dir():
        for py in sorted(tdir.rglob("*.py")):
            try:
                text = py.read_text(errors="ignore")
            except OSError:
                continue
            for arg in call.findall(text):
                arg = arg.strip().split(",")[0].strip()
                if len(arg) >= 2 and arg[0] in "\"'" and arg[-1] == arg[0]:
                    found.append(arg[1:-1])
    found = sorted(set(found))
    warning = None
    if "start" in found and "stop" in found:
        start, end = "start", "stop"
    elif not found:
        start, end = default_start, default_end
        warning = "no tracy signposts in %s/ -- using default %r/%r (full capture)" % (tdir, start, end)
    else:
        start, end = default_start, default_end
        warning = (
            "custom signposts %s but no 'start'/'stop' -- using default %r/%r; set start_signpost/end_signpost to override"
            % (found, start, end)
        )
    return {"start_signpost": start, "end_signpost": end, "found": found, "warning": warning}
