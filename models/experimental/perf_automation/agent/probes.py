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
import time
from pathlib import Path
from typing import Any, Callable

from .environment import EnvironmentError_

# ---------------------------------------------------------------------------
# 7.1 environment probe — `tt-smi -s` (TBD(env-script): CLOSED)
# ---------------------------------------------------------------------------

# board_type prefix -> arch token understood by environment.ARCH_FACTS
BOARD_ARCH_PREFIXES: tuple[tuple[str, str], ...] = (
    ("n150", "wormhole"),
    ("n300", "wormhole"),
    ("galaxy", "wormhole"),
    ("p100", "blackhole"),
    ("p150", "blackhole"),
    ("p300", "blackhole"),
)


def board_to_arch(board_type: str) -> str | None:
    b = (board_type or "").strip().lower()
    for prefix, arch in BOARD_ARCH_PREFIXES:
        if b.startswith(prefix):
            return arch
    return None


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


def sdk_model_files_runner(
    env_agent_path: str | os.PathLike[str] = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 24,
) -> Callable[[str], str]:
    """Build the production runner for read_model_files (PLAN section 7.3).

    Native Anthropic auth: ANTHROPIC_API_KEY if exported, else `claude` login. No creds file
    required, so .env.agent is optional (it only supplies model/effort overrides when present).
    """
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("sub", resolved)

    def runner(prompt: str) -> str:
        pass

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            model=model,
            system_prompt=(
                "You map model source trees for performance tooling. Use ONLY the "
                "read-only tools provided (Read, Glob, Grep). Your FINAL message "
                "must be exactly one JSON object — no prose, no code fences."
            ),
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
            max_buffer_size=50 * 1024 * 1024,
        )
        chunks: list[str] = []

        async def _go() -> None:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
                elif isinstance(msg, ResultMessage):
                    runner.last_usage = _usage_summary(msg)

        from .sdk_retry import run_with_retry

        run_with_retry(_go, lambda: chunks.clear())
        return _extract_json_object("\n".join(chunks))

    runner.last_usage = None
    runner.model = model

    return runner


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


def build_tracy_command(perf_test: str, case: str | None, out_dir: str | Path) -> list[str]:
    """The raw profile_this command (C++ post-processing default) + -o.

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -o <out> -m pytest ... -sv
    Run directly (never via profile_this.py: it swallows the exit code).
    """
    cmd = ["python", "-m", "tracy", "-v", "-r", "-p", "-o", str(out_dir), "-m", "pytest", "-o", "timeout=0", perf_test]
    if case:
        cmd += ["-k", case]
    cmd += ["-sv"]
    return cmd


def _pgroup_cpu_jiffies(pgid: int) -> int:
    """Sum utime+stime (jiffies) over all live PIDs in process group `pgid`, from /proc.
    Liveness signal: a process doing real work (e.g. compiling kernels) keeps accruing CPU;
    a hung/deadlocked one blocked on a lock or I/O accrues ~none. Best-effort; 0 on any error."""
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


def _descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
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
        if len(fields) > 1:
            children.setdefault(int(fields[1]), []).append(int(entry))
    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        for c in children.get(pid, []):
            out.append(c)
            stack.append(c)
    return out


def _kill_tree(root_pid: int) -> None:
    import signal

    pids = _descendant_pids(root_pid) + [root_pid]
    pgids = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
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


def _reset_arg_sets() -> list[list[str]]:
    """The tt-smi reset invocations to try, in order, for THIS host. An explicit override wins; else a
    Galaxy host uses the galaxy-tray reset (auto-retry first) with the plain reset as a last-ditch
    fallback, and a non-Galaxy host uses the plain per-device reset."""
    override = os.environ.get("TT_HW_PLANNER_RESET_ARGS")
    if override:
        return [override.split()]
    galaxy = _GALAXY_HOST
    if galaxy is None:
        galaxy = os.environ.get("TT_HW_PLANNER_GALAXY", "").strip().lower() in ("1", "true", "yes")
    if galaxy:
        return [["-glx_reset_auto"], ["-glx_reset"], ["-r"]]
    return [["-r"]]


def _device_reset() -> bool:
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    for args in _reset_arg_sets():
        try:
            proc = subprocess.run([tt_smi, *args], capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False


_DEVICE_OVERHEAT_RE = re.compile(r"Waiting for AICLK value to settle failed|possible overheating|AICLK clamped")
_COOL_MARGIN_C = float(os.environ.get("PERF_MCP_COOL_MARGIN_C", "5") or "5")
_COOL_POLL_S = float(os.environ.get("PERF_MCP_COOL_POLL_S", "5") or "5")
_COOL_MAX_S = float(os.environ.get("PERF_MCP_COOL_MAX_S", "120") or "120")


def detect_overheat(log_text: str) -> str | None:
    """A run's log carrying the device's OWN thermal-distress signal (AICLK failed to settle / clamped /
    possible overheating). Returns the matched phrase, else None. Distinct from a crash: the run may
    complete, but the chip is throttling and the next run should let it cool first."""
    if not log_text:
        return None
    m = _DEVICE_OVERHEAT_RE.search(log_text)
    return m.group(0) if m else None


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


def _read_asic_temp():
    """Max ASIC temperature (deg C) across chips from `tt-smi -s`, or None if unavailable. Only safe at a
    run boundary (device idle) -- tt-smi contends with an active profiler run."""
    tt_smi = shutil.which("tt-smi") or "/home/ttuser/.tenstorrent-venv/bin/tt-smi"
    try:
        proc = subprocess.run([tt_smi, "-s"], capture_output=True, text=True, timeout=30)
        return _max_asic_temp(json.loads(proc.stdout))
    except Exception:  # noqa: BLE001
        return None


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
        last_size = -1
        last_cpu = _pgroup_cpu_jiffies(pgid)
        poll = 5.0
        while True:
            try:
                return proc.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            try:
                size = log_path.stat().st_size
            except OSError:
                size = last_size
            cpu = _pgroup_cpu_jiffies(pgid)
            if size > last_size or cpu > last_cpu + 10:
                last_progress = now
            last_size, last_cpu = size, cpu
            if stall_timeout_s and now - last_progress >= stall_timeout_s:
                _kill_and_raise(
                    f"made no forward progress for {stall_timeout_s}s "
                    f"(stalled/hung: no log growth and ~no CPU) — process group killed"
                )
            if now - start >= timeout_s:
                _kill_and_raise(f"exceeded absolute backstop of {timeout_s}s (process group killed)")


_CSV_STDOUT_RE = re.compile(r"OPs csv generated at:\s*(\S+ops_perf_results_\S+\.csv)")


def _validate_csv(path: Path, log_path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise TracyRunError(f"ops CSV missing/empty: {path}; log: {log_path}")
    header = path.open().readline()
    if not header.startswith("OP CODE"):
        raise TracyRunError(f"unexpected CSV header in {path}: {header[:60]!r}; log: {log_path}")


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
    cmd = ["python", "-m", "pytest", "-o", "addopts=", perf_test, "--collect-only", "-q"]
    proc = runner(
        cmd, cwd=Path(tt_metal_root), env=env or dict(os.environ), capture_output=True, text=True, timeout=120
    )
    ids = [ln.strip() for ln in (proc.stdout or "").splitlines() if "::" in ln and not ln.startswith("=")]
    tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-6:])
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
    cmd = ["python", "-m", "pytest", "-o", "addopts=", perf_test, "--collect-only", "-q"]
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
        env["TT_METAL_DEVICE_PROFILER"] = "1"
        env.update(extra_env or {})
        try:
            from .profiler_heal import ensure_profiler_patched

            ensure_profiler_patched(root)
        except Exception:
            pass
        node_id = resolve_node_id(root, perf_test, case, env=env, runner=collect_runner)
        cmd = build_tracy_command(node_id, None, out_dir)
        support_count = int(env.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT") or 0)
        t_start = time.monotonic()
        partial_reason = None
        heal_attempt = 0
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
                _await_cool()
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
    'Respond with ONLY a JSON object: {{"decision": "continue"|"stop", '
    '"reasoning": <2-3 sentences>}}. Stop only for genuine blockers — '
    "warnings with a sensible fallback are acceptable."
)


def lead_review_gate(
    pathmap: dict[str, Any],
    env_agent_path: str | os.PathLike[str] = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 4,
) -> dict[str, Any]:
    """One lead-model call: read the evidence notes, decide continue/stop.

    Returns {"decision", "reasoning", "model"}; raises DiscoveryRejected on stop.
    No tools — pure judgment over the structured findings."""

    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)
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
    options = ClaudeAgentOptions(
        model=model,
        system_prompt="You make go/no-go calls for an automated perf-optimization harness.",
        allowed_tools=[],
        permission_mode="bypassPermissions",
        setting_sources=[],
        max_turns=max_turns,
        max_buffer_size=50 * 1024 * 1024,
    )
    chunks: list[str] = []
    usage: dict[str, Any] = {}

    async def _go() -> None:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(msg, ResultMessage):
                usage["summary"] = _usage_summary(msg)

    from .sdk_retry import run_with_retry

    run_with_retry(_go, lambda: (chunks.clear(), usage.clear()))
    try:
        verdict = json.loads(_extract_json_object("\n".join(chunks)))
    except json.JSONDecodeError as exc:
        raise DiscoveryRejected(f"lead review returned unparseable verdict: {exc}") from exc
    decision = verdict.get("decision")
    reasoning = str(verdict.get("reasoning", ""))
    if decision not in ("continue", "stop"):
        raise DiscoveryRejected(f"lead review returned invalid decision: {decision!r}")
    if decision == "stop":
        raise DiscoveryRejected(f"lead agent stopped the run: {reasoning}")
    return {"decision": decision, "reasoning": reasoning, "model": model, "usage": usage.get("summary")}


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
