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
    proc = subprocess.run(["tt-smi", "-s"], check=True, capture_output=True, text=True)
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

    Fails fast per section 3.1 if .env.agent is missing — BEFORE any SDK call.
    """
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)  # ConfigError here if absent
    model = get_model("sub", resolved)

    def runner(prompt: str) -> str:
        import asyncio

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

        asyncio.run(_go())
        return _extract_json_object("\n".join(chunks))

    runner.last_usage = None
    runner.model = model

    return runner


# ---------------------------------------------------------------------------
# 7.4 tracy stage-1 RUN + preflight — per the FINAL stage-1 contract
# ---------------------------------------------------------------------------


class TracyRunError(Exception):
    """Stage-1 crash: nonzero exit, timeout, or no usable CSV produced."""


class PreflightError(Exception):
    """The discovered perf test selects zero tests (the S512 trap)."""


def build_tracy_command(perf_test: str, case: str | None, out_dir: str | Path) -> list[str]:
    """The raw profile_this command (C++ post-processing default) + -o.

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -o <out> -m pytest ... -sv
    Run directly (never via profile_this.py: it swallows the exit code).
    """
    cmd = ["python", "-m", "tracy", "-v", "-r", "-p", "-o", str(out_dir), "-m", "pytest", perf_test]
    if case:
        cmd += ["-k", case]
    cmd += ["-sv"]
    return cmd


def _execute(cmd: list[str], cwd: Path, env: dict, timeout_s: int, log_path: Path) -> int:
    """Run cmd with output streamed to log_path (live-tailable). Hang-proof:
    no pipes (a daemon child inheriting them cannot deadlock us), and the
    whole process GROUP is killed on timeout (so orphaned capture-release
    daemons die too). Returns the exit code. Injectable seam for tests."""
    import signal

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            raise TracyRunError(
                f"tracy run timed out after {timeout_s}s (process group killed); log: {log_path}"
            ) from None


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
    timeout_s: int = 1800,  # TBD(run-timeout): provisional default
    execute: Callable[..., int] = _execute,
    extra_env: dict[str, str] | None = None,  # e.g. TT_METAL_VISIBLE_DEVICES
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
        cmd = build_tracy_command(perf_test, case, out_dir)
        watermark = time.time() - 0.05
        t_start = time.monotonic()
        code = execute(cmd, root, env, timeout_s, log_path)
        wall_ms = (time.monotonic() - t_start) * 1000.0
        if code != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-15:]) if log_path.is_file() else ""
            raise TracyRunError(f"tracy run exit {code}; log {log_path}; tail:\n{tail}")

        # layer 1: directed output (-o) — a dir only this run writes to
        found = sorted(out_dir.glob("**/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
        # layer 2: stdout regex cross-check
        log_text = log_path.read_text() if log_path.is_file() else ""
        m = _CSV_STDOUT_RE.search(log_text)
        if m:
            reported = Path(m.group(1))
            if found and reported.resolve() != found[-1].resolve():
                with open(log_path, "a") as fh:
                    fh.write(f"\n[harness] WARNING: -o glob {found[-1]} != stdout path {reported}\n")
            elif not found and reported.is_file():
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
    import asyncio

    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("lead", resolved)
    findings = json.dumps({k: pathmap[k] for k in ("perf_test", "pcc", "components", "summary", "warnings")}, indent=1)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt="You make go/no-go calls for an automated perf-optimization harness.",
        allowed_tools=[],
        permission_mode="bypassPermissions",
        setting_sources=[],
        max_turns=max_turns,
    )
    chunks: list[str] = []
    usage: dict[str, Any] = {}

    async def _go() -> None:
        async for msg in query(prompt=REVIEW_PROMPT.format(findings=findings), options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(msg, ResultMessage):
                usage["summary"] = _usage_summary(msg)

    asyncio.run(_go())
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
