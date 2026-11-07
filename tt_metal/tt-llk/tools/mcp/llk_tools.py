# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LLK tools for AI assistants, served over MCP (stdio) on the machine with the card."""

import ast
import asyncio
import atexit
import contextlib
import csv
import os
import re
import signal
import sys
from pathlib import Path

LLK = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = LLK / ".claude" / "scripts" / "perf_compare_commits.sh"
PERF_TESTS = LLK / "tests" / "python_tests"
ARCHES = ("wormhole", "blackhole")
POSITIONS = (0, 4, 24, 44, 60)
REPORT_LINE = re.compile(r"^report: (.+)$", re.M)
REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^@{}-]*$")
RUN_TYPES = re.compile(r"^(all|[A-Z0-9_]+(,[A-Z0-9_]+)*)$")
MODULE_THRESHOLDS = re.compile(r"^([a-z0-9_]+=[0-9.]+(,[a-z0-9_]+=[0-9.]+)*)?$")
STATS = ("current", "baseline", "delta_pct", "delta_cycles")
LOG_TAIL = 40
REPORT_LIMIT = 200_000
TOP_N = 50
RUNNING = set()


def _decorator_name(node):
    if isinstance(node, ast.Call):
        node = node.func
    return getattr(node, "attr", None) or getattr(node, "id", None)


def _runs_on(path, arch):
    for fn in ast.parse(path.read_text()).body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test"):
            continue
        names = {_decorator_name(d) for d in fn.decorator_list}
        if "perf" in names and f"skip_for_{arch}" not in names:
            return True
    return False


def perf_tests(arch=None):
    arches = (arch,) if arch else ARCHES
    return sorted(
        p.stem
        for p in PERF_TESTS.glob("perf_*.py")
        if any(_runs_on(p, a) for a in arches)
    )


def reports_home():
    home = os.environ.get(
        "PERF_COMPARE_HOME", Path.home() / ".cache" / "tt-llk-perf-compare"
    )
    return Path(home).resolve() / "reports"


def compare_command(
    arch,
    test,
    baseline=None,
    current=None,
    iterations=None,
    threshold=None,
    module_threshold=None,
    run_types=None,
    min_cycles=None,
    code_shift=None,
    speed_of_light=False,
    refresh=False,
    dry_run=False,
):
    if arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, not {arch!r}")
    if test not in perf_tests(arch):
        raise ValueError(
            f"{test!r} is not a perf test that runs on {arch}; list_perf_tests names them"
        )
    for name, ref in (("baseline", baseline), ("current", current)):
        if ref is not None and not REF.match(ref):
            raise ValueError(f"{name} must be a git ref, not {ref!r}")
    if iterations is not None and (not isinstance(iterations, int) or iterations < 1):
        raise ValueError(f"iterations must be a positive integer, not {iterations!r}")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError(f"threshold is a fraction (0.02 = 2%), not {threshold!r}")
    if min_cycles is not None and min_cycles < 0:
        raise ValueError(f"min_cycles must be 0 or more, not {min_cycles!r}")
    if code_shift is not None and (not isinstance(code_shift, int) or code_shift < 0):
        raise ValueError(f"code_shift must be a byte count, not {code_shift!r}")
    if run_types is not None and not RUN_TYPES.match(run_types):
        raise ValueError(
            f"run_types must be run type names or 'all', not {run_types!r}"
        )
    if module_threshold is not None and not MODULE_THRESHOLDS.match(module_threshold):
        raise ValueError(
            f"module_threshold must be module=fraction pairs, not {module_threshold!r}"
        )

    cmd = [str(COMPARE_SCRIPT), arch, test]
    for flag, value in (
        ("--baseline", baseline),
        ("--current", current),
        ("--iterations", iterations),
        ("--threshold", threshold),
        ("--module-threshold", module_threshold),
        ("--run-types", run_types),
        ("--min-cycles", min_cycles),
        ("--code-shift", code_shift),
    ):
        if value is not None:
            cmd += [flag, str(value)]
    for flag, on in (
        ("--speed-of-light", speed_of_light),
        ("--refresh", refresh),
        ("--dry-run", dry_run),
    ):
        if on:
            cmd.append(flag)
    return cmd


def _report_path(output):
    match = REPORT_LINE.search(output)
    if not match:
        return None
    path = Path(match.group(1)).resolve()
    if not path.is_relative_to(reports_home()) or not path.is_file():
        return None
    return path


def compare_result(returncode, output, dry_run=False):
    path = _report_path(output)
    report = path.read_text()[:REPORT_LIMIT] if path else None
    if dry_run and returncode == 0:
        verdict = "dry run"
    elif report is None or returncode not in (0, 1):
        verdict = "failed"
    else:
        verdict = "clean" if returncode == 0 else "regressed"
    return {
        "verdict": verdict,
        "report_path": str(path) if path else None,
        "report": report,
        "log_tail": "\n".join(output.splitlines()[-LOG_TAIL:]),
    }


def _points(report_path, suffix):
    path = Path(report_path).with_suffix(suffix)
    if not path.is_file():
        return {}
    with open(path, newline="") as fh:
        return {
            tuple(
                sorted((k, v) for k, v in row.items() if k not in STATS and v)
            ): float(row["delta_pct"])
            for row in csv.DictReader(fh)
        }


def placement_outcomes(runs):
    """Stage-2 outcome of each shift-0 regression, from {shift: compare_result} at every position."""
    candidates = _points(runs[0]["report_path"], ".regressions.csv")
    regressed = {
        s: (
            set(_points(r["report_path"], ".regressions.csv"))
            if r["verdict"] in ("clean", "regressed")
            else None
        )
        for s, r in runs.items()
    }
    deltas = {
        s: _points(r["report_path"], ".points.csv") if r["report_path"] else {}
        for s, r in runs.items()
    }
    out = []
    for key, delta in sorted(candidates.items(), key=lambda kv: -kv[1]):
        if any(regressed.get(s) is None for s in POSITIONS):
            outcome = "not re-measured"
        elif all(key in regressed[s] for s in POSITIONS):
            outcome = "regression"
        else:
            outcome = "placement"
        out.append(
            {
                "point": dict(key),
                "delta_pct": {f"p{s}": deltas.get(s, {}).get(key) for s in POSITIONS},
                "outcome": outcome,
            }
        )
    return out


def _stop(proc):
    if proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


atexit.register(lambda: [_stop(p) for p in list(RUNNING)])


async def run(cmd):
    env = dict(
        os.environ,
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=LLK,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    RUNNING.add(proc)
    try:
        out, _ = await proc.communicate()
    except BaseException:
        _stop(proc)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(proc.wait()), 5)
        raise
    finally:
        RUNNING.discard(proc)
    return proc.returncode, out.decode(errors="replace")


async def perf_compare(arch, test, **options):
    try:
        cmd = compare_command(arch, test, **options)
    except ValueError as e:
        return {"verdict": "refused", "error": str(e)}
    returncode, output = await run(cmd)
    return compare_result(returncode, output, options.get("dry_run", False))


async def perf_placement_check(arch, test, iterations=1, **options):
    probe = await perf_compare(
        arch, test, code_shift=POSITIONS[1], dry_run=True, **options
    )
    if probe["verdict"] != "dry run":
        return {**probe, "outcomes": []}
    runs = {0: await perf_compare(arch, test, **options)}
    if runs[0]["verdict"] != "regressed":
        return {**runs[0], "outcomes": []}
    for shift in POSITIONS[1:]:
        runs[shift] = await perf_compare(
            arch, test, code_shift=shift, iterations=iterations, **options
        )
    outcomes = placement_outcomes(runs)
    counts = {
        o: sum(r["outcome"] == o for r in outcomes)
        for o in ("regression", "placement", "not re-measured")
    }
    return {
        "verdict": (
            "regression"
            if counts["regression"] or counts["not re-measured"]
            else "placement"
        ),
        "counts": counts,
        "outcomes": outcomes[:TOP_N],
        "reports": {f"p{s}": r["report_path"] for s, r in runs.items()},
    }


def main():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError:
            raise SystemExit(
                f"llk-tools needs the MCP SDK in {sys.executable}: pip install mcp"
            )

    server = MCPServer("llk-tools")

    @server.tool(name="list_perf_tests")
    def list_perf_tests_tool(arch: str | None = None) -> list[str]:
        """The LLK perf test modules that perf_compare can measure on arch (wormhole or blackhole; both if not given)."""
        return perf_tests(arch)

    @server.tool(name="perf_compare")
    async def perf_compare_tool(
        arch: str,
        test: str,
        baseline: str | None = None,
        current: str | None = None,
        iterations: int | None = None,
        threshold: float | None = None,
        module_threshold: str | None = None,
        run_types: str | None = None,
        min_cycles: float | None = None,
        speed_of_light: bool = False,
        refresh: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Check one LLK perf test for regressions between two commits, on this machine's card.

        This is the default perf check. Use it before you push and instead of starting
        the CI perf gate: CI runner time is expensive and the gate only reports later.
        Use the CI gate only when no card is attached, or as the final confirmation.

        arch is wormhole or blackhole. baseline defaults to the commit the branch was
        cut from (merge-base with origin/main), current to HEAD; any git ref works.
        Only committed code is measured. The defaults are the merge gate's: L1_TO_L1
        run type only, threshold 0.02 on blackhole and 0.13 on wormhole (0.25 for
        perf_matmul and perf_math_matmul), and 30 cycles. A point regresses when it is
        more than threshold slower AND more than min_cycles slower, median of the
        iterations (default 3) per side. threshold is a fraction: 0.02 = 2%.
        module_threshold is module=fraction pairs, comma-separated. run_types is a
        comma-separated list of run types, or "all". speed_of_light measures the
        compile-time-parameter build on both sides. refresh measures again instead of
        reusing cached runs.

        Runs are cached per commit, so a repeat or more iterations only measures what
        is missing. A full run takes from minutes to about an hour; dry_run first
        shows what would be measured. Wormhole results on main move from run to run
        with test order: before you call a wormhole point a regression, repeat with
        more iterations, or call perf_placement_check.

        Returns the verdict (clean, regressed, failed or dry run), the Markdown report,
        its path and the last lines of the log.
        """
        return await perf_compare(
            arch,
            test,
            baseline=baseline,
            current=current,
            iterations=iterations,
            threshold=threshold,
            module_threshold=module_threshold,
            run_types=run_types,
            min_cycles=min_cycles,
            speed_of_light=speed_of_light,
            refresh=refresh,
            dry_run=dry_run,
        )

    @server.tool(name="perf_placement_check")
    async def perf_placement_check_tool(
        arch: str,
        test: str,
        baseline: str | None = None,
        current: str | None = None,
        iterations: int = 1,
        threshold: float | None = None,
        module_threshold: str | None = None,
        run_types: str | None = None,
        min_cycles: float | None = None,
    ) -> dict:
        """Tell a real regression from a code placement effect, on this machine's card, as stage 2 of the gate does.

        Runs perf_compare, then measures both commits again with the TRISC code moved
        by 4, 24, 44 and 60 bytes (iterations per position, default 1). Each
        regression gets an outcome: "regression" (slower at all 5 positions),
        "placement" (slower at some positions only, so the change follows where the
        code sits) or "not re-measured" (a position failed). The arguments are those of
        perf_compare. Both commits must have the LLK_CODE_SHIFT knob (#62981);
        if not, the verdict is failed and the log says so. It costs about 5 compares.
        """
        return await perf_placement_check(
            arch,
            test,
            iterations=iterations,
            baseline=baseline,
            current=current,
            threshold=threshold,
            module_threshold=module_threshold,
            run_types=run_types,
            min_cycles=min_cycles,
        )

    server.run()


if __name__ == "__main__":
    main()
