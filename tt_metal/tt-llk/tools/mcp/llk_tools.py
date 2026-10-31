# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LLK tools for AI assistants, served over MCP (stdio) on the machine with the card."""

import re
import subprocess
from pathlib import Path

LLK = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = LLK / ".claude" / "scripts" / "perf_compare_commits.sh"
PERF_TESTS = LLK / "tests" / "python_tests"
ARCHES = ("wormhole", "blackhole")
REPORT_LINE = re.compile(r"^report: (.+)$", re.M)
LOG_TAIL = 40


def perf_tests():
    return sorted(p.stem for p in PERF_TESTS.glob("perf_*.py"))


def compare_command(
    arch,
    test,
    baseline=None,
    current=None,
    iterations=3,
    threshold=0.02,
    min_cycles=30,
    speed_of_light=False,
    refresh=False,
    dry_run=False,
):
    if arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, not {arch!r}")
    if test not in perf_tests():
        raise ValueError(f"{test!r} is not a perf test; list_perf_tests names them")
    cmd = [str(COMPARE_SCRIPT), arch, test]
    cmd += ["--iterations", str(iterations), "--threshold", str(threshold)]
    cmd += ["--min-cycles", str(min_cycles)]
    for flag, value in (("--baseline", baseline), ("--current", current)):
        if value:
            cmd += [flag, value]
    for flag, on in (
        ("--speed-of-light", speed_of_light),
        ("--refresh", refresh),
        ("--dry-run", dry_run),
    ):
        if on:
            cmd.append(flag)
    return cmd


def compare_result(returncode, output, dry_run=False):
    report_path = REPORT_LINE.search(output)
    report = None
    if report_path and Path(report_path.group(1)).is_file():
        report = Path(report_path.group(1)).read_text()
    if dry_run and returncode == 0:
        verdict = "dry run"
    elif report is None:
        verdict = "failed"
    else:
        verdict = "clean" if returncode == 0 else "regressed"
    return {
        "verdict": verdict,
        "report_path": report_path.group(1) if report_path else None,
        "report": report,
        "log_tail": "\n".join(output.splitlines()[-LOG_TAIL:]),
    }


def perf_compare(arch, test, **options):
    cmd = compare_command(arch, test, **options)
    proc = subprocess.run(
        cmd, cwd=LLK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return compare_result(proc.returncode, proc.stdout, options.get("dry_run", False))


def main():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError:
            raise SystemExit("llk-tools needs the MCP SDK: pip install mcp")

    server = MCPServer("llk-tools")

    @server.tool(name="list_perf_tests")
    def list_perf_tests_tool() -> list[str]:
        """The LLK perf test modules that perf_compare can measure."""
        return perf_tests()

    @server.tool(name="perf_compare")
    def perf_compare_tool(
        arch: str,
        test: str,
        baseline: str | None = None,
        current: str | None = None,
        iterations: int = 3,
        threshold: float = 0.02,
        min_cycles: float = 30,
        speed_of_light: bool = False,
        refresh: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Compare one LLK perf test between two commits on this machine's card.

        Use this instead of a PR gate to check a change for perf regressions.
        arch is wormhole or blackhole. baseline defaults to the commit the branch
        was cut from (merge-base with origin/main), current to HEAD; any git ref
        works. Only committed code is measured. A point regresses when it is more
        than threshold slower AND more than min_cycles slower, median of the
        iterations per side. Runs are cached per commit, so repeating a comparison
        or adding iterations only measures what is missing. A full run takes from
        minutes to about an hour; dry_run first shows what would be measured.
        Returns the verdict (clean, regressed, failed or dry run), the Markdown
        report and the last lines of the log.
        """
        return perf_compare(
            arch,
            test,
            baseline=baseline,
            current=current,
            iterations=iterations,
            threshold=threshold,
            min_cycles=min_cycles,
            speed_of_light=speed_of_light,
            refresh=refresh,
            dry_run=dry_run,
        )

    server.run()


if __name__ == "__main__":
    main()
