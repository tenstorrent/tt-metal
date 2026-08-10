# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Failure records: DWARF resolution, the JSONL log, and the markdown rendered from it.

The JSONL is the source of truth and the markdown is a view of it. Records are
appended one short line at a time, which is atomic enough for xdist workers to
share a file without taking a lock.
"""

import json
import os
import shlex
import socket
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILURES = "failures.jsonl"
MARKDOWN = "report.md"


def _sfpi_bin() -> Path:
    llk_home = os.environ.get("LLK_HOME") or str(HERE.parents[3])
    return Path(llk_home) / "tests" / "sfpi" / "compiler" / "bin"


def compile_path_to_repo_path(location: str) -> str:
    """DWARF records the path as the compiler saw it; normalize to repo-relative."""
    path, _, line = location.rpartition(":")
    full = os.path.normpath(path)
    marker = full.find("/tt_metal/")
    return f"{full[marker + 1:] if marker >= 0 else full}:{line}"


@lru_cache(maxsize=None)
def source_chain(elf: str, vaddr: int) -> tuple:
    """Inline C++ call chain for a site (innermost first): "function  file:line"."""
    addr2line = _sfpi_bin() / "riscv-tt-elf-addr2line"
    try:
        out = subprocess.run(
            [str(addr2line), "-e", elf, "-f", "-C", "-i", f"0x{vaddr:x}"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # A missing or broken toolchain should cost us the "where", not the sweep.
        return ()
    # addr2line -f prints two lines per frame: function name, then file:line.
    frames = [line for line in out.splitlines() if line.strip()]
    return tuple(
        f"{function.split('(')[0]}  {compile_path_to_repo_path(location)}"
        for function, location in zip(frames[::2], frames[1::2])
        if function != "??"
    )


def _run(*command) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(arch: str, site_mode: str, filler: str) -> dict:
    sfpi_version = _sfpi_bin().parents[1] / "sfpi.version"
    boards = sorted(Path("/sys/class/tenstorrent").glob("*/device/device"))
    return {
        "arch": arch,
        "site_mode": site_mode,
        "filler_policy": filler,
        "commit": _run("git", "-C", str(HERE), "rev-parse", "--short", "HEAD"),
        "sfpi": (
            sfpi_version.read_text().strip() if sfpi_version.exists() else "unknown"
        ),
        "host": socket.gethostname(),
        "board": (
            f"{boards[0].read_text().strip()} x{len(boards)}" if boards else "unknown"
        ),
    }


def append(report_dir: Path, record: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / FAILURES, "a") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def load(report_dir: Path) -> list:
    path = report_dir / FAILURES
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reproduce_command(record: dict, delays=()) -> str:
    """Re-run just this site, at just the counts that broke it, as a failure rate."""
    env = {
        "CHIP_ARCH": record["arch"],
        "TTNOP_SITE_MODE": record["site_mode"],
        "TTNOP_THREADS": record["thread"],
        "TTNOP_SITES": f"{record['thread']}:{record['site_index']}",
        "TTNOP_FILLER": record["filler"],
        "TTNOP_DELAYS": as_ranges(delays or [record["delay"]], ","),
        "TTNOP_REPEATS": "50",
    }
    assignments = " ".join(f"{key}={value}" for key, value in env.items())
    return f"{assignments} ./focus.sh {shlex.quote(record['case'])}"


def as_ranges(values, separator: str = ", ") -> str:
    """Exact counts, folded where they run on: {1,2,3,7} -> "1-3, 7".

    A 1-100 sweep can fail at eighty of them, and eighty comma-separated numbers
    is a wall nobody reads.
    """
    runs = []
    for value in sorted(set(values)):
        if runs and value == runs[-1][1] + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return separator.join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs)


def _by_site(records: list):
    """Group failures per (case, thread, site) and label them 1a, 1b, 2a, ... .

    The label numbers the pytest case and letters the sites within it, so the
    summary table and the section below it can point at each other.
    """
    sites = defaultdict(list)
    for record in records:
        sites[(record["case"], record["thread"], record["addr"])].append(record)
    labels, case_number, seen_sites = {}, {}, defaultdict(int)
    for key in sorted(sites):
        case = key[0]
        case_number.setdefault(case, len(case_number) + 1)
        labels[key] = f"{case_number[case]}{chr(ord('a') + seen_sites[case])}"
        seen_sites[case] += 1
    return sites, labels, len(case_number)


def render(report_dir: Path, env: dict) -> str:
    records = load(report_dir)
    if not records:
        return ""
    sites, labels, cases = _by_site(records)

    out = ["# ttnop timing-perturbation findings", "", "| | |", "| --- | --- |"]
    out += [f"| {key} | `{value}` |" for key, value in env.items()]
    out += [
        "",
        f"{len(records)} failing variant(s) at {len(sites)} site(s) across {cases} case(s).",
        "",
        "## Failing sites",
        "",
        "| # | thread | site | failing NOP counts | how |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(sites):
        group = sites[key]
        tags = sorted({record["tag"] for record in group})
        out.append(
            f"| {labels[key]} | {key[1]} | `{group[0]['op']}@0x{key[2]:05x}` | "
            f"{as_ranges(record['delay'] for record in group)} | {', '.join(tags)} |"
        )

    for key in sorted(sites):
        case, thread, addr = key
        group = sorted(sites[key], key=lambda record: record["delay"])
        head = group[0]
        delays = [record["delay"] for record in group]
        out += [
            "",
            f"## {labels[key]}. {thread} {head['op']} @ 0x{addr:05x}",
            "",
            f"- case: `{case}`",
            f"- failing NOP counts ({len(set(delays))} of them): {as_ranges(delays)}",
            f"- filler: `{head['filler']}` = `0x{head['filler_word']:08x}`",
            f"- site index (mode `{head['site_mode']}`): {head['site_index']}",
        ]
        first_error = next(
            (record["error"] for record in group if record.get("error")), ""
        )
        if first_error:
            out.append(f"- first error: `{first_error}`")

        # Only a depth run repeats a variant, and only then is a rate meaningful.
        if any(record["runs"] > 1 for record in group):
            out += [
                "",
                "### Failure rate",
                "",
                "| nops | fails / runs | how |",
                "| --- | --- | --- |",
            ]
            out += [
                f"| {record['delay']} | {record['fails']} / {record['runs']} | {record['tag']} |"
                for record in group
            ]

        out += ["", "### Where the NOPs went in (innermost frame first)", ""]
        chain = head.get("chain") or ()
        out += [f"{i}. `{frame}`" for i, frame in enumerate(chain, 1)] or [
            "_no DWARF available_"
        ]
        out += [
            "",
            "### Reproduce",
            "",
            "```bash",
            reproduce_command(head, delays),
            "```",
        ]

    return "\n".join(out) + "\n"


def write_markdown(report_dir: Path, env: dict) -> Path:
    """Render the markdown, or do nothing at all when there is nothing to report."""
    text = render(report_dir, env)
    if not text:
        return None
    path = report_dir / MARKDOWN
    path.write_text(text)
    return path
