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
SKIPS = "skips.jsonl"
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


def environment(arch: str, site_mode: str, filler: str, drift: bool = True) -> dict:
    sfpi_version = _sfpi_bin().parents[1] / "sfpi.version"
    boards = sorted(Path("/sys/class/tenstorrent").glob("*/device/device"))
    return {
        "arch": arch,
        "site_mode": site_mode,
        "filler_policy": filler,
        "drift": "on (frozen stimuli)" if drift else "off (rolling stimuli)",
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


def append_skips(report_dir: Path, hung: str, cases) -> None:
    """Sibling params stepped over because `hung` wedged a core. Survives the reset."""
    wanted = [case for case in cases if case]
    if not wanted:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / SKIPS, "a") as handle:
        for case in wanted:
            handle.write(
                json.dumps({"case": case, "hung": hung}, separators=(",", ":")) + "\n"
            )


def load_skips(report_dir: Path) -> list:
    path = report_dir / SKIPS
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
        # Repeats is left to focus.sh's own default, so the rate a reproduce line
        # asks for does not have to be kept in step with the runner's.
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
    summary table and the section below it can point at each other. Keys stay
    in first-seen (sweep) order so leftover math sites cannot sort in front
    of the unpack miss that actually started the case.
    """
    sites = defaultdict(list)
    order = []
    for record in records:
        key = (record["case"], record["thread"], record["addr"])
        if key not in sites:
            order.append(key)
        sites[key].append(record)
    labels, case_number, seen_sites = {}, {}, defaultdict(int)
    for key in order:
        case = key[0]
        case_number.setdefault(case, len(case_number) + 1)
        labels[key] = f"{case_number[case]}{chr(ord('a') + seen_sites[case])}"
        seen_sites[case] += 1
    return sites, labels, len(case_number), order


def in_plan_order(records: list) -> list:
    """Sweep order, even when several workers appended to the same log.

    A depth run can split one case over eight workers, whose records then land
    interleaved. Sorting on the plan index each record carries restores that.
    Cases keep the order they first appear in, so a run that gave each worker a
    whole case (records already in plan order) is left exactly as it was.
    """
    first_seen = {}
    for record in records:
        first_seen.setdefault(record["case"], len(first_seen))
    return sorted(
        records, key=lambda record: (first_seen[record["case"]], record.get("seq", 0))
    )


def _pcc_cell(rows: list) -> str:
    """Worst (lowest) PCC vs the clean run in this group; empty if none recorded."""
    pccs = [record["pcc"] for record in rows if "pcc" in record]
    if not pccs:
        return ""
    worst = min(pccs)
    return f"{worst:.6f} (Δ {1.0 - worst:.2g})"


def _by_filler(group: list):
    """Split one site's records by filler, strongest cliff first.

    The same ATGETM can fail with unpacr1 at n=42 and tti_nop only at n=91.
    Grouping them together and then taking the lowest-delay row hid every
    filler except the earliest.
    """
    buckets = defaultdict(list)
    for record in group:
        buckets[record["filler"]].append(record)
    return [
        (name, buckets[name])
        for name in sorted(
            buckets, key=lambda name: (min(r["delay"] for r in buckets[name]), name)
        )
    ]


def _skip_section(skips: list) -> list:
    """Sibling params not run because another param of the same test hung a core."""
    if not skips:
        return []
    by_hung = defaultdict(list)
    for record in skips:
        hung = record.get("hung") or "unknown"
        case = record.get("case") or ""
        if case:
            by_hung[hung].append(case)
    out = [
        "",
        "## Skipped after a hang",
        "",
        "These cases were not run. A sibling param of the same test hung a core, "
        "and the rest of that family hits the same site — so they were stepped over "
        "rather than spending another recovery on the same race. They are not findings.",
        "",
        "| skipped | hung case |",
        "| --- | --- |",
    ]
    for hung, cases in by_hung.items():
        for case in cases:
            out.append(f"| `{case}` | `{hung}` |")
    return out


def render(report_dir: Path, env: dict) -> str:
    records = in_plan_order(load(report_dir))
    skips = load_skips(report_dir)
    if not records and not skips:
        return ""
    sites, labels, cases, order = _by_site(records) if records else ({}, {}, 0, [])

    out = [
        "# ttnop timing-perturbation findings",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    out += [f"| {key} | `{value}` |" for key, value in env.items()]
    out += [
        "",
        "Stimuli are frozen: the clean (no-NOP) run and every NOP run see the same input.",
        "A **mismatch** is the NOP run failing the test's own golden check "
        "(per-element tolerance or PCC vs golden).",
        "A **drift** is the NOP run still passing that check, but its result "
        "differing from the clean run.",
        "**PCC vs clean** is the Pearson correlation of those two hardware tensors "
        "(`Δ` = `1 − pcc`). It is not PCC vs golden.",
        "",
        f"{len(records)} recorded variant(s) across {len(order)} site(s), {cases} case(s)"
        + (f"; {len(skips)} case(s) skipped after a hang." if skips else "."),
    ]
    out += _skip_section(skips)
    if not order:
        return "\n".join(out) + "\n"
    out += [
        "",
        "## Sites",
        "",
        "| # | thread | site | NOP_TYPE | NOP counts | how | PCC vs clean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in order:
        group = sites[key]
        site = f"`{group[0]['op']}@0x{key[2]:05x}`"
        for filler, rows in _by_filler(group):
            tags = sorted({record["tag"] for record in rows})
            out.append(
                f"| {labels[key]} | {key[1]} | {site} | `{filler}` | "
                f"{as_ranges(record['delay'] for record in rows)} | {', '.join(tags)} | "
                f"{_pcc_cell(rows)} |"
            )

    for key in order:
        case, thread, addr = key
        group = sites[key]
        head = group[0]
        fillers = _by_filler(group)
        out += [
            "",
            f"## {labels[key]}. {thread} {head['op']} @ 0x{addr:05x}",
            "",
            f"- case: `{case}`",
            f"- site index (mode `{head['site_mode']}`): {head['site_index']}",
        ]
        first_error = next(
            (record["error"] for record in group if record.get("error")), ""
        )
        if first_error:
            out.append(f"- first finding: `{first_error}`")

        out += [
            "",
            "### NOP types",
            "",
            "| NOP_TYPE | word | NOP counts | how | PCC vs clean |",
            "| --- | --- | --- | --- | --- |",
        ]
        for filler, rows in fillers:
            tags = sorted({record["tag"] for record in rows})
            delays = [record["delay"] for record in rows]
            out.append(
                f"| `{filler}` | `0x{rows[0]['filler_word']:08x}` | "
                f"{as_ranges(delays)} ({len(set(delays))}) | {', '.join(tags)} | "
                f"{_pcc_cell(rows)} |"
            )

        # Only a depth run repeats a variant, and only then is a rate meaningful.
        if any(record["runs"] > 1 for record in group):
            out += [
                "",
                "### Failure rate",
                "",
                "| NOP_TYPE | nops | fails / runs | how |",
                "| --- | --- | --- | --- |",
            ]
            out += [
                f"| `{record['filler']}` | {record['delay']} | "
                f"{record['fails']} / {record['runs']} | {record['tag']} |"
                for filler, rows in fillers
                for record in sorted(rows, key=lambda record: record["delay"])
            ]

        out += ["", "### Where the NOPs went in (innermost frame first)", ""]
        chain = head.get("chain") or ()
        out += [f"{i}. `{frame}`" for i, frame in enumerate(chain, 1)] or [
            "_no DWARF available_"
        ]
        out += ["", "### Reproduce", "", "```bash"]
        out += [
            reproduce_command(rows[0], [record["delay"] for record in rows])
            for filler, rows in fillers
        ]
        out.append("```")

    return "\n".join(out) + "\n"


def write_markdown(report_dir: Path, env: dict) -> Path:
    """Render the markdown, or do nothing at all when there is nothing to report."""
    text = render(report_dir, env)
    if not text:
        return None
    path = report_dir / MARKDOWN
    path.write_text(text)
    return path
