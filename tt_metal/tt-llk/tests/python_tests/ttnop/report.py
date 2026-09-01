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

TTNOP_DIR = Path(__file__).resolve().parent
TT_METAL_DIR = TTNOP_DIR.parents[3]
SFPI_BIN = Path("tests/sfpi/compiler/bin")
SFPI_VERSION = Path("tests/sfpi/sfpi.version")
TENSTORRENT_DEVICES = Path("/sys/class/tenstorrent")
REPO_PATH_MARKER = f"/{TT_METAL_DIR.name}/"

FAILURES = "failures.jsonl"
SKIPS = "skips.jsonl"
MARKDOWN = "report.md"


def _sfpi_bin() -> Path:
    return Path(os.environ.get("LLK_HOME") or TT_METAL_DIR) / SFPI_BIN


def compile_path_to_repo_path(location: str) -> str:
    """Turn a compiler path from DWARF into a repository path."""
    path, _, line = location.rpartition(":")
    full = os.path.normpath(path)
    marker = full.find(REPO_PATH_MARKER)
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
        # Missing DWARF tooling should not stop the sweep.
        return ()
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
    llk_home = Path(os.environ.get("LLK_HOME") or TT_METAL_DIR)
    sfpi_version = llk_home / SFPI_VERSION
    boards = sorted(TENSTORRENT_DEVICES.glob("*/device/device"))
    return {
        "arch": arch,
        "site_mode": site_mode,
        "filler_policy": filler,
        "drift": "on (frozen stimuli)" if drift else "off (rolling stimuli)",
        "commit": _run("git", "-C", str(TTNOP_DIR), "rev-parse", "--short", "HEAD"),
        "sfpi": (
            sfpi_version.read_text().strip() if sfpi_version.exists() else "unknown"
        ),
        "host": socket.gethostname(),
        "board": (
            f"{boards[0].read_text().strip()} x{len(boards)}" if boards else "unknown"
        ),
    }


def _append_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(report_dir: Path, record: dict) -> None:
    _append_jsonl(report_dir / FAILURES, (record,))


def load(report_dir: Path) -> list:
    return _load_jsonl(report_dir / FAILURES)


def append_skips(report_dir: Path, hung: str, cases) -> None:
    """Sibling params stepped over because `hung` wedged a core. Survives the reset."""
    records = [{"case": case, "hung": hung} for case in cases if case]
    if records:
        _append_jsonl(report_dir / SKIPS, records)


def load_skips(report_dir: Path) -> list:
    return _load_jsonl(report_dir / SKIPS)


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
    """Fold consecutive counts: {1,2,3,7} -> "1-3, 7"."""
    runs = []
    for value in sorted(set(values)):
        if runs and value == runs[-1][1] + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return separator.join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs)


def _by_site(records: list):
    """Group and label sites 1a, 1b, 2a in first-seen sweep order."""
    sites = defaultdict(list)
    for record in records:
        key = (record["case"], record["thread"], record["addr"])
        sites[key].append(record)

    labels, case_number, seen_sites = {}, {}, defaultdict(int)
    for key in sites:
        case = key[0]
        case_number.setdefault(case, len(case_number) + 1)
        labels[key] = f"{case_number[case]}{chr(ord('a') + seen_sites[case])}"
        seen_sites[case] += 1
    return sites, labels, len(case_number), list(sites)


def in_plan_order(records: list) -> list:
    """Restore per-case sweep order after concurrent workers append the log."""
    first_seen = {
        case: i for i, case in enumerate(dict.fromkeys(r["case"] for r in records))
    }
    return sorted(
        records, key=lambda record: (first_seen[record["case"]], record.get("seq", 0))
    )


def _pcc_cell(rows: list) -> str:
    """Lowest PCC vs clean, if recorded."""
    pccs = [record["pcc"] for record in rows if "pcc" in record]
    if not pccs:
        return ""
    worst = min(pccs)
    return f"{worst:.6f} (Δ {1.0 - worst:.2g})"


def _band(rows: list) -> list:
    """Distinct failing counts."""
    return sorted({record["delay"] for record in rows})


def _rate(rows: list) -> float:
    """Highest failure rate. Breadth variants have rate 1."""
    return max(record["fails"] / max(record["runs"], 1) for record in rows)


def _strength(rows: list) -> tuple:
    """Prefer a wider band, then a higher rate, then an earlier count."""
    band = _band(rows)
    return -len(band), -_rate(rows), band[0]


def _by_filler(group: list):
    """Split one site by filler, strongest witness first."""
    buckets = defaultdict(list)
    for record in group:
        buckets[record["filler"]].append(record)
    return [
        (name, buckets[name])
        for name in sorted(buckets, key=lambda name: (*_strength(buckets[name]), name))
    ]


def _witness(rows: list) -> str:
    """Format one site/filler group's band and rate."""
    head = rows[0]
    band = _band(rows)
    fails = sum(record["fails"] for record in rows)
    runs = sum(record["runs"] for record in rows)
    return (
        f"`{head['thread']} {head['op']}@0x{head['addr']:05x}` `{head['filler']}` "
        f"{as_ranges(band)} ({fails}/{runs})"
    )


def _strongest_section(records: list) -> list:
    """Lead with the widest band and, when different, the highest rate."""
    groups = defaultdict(list)
    for record in records:
        key = (record["case"], record["thread"], record["addr"], record["filler"])
        groups[key].append(record)
    widest = min(groups, key=lambda key: _strength(groups[key]))
    frequent = max(
        groups, key=lambda key: (_rate(groups[key]), len(_band(groups[key])))
    )
    out = [
        "",
        "## Strongest signal / tie-breaker",
        "",
        f"- widest band: {_witness(groups[widest])}",
    ]
    if frequent != widest:
        out.append(f"- highest rate: {_witness(groups[frequent])}")
    return out


def _table(headers, rows) -> list:
    """Render a markdown table."""
    return [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
        *(f"| {' | '.join(str(value) for value in row)} |" for row in rows),
    ]


def _tags(rows: list) -> str:
    return ", ".join(sorted({record["tag"] for record in rows}))


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
        "and the rest of that family hits the same site. They were skipped "
        "rather than spending another recovery on the same race. They are not findings.",
        "",
        *_table(
            ("skipped", "hung case"),
            (
                (f"`{case}`", f"`{hung}`")
                for hung, cases in by_hung.items()
                for case in cases
            ),
        ),
    ]
    return out


def _site_section(key, group: list, label: str) -> list:
    case, thread, addr = key
    head = group[0]
    fillers = _by_filler(group)
    out = [
        "",
        f"## {label}. {thread} {head['op']} @ 0x{addr:05x}",
        "",
        f"- case: `{case}`",
        f"- site index (mode `{head['site_mode']}`): {head['site_index']}",
    ]
    first_error = next((r["error"] for r in group if r.get("error")), "")
    if first_error:
        out.append(f"- first finding: `{first_error}`")

    out += [
        "",
        "### NOP types",
        "",
        *_table(
            ("NOP_TYPE", "word", "NOP counts", "how", "PCC vs clean"),
            (
                (
                    f"`{filler}`",
                    f"`0x{rows[0]['filler_word']:08x}`",
                    f"{as_ranges(_band(rows))} ({len(_band(rows))})",
                    _tags(rows),
                    _pcc_cell(rows),
                )
                for filler, rows in fillers
            ),
        ),
    ]

    if any(record["runs"] > 1 for record in group):
        out += [
            "",
            "### Failure rate",
            "",
            *_table(
                ("NOP_TYPE", "nops", "fails / runs", "how"),
                (
                    (
                        f"`{record['filler']}`",
                        record["delay"],
                        f"{record['fails']} / {record['runs']}",
                        record["tag"],
                    )
                    for _, rows in fillers
                    for record in sorted(rows, key=lambda r: r["delay"])
                ),
            ),
        ]

    out += ["", "### Where the NOPs went in (innermost frame first)", ""]
    chain = head.get("chain") or ()
    out += [f"{i}. `{frame}`" for i, frame in enumerate(chain, 1)] or [
        "_no DWARF available_"
    ]
    out += ["", "### Reproduce", "", "```bash"]
    out += [
        reproduce_command(rows[0], [record["delay"] for record in rows])
        for _, rows in fillers
    ]
    return [*out, "```"]


def render(report_dir: Path, env: dict) -> str:
    records = in_plan_order(load(report_dir))
    skips = load_skips(report_dir)
    if not records and not skips:
        return ""
    sites, labels, cases, order = _by_site(records) if records else ({}, {}, 0, [])

    out = ["# ttnop timing-perturbation findings", ""]
    out += _table(
        ("field", "value"), ((key, f"`{value}`") for key, value in env.items())
    )
    out += [
        "",
        "Frozen stimuli: clean and NOP runs share the same input.",
        "**mismatch**: failed the test golden. **drift**: passed golden, differed from clean.",
        "**PCC vs clean**: Pearson of those two tensors (`Δ` = `1 − pcc`), not vs golden.",
        "",
        f"{len(records)} recorded variant(s) across {len(order)} site(s), {cases} case(s)"
        + (f". {len(skips)} case(s) skipped after a hang." if skips else "."),
    ]
    if records:
        out += _strongest_section(records)
    out += _skip_section(skips)
    if not order:
        return "\n".join(out) + "\n"
    out += ["", "## Sites", ""]
    out += _table(
        ("#", "thread", "site", "NOP_TYPE", "NOP counts", "how", "PCC vs clean"),
        (
            (
                labels[key],
                key[1],
                f"`{sites[key][0]['op']}@0x{key[2]:05x}`",
                f"`{filler}`",
                f"{as_ranges(_band(rows))} ({len(_band(rows))})",
                _tags(rows),
                _pcc_cell(rows),
            )
            for key in order
            for filler, rows in _by_filler(sites[key])
        ),
    )
    for key in order:
        out += _site_section(key, sites[key], labels[key])

    return "\n".join(out) + "\n"


def write_markdown(report_dir: Path, env: dict) -> Path:
    """Render the markdown, or do nothing at all when there is nothing to report."""
    text = render(report_dir, env)
    if not text:
        return None
    path = report_dir / MARKDOWN
    path.write_text(text)
    return path
