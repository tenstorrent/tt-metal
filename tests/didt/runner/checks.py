# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pass/fail parsers for didt_runner, forked from the galaxy workload deployment test's gwdt_checks.py."""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    PENDING = "PENDING"
    RUN = "RUN"
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    HANG = "HANG"
    SKIP = "SKIP"


@dataclass
class Result:
    status: Status
    reason: str = ""
    details: list[str] = field(default_factory=list)


def _summary_counts(text: str) -> dict[str, int]:
    lines = re.findall(r"^=+ (.*) in [\d.]+s", text, re.MULTILINE)
    if not lines:
        return {}
    return {k.rstrip("s"): int(n) for n, k in re.findall(r"(\d+) (\w+)", lines[-1])}


def _first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def classify_pytest(log_text: str, expected_passed: int) -> Result:
    counts = _summary_counts(log_text)
    if counts.get("skipped"):
        why = _first(r"^SKIPPED \[\d+\] [^:]+:\d+: (.*)$", log_text) or ""
        return Result(Status.FAIL, f"{counts['skipped']} skipped: {why}".rstrip(": "))
    if counts.get("failed") or counts.get("error"):
        why = _first(r"^(?:FAILED|ERROR) \S+ - (.*)$", log_text) or _first(r"^E\s+(.*)$", log_text) or "failed"
        return Result(Status.FAIL, why)
    passed = counts.get("passed", 0)
    if passed != expected_passed:
        return Result(Status.FAIL, f"{passed} passed, expected {expected_passed}")
    return Result(Status.PASS, f"{passed} passed")


TAIL_BYTES = 8 << 20


def read_tail(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - nbytes))
        return f.read().decode(errors="replace")


def _tree_node_ids(log_text: str) -> list[str]:
    """Full node ids from the `<Dir>/<Package>/<Module>/<Class>/<Function>` tree. The top `<Dir>` is the rootdir
    itself, so paths start below it, as in flat ids."""
    ids, stack = [], []  # stack of (indent, kind, name)
    for indent, kind, name in re.findall(r"^( *)<(Dir|Package|Module|Class|Function) (.+)>$", log_text, re.MULTILINE):
        while stack and stack[-1][0] >= len(indent):
            stack.pop()
        if kind == "Function":
            path = "/".join(n for _, k, n in stack[1:] if k != "Class")
            classes = [n for _, k, n in stack if k == "Class"]
            ids.append("::".join([path, *classes, name]))
        else:
            stack.append((len(indent), kind, name))
    return ids


def collected_node_ids(log_text: str) -> list[str]:
    """Node ids from `pytest --collect-only`, in either the tree or the flat (`f.py::x[..]`) form."""
    return _tree_node_ids(log_text) or re.findall(r"^(\S+::\S+)$", log_text, re.MULTILINE)


def missing_node_ids(wanted: list[str], collected: list[str]) -> list[str]:
    """Full ids, file included: two test files can share a test name and parametrization."""
    have = set(collected)
    return [w for w in wanted if w not in have]


def sweep_skipped_windows(log_text: str) -> int | None:
    """Windows the sweep caught an exception on: it skips them and still passes."""
    found = re.findall(r"skipped windows: (\d+) ->", log_text)
    return int(found[-1]) if found else None


def write_ack_chips(log_text: str) -> tuple[Result, dict[int, str]]:
    """tests/didt/test_mm_mcast_write_ack_failure.py run as a script: one `chip N PASS` or `chip N FAIL: why` line
    per chip, N being the chip's BusDeviceFunction rank on the host. Returns the step result and chip N -> why."""
    passed = re.findall(r"chip (\d+) PASS$", log_text, re.MULTILINE)
    failed = sorted(re.findall(r"chip (\d+) FAIL: (.*)$", log_text, re.MULTILINE), key=lambda f: int(f[0]))
    total = len(passed) + len(failed)
    if failed:
        details = [f"chip {c}: " + re.sub(r"^HUNG chip \d+ \(runtime id \d+\): ", "hang ", why) for c, why in failed]
        return Result(Status.FAIL, f"hang or ND on {len(failed)}/{total} chips", details), {
            int(c): why.strip() for c, why in failed
        }
    if not re.search(r"all \d+ chips passed", log_text):
        return Result(Status.FAIL, _first(r"^(\w+(?:Error|Exception): .*)$", log_text) or "no result"), {}
    return Result(Status.PASS, f"no hang on {total}/{total} chips"), {}


@dataclass
class GddrEdc:
    bus: str
    instance: int
    bank: int
    corr_rd: int
    corr_wr: int
    uncorr_rd: int
    uncorr_wr: int


def _telem(t: dict, key: str) -> int:
    v = t.get(key, 0)
    return int(v, 16) if isinstance(v, str) else int(v or 0)


def gddr_edc(snapshot: dict) -> list[GddrEdc]:
    """Non-zero GDDR EDC counters per chip and GDDR instance from a tt-smi snapshot, decoded as in tt-triage
    check_dram_health: GDDR_<2k>_<2k+1>_CORR_ERRS packs rd/wr bytes for instance 2k in bits 0-15 and 2k+1 in 16-31;
    GDDR_UNCORR_ERRS has an rd/wr bit pair per instance. The bank counts only enabled instances. The counters clear on
    reset, so a snapshot right after a step holds exactly that step's errors."""
    out = []
    for chip in snapshot.get("device_info") or []:
        t = chip.get("smbus_telem", {})
        enabled, uncorr, bank = _telem(t, "ENABLED_GDDR") & 0xFF, _telem(t, "GDDR_UNCORR_ERRS"), 0
        for i in range(8):
            if not (enabled >> i) & 1:
                continue
            word, shift = _telem(t, f"GDDR_{i & ~1}_{i | 1}_CORR_ERRS"), 16 if i % 2 else 0
            e = GddrEdc(
                chip.get("board_info", {}).get("bus_id", "?"),
                i,
                bank,
                (word >> shift) & 0xFF,
                (word >> (shift + 8)) & 0xFF,
                (uncorr >> (2 * i)) & 1,
                (uncorr >> (2 * i + 1)) & 1,
            )
            if e.corr_rd or e.corr_wr or e.uncorr_rd or e.uncorr_wr:
                out.append(e)
            bank += 1
    return out


def umd_metal_ids(log_text: str) -> dict[int, int]:
    """PCIe (/dev/tenstorrent) index -> metal device id, from UMD's 'Opening local chip ids/PCIe ids' line."""
    m = re.search(r"Opening local chip ids/PCIe ids: \{([\d, ]*)\}/\[([\d, ]*)\]", log_text)
    if not m:
        return {}
    metal, pcie = ([int(x) for x in g.split(",") if x.strip()] for g in m.groups())
    return dict(zip(pcie, metal))


def gddr_result(r: Result, found: list[tuple[str, GddrEdc]], metal_by_bus: dict[str, int]) -> Result:
    """Fold tt-smi GDDR EDC into a step result: uncorrected fails, corrected warns; a worse status is kept."""
    if not found:
        return r
    lines = []
    for snap, e in found:
        dev = metal_by_bus.get(e.bus)
        lines.append(
            f"dev {dev if dev is not None else '?'} ({e.bus.removeprefix('0000:')}) GDDR {e.instance} / bank {e.bank}: "
            f"corrected rd/wr {e.corr_rd}/{e.corr_wr}, uncorrected rd/wr {e.uncorr_rd}/{e.uncorr_wr} [{snap}]"
        )
    uncorrected = any(e.uncorr_rd or e.uncorr_wr for _, e in found)
    worse = Status.FAIL if uncorrected else Status.WARN
    status = worse if r.status in (Status.PASS, Status.WARN) else r.status
    reason = f"{'uncorrected' if uncorrected else 'corrected'} GDDR EDC on {len(found)} instance(s)"
    return Result(status, f"{r.reason}; {reason}" if r.reason else reason, r.details + lines)
