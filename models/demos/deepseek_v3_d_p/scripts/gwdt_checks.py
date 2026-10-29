# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

TOOL_NAME = "galaxy-workload-deployment-test"
GALAXY_CHIPS = 32
MIN_TDP_WATTS = 130


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    HANG = "HANG"
    WARN = "WARN"
    SKIP = "SKIP"
    RUN = "RUN"
    PENDING = "·"


@dataclass
class Result:
    status: Status
    reason: str = ""
    details: list[str] = field(default_factory=list)


def log_root(user: str, home: Path) -> Path:
    data = Path("/data") / user
    if data.is_dir() and os.access(data, os.W_OK):
        return data / TOOL_NAME
    return home / TOOL_NAME


def run_name(now: datetime, host: str, commit: str) -> str:
    return f"LOG_{now:%Y_%m_%d_%H_%M}_{host}_{commit}"


def _chip_ranges(chips: list[int]) -> str:
    ranges, start = [], chips[0]
    for prev, cur in zip(chips, chips[1:] + [None]):
        if cur != prev + 1:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ",".join(ranges)


def _per_value(label: str, values: list[str]) -> list[str]:
    if len(set(values)) == 1:
        return [f"{label} {values[0]} ({len(values)}/{len(values)})"]
    return [
        f"{label} {v} chips {_chip_ranges([i for i, x in enumerate(values) if x == v])}"
        for v in sorted(set(values), key=values.index)
    ]


def _tdp_watts(telem: dict) -> int | None:
    tdp = telem.get("TDP_LIMIT_MAX") or telem.get("SMBUS_TX_TDP_LIMIT_MAX")
    if tdp is None:
        return None
    return tdp if isinstance(tdp, int) else int(str(tdp), 16)


def check_smi(snapshot: dict) -> Result:
    chips = snapshot.get("device_info") or []
    if len(chips) != GALAXY_CHIPS:
        return Result(Status.FAIL, f"found {len(chips)} chips, need {GALAXY_CHIPS}")
    fw = [str(c.get("firmwares", {}).get("fw_bundle_version")) for c in chips]
    tdp = [_tdp_watts(c.get("smbus_telem", {})) for c in chips]
    details = _per_value("fw_bundle_version", fw) + _per_value("TDP_LIMIT_MAX", [f"{w} W" for w in tdp])
    warnings = [
        f"{label} not uniform" for label, v in (("fw_bundle_version", fw), ("TDP_LIMIT_MAX", tdp)) if len(set(v)) > 1
    ]
    low = min((w for w in tdp if w is not None), default=None)
    if low is None or low < MIN_TDP_WATTS:
        warnings.append(f"TDP {low} W < {MIN_TDP_WATTS} W -> DS_PERF_IGNORE_POWER=1")
    return Result(Status.WARN if warnings else Status.PASS, "; ".join(warnings), details)


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


_SIGNALS = {6: "SIGABRT", 7: "SIGBUS", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}


def sig_name(rc: int) -> str:
    if rc < 128:
        return f"exit {rc}"
    return _SIGNALS.get(rc - 128, f"signal {rc - 128}")


def classify_loop(log: Path, triage_csv: Path) -> Result:
    if triage_csv.is_file() and triage_csv.stat().st_size > 0:
        return Result(Status.HANG, f"triage {triage_csv}")
    text = log.read_text(errors="replace") if log.is_file() else ""
    if classify_pytest(text, 1).status == Status.PASS:
        return Result(Status.PASS, "1 passed")
    rc = _first(r"TEST_DONE_EXIT=(\d+)\s*\Z", text) or _first(r"TEST_DONE_EXIT=(\d+)", text)
    if rc is not None and int(rc) >= 128:
        return Result(Status.FAIL, sig_name(int(rc)))
    why = (
        _first(r"\[determinism\] (.*DIFFERS.*)$", text)
        or _first(r"^E\s+(.*)$", text)
        or (sig_name(int(rc)) if rc is not None else "no result")
    )
    return Result(Status.FAIL, why)


def collected_node_ids(log_text: str) -> list[str]:
    """Node ids from `pytest --collect-only -q`, in either the tree (`<Function x[..]>`) or flat (`f.py::x[..]`) form."""
    tree = re.findall(r"^\s*<Function (\S+)>$", log_text, re.MULTILINE)
    return tree or re.findall(r"^(\S+::\S+)$", log_text, re.MULTILINE)


def didt_bad_devices(log_text: str) -> dict[int, int]:
    found = re.findall(r"non-deterministic outputs on device (\d+) is (\d+)", log_text)
    return {int(d): int(k) for d, k in found if int(k) > 0}


def fabric_results(log_text: str) -> dict[str, bool]:
    found = re.findall(r"^(PASSED|FAILED) \S*test_fabric_opens\[(\w+)\]", log_text, re.MULTILINE)
    return {name: verdict == "PASSED" for verdict, name in found}


def dur(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _log_ts(pattern: str, text: str) -> float | None:
    ts = _first(rf"^(\d{{4}}-\d{{2}}-\d{{2}} [\d:]{{8}})[^\n]*{pattern}", text)
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() if ts else None


def progress(log_text: str, iters: int, now: float, mtime: float) -> str:
    done = len(re.findall(r"iter \d+ done \(\d+ chunks\)", log_text))
    layer = re.findall(r"forward_layer_\d+_(?:start|end)", log_text)
    loading = re.findall(r"Building layer (\d+/\d+)", log_text)
    parts = [f"iter {done}/{iters}"]
    if layer:
        parts.append(layer[-1])
    elif loading:
        parts.append(f"loading layer {loading[-1]}")
    start, fwd = _log_ts("Building TtPrefillTransformer", log_text), _log_ts("forward_layer_0_start", log_text)
    if start is not None and fwd is not None:
        parts.append(f"load {dur(fwd - start)}")
    parts.append(f"idle {dur(now - mtime)}")
    return "  ".join(parts)
