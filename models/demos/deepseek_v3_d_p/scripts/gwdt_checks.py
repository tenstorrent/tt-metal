# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sqlite3
from contextlib import closing
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


TAIL_BYTES = 8 << 20


def read_tail(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - nbytes))
        return f.read().decode(errors="replace")


def classify_loop(log: Path, triage_csv: Path) -> Result:
    """One stress.sh loop. Reads only the log tail: soak logs reach GBs."""
    if triage_csv.is_file() and triage_csv.stat().st_size > 0:
        return Result(Status.HANG, f"triage {triage_csv}")
    text = read_tail(log) if log.is_file() else ""
    verdict = classify_pytest(text, 1)
    if verdict.status == Status.PASS:
        return verdict
    rc = _first(r"TEST_DONE_EXIT=(\d+)\s*\Z", text) or _first(r"TEST_DONE_EXIT=(\d+)", text)
    if rc is not None and int(rc) >= 128:
        return Result(Status.FAIL, sig_name(int(rc)))
    why = (
        _first(r"\[determinism\] (.*DIFFERS.*)$", text)
        or (verdict.reason if _summary_counts(text) else None)
        or _first(r"^E\s+(.*)$", text)
        or (sig_name(int(rc)) if rc is not None else "no result")
    )
    return Result(Status.FAIL, why)


def stress_result(loops: list[Result], stress_log: Path) -> Result:
    for i, r in enumerate(loops, 1):
        if r.status == Status.PENDING:
            return Result(Status.FAIL, f"stress.sh ended before loop {i}, see {stress_log}")
        if r.status != Status.PASS:
            return r
    return Result(Status.PASS, f"{len(loops)}/{len(loops)} loops passed")


def collected_node_ids(log_text: str) -> list[str]:
    """Node ids from `pytest --collect-only -q`, in either the tree (`<Function x[..]>`) or flat (`f.py::x[..]`) form."""
    tree = re.findall(r"^\s*<Function (\S+)>$", log_text, re.MULTILINE)
    return tree or re.findall(r"^(\S+::\S+)$", log_text, re.MULTILINE)


def missing_node_ids(wanted: list[str], collected: list[str]) -> list[str]:
    names = {c.split("::")[-1] for c in collected}
    return [w for w in wanted if w.split("::")[-1] not in names]


def layer_label(node_id: str) -> str:
    m = re.search(r"-(dense_L\d+|moe_(?:full_|shared_)?L\d+|dense|moe)-", node_id)
    return m.group(1) if m else node_id.split("::")[-1]


_VERDICT = {"PASSED": Status.PASS, "FAILED": Status.FAIL, "ERROR": Status.FAIL, "SKIPPED": Status.FAIL}


def row_results(log_text: str, ids: list[str]) -> list[Result]:
    """Per-node-id result of one pytest run: the -rA summary when it is there, else the live -v stream."""
    names = [i.split("::")[-1] for i in ids]
    summary = {n: v for v, n in re.findall(r"^(PASSED|FAILED|ERROR|SKIPPED) \S*::(\S+)", log_text, re.MULTILINE)}
    starts = [log_text.find(n) for n in names]
    rows = []
    for k, n in enumerate(names):
        if n in summary:
            rows.append(Result(_VERDICT[summary[n]]))
            continue
        if starts[k] < 0:
            rows.append(Result(Status.PENDING))
            continue
        later = [p for p in starts[k + 1 :] if p > starts[k]]
        seg = log_text[starts[k] : min(later) if later else len(log_text)]
        verdict = re.search(r"\b(PASSED|FAILED|ERROR|SKIPPED)\b", seg[len(n) :])
        if verdict:
            rows.append(Result(_VERDICT[verdict.group(1)]))
        else:
            done = re.findall(r"pass (\d+/\d+) done", seg)
            rows.append(Result(Status.RUN, f"pass {done[-1]}" if done else ""))
    return rows


def didt_bad_devices(log_text: str) -> dict[int, int]:
    found = re.findall(r"non-deterministic outputs on device (\d+) is (\d+)", log_text)
    return {int(d): int(k) for d, k in found if int(k) > 0}


def run_to_run_bad_devices(log_text: str) -> dict[int, dict[str, int]]:
    """tests/didt/test_matmul_run_to_run_determinism.py: differing elements per device id and shard / core, summed
    over the windows."""
    bad: dict[int, dict[str, int]] = {}
    found = re.findall(
        r"shard (\d+) \(device id (\d+)\), logical core (\(\d+, \d+\)) -> physical (\(\d+,\d+\)): "
        r"run-to-run NOT deterministic, (\d+) elements",
        log_text,
    )
    for shard, dev, logical, physical, n in found:
        per = bad.setdefault(int(dev), {})
        where = f"shard {shard} core {logical} phys {physical}"
        per[where] = per.get(where, 0) + int(n)
    return bad


def sweep_bad_devices(log_text: str) -> dict[int, int]:
    """tests/didt/scratch_core_sweep.py: differing elements per device id, summed over the windows."""
    bad: dict[int, int] = {}
    for dev, n in re.findall(r"device (\d+) NOT deterministic, (\d+) elements", log_text):
        bad[int(dev)] = bad.get(int(dev), 0) + int(n)
    return bad


def sweep_skipped_windows(log_text: str) -> int | None:
    """Windows the sweep caught an exception on: it skips them and still passes."""
    found = re.findall(r"skipped windows: (\d+) ->", log_text)
    return int(found[-1]) if found else None


def write_ack_result(log_text: str) -> Result:
    """tests/didt/test_mm_mcast_write_ack_failure.py run as a script, a hang check with no output compare: one
    `chip N PASS` or `chip N FAIL: why` line per chip, `why` naming the card when the chip's own log does."""
    passed = re.findall(r"chip (\d+) PASS$", log_text, re.MULTILINE)
    failed = sorted(re.findall(r"chip (\d+) FAIL: (.*)$", log_text, re.MULTILINE), key=lambda f: int(f[0]))
    total = len(passed) + len(failed)
    if failed:
        details = [f"chip {c}: " + re.sub(r"^HUNG chip \d+ \(runtime id \d+\): ", "hang ", why) for c, why in failed]
        return Result(Status.FAIL, f"hang on {len(failed)}/{total} chips", details)
    if not re.search(r"all \d+ chips passed", log_text):
        return Result(Status.FAIL, _first(r"^(\w+(?:Error|Exception): .*)$", log_text) or "no result")
    return Result(Status.PASS, f"no hang on {total}/{total} chips")


def mla_determinism_reps(log_text: str) -> int | None:
    found = re.findall(r"Chunked prefill determinism: (\d+) reps", log_text)
    return int(found[-1]) if found else None


def dram_health(db: Path) -> Result:
    """tt-triage check_dram_health from its sqlite output: the script's own errors (uncorrected EDC,
    training, BIST) fail; corrected EDC counts are only reported."""
    if not db.is_file():
        return Result(Status.FAIL, f"no DRAM health output {db}")
    with closing(sqlite3.connect(db)) as con:
        errors = con.execute(
            "SELECT Device, Message FROM diagnostics WHERE Script LIKE 'check_dram_health%' AND Type = 'error'"
        ).fetchall()
        try:
            rows = con.execute(
                'SELECT "Dev", "GDDR Inst", "Corrected Reads", "Corrected Writes" FROM check_dram_health'
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    corrected = [f"dev {d} GDDR {i}: corrected rd/wr {rd}/{wr}" for d, i, rd, wr in rows if rd or wr]
    if errors:
        return Result(Status.FAIL, f"{len(errors)} DRAM errors", [f"dev {d}: {m}" for d, m in errors] + corrected)
    if not rows:
        return Result(Status.FAIL, f"no DRAM health rows in {db}")
    return Result(Status.PASS, f"{len(rows)} GDDR instances, {len(corrected)} with corrected EDC", corrected)


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


_TS = r"^(\d{4}-\d{2}-\d{2} [\d:]{8})"


class LogFollower:
    """Progress of a growing pytest log, read incrementally so a GB-scale soak log is scanned once."""

    def __init__(self, path: Path):
        self.path, self.offset, self.partial = path, 0, ""
        self.iters, self.marker, self.start, self.fwd = 0, "", None, None

    def update(self) -> None:
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        self.offset += len(chunk)
        *lines, self.partial = (self.partial + chunk.decode(errors="replace")).split("\n")
        for line in lines:
            if re.search(r"iter \d+ done \(\d+ chunks\)", line):
                self.iters += 1
            if m := re.search(r"forward_layer_\d+_(?:start|end)", line):
                self.marker = m.group(0)
            elif not self.fwd and (m := re.search(r"Building layer (\d+/\d+)", line)):
                self.marker = f"loading layer {m.group(1)}"
            if self.start is None and "Building TtPrefillTransformer" in line:
                self.start = _ts(line)
            if self.fwd is None and "forward_layer_0_start" in line:
                self.fwd = _ts(line)

    def line(self, iters: int, now: float) -> str:
        parts = [f"iter {self.iters}/{iters}"] + ([self.marker] if self.marker else [])
        if self.start is not None and self.fwd is not None:
            parts.append(f"load {dur(self.fwd - self.start)}")
        parts.append(f"idle {dur(now - self.path.stat().st_mtime)}")
        return "  ".join(parts)


def _ts(line: str) -> float | None:
    m = re.match(_TS, line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() if m else None
