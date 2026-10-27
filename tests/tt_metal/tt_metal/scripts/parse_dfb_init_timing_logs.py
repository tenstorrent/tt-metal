#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parse DFB init timing lines from unit test logs into CSV.

Reads LogDfbInitTimingFromL1() output emitted by the dfb_init_timing_bench binary
(requires TT_METAL_MEASURE_DFB_INIT_TIME=1).

Example:
    ./build/test/tt_metal/dfb_init_timing_bench --case all 2>&1 | tee case-all.log
    python3 tests/tt_metal/tt_metal/scripts/parse_dfb_init_timing_logs.py \\
        case-*-75.log -o dfb_init_timing.csv --summary dfb_init_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TIMING_HEADER_RE = re.compile(
    r"DFB init timing \[(?P<benchmark>[^\]]+)\] @ L1 0x[0-9a-fA-F]+" r"(?: \(used_slots_mask=0x[0-9a-fA-F]+\))?:"
)
TEST_RESULT_RE = re.compile(r"\[\s*(PASSED|FAILED)\s*\]\s+\d+ test")

DM0_RE = re.compile(
    r"(?P<hart>DM0): e2e=(?P<e2e>\d+) "
    r"pre_loop_sw=(?P<pre_loop_sw>\d+) subpassB_desc=(?P<subpassB_desc>\d+) "
    r"(?:between_dfb_sw|hw_reg_write_cycles)=(?P<between_dfb_sw>\d+) "
    r"subpassB_l1_read=(?P<subpassB_l1_read>\d+) "
    r"subpassB_rocc_issue=(?P<subpassB_rocc_issue>\d+) subpassB_hw=(?P<subpassB_hw>\d+) "
    r"first_ie_rmw=(?P<first_ie_rmw>\d+) second_ie_rmw=(?P<second_ie_rmw>\d+) "
    r"isr_enable=(?P<isr_enable>\d+) "
    r"(?:(?:implicit_sync_stores|hw_reg_writes)=(?P<implicit_sync_stores>\d+) )?"
    r"start=(?P<start>\d+) end=(?P<end>\d+)"
)

DM1_RE = re.compile(
    r"(?P<hart>DM1): e2e=(?P<e2e>\d+) "
    r"blob_l1_read_sw=(?P<blob_l1_read_sw>\d+) blob_loop_ovhd=(?P<blob_loop_ovhd>\d+) "
    r"pairs_reg_hw=(?P<pairs_reg_hw>\d+) enable_remapper_hw=(?P<enable_remapper_hw>\d+) "
    r"first_pair_clientR_hw=(?P<first_pair_clientR_hw>\d+) "
    r"first_pair_clientL_hw=(?P<first_pair_clientL_hw>\d+) "
    r"last_pair_hw=(?P<last_pair_hw>\d+) "
    r"(?:pairs_slots_written=\d+ )?"
    r"(?:hw_reg_write_cycles=\d+ hw_reg_writes=\d+ )?"
    r"pairs_slots_written=(?P<pairs_slots_written>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+)"
)

DM1_RE_LEGACY = re.compile(
    r"(?P<hart>DM1): e2e=(?P<e2e>\d+) "
    r"blob_l1_read_sw=(?P<blob_l1_read_sw>\d+) blob_loop_ovhd=(?P<blob_loop_ovhd>\d+) "
    r"pairs_reg_hw=(?P<pairs_reg_hw>\d+) enable_remapper_hw=(?P<enable_remapper_hw>\d+) "
    r"first_pair_clientR_hw=(?P<first_pair_clientR_hw>\d+) "
    r"first_pair_clientL_hw=(?P<first_pair_clientL_hw>\d+) "
    r"last_pair_hw=(?P<last_pair_hw>\d+) "
    r"hw_reg_write_cycles=(?P<hw_reg_write_cycles>\d+) hw_reg_writes=(?P<hw_reg_writes>\d+) "
    r"pairs_slots_written=(?P<pairs_slots_written>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+)"
)

DM_LOCAL_RE = re.compile(
    r"(?P<hart>DM[2-7]): e2e=(?P<e2e>\d+) "
    r"merged_sw=(?P<merged_sw>\d+) remapper_spin=(?P<remapper_spin>\d+) "
    r"tc_hw=(?P<tc_hw>\d+) (?:wait_all|hw_reg_writes)=(?P<wait_all>\d+) "
    r"tc_reset_hw=(?P<tc_reset_hw>\d+) tc_capacity_hw=(?P<tc_capacity_hw>\d+) "
    r"pre_loop=(?P<pre_loop>\d+) entry_hdr=(?P<entry_hdr>\d+) "
    r"tc_slots=(?P<tc_slots>\d+) sig_write=(?P<sig_write>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+)"
)

TRISC_RE = re.compile(
    r"(?P<hart>Neo\d+ (?:unpack|pack)): e2e=(?P<e2e>\d+) "
    r"merged_sw=(?P<merged_sw>\d+) remapper_spin=(?P<remapper_spin>\d+) "
    r"tc_hw=(?P<tc_hw>\d+) (?:wait_all|hw_reg_writes)=(?P<wait_all>\d+) "
    r"tc_reset_hw=(?P<tc_reset_hw>\d+) tc_capacity_hw=(?P<tc_capacity_hw>\d+) "
    r"pre_loop=(?P<pre_loop>\d+) entry_hdr=(?P<entry_hdr>\d+) "
    r"tc_slots=(?P<tc_slots>\d+) sig_write=(?P<sig_write>\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+)"
)

NOT_WRITTEN_RE = re.compile(r"(?P<hart>DM\d|Neo\d+ (?:unpack|pack)): \(not written\)")

DETAIL_COLUMNS = [
    "pre_loop_sw",
    "subpassB_desc",
    "between_dfb_sw",
    "subpassB_l1_read",
    "subpassB_rocc_issue",
    "subpassB_hw",
    "first_ie_rmw",
    "second_ie_rmw",
    "isr_enable",
    "implicit_sync_stores",
    "hw_reg_write_cycles",
    "hw_reg_writes",
    "blob_l1_read_sw",
    "blob_loop_ovhd",
    "pairs_reg_hw",
    "enable_remapper_hw",
    "first_pair_clientR_hw",
    "first_pair_clientL_hw",
    "last_pair_hw",
    "pairs_slots_written",
    "merged_sw",
    "remapper_spin",
    "tc_hw",
    "wait_all",
    "tc_reset_hw",
    "tc_capacity_hw",
    "pre_loop",
    "entry_hdr",
    "tc_slots",
    "sig_write",
]

ROW_COLUMNS = [
    "log_file",
    "case_name",
    "benchmark_name",
    "test_result",
    "hart",
    "role",
    "written",
    "e2e",
    "start",
    "end",
    *DETAIL_COLUMNS,
]

SUMMARY_COLUMNS = [
    "log_file",
    "case_name",
    "benchmark_name",
    "test_result",
    "num_harts_written",
    "worst_e2e",
    "worst_hart",
    "dm0_e2e",
    "dm1_e2e",
    "max_dm2_7_e2e",
    "max_dm2_7_hart",
    "max_neo_e2e",
    "max_neo_hart",
]


@dataclass
class TimingRow:
    log_file: str
    case_name: str
    benchmark_name: str
    test_result: str
    hart: str
    role: str
    written: bool
    e2e: int | None = None
    start: int | None = None
    end: int | None = None
    details: dict[str, int] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, str | int]:
        row: dict[str, str | int] = {
            "log_file": self.log_file,
            "case_name": self.case_name,
            "benchmark_name": self.benchmark_name,
            "test_result": self.test_result,
            "hart": self.hart,
            "role": self.role,
            "written": int(self.written),
            "e2e": self.e2e if self.e2e is not None else "",
            "start": self.start if self.start is not None else "",
            "end": self.end if self.end is not None else "",
        }
        for col in DETAIL_COLUMNS:
            row[col] = self.details.get(col, "")
        return row


def infer_role(hart: str) -> str:
    if hart == "DM0":
        return "dm0_isr"
    if hart == "DM1":
        return "dm1_remapper"
    if hart.startswith("DM"):
        return "dm_local"
    return "trisc_local"


def parse_log_file(path: Path) -> list[TimingRow]:
    text = path.read_text(encoding="utf-8", errors="replace")
    case_name = path.stem
    log_file = path.name

    result_match = TEST_RESULT_RE.search(text)
    test_result = result_match.group(1) if result_match else "UNKNOWN"

    rows: list[TimingRow] = []
    current_benchmark = ""

    for line in text.splitlines():
        header_match = TIMING_HEADER_RE.search(line)
        if header_match:
            current_benchmark = header_match.group("benchmark")
            continue

        if not current_benchmark:
            continue

        for regex, role in (
            (DM0_RE, "dm0_isr"),
            (DM1_RE_LEGACY, "dm1_remapper"),
            (DM1_RE, "dm1_remapper"),
            (DM_LOCAL_RE, "dm_local"),
            (TRISC_RE, "trisc_local"),
        ):
            match = regex.search(line)
            if match:
                data = match.groupdict()
                hart = data.pop("hart")
                e2e = int(data.pop("e2e"))
                start = int(data.pop("start"))
                end = int(data.pop("end"))
                # Optional named groups (legacy field aliases) are None when absent.
                details = {k: int(v) for k, v in data.items() if v is not None}
                rows.append(
                    TimingRow(
                        log_file=log_file,
                        case_name=case_name,
                        benchmark_name=current_benchmark,
                        test_result=test_result,
                        hart=hart,
                        role=role,
                        written=True,
                        e2e=e2e,
                        start=start,
                        end=end,
                        details=details,
                    )
                )
                break
        else:
            not_written = NOT_WRITTEN_RE.search(line)
            if not_written:
                hart = not_written.group("hart")
                rows.append(
                    TimingRow(
                        log_file=log_file,
                        case_name=case_name,
                        benchmark_name=current_benchmark,
                        test_result=test_result,
                        hart=hart,
                        role=infer_role(hart),
                        written=False,
                    )
                )

    return rows


def summarize(rows: Iterable[TimingRow]) -> list[dict[str, str | int]]:
    by_case: dict[tuple[str, str, str], list[TimingRow]] = {}
    for row in rows:
        if not row.written:
            continue
        key = (row.log_file, row.case_name, row.benchmark_name)
        by_case.setdefault(key, []).append(row)

    summaries: list[dict[str, str | int]] = []
    for (log_file, case_name, benchmark_name), case_rows in sorted(by_case.items()):
        worst = max(case_rows, key=lambda r: r.e2e or 0)
        dm0 = next((r for r in case_rows if r.hart == "DM0"), None)
        dm1 = next((r for r in case_rows if r.hart == "DM1"), None)
        dm_prod = [r for r in case_rows if r.hart.startswith("DM") and r.hart not in {"DM0", "DM1"}]
        neo = [r for r in case_rows if r.hart.startswith("Neo")]

        max_dm = max(dm_prod, key=lambda r: r.e2e or 0) if dm_prod else None
        max_neo = max(neo, key=lambda r: r.e2e or 0) if neo else None

        summaries.append(
            {
                "log_file": log_file,
                "case_name": case_name,
                "benchmark_name": benchmark_name,
                "test_result": case_rows[0].test_result,
                "num_harts_written": len(case_rows),
                "worst_e2e": worst.e2e,
                "worst_hart": worst.hart,
                "dm0_e2e": dm0.e2e if dm0 else "",
                "dm1_e2e": dm1.e2e if dm1 else "",
                "max_dm2_7_e2e": max_dm.e2e if max_dm else "",
                "max_dm2_7_hart": max_dm.hart if max_dm else "",
                "max_neo_e2e": max_neo.e2e if max_neo else "",
                "max_neo_hart": max_neo.hart if max_neo else "",
            }
        )
    return summaries


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            candidate = Path(pattern)
            if candidate.is_file():
                paths.append(candidate)
    # Preserve order while deduplicating.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="+",
        help="Log file paths or glob patterns (e.g. case-*-75.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dfb_init_timing.csv"),
        help="Per-hart CSV output path (default: dfb_init_timing.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional per-case worst-e2e summary CSV path",
    )
    args = parser.parse_args()

    log_paths = expand_inputs(args.logs)
    if not log_paths:
        print("No log files matched.", file=sys.stderr)
        return 1

    all_rows: list[TimingRow] = []
    for path in log_paths:
        parsed = parse_log_file(path)
        if not parsed:
            print(f"warning: no DFB init timing block in {path}", file=sys.stderr)
        all_rows.extend(parsed)

    if not all_rows:
        print("No timing rows parsed.", file=sys.stderr)
        return 1

    write_csv(args.output, ROW_COLUMNS, (row.to_csv_row() for row in all_rows))
    print(f"Wrote {len(all_rows)} rows to {args.output}")

    if args.summary:
        summary_rows = summarize(all_rows)
        write_csv(args.summary, SUMMARY_COLUMNS, summary_rows)
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
