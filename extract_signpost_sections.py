#!/usr/bin/env python3
"""Extract rows from a Tracy ops_perf_results CSV that fall between
`shared_expert_and_dispatch_start` and `shared_expert_and_dispatch_end`
signposts. Handles multiple start/end pairs by concatenating all matching
sections into a single output CSV (header and the bracketing signpost rows
are preserved so section boundaries remain visible).

With --summary, also prints per-section, per-op-invocation min/max of
DEVICE FW START CYCLE and DEVICE FW END CYCLE across the devices that
ran that invocation. An invocation is a consecutive run of rows with the
same OP CODE; if a DEVICE ID repeats within that run, it is treated as a
new invocation (so e.g. Matmul-Matmul back-to-back on 8 devices is split
into two 8-device invocations, not one 16-device group).
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

START_SIGNPOST = "shared_expert_and_dispatch_start"
END_SIGNPOST = "shared_expert_and_dispatch_end"


@dataclass
class Invocation:
    op_code: str
    device_ids: list[str] = field(default_factory=list)
    fw_starts: list[int] = field(default_factory=list)
    fw_ends: list[int] = field(default_factory=list)


def _to_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract_sections(
    input_path: Path,
    output_path: Path,
    start: str,
    end: str,
) -> tuple[int, int, list[list[Invocation]]]:
    sections_count = 0
    rows_written = 0
    inside = False
    sections: list[list[Invocation]] = []
    current_section: list[Invocation] = []
    current_invocation: Invocation | None = None

    def flush_invocation() -> None:
        nonlocal current_invocation
        if current_invocation is not None:
            current_section.append(current_invocation)
            current_invocation = None

    with input_path.open("r", newline="") as f_in, output_path.open("w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input CSV {input_path} is empty")
        writer.writerow(header)

        def col_idx(name: str, fallback: int) -> int:
            try:
                return header.index(name)
            except ValueError:
                return fallback

        op_code_idx = col_idx("OP CODE", 0)
        op_type_idx = col_idx("OP TYPE", 1)
        device_id_idx = col_idx("DEVICE ID", 3)
        fw_start_idx = col_idx("DEVICE FW START CYCLE", 13)
        fw_end_idx = col_idx("DEVICE FW END CYCLE", 14)

        def get(row: list[str], idx: int) -> str:
            return row[idx] if len(row) > idx else ""

        for row in reader:
            if not row:
                continue
            op_code = get(row, op_code_idx)
            op_type = get(row, op_type_idx)
            is_signpost = op_type == "signpost"

            if is_signpost and op_code == start:
                if inside:
                    flush_invocation()
                inside = True
                sections_count += 1
                current_section = []
                sections.append(current_section)
                writer.writerow(row)
                continue
            if is_signpost and op_code == end:
                if inside:
                    flush_invocation()
                    writer.writerow(row)
                inside = False
                continue

            if not inside:
                continue

            writer.writerow(row)
            rows_written += 1

            device_id = get(row, device_id_idx)
            fw_start = _to_int(get(row, fw_start_idx))
            fw_end = _to_int(get(row, fw_end_idx))

            need_new = (
                current_invocation is None
                or current_invocation.op_code != op_code
                or device_id in current_invocation.device_ids
            )
            if need_new:
                flush_invocation()
                current_invocation = Invocation(op_code=op_code)

            current_invocation.device_ids.append(device_id)
            if fw_start is not None:
                current_invocation.fw_starts.append(fw_start)
            if fw_end is not None:
                current_invocation.fw_ends.append(fw_end)

        if inside:
            flush_invocation()

    return sections_count, rows_written, sections


def print_summary(sections: list[list[Invocation]]) -> None:
    if not sections:
        print("(no sections found)")
        return

    for sec_i, invocations in enumerate(sections, start=1):
        print(f"\n=== Section {sec_i} ({len(invocations)} op invocation(s)) ===")
        op_w = max((len(inv.op_code) for inv in invocations), default=8)
        header = (
            f"  {'#':>3}  {'op_code':<{op_w}}  {'devs':>4}  "
            f"{'fw_start_min':>14}  {'fw_start_max':>14}  {'start_spread':>12}  "
            f"{'fw_end_min':>14}  {'fw_end_max':>14}  {'end_spread':>12}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, inv in enumerate(invocations, start=1):
            s_min = min(inv.fw_starts) if inv.fw_starts else None
            s_max = max(inv.fw_starts) if inv.fw_starts else None
            e_min = min(inv.fw_ends) if inv.fw_ends else None
            e_max = max(inv.fw_ends) if inv.fw_ends else None
            s_spread = (s_max - s_min) if s_min is not None and s_max is not None else None
            e_spread = (e_max - e_min) if e_min is not None and e_max is not None else None

            def fmt(v: int | None) -> str:
                return "-" if v is None else str(v)

            print(
                f"  {i:>3}  {inv.op_code:<{op_w}}  {len(inv.device_ids):>4}  "
                f"{fmt(s_min):>14}  {fmt(s_max):>14}  {fmt(s_spread):>12}  "
                f"{fmt(e_min):>14}  {fmt(e_max):>14}  {fmt(e_spread):>12}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input ops_perf_results CSV path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: shared_expert_and_dispatch_overlap_sections.csv next to the input)",
    )
    parser.add_argument("--start", default=START_SIGNPOST, help=f"Start signpost (default: {START_SIGNPOST})")
    parser.add_argument("--end", default=END_SIGNPOST, help=f"End signpost (default: {END_SIGNPOST})")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print per-section, per-op-invocation min/max of DEVICE FW START/END CYCLE to stdout",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if args.output else input_path.with_name("shared_expert_and_dispatch_overlap_sections.csv")
    )

    sections_count, rows, sections = extract_sections(input_path, output_path, args.start, args.end)
    print(f"extracted {sections_count} section(s), {rows} row(s) -> {output_path}")

    if args.summary:
        print_summary(sections)

    return 0


if __name__ == "__main__":
    sys.exit(main())
