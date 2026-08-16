# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Drop the columns nothing in this tree reads, so the windows fit the repo's limit.

``tt-metal``'s ``check-large-files`` pre-commit hook rejects any artifact over
**500 KB**. The three archived profile windows are already the *windowed*
captures -- 14048, 14048 and 18424 rows, one verified iteration each, cut by
``window_full_model_48{,_prefill}.py`` out of a ~139 MB raw capture -- so there
is no row left to trim without breaking the boundary check that makes them
evidence. What there is instead is **width**: Tracy writes 128 columns and the
analysis in this tree reads eleven of them.

So the archived copies keep the columns the analysis consumes, at 2728/2732/2308
KB -> 259/262/350 KB, and the full 128-column capture stays regenerable from the
command in ``README.md``'s reproduction block.

**The kept set is not a guess.** ``KEPT`` below is the union of every column
name that appears in a subscript of a CSV row anywhere in ``probes/``, widened
deliberately -- see ``CONSUMED`` for the eleven that are actually read and the
notes on each block for why the rest are kept anyway. ``--audit`` re-derives
``CONSUMED`` by scanning the consumers, so this docstring cannot drift from
them:

    python reduce_profile_csv.py --audit

and the reduction itself is::

    python reduce_profile_csv.py ../ops_perf_full_model_48layer_decode.csv.gz
    python reduce_profile_csv.py ../ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz
    python reduce_profile_csv.py ../ops_perf_full_model_48layer_prefill_s128.csv.gz

Re-running it on an already-reduced file is a no-op, so it is safe to repeat.

**What is lost.** ``tt-perf-report`` reads ~40 columns including ``ATTRIBUTES``,
``PM IDEAL [ns]`` per-RISC breakdowns and the DRAM/NoC utilisations, and it
cannot be re-run from a reduced file. Its output is already archived whole, as
``tt_perf_report_full_model_48layer_*.txt.gz``. ``ATTRIBUTES`` in particular is
**dropped**: it is 280 KB compressed on the decode window on its own -- more
than half the entire budget -- nothing under ``probes/`` subscripts it, and no
figure in ``README.md`` or either work log is derived from it. Keeping it would
have put every one of the three files back over the limit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The repo's ``check-large-files`` ceiling, in KB.
LIMIT_KB = 500

#: The columns some script under ``probes/`` actually subscripts. Kept in sync
#: with the scripts by ``--audit``, which greps them rather than trusting this.
CONSUMED = (
    "OP CODE",  # every windower, ranker and analysis
    "DEVICE ID",  # the per-device split, in all of them
    "CORE COUNT",  # the ``cores`` column of both rankings
    "DEVICE KERNEL DURATION [ns]",  # every published microsecond
    # ``rank_full_model_48.shape(row, i)``, for i in (0, 1) only -- the operand
    # shapes that tell two call sites of the same op code apart, and that
    # ``profile_summary.py`` uses to find the LM head by its 37984 vocabulary
    # shard.
    "INPUT_0_W_PAD[LOGICAL]",
    "INPUT_0_Z_PAD[LOGICAL]",
    "INPUT_0_Y_PAD[LOGICAL]",
    "INPUT_0_X_PAD[LOGICAL]",
    "INPUT_1_W_PAD[LOGICAL]",
    "INPUT_1_Z_PAD[LOGICAL]",
    "INPUT_1_Y_PAD[LOGICAL]",
    "INPUT_1_X_PAD[LOGICAL]",
)

#: What is archived. ``CONSUMED`` plus a margin, because a column that turns out
#: to be needed costs a re-capture on hardware and a column that turns out not to
#: be needed costs a few KB. Each block says what it is for.
KEPT = (
    # --- read by the consumers -------------------------------------------
    "OP CODE",
    "DEVICE ID",
    "CORE COUNT",
    "DEVICE KERNEL DURATION [ns]",
    # --- identity of the row, so a reduced row is still traceable back to a
    # --- re-capture: which op, in what order, on what part
    "OP TYPE",
    "GLOBAL CALL COUNT",
    "DEVICE ARCH",
    "MATH FIDELITY",
    "PROGRAM CACHE HIT",
    # --- the other two durations any "is this op slow, or is it waiting"
    # --- question needs. ``OP TO OP LATENCY`` is the dispatch gap; the
    # --- MoE-skew analysis regresses on kernel duration today but the gap is
    # --- the natural next question and it is 21 KB.
    "OP TO OP LATENCY [ns]",
    "DEVICE FW DURATION [ns]",
    # --- operand and result shapes/layouts. Only INPUT_0/1's four PAD columns
    # --- are read today; the LAYOUT/DATATYPE/MEMORY triples come along because
    # --- ``tt-perf-report`` groups by in0 memory config and the README quotes
    # --- that grouping, and OUTPUT_0 because "how big was the result" is asked
    # --- of half the rows in the rankings.
    *(f"INPUT_{i}_{axis}_PAD[LOGICAL]" for i in (0, 1) for axis in "WZYX"),
    *(f"INPUT_{i}_{part}" for i in (0, 1) for part in ("LAYOUT", "DATATYPE", "MEMORY")),
    *(f"OUTPUT_0_{axis}_PAD[LOGICAL]" for axis in "WZYX"),
    "OUTPUT_0_LAYOUT",
    "OUTPUT_0_DATATYPE",
    "OUTPUT_0_MEMORY",
    # --- the three utilisation figures the Limitations section reasons about
    # --- (LM-head DRAM bandwidth headroom, FPU utilisation on the experts).
    "PM IDEAL [ns]",
    "PM FPU UTIL (%)",
    "DRAM BW UTIL (%)",
)

#: The consumers ``--audit`` scans. Every script under ``probes/`` that opens one
#: of the three archived windows.
CONSUMERS = (
    "profile_summary.py",
    "rank_full_model_48.py",
    "window_full_model_48.py",
    "window_full_model_48_prefill.py",
    "moe_skew_analysis.py",
    "check_published_figures.py",
    "mutation_test_checker.py",
)

#: ``row["COLUMN NAME"]`` and ``row[f"INPUT_{i}_{a}_PAD[LOGICAL]"]``. The second
#: form is why this is a regex over the source and not an import: the column name
#: is assembled at run time, so it has to be expanded by hand -- see ``_expand``.
SUBSCRIPT = re.compile(r"""\[\s*f?["']([A-Z][^"']*)["']\s*\]""")


def _expand(name: str) -> list[str]:
    """Expand the one f-string column template the consumers use."""
    if "{i}" in name and "{a}" in name:
        return [name.replace("{i}", str(i)).replace("{a}", a) for i in (0, 1) for a in "WZYX"]
    return [name] if "{" not in name else []


def audit(capture: Path) -> int:
    """Re-derive the consumed column set from the consumers, and check it.

    ``capture`` supplies the authoritative list of column names -- the header of
    a real capture, reduced or not. Anything the consumers subscript that is not
    in it is a subscript of something else (a ``profile_summary`` JSON payload
    keyed by op label, say) and is reported separately rather than silently
    treated as a column.
    """
    with gzip.open(capture, "rt", newline="") as handle:
        header = set(csv.DictReader(handle).fieldnames or ())

    found: dict[str, list[str]] = {}
    for consumer in CONSUMERS:
        text = (HERE / consumer).read_text(encoding="utf-8")
        for match in SUBSCRIPT.findall(text):
            for column in _expand(match):
                found.setdefault(column, []).append(consumer)

    not_columns = sorted(c for c in found if c not in header and c not in KEPT)
    columns = {c: sorted(set(v)) for c, v in found.items() if c not in not_columns}
    print(f"header of {capture.name}: {len(header)} columns")
    print(f"scanned {len(CONSUMERS)} consumers; {len(columns)} distinct column subscripts:")
    for column, users in sorted(columns.items()):
        mark = "kept" if column in KEPT else "NOT KEPT"
        print(f"  {mark:<8} {column:<32} {', '.join(users)}")
    if not_columns:
        print("\nsubscripts that are not column names of this capture (inspect these by eye --")
        print("they should all be dict keys of something other than a profile row):")
        for name in not_columns:
            print(f"  {name}    {', '.join(sorted(set(found[name])))}")

    missing = sorted(set(CONSUMED) - set(columns))
    extra = sorted(set(columns) - set(CONSUMED))
    if missing:
        print(f"\nCONSUMED lists columns no consumer subscripts: {missing}")
    if extra:
        print(f"\nconsumers subscript columns CONSUMED does not list: {extra}")
    unkept = sorted(set(columns) - set(KEPT))
    if unkept:
        print(f"\nNOT KEPT but subscripted: {unkept}")
        return 1
    if missing or extra:
        return 1
    print("\nevery subscripted column is in KEPT, and CONSUMED lists exactly them.")
    return 0


def reduce_one(path: Path, dry_run: bool = False) -> None:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)

    keep = [c for c in KEPT if c in fieldnames]
    dropped = [c for c in fieldnames if c not in keep]
    if not dropped:
        print(f"{path.name}: already reduced ({len(keep)} columns, {len(rows)} rows) -- nothing to do")
        return

    missing = [c for c in KEPT if c not in fieldnames]
    if missing:
        raise SystemExit(f"{path.name}: capture is missing columns this reducer keeps: {missing}")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keep, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    before = path.stat().st_size
    if dry_run:
        probe = io.BytesIO()
        with gzip.open(probe, "wt", newline="") as handle:
            handle.write(buffer.getvalue())
        after = probe.tell()
    else:
        # The mutation tester's scratch trees hard-link these files, so writing
        # through the existing inode would corrupt a tree that is meant to be a
        # pristine copy. Unlink first, exactly as ``mutation_test_checker.py``
        # does for the same reason.
        path.unlink()
        with gzip.open(path, "wt", newline="") as handle:
            handle.write(buffer.getvalue())
        after = path.stat().st_size

    print(
        f"{path.name}: {len(fieldnames)} -> {len(keep)} columns, {len(rows)} rows kept, "
        f"{before/1024:,.0f} KB -> {after/1024:,.0f} KB"
        f"{'  OVER THE %d KB LIMIT' % LIMIT_KB if after > LIMIT_KB * 1024 else ''}"
    )
    print(f"  dropped: {', '.join(dropped)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="*", help="gzipped windowed ops CSVs to reduce in place")
    parser.add_argument("--audit", action="store_true", help="re-derive the consumed column set and check KEPT")
    parser.add_argument(
        "--audit-capture",
        type=Path,
        default=HERE.parent / "ops_perf_full_model_48layer_decode.csv.gz",
        help="the capture whose header names the columns --audit checks against",
    )
    parser.add_argument("--dry-run", action="store_true", help="report the sizes without rewriting")
    args = parser.parse_args()

    status = audit(args.audit_capture) if args.audit else 0
    for path in args.csv:
        reduce_one(path, dry_run=args.dry_run)
    return status


if __name__ == "__main__":
    sys.exit(main())
