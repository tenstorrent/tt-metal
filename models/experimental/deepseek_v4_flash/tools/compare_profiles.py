# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Region-by-region diff of two ttnn ``ops_perf_results_*.csv`` profiles.

The ttnn/Tracy op profile is a *flat* list of device operations, but the
``DeepSeekV4DecoderLayer`` decode path wraps each sub-module in a ``_region``
(``models/experimental/deepseek_v4_flash/tt/common.py``) that emits
``<NAME>_START`` / ``<NAME>_END`` signpost pseudo-rows into the CSV. This tool
uses those markers to slice out a single named region (e.g. ``ATTENTION`` or
``MOE``) from *both* CSVs and compares the op sequences side by side.

For each region the op sequence repeats once per captured layer/iteration, so a
region occurs many times in a file. Per-op fields that are invariant across
occurrences (op code, core count, input/output shapes + memory configs) are read
from the first occurrence; the device-kernel duration is aggregated (mean by
default) across all occurrences to smooth run-to-run jitter. The two files are
then aligned by op position and only the differences are highlighted.

Because ``MOE`` nests ``MOE_ROUTER`` / ``MOE_EXPERTS`` / ``MOE_SHARED``, selecting
``MOE`` includes every device op between ``MOE_START`` and ``MOE_END`` (i.e. the
sub-regions too); select a sub-region name directly to narrow it.

Usage:
    python compare_profiles.py A.csv B.csv --list
    python compare_profiles.py A.csv B.csv --region ATTENTION
    python compare_profiles.py A.csv B.csv --region MOE --agg median
    python compare_profiles.py A.csv B.csv --region ATTENTION --occurrence 0
"""

from __future__ import annotations

import argparse
import csv
import difflib
import statistics
import sys
from dataclasses import dataclass, field

# ttnn dumps very wide rows (100+ columns); the attributes / memory-config cells
# can be long, so lift the field-size cap.
csv.field_size_limit(sys.maxsize)

_MAX_TENSORS = 8  # INPUT_0..7 columns present in the CSV schema
_OUT_TENSORS = 2  # OUTPUT_0..1


@dataclass
class Tensor:
    """One input/output tensor's shape + layout signature for a single op."""

    shape: str  # "WxZxYxX" (padded), each dim "p" or "p(logical)" when they differ
    layout: str
    dtype: str
    mem: str

    def __str__(self) -> str:
        return f"{self.shape} {self.layout} {self.dtype} {self.mem}"

    def sig(self) -> tuple:
        return (self.shape, self.layout, self.dtype, self.mem)


@dataclass
class Op:
    """A single device op with the fields the comparison cares about."""

    code: str
    cores: str
    device_ns: float
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]

    def struct_sig(self) -> tuple:
        """Everything except timing -- used to detect real config changes."""
        return (
            self.code,
            self.cores,
            tuple(t.sig() for t in self.inputs),
            tuple(t.sig() for t in self.outputs),
        )


@dataclass
class OpAgg:
    """An op position aggregated across all occurrences of a region."""

    op: Op  # representative (first occurrence): code / cores / shapes
    device_ns: list[float] = field(default_factory=list)

    def dur(self, agg: str) -> float:
        if not self.device_ns:
            return 0.0
        if agg == "median":
            return statistics.median(self.device_ns)
        if agg == "first":
            return self.device_ns[0]
        return statistics.fmean(self.device_ns)

    def token(self) -> tuple:
        """Alignment key: op code + input/output *shapes* (memory config excluded).

        Matching on shapes (not memory) lets the same logical op line up across the
        two runs even when one shards to L1 and the other keeps it in DRAM -- that
        memory-config difference then surfaces in the per-op shape detail.
        """
        return (
            self.op.code,
            tuple(t.shape for t in self.op.inputs),
            tuple(t.shape for t in self.op.outputs),
        )


def align(aggs_a: list[OpAgg], aggs_b: list[OpAgg]) -> list[tuple[OpAgg | None, OpAgg | None]]:
    """Diff-align two op sequences by (op code + shapes) into aligned pairs.

    Uses :class:`difflib.SequenceMatcher` so structurally-identical ops pair up
    (``(a, b)``) while ops that exist in only one run appear as ``(a, None)`` /
    ``(None, b)`` instead of being force-matched by position.
    """
    ta = [x.token() for x in aggs_a]
    tb = [x.token() for x in aggs_b]
    pairs: list[tuple[OpAgg | None, OpAgg | None]] = []
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((aggs_a[i1 + k], aggs_b[j1 + k]))
        else:  # replace / delete / insert -> emit one-sided rows
            for k in range(i1, i2):
                pairs.append((aggs_a[k], None))
            for k in range(j1, j2):
                pairs.append((None, aggs_b[k]))
    return pairs


def _dim(cell: str) -> str:
    """``32[1]`` -> ``32(1)`` (padded, logical when it differs); ``32[32]`` -> ``32``."""
    cell = (cell or "").strip()
    if not cell:
        return "?"
    if "[" in cell:
        pad, log = cell.split("[", 1)
        pad, log = pad.strip(), log.rstrip("]").strip()
        return pad if pad == log else f"{pad}({log})"
    return cell


def _tensor(row: dict, prefix: str) -> Tensor | None:
    """Build a :class:`Tensor` from the ``{prefix}_*`` columns, or None if absent."""
    dims = [row.get(f"{prefix}_{a}_PAD[LOGICAL]", "") for a in ("W", "Z", "Y", "X")]
    if not any((d or "").strip() for d in dims):
        return None
    return Tensor(
        shape="x".join(_dim(d) for d in dims),
        layout=(row.get(f"{prefix}_LAYOUT", "") or "?").strip(),
        dtype=(row.get(f"{prefix}_DATATYPE", "") or "?").strip(),
        mem=(row.get(f"{prefix}_MEMORY", "") or "?").strip(),
    )


def _op_from_row(row: dict) -> Op:
    def num(col: str) -> float:
        try:
            return float(row.get(col) or 0)
        except ValueError:
            return 0.0

    inputs = tuple(t for i in range(_MAX_TENSORS) if (t := _tensor(row, f"INPUT_{i}")))
    outputs = tuple(t for i in range(_OUT_TENSORS) if (t := _tensor(row, f"OUTPUT_{i}")))
    return Op(
        code=row["OP CODE"],
        cores=(row.get("CORE COUNT", "") or "?").strip(),
        device_ns=num("DEVICE KERNEL DURATION [ns]"),
        inputs=inputs,
        outputs=outputs,
    )


def extract_region(path: str, region: str) -> list[list[Op]]:
    """All occurrences of ``region`` in ``path`` as a list of op sequences.

    Walks the rows with a signpost stack; an op belongs to the occurrence if the
    target region is anywhere on the stack (so ``MOE`` picks up its sub-regions).
    A new occurrence starts each time ``region`` is (re-)pushed onto an empty-of-
    that-region stack.
    """
    occurrences: list[list[Op]] = []
    stack: list[str] = []
    current: list[Op] | None = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row["OP CODE"]
            if code.endswith("_START"):
                name = code[: -len("_START")]
                stack.append(name)
                if name == region and current is None:
                    current = []
                    occurrences.append(current)
                continue
            if code.endswith("_END"):
                name = code[: -len("_END")]
                if name in stack:
                    del stack[stack.index(name) :]
                if name == region:
                    current = None
                continue
            if current is not None and region in stack:
                current.append(_op_from_row(row))
    return occurrences


def list_regions(path: str) -> list[tuple[str, int]]:
    """Region names (in first-seen order) and their occurrence counts."""
    counts: dict[str, int] = {}
    order: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row["OP CODE"]
            if code.endswith("_START"):
                name = code[: -len("_START")]
                if name not in counts:
                    counts[name] = 0
                    order.append(name)
                counts[name] += 1
    return [(n, counts[n]) for n in order]


def aggregate(occurrences: list[list[Op]], occurrence: int | None) -> tuple[list[OpAgg], set[int]]:
    """Collapse occurrences into per-position :class:`OpAgg`.

    Returns the aggregated ops plus the set of positions where op *structure*
    (code/cores/shapes) varied across occurrences (a red flag for alignment).
    Positions are aligned by index; occurrences with differing lengths still
    contribute to the positions they cover.
    """
    if occurrence is not None:
        occurrences = [occurrences[occurrence]]
    width = max((len(o) for o in occurrences), default=0)
    aggs: list[OpAgg] = []
    unstable: set[int] = set()
    for i in range(width):
        rep: Op | None = None
        sigs: set[tuple] = set()
        durs: list[float] = []
        for occ in occurrences:
            if i < len(occ):
                op = occ[i]
                rep = rep or op
                sigs.add(op.struct_sig())
                durs.append(op.device_ns)
        if rep is None:
            continue
        if len(sigs) > 1:
            unstable.add(i)
        aggs.append(OpAgg(op=rep, device_ns=durs))
    return aggs, unstable


def _us(ns: float) -> float:
    return ns / 1e3


def _fmt_pct(a: float, b: float) -> str:
    if a == 0:
        return "  n/a " if b == 0 else " +inf "
    return f"{100 * (b - a) / a:+6.1f}%"


def _tensor_lines(label: str, ta: Tensor | None, tb: Tensor | None) -> list[str]:
    """A '<label>' line showing A, and B only when it differs from A."""
    if ta is None and tb is None:
        return []
    if ta is None:
        return [f"      {label:<5} (only B) {tb}"]
    if tb is None:
        return [f"      {label:<5} (only A) {ta}"]
    if ta.sig() == tb.sig():
        return [f"      {label:<5} {ta}"]
    return [f"      {label:<5} A: {ta}", f"      {'':<5} B: {tb}"]


def compare(
    name_a: str,
    aggs_a: list[OpAgg],
    unstable_a: set[int],
    name_b: str,
    aggs_b: list[OpAgg],
    unstable_b: set[int],
    region: str,
    n_occ_a: int,
    n_occ_b: int,
    agg: str,
    show_shapes: bool,
) -> None:
    print()
    print("=" * 88)
    print(f"  REGION: {region}     duration = {agg} of DEVICE KERNEL DURATION across occurrences")
    print(f"    A = {name_a}   ({n_occ_a} occurrences, {len(aggs_a)} ops/occurrence)")
    print(f"    B = {name_b}   ({n_occ_b} occurrences, {len(aggs_b)} ops/occurrence)")
    if unstable_a or unstable_b:
        print(
            f"    ! op shapes vary across occurrences (A: {len(unstable_a)} positions, B: {len(unstable_b)}); "
            f"durations are aggregated over mixed shapes, detail is from occurrence 0. "
            f"Use --occurrence N to pin one."
        )
    print("=" * 88)
    print("  mark:  '='=matched op   '-'=only in A (removed)   '+'=only in B (added)")
    print(f"  {'':>2} {'op code':<38}{'cores':>11}{'A [us]':>10}{'B [us]':>10}{'Δ':>8}")
    print("-" * 88)

    pairs = align(aggs_a, aggs_b)
    tot_a = tot_b = 0.0
    n_match = n_only_a = n_only_b = 0
    for a, b in pairs:
        da = a.dur(agg) if a else 0.0
        db = b.dur(agg) if b else 0.0
        tot_a += da
        tot_b += db

        if a and b:
            n_match += 1
            mark = "="
            code = a.op.code
            cores = a.op.cores if a.op.cores == b.op.cores else f"{a.op.cores}/{b.op.cores}"
            print(f"  {mark:>2} {code:<38}{cores:>11}{_us(da):>10.2f}{_us(db):>10.2f}{_fmt_pct(da, db):>8}")
        elif a:
            n_only_a += 1
            print(f"  {'-':>2} {a.op.code:<38}{a.op.cores:>11}{_us(da):>10.2f}{'':>10}{'':>8}")
        else:
            n_only_b += 1
            print(f"  {'+':>2} {b.op.code:<38}{b.op.cores:>11}{'':>10}{_us(db):>10.2f}{'':>8}")

        if not show_shapes:
            continue
        op_a = a.op if a else None
        op_b = b.op if b else None
        ins_a = op_a.inputs if op_a else ()
        ins_b = op_b.inputs if op_b else ()
        outs_a = op_a.outputs if op_a else ()
        outs_b = op_b.outputs if op_b else ()
        lines: list[str] = []
        for j in range(max(len(ins_a), len(ins_b))):
            lines += _tensor_lines(f"in{j}", ins_a[j] if j < len(ins_a) else None, ins_b[j] if j < len(ins_b) else None)
        for j in range(max(len(outs_a), len(outs_b))):
            lines += _tensor_lines(
                f"out{j}", outs_a[j] if j < len(outs_a) else None, outs_b[j] if j < len(outs_b) else None
            )
        for ln in lines:
            print(ln)

    print("-" * 88)
    print(f"  matched={n_match}   only-A={n_only_a}   only-B={n_only_b}")
    print(f"  {'TOTAL region':<41}{'':>11}{_us(tot_a):>10.2f}{_us(tot_b):>10.2f}{_fmt_pct(tot_a, tot_b):>8}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_a", help="baseline ops_perf_results_*.csv (A)")
    ap.add_argument("csv_b", help="comparison ops_perf_results_*.csv (B)")
    ap.add_argument("--region", help="region name to compare (e.g. ATTENTION, MOE)")
    ap.add_argument("--list", action="store_true", help="list regions + occurrence counts in both files and exit")
    ap.add_argument(
        "--agg",
        choices=("mean", "median", "first"),
        default="mean",
        help="how to aggregate device kernel duration across a region's occurrences (default: mean)",
    )
    ap.add_argument(
        "--occurrence",
        type=int,
        default=None,
        help="use only this occurrence index (0-based) instead of aggregating across all",
    )
    ap.add_argument("--no-shapes", action="store_true", help="hide the per-op input/output shape + memory detail")
    args = ap.parse_args()

    if args.list or not args.region:
        ra = dict(list_regions(args.csv_a))
        rb = dict(list_regions(args.csv_b))
        names = list(ra) + [n for n in rb if n not in ra]
        print(f"\n  {'region':<18}{'A occ':>8}{'B occ':>8}")
        print("  " + "-" * 32)
        for n in names:
            print(f"  {n:<18}{ra.get(n, 0):>8}{rb.get(n, 0):>8}")
        print()
        if not args.region:
            print("  pass --region <NAME> to compare a region.")
            return 0

    occ_a = extract_region(args.csv_a, args.region)
    occ_b = extract_region(args.csv_b, args.region)
    if not occ_a or not occ_b:
        missing = args.csv_a if not occ_a else args.csv_b
        print(f"  region {args.region!r} not found in {missing}")
        return 1

    aggs_a, unstable_a = aggregate(occ_a, args.occurrence)
    aggs_b, unstable_b = aggregate(occ_b, args.occurrence)
    compare(
        args.csv_a,
        aggs_a,
        unstable_a,
        args.csv_b,
        aggs_b,
        unstable_b,
        args.region,
        len(occ_a),
        len(occ_b),
        args.agg,
        show_shapes=not args.no_shapes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
