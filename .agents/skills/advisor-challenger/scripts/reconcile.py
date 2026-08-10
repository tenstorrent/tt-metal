#!/usr/bin/env python3
"""Reconcile the advisor's placement against the shipped graph, and account for every device op.

  reconcile.py --report shard_advise/<kind>/report.json \
               --ir     shard_advise/<kind>/final_ir.mlir \
               --perf   tracy/incumbent_<kind>_ops.csv \
               --incumbent incumbent.json \
               --layer-kind <kind> --layers-of-kind N --total-layers M \
               --out    reconciliation_<kind>.json

  reconcile.py --legal-grids <tensor-width-in-tiles>   # the legal core-count ladder, no other inputs
  reconcile.py --self-test                             # regression fixtures, no inputs needed

Partitions the measured window -- every device op gets exactly one bucket, and the buckets sum to 100 %:

  chain:<id>            in a maximal L1-resident run whose placement differs from shipped
  boundary              a conversion op -- what joining the chains either side would remove
  dram_resident         the advisor placed it in DRAM; that is advice too
  advisor_unfixable     the advisor declared it UNPLACEABLE with the exact runtime error. Its layout in
                        ops[] is a fallback after a declared failure, not advice, so there is nothing
                        here to screen -- see honour_unfixable()
  agrees_with_shipped   advised == shipped, in BOTH the memory space and the grid (or the DS family)
  untraced              in the profile, absent from the advisor's graph

Then answers the two questions that decide whether screening is worth device time:

  feasibility           is the advisor's ceiling above this harness's own noise floor? A chain worth less
                        than the spread of the incumbent's repeats cannot be resolved by non-overlap, so
                        measuring it returns a zero that says nothing about the advice. The ceiling has
                        TWO channels (see cliff_candidates); a zero on the first is not a stop condition.
  advised_boundaries    of the shipped conversions, which does the advice NOT place? Only those are
                        attributable to the advisor; the rest are real cost belonging to $optimize.

Chains are ranked by that attributable value, NOT by the advised ops' window share: a chain-lengthening
change removes conversions, which are separate device ops an op-share threshold never counts.

What it is authoritative about, and what it is only suggesting, is in the `confidence` block of every
output, alongside `limitations`. It fails loudly -- a report covering more than one replay, an accounting
that does not close, or inputs that appear to describe different graphs are errors, not silent gaps.

WHY final_ir.mlir IS NOW AN INPUT (v3). report.json is a lossy summary of the advice in two ways that
each produced a wrong published conclusion in the v2 corpus: its `cores=(x0,y0)-(x1,y1)` field is the
FIRST RANGE ONLY of a multi-range CoreRangeSet -- 58.3 % of advised core counts corpus-wide came out
understated, and 34.4 % of the "disagreement" it reported was phantom -- and it carries no shard shape at
all, so a plan read from it cannot be implemented. The IR carries both. Pass --ir; the tool cross-checks
the two and says which it used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from pathlib import Path

# Pairing is automatic by normalised name (see matches()). These two tables hold only what that cannot
# reach, and both are kept SMALL and GENERAL on purpose -- enumerating the op names of the models you
# happen to have seen is how this script goes stale.
#
# RENAMES: a ttnn op whose device class is a different word.
TTNN_TO_DEVICE = {
    "rms_norm": "LayerNorm",                 # rms_norm is served by the layernorm device op
    "linear": "Matmul",
    "concatenate_heads": "NLPConcatHeads",
    "transpose": "Permute",                  # ttnn.transpose lowers to the permute device op
    "paged_scaled_dot_product_attention_decode": "SdpaDecode",
    "scaled_dot_product_attention_decode": "SdpaDecode",
}
# ADVISED OPS EXCLUDED FROM PAIRING: their device counterpart is either nothing at all, or a movement op
# that is already classified by role below. `to_layout` / `to_memory_config` are here for the second reason,
# NOT because they are free -- a to_layout is exactly how a 6.7-10 us retilize enters the graph. Its cost is
# counted, as a `boundary` op on the device side. Excluding them only stops them stealing a compute op.
UNPAIRED_ADVISED = {"to_layout", "to_memory_config", "reshape", "view",
                    "full", "zeros", "ones", "empty", "arange"}
# MOVEMENT device ops -- their only effect is placement or layout, so they are chain BOUNDARIES whatever
# the advisor said about them. Matched by PREFIX, because tt-metal spells variants: UntilizeWithUnpadding,
# TilizeWithValPadding. Ops that also compute (Slice, Permute, Concat, Repeat) are deliberately absent:
# those pair with an advised op and belong to a chain.
MOVEMENT_PREFIXES = (("Reshard", "l1_regrid"), ("ShardedToInterleaved", "l1_to_dram"),
                     ("InterleavedToSharded", "dram_to_l1"), ("Untilize", "retilize"),
                     ("Tilize", "retilize"), ("ReshapeView", "reshape_view"),
                     ("FillPad", "fill_pad"), ("Copy", "copy"))


# Only these four are placement decisions the advisor states as a to_memory_config edge, so only these can be
# compared against reshards[] and attributed. The rest -- ReshapeView, FillPad, Copy -- are UNRESOLVED, not
# out of scope: a ReshapeView can carry a hidden layout change and act as a chain boundary despite looking
# like a shape-only op, and it is not cheap (8-19 us on 110 and 16 cores here, 88.5 us of one 1355 us
# window). The profile cannot settle it: `Input 0 Memory` gives the memory class with no grid and no output
# column, so even a Reshard that provably regrids 30->40 cores reads as unchanged. Deciding needs the IR.
ADVISOR_COMPARABLE = {"l1_regrid", "l1_to_dram", "dram_to_l1", "retilize"}


def movement_class(dev_cls: str):
    for pref, name in MOVEMENT_PREFIXES:
        if dev_cls.startswith(pref):
            return name
    return None


def device_class(op_code: str) -> str:
    return op_code.split()[0].replace("DeviceOperation", "")


def parse_perf(path: Path, incumbent_ms: float | None = None, layers_in_window: int = 1):
    """Device ops in program order, plus the window total.

    Column names differ between producers and cases: tt-perf-report writes `OP Code` / `Device Time` (us),
    a raw Tracy export writes `OP CODE` / `DEVICE KERNEL DURATION [ns]` / `CORE COUNT`. Look them up
    case-insensitively by substring rather than by exact spelling.
    """
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        sys.exit(f"FATAL: {path} has no rows")
    keys = {(k or "").strip().lower(): k for k in rows[0]}

    def find(*needles, exact=True):
        for n in needles:
            if exact and n in keys:
                return keys[n]
        for n in needles:
            for k, orig in keys.items():
                if n in k:
                    return orig
        return None

    code_col = find("op code", "op type", "name")
    dur_col = find("device time", "device kernel duration [ns]", "device kernel duration")
    core_col = find("cores", "core count")
    mem_col = find("input 0 memory", "input_0_memory")   # memory CLASS only: no grid, no output column
    if code_col is None or dur_col is None:
        sys.exit(f"FATAL: {path}: need an op-code and a duration column; saw {sorted(keys)}")
    ns = "ns" in dur_col.lower() or "cycle" in dur_col.lower()
    scale = 1 / 1000.0 if ns else 1.0

    dsc = find("dram sharded", exact=False)          # hoisted: this used to be looked up once per row
    ops, total = [], 0.0
    for r in rows:
        code = (r.get(code_col) or "").strip()
        if not code:
            continue
        try:
            us = float(str(r[dur_col]).replace(",", "")) * scale
        except (TypeError, ValueError):
            us = 0.0
        cores = None
        if core_col:
            try:
                cores = int(float(str(r.get(core_col, "")).replace(",", "")))
            except (TypeError, ValueError):
                cores = None
        ops.append({"id": r.get(find("id") or "", None), "cls": device_class(code), "code": code,
                    "us": us, "cores": cores,
                    "mem": ((r.get(mem_col) or "").strip() if mem_col else ""),
                    "dram_sharded": (r.get(dsc) or "").strip().lower() in ("true", "1", "yes")
                                    if dsc else False})
        total += us

    # A report that was not bounded to one iteration by signposts repeats its op sequence. Detect that
    # structurally rather than by a magic ratio: shares computed over N iterations are N times too small.
    seq = [o["cls"] for o in ops]
    found_reps = 1
    for period in range(1, len(seq) // 2 + 1):
        if len(seq) % period == 0 and seq == seq[:period] * (len(seq) // period):
            reps = len(seq) // period
            if reps == layers_in_window:
                found_reps = reps
                break          # a deliberate N-consecutive-layer capture, declared with --layers-in-window
            sys.exit(f"FATAL: {path}: the op sequence repeats {reps}x (period {period} of {len(seq)} "
                     f"rows), so this report covers {reps} iterations rather than "
                     f"{layers_in_window}. Re-run tt-perf-report with --start-signpost/--end-signpost "
                     f"bounding one replay, or pass --layers-in-window {reps} if the capture really does "
                     f"cover {reps} consecutive layers; every share computed here would be "
                     f"{reps // max(1, layers_in_window)}x too small.")

    # BUG FIX (v3): a declared --layers-in-window N that the profile does not actually contain used to pass
    # silently, and per_layer_window_us then divided by N -- so the whole model estimate came out N times too
    # small with nothing saying so. Declaring N is a claim about the profile; check it.
    if layers_in_window > 1 and found_reps != layers_in_window:
        sys.exit(f"FATAL: {path}: --layers-in-window {layers_in_window} was declared, but the op sequence "
                 f"does not repeat {layers_in_window}x -- so this profile does not cover "
                 f"{layers_in_window} consecutive layers. Dividing the window by {layers_in_window} would "
                 f"understate every per-layer and per-model number by that factor.")

    # And a window far from the harness's own number means the CSV is not the decode window at all.
    if incumbent_ms is not None and incumbent_ms > 0:
        ratio = total / (incumbent_ms * 1000.0)
        if ratio > 5 or ratio < 0.2:
            sys.exit(f"FATAL: {path}: window {total:.1f} us is {ratio:.1f}x the harness's "
                     f"{incumbent_ms * 1000:.1f} us. Either the harness does not measure the decode "
                     f"path, or the report is not signpost-bounded to one iteration. Shares computed "
                     f"against this window would be wrong.")
    return ops, total


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def matches(ttnn_op: str, dev_cls: str) -> bool:
    """Pair a ttnn op with a device class by normalised name, either direction as a prefix.

    Prefix-either-way covers the common shapes without an entry each: slice_static/Slice, concat/Concat,
    embedding/Embeddings, topk/TopK, repeat/Repeat, the rotary variants. TTNN_TO_DEVICE holds only genuine
    renames.
    """
    a, b = norm(ttnn_op), norm(dev_cls)
    return a == b or a.startswith(b) or b.startswith(a)


def grid_cores(layout: str):
    """The advised core count, from the `AxB` GRID STRING in the layout.

    `l1/height_sharded/32x1 cores=(0,0)-(10,1)` is 32 cores, not 22. The `cores=` field prints only the
    first range of the chosen CoreRangeSet -- here `[(0,0)-(10,1), (0,2)-(9,2)]` = 22 + 10 -- while the
    grid string printed immediately before it is the true coreCount, the value the advisor's own
    LayoutScore compares. Validated against three decision traces (`beam[0].score.coreCount`): the grid
    product was right 22/22, 10/10 and 17/17; the bounding box 2/22, 1/10 and 2/17.

    This also removes a whole class of phantom disagreement. An `l1/interleaved/10x11` advice carries no
    `cores=` field at all, so the old parse produced None and the row could never register as agreement
    even against a shipped op on exactly those 110 cores.
    """
    m = re.search(r"/(\d+)x(\d+)", layout)
    return int(m.group(1)) * int(m.group(2)) if m else None


def bbox_cores(layout: str):
    """The old, lossy reading -- kept only so the two can be reported side by side."""
    m = re.search(r"cores=\((\d+),(\d+)\)-\((\d+),(\d+)\)", layout)
    return ((int(m.group(3)) - int(m.group(1)) + 1) * (int(m.group(4)) - int(m.group(2)) + 1)
            if m else None)


def parse_advised(report: dict):
    """Advised ops in program order, with the core count read off the grid string (see grid_cores)."""
    out = []
    for o in report.get("ops", []):
        lay = o.get("layout", "")
        name = o["op"].replace("ttnn.", "")
        explicit = None if name in UNPAIRED_ADVISED else TTNN_TO_DEVICE.get(name, "__auto__")
        out.append({"index": o["index"], "op": name, "explicit": explicit, "layout": lay,
                    "cores": grid_cores(lay), "cores_bbox": bbox_cores(lay),
                    "program_config": o.get("program_config", "") or "",
                    "space": "dram" if lay.startswith("dram") else
                             ("l1" if lay.startswith("l1") else None)})
    return out


def parse_unfixable(report: dict):
    """op name -> the exact runtime error the advisor recorded for it.

    `unfixable_ops` is the advisor telling you, in writing, which ops it could not place and why -- it
    obtained each string by querying tt-metal's own constraint machinery. For such an op the layout that
    appears in `ops[]` is a FALLBACK AFTER A DECLARED FAILURE, not a recommendation, so the
    `dram_resident` bucket's premise ("the advisor put it in DRAM, that is advice") does not hold and
    there is nothing to screen. In the v2 corpus 54 declarations were made and 41 were presented to the
    cell as screenable advice anyway; cells then spent device time rediscovering the identical error
    string. One cell recorded it back verbatim, twice.
    """
    out = {}
    for x in report.get("unfixable_ops") or []:
        name = str(x.get("op", "")).replace("ttnn.", "").split(".")[-1]
        if name:
            out[name] = x.get("reason", "")
    return out


def legal_ladder(width_tiles: int, max_cores: int = 110, row_width: int = 11):
    """The core-count ladder for a width shard of `width_tiles` tiles, and how it was derived.

    THREE separate mechanisms decide which grids can run, and every v2 cell rediscovered them by
    launching processes that died on TT_FATAL. Two are computable here:

      1. shard padding. `shard_spec_validation.cpp` refuses a grid whose padded width exceeds the tensor
         by a whole shard width -- one where some core would own no data. For C cores each owning
         ceil(W/C) tiles that is `(C - 1) * ceil(W / C) < W`.
      2. the shape the model's own grid helper will build. In practice these accept the exact tile
         divisors of W, plus whole rows of the compute grid (`row_width`, which is 11 on Blackhole and 8
         on Wormhole -- pass --grid-row-width for the part you are on).

    The third -- the op's own layout rulebook -- is not computable from here, which is why this is a bound
    on the sweep and not a proof. Confirm a surprising rung with an isolated single-op test.

    Derived, not fitted: at W = 64 with row_width 11 `ladder` is exactly the set measured on north-mini,
    {1,2,4,8,11,16,22,32,64}, and 40/44/48/55/88 -- which hard-failed there, and which one cell drew a
    conclusion from anyway -- are all excluded by rule 1.
    """
    if width_tiles <= 0:
        return {"width_tiles": width_tiles, "ladder": [], "padding_legal": [], "note": "no width given"}
    padding = [c for c in range(1, max_cores + 1)
               if (c - 1) * -(-width_tiles // c) < width_tiles]
    divisors = {c for c in padding if width_tiles % c == 0}
    rows = {c for c in padding if row_width and c % row_width == 0}
    return {
        "width_tiles": width_tiles, "grid_row_width": row_width, "max_cores": max_cores,
        "ladder": sorted(divisors | rows),
        "exact_divisors": sorted(divisors), "whole_grid_rows": sorted(rows),
        "padding_legal": padding,
        "note": "`ladder` is what to sweep: the exact tile divisors plus whole compute-grid rows, both "
                "filtered by the shard-padding rule. `padding_legal` is the wider set the padding rule "
                "alone admits -- a rung there is worth trying if the ladder is short, but the model's grid "
                "helper often will not build it. Neither set proves the OP will accept a rung: its own "
                "rulebook is not modelled here. Sweep the ladder on BOTH sides of the advised value; the "
                "response is not monotonic, and in the v2 corpus the advised value was once a local "
                "maximum with the optimum below it.",
    }


# One layout definition in final_ir.mlir, e.g.
#   #ttnn_layout22 = #ttnn.ttnn_layout<(...) -> (...), <32x1>,
#       memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>,
#       core_ranges = <[#ttnn.core_range<(0,0), (10,1)>, #ttnn.core_range<(0,2), (9,2)>]>>
_IR_LAYOUT = re.compile(
    r"^#(ttnn_layout\w*)\s*=.*?<(\d+)x(\d+)>,\s*memref<([0-9x]+)x!?[^,]*?tile<(\d+)x(\d+)[^>]*>,\s*#(\w+)>,"
    r"\s*<(\w+)>(.*)$")
# the result type, after the last `->`. Parenthesised for a multi-result op (nlp_create_qkv_heads_decode
# returns query/key/value); take the first result, which is the one the op is named for.
_IR_RESULT = re.compile(r"tensor<([0-9x]+)x[a-z_0-9]+,\s*#(ttnn_layout\w*)>")
_IR_OPNAME = re.compile(r'"ttnn\.([a-z_0-9]+)"')


def parse_ir(path: Path):
    """The advised plan as the advisor actually wrote it: shard shape, grid and true core count per op.

    This is the artefact to implement the advice FROM. report.json has no shard shape anywhere, and a
    plan reconstructed without one is a different plan: a v2 analysis spent a week believing the advice
    was illegal because it guessed the logical shard width (32,48) where the advisor had specified the
    tile-aligned (32,64).

    Returns (layouts, ops). `ops` is in program order and carries the op's RESULT layout, which is the
    placement the advisor chose for it.
    """
    layouts, ops = {}, []
    for line in path.read_text().splitlines():
        m = _IR_LAYOUT.match(line.strip())
        if m:
            name, gy, gx, memref, th, tw, space, kind, tail = m.groups()
            dims = [int(x) for x in memref.split("x")]
            ranges = [(int(a), int(b), int(c), int(d)) for a, b, c, d in
                      re.findall(r"core_range<\((\d+),(\d+)\),\s*\((\d+),(\d+)\)>", tail)]
            layouts[name] = {
                "grid": f"{gy}x{gx}", "grid_cores": int(gy) * int(gx),
                "space": "l1" if space == "l1" else "dram", "layout": kind,
                # the shard's own extent, in elements. dims are TILES; scale by the tile shape.
                "shard_shape": [dims[-2] * int(th), dims[-1] * int(tw)] if len(dims) >= 2 else None,
                "core_ranges": [f"({a},{b})-({c},{d})" for a, b, c, d in ranges] or None,
                "core_range_cores": (sum((c - a + 1) * (d - b + 1) for a, b, c, d in ranges)
                                     if ranges else None),
            }
            continue
        if '"ttnn.' not in line:
            continue
        op = _IR_OPNAME.search(line)
        if not op:
            continue
        res = _IR_RESULT.search(line, line.rfind("->"))
        if not res:
            continue
        shape, lay = res.groups()
        ops.append({"op": op.group(1), "logical_shape": [int(x) for x in shape.split("x")],
                    "layout_ref": lay, "unfixable": "ttnn.validation_unfixable" in line,
                    **{k: v for k, v in (layouts.get(lay) or {}).items()}})
    return layouts, ops


def pair_score(adv, dev_cls, nameless=False):
    """How well an advised op pairs with a device class. 0 = incompatible.

    An exact rename or a normalised-prefix match scores 2. A `nameless` advised op -- one that matches no
    device class anywhere in this profile -- scores 1 against anything, so it can pair by position alone.
    tt-metal funnels whole op families through one device class (`neg` and `sigmoid` both run as `Unary`)
    and that mapping is not enumerable from here, so position is the only remaining evidence. Scoring it
    below a real match means the alignment falls back to position only when nothing better lines up; an
    unconditional fallback lets any unmatched op grab a distant device op and drag everything between it
    into `untraced`.
    """
    want = adv["explicit"]
    if want != "__auto__":
        return 2 if dev_cls == want else 0
    if matches(adv["op"], dev_cls):
        return 2
    return 1 if nameless else 0


def align(advised, device):
    """Align the two program-order sequences by dynamic programming.

    Both sides have insertions -- the advised graph omits ops the tracer never saw and contains ops with
    no device counterpart, while the profile contains ops the advisor never placed. A greedy scan is not
    adequate: one unpairable advised op consumes an unbounded run of device ops into `untraced`, which
    understates what the advisor saw. This is an LCS with a weighted match, so a poor local pairing loses
    to leaving both sides unpaired.

    Returns [(advised|None, device|None)] in program order. Movement ops are excluded by the caller: they
    are boundaries whatever the advisor said, so they must not participate in pairing.
    """
    n, m = len(advised), len(device)
    # an advised op that matches no device class here may pair by position alone; see pair_score
    nameless = [not any(pair_score(a, d["cls"]) == 2 for d in device) for a in advised]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            sc = pair_score(advised[i], device[j]["cls"], nameless[i])
            best = max(dp[i + 1][j], dp[i][j + 1])
            if sc:
                best = max(best, sc + dp[i + 1][j + 1])
            dp[i][j] = best
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        sc = pair_score(advised[i], device[j]["cls"], nameless[i])
        if sc and dp[i][j] == sc + dp[i + 1][j + 1]:
            pairs.append((advised[i], device[j], sc)); i += 1; j += 1
        elif dp[i][j] == dp[i][j + 1]:
            pairs.append((None, device[j], 0)); j += 1
        else:
            pairs.append((advised[i], None, 0)); i += 1
    while i < n:
        pairs.append((advised[i], None, 0)); i += 1
    while j < m:
        pairs.append((None, device[j], 0)); j += 1
    return pairs


def layer_handoff(device, layers_of_kind):
    """Does this layer take its input from DRAM while leaving its output in L1?

    Consecutive decoder layers can hand off in L1, and one corpus cell does exactly that -- its profile opens
    and closes on L1_WIDTH_SHARDED. Two others open with an InterleavedToSharded off DRAM_INTERLEAVED and
    close in L1, so every layer re-loads what its predecessor had already placed. That conversion is paid
    layers_of_kind - 1 times more than it needs to be, and it belongs to whoever built the decoder, not to
    this stage -- the advisor is never asked about a layer boundary. Report it so it is not lost.
    """
    if not device:
        return None
    first, last = device[0], device[-1]
    entry_dram = movement_class(first["cls"]) == "dram_to_l1" or (first.get("mem") or "").find("DRAM") >= 0
    exit_l1 = "L1" in (last.get("mem") or "")
    if not (entry_dram and exit_l1):
        return {"entry_from_dram": entry_dram, "exit_in_l1": exit_l1,
                "note": "no layer-boundary DRAM round trip detected, or the profile does not show it"}
    return {
        "entry_from_dram": True, "exit_in_l1": True,
        "entry_op": first["code"], "entry_us": round(first["us"], 3),
        "redundant_per_model_us": round(first["us"] * max(0, layers_of_kind - 1), 3),
        "note": "This layer loads its input from DRAM but leaves its output in L1, so consecutive layers do "
                "not hand off in L1 and this conversion is paid once per layer. NOT this stage's to fix -- "
                "the advisor is not asked about layer boundaries -- but report it upstream.",
    }


def parse_reshards(report):
    """Index the advice's own chain boundaries by the (producer, consumer) edge they sit on.

    `reshards[]` is where the advisor declares where it believes L1 residency breaks -- in tt-mlir these are
    the ToLayout / ToMemoryConfig ops the optimizer materializes on a chain edge, and they do NOT appear in
    `ops[]`. Comparing them against the shipped movement ops is the point: a shipped boundary with no advised
    reshard on the same edge means the advisor keeps that chain in L1, which is the highest-value candidate
    this stage can produce.

    Keyed on op NAMES, so an edge a layer repeats (several `linear` -> `slice_static`) collapses to one key.
    That makes presence a strong hint and absence a weaker one; it is ranking input, not a measurement.
    """
    idx = {}
    for x in report.get("reshards", []) or []:
        src, dst = (x.get("from") or "?"), (x.get("to") or "?")
        space = lambda t: "dram" if t.startswith("dram") else ("l1" if t.startswith("l1") else "?")
        idx.setdefault((x.get("producer"), x.get("consumer")), []).append(
            {"kind": x.get("kind"), "from": src, "to": dst,
             "transition": f"{space(src)}->{space(dst)}", "output_revert": x.get("output_revert")})
    return idx


def bsum(rows):
    """Split the shipped conversions by who wants them, and by which move they are.

    `us_advisor_drops` is this stage's ceiling on conversion-removal value: it is the DRAM round trips and
    regrids the advice does NOT place, so removing them is attributable to the advisor. `us_advisor_agrees`
    is conversion time that is real and worth attacking but is OUT OF SCOPE here -- removing it is the
    agent's own idea, and crediting it to the advisor would contaminate the measurement. Report it, hand it
    to $optimize, do not screen it in this stage.
    """
    allb = [r for r in rows if r["bucket"] == "boundary"]
    b = [r for r in allb if r.get("advisor_comparable") != "unresolved"]
    us = lambda pred: round(sum(r["us"] for r in b if pred(r)), 3)
    by_class = {}
    for r in allb:
        e = by_class.setdefault(r.get("conversion_class", "?"), {"us": 0.0, "ops": 0})
        e["us"] = round(e["us"] + r["us"], 3)
        e["ops"] += 1
    return {
        "advisor_drops_ops": sum(1 for r in b if r.get("advised_here") is False),
        "us_advisor_drops": us(lambda r: r.get("advised_here") is False),
        "advisor_agrees_ops": sum(1 for r in b if r.get("advised_here") is True),
        "us_advisor_agrees": us(lambda r: r.get("advised_here") is True),
        "undetermined_ops": sum(1 for r in b if r.get("advised_here") is None),
        "us_undetermined": us(lambda r: r.get("advised_here") is None),
        "unresolved_ops": sum(1 for r in allb if r.get("advisor_comparable") == "unresolved"),
        "us_unresolved": round(
            sum(r["us"] for r in allb if r.get("advisor_comparable") == "unresolved"), 3),
        "by_conversion_class": by_class,
        "note": "us_advisor_drops is what this stage can attribute to the advisor. us_advisor_agrees is real "
                "conversion time the advisor endorses: attacking it is a separate activity and crediting it "
                "here would contaminate the measurement. us_unresolved needs the IR to classify -- resolve "
                "it, do not skip it.",
    }


CLIFF_MAX_CORES = 2        # "on the cliff" = a reduction that never got off one or two cores
CLIFF_MIN_SHARE_PCT = 2.0  # and is big enough for the first step off it to be measurable


def main() -> int:
    ap = argparse.ArgumentParser()
    if "--self-test" in sys.argv:
        return self_test()
    if "--legal-grids" in sys.argv:
        i = sys.argv.index("--legal-grids")
        row = int(sys.argv[sys.argv.index("--grid-row-width") + 1]) if "--grid-row-width" in sys.argv else 11
        lad = legal_ladder(int(sys.argv[i + 1]), row_width=row)
        print(json.dumps(lad, indent=2))
        return 0
    ap.add_argument("--self-test", action="store_true", help="run the regression fixtures and exit")
    ap.add_argument("--legal-grids", type=int, metavar="WIDTH_TILES",
                    help="print the legal core-count ladder for a tensor this many tiles wide, and exit")
    ap.add_argument("--grid-row-width", type=int, default=11,
                    help="cores per compute-grid row: 11 on Blackhole, 8 on Wormhole. Used by the ladder")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--perf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layer-kind", default="unknown")
    ap.add_argument("--layers-of-kind", type=int, default=1)
    ap.add_argument("--layers-in-window", type=int, default=1,
                    help="how many consecutive decoder layers the capture and the profile cover. >1 puts the "
                         "interior layer boundaries INSIDE the advised graph, where the advisor can choose "
                         "to keep them in L1 instead of having them hardcoded to DRAM")
    ap.add_argument("--total-layers", type=int, default=1)
    ap.add_argument("--incumbent-ms", type=float, default=None,
                    help="incumbent_ms from incumbent.json; enables a window sanity check")
    ap.add_argument("--incumbent", type=Path, default=None,
                    help="incumbent.json itself. Strongly preferred: repeats_ms gives the harness noise "
                         "floor, without which this script cannot tell you whether a chain is measurable")
    ap.add_argument("--evidence", type=Path, default=None,
                    help="post-screen measurement evidence, merged into a FRESHLY GENERATED "
                         "reconciliation by stable identifier. This is the supported way to record "
                         "verdicts; never edit the output JSON")
    ap.add_argument("--ir", type=Path, default=None,
                    help="final_ir.mlir from the same capture. THE AUTHORITATIVE ADVICE: report.json "
                         "carries no shard shape and understates multi-range core counts, so a plan built "
                         "from it alone cannot be implemented. Strongly preferred")
    a = ap.parse_args()

    for f in (a.report, a.perf):
        if not f.is_file():
            sys.exit(f"FATAL: no such file: {f}")
    evidence = {}
    if a.evidence:
        if not a.evidence.is_file():
            sys.exit(f"FATAL: no such evidence file: {a.evidence}")
        evidence = json.loads(a.evidence.read_text())
    incumbent, repeats = {}, []
    if a.incumbent:
        if not a.incumbent.is_file():
            sys.exit(f"FATAL: no such file: {a.incumbent}")
        incumbent = json.loads(a.incumbent.read_text())
        repeats = [float(x) for x in (incumbent.get("repeats_ms") or [])]
        if a.incumbent_ms is None:
            a.incumbent_ms = incumbent.get("incumbent_ms")

    # LAYER COUNTS ARE LOAD-BEARING: the model estimate multiplies by layers_of_kind, so a wrong count
    # scales the headline silently. Cross-check against the counts the incumbent recorded rather than
    # trusting the flag.
    counts = incumbent.get("layer_counts") or {}
    if counts:
        want = counts.get(a.layer_kind)
        if want is None:
            sys.exit(f"FATAL: incumbent.json records layer_counts {sorted(counts)} with no entry for "
                     f"{a.layer_kind!r}. Name the kinds the same way in both places.")
        if want != a.layers_of_kind:
            sys.exit(f"FATAL: --layers-of-kind {a.layers_of_kind} but incumbent.json says {a.layer_kind} "
                     f"has {want} layers. The model estimate scales by this number; fix whichever is wrong.")
        if sum(counts.values()) != a.total_layers:
            sys.exit(f"FATAL: layer_counts sum to {sum(counts.values())} but --total-layers is "
                     f"{a.total_layers}. Every layer belongs to exactly one kind.")

    report = json.loads(a.report.read_text())
    advised = parse_advised(report)
    unfixable = parse_unfixable(report)
    reshard_idx = parse_reshards(report)

    # THE ADVICE AS THE ADVISOR WROTE IT. Read the IR if it is there, and cross-check report.json against
    # it: the summary's core counts are understated wherever the chosen CoreRangeSet has more than one
    # range, and the summary has no shard shape at all. A disagreement here is not an error, it is the
    # known lossiness -- record it so the next reader does not have to rediscover it.
    ir_ops, ir_note = [], None
    if a.ir:
        if not a.ir.is_file():
            sys.exit(f"FATAL: no such IR file: {a.ir}")
        _, ir_ops = parse_ir(a.ir)
        for o in ir_ops:
            if o.get("unfixable"):
                unfixable.setdefault(o["op"], "declared ttnn.validation_unfixable in final_ir.mlir")
        by_op = {}
        for o in ir_ops:
            by_op.setdefault(o["op"], []).append(o)
        understated = 0
        for adv in advised:
            same = by_op.get(adv["op"]) or []
            if same and same[0].get("grid_cores") is not None:
                adv["ir_cores"] = same[0]["grid_cores"]
                adv["ir_shard_shape"] = same[0].get("shard_shape")
                adv["ir_core_ranges"] = same[0].get("core_ranges")
                if adv["cores_bbox"] is not None and adv["cores_bbox"] < same[0]["grid_cores"]:
                    understated += 1
        ir_note = (f"{understated} of {len(advised)} advised ops have a report.json cores= bounding box "
                   f"below the true core count. The grid product is used throughout.")
    device, window = parse_perf(a.perf, a.incumbent_ms, a.layers_in_window)
    handoff = layer_handoff(device, a.layers_of_kind)
    if window <= 0:
        sys.exit(f"FATAL: measured window is 0 us from {a.perf}")

    # movement ops are boundaries regardless of what the advisor said, so keep them out of the pairing
    pairable = [d for d in device if not movement_class(d["cls"])]
    alignment = align([a for a in advised if a["explicit"] is not None], pairable)
    paired = {id(d): (a, sc) for a, d, sc in alignment if d is not None}
    unpaired_adv = [a for a, d, _ in alignment if d is None]
    seq = [paired.get(id(d), (None, 0)) + (d,) for d in device] + [(a, 0, None) for a in unpaired_adv]

    rows, chains, cur = [], [], None
    for adv, pair_sc, dev in seq:
        if dev is None:
            auto = adv["explicit"] == "__auto__"
            rows.append({"op": adv["op"],
                         "bucket": "unmapped_advised" if auto else "advised_but_not_present",
                         "us": 0.0, "share_pct": 0.0, "advised": adv["layout"],
                         "reason": "no device op paired with this advised op. Either it did not run "
                                   "(a capture/shape mismatch worth a look), or its name does not pair "
                                   "with any device class -- add it to TTNN_TO_DEVICE if so."})
            continue
        r = {"op": adv["op"] if adv else None, "device": dev["code"], "cls": dev["cls"],
             # how this pairing was made: a name match is evidence, a positional one is a guess
             "pair_confidence": (None if adv is None else "name" if pair_sc == 2 else "position"),
             "us": round(dev["us"], 3), "share_pct": round(100 * dev["us"] / window, 3),
             "shipped_cores": dev["cores"]}
        mv = movement_class(dev["cls"])
        if mv:
            # movement ops mostly arrive unpaired, because the advisor lists them in reshards[] rather
            # than ops[]; classify them first or they fall through to `untraced` and the ranking loses
            # its input entirely
            r.update(bucket="boundary", conversion_class=mv)
            cur = None                                   # a conversion ends the current L1 run
        elif adv is None:
            r.update(bucket="untraced", reason="in the profile, absent from the advisor's graph")
            cur = None
        elif adv["op"] in unfixable:
            # THE ADVISOR ALREADY TOLD YOU THIS ONE IS IMPOSSIBLE, with the exact runtime error. 41 of 54
            # such declarations in the v2 corpus were presented to a cell as screenable advice anyway, and
            # cells then spent device time rediscovering the string below.
            r.update(bucket="advisor_unfixable", advised=adv["layout"], advisor_unfixable=True,
                     unfixable_reason=unfixable[adv["op"]],
                     reason="the advisor declared this op UNPLACEABLE, with the runtime error in "
                            "unfixable_reason. The layout it shows is a fallback after that declared "
                            "failure, not a recommendation -- do not screen it, and do not count it as a "
                            "disagreement. If you think the error is wrong, disprove it with an isolated "
                            "single-op test, not with a whole-decoder measurement.")
            cur = None
        elif adv["space"] == "dram":
            r.update(bucket="dram_resident", advised=adv["layout"],
                     reason="advisor placed it in DRAM -- that is advice, and it disagrees with a "
                            "sharded shipped op")
            cur = None
        else:
            # AGREEMENT, AND A SPACE HINT THAT IS DELIBERATELY NOT PART OF IT.
            #
            # An op advised into L1 and shipped in DRAM should not read as agreement just because the core
            # counts coincide -- one v2 cell had exactly that, on a `typecast`. But THE PROFILE CANNOT
            # SETTLE IT: `Input 0 Memory` is the space of the op's INPUT, and what the advisor states is a
            # placement for its OUTPUT. There is no output-memory column at all. A matmul legitimately
            # reads DRAM and writes L1 width-sharded, so treating input-0 as the op's space and moving the
            # bucket on it manufactures disagreements. Measured over 21 corpus kind-runs, it did: 15 rows
            # left agreement and only ONE -- the documented typecast -- was real; the other 14 were
            # `linear` and `nlp_create_qkv_heads_decode` reading DRAM and writing L1, six of them on the
            # matmul class a ladder sweep is already known to lose on.
            #
            # So the space mismatch is recorded as a HINT to check the IR, and agreement is decided as
            # before. Do not promote this to a bucket rule without the output memory config.
            dev_space = ("l1" if "L1" in (dev.get("mem") or "") else
                         "dram" if "DRAM" in (dev.get("mem") or "") else None)
            grid_same = adv["cores"] is not None and adv["cores"] == dev["cores"]
            ds_same = dev["dram_sharded"] and "dram_sharded" in adv["program_config"]
            same = grid_same or ds_same
            r.update(bucket="agrees_with_shipped" if same else "chain",
                     advised=adv["layout"], advised_cores=adv["cores"],
                     advised_cores_bbox=adv["cores_bbox"], advised_shard_shape=adv.get("ir_shard_shape"),
                     shipped_input0_space=dev_space)
            if dev_space is not None and dev_space != adv["space"]:
                r["space_hint"] = (
                    f"advised into {adv['space']}, and this op's INPUT 0 is in {dev_space}. The profile "
                    "carries no output memory config, so this is not a disagreement -- an op can read one "
                    "space and write another. Check the edge in final_ir.mlir before treating it as one.")
            if same:
                # WHAT they agree on. `ds_family` says nothing about the grid -- DS beats core count in the
                # advisor's ordering, so a 12-vs-99-core DS pair is agreement. That is correct (widening a
                # DS matmul measured +65 % slower, 1 win in 7) but reads as a grid match without this field.
                r["agreed_on"] = "grid" if grid_same else "ds_family"
            if not same:
                if cur is None:
                    cur = {"chain": f"{a.layer_kind}:{len(chains)}", "ops": [], "us": 0.0,
                           "boundary_us": 0.0, "verdict": "pending", "measured_ms": None,
                           "repeats_ms": None}
                    chains.append(cur)
                cur["ops"].append(adv["op"])
                cur["us"] += dev["us"]
                cur["_positional"] = cur.get("_positional", 0) + (r["pair_confidence"] == "position")
                r["chain"] = cur["chain"]
        rows.append(r)

    # Does the advice put a conversion on this same edge? Absent => the advisor keeps the chain in L1.
    dev_rows = [r for r in rows if "device" in r]
    for i, r in enumerate(dev_rows):
        if r["bucket"] != "boundary":
            continue
        prev = next((dev_rows[j]["op"] for j in range(i - 1, -1, -1) if dev_rows[j].get("op")), None)
        nxt = next((dev_rows[j]["op"] for j in range(i + 1, len(dev_rows)) if dev_rows[j].get("op")), None)
        hits = reshard_idx.get((prev, nxt))
        r["edge"] = f"{prev} -> {nxt}"
        if r.get("conversion_class") not in ADVISOR_COMPARABLE:
            r["advisor_comparable"] = "unresolved"
            r["reason"] = (
                f"{r.get('conversion_class')} is not a placement edge the advisor states, and the profile "
                "cannot show whether it changed layout. It may still be a chain boundary -- check this "
                "edge's layouts in the IR. Do not drop it: it is unresolved, not out of scope")
        elif prev is None or nxt is None:
            r["advised_here"] = None
            r["reason"] = "boundary on an edge with no paired advised op either side -- undetermined"
        elif hits:
            r["advised_here"] = True
            r["advised_reshard"] = hits[0]
            r["reason"] = "the advisor puts a conversion here too -- agreement, not a candidate"
        else:
            r["advised_here"] = False
            r["reason"] = ("no advised reshard on this edge: the advisor keeps this run in L1. Removing "
                           "this conversion is the change to measure.")

    # Conversion value: boundary us adjacent to a chain -- what joining across it would remove.
    #
    # BUG FIX (v3): this used to charge each adjacent chain HALF the conversion unconditionally, so a
    # boundary with a chain on only one side contributed half its cost and no chain ever claimed the other
    # half. On phi fuse-noadvise that lost 11.5 of the 71.6 us ceiling -- value that showed in the ceiling
    # and in no candidate, which is exactly the suppression this stage's own bias rule forbids. Split only
    # when BOTH sides are chains, since then one change joins across it and additivity must hold.
    for i, r in enumerate(rows):
        if r["bucket"] != "boundary":
            continue
        sides = [rows[j]["chain"] for j in (i - 1, i + 1)
                 if 0 <= j < len(rows) and rows[j].get("chain")]
        if not sides:
            continue
        share = r["us"] / len(set(sides)) if len(set(sides)) > 1 else r["us"]
        for cid in set(sides):
            ch = next(c for c in chains if c["chain"] == cid)
            ch["boundary_us"] += share
            if r.get("advisor_comparable") == "unresolved":
                ch["_unresolved_us"] = round(ch.get("_unresolved_us", 0.0) + share, 3)
            if r.get("advised_here") is False:
                ch["advisor_removes_us"] = round(ch.get("advisor_removes_us", 0.0) + share, 3)
    # An attributable boundary with no chain either side is still a candidate -- arguably the cleanest one,
    # since the advisor is saying these two placements can be adjacent with no conversion and no geometry
    # change. Without this the value shows in the ceiling and in no candidate, and the agent sees nothing to
    # screen. Found by running the tool against a synthetic well-formed cell.
    for i, r in enumerate(rows):
        if r["bucket"] != "boundary" or r.get("advised_here") is not False:
            continue
        if any(0 <= j < len(rows) and rows[j].get("chain") for j in (i - 1, i + 1)):
            continue
        ch = {"chain": f"{a.layer_kind}:b{i}", "ops": [], "us": 0.0, "boundary_us": round(r["us"], 3),
              "advisor_removes_us": round(r["us"], 3), "verdict": "pending", "measured_ms": None,
              "repeats_ms": None, "kind": "boundary_only", "edge": r.get("edge"),
              "how": f"remove the {r.get('conversion_class')} on this edge; the advice places no conversion "
                     "here and both neighbours already match shipped, so no geometry change is needed"}
        chains.append(ch)

    for c in chains:
        c["us"] = round(c["us"], 3)
        c["boundary_us"] = round(c["boundary_us"], 3)
        c["conversion_value_us"] = c["boundary_us"]
        c.setdefault("advisor_removes_us", 0.0)
        c.setdefault("_positional", 0)
        c.setdefault("_unresolved_us", 0.0)
        c["op_share_pct"] = round(100 * c["us"] / window, 3)
        c["per_model_us"] = round((c["us"] + c["boundary_us"]) * a.layers_of_kind, 3)
        # DECIDE ON THIS, not on the per-layer number: a 1 us win on 40 layers beats a 3 us win on 8.
        c["advisor_removes_per_model_us"] = round(c["advisor_removes_us"] * a.layers_of_kind, 3)
    # advisor-attributable value first: a boundary the advice also wants is not this stage's candidate.
    # Losing chains are kept in the list, only ordered lower -- suppressing one understates the contribution.
    chains.sort(key=lambda c: (-c["advisor_removes_us"], -(c["conversion_value_us"] + c["us"])))

    acct, accounted = {}, 0.0
    for r in rows:
        if r["bucket"] in ("advised_but_not_present", "unmapped_advised"):
            continue
        accounted += r["us"]
        b = acct.setdefault(r["bucket"], {"us": 0.0, "ops": 0})
        b["us"] += r["us"]
        b["ops"] += 1
    for v in acct.values():
        v["us"] = round(v["us"], 3)
        v["share_pct"] = round(100 * v["us"] / window, 3)
    closes = abs(accounted - window) < 0.01

    # A material op on almost no cores is only THIS stage's business when the advisor disagrees with that
    # placement -- then it is advisor-attributable and needs an attempt. Where the advisor wants the same
    # 1-core placement, fixing it is a direct grid sweep with no advisor contribution: report it and hand it
    # to $optimize. Keeping those two apart is what stops this stage growing into a second optimize pass.
    starved = [r for r in rows if (r.get("shipped_cores") or 99) <= 2 and r["share_pct"] >= 1.0
               and r["bucket"] in ("chain", "agrees_with_shipped")]
    low = [r for r in starved if r["bucket"] == "chain"]
    low_out_of_scope = [dict(r, note="the advisor wants this placement too, so improving it is a direct grid "
                                     "sweep with no advisor contribution -- report it, hand it to $optimize, "
                                     "and do not screen it here")
                        for r in starved if r["bucket"] == "agrees_with_shipped"]

    # THE CLIFF CHECK -- costs no device time, and screen these FIRST.
    #
    # The win in this class is a threshold, not a gradient: a reduction's response to core count is flat
    # over a wide middle and falls off a cliff at one core, so essentially all the value is the first step
    # OFF one core. Over the v2 corpus this rule flags 5 of 14 cells, contains all three double-digit wins,
    # and no unflagged cell produced one; it then predicted an unscreened -12.44 %/layer win before it was
    # measured. Ranking by it puts the winner at rank 1 in all four cells whose win was of this class,
    # against 2nd/2nd/2nd/4th-of-27 under the boundary-value order.
    #
    # Positional pairings are excluded: the row's identity is a guess, and a guess cannot support a finding.
    cliff = []
    for r in rows:
        if r["bucket"] != "chain" or r.get("pair_confidence") != "name":
            continue
        sc, ac = r.get("shipped_cores"), r.get("advised_cores")
        if sc is None or ac is None or sc > CLIFF_MAX_CORES or ac <= sc:
            continue
        if r["share_pct"] < CLIFF_MIN_SHARE_PCT:
            continue
        width_tiles = None
        for o in ir_ops:
            if o["op"] == r["op"] and o.get("logical_shape"):
                width_tiles = -(-o["logical_shape"][-1] // 32)
                break
        lad = legal_ladder(width_tiles, row_width=a.grid_row_width) if width_tiles else None
        cliff.append({
            "device": r["device"], "op": r["op"], "chain": r.get("chain"),
            "us": r["us"], "share_pct": r["share_pct"], "per_model_us": round(r["us"] * a.layers_of_kind, 3),
            "shipped_cores": sc, "advised_cores": ac,
            "advised_shard_shape": r.get("advised_shard_shape"),
            "width_tiles": width_tiles, "legal_ladder": (lad or {}).get("ladder"),
            "verdict": None, "measured_ms": None, "repeats_ms": None,
            "how": f"{r['op']} runs on {sc} core(s) at {r['share_pct']}% of the window and the advisor wants "
                   f"{ac}. Sweep the ladder on both sides of {ac}. The first step off {sc} core(s) is where "
                   f"the value is; which middle rung you land on is worth about 1 pp.",
        })
    cliff.sort(key=lambda c: -c["per_model_us"])
    # Cost sitting on cliff ops. NOT a saving: it is the op's own incumbent cost, so "57x the floor" means
    # "a 1.7 % saving here would be measurable", not "a 57x win is available". Measured savings in this
    # class were 6-13 % of the LAYER, four times out of four.
    regrid_pool_us = round(sum(c["us"] for c in cliff), 3)

    # How much of the accounting rests on positional guesses rather than name evidence. This is the tool's
    # generality check: on a model whose ops it does not recognise the buckets degrade, and it must SAY so
    # rather than emit confident-looking wrong ones.
    byname = [r for r in rows if r.get("pair_confidence") == "name"]
    bypos = [r for r in rows if r.get("pair_confidence") == "position"]
    pos_us = round(sum(r["us"] for r in bypos), 3)
    pos_share = round(100 * pos_us / window, 3)

    # DO THE TWO GRAPHS EVEN MATCH? Positional-pair share does not answer this: generic names (rms_norm,
    # linear, add) match across any transformer, so feeding one model's advice against another's profile
    # pairs happily and emits a confident, wrong accounting. What does answer it is unexplained blindness --
    # a large `untraced` share that the report itself does not own up to. A report that declares
    # `uncapturable` (an MoE whose experts are terminal in the tracer) is expected to be blind; one that
    # declares nothing and still fails to account for a third of the window is describing a different graph.
    untraced_share = acct.get("untraced", {}).get("share_pct", 0.0)
    declared_blind = bool(report.get("uncapturable"))
    fanout = round(len(device) / max(1, len(advised)), 2)
    suspect = [x for x, bad in (
        (f"untraced is {untraced_share:.2f} % of the window", untraced_share > 30.0),
        (f"{len(device)} device ops against {len(advised)} advised ({fanout}x)", fanout > 2.5)) if bad]
    degraded = bool(suspect) and not declared_blind

    # THE HARNESS NOISE FLOOR. A chain whose value is below the spread of the incumbent's own repeats cannot
    # be resolved by the non-overlap rule no matter how good the advice is, so measuring it burns device time
    # and returns a zero that says nothing about the advisor. Both failure directions are real: one corpus
    # cell had a whole-stage ceiling of 0.65x its floor and still shipped a win, and another had a ceiling of
    # 4.31x its floor but no single chain above it, screened them one at a time, and reported no change.
    floor_us = round((max(repeats) - min(repeats)) * 1000, 3) if len(repeats) >= 2 else None
    # IS THE FLOOR NOISE, OR AN UNFINISHED WARM-UP? A device that is still settling produces repeats that
    # fall monotonically, and then the "floor" is a systematic ramp rather than run-to-run variance. That
    # matters twice over: it inflates the floor, and it breaks the exchangeability the non-overlap rule
    # assumes, because a candidate measured after the incumbent in the same process is simply warmer.
    floor_exfirst_us = round((max(repeats[1:]) - min(repeats[1:])) * 1000, 3) if len(repeats) >= 3 else None
    first_gap_share = (round(1 - floor_exfirst_us / floor_us, 3)
                       if floor_us and floor_exfirst_us is not None else None)
    monotone = len(repeats) >= 3 and all(repeats[i] >= repeats[i + 1] for i in range(len(repeats) - 1))
    ceiling_us = round(sum(
        r["us"] for r in rows if r["bucket"] == "boundary" and r.get("advised_here") is False), 3)
    for c in chains:
        c["vs_noise_floor"] = round(c["advisor_removes_us"] / floor_us, 2) if floor_us else None
        c["resolvable_alone"] = (c["advisor_removes_us"] > floor_us) if floor_us else None
        c["confidence"] = "low" if (c.pop("_positional", 0) or c.pop("_unresolved_us", 0)) else "high"

    declared = set()
    for src in (report.get("uncapturable") or {}, ):
        declared |= {str(x).split(".")[-1] for x in (src.get("ops") or [])}
    declared |= {str(x.get("op", "")).split(".")[-1] for x in (report.get("unfixable_ops") or [])}
    untraced_rows = [r for r in rows if r["bucket"] == "untraced"]
    untraced_detail = {
        "ops": [{"device": r["device"], "us": r["us"], "share_pct": r["share_pct"]} for r in untraced_rows],
        "declared_uncapturable_by_report": sorted(declared),
        "note": "untraced means the profile has it and the advisor's graph does not. That is expected for "
                "ops the report declares uncapturable (terminal in the tracer) and a problem otherwise. "
                "This tool cannot tell the two apart per op -- the capture log can.",
    }

    feasibility = {"noise_floor_us": floor_us, "noise_floor_source": "max-min of incumbent repeats_ms",
                   "noise_floor_excluding_first_us": floor_exfirst_us,
                   "first_repeat_share_of_floor": first_gap_share, "repeats_monotone_decreasing": monotone,
                   "repeats": len(repeats), "ceiling_us": ceiling_us,
                   "ceiling_vs_floor": round(ceiling_us / floor_us, 2) if floor_us else None,
                   # SECOND CHANNEL. The ceiling above counts only boundary conversions the advice does not
                   # place, so re-gridding an op that stays inside its L1 chain removes no boundary and
                   # prices at exactly 0.000 us -- while measuring up to 236.8 us/layer on hardware. Two of
                   # the v2 corpus's three biggest wins came from cells whose ceiling said 0 and whose every
                   # chain read `below_threshold`. So the ceiling is not a stopping condition on its own.
                   "regrid_pool_us": regrid_pool_us, "cliff_ops": len(cliff),
                   "regrid_pool_note": "cost sitting on cliff ops -- the op's own incumbent cost, not an "
                                       "achievable saving. Measured savings in this class were 6-13 % of "
                                       "the layer, 4 times out of 4.",
                   "chains_resolvable_alone": sum(1 for c in chains if c["resolvable_alone"])}
    if floor_us is None:
        feasibility["verdict"] = "unknown"
        feasibility["advice"] = ("Pass --incumbent incumbent.json with at least 2 repeats_ms. Without the "
                                "noise floor there is no way to tell a real zero from an unmeasurable one.")
    elif ceiling_us <= floor_us and cliff:
        feasibility["verdict"] = "regrid_only"
        feasibility["advice"] = (
            f"The boundary ceiling is {ceiling_us} us, at or below the {floor_us} us floor -- and that is "
            f"NOT a reason to report zero here. {len(cliff)} op(s) carrying {regrid_pool_us} us sit on <= "
            f"{CLIFF_MAX_CORES} core(s) where the advisor wants more; re-gridding an op inside its own L1 "
            f"chain removes no boundary, so this channel is invisible to the ceiling by construction. "
            f"Screen cliff_candidates in the order given before publishing anything.")
    elif ceiling_us <= floor_us:
        feasibility["verdict"] = "not_measurable"
        feasibility["advice"] = (
            f"STOP. Everything the advisor proposes removing here totals {ceiling_us} us, below this "
            f"harness's own {floor_us} us spread, and no op is on the cliff. No non-overlap decision on "
            f"this cell can be attributed to the advice. Record a contribution of zero WITH THIS ARITHMETIC "
            f"as the reason -- do not screen chains and call the result zero. Do NOT try to rescue it with "
            f"more replays per block: measured, 250 -> 1,800 replays made the floor 3-4x WORSE and still did "
            f"not separate the candidate. The term worth attacking is cross-process (see process_ordinal).")
    elif not chains:
        feasibility["verdict"] = "no_candidates"
        feasibility["advice"] = (
            f"{ceiling_us} us is attributable and above the {floor_us} us floor, but it resolved to no "
            "candidate. Read `disagreements` for boundary rows with advised_here=false and work out what "
            "change would remove them; do not report zero without explaining this gap.")
    elif not feasibility["chains_resolvable_alone"]:
        feasibility["verdict"] = "aggregate_only"
        feasibility["advice"] = (
            f"Do not screen these chains one at a time: the total is {ceiling_us} us "
            f"({feasibility['ceiling_vs_floor']}x the {floor_us} us floor) but no single chain clears it, so "
            f"each one measured alone returns a zero regardless of the advice. Apply the top chains together "
            f"as one candidate first to establish that anything is there, then split only what wins.")
    else:
        feasibility["verdict"] = "measurable"
        feasibility["advice"] = (f"{feasibility['chains_resolvable_alone']} chain(s) exceed the {floor_us} us "
                                 f"floor; screen those individually and group the rest.")
    if first_gap_share is not None and (first_gap_share > 0.5 or monotone):
        would = ("measurable" if any(c["advisor_removes_us"] > floor_exfirst_us for c in chains)
                 else "aggregate_only" if ceiling_us > floor_exfirst_us else "not_measurable")
        feasibility["warmup_suspect"] = {
            "first_repeat_share_of_floor": first_gap_share, "monotone": monotone,
            "floor_without_first_repeat_us": floor_exfirst_us,
            "verdict_if_warmed_properly": would,
            "note": "The first timed repeat carries most of the spread, and/or the repeats fall "
                    "monotonically, so this is a settling ramp rather than noise. Re-measure with at least "
                    "10 untimed warm-up replays before accepting the verdict above -- and never measure a "
                    "candidate after the incumbent in one process, since the later run is simply warmer.",
        }
    rec = incumbent.get("noise_floor_ms")
    if rec is not None and floor_us is not None and abs(rec * 1000 - floor_us) > 0.002:
        feasibility["recorded_noise_floor_disagrees"] = {
            "recorded_ms": rec, "computed_us": floor_us,
            "note": "incumbent.json records a noise floor that is not the spread of its own repeats_ms."}
    if repeats and incumbent.get("incumbent_ms") is not None:
        med = sorted(repeats)[len(repeats) // 2]
        if abs(incumbent["incumbent_ms"] - med) > 1e-9:
            feasibility["incumbent_ms_is_not_median"] = {
                "recorded": incumbent["incumbent_ms"], "median": med, "min": min(repeats),
                "note": "min-of-n is biased low and the bias grows with n; cells with different n are not "
                        "comparable. Fix the incumbent before screening."}

    # WHAT THE ADVICE ITSELF SAYS ABOUT ITS OWN CONSTRUCTION. `spill.ran` is true in every corpus cell, and
    # a plan the optimizer had to spill out of L1 is less likely to survive being applied one chain at a time.
    # The three op counts disagree in every cell and nothing defines their relationship, so report all three
    # rather than trusting one.
    sp = report.get("spill") or {}
    capture_provenance = {
        "capture_batch": report.get("capture_batch"),
        "capture_policy_source": report.get("capture_policy_source"),
        "traced_weight_dtypes": report.get("traced_weight_dtypes"),
        "allow_bf16_dram_sharded_matmul": report.get("allow_bf16_dram_sharded_matmul"),
        "spills": sp.get("total_spills"), "spill_ran": sp.get("ran"),
        "op_counts": {"report_total_ops": report.get("total_ops"), "ops_listed": len(advised),
                      "final_choices": report.get("final_choices")},
        "dram_sharded_advised": report.get("dram_sharded_advised"),
        "layers_in_window": a.layers_in_window,
        "graph_input_reshards": sum(1 for x in (report.get("reshards") or [])
                                    if x.get("producer") == "input"),
        "graph_input_note": "reshards with producer=='input' come off a graph input the py-to-IR boundary "
                            "hardcodes to DRAM interleaved. Some are genuine weight placement (DRAM is where "
                            "weights live); the one feeding the first op is the layer's activation entry and "
                            "is an artifact. Capturing N consecutive layers reduces the artifact to 1/N of "
                            "the layer boundaries.",
    }
    if sp.get("total_spills"):
        capture_provenance["spill_caution"] = (
            f"the optimizer spilled {sp['total_spills']} time(s) building this plan, so it did not fit L1 as "
            "a whole. Expect single-chain application to hit the same wall, and treat a capacity failure as "
            "a partial-application artifact rather than evidence the direction is wrong.")
    if not report.get("capture_policy_source"):
        capture_provenance["provenance_gap"] = (
            "capture_policy_source is unset, so there is no record that the traced decoder was built with "
            "the SHIPPED policy rather than class defaults. Dtypes are checked by the gate; layouts and "
            "DRAM-sharding flags are not.")

    per_layer_window = window / max(1, a.layers_in_window)
    model_estimate = {
        "this_kind_us": round(per_layer_window * a.layers_of_kind, 3),
        "per_layer_window_us": round(per_layer_window, 3), "layers_in_window": a.layers_in_window,
        "layers_of_kind": a.layers_of_kind, "total_layers": a.total_layers,
        "kind_share_of_layers": round(a.layers_of_kind / max(1, a.total_layers), 3),
        "layer_counts": counts or None,
        "layer_counts_source": ("incumbent.json layer_counts (cross-checked)" if counts else
                                "UNVERIFIED -- incumbent.json records no layer_counts, so layers_of_kind is "
                                "taken on trust and the model estimate is unchecked"),
        "ceiling_per_model_us": round(ceiling_us * a.layers_of_kind / max(1, a.layers_in_window), 3),
        # the model number inherits the per-layer measurement error MULTIPLIED by the layer count, so it is
        # far less precise than its digits suggest. Quote it with this band or not at all.
        "uncertainty_per_model_us": (round(floor_us * a.layers_of_kind / max(1, a.layers_in_window), 3)
                                     if floor_us else None),
        "layer_handoff": handoff,
        "note": "Sum this_kind_us over every layer kind for the full-model estimate, and choose between "
                "candidates on THAT. Per-layer microseconds are for detection against the per-layer noise "
                "floor; per-model microseconds are for deciding, because layer counts differ by kind. Report "
                "the model estimate as before/after with the uncertainty band: a per-layer delta scaled by "
                "the layer count carries the per-layer floor scaled by the same factor.",
    }

    out = {
        "feasibility": feasibility, "model_estimate": model_estimate,
        "capture_provenance": capture_provenance, "untraced_detail": untraced_detail,
        "generated_by": "advisor-challenger/scripts/reconcile.py", "tool_version": 6,
        "advised_plan": {
            "source": str(a.ir) if a.ir else None,
            "ops": ir_ops or None,
            "advised_cores_source": "grid-string product from report.json"
                                    + (", cross-checked against final_ir.mlir" if a.ir else ""),
            "cores_bbox_understated": ir_note,
            "unfixable_ops": {k: v for k, v in unfixable.items()} or None,
            "note": "CANDIDATE #1 IS THIS PLAN, APPLIED WHOLE. Build it from the shard shapes here, not "
                    "from report.json, which has none. Drop the unfixable_ops entries first -- the advisor "
                    "has already declared those impossible, with the exact runtime error. If what remains "
                    "will not run, remove ONLY the failing item and record the single-op test that isolates "
                    "it. Then ablate: an advised item whose REMOVAL is faster is feedback about the advisor "
                    "that no build-up order can produce."
                    if a.ir else
                    "NO IR WAS PASSED, so the advice cannot be implemented as written from this file: "
                    "report.json carries no shard shape and understates multi-range core counts. Re-run "
                    "with --ir shard_advise/<kind>/final_ir.mlir.",
        },
        "confidence": {
            "paired_by_name": len(byname), "paired_by_position": len(bypos),
            "us_paired_by_position": pos_us, "pct_paired_by_position": pos_share,
            "device_to_advised_fanout": fanout, "report_declares_uncapturable": declared_blind,
            "degraded": degraded, "degraded_because": suspect if degraded else [],
            "hard": ["measured_window_us", "per-op us and share_pct", "accounting_closes_100pct",
                     "the single-replay and window-ratio guards", "by_conversion_class us"],
            "soft": ["which advised op each device op is", "advised_here / advisor_removes_us",
                     "the chain ranking, since it is computed from the soft items above"]},
        "limitations": [
            "Ops are paired by normalised name, then by position when no name matches. A positional pair is "
            "a guess; rows carry pair_confidence, and pct_paired_by_position says how much of the window "
            "rests on guesses. On an unfamiliar model expect this to rise.",
            "advised_here is keyed on producer/consumer op NAMES, so a repeated edge collapses to one key: "
            "presence is a strong signal, absence a weak one.",
            "untraced conflates 'terminal in the tracer' with 'ran but the advisor never placed it'. This "
            "tool cannot tell them apart; the capture log can.",
            "The profile exposes no grid and no output memory config, so a conversion that only regrids is "
            "invisible here and unresolved classes cannot be settled without the IR.",
            "agrees_with_shipped compares the space using the profile's INPUT-0 memory class, the only one "
            "it exposes; where that column is absent the row is `grid_only_space_unknown` and the space is "
            "unchecked. A ds_family agreement says nothing about the grid, deliberately.",
            "The legal ladder models the shard-padding rule and the model's grid helper, not the op's own "
            "layout rulebook. It bounds a sweep; it does not prove a rung will run.",
            "Nothing here is a measurement. Every verdict comes from the device.",
        ],
        "layer_kind": a.layer_kind, "layers_of_kind": a.layers_of_kind,
        "total_layers": a.total_layers, "measured_window_us": round(window, 3),
        "accounting": acct, "accounting_closes_100pct": closes,
        "accounted_us": round(accounted, 3),
        "scope": {
            "window_us": round(window, 3),
            "incumbent_us": round(a.incumbent_ms * 1000, 3) if a.incumbent_ms else None,
            "note": "The profile and the harness both cover ONE decoder layer, and window_us should be "
                    "within a few percent of incumbent_us. per_model_us on each chain scales by "
                    "layers_of_kind to the whole model and is an extrapolation -- never compare it to "
                    "incumbent_ms, which is per layer.",
            "layers_of_kind": a.layers_of_kind, "total_layers": a.total_layers},
        "ranked_by": "advisor_removes_us (conversions the advice does not place), then total conversion "
                     "value + chain op us. NOT the advised ops' window share. Compare each chain's "
                     "vs_noise_floor before spending a measurement on it.",
        "advised_boundaries": bsum(rows),
        "cliff_candidates": cliff,
        "chains": chains, "material_ops_on_le_2_cores": low,
        "starved_ops_not_attributable": low_out_of_scope,
        "screening_order": [
            "1. the advised plan applied WHOLE, built from advised_plan.ops (minus unfixable_ops)",
            "2. cliff_candidates, in the order given -- highest per_model_us first",
            "3. chains, in the order given",
            "4. ablate the plan: drop one advised item at a time from the apply-all candidate",
        ],
        "note": "Screen in the order in `screening_order`, and record repeats_ms for everything measured. "
                "The advised core count is a DETECTION signal, not a SELECTION one: the advisor's objective "
                "has no latency term at any level and for normalization ops the core-count term is "
                "overridden with a value that cannot vary with the candidate. So trust it for WHICH op and "
                "WHICH DIRECTION, and sweep `legal_ladder` on both sides of its number for the geometry.",
        "disagreements": rows,
    }

    # ACCEPT MEASURED OUTCOMES. This script used to write `verdict: pending` and never fill it, while the
    # gate required it filled and the skill forbade hand-editing the output. Four v2 cells escaped that
    # impossible contract in four different ways, so which violation a cell chose is what decided whether
    # its work was publishable. Merging by stable identifier keeps this output a reproducible tool product;
    # an unknown identifier is fatal, so a stale evidence file cannot annotate the wrong row.
    if evidence:
        allowed = {"verdict", "measured_ms", "repeats_ms", "oracle_passed", "oracle_pcc", "oracle_kind",
                   "oracle_pcc_bar", "oracle_bar_source", "incumbent_pcc_vs_reference", "combined_with",
                   "hard_error", "perf_report", "oracle_weights", "op_under_test",
                   "candidate_shape_assumptions", "ops"}

        def merge(section, index, label):
            for key, update in (evidence.get(section) or {}).items():
                if key not in index:
                    sys.exit(f"FATAL: evidence names unknown {label} {key!r}. Regenerate the "
                             f"reconciliation and re-key the evidence; a stale identifier would annotate "
                             f"the wrong row.")
                unknown = set(update) - allowed
                if unknown:
                    sys.exit(f"FATAL: evidence for {key} has unsupported fields {sorted(unknown)}")
                index[key].update(update)

        merge("chains", {c["chain"]: c for c in out["chains"]}, "chain")
        merge("cliff_candidates", {c["device"]: c for c in out["cliff_candidates"]}, "cliff candidate")
        merge("material_ops", {r["device"]: r for r in out["material_ops_on_le_2_cores"]}, "material op")
        occurrence, by_id = {}, {}
        for row in out["disagreements"]:            # repeated device names get a 0-based occurrence suffix
            dev = row.get("device")
            if dev:
                n = occurrence.get(dev, 0)
                by_id[f"{dev}#{n}"] = row
                occurrence[dev] = n + 1
        merge("disagreements", by_id, "disagreement")
        out["measurement_evidence"] = str(a.evidence)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{a.layer_kind}: window {window:.3f} us, {len(rows)} rows, closes: {closes}")
    print(f"   paired: {len(byname)} by name, {len(bypos)} by position "
          f"({pos_us:.3f} us, {pos_share:.2f} % of window)")
    if degraded:
        print("   ** DEGRADED -- " + "; ".join(suspect) + ", and the report declares nothing uncapturable.")
        print("   ** The profile and the advice may describe different graphs: wrong file, a stale capture, "
              "or a capture that stopped early. Resolve this before screening -- the buckets and the "
              "ranking below are unsafe, and every share is measured against a window that may not be the "
              "graph the advice describes.")
    elif untraced_share > 30.0:
        print(f"   note: untraced is {untraced_share:.2f} %, expected -- the report declares uncapturable "
              f"ops. State this share as the advisor's reach, do not imply coverage.")
    for b, v in sorted(acct.items(), key=lambda x: -x[1]["us"]):
        print(f"   {b:22s} {v['us']:9.3f} us {v['share_pct']:6.2f} %  ({v['ops']} ops)")
    ab = out["advised_boundaries"]
    print(f"   boundaries: advisor drops {ab['advisor_drops_ops']} ({ab['us_advisor_drops']:.3f} us, "
          f"in scope) | agrees {ab['advisor_agrees_ops']} ({ab['us_advisor_agrees']:.3f} us, out of scope) "
          f"| undetermined {ab['undetermined_ops']} ({ab['us_undetermined']:.3f} us) "
          f"| UNRESOLVED {ab['unresolved_ops']} ({ab['us_unresolved']:.3f} us, check the IR)")
    print("   " + "  ".join(f"{k} {v['us']:.3f}us/{v['ops']}" for k, v in
                            sorted(ab["by_conversion_class"].items(), key=lambda x: -x[1]["us"])))
    ap_ = out["advised_plan"]
    if ap_["source"]:
        print(f"   ADVISED PLAN from {ap_['source']}: {len(ir_ops)} ops with shard shapes"
              + (f"; {ir_note}" if ir_note else ""))
    else:
        print("   !! NO --ir: report.json has no shard shape, so the plan cannot be implemented as written. "
              "Re-run with --ir shard_advise/<kind>/final_ir.mlir.")
    if unfixable:
        print(f"   advisor declares {len(unfixable)} op(s) UNPLACEABLE -- not screenable, do not spend "
              f"device time on them: {', '.join(sorted(unfixable))}")
    f = out["feasibility"]
    print(f"   FEASIBILITY [{f['verdict']}] floor {f['noise_floor_us']} us (n={f['repeats']}), "
          f"ceiling {f['ceiling_us']} us ({f['ceiling_vs_floor']}x), "
          f"cliff pool {f['regrid_pool_us']} us on {f['cliff_ops']} op(s)")
    for c in cliff:
        print(f"   ** CLIFF: {c['op']} on {c['shipped_cores']} core(s), {c['us']:.3f} us "
              f"({c['share_pct']} % of window, {c['per_model_us']:.1f} us/model), advisor wants "
              f"{c['advised_cores']}"
              + (f", legal ladder {c['legal_ladder']}" if c["legal_ladder"] else "")
              + " -- SCREEN THIS FIRST")
    for line in textwrap.wrap(f["advice"], 108):
        print("      " + line)
    if "warmup_suspect" in f:
        w = f["warmup_suspect"]
        print(f"   !! WARM-UP: the first repeat is {100 * (w['first_repeat_share_of_floor'] or 0):.0f} % of "
              f"the floor" + (" and the repeats fall monotonically" if w["monotone"] else "") +
              f". Without it the floor is {w['floor_without_first_repeat_us']} us and the verdict would be "
              f"[{w['verdict_if_warmed_properly']}]. Add warm-up replays and re-measure.")
    if "incumbent_ms_is_not_median" in f:
        print(f"   !! incumbent_ms {f['incumbent_ms_is_not_median']['recorded']} is not the median "
              f"{f['incumbent_ms_is_not_median']['median']} -- fix before screening")
    for r in low_out_of_scope:
        print(f"   -- {r['device']} on {r['shipped_cores']} core(s), {r['us']:.3f} us ({r['share_pct']} %) "
              f"-- advisor agrees, so NOT this stage's: report it and hand it to $optimize")
    cliff_devices = {c["device"] for c in cliff}
    for r in (r for r in low if r["device"] not in cliff_devices):   # cliff rows are already printed above
        print(f"   !! BIGGER THAN THE CEILING: {r['device']} on {r['shipped_cores']} core(s), "
              f"{r['us']:.3f} us ({r['share_pct']} % of the window) -- needs a measured attempt or a quoted "
              f"hard error" if r["us"] > ceiling_us else
              f"   !! {r['device']} on {r['shipped_cores']} core(s), {r['us']:.3f} us ({r['share_pct']} %) "
              f"-- needs a measured attempt or a quoted hard error")

    attrib = [c for c in chains if c["advisor_removes_us"] > 0]
    rest = [c for c in chains if c["advisor_removes_us"] <= 0]

    def show(c):
        print(f"      {c['chain']:20s} ops {c['us']:8.3f} + boundaries {c['boundary_us']:7.3f} us"
              f"  (attributable {c['advisor_removes_us']:6.3f} = {c['vs_noise_floor'] or 0:.2f}x floor,"
              f" conf {c['confidence']}, {c['advisor_removes_per_model_us']:7.1f} us/model)"
              f"   {'/'.join(c['ops'])[:36]}")

    cp = out["capture_provenance"]
    if cp.get("spills"):
        print(f"   note: the advice spilled {cp['spills']}x out of L1 while being built -- a capacity "
              f"failure on one chain is a partial-application artifact, not a wrong direction")
    if cp.get("provenance_gap"):
        print("   note: capture_policy_source unset -- no record the capture used the shipped policy")
    me = out["model_estimate"]
    print(f"   MODEL ESTIMATE this kind: {me['this_kind_us']:.1f} us "
          f"({a.layers_of_kind}/{a.total_layers} layers); ceiling per model {me['ceiling_per_model_us']:.1f} us")
    if handoff and handoff.get("redundant_per_model_us"):
        print(f"   !! LAYER HANDOFF: input from DRAM, output left in L1 -- {handoff['entry_op'][:34]} costs "
              f"{handoff['entry_us']} us/layer, {handoff['redundant_per_model_us']} us across the model. "
              f"Upstream decoder issue, not this stage's.")
    print(f"   {len(attrib)} chain(s) with advisor-attributable value, ranked by it:")
    for c in attrib:
        show(c)
    if rest:
        print(f"   {len(rest)} chain(s) with NO attributable value -- the advisor places the same "
              f"conversions, or they are unresolved. Listed for completeness, not as candidates:")
        for c in rest:
            show(c)
    return 0


def self_test() -> int:
    """Regression fixtures. The pairing and the guards are the parts that silently mislead when wrong, and
    this file has broken twice in ways that still produced plausible output, so assert on both."""
    import subprocess
    import tempfile

    def csv_of(pairs):
        head = "OP CODE,DEVICE KERNEL DURATION [ns],CORE COUNT\n"
        return head + "".join(f"{c},{int(us * 1000)},32\n" for c, us in pairs)

    def report_of(ops, reshards=(), **kw):
        return json.dumps({"ops": [{"index": i, "op": f"ttnn.{o}", "layout": "l1/block_sharded/1x32",
                                    "program_config": ""} for i, o in enumerate(ops)],
                           "reshards": list(reshards), "total_ops": len(ops), **kw})

    fails = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        def run(report, csv, extra=()):
            (d / "r.json").write_text(report)
            (d / "p.csv").write_text(csv)
            return subprocess.run([sys.executable, __file__, "--report", str(d / "r.json"),
                                   "--perf", str(d / "p.csv"), "--out", str(d / "o.json"), *extra],
                                  capture_output=True, text=True)

        def load():
            return json.loads((d / "o.json").read_text())

        def check(cond, name):
            if not cond:
                fails.append(name)
            print(f"  {'ok  ' if cond else 'FAIL'} {name}")

        # 1. plain pairing, closure, and a boundary that the advice does not place
        r = run(report_of(["rms_norm", "linear", "add"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]))
        check(r.returncode == 0, "clean input exits 0")
        o = load()
        check(o["accounting_closes_100pct"], "accounting closes")
        check(abs(o["measured_window_us"] - 117.0) < 1e-6, "window is the sum of device times")
        # rms_norm->LayerNorm and linear->Matmul are name matches; `add` does NOT resemble `BinaryNg`, so it
        # pairs by position. That is the documented fallback, and it is why the corpus shows 1-5 % positional.
        check(o["confidence"]["paired_by_name"] == 2 and o["confidence"]["paired_by_position"] == 1,
              "renames pair by name, binary elementwise falls back to position")
        check([r["pair_confidence"] for r in o["disagreements"] if r.get("op") == "add"] == ["position"],
              "the positional pair is flagged as such")
        check(o["advised_boundaries"]["us_advisor_drops"] == 2.0, "unplaced Reshard is attributable")
        # and the chain beside it claims the WHOLE conversion, not half: the old halving lost the other
        # half to nothing, so the ceiling exceeded the sum of the candidates it was supposed to explain
        check(abs(sum(c["advisor_removes_us"] for c in o["chains"])
                  - o["feasibility"]["ceiling_us"]) < 0.01,
              "chain attributable value sums to the ceiling")

        # 2. the same graph, with the advice placing that conversion too -> not attributable
        r = run(report_of(["rms_norm", "linear", "add"],
                          [{"kind": "to_memory_config", "producer": "linear", "consumer": "add",
                            "from": "l1/x", "to": "l1/y"}]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]))
        o = load()
        check(o["advised_boundaries"]["us_advisor_drops"] == 0.0 and
              o["advised_boundaries"]["us_advisor_agrees"] == 2.0, "advised conversion is not attributable")

        # 3. an unpairable advised op must not drag device ops into untraced
        r = run(report_of(["rms_norm", "sparse_matmul", "linear"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0)]))
        o = load()
        check(o["accounting"].get("untraced", {}).get("us", 0) == 0.0,
              "an unpairable advised op leaves the device ops paired")

        # 4. a report covering two replays must be refused, not scaled
        r = run(report_of(["rms_norm", "linear"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0)] * 2))
        check(r.returncode != 0 and "repeats 2x" in r.stderr, "two-replay report is refused")

        # 4b. and a declared N the profile does not contain must be refused, not silently divided by
        r = run(report_of(["rms_norm", "linear"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0)]),
                ["--layers-in-window", "2"])
        check(r.returncode != 0 and "does not repeat 2x" in r.stderr,
              "an undeclared-but-claimed multi-layer window is refused")

        # 5. a profile the advice does not describe must degrade, not produce confident buckets
        r = run(report_of(["rms_norm"]),
                csv_of([("LayerNormDeviceOperation", 10.0)] + [(f"Op{i}Operation", 50.0)
                                                               for i in range(8)]))
        o = load()
        check(o["confidence"]["degraded"], "mismatched inputs report DEGRADED")

        # 6. the noise floor decides measurability
        (d / "inc.json").write_text(json.dumps({"incumbent_ms": 0.1, "repeats_ms": [0.100, 0.150]}))
        r = run(report_of(["rms_norm", "linear", "add"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]),
                ["--incumbent", str(d / "inc.json")])
        o = load()
        check(o["feasibility"]["verdict"] == "not_measurable",
              "a 2.0 us ceiling under a 50 us floor is not_measurable")

        # ---- v3 ----------------------------------------------------------------------------------
        # 7. the advised core count is the GRID PRODUCT, not the cores= bounding box
        check(grid_cores("l1/height_sharded/32x1 cores=(0,0)-(10,1)") == 32 and
              bbox_cores("l1/height_sharded/32x1 cores=(0,0)-(10,1)") == 22 and
              grid_cores("l1/interleaved/10x11") == 110,
              "advised cores come from the grid string, and an interleaved layout has one")

        # 8. the legal ladder reproduces the one measured on north-mini, and excludes what hard-failed
        lad = legal_ladder(64)
        check(lad["ladder"] == [1, 2, 4, 8, 11, 16, 22, 32, 64] and
              not any(c in lad["padding_legal"] for c in (40, 44, 48, 55, 88)),
              "legal ladder at W=64 is {1,2,4,8,11,16,22,32,64}, excluding 40/44/48/55/88")

        # 9. an op the advisor declared unplaceable is not screenable, and not a disagreement
        r = run(report_of(["rms_norm", "nlp_concat_heads_decode"],
                          unfixable_ops=[{"op": "ttnn.nlp_concat_heads_decode", "reason": "TT_FATAL x"}]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("NLPConcatHeadsDecodeDeviceOperation", 90.0)]))
        o = load()
        check(o["accounting"].get("advisor_unfixable", {}).get("us") == 90.0 and
              not any(x.get("chain") for x in o["disagreements"] if x.get("op") == "nlp_concat_heads_decode"),
              "an unfixable op gets its own bucket and no chain")

        # 10. the cliff check fires on a 1-core op the advisor wants widened, and the ceiling does not veto
        (d / "inc2.json").write_text(json.dumps({"incumbent_ms": 0.1, "repeats_ms": [0.1000, 0.1002]}))
        rep = json.loads(report_of(["rms_norm", "linear"]))
        rep["ops"][0]["layout"] = "l1/width_sharded/1x22 cores=(0,0)-(10,1)"
        r = run(json.dumps(rep),
                "OP CODE,DEVICE KERNEL DURATION [ns],CORE COUNT,INPUT 0 MEMORY\n"
                "LayerNormDeviceOperation,20000,1,DEV_0_L1_WIDTH_SHARDED\n"
                "MatmulDeviceOperation,80000,32,DEV_0_L1_WIDTH_SHARDED\n",
                ["--incumbent", str(d / "inc2.json")])
        o = load()
        check(len(o["cliff_candidates"]) == 1 and o["cliff_candidates"][0]["advised_cores"] == 22,
              "a 1-core op the advisor wants on 22 is a cliff candidate")
        check(o["feasibility"]["verdict"] == "regrid_only",
              "a zero boundary ceiling with a cliff op is regrid_only, not not_measurable")

        # 11. an advised-L1/shipped-DRAM-input row is HINTED, not rebucketed. Input 0 is the input's space
        # and the advisor states a placement for the OUTPUT, so a bucket rule on it invents disagreements:
        # over 21 corpus kind-runs it moved 15 rows and 14 of them were an op reading DRAM and writing L1.
        rep = json.loads(report_of(["rms_norm"]))
        rep["ops"][0]["layout"] = "l1/height_sharded/1x1 cores=(0,0)-(0,0)"
        r = run(json.dumps(rep),
                "OP CODE,DEVICE KERNEL DURATION [ns],CORE COUNT,INPUT 0 MEMORY\n"
                "LayerNormDeviceOperation,5000,1,DEV_0_DRAM_INTERLEAVED\n")
        o = load()
        row = o["disagreements"][0]
        check(row["bucket"] == "agrees_with_shipped" and row.get("agreed_on") == "grid"
              and "space_hint" in row,
              "advised L1 vs a DRAM input is a recorded hint, not a rebucketing")

        # 12. evidence merges by identifier, and a stale identifier is fatal
        r = run(report_of(["rms_norm", "linear", "add"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]))
        chain_id = load()["chains"][0]["chain"]
        (d / "ev.json").write_text(json.dumps({"chains": {chain_id: {"verdict": "rejected",
                                                                    "measured_ms": 1.0}}}))
        r = run(report_of(["rms_norm", "linear", "add"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]),
                ["--evidence", str(d / "ev.json")])
        o = load()
        check(o["chains"][0]["verdict"] == "rejected" and o["chains"][0]["measured_ms"] == 1.0,
              "evidence fills a verdict without hand-editing the output")
        (d / "ev.json").write_text(json.dumps({"chains": {"nope:99": {"verdict": "kept"}}}))
        r = run(report_of(["rms_norm", "linear", "add"]),
                csv_of([("LayerNormDeviceOperation", 10.0), ("MatmulDeviceOperation", 100.0),
                        ("ReshardDeviceOperation", 2.0), ("BinaryNgDeviceOperation", 5.0)]),
                ["--evidence", str(d / "ev.json")])
        check(r.returncode != 0 and "unknown chain" in r.stderr, "a stale evidence identifier is fatal")

    print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
