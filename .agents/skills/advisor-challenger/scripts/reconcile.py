#!/usr/bin/env python3
"""Reconcile the advisor's placement against the shipped graph, and account for every device op.

  reconcile.py --report shard_advise/<kind>/report.json \
               --perf   tracy/incumbent_<kind>_ops.csv \
               --layer-kind <kind> --layers-of-kind N --total-layers M \
               --out    reconciliation_<kind>.json

Emits a partition of the measured window: every device op gets exactly one bucket, and the buckets sum
to 100 %:

  chain:<id>            in a maximal L1-resident run whose placement differs from shipped
  boundary              a conversion op -- what joining the chains either side would remove
  dram_resident         the advisor placed it in DRAM; that is advice too
  agrees_with_shipped   advised == shipped
  untraced              in the profile, absent from the advisor's graph

Candidates are ranked by CONVERSION VALUE -- the microseconds of boundary ops a change would remove --
not by the advised ops' window share. A chain-lengthening change removes conversions, and those are
separate device ops that an op-share threshold never counts.

Fails loudly: an unmapped op or an accounting that does not close is an error, not a silent gap.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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
# NO DEVICE OP: host-side tensor creation, and the two ops the advisor lists in reshards[] rather than
# ops[]. NOT reshape: a reshape is sometimes a free view and sometimes a real ReshapeView kernel, so let
# the pairing decide instead of asserting it is free.
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


def parse_perf(path: Path, incumbent_ms: float | None = None):
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
    if code_col is None or dur_col is None:
        sys.exit(f"FATAL: {path}: need an op-code and a duration column; saw {sorted(keys)}")
    ns = "ns" in dur_col.lower() or "cycle" in dur_col.lower()
    scale = 1 / 1000.0 if ns else 1.0

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
        dsc = find("dram sharded", exact=False)
        ops.append({"id": r.get(find("id") or "", None), "cls": device_class(code), "code": code,
                    "us": us, "cores": cores,
                    "dram_sharded": (r.get(dsc) or "").strip().lower() in ("true", "1", "yes")
                                    if dsc else False})
        total += us

    # A report that was not bounded to one iteration by signposts repeats its op sequence. Detect that
    # structurally rather than by a magic ratio: shares computed over N iterations are N times too small.
    seq = [o["cls"] for o in ops]
    for period in range(1, len(seq) // 2 + 1):
        if len(seq) % period == 0 and seq == seq[:period] * (len(seq) // period):
            reps = len(seq) // period
            sys.exit(f"FATAL: {path}: the op sequence repeats {reps}x (period {period} of {len(seq)} "
                     f"rows), so this report covers {reps} iterations rather than one. Re-run "
                     f"tt-perf-report with --start-signpost/--end-signpost bounding a single replay; "
                     f"every share computed here would be {reps}x too small.")

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


def parse_advised(report: dict):
    """Advised ops in program order. Geometry comes from report.json; the IR is not needed."""
    out = []
    for o in report.get("ops", []):
        lay = o.get("layout", "")
        m = re.search(r"cores=\((\d+),(\d+)\)-\((\d+),(\d+)\)", lay)
        cores = ((int(m.group(3)) - int(m.group(1)) + 1) * (int(m.group(4)) - int(m.group(2)) + 1)
                 if m else None)
        name = o["op"].replace("ttnn.", "")
        explicit = None if name in UNPAIRED_ADVISED else TTNN_TO_DEVICE.get(name, "__auto__")
        out.append({"index": o["index"], "op": name, "explicit": explicit, "layout": lay,
                    "cores": cores, "program_config": o.get("program_config", "") or "",
                    "space": "dram" if lay.startswith("dram") else
                             ("l1" if lay.startswith("l1") else None)})
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--perf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layer-kind", default="unknown")
    ap.add_argument("--layers-of-kind", type=int, default=1)
    ap.add_argument("--total-layers", type=int, default=1)
    ap.add_argument("--incumbent-ms", type=float, default=None,
                    help="incumbent_ms from incumbent.json; enables a window sanity check")
    ap.add_argument("--ir", type=Path, help="accepted and ignored; geometry comes from report.json")
    a = ap.parse_args()

    for f in (a.report, a.perf):
        if not f.is_file():
            sys.exit(f"FATAL: no such file: {f}")
    report = json.loads(a.report.read_text())
    advised = parse_advised(report)
    reshard_idx = parse_reshards(report)
    device, window = parse_perf(a.perf, a.incumbent_ms)
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
        elif adv["space"] == "dram":
            r.update(bucket="dram_resident", advised=adv["layout"],
                     reason="advisor placed it in DRAM -- that is advice, and it disagrees with a "
                            "sharded shipped op")
            cur = None
        else:
            same = (adv["cores"] is not None and adv["cores"] == dev["cores"]) or \
                   (dev["dram_sharded"] and "dram_sharded" in adv["program_config"])
            r.update(bucket="agrees_with_shipped" if same else "chain",
                     advised=adv["layout"], advised_cores=adv["cores"])
            if not same:
                if cur is None:
                    cur = {"chain": f"{a.layer_kind}:{len(chains)}", "ops": [], "us": 0.0,
                           "boundary_us": 0.0, "verdict": "pending", "measured_ms": None,
                           "repeats_ms": None}
                    chains.append(cur)
                cur["ops"].append(adv["op"])
                cur["us"] += dev["us"]
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

    # conversion value: boundary us adjacent to a chain -- what joining across it would remove
    for i, r in enumerate(rows):
        if r["bucket"] != "boundary":
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(rows) and rows[j].get("chain"):
                ch = next(c for c in chains if c["chain"] == rows[j]["chain"])
                ch["boundary_us"] += r["us"] / 2.0
                if r.get("advised_here") is False:
                    ch["advisor_removes_us"] = round(ch.get("advisor_removes_us", 0.0) + r["us"] / 2.0, 3)
    for c in chains:
        c["us"] = round(c["us"], 3)
        c["boundary_us"] = round(c["boundary_us"], 3)
        c["conversion_value_us"] = c["boundary_us"]
        c.setdefault("advisor_removes_us", 0.0)
        c["op_share_pct"] = round(100 * c["us"] / window, 3)
        c["per_model_us"] = round((c["us"] + c["boundary_us"]) * a.layers_of_kind, 3)
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

    low = [r for r in rows if r["bucket"] in ("chain", "agrees_with_shipped")
           and (r.get("shipped_cores") or 99) <= 2 and r["share_pct"] >= 1.0]

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

    out = {
        "generated_by": "advisor-challenger/scripts/reconcile.py", "tool_version": 4,
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
            "Nothing here is a measurement. Every verdict comes from the device.",
        ],
        "layer_kind": a.layer_kind, "layers_of_kind": a.layers_of_kind,
        "total_layers": a.total_layers, "measured_window_us": round(window, 3),
        "accounting": acct, "accounting_closes_100pct": closes,
        "accounted_us": round(accounted, 3),
        "ranked_by": "advisor_removes_us, then conversion_value_us + chain op us -- NOT advised-op share",
        "advised_boundaries": bsum(rows),
        "chains": chains, "material_ops_on_le_2_cores": low,
        "note": "Screen chains in the order given, each as one unit, and record repeats_ms. Do NOT copy "
                "advised core counts: they are selected under a bytes-shaped objective and are often "
                "smaller than the divisors of the tensor's tile count. The advisor's contribution is the "
                "op SET and the DIRECTION of the change; own the geometry.",
        "disagreements": rows,
    }
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
    print(f"   {len(chains)} chain(s), ranked by advisor-attributable conversion value:")
    for c in chains:
        print(f"      {c['chain']:16s} ops {c['us']:8.3f} + boundaries {c['boundary_us']:7.3f} us"
              f" (advisor drops {c['advisor_removes_us']:6.3f})"
              f"   {'/'.join(c['ops'])[:52]}")
    for r in low:
        print(f"   !! {r['device']} on {r['shipped_cores']} core(s), {r['us']:.3f} us "
              f"({r['share_pct']} %) -- needs a measured attempt or a quoted hard error")
    if not closes:
        sys.exit(f"FATAL: accounting does not close: {accounted:.3f} of {window:.3f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
