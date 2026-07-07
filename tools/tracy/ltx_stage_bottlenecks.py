#!/usr/bin/env python3
"""Per-stage bottleneck rollup for the LTX video pipeline on top of an nsight ops-perf CSV.

nsight (tools/tracy) already emits per-OP device time plus FPU / NoC / CCL utilisation into
ops_perf_results_*.csv. It has no per-STAGE view, so the LTX fast pipeline
(text-encode -> Stage-1 denoise -> Stage-2 denoise -> VAE decode -> audio decode -> video
export) can't be read as a budget. This rolls the per-op rows up into those six stages,
ranks the dominant ops per stage, tags each op compute- / bandwidth- / dispatch-bound from
its counters, and checks the total against an e2e budget.

Two independent time sources are surfaced, because they answer different questions:
  * device-op time from the CSV  -> WHERE inside a stage the device cycles go (rollup + top-N).
  * per-stage wall-clock from the pipeline stderr timers (--log) -> the e2e BUDGET truth,
    since it also contains host / dispatch / D2H gaps that never appear as a device op.
Device-op time for a stage is always <= its wall-clock; the difference is the host/dispatch tax.

Op -> stage mapping precedence (see assign_stages for the why of each rung):
  1. --boundaries      explicit GLOBAL CALL COUNT ranges per stage (fully deterministic).
  2. --csv "STAGE=..."  a whole scoped CSV pinned to one stage (combine per-stage captures).
  3. --assume-stage    a single-family scoped capture pinned to one stage.
  4. signature segment  walk the call-count-ordered ops, switch stage at family-boundary ops
                        (first VAE conv, the S1->S2 token-count jump); documented, LTX-specific.

Cross-device rollup is CRITICAL-PATH: ops on the 8-chip mesh run concurrently, so device
time is max-across-devices per rank-aligned op-instance, never summed (summing 8 SP shards
of one op inflates ~8x). Pin one device with --device to bypass it.

stdlib only (no pandas / numpy) so it runs anywhere the CSV lands.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# The six canonical LTX fast-pipeline stages, in execution order.
CANONICAL_STAGES = [
    "text-encode",
    "Stage 1 denoise",
    "Stage 2 denoise",
    "VAE decode",
    "Audio decode",
    "Video export",
]

# CSV columns consumed. Names track process_ops_logs.py OPS_CSV_HEADER; a run without
# --collect-noc-traces / perf counters leaves the util cells blank, which is handled as "n/a".
COL_OP = "OP CODE"
COL_CALL = "GLOBAL CALL COUNT"
COL_DEV = "DEVICE ID"
COL_FW = "DEVICE FW DURATION [ns]"
COL_FPU = "PM FPU UTIL (%)"
COL_NOC = "NOC BW UTIL FROM COUNTERS (%)"
COL_ETH = "ETH BW UTIL FROM COUNTERS (%)"
COL_CCL = "CCL FABRIC BW UTIL (%)"
COL_OUT_Y = "OUTPUT_0_Y_PAD[LOGICAL]"
COL_OUT_X = "OUTPUT_0_X_PAD[LOGICAL]"

# Op-code substrings that place an op in a stage family. Matching is case-insensitive
# substring. This is the LTX DiT / VAE op vocabulary; a different model needs it retuned.
# Layout/elementwise glue (Tilize, Typecast, Slice, ...) is intentionally NOT keyed here: it
# recurs in every stage, so it is assigned by temporal position, not by op code.
VAE_SIGNATURES = ("conv", "groupnorm", "upsample", "downsample", "interpolate")
DENOISE_SIGNATURES = (
    "sdpa",
    "ringjointsdpa",
    "rmsnorm",
    "allgather",
    "reducescatter",
    "allreduce",
    "minimalmatmul",
    "rotaryembedding",
    "createheads",
    "concatheads",
)
TEXTENCODE_SIGNATURES = ("embedding", "encoder", "t5")
# Ops that are compute by construction, so a blank/zero perf-model FPU is "unmodelled", not
# proof of a dispatch stall.
COMPUTE_OPS = ("matmul", "sdpa", "conv")


def _f(v: str) -> float | None:
    """Parse a CSV numeric cell; blank / non-numeric -> None (an absent counter, not a zero)."""
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _dim(v: str) -> int:
    """OUTPUT dims arrive as '9696[9696]'; take the logical value, tolerate junk."""
    if not v:
        return 0
    m = re.match(r"\s*(\d+)", v)
    return int(m.group(1)) if m else 0


@dataclass
class Op:
    """One op-instance after critical-path collapse across the mesh."""

    code: str
    rank: int  # position in per-device call-count order; the cross-device alignment key
    call: int
    fw_ns: float  # critical-path device-FW time (max over devices for this rank)
    fpu: float | None
    noc: float | None
    eth: float | None
    ccl: float | None
    tokens: int  # OUTPUT_0 Y*X, the S1->S2 resolution-jump signal
    dev_spread_ns: tuple[float, float] = (0.0, 0.0)  # (min,max) over devices, for skew visibility


@dataclass
class StageRoll:
    name: str
    ops: list[Op] = field(default_factory=list)
    wall_s: float | None = None  # from --log timers, if present
    budget_s: float | None = None

    @property
    def device_ms(self) -> float:
        return sum(o.fw_ns for o in self.ops) / 1e6


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def collapse_critical_path(rows: list[dict], pin_device: int | None) -> list[Op]:
    """Collapse the mesh to a single critical-path op stream.

    Ops are SPMD across devices, so rank-align each device's call-count-ordered stream and
    take max device-FW per rank (the slowest shard gates the step). Falls back to a single
    device if the per-device op counts differ (non-SPMD capture) or --device is set.
    """
    by_dev: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dev[r.get(COL_DEV, "0")].append(r)
    for d in by_dev:
        by_dev[d].sort(key=lambda r: int(r[COL_CALL] or 0))

    devs = sorted(by_dev, key=lambda d: int(d) if d.isdigit() else d)
    counts = {d: len(by_dev[d]) for d in devs}
    aligned = len(set(counts.values())) == 1 and len(devs) > 1

    def op_from(r: dict, rank: int, fw: float, spread: tuple[float, float]) -> Op:
        return Op(
            code=r[COL_OP],
            rank=rank,
            call=int(r[COL_CALL] or 0),
            fw_ns=fw,
            fpu=_f(r.get(COL_FPU, "")),
            noc=_f(r.get(COL_NOC, "")),
            eth=_f(r.get(COL_ETH, "")),
            ccl=_f(r.get(COL_CCL, "")),
            tokens=_dim(r.get(COL_OUT_Y, "")) * _dim(r.get(COL_OUT_X, "")),
            dev_spread_ns=spread,
        )

    if pin_device is not None or not aligned:
        d = str(pin_device) if pin_device is not None else devs[0]
        if d not in by_dev:
            sys.exit(f"device {d} not in CSV (have {devs})")
        return [op_from(r, i, _f(r.get(COL_FW, "")) or 0.0, (0.0, 0.0)) for i, r in enumerate(by_dev[d])]

    ref = by_dev[devs[0]]
    out: list[Op] = []
    for i in range(len(ref)):
        fws = [_f(by_dev[d][i].get(COL_FW, "")) or 0.0 for d in devs]
        # Attribute the critical-path row to the slowest device so its util reflects that shard.
        slow = max(range(len(devs)), key=lambda k: fws[k])
        out.append(op_from(by_dev[devs[slow]][i], i, max(fws), (min(fws), max(fws))))
    return out


def parse_boundaries(spec: str) -> dict[str, tuple[int, int]]:
    """--boundaries 'Stage 1 denoise=1024:8191,Stage 2 denoise=8192:20000' -> {name:(lo,hi)}."""
    out: dict[str, tuple[int, int]] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, rng = part.partition("=")
        lo, _, hi = rng.partition(":")
        out[name.strip()] = (int(lo), int(hi))
    return out


def family_of(code: str) -> str | None:
    c = code.lower()
    if any(s in c for s in VAE_SIGNATURES):
        return "VAE decode"
    if any(s in c for s in DENOISE_SIGNATURES):
        return "denoise"
    if any(s in c for s in TEXTENCODE_SIGNATURES):
        return "text-encode"
    return None


def assign_stages(
    ops: list[Op],
    boundaries: dict[str, tuple[int, int]] | None,
    assume_stage: str | None,
    s1_s2_split: int | None,
    s1_steps: int,
) -> dict[str, StageRoll]:
    """Return stage-name -> StageRoll. See module docstring for the precedence rationale.

    Each op gets a base stage (assume_stage, else signature segmentation); explicit
    --boundaries then override any op whose call count falls in a range. Boundaries are a
    partial override, not a whole spec, so unlisted ops are never silently dropped.
    """
    rolls: dict[str, StageRoll] = {name: StageRoll(name) for name in CANONICAL_STAGES}

    def roll(name: str) -> StageRoll:
        return rolls.setdefault(name, StageRoll(name))

    ordered = sorted(ops, key=lambda o: o.rank)

    # Base assignment.
    base: dict[int, str] = {}
    if assume_stage:
        # A scoped single-stage capture: caller asserts which stage the whole CSV is.
        for op in ordered:
            base[id(op)] = assume_stage
    else:
        # Signature segmentation. Walk in temporal (rank) order and switch the active stage at
        # family-boundary ops. Glue ops (family None) inherit the current stage, so weight-quant
        # tilize before the first transformer op stays with the stage it precedes.
        fam_seq = [(o, family_of(o.code)) for o in ordered]
        has_text = any(f == "text-encode" for _, f in fam_seq)

        # S1->S2 boundary: denoise runs Stage-1 (s1_steps steps, low-res) then Stage-2 (full-res).
        # Token count is NOT monotonic across a block (matmuls spike Y*X), so split by STEP COUNT:
        # the block's first op code recurs once per denoise step, so Stage 2 begins at the
        # (s1_steps+1)-th occurrence. Overridable by --s1-s2-split (a call count).
        dn = [o for o, f in fam_seq if f == "denoise"]
        split_call = math.inf
        if dn and s1_s2_split is not None:
            split_call = s1_s2_split
        elif dn and s1_steps > 0:
            marker = dn[0].code
            starts = [o.call for o in dn if o.code == marker]
            if len(starts) > s1_steps:
                split_call = starts[s1_steps]

        current = "text-encode" if has_text else "Stage 1 denoise"
        for op, fam in fam_seq:
            if fam == "text-encode":
                current = "text-encode"
            elif fam == "VAE decode":
                current = "VAE decode"
            elif fam == "denoise":
                current = "Stage 2 denoise" if op.call >= split_call else "Stage 1 denoise"
            # Audio decode has no distinct device-op signature; it is resolved by --log or
            # --boundaries. Glue ops after VAE stay in VAE unless a boundary moves them.
            base[id(op)] = current

    for op in ordered:
        stage = base[id(op)]
        if boundaries:
            for name, (lo, hi) in boundaries.items():
                if lo <= op.call <= hi:
                    stage = name
                    break
        roll(stage).ops.append(op)
    return rolls


# --- pipeline stderr timers ------------------------------------------------------------

# Each canonical stage's wall-clock line as emitted by pipeline_ltx_distilled.py. The last
# match wins (warm gen; warmup lines are skipped explicitly). Aux timers are shown but are
# not device-mapped stages.
TIMER_PATTERNS = {
    "text-encode": re.compile(r"Encoding \([^)]*\):\s*([\d.]+)s"),
    "Stage 1 denoise": re.compile(r"Stage 1 denoise:\s*([\d.]+)s"),
    "Stage 2 denoise": re.compile(r"Stage 2 denoise:\s*([\d.]+)s"),
    "VAE decode": re.compile(r"VAE decode \(forward\):\s*([\d.]+)s"),
    "Audio decode": re.compile(r"Audio decode:\s*([\d.]+)s"),
    "Video export": re.compile(r"Video export:\s*([\d.]+)s"),
}
AUX_TIMER_PATTERNS = {
    "Transformer prepare": re.compile(r"Transformer prepare:\s*([\d.]+)s"),
    "Latent upsample": re.compile(r"Latent upsample:\s*([\d.]+)s"),
    "VAE prepare": re.compile(r"VAE prepare:\s*([\d.]+)s"),
}


_STEP_RE = re.compile(r"Step \d+/(\d+):")


def parse_log(path: str) -> tuple[dict[str, float], dict[str, float], int | None]:
    """Return (stage wall-clock, aux wall-clock, Stage-1 step count).

    The step count is the N in the "Step i/N" lines that precede "Stage 1 denoise:"; it feeds
    the S1->S2 op split so the segmenter needs no shape guessing.
    """
    stage_s: dict[str, float] = {}
    aux_s: dict[str, float] = {}
    s1_steps: int | None = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "warmup" in line.lower():
                continue
            if "Stage 1 denoise:" not in line:
                m = _STEP_RE.search(line)
                if m and s1_steps is None:
                    s1_steps = int(m.group(1))
            for name, pat in TIMER_PATTERNS.items():
                m = pat.search(line)
                if m:
                    stage_s[name] = float(m.group(1))
            for name, pat in AUX_TIMER_PATTERNS.items():
                m = pat.search(line)
                if m:
                    aux_s[name] = float(m.group(1))
    return stage_s, aux_s, s1_steps


# --- classification --------------------------------------------------------------------


def classify(op: Op, fpu_thresh: float, bw_thresh: float) -> tuple[str, str]:
    """Return (label, css-class) tagging what bounds the op, from its counters."""
    ccl = op.ccl or 0.0
    noc = op.noc or 0.0
    eth = op.eth or 0.0
    if ccl >= bw_thresh:
        return f"CCL-bound ({ccl:.0f}% fabric)", "bw"
    if max(noc, eth) >= bw_thresh:
        which = "NoC" if noc >= eth else "ETH"
        return f"{which}-bound ({max(noc, eth):.0f}%)", "bw"
    if op.fpu is not None and op.fpu >= fpu_thresh:
        return f"compute-bound (FPU {op.fpu:.0f}%)", "cmp"
    if op.fpu in (None, 0.0) and any(s in op.code.lower() for s in COMPUTE_OPS):
        return "compute (FPU unmodelled)", "cmp"
    return "dispatch/host-bound (util low)", "disp"


# --- reporting -------------------------------------------------------------------------


def _util(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}%"


def build_report(
    rolls: dict[str, StageRoll],
    aux_wall: dict[str, float],
    budget_s: float,
    top_n: int,
    fpu_thresh: float,
    bw_thresh: float,
    provenance: str,
) -> tuple[str, str]:
    """Return (plaintext, html). Ordered by CANONICAL_STAGES then any extra stages."""
    order = [n for n in CANONICAL_STAGES if n in rolls] + [n for n in rolls if n not in CANONICAL_STAGES]
    active = [rolls[n] for n in order if rolls[n].ops or (rolls[n].wall_s or 0) > 0]

    total_device_ms = sum(r.device_ms for r in active) or 1.0
    total_wall_s = sum((r.wall_s or 0.0) for r in active)

    # ---- plaintext ----
    lines = []
    lines.append("LTX per-stage bottleneck rollup")
    lines.append(provenance)
    lines.append("")
    hdr = f"{'stage':<20} {'device ms':>10} {'% dev':>6} {'wall s':>8} {'% e2e':>6} {'budget':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in active:
        pct_dev = 100 * r.device_ms / total_device_ms
        wall = "" if r.wall_s is None else f"{r.wall_s:8.2f}"
        pct_e2e = "" if (r.wall_s is None or not total_wall_s) else f"{100 * r.wall_s / total_wall_s:5.0f}%"
        over = ""
        if r.wall_s is not None and r.budget_s is not None:
            d = r.wall_s - r.budget_s
            over = f"{'+' if d >= 0 else ''}{d:.2f}s"
        lines.append(f"{r.name:<20} {r.device_ms:10.2f} {pct_dev:5.0f}% {wall:>8} {pct_e2e:>6} {over:>7}")
    lines.append("-" * len(hdr))
    lines.append(
        f"{'TOTAL':<20} {total_device_ms:10.2f} {'100%':>6} " f"{total_wall_s:8.2f} {'':>6} budget={budget_s:.2f}s"
    )
    if total_wall_s:
        d = total_wall_s - budget_s
        verdict = f"OVER by {d:.2f}s (need -{100 * d / total_wall_s:.0f}%)" if d > 0 else f"under by {-d:.2f}s"
        lines.append(f"{'':<20} e2e wall {total_wall_s:.2f}s vs budget {budget_s:.2f}s -> {verdict}")
    if aux_wall:
        lines.append("aux timers: " + ", ".join(f"{k} {v:.2f}s" for k, v in aux_wall.items()))
    lines.append("")

    for r in active:
        if not r.ops:
            lines.append(f"[{r.name}] no device ops (host/dispatch stage; time from log)")
            lines.append("")
            continue
        agg: dict[str, list[Op]] = defaultdict(list)
        for op in r.ops:
            agg[op.code].append(op)
        ranked = sorted(agg.items(), key=lambda kv: sum(o.fw_ns for o in kv[1]), reverse=True)
        lines.append(f"[{r.name}]  device {r.device_ms:.2f} ms across {len(r.ops)} ops — top {top_n}:")
        for code, ins in ranked[:top_n]:
            ms = sum(o.fw_ns for o in ins) / 1e6
            rep = max(ins, key=lambda o: o.fw_ns)
            label, _ = classify(rep, fpu_thresh, bw_thresh)
            lines.append(
                f"    {ms:8.2f} ms  x{len(ins):<3d} {code:<34.34} "
                f"FPU {_util(rep.fpu):>4} NoC {_util(rep.noc):>4} CCL {_util(rep.ccl):>4}  {label}"
            )
        lines.append("")

    text = "\n".join(lines)
    html_doc = _html(
        active, aux_wall, budget_s, total_device_ms, total_wall_s, top_n, fpu_thresh, bw_thresh, provenance
    )
    return text, html_doc


def _bar(pct: float, cls: str) -> str:
    pct = max(0.0, min(100.0, pct))
    return f'<div class="bar"><span class="{cls}" style="width:{pct:.1f}%"></span></div>'


def _html(active, aux_wall, budget_s, total_device_ms, total_wall_s, top_n, fpu_thresh, bw_thresh, provenance):
    e = html.escape
    peak_ms = max((r.device_ms for r in active), default=1.0) or 1.0
    over = total_wall_s - budget_s if total_wall_s else 0.0

    cards = []
    cards.append(
        _card(
            "e2e wall-clock",
            f"{total_wall_s:.2f}<small> s</small>" if total_wall_s else "n/a",
            "sum of per-stage timers (--log)",
        )
    )
    cards.append(
        _card(
            "e2e budget",
            f"{budget_s:.2f}<small> s</small>",
            ("OVER by %.2fs" % over) if over > 0 else ("under by %.2fs" % -over) if total_wall_s else "set --budget",
        )
    )
    cards.append(
        _card("device critical-path", f"{total_device_ms:.1f}<small> ms</small>", "max-across-mesh per op, summed")
    )
    slow = max(active, key=lambda r: r.device_ms, default=None)
    cards.append(
        _card("hottest stage (device)", e(slow.name) if slow else "n/a", f"{slow.device_ms:.1f} ms" if slow else "")
    )

    # Stage rollup + budget table.
    rows = []
    for r in active:
        pct_dev = 100 * r.device_ms / (total_device_ms or 1)
        wall = "—" if r.wall_s is None else f"{r.wall_s:.2f} s"
        pct_e2e = 100 * r.wall_s / total_wall_s if (r.wall_s and total_wall_s) else 0.0
        budget_cell = "—"
        cls = "ok"
        if r.wall_s is not None and r.budget_s is not None:
            d = r.wall_s - r.budget_s
            cls = "bad" if d > 0 else "ok"
            budget_cell = f"{'+' if d >= 0 else ''}{d:.2f} s vs {r.budget_s:.2f}"
        rows.append(
            f"<tr><td><strong>{e(r.name)}</strong></td>"
            f"<td class='num'>{r.device_ms:.2f} ms</td>"
            f"<td>{_bar(100 * r.device_ms / peak_ms, 'cmp')}</td>"
            f"<td class='num'>{wall}</td>"
            f"<td class='num'>{pct_e2e:.0f}%</td>"
            f"<td class='num {cls}'>{budget_cell}</td></tr>"
        )
    budget_row = ""
    if total_wall_s:
        vcls = "bad" if over > 0 else "good"
        vtxt = (
            f"OVER by {over:.2f}s — need −{100 * over / total_wall_s:.0f}%"
            if over > 0
            else f"under budget by {-over:.2f}s"
        )
        budget_row = (
            f"<tr class='total'><td>TOTAL</td><td class='num'>{total_device_ms:.1f} ms</td><td></td>"
            f"<td class='num'>{total_wall_s:.2f} s</td><td class='num'>100%</td>"
            f"<td class='num {vcls}'>{vtxt}</td></tr>"
        )

    aux = ""
    if aux_wall:
        aux = (
            "<p class='muted small'>Aux (non-stage) timers: "
            + " · ".join(f"{e(k)} {v:.2f}s" for k, v in aux_wall.items())
            + "</p>"
        )

    # Per-stage top-N.
    sections = []
    for r in active:
        if not r.ops:
            sections.append(
                f"<h3>{e(r.name)}</h3><p class='muted'>No device ops — host/dispatch stage; time from the pipeline log only.</p>"
            )
            continue
        agg = defaultdict(list)
        for op in r.ops:
            agg[op.code].append(op)
        ranked = sorted(agg.items(), key=lambda kv: sum(o.fw_ns for o in kv[1]), reverse=True)
        trs = []
        stage_ms = r.device_ms or 1.0
        for code, ins in ranked[:top_n]:
            ms = sum(o.fw_ns for o in ins) / 1e6
            rep = max(ins, key=lambda o: o.fw_ns)
            label, kind = classify(rep, fpu_thresh, bw_thresh)
            pill = {"cmp": "cmp", "bw": "bw", "disp": "disp"}[kind]
            trs.append(
                f"<tr><td class='num'>{ms:.2f}</td><td class='num'>{100 * ms / stage_ms:.0f}%</td>"
                f"<td class='num'>×{len(ins)}</td><td class='mono'>{e(code)}</td>"
                f"<td class='num'>{_util(rep.fpu)}</td><td class='num'>{_util(rep.noc)}</td>"
                f"<td class='num'>{_util(rep.ccl)}</td>"
                f"<td><span class='pill {pill}'>{e(label)}</span></td></tr>"
            )
        sections.append(
            f"<h3>{e(r.name)} <span class='muted small'>· {r.device_ms:.2f} ms device · {len(r.ops)} ops</span></h3>"
            f"<table class='ops'><tr><th>ms</th><th>%stg</th><th>n</th><th>op</th>"
            f"<th>FPU</th><th>NoC</th><th>CCL</th><th>bound</th></tr>{''.join(trs)}</table>"
        )

    return _TEMPLATE.format(
        cards="".join(cards),
        stage_rows="".join(rows) + budget_row,
        aux=aux,
        sections="".join(sections),
        provenance=e(provenance),
        fpu_thresh=fpu_thresh,
        bw_thresh=bw_thresh,
    )


def _card(k: str, v: str, note: str) -> str:
    return f"<div class='card'><div class='k'>{html.escape(k)}</div><div class='v'>{v}</div><div class='muted small'>{html.escape(note)}</div></div>"


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LTX stage bottlenecks — nsight</title>
<style>
 :root{{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--border:#30363d;--fg:#e6edf3;
  --muted:#9da7b3;--accent:#4493f8;--good:#3fb950;--warn:#d29922;--bad:#f85149;
  --cmp:#4493f8;--bw:#d29922;--disp:#8b949e;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;}}
 @media (prefers-color-scheme:light){{:root{{--bg:#f6f8fa;--panel:#fff;--panel2:#eef1f5;
  --border:#d0d7de;--fg:#1f2328;--muted:#59636e;}}}}
 :root[data-theme=dark]{{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--border:#30363d;--fg:#e6edf3;--muted:#9da7b3;}}
 :root[data-theme=light]{{--bg:#f6f8fa;--panel:#fff;--panel2:#eef1f5;--border:#d0d7de;--fg:#1f2328;--muted:#59636e;}}
 *{{box-sizing:border-box;}}
 body{{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}}
 .wrap{{max-width:1080px;margin:0 auto;padding:32px 24px 80px;}}
 h1{{font-size:25px;margin:0 0 4px;letter-spacing:-.3px;}}
 h2{{font-size:18px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border);}}
 h3{{font-size:15px;margin:22px 0 6px;}}
 .sub{{color:var(--muted);margin:0 0 8px;}}
 .small{{font-size:12.5px;}} .muted{{color:var(--muted);}}
 .mono{{font-family:var(--mono);font-size:.9em;}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0;}}
 .card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}}
 .card .k{{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.4px;}}
 .card .v{{font-size:22px;font-weight:650;margin:4px 0;}} .card .v small{{font-size:13px;color:var(--muted);font-weight:400;}}
 .tablewrap{{overflow-x:auto;}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;background:var(--panel);
  border:1px solid var(--border);border-radius:10px;overflow:hidden;font-size:13.5px;}}
 th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap;}}
 th{{background:var(--panel2);color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.3px;}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
 tr:last-child td{{border-bottom:none;}}
 tr.total td{{font-weight:650;background:var(--panel2);}}
 .good{{color:var(--good);}} .bad{{color:var(--bad);}} .ok{{color:var(--fg);}}
 .bar{{background:var(--panel2);border-radius:5px;height:9px;min-width:90px;overflow:hidden;}}
 .bar span{{display:block;height:100%;}} .bar .cmp{{background:var(--accent);}}
 .pill{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;}}
 .pill.cmp{{background:rgba(68,147,248,.16);color:var(--cmp);}}
 .pill.bw{{background:rgba(210,153,34,.16);color:var(--warn);}}
 .pill.disp{{background:rgba(139,148,158,.18);color:var(--disp);}}
 .note{{background:var(--panel2);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px;margin:14px 0;font-size:13.5px;}}
 .foot{{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--border);padding-top:14px;}}
</style></head><body><div class="wrap">
 <h1>LTX per-stage bottlenecks</h1>
 <p class="sub">nsight ops-perf rollup · critical-path device time + per-stage wall-clock budget</p>
 <div class="note">{provenance}</div>
 <div class="grid">{cards}</div>
 <h2>Stage rollup &amp; e2e budget</h2>
 <div class="tablewrap"><table>
  <tr><th>stage</th><th>device time</th><th>share (device)</th><th>wall-clock</th><th>% e2e</th><th>vs budget</th></tr>
  {stage_rows}
 </table></div>
 {aux}
 <p class="muted small">Device time = mesh critical-path (max-across-chip per op, summed) — never the
  cross-chip sum. It is ≤ wall-clock; the gap is host / dispatch / D2H that is not a device op.
  Bound tag thresholds: FPU ≥ {fpu_thresh:.0f}% = compute, NoC/CCL ≥ {bw_thresh:.0f}% = bandwidth, else dispatch/host.</p>
 <h2>Top ops per stage</h2>
 {sections}
 <div class="foot">Generated by tools/tracy/ltx_stage_bottlenecks.py. Self-contained; regenerate after each capture.</div>
</div></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="[STAGE=]PATH",
        help="ops_perf_results CSV; repeatable. Prefix 'STAGE=' pins that whole CSV to a stage.",
    )
    ap.add_argument("--log", help="pipeline stderr with the per-stage 'Stage 1 denoise: Xs' timers")
    ap.add_argument("--budget", type=float, default=6.0, help="total e2e budget in seconds (default 6.0)")
    ap.add_argument("--stage-budgets", help="per-stage targets, e.g. 'Stage 1 denoise=2.0,Stage 2 denoise=1.5'")
    ap.add_argument("--boundaries", help="explicit GLOBAL CALL COUNT ranges, e.g. 'Stage 1 denoise=1024:8191,...'")
    ap.add_argument("--assume-stage", help="pin a single-family scoped CSV to one stage")
    ap.add_argument(
        "--s1-s2-split", type=int, help="GLOBAL CALL COUNT where Stage 1 -> Stage 2 (else auto by step count)"
    )
    ap.add_argument(
        "--s1-steps",
        type=int,
        default=6,
        help="Stage-1 denoise steps for the auto S1/S2 split (default 6; --log overrides)",
    )
    ap.add_argument("--device", type=int, help="pin one DEVICE ID instead of the mesh critical-path")
    ap.add_argument("--top", type=int, default=6, help="top-N ops per stage (default 6)")
    ap.add_argument("--fpu-thresh", type=float, default=30.0, help="FPU%% at/above which an op is compute-bound")
    ap.add_argument("--bw-thresh", type=float, default=20.0, help="NoC/CCL%% at/above which an op is bandwidth-bound")
    ap.add_argument("--html", help="write the HTML report here")
    args = ap.parse_args(argv)

    boundaries = parse_boundaries(args.boundaries) if args.boundaries else None
    sb: dict[str, float] = {}
    if args.stage_budgets:
        for part in args.stage_budgets.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                sb[k.strip()] = float(v)

    # The log is parsed first: its Stage-1 step count drives the S1/S2 op split.
    stage_wall, aux_wall, log_s1_steps = ({}, {}, None)
    if args.log:
        stage_wall, aux_wall, log_s1_steps = parse_log(args.log)
    s1_steps = log_s1_steps if log_s1_steps else args.s1_steps

    # Merge all CSVs. A 'STAGE=path' CSV is pinned to that stage; stage names hold no '=' and
    # paths hold none either, so the first '=' cleanly separates the two.
    merged: dict[str, StageRoll] = {name: StageRoll(name) for name in CANONICAL_STAGES}
    srcs = []
    for spec in args.csv:
        if "=" in spec:
            stage_tag, path = (s.strip() for s in spec.split("=", 1))
        else:
            stage_tag, path = "", spec
        rows = load_rows(path)
        srcs.append(path)
        ops = collapse_critical_path(rows, args.device)
        rolls = assign_stages(
            ops,
            boundaries,
            assume_stage=(stage_tag or args.assume_stage) or None,
            s1_s2_split=args.s1_s2_split,
            s1_steps=s1_steps,
        )
        for name, r in rolls.items():
            merged.setdefault(name, StageRoll(name)).ops.extend(r.ops)

    for name, r in merged.items():
        if name in stage_wall:
            r.wall_s = stage_wall[name]
        if name in sb:
            r.budget_s = sb[name]

    provenance = f"Source CSV(s): {', '.join(srcs)}"
    if args.log:
        provenance += f" · log: {args.log}"
    provenance += f" · rollup: {'device ' + str(args.device) if args.device is not None else 'mesh critical-path'}"

    text, html_doc = build_report(merged, aux_wall, args.budget, args.top, args.fpu_thresh, args.bw_thresh, provenance)
    print(text)
    if args.html:
        with open(args.html, "w") as fh:
            fh.write(html_doc)
        print(f"\nHTML report -> {args.html}")


if __name__ == "__main__":
    main()
