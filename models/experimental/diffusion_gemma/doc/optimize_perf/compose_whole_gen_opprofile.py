# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compose the WHOLE-GENERATION op-level breakdown for DiffusionGemma (#47465).

Reads two reduced-layer Tracy op CSVs produced by prof_denoise_step.py at two
layer counts (default 2 and 6, the 6-layer point includes a full-attention
layer), then linearly extrapolates every op-code's per-phase device time to the
full 30-layer model and composes the model-faithful whole generation:

    whole_gen = prefill(30L) x 1  +  denoise_step(30L) x STEPS  +  commit_token(30L) x COMMIT_TOKENS

Why reduced-layer + scale (not a single 30-layer capture): the on-device
profiler op buffer (PROGRAM_SUPPORT_COUNT) holds only a few thousand ops; a
single 30-layer forward is ~30k ops, so a direct 30-layer capture silently drops
device timing after the first few thousand ops. The op topology is per-layer and
linear (SparseMatmul count scales exactly xN with layers; sliding vs full
attention layers see identical positions at this short context so their per-layer
cost is identical), so the 2-point fit is exact up to measurement noise.

Two device metrics are reported per phase:
  * sum-of-device-FW per op-code -> the OP MIX (which ops dominate). Standard
    tt-perf-report style attribution. NOTE: on the mesh the per-op FW windows
    OVERLAP (~1.74x in denoise: concurrent programs), so sum-FW is larger than
    wall time; it is used only for RELATIVE op share.
  * device-busy SPAN (max FW end - min FW start, cycles / AICLK) -> the true
    per-phase device time. Validated to match the warmed wall-clock exactly
    (denoise 2L span 352ms == wall 352.6ms; 6L span 916.9ms == wall 917.5ms).

Usage:
    python compose_whole_gen_opprofile.py \
        --csv2 /tmp/dg_run1_2L.csv --n2 2 --ncommit2 8 \
        --csv6 /tmp/dg_run2_6L.csv --n6 6 --ncommit6 4 \
        --out-dir models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

AICLK_HZ = 1.35e9  # Blackhole AICLK; validated: denoise span(cyc)/AICLK == warmed wall-clock.

PHASES = [
    ("PREFILL", "PREFILL_START", "PREFILL_END"),
    ("DENOISE", "DENOISE_START", "DENOISE_END"),
    ("COMMIT", "COMMIT_START", "COMMIT_END"),
]


def _load(path):
    rows = list(csv.DictReader(open(path)))
    h = list(rows[0].keys())

    def find(*ts):
        for c in h:
            if all(t in c.lower() for t in ts):
                return c
        return None

    cols = {
        "op": find("op", "code"),
        "id": find("device", "id"),
        "fw": find("device", "fw", "dur"),
        "kn": "DEVICE KERNEL DURATION [ns]",
        "gap": "OP TO OP LATENCY [ns]",
        "cs": "DEVICE FW START CYCLE",
        "ce": "DEVICE FW END CYCLE",
        "attr": "ATTRIBUTES",
    }
    return rows, cols


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _perm_label(attr):
    """Give Permute/Transpose a dims-signature label so the MoE-gather (6D) and
    attention head (4D) permutes are separated in the op mix."""
    m = re.search(r"'dims':\s*'\{([^}]*)\}'", attr or "")
    if not m:
        return None
    dims = m.group(1).replace(" ", "")
    ndim = dims.count(";") + 1
    kind = "MoE-gather-6D" if ndim >= 5 else ("attn-head-4D" if ndim == 4 else f"{ndim}D")
    return f"[{kind} {{{dims}}}]"


def _phase_agg(rows, cols, a, b):
    """Return (agg, span_ms, n_ops) for one signpost region on device0.
    agg[label] = [count, fw_ns, kn_ns, gap_ns]."""

    def idxs(name):
        return [i for i, r in enumerate(rows) if (r[cols["op"]] or "").strip() == name]

    lo, hi = idxs(a), idxs(b)
    if not lo or not hi:
        raise SystemExit(f"signpost {a}/{b} missing")
    region = [r for r in rows[min(lo) : max(hi)] if (r.get(cols["id"], "") or "").strip() == "0"]

    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    starts, ends = [], []
    for r in region:
        op = (r[cols["op"]] or "").strip() or "(blank)"
        if op.endswith("_START") or op.endswith("_END"):
            continue
        label = op
        if op in ("PermuteDeviceOperation", "TransposeDeviceOperation"):
            pl = _perm_label(r.get(cols["attr"], ""))
            if pl:
                label = op + pl
        a_ = agg[label]
        a_[0] += 1
        a_[1] += _fnum(r[cols["fw"]]) or 0.0
        a_[2] += _fnum(r[cols["kn"]]) or 0.0
        a_[3] += _fnum(r[cols["gap"]]) or 0.0
        cs, ce = _fnum(r[cols["cs"]]), _fnum(r[cols["ce"]])
        if cs is not None:
            starts.append(cs)
        if ce is not None:
            ends.append(ce)
    span_ms = ((max(ends) - min(starts)) / AICLK_HZ * 1e3) if starts and ends else 0.0
    return agg, span_ms, len(region)


def fit(v2, v6, n2, n6, target):
    """2-point linear extrapolation to `target` layers; clamp >=0."""
    per_layer = (v6 - v2) / (n6 - n2)
    v = v2 + per_layer * (target - n2)
    return max(v, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv2", required=True)
    ap.add_argument("--n2", type=int, default=2)
    ap.add_argument("--ncommit2", type=int, default=8)
    ap.add_argument("--csv6", required=True)
    ap.add_argument("--n6", type=int, default=6)
    ap.add_argument("--ncommit6", type=int, default=4)
    ap.add_argument("--target-layers", type=int, default=30)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--commit-tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--out-dir", default="models/experimental/diffusion_gemma/doc/optimize_perf/whole_gen_opprofile")
    args = ap.parse_args()

    rows2, cols2 = _load(args.csv2)
    rows6, cols6 = _load(args.csv6)

    # Per-phase aggregates at both layer counts.
    data = {}  # phase -> {"agg2","span2","agg6","span6"}
    for name, a, b in PHASES:
        agg2, span2, _ = _phase_agg(rows2, cols2, a, b)
        agg6, span6, _ = _phase_agg(rows6, cols6, a, b)
        # Commit region is N commit tokens -> normalize to per-token.
        if name == "COMMIT":
            for agg, nt in ((agg2, args.ncommit2), (agg6, args.ncommit6)):
                for k in agg:
                    agg[k] = [x / nt for x in agg[k]]
            span2 /= args.ncommit2
            span6 /= args.ncommit6
        data[name] = dict(agg2=agg2, span2=span2, agg6=agg6, span6=span6)

    # Extrapolate every op-label per phase to target layers, then compose whole gen.
    mult = {"PREFILL": args.blocks, "DENOISE": args.steps * args.blocks, "COMMIT": args.commit_tokens * args.blocks}
    phase30 = {}  # phase -> {label: fw_ns_at_30L (per-unit)}
    phase_span30 = {}
    whole = defaultdict(float)  # label -> whole-gen fw_ns
    whole_cnt = defaultdict(float)
    for name, _, _ in PHASES:
        d = data[name]
        labels = set(d["agg2"]) | set(d["agg6"])
        p30 = {}
        for lab in labels:
            v2 = d["agg2"].get(lab, [0, 0, 0, 0])
            v6 = d["agg6"].get(lab, [0, 0, 0, 0])
            fw30 = fit(v2[1], v6[1], args.n2, args.n6, args.target_layers)
            cnt30 = fit(v2[0], v6[0], args.n2, args.n6, args.target_layers)
            p30[lab] = dict(fw=fw30, cnt=cnt30)
            whole[lab] += fw30 * mult[name]
            whole_cnt[lab] += cnt30 * mult[name]
        phase30[name] = p30
        phase_span30[name] = fit(d["span2"], d["span6"], args.n2, args.n6, args.target_layers)

    # Whole-gen phase device-busy time (span basis) and sum-FW basis.
    span_whole = {n: phase_span30[n] * mult[n] for n, _, _ in PHASES}  # ms
    fw_phase_whole = {n: sum(p["fw"] for p in phase30[n].values()) * mult[n] / 1e6 for n, _, _ in PHASES}  # ms
    total_fw_ms = sum(whole.values()) / 1e6
    total_span_ms = sum(span_whole.values())

    os.makedirs(args.out_dir, exist_ok=True)
    lines = []

    def P(s=""):
        lines.append(s)

    # Phase split on the robust device-FW attribution basis (prefill span is cold /
    # JIT-compile-polluted, so span is NOT used for the phase split).
    total_phase_fw = sum(fw_phase_whole.values())
    P("=" * 92)
    P("DiffusionGemma #47465 - WHOLE-GENERATION op-level profile (CURRENT optimized model)")
    P(
        f"model-faithful: {args.target_layers} layers, canvas 256, {args.steps} denoise steps/block, "
        f"{args.commit_tokens} commit tokens/block, {args.blocks} block(s)"
    )
    P(
        f"reduced-layer 2-point fit: L={args.n2} and L={args.n6} (6L includes a full-attention layer) -> L={args.target_layers}"
    )
    P("=" * 92)
    P()
    P("PHASE SPLIT over the whole generation (device-FW attribution basis; robust):")
    P(f"{'PHASE':9s} {'unit sum-FW ms @30L':>20s} {'x mult':>8s} {'whole-gen sum-FW ms':>20s} {'% of gen':>10s}")
    for n, _, _ in PHASES:
        unit = fw_phase_whole[n] / mult[n]
        P(
            f"{n:9s} {unit:>20.1f} {mult[n]:>8d} {fw_phase_whole[n]:>20.0f} {fw_phase_whole[n]/total_phase_fw*100:>9.2f}%"
        )
    P(f"{'TOTAL':9s} {'':>20s} {'':>8s} {total_phase_fw:>20.0f} {100.0:>9.2f}%")
    P()
    P("Supplementary warmed per-unit device-busy SPAN (eager-under-Tracy; profiler-inflated, NOT serving speed):")
    P(f"  denoise step @30L : {phase_span30['DENOISE']:8.0f} ms   (x48 = {span_whole['DENOISE']/1000:.1f} s eager)")
    P(f"  commit token @30L : {phase_span30['COMMIT']:8.1f} ms   (x256 = {span_whole['COMMIT']/1000:.1f} s eager)")
    P(
        f"  prefill      @30L : cold/JIT-compile-polluted span; device-FW work ~= {fw_phase_whole['PREFILL']:.0f} ms (one-time, negligible)"
    )
    P("  -> ACTUAL serving speed is the TRACED path (~17.92 t/s model-faithful, prior #47465 work), not this")
    P("     eager-under-profiler figure, which exists only to attribute per-op device time.")
    P()
    P(
        f"  whole-gen sum-of-FW total (op-attribution basis)  : {total_fw_ms/1000:.1f} s "
        f"(FW windows overlap ~{total_fw_ms/max(total_span_ms,1):.2f}x on the mesh; use for op-mix % only)"
    )
    P()
    P("-" * 92)
    P("WHOLE-GENERATION op mix (sum-of-device-FW attribution; which ops dominate the full generation)")
    P("-" * 92)
    P(f"{'op-code (Permute/Transpose split by dims signature)':58s} {'whole-gen FW ms':>15s} {'%':>7s} {'~ops':>9s}")
    ranked = sorted(whole.items(), key=lambda kv: -kv[1])
    for lab, fw in ranked:
        pct = fw / (total_fw_ms * 1e6) * 100
        if pct < 0.05 and fw / 1e6 < 50:
            continue
        P(f"{lab:58s} {fw/1e6:>15.0f} {pct:>6.1f}% {whole_cnt[lab]:>9.0f}")
    P()
    P("-" * 92)
    P("PER-PHASE op mix at 30 layers (sum-FW ms per single unit: 1 prefill / 1 denoise step / 1 commit token)")
    P("-" * 92)
    for n, _, _ in PHASES:
        tot = sum(p["fw"] for p in phase30[n].values()) / 1e6
        P(
            f"\n[{n}]  unit sum-FW = {tot:.1f} ms   (device-busy span = {phase_span30[n]:.1f} ms)   x{mult[n]} = "
            f"{fw_phase_whole[n]:.0f} ms sum-FW over generation"
        )
        for lab, p in sorted(phase30[n].items(), key=lambda kv: -kv[1]["fw"])[:9]:
            fwm = p["fw"] / 1e6
            if fwm < 0.05:
                continue
            P(f"    {lab:56s} {fwm:>10.1f} ms {fwm/max(tot,1e-9)*100:>6.1f}%  n={p['cnt']:.0f}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "whole_gen_op_breakdown.txt"), "w") as f:
        f.write(report + "\n")

    # Machine-readable summary.
    summary = {
        "workload": {
            "profile": "whole_generation_denoise_diffusion",
            "target_layers": args.target_layers,
            "canvas_length": 256,
            "denoise_steps_per_block": args.steps,
            "commit_tokens_per_block": args.commit_tokens,
            "blocks": args.blocks,
            "model_faithful": True,
        },
        "method": {
            "reduced_layer_fit": [args.n2, args.n6],
            "reason": "on-device profiler op buffer (~3.3k ops) cannot hold a 30-layer forward (~30k ops); "
            "per-layer op topology is linear so 2-point fit -> 30L is exact up to noise",
            "aiclk_hz": AICLK_HZ,
            "fw_overlap_ratio_denoise": round(total_fw_ms / max(total_span_ms, 1), 3),
        },
        "whole_gen_device_busy_span_s": round(total_span_ms / 1000, 2),
        "whole_gen_sum_fw_s": round(total_fw_ms / 1000, 2),
        "phase_span_ms": {n: round(span_whole[n], 1) for n, _, _ in PHASES},
        "phase_span_pct": {n: round(span_whole[n] / total_span_ms * 100, 1) for n, _, _ in PHASES},
        "unit_span_ms_30L": {n: round(phase_span30[n], 2) for n, _, _ in PHASES},
        "whole_gen_op_mix_sum_fw_ms": {lab: round(fw / 1e6, 1) for lab, fw in ranked if fw / 1e6 >= 10},
        "whole_gen_op_mix_pct": {
            lab: round(fw / (total_fw_ms * 1e6) * 100, 1) for lab, fw in ranked if fw / (total_fw_ms * 1e6) * 100 >= 0.1
        },
    }
    with open(os.path.join(args.out_dir, "whole_gen_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_dir}/whole_gen_op_breakdown.txt and whole_gen_summary.json")


if __name__ == "__main__":
    main()
