# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Roofline post-processing for the BH Galaxy MoE benchmark.

    python roofline.py RESULTS_DIR      # reads RESULTS_DIR/prof_*/**/ops_perf_results*.csv, writes roofline.{json,md}

Everything is per chip. The memory roof is DRAM bandwidth (expert weights must stream from DRAM every call); the
compute roof is the chip's LoFi matmul peak (cores x 4096 FLOP/cycle x AICLK). The decode op runs its matmuls on a
ring of DRAM-adjacent cores only, so its table also shows that ring's own compute peak.

Weights are BFP4 (576 B per 1024 elements) on both paths; prefill shared experts are BFP8.
"""

import glob
import json
import math
import os
import re
import statistics
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_models import (  # noqa: E402
    BH_AICLK_MHZ,
    BH_DRAM_GBPS,
    BH_FLOP_PER_CYCLE_PER_CORE_LOFI,
    BYTES_PER_ELEM,
    GALAXY_CHIPS,
)

BF4 = BYTES_PER_ELEM["bfloat4_b"]
BF8 = BYTES_PER_ELEM["bfloat8_b"]
DUR = "DEVICE KERNEL DURATION [ns]"

# Prefill op families, matched on OP CODE (case-insensitive) in order; the first match wins.
PREFILL_FAMILIES = [
    ("routed_expert", r"routed.?expert|fused.?swiglu|expert.?ffn|extract|insert|hybrid"),
    ("dispatch", r"dispatch"),
    ("combine", r"combine|reduce.?nc|post.?combine"),
    ("gate", r"gate|topk|top_k|grouped|hash|sigmoid|softmax|bincount|routing"),
    ("shared_expert", r"matmul|minimal|linear|silu|glu|mul|binary|unary"),
    ("ccl", r"all.?gather|reduce.?scatter|all.?reduce|all.?to.?all"),
]


def _family(op_code: str) -> str:
    low = op_code.lower()
    for fam, pat in PREFILL_FAMILIES:
        if re.search(pat, low):
            return fam
    return "other"


def _parse_tag(tag: str) -> dict:
    parts = tag.split("|")
    out = {"kind": parts[1]}
    for p in parts[2:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return out


def platform(res_dir: str) -> dict:
    """Peak AICLK and DRAM bandwidth from the tt-smi snapshot when present; the spec values otherwise.

    The snapshot is taken at idle, so its live `aiclk` reads the idle clock; the peak is AICLK_LIMIT_MAX. DRAM
    bandwidth scales with the GDDR data rate (DDR_SPEED, MT/s): 512 GB/s at 16000. Galaxies ship with 14000 and
    16000 parts, so the slowest chip's rate sets the roof.
    """
    aiclk = ddr = None
    path = os.path.join(res_dir, "tt_smi_snapshot.json")
    if os.path.exists(path):
        text = open(path).read()
        clk = [int(v, 16) for v in re.findall(r'"AICLK_LIMIT_MAX"\s*:\s*"(0x[0-9a-fA-F]+)"', text)]
        clk = [v for v in clk if 200 < v < 3000]
        aiclk = min(clk) if clk else None
        rates = [int(v, 16) for v in re.findall(r'"DDR_SPEED"\s*:\s*"(0x[0-9a-fA-F]+)"', text)]
        rates = [v for v in rates if 4000 < v < 32000]
        ddr = min(rates) if rates else None
    return {
        "aiclk_mhz": aiclk or BH_AICLK_MHZ,
        "aiclk_source": "tt-smi AICLK_LIMIT_MAX" if aiclk else "spec",
        "ddr_mtps": ddr or 16000,
        "dram_gbps": BH_DRAM_GBPS * (ddr or 16000) / 16000,
        "dram_source": "tt-smi DDR_SPEED" if ddr else "spec",
    }


def split_cases(csv_path: str):
    """Yield (tag dict, DataFrame of device-op rows) between each moe_bench START signpost and its END."""
    df = pd.read_csv(csv_path, low_memory=False)
    is_sp = df["OP TYPE"].astype(str).str.contains("signpost")
    cur, rows = None, []
    for idx, (sp, code) in enumerate(zip(is_sp, df["OP CODE"].astype(str))):
        if sp and code.startswith("moe_bench|"):
            if cur is not None:
                yield cur, df.iloc[rows]
            cur, rows = (code.rsplit("|", 1)[0], []) if code.endswith("|START") else (None, [])
            continue
        if cur is not None and not sp:
            rows.append(idx)
    if cur is not None:
        yield cur, df.iloc[rows]


def decode_row(tag: str, ops: pd.DataFrame, plat: dict) -> dict:
    t = _parse_tag(tag)
    mc = ops[ops["OP CODE"].astype(str).str.contains("moecompute", case=False, regex=False)]
    durs = pd.to_numeric(mc[DUR], errors="coerce").dropna().tolist()
    reps = int(t.get("repeats", 1))
    warm = [d for i, d in enumerate(durs) if reps <= 1 or i % reps != 0]
    row = dict(t, calls=len(durs) // max(reps, 1), samples=len(warm))
    if not warm:
        row["status"] = "no moe_compute rows (failed or skipped)"
        return row
    t_ns = statistics.median(warm)
    H, N, El, k, T = t["H"], t["N"], t["E_local"], t["k"], t["T"]
    active = El * (1 - (1 - k / El) ** T)  # expected distinct experts hit (uniform top-k routing)
    w_bytes_active = active * 3 * H * N * BF4
    w_bytes_all = El * 3 * H * N * BF4
    act_bytes = T * H * 2 + T * k * H * 2
    flops = 2 * T * k * 3 * H * N
    gx, gy = (int(x) for x in str(t["grid"]).split("x"))
    clk = plat["aiclk_mhz"] * 1e6
    peak_chip = gx * gy * BH_FLOP_PER_CYCLE_PER_CORE_LOFI * clk
    peak_ring = int(t["ring"]) * BH_FLOP_PER_CYCLE_PER_CORE_LOFI * clk
    bw = plat["dram_gbps"] * 1e9
    t_mem = (w_bytes_active + act_bytes) / bw
    t_cmp = flops / peak_chip
    t_s = t_ns * 1e-9
    row.update(
        status="ok",
        time_us=t_ns / 1e3,
        time_us_min=min(warm) / 1e3,
        active_experts=active,
        weight_MB=w_bytes_active / 1e6,
        weight_MB_all_local=w_bytes_all / 1e6,
        gflop=flops / 1e9,
        arith_intensity=flops / (w_bytes_active + act_bytes),
        achieved_GBps=(w_bytes_active + act_bytes) / t_s / 1e9,
        pct_dram_peak=100 * (w_bytes_active + act_bytes) / t_s / bw,
        achieved_TFLOPs=flops / t_s / 1e12,
        pct_chip_compute_peak=100 * flops / t_s / peak_chip,
        pct_ring_compute_peak=100 * flops / t_s / peak_ring,
        roofline_us=max(t_mem, t_cmp) * 1e6,
        bound="memory" if t_mem >= t_cmp else "compute",
        pct_of_roofline=100 * max(t_mem, t_cmp) / t_s,
        ring_roofline_us=max(t_mem, flops / peak_ring) * 1e6,
        pct_of_ring_roofline=100 * max(t_mem, flops / peak_ring) / t_s,
    )
    return row


def prefill_row(tag: str, ops: pd.DataFrame, plat: dict) -> dict:
    t = _parse_tag(tag)
    ops = ops.copy()
    ops[DUR] = pd.to_numeric(ops[DUR], errors="coerce")
    ops = ops.dropna(subset=[DUR])
    row = dict(t)
    if ops.empty:
        row["status"] = "no device ops (failed or skipped)"
        return row
    ndev = ops["DEVICE ID"].nunique()
    ops["family"] = ops["OP CODE"].astype(str).map(_family)
    per_dev = ops.groupby(["DEVICE ID"])[DUR].sum()
    fam_per_dev = ops.groupby(["family", "DEVICE ID"])[DUR].sum().groupby("family").mean()
    by_op = (
        ops.groupby("OP CODE")
        .agg(calls=(DUR, "size"), mean_ns=(DUR, "mean"), sum_ns=(DUR, "sum"))
        .assign(per_device_us=lambda d: d.sum_ns / ndev / 1e3, family=lambda d: d.index.map(_family))
        .sort_values("per_device_us", ascending=False)
    )
    span_us = None
    if {"DEVICE FW START CYCLE", "DEVICE FW END CYCLE"} <= set(ops.columns):
        s = pd.to_numeric(ops["DEVICE FW START CYCLE"], errors="coerce")
        e = pd.to_numeric(ops["DEVICE FW END CYCLE"], errors="coerce")
        spans = (e.groupby(ops["DEVICE ID"]).max() - s.groupby(ops["DEVICE ID"]).min()) / (plat["aiclk_mhz"])
        span_us = float(spans.max())

    H, Hr, N, E, k, Nsh = t["H"], t["Hr"], t["N"], t["E"], t["k"], t["Nsh"]
    tpc = t["tokens_per_chip"]
    sp, tp = (int(x) for x in str(t["mesh"]).split("x"))
    chips = sp * tp
    tokens = tpc * sp  # one prefill chunk: SP shards the sequence, TP columns share it
    pairs = tokens * k / chips  # token-expert pairs per chip (experts spread over all chips)
    routed_flops = pairs * 2 * 3 * Hr * N
    routed_bytes = (E / chips) * 3 * Hr * N * BF4 + pairs * Hr * (BF8 + 2)
    shared_flops = tpc * 2 * 3 * H * Nsh / tp
    shared_bytes = 3 * H * Nsh * BF8 / tp
    gx, gy = (int(x) for x in str(t["grid"]).split("x"))
    clk = plat["aiclk_mhz"] * 1e6
    peak = gx * gy * BH_FLOP_PER_CYCLE_PER_CORE_LOFI * clk
    bw = plat["dram_gbps"] * 1e9

    def roof(f, b):
        return max(f / peak, b / bw)

    routed_us = float(fam_per_dev.get("routed_expert", float("nan"))) / 1e3
    layer_us = float(per_dev.mean()) / 1e3
    r_roof = roof(routed_flops, routed_bytes)
    l_roof = roof(routed_flops + shared_flops, routed_bytes + shared_bytes)
    row.update(
        status="ok",
        devices=ndev,
        tokens_per_chunk=tokens,
        pairs_per_chip=pairs,
        routed_gflop_per_chip=routed_flops / 1e9,
        routed_weight_MB_per_chip=(E / chips) * 3 * Hr * N * BF4 / 1e6,
        routed_arith_intensity=routed_flops / routed_bytes,
        routed_expert_us=routed_us,
        routed_TFLOPs=routed_flops / (routed_us * 1e-6) / 1e12 if routed_us == routed_us and routed_us > 0 else None,
        routed_pct_compute_peak=100 * routed_flops / (routed_us * 1e-6) / peak if routed_us > 0 else None,
        routed_roofline_us=r_roof * 1e6,
        routed_bound="memory" if routed_bytes / bw >= routed_flops / peak else "compute",
        routed_pct_of_roofline=100 * r_roof / (routed_us * 1e-6) if routed_us > 0 else None,
        layer_kernel_sum_us=layer_us,
        layer_kernel_sum_max_dev_us=float(per_dev.max()) / 1e3,
        layer_span_us_untraced=span_us,
        layer_roofline_us=l_roof * 1e6,
        layer_pct_of_roofline=100 * l_roof / (layer_us * 1e-6),
        family_us={f: float(v) / 1e3 for f, v in fam_per_dev.items()},
        top_ops=[
            {"op": op, "family": r.family, "calls_per_dev": r.calls / ndev, "per_device_us": r.per_device_us}
            for op, r in by_op.head(15).iterrows()
        ],
    )
    return row


def _fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.{nd}f}" if isinstance(v, float) else str(v)


def main(res_dir: str):
    plat = platform(res_dir)
    out = {"platform": plat, "decode": [], "prefill": [], "gates": {}}
    for csv in sorted(glob.glob(os.path.join(res_dir, "prof_*", "**", "ops_perf_results*.csv"), recursive=True)):
        for tag, ops in split_cases(csv):
            kind = _parse_tag(tag)["kind"]
            (out["decode"] if kind == "decode" else out["prefill"]).append(
                (decode_row if kind == "decode" else prefill_row)(tag, ops, plat)
            )
    for log in glob.glob(os.path.join(res_dir, "gate_*.log")):
        hits = re.findall(r"([\w\-\[\]\.]+): ([0-9,.]+) ns \(([0-9.]+) ms\), expected ([0-9,]+) ns", open(log).read())
        out["gates"][os.path.basename(log)] = [
            {"case": c, "measured_ns": float(m.replace(",", "")), "expected_ns": int(e.replace(",", ""))}
            for c, m, _, e in hits
        ]
    json.dump(out, open(os.path.join(res_dir, "roofline.json"), "w"), indent=1, default=str)

    md = [
        f"# BH Galaxy MoE roofline ({os.path.basename(os.path.normpath(res_dir))})",
        "",
        f"AICLK {plat['aiclk_mhz']:.0f} MHz ({plat['aiclk_source']}), DRAM {plat['dram_gbps']:.0f} GB/s per chip "
        f"(GDDR {plat['ddr_mtps']} MT/s, {plat['dram_source']}), "
        f"LoFi peak {BH_FLOP_PER_CYCLE_PER_CORE_LOFI} FLOP/cycle/core. Galaxy = {GALAXY_CHIPS} chips.",
        "",
    ]
    if out["decode"]:
        md += [
            "## Decode: moe_compute (compute_only, one chip, experts per chip = E/32)",
            "",
            "| model | users/chip | fp32 dest | E/chip | k | time µs | weights MB | GB/s | % DRAM | TFLOP/s | % chip peak | % ring peak | bound | % of roofline | ring roofline µs | % of ring roofline |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in out["decode"]:
            if r.get("status") != "ok":
                md.append(f"| {r.get('model')} | {r.get('T')} | {r.get('fp32_dest')} | | | {r.get('status')} |||||||||")
                continue
            md.append(
                f"| {r['model']} | {r['T']} | {r['fp32_dest']} | {r['E_local']} | {r['k']} | {_fmt(r['time_us'])} | "
                f"{_fmt(r['weight_MB'])} | {_fmt(r['achieved_GBps'], 0)} | {_fmt(r['pct_dram_peak'])} | "
                f"{_fmt(r['achieved_TFLOPs'], 2)} | {_fmt(r['pct_chip_compute_peak'])} | {_fmt(r['pct_ring_compute_peak'])} | "
                f"{r['bound']} | {_fmt(r['pct_of_roofline'])} | {_fmt(r['ring_roofline_us'])} | "
                f"{_fmt(r['pct_of_ring_roofline'])} |"
            )
        md.append("")
    if out["prefill"]:
        md += [
            "## Prefill: TtMoe layer, 8x4 TorusXY, 5k-token chunk (640 tokens/chip)",
            "",
            "| model | routed expert µs | roofline µs | bound | % of roofline | routed TFLOP/s/chip | layer kernel-sum µs | layer roofline µs | % | dispatch µs | combine µs | gate µs | shared µs |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in out["prefill"]:
            if r.get("status") != "ok":
                md.append(f"| {r.get('model')} | {r.get('status')} ||||||||||||")
                continue
            f = r["family_us"]
            md.append(
                f"| {r['model']} | {_fmt(r['routed_expert_us'])} | {_fmt(r['routed_roofline_us'])} | {r['routed_bound']} | "
                f"{_fmt(r['routed_pct_of_roofline'])} | {_fmt(r['routed_TFLOPs'], 1)} | {_fmt(r['layer_kernel_sum_us'])} | "
                f"{_fmt(r['layer_roofline_us'])} | {_fmt(r['layer_pct_of_roofline'])} | {_fmt(f.get('dispatch'))} | "
                f"{_fmt(f.get('combine'))} | {_fmt(f.get('gate'))} | {_fmt(f.get('shared_expert'))} |"
            )
        md.append("")
        for r in out["prefill"]:
            if r.get("status") == "ok":
                md += [f"### {r['model']} top ops (per device)", "", "| op | family | calls | µs |", "|---|---|---|---|"]
                md += [
                    f"| {o['op']} | {o['family']} | {o['calls_per_dev']:.0f} | {o['per_device_us']:,.1f} |"
                    for o in r["top_ops"]
                ]
                md.append("")
    if any(out["gates"].values()):
        md += ["## Traced CI gates (wall clock, full MoE layer)", "", "| case | measured ms | gate ms |", "|---|---|---|"]
        for hits in out["gates"].values():
            md += [f"| {h['case']} | {h['measured_ns'] / 1e6:.3f} | {h['expected_ns'] / 1e6:.3f} |" for h in hits]
    open(os.path.join(res_dir, "roofline.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
