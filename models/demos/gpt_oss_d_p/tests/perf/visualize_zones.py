#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Turn a profiled ops CSV into a report: a table on stdout and a standalone HTML page.

    python3 visualize_zones.py <ops_perf_results_*.csv> [-o report.html]

Reads the zone hierarchy, per-chip spread and wall-clock accounting via parse_zone_perf, then writes
a self-contained HTML page (no external assets) alongside a text summary. Both cover:

  * per-layer-class zone breakdown (sliding vs full attention), split compute / communication /
    KV-cache memory
  * per-chip spread for the zones with the widest imbalance, one cell per device
  * op-level detail inside each zone — which device kernels a zone is actually made of
  * device-busy accounting: kernel time, per-op firmware, and what that projects to at 36 layers

Run the profile first with ./run_prefill_profile.sh, which prints the CSV path when it finishes.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import parse_zone_perf as P  # noqa: E402

# NOTE: ring_joint_sdpa is deliberately NOT a comm key. The cache-backed ring SDPA fuses the SP
# ring-rotation CCL with the attention compute in one device op, so its comm share cannot be split
# out; it is reported as compute. The one-shot fallback's ag_qkv / sdpa_reduce_scatter ARE separate
# ops, which is what makes the CACHE=0 capture the comm/compute reference point for attention.
COMM_KEYS = (
    "ccl_out_allreduce",
    "ccl_out_allgather",
    "ag_qkv",
    "sdpa_reduce_scatter",
    "tp_allgather",
    "dispatch",
    "combine",
    "moe_reduce",
    "pre_dispatch_allgather",
)
MEM_KEYS = ("kv_write", "defrag_move")


def parent_rels(rels):
    """Zones in `rels` that actually have a descendant present in THIS capture.

    Parents hold their children's time too, so they must be excluded from any sum over zones — but
    whether a zone IS a parent depends on the detail level the capture ran at, not on a fixed list:
    a zone whose children are all suppressed at the current LEVEL is a leaf, and its time has to be
    counted; a static exclusion list would silently drop it from the totals.
    """
    keys = set(rels)
    return {k for k in keys if any(o != k and o.startswith(k + "/") for o in keys)} | {"(layer total)"}


# GPT-OSS-120B: 36 layers alternating sliding (even) / full (odd) attention — 18 of each. Per-layer
# zone costs are layer-count free (no whole-cache de-shard like M3), so the projection is a plain
# per-class scale-up.
FULL_MODEL_LAYERS = 36
FULL_MODEL_SLIDING = 18
FULL_MODEL_FULL = 18


def cat(rel):
    if any(k in rel for k in MEM_KEYS):
        return "memory"
    if any(rel.endswith(k) for k in COMM_KEYS):
        return "comm"
    return "compute"


def collect(csv_path):
    """Run the parser over the CSV and return everything the report needs."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    byte_cols = P.io_byte_columns(header)
    usecols = P.BASE_COLS + [c for c in P.OPTIONAL_COLS if c in header]
    usecols += [c for c in P.HOST_MOVEMENT_COLS if c in header]
    usecols += sorted({c for cols in byte_cols.values() for c in cols.values()})

    acc = P.ZoneAccumulator()
    acc.collect_timeline = True
    acc.movement_cols_present = any(c in header for c in P.HOST_MOVEMENT_COLS)
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000, low_memory=False):
        for row in chunk.to_dict("records"):
            acc.feed(row, byte_cols)
    return acc, P.summarize(acc), P.aggregate_by_class(acc and P.summarize(acc))


def accounting(csv_path, sliding_ms, full_ms, n_layers):
    """Kernel vs per-op firmware on the busiest chip, and what it projects to for the full model.

    Inter-op gaps are deliberately excluded: under tracy the host cannot dispatch fast enough, so
    OP TO OP LATENCY measures instrumentation overhead rather than dispatch cost. Kernel and firmware
    are on-device measurements and are unaffected by host-side profiling.
    """
    marks = pd.read_csv(csv_path, usecols=["OP CODE", "OP TYPE"], low_memory=False)
    sp = marks["OP TYPE"] == "signpost"
    lo = marks.index[sp & (marks["OP CODE"] == f"{P.ZONE_START} {P.ROOT_ZONE}")]
    hi = marks.index[sp & (marks["OP CODE"] == f"{P.ZONE_END} {P.ROOT_ZONE}")]
    if not len(lo) or not len(hi):
        return None
    cols = ["OP CODE", "OP TYPE", "DEVICE ID", P.DURATION_COL, "DEVICE FW DURATION [ns]"]
    d = pd.read_csv(csv_path, usecols=cols, low_memory=False).iloc[int(lo[0]) : int(hi[-1])]
    d = d[d["OP TYPE"] == P.DEVICE_OP_TYPE]
    g = d.groupby("DEVICE ID").agg(kernel=(P.DURATION_COL, "sum"), fw=("DEVICE FW DURATION [ns]", "sum")) / 1e6
    dev = int(g.fw.idxmax())
    kernel, fw = float(g.loc[dev, "kernel"]), float(g.loc[dev, "fw"])
    mult = fw / kernel if kernel else 1.0

    kern_full = FULL_MODEL_SLIDING * sliding_ms + FULL_MODEL_FULL * full_ms
    return {
        "device": dev,
        "ops": int(len(d[d["DEVICE ID"] == dev])),
        "kernel_ms": round(kernel, 3),
        "fw_overhead_ms": round(fw - kernel, 3),
        "busy_ms": round(fw, 3),
        "fw_multiplier": round(mult, 4),
        "layers": n_layers,
        "proj_kernel_ms": round(kern_full, 1),
        "proj_busy_ms": round(kern_full * mult, 1),
    }


def text_report(byclass, acc, acct, csv_path):
    out = [f"\n{'='*94}", f"GPT-OSS prefill zone profile — {os.path.basename(csv_path)}", "=" * 94]
    for cls in ("sliding", "full"):
        rels = byclass.get(cls)
        if not rels:
            continue
        total = rels.get("(layer total)", {}).get("ms_per_layer", 0)
        n = rels.get("(layer total)", {}).get("layers", 0)
        out += [
            "",
            f"--- {cls.upper()} layer · {total:.3f} ms/layer · {n} layer(s) sampled ---",
            f"  {'zone':<34} {'ms/layer':>9} {'% layer':>8} {'ops':>5} {'MiB':>8} {'GB/s':>7} "
            f"{'DRAM%':>6} {'NOC%':>6}  kind",
            f"  {'-'*34} {'-'*9} {'-'*8} {'-'*5} {'-'*8} {'-'*7} {'-'*6} {'-'*6}  {'-'*6}",
        ]
        parents = parent_rels(rels)
        for rel, v in sorted(rels.items(), key=lambda kv: -kv[1]["ms_per_layer"]):
            if rel in parents and rel != "(layer total)":
                mark = "  (parent)"
            elif rel == "(layer total)":
                continue
            else:
                mark = ""
            pct = 100 * v["ms_per_layer"] / total if total else 0
            du = f"{v['dram_util']:.1f}" if v.get("dram_util") is not None else "-"
            nu = f"{v['noc_util']:.1f}" if v.get("noc_util") is not None else "-"
            out.append(
                f"  {rel:<34} {v['ms_per_layer']:>9.3f} {pct:>7.1f}% {v['ops_per_layer']:>5.0f} "
                f"{v['mib_per_layer']:>8.1f} {v['gbs_mean']:>7.0f} {du:>6} {nu:>6}  {cat(rel)}{mark}"
            )
    if not any(v.get("dram_util") is not None for rels in byclass.values() for v in rels.values()):
        out += [
            "",
            "  DRAM% / NOC% not measured: re-run the capture with NOC_TRACES=1 (needs tt-npe built and",
            "  on PYTHONPATH via its ENV_SETUP). Without it the columns stay '-'.",
        ]
    if acct:
        out += [
            "",
            f"--- device-busy accounting (device {acct['device']}, busiest of 32) ---",
            f"  kernel time                    {acct['kernel_ms']:>9.2f} ms",
            f"  per-op firmware                {acct['fw_overhead_ms']:>9.2f} ms   "
            f"({acct['fw_multiplier']:.3f}x multiplier)",
            f"  device busy, {acct['layers']} layers          {acct['busy_ms']:>9.2f} ms",
            f"  projected to {FULL_MODEL_LAYERS} layers        {acct['proj_busy_ms']:>9.1f} ms   "
            f"({FULL_MODEL_SLIDING} sliding + {FULL_MODEL_FULL} full)",
        ]

    # op detail for the heaviest zones
    def worst_ms(z):
        per = defaultdict(float)
        for by_dev in acc.op_detail[z].values():
            for dv, ns in by_dev.items():
                per[dv] += ns
        return max(per.values()) / 1e6 if per else 0.0

    out += ["", "--- ops inside the heaviest zones (worst chip) ---"]
    for z in sorted(acc.op_detail, key=worst_ms, reverse=True)[:10]:
        out.append(f"  {z:<44} {worst_ms(z):>8.2f} ms")
        for code, by_dev in sorted(acc.op_detail[z].items(), key=lambda kv: -max(kv[1].values()))[:4]:
            out.append(f"      {code:<50} {max(by_dev.values())/1e6:>8.3f} ms")
    return "\n".join(out)


def build_html(byclass, summary, acc, acct, csv_path):
    def leaves(cls):
        parents = parent_rels(byclass.get(cls, {}))
        return {k: v for k, v in byclass.get(cls, {}).items() if k not in parents}

    def rows(cls):
        return [
            [
                r,
                round(v["ms_per_layer"], 4),
                int(round(v["ops_per_layer"])),
                round(v["mib_per_layer"], 1),
                round(v["gbs_mean"]),
                cat(r),
                round(v["dram_util"], 1) if v.get("dram_util") is not None else None,
                round(v["noc_util"], 1) if v.get("noc_util") is not None else None,
            ]
            for r, v in sorted(leaves(cls).items(), key=lambda kv: -kv[1]["ms_per_layer"])
        ]

    # per-chip spread, widest first
    imb = []
    per_zone_dev = defaultdict(dict)
    for (zone, dv), st in acc.stats.items():
        per_zone_dev[zone][dv] = st["ns"] / 1e6
    # Same leaf rule as the tables: a zone counts here unless a descendant is present in this capture.
    all_rels = {"/".join(z.split("/")[2:]) for z in per_zone_dev if len(z.split("/")) >= 3}
    imb_parents = parent_rels(all_rels)
    for zone, per in per_zone_dev.items():
        parts = zone.split("/")
        if len(parts) < 3 or "/".join(parts[2:]) in imb_parents or len(per) < 8:
            continue
        lo, hi = min(per.values()), max(per.values())
        if lo <= 0:
            continue
        imb.append(
            {
                "layer": parts[1],
                "rel": "/".join(parts[2:]),
                "lo": lo,
                "hi": hi,
                "ratio": hi / lo,
                "worst": max(per, key=per.get),
                "devs": sorted(per),
                "vals": [round(per[d], 4) for d in sorted(per)],
            }
        )
    imb.sort(key=lambda r: -r["ratio"])

    def worst_ms(z):
        per = defaultdict(float)
        for by_dev in acc.op_detail[z].values():
            for dv, ns in by_dev.items():
                per[dv] += ns
        return max(per.values()) / 1e6 if per else 0.0

    opdetail = []
    for z in sorted(acc.op_detail, key=worst_ms, reverse=True)[:14]:
        ops = sorted(((c, max(bd.values()) / 1e6) for c, bd in acc.op_detail[z].items()), key=lambda kv: -kv[1])
        opdetail.append(
            {
                "zone": z,
                "ms": round(worst_ms(z), 4),
                "ops": [[c.replace("DeviceOperation", "").replace("Operation", ""), round(m, 4)] for c, m in ops[:6]],
            }
        )

    # Per-layer op-by-op panels, one device (the busiest), ops in execution order. Kernel durations
    # only — idle between ops is not measurable under tracy, see the note rendered with the panels.
    tl_dev = acct["device"] if acct else (acc.timeline[0]["dev"] if acc.timeline else None)
    bylayer = defaultdict(list)
    for r in acc.timeline:
        if r["dev"] != tl_dev:
            continue
        parts = r["zone"].split("/")
        key = parts[1] if len(parts) > 1 and parts[1].startswith("layer") else "_pre"
        bylayer[key].append(
            {
                "i": len(bylayer[key]),
                "c": r["code"].replace("DeviceOperation", "").replace("Operation", ""),
                "z": "/".join(parts[2:]) or "(layer)",
                "ms": round(r["ns"] / 1e6, 5),
                "k": cat("/".join(parts[2:])),
            }
        )
    panels = [
        {"id": k, "ms": round(sum(o["ms"] for o in v), 4), "ops": v} for k, v in sorted(bylayer.items()) if k != "_pre"
    ]

    payload = json.dumps(
        {
            "panels": panels,
            "tlDevice": tl_dev,
            "hasUtil": any(v.get("dram_util") is not None for rels in byclass.values() for v in rels.values()),
            "sliding": rows("sliding"),
            "full": rows("full"),
            "slidingTotal": byclass.get("sliding", {}).get("(layer total)", {}).get("ms_per_layer", 0),
            "fullTotal": byclass.get("full", {}).get("(layer total)", {}).get("ms_per_layer", 0),
            "slidingLayers": byclass.get("sliding", {}).get("(layer total)", {}).get("layers", 0),
            "fullLayers": byclass.get("full", {}).get("(layer total)", {}).get("layers", 0),
            "imb": imb[:14],
            "opdetail": opdetail,
            "acct": acct,
            "movementClean": not acc.host_ops,
            "movementMeasured": acc.movement_cols_present,
            "src": os.path.basename(csv_path),
        },
        separators=(",", ":"),
    )
    return TEMPLATE.replace("__PAYLOAD__", payload)


TEMPLATE = r"""<title>GPT-OSS prefill zone profile</title>
<style>
:root{--ground:#f4f6f8;--surface:#fff;--raised:#eaeef2;--ink:#141a21;--ink-2:#48545f;--ink-3:#7c8894;
 --rule:#d3dae1;--compute:#2f6f9f;--comm:#b5742f;--memory:#3f8574;--hot:#a3413f;--cool:#4a7fb5;
 --shadow:0 1px 2px rgba(20,26,33,.06),0 4px 14px rgba(20,26,33,.05)}
@media (prefers-color-scheme:dark){:root{--ground:#0d1218;--surface:#141b23;--raised:#1c252f;--ink:#e6ecf2;
 --ink-2:#a3b0bd;--ink-3:#6f7d8b;--rule:#26313c;--compute:#5a9fd0;--comm:#d99a54;--memory:#5aab96;
 --hot:#d1706d;--cool:#6ba3d8;--shadow:none}}
:root[data-theme=dark]{--ground:#0d1218;--surface:#141b23;--raised:#1c252f;--ink:#e6ecf2;--ink-2:#a3b0bd;
 --ink-3:#6f7d8b;--rule:#26313c;--compute:#5a9fd0;--comm:#d99a54;--memory:#5aab96;--hot:#d1706d;
 --cool:#6ba3d8;--shadow:none}
:root[data-theme=light]{--ground:#f4f6f8;--surface:#fff;--raised:#eaeef2;--ink:#141a21;--ink-2:#48545f;
 --ink-3:#7c8894;--rule:#d3dae1;--compute:#2f6f9f;--comm:#b5742f;--memory:#3f8574;--hot:#a3413f;
 --cool:#4a7fb5;--shadow:0 1px 2px rgba(20,26,33,.06),0 4px 14px rgba(20,26,33,.05)}
*{box-sizing:border-box}
body{margin:0;padding:2.4rem 1.5rem 5rem;background:var(--ground);color:var(--ink);
 font:400 15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1140px;margin:0 auto;display:flex;flex-direction:column;gap:2.2rem}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace}
header{border-bottom:2px solid var(--ink);padding-bottom:1rem;display:flex;flex-direction:column;gap:.5rem}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;letter-spacing:.14em;
 text-transform:uppercase;color:var(--ink-3)}
h1{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:clamp(1.4rem,3vw,1.9rem);font-weight:600;
 letter-spacing:-.02em;margin:0;text-wrap:balance}
h2{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;
 color:var(--ink-3);margin:0 0 .8rem;font-weight:600}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:1.05rem;
 box-shadow:var(--shadow)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:1.1rem;
 box-shadow:var(--shadow);display:flex;flex-direction:column;gap:.75rem}
.card h3{margin:0;font-family:ui-monospace,Menlo,monospace;font-size:.75rem;letter-spacing:.1em;
 text-transform:uppercase;color:var(--ink-3);font-weight:600}
.big{font-family:ui-monospace,Menlo,monospace;font-size:2.4rem;line-height:1;font-weight:600;
 letter-spacing:-.03em}
.big small{font-size:.88rem;font-weight:400;color:var(--ink-3);letter-spacing:0;margin-left:.3rem}
.split{display:flex;height:11px;border-radius:2px;overflow:hidden;background:var(--raised)}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;font-size:.8rem;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.sw{width:10px;height:10px;border-radius:2px;flex:none}
.scroller{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:6px;
 box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;min-width:620px;font-size:.85rem}
th,td{padding:.38rem .7rem;text-align:right;white-space:nowrap}
th{font-family:ui-monospace,Menlo,monospace;font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;
 color:var(--ink-3);font-weight:600;border-bottom:1px solid var(--rule)}
td{border-bottom:1px solid color-mix(in srgb,var(--rule) 55%,transparent)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--raised)}
.zname{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:.81rem}
.barcell{width:28%;text-align:left}
.bar{height:9px;border-radius:2px;min-width:2px;display:block}
.chips{display:grid;grid-template-columns:repeat(32,1fr);gap:2px}
.chips i{display:block;height:19px;border-radius:2px}
.imbrow{display:flex;flex-direction:column;gap:.22rem;padding:.5rem 0;
 border-bottom:1px solid color-mix(in srgb,var(--rule) 55%,transparent)}
.imbrow:last-child{border-bottom:none}
.imbhead{display:flex;justify-content:space-between;gap:1rem;font-size:.83rem;align-items:baseline}
.imbhead .z{font-family:ui-monospace,Menlo,monospace}
.imbhead .r{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.acctrow{display:grid;grid-template-columns:1fr auto;gap:1rem;font-size:.9rem;padding:.12rem 0}
.acctbar{height:24px;border-radius:3px;display:flex;overflow:hidden;background:var(--raised);
 margin-bottom:.5rem}
.opz{padding:.5rem 0;border-bottom:1px solid color-mix(in srgb,var(--rule) 55%,transparent)}
.opz:last-child{border-bottom:none}
.opz .h{display:flex;justify-content:space-between;font-family:ui-monospace,Menlo,monospace;font-size:.82rem}
.opz .o{display:flex;justify-content:space-between;font-size:.79rem;color:var(--ink-2);
 padding-left:1.2rem;font-family:ui-monospace,Menlo,monospace}
.tag{font-family:ui-monospace,Menlo,monospace;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;
 padding:.18rem .45rem;border-radius:3px;font-weight:600}
.tag.ok{background:color-mix(in srgb,var(--memory) 18%,transparent);color:var(--memory)}
.tag.unk{background:color-mix(in srgb,var(--ink-3) 22%,transparent);color:var(--ink-2)}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--rule);
 border-radius:4px;padding:.4rem .6rem;font-family:ui-monospace,Menlo,monospace;font-size:.75rem;
 box-shadow:0 4px 18px rgba(0,0,0,.22);opacity:0;transition:opacity .1s;z-index:9;color:var(--ink)}
strong{color:var(--ink)}
code{font-family:ui-monospace,Menlo,monospace;font-size:.85em;background:var(--raised);padding:.1rem .3rem;
 border-radius:3px}
</style>
<div class="wrap">
<header>
  <div class="eyebrow" id="eyebrow"></div>
  <h1>GPT-OSS prefill — zone profile</h1>
</header>
<section><h2>Per-layer device-kernel time</h2><div class="cards" id="cards"></div></section>
<section id="acct-sec"><h2>Device-busy accounting</h2><div class="panel" id="acct"></div></section>
<section><h2>Per-chip spread — one cell per device, widest first</h2>
 <div class="panel"><div id="imb"></div>
 <div class="legend" style="margin-top:.7rem">
   <span><i class="sw" style="background:var(--cool)"></i>fastest chip</span>
   <span><i class="sw" style="background:var(--hot)"></i>slowest chip</span>
   <span style="color:var(--ink-3)">hover for device id and time</span></div></div></section>
<section id="utilnote" style="display:none"><div class="panel" style="border-left:3px solid var(--ink-3)">
 <b>DRAM / NOC utilization not measured in this capture.</b> Re-run with <code>NOC_TRACES=1</code> to add
 per-op <code>DRAM BW UTIL (%)</code> and <code>NOC UTIL (%)</code> columns — it needs tt-npe built and on
 <code>PYTHONPATH</code> (see its <code>ENV_SETUP</code>). Without it the profiler reports time and bytes
 moved, but cannot say whether a zone is bandwidth-bound or waiting.
</div></section>
<section><h2>Sliding-attention layer</h2><div class="scroller"><table id="t-sliding"></table></div></section>
<section><h2>Full-attention layer</h2><div class="scroller"><table id="t-full"></table></div></section>
<section><h2>Ops inside the heaviest zones</h2><div class="panel" id="opdetail"></div></section>
<section><h2>Host / device movement</h2><div class="panel" id="move"></div></section>
<section id="perlayer-sec"><h2>Op by op, one panel per layer</h2>
 <div class="panel" style="border-left:3px solid var(--hot);margin-bottom:1rem">
   <b>Kernel durations laid end to end — not wall-clock positions.</b> Idle time between ops is not
   measurable under tracy (the host cannot dispatch fast enough, so <code>OP TO OP LATENCY</code>
   reflects instrumentation). What these show exactly is which ops consume each layer's device time.
 </div>
 <div id="perlayer" style="display:flex;flex-direction:column;gap:1.2rem"></div></section>
</div>
<div class="tip" id="tip"></div>
<script>
const D=__PAYLOAD__;
const CAT={compute:"--compute",comm:"--comm",memory:"--memory"};
const cs=getComputedStyle(document.documentElement);
const col=k=>cs.getPropertyValue(CAT[k]).trim()||"#888";
const tip=document.getElementById("tip");
const showTip=(e,t)=>{tip.textContent=t;tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+"px";
  tip.style.top=Math.max(8,e.clientY-tip.offsetHeight-12)+"px";};
const hideTip=()=>tip.style.opacity=0;

document.getElementById("eyebrow").textContent=
  `${D.src} · ${D.slidingLayers} sliding + ${D.fullLayers} full layers sampled · 32 chips`;

function totals(rows){const t={compute:0,comm:0,memory:0};rows.forEach(r=>t[r[5]]+=r[1]);return t;}
document.getElementById("cards").innerHTML=[
 {n:"Sliding-attention layer",t:D.slidingTotal,r:D.sliding},{n:"Full-attention layer",t:D.fullTotal,r:D.full}
].filter(c=>c.r.length).map(c=>{
  const t=totals(c.r),s=t.compute+t.comm+t.memory;
  return `<div class="card"><h3>${c.n}</h3>
   <div class="big">${c.t.toFixed(2)}<small>ms / layer</small></div>
   <div class="split">${["compute","comm","memory"].filter(k=>t[k]>0).map(k=>
     `<i style="width:${(100*t[k]/s).toFixed(2)}%;background:${col(k)}"></i>`).join("")}</div>
   <div class="legend">${["compute","comm","memory"].filter(k=>t[k]>0).map(k=>
     `<span><i class="sw" style="background:${col(k)}"></i>${k==="comm"?"communication":k}
      <b class="mono">${(100*t[k]/s).toFixed(0)}%</b></span>`).join("")}</div></div>`;
}).join("");

const A=D.acct;
if(A){
 const segs=[["kernel",A.kernel_ms,"--compute"],["per-op firmware",A.fw_overhead_ms,"--memory"]];
 const tot=segs.reduce((a,s)=>a+s[1],0);
 document.getElementById("acct").innerHTML=
  `<div class="acctbar">${segs.map(s=>`<i style="width:${(100*s[1]/tot).toFixed(2)}%;
     background:${cs.getPropertyValue(s[2]).trim()}"></i>`).join("")}</div>`+
  segs.map(s=>`<div class="acctrow"><span><i class="sw" style="display:inline-block;
     background:${cs.getPropertyValue(s[2]).trim()}"></i> ${s[0]}</span>
     <span class="mono">${s[1].toFixed(2)} ms</span></div>`).join("")+
  `<div class="acctrow" style="border-top:1px solid var(--rule);margin-top:.35rem;padding-top:.35rem">
     <span><b>device busy · ${A.layers} layers · chip ${A.device}</b></span>
     <span class="mono"><b>${A.busy_ms.toFixed(2)} ms</b></span></div>
   <div class="acctrow"><span>firmware multiplier over kernel time</span>
     <span class="mono">${A.fw_multiplier.toFixed(3)}×</span></div>
   <div class="acctrow"><span>projected to 36 layers (18 sliding + 18 full)</span>
     <span class="mono"><b>${A.proj_busy_ms} ms</b></span></div>
   <p style="margin:.7rem 0 0;color:var(--ink-2);font-size:.88rem;max-width:80ch">
     Kernel and firmware are on-device measurements. Inter-op gaps are excluded: under tracy the host
     cannot dispatch fast enough, so <code>OP TO OP LATENCY</code> measures instrumentation, not
     dispatch. The projection scales each layer class to its full-model count and leaves every zone
     as measured.</p>`;
}else{document.getElementById("acct-sec").style.display="none";}

document.getElementById("imb").innerHTML=D.imb.map(r=>{
 const lo=Math.min(...r.vals),hi=Math.max(...r.vals);
 return `<div class="imbrow">
  <div class="imbhead"><span class="z">${r.layer} / ${r.rel}</span>
   <span class="r" style="color:${r.ratio>=2?"var(--hot)":"var(--ink-2)"}">${r.ratio.toFixed(2)}×
    <span style="color:var(--ink-3);font-weight:400">${r.lo.toFixed(3)}–${r.hi.toFixed(3)} ms</span></span></div>
  <div class="chips">${r.vals.map((v,i)=>{const t=hi>lo?(v-lo)/(hi-lo):0;
    return `<i data-t="dev ${r.devs[i]}: ${v.toFixed(3)} ms"
      style="background:color-mix(in srgb,var(--hot) ${(t*100).toFixed(0)}%,var(--cool))"></i>`;}).join("")}</div>
 </div>`;}).join("");
document.getElementById("imb").addEventListener("mousemove",e=>{
 const el=e.target.closest("i[data-t]");el?showTip(e,el.dataset.t):hideTip();});
document.getElementById("imb").addEventListener("mouseleave",hideTip);

// Utilization gets a saturation tint so a bandwidth-bound zone is visible without reading numbers.
function util(v){
  const t=Math.min(Math.max(v,0),100)/100;
  const c=t>0.75?"var(--hot)":t>0.4?"var(--comm)":"var(--ink-2)";
  return `<span style="color:${c};font-weight:${t>0.75?600:400}">${v.toFixed(1)}</span>`;
}
function table(id,rows){
 if(!rows.length){document.getElementById(id).closest("section").style.display="none";return;}
 const max=Math.max(...rows.map(r=>r[1]));
 document.getElementById(id).innerHTML=
  `<thead><tr><th class="zname">zone</th><th class="barcell"></th><th>ms</th><th>ops</th><th>MiB</th>
   <th>GB/s</th>${D.hasUtil?"<th>DRAM %</th><th>NOC %</th>":""}</tr></thead><tbody>`+
   rows.map(([z,ms,o,mib,g,k,du,nu])=>
   `<tr><td class="zname">${z}</td><td class="barcell"><span class="bar"
     style="width:${Math.max(2,100*ms/max).toFixed(1)}%;background:${col(k)}"></span></td>
    <td class="mono">${ms.toFixed(3)}</td><td class="mono">${o}</td>
    <td class="mono">${mib.toFixed(1)}</td><td class="mono">${g}</td>`+
    (D.hasUtil?`<td class="mono">${du==null?"—":util(du)}</td>
                <td class="mono">${nu==null?"—":util(nu)}</td>`:"")+
   `</tr>`).join("")+"</tbody>";
}
table("t-sliding",D.sliding);table("t-full",D.full);
if(!D.hasUtil){document.getElementById("utilnote").style.display="";}

document.getElementById("opdetail").innerHTML=D.opdetail.map(z=>
 `<div class="opz"><div class="h"><span>${z.zone}</span><b>${z.ms.toFixed(3)} ms</b></div>
  ${z.ops.map(([c,m])=>`<div class="o"><span>${c}</span><span>${m.toFixed(3)} ms</span></div>`).join("")}
 </div>`).join("");

if(D.panels&&D.panels.length){
 const host=document.getElementById("perlayer");
 host.innerHTML=D.panels.map((p,pi)=>{
  const nm=p.id.replace("layer","Layer ").replace("_sliding"," · sliding").replace("_full"," · full");
  return `<div><div style="display:flex;justify-content:space-between;align-items:baseline;
    font-family:ui-monospace,Menlo,monospace;font-size:.86rem;margin-bottom:.3rem">
    <b>${nm}</b><span style="color:var(--ink-2)">${p.ms.toFixed(3)} ms · ${p.ops.length} ops ·
      slowest ${Math.max(...p.ops.map(o=>o.ms)).toFixed(3)} ms</span></div>
   <div style="display:flex;height:22px;border-radius:3px;overflow:hidden;cursor:crosshair">
    ${p.ops.map((o,oi)=>`<i data-p="${pi}" data-o="${oi}" style="flex:${o.ms.toFixed(6)} 0 0;
      background:${col(o.k)};border-right:.5px solid var(--surface)"></i>`).join("")}</div>
   <canvas data-p="${pi}" style="width:100%;height:150px;display:block;cursor:crosshair"></canvas></div>`;
 }).join("");
 host.addEventListener("mousemove",e=>{const el=e.target.closest("i[data-o]");if(!el)return;
  const o=D.panels[+el.dataset.p].ops[+el.dataset.o];
  showTip(e,`#${o.i} ${o.c} · ${o.z} · ${o.ms.toFixed(4)} ms`);});
 host.addEventListener("mouseleave",hideTip);
 const geoms={};
 function drawPanels(){
  host.querySelectorAll("canvas").forEach(cv=>{
   const pi=+cv.dataset.p,p=D.panels[pi],ctx=cv.getContext("2d");
   const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
   cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
   const padL=46,padT=8,plotW=W-padL-6,plotH=H-14-padT;
   const maxMs=Math.max(...p.ops.map(o=>o.ms));
   const lo=Math.log10(0.002),hi=Math.log10(Math.max(maxMs*1.3,0.02));
   const y=v=>padT+plotH*(1-(Math.log10(Math.max(v,0.002))-lo)/(hi-lo));
   const bw=plotW/p.ops.length;
   ctx.font="10px ui-monospace,Menlo,monospace";ctx.textAlign="right";ctx.textBaseline="middle";
   [0.01,0.1,1].filter(g=>g<maxMs*1.3).forEach(g=>{
    ctx.strokeStyle=cs.getPropertyValue("--rule").trim();ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(padL,y(g)+.5);ctx.lineTo(W-6,y(g)+.5);ctx.stroke();
    ctx.fillStyle=cs.getPropertyValue("--ink-3").trim();ctx.fillText(g+" ms",padL-6,y(g));});
   ctx.textAlign="left";let prev=null;
   p.ops.forEach((o,i)=>{const g=o.z.split("/")[0];
    if(g!==prev){if(prev!==null){ctx.strokeStyle=cs.getPropertyValue("--ink-3").trim();
      ctx.globalAlpha=.3;ctx.beginPath();ctx.moveTo(padL+i*bw,padT);ctx.lineTo(padL+i*bw,padT+plotH);
      ctx.stroke();ctx.globalAlpha=1;}
     ctx.fillStyle=cs.getPropertyValue("--ink-3").trim();ctx.fillText(g,padL+i*bw+3,padT+7);prev=g;}});
   p.ops.forEach((o,i)=>{ctx.fillStyle=col(o.k);const yy=y(o.ms);
    ctx.fillRect(padL+i*bw,yy,Math.max(bw-0.8,1),padT+plotH-yy);});
   geoms[pi]={padL,bw,n:p.ops.length};});
 }
 drawPanels();addEventListener("resize",drawPanels);
 new MutationObserver(drawPanels).observe(document.documentElement,
   {attributes:true,attributeFilter:["data-theme"]});
 host.querySelectorAll("canvas").forEach(cv=>{
  cv.addEventListener("mousemove",e=>{const pi=+cv.dataset.p,g=geoms[pi];if(!g)return;
   const r=cv.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-g.padL)/g.bw);
   if(i<0||i>=g.n){hideTip();return;}
   const o=D.panels[pi].ops[i];showTip(e,`#${o.i} ${o.c} · ${o.z} · ${o.ms.toFixed(4)} ms`);});
  cv.addEventListener("mouseleave",hideTip);});
}else{document.getElementById("perlayer-sec").style.display="none";}

document.getElementById("move").innerHTML=
 `<div style="display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap;margin-bottom:.4rem">
   <span class="tag ${D.movementClean?"ok":"unk"}">${D.movementClean?"no host ops":"host ops present"}</span>
   <span>${D.movementClean?"Every op in the profiled chunk ran as a device kernel — no CPU fallbacks, no host ops."
    :"Some ops did not run as device kernels; see the text report."}</span></div>
  <div style="display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap">
   <span class="tag ${D.movementMeasured?"ok":"unk"}">${D.movementMeasured?"transfers measured":"transfers not measured"}</span>
   <span>${D.movementMeasured?"Buffer-transfer child calls were instrumented and none were recorded inside the chunk."
    :"This capture has no <code>*_TT_HOST_FUNC</code> columns, so H2D/D2H copies cannot be ruled out. Re-run with <code>--child-functions HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram</code>."}</span></div>`;
</script>
"""


def main():
    ap = argparse.ArgumentParser(description="Visualize a GPT-OSS prefill zone profile")
    ap.add_argument("csv", help="ops_perf_results_*.csv produced by run_prefill_profile.sh")
    ap.add_argument("-o", "--out", help="HTML output path (default: alongside the CSV)")
    ap.add_argument(
        "--open",
        dest="do_open",
        action="store_true",
        help="serve the report over HTTP and print the URL. Use this — the output is an HTML page, and "
        "opening the file in an editor shows you its source instead of the report.",
    )
    ap.add_argument("--port", type=int, default=8090, help="port for --open (default 8090)")
    ap.add_argument(
        "--bind",
        default="127.0.0.1",
        help="interface for --open. 127.0.0.1 (default) is reachable from your laptop through the "
        "editor's port forwarding or an SSH tunnel; 0.0.0.0 also lets colleagues open the URL directly.",
    )
    args = ap.parse_args()

    acc, summary, byclass = collect(args.csv)
    if not summary.get(P.ROOT_ZONE):
        sys.exit(
            f"ERROR: no `{P.ROOT_ZONE}` zone in {args.csv}.\n"
            "       Was GPTOSS_PROFILE_ZONES=1 set, and did the run reach the profiled chunk?"
        )
    sliding = byclass.get("sliding", {}).get("(layer total)", {}).get("ms_per_layer", 0)
    full = byclass.get("full", {}).get("(layer total)", {}).get("ms_per_layer", 0)
    n_layers = byclass.get("sliding", {}).get("(layer total)", {}).get("layers", 0) + byclass.get("full", {}).get(
        "(layer total)", {}
    ).get("layers", 0)
    acct = accounting(args.csv, sliding, full, n_layers)

    print(text_report(byclass, acc, acct, args.csv))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "zone_report.html")
    with open(out, "w") as f:
        f.write(build_html(byclass, summary, acc, acct, args.csv))
    print(f"\n[visualize] HTML report -> {out}")
    if args.do_open:
        serve(out, args.port, args.bind)
    else:
        print(f"[visualize] to view it:  {sys.argv[0]} <csv> --open")
    return 0


def serve(path, port, bind):
    """Serve the report over HTTP until interrupted, and print how to reach it.

    The report is a single self-contained file, so this serves the same bytes for every path — no
    directory listing, nothing else on the box exposed.
    """
    import http.server
    import socket

    body = open(path, "rb").read()
    name = os.path.basename(path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = None
    for p in range(port, port + 10):
        try:
            srv = http.server.ThreadingHTTPServer((bind, p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        print(f"[visualize] could not bind a port in {port}..{port+9}; pass --port")
        return

    host, user = socket.gethostname(), os.environ.get("USER", "you")
    print("")
    print("=" * 78)
    print(f"  REPORT: http://localhost:{port}/{name}")
    print("=" * 78)
    if bind == "127.0.0.1":
        print("  In VS Code / Cursor over SSH: a notification offers to open the forwarded port —")
        print(f"  accept it, or use the Ports panel to forward {port}. Otherwise, from your laptop:")
        print(f"      ssh -NL {port}:127.0.0.1:{port} {user}@{host}")
        print("  To let colleagues open it directly instead, re-run with --bind 0.0.0.0")
    else:
        print(f"  Reachable on the lab network — send colleagues:  http://{host}:{port}/{name}")
    print(f"  Ctrl-C to stop. The file itself is at {path} (self-contained; scp it anywhere).")
    print("")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[visualize] stopped")


if __name__ == "__main__":
    sys.exit(main())
