"""Rebuild the local analysis report from calculator outputs and workbook evidence.

Run from the repository root after generating the three *_validation/recomputed JSON files.
This script reads the implementation; it does not change or refit it.
"""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = ROOT / "tools/perf_calculator"
sys.path.insert(0, str(MODEL))
from routing import links
from noc_routing_model import predict


def read(p):
    return json.loads(p.read_text())


topology = read(MODEL / "topology_bh.json")
read_results = read(OUT / "read_validation.json")
write_results = read(OUT / "write_recomputed.json")
multi_results = read(OUT / "multicore_recomputed.json")
saved = read(MODEL / "write_validation_bh.json")
rows = []
for r in read_results:
    c = r["case"]
    training = c["cores"] == 110 and c["payload_bytes"] == 128 and c["mode"] != "split"
    rows.append(
        dict(
            **{k: c[k] for k in ["cores", "payload_bytes", "batch", "mode", "banks"]},
            traffic="read",
            estimated_us=r["prediction"]["estimated_us"],
            measured_us=r["measured_us"],
            error_percent=r["error_pct"],
            evidence="admission calibration" if training else "read comparison",
            bank_bound_us=r["prediction"]["bank_occupancy_lower_bound_us"],
            link_bound_us=r["prediction"]["link_occupancy_lower_bound_us"],
        )
    )
for group, results in [("write development", write_results), ("write holdout", multi_results)]:
    for r in results["cases"]:
        rows.append(
            dict(
                **r["case"],
                traffic="write",
                estimated_us=r["prediction"]["estimated_us"],
                measured_us=r["measured_us"],
                error_percent=r["error_percent"],
                evidence=group,
                bank_bound_us=r["prediction"]["bank_occupancy_lower_bound_us"],
                link_bound_us=r["prediction"]["link_occupancy_lower_bound_us"],
            )
        )
with (OUT / "comparisons.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

# Check regenerated predictions against the retained validation snapshot.
for old, new in zip(saved["single_core"]["cases"], write_results["cases"]):
    assert old["case"] == new["case"]
    assert abs(old["estimated_us"] - new["prediction"]["estimated_us"]) < 1e-10
for old in saved["multicore"]:
    new = next(r for r in multi_results["cases"] if all(old[k] == r["case"][k] for k in r["case"]))
    assert abs(old["estimated_us"] - new["prediction"]["estimated_us"]) < 1e-10
    assert abs(old["measured_us"] - new["measured_us"]) < 1e-10

# Read workbook cells and cached results directly, preserving the formula/cached distinction.
ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
sheets = {}
with zipfile.ZipFile(ROOT / "Data movement perf calculator.xlsx") as z:
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("s:sheets", ns):
        target = rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        cells = {}
        for c in ET.fromstring(z.read(target)).findall(".//s:sheetData/s:row/s:c", ns):
            v, formula, inline = c.find("s:v", ns), c.find("s:f", ns), c.find("s:is", ns)
            value = v.text if v is not None else None
            if c.attrib.get("t") == "s":
                value = ss[int(value)]
            if inline is not None:
                value = "".join(inline.itertext())
            cells[c.attrib["r"]] = dict(value=value, formula=formula.text if formula is not None else None)
        sheets[sheet.attrib["name"]] = cells
(OUT / "workbook_extract.json").write_text(json.dumps(sheets, indent=2) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f7f9fc",
        "axes.facecolor": "#f7f9fc",
        "axes.labelcolor": "#22334b",
        "text.color": "#22334b",
        "svg.fonttype": "none",
    }
)
BLUE, ORANGE, GREEN, RED = "#2667c9", "#d47420", "#188476", "#bb3c52"


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    svg = OUT / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


# Figure 1: measured accuracy, grouped by evidence rather than a misleading global average.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
groups = [
    ("Reads · 24 saved probes", [r for r in rows if r["traffic"] == "read"]),
    ("Writes · 24 development cases", [r for r in rows if r["evidence"] == "write development"]),
    ("Writes · 36 multicore holdouts", [r for r in rows if r["evidence"] == "write holdout"]),
]
for ax, (title, group) in zip(axes, groups):
    upper = max(max(r["measured_us"], r["estimated_us"]) for r in group) * 1.08
    ax.plot([0, upper], [0, upper], color="#7f8da0", lw=1, label="Exact agreement")
    ax.fill_between([0, upper], [0, 0.9 * upper], [0, 1.1 * upper], color="#dae2ee", alpha=0.6, label="±10%")
    for bank, color in [("balanced", BLUE), ("hotspot", ORANGE)]:
        g = [r for r in group if r["banks"] == bank and r["evidence"] != "admission calibration"]
        if g:
            ax.scatter([r["measured_us"] for r in g], [r["estimated_us"] for r in g], c=color, label=bank, s=32)
    g = [r for r in group if r["evidence"] == "admission calibration"]
    if g:
        ax.scatter(
            [r["measured_us"] for r in g],
            [r["estimated_us"] for r in g],
            marker="*",
            s=130,
            c=GREEN,
            label="Admission training",
        )
    ax.set(
        title=title, xlabel="Measured duration (µs)", ylabel="Estimated duration (µs)", xlim=(0, upper), ylim=(0, upper)
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)
fig.suptitle("Accuracy depends on the traffic regime", fontsize=18, fontweight="bold")
save(fig, "accuracy")

# Figure 2: core scaling isolates the balanced-write gap.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for ax, bank in zip(axes, ["balanced", "hotspot"]):
    for mode, color in [("noc0", BLUE), ("noc1", ORANGE), ("split", GREEN)]:
        g = sorted(
            [
                r
                for r in rows
                if r["evidence"] == "write holdout"
                and r["payload_bytes"] == 1024
                and r["banks"] == bank
                and r["mode"] == mode
            ],
            key=lambda r: r["cores"],
        )
        ax.plot([r["cores"] for r in g], [r["measured_us"] for r in g], "o-", c=color, label=mode + " measured")
        ax.plot([r["cores"] for r in g], [r["estimated_us"] for r in g], "x--", c=color, label=mode + " estimated")
    ax.set(
        title=bank.capitalize() + " banks",
        xlabel="Active cores",
        ylabel="Duration per 64 writes/core (µs)",
        xticks=[32, 96, 110],
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("1 KiB writes · batch 16 · 1350 MHz", fontsize=17, fontweight="bold")
save(fig, "write_scaling")

# Figure 3: reproduce the earlier workbook batching illustration, with computed and measured values separate.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
s = sheets["L1 copy algorithm"]
batch_data = []
for i in range(10, 18):
    batch_data.append(
        {
            "placement": s[f"A{i}"]["value"],
            "variant": s[f"B{i}"]["value"],
            "pages": int(s[f"C{i}"]["value"]),
            "batch": int(s[f"D{i}"]["value"]),
            "estimated_us": float(s[f"G{i}"]["value"]),
            "measured_us": float(s[f"H{i}"]["value"]),
        }
    )
for ax, placement in zip(axes, ["near", "far"]):
    g = [r for r in batch_data if r["placement"] == placement and r["pages"] == 32]
    g.sort(key=lambda r: r["batch"])
    ax.bar(np.arange(2) - 0.17, [r["estimated_us"] for r in g], 0.34, color=BLUE, label="Workbook formula")
    ax.bar(np.arange(2) + 0.17, [r["measured_us"] for r in g], 0.34, color=GREEN, label="Saved measurement")
    for j, r in enumerate(g):
        ax.text(j + 0.17, r["measured_us"] + 0.12, f'{r["measured_us"]:.3f}', ha="center", fontsize=9)
    ax.set(
        xticks=[0, 1],
        xticklabels=["1 page / barrier", "8 pages / barrier"],
        title=placement.capitalize() + " L1 source",
        ylabel="Duration per 64 KiB copy (µs)",
        ylim=(0, 10),
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
fig.suptitle("Batching amortizes completion latency · historical L1 example", fontsize=17, fontweight="bold")
save(fig, "batching")

# Figure 4: exact uncontended packet arithmetic and illustrative resource reservations.
fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout="constrained", gridspec_kw={"height_ratios": [1, 1.2]})
ax = axes[0]
segments = [
    (
        "Read",
        [(10 + 25 * 9, BLUE, "request 235"), (131, ORANGE, "bank delay 131"), (10 + 4 * 9 + 16, GREEN, "response 62")],
    ),
    (
        "Write",
        [
            (40, "#8290a5", "issue 40"),
            (10 + 25 * 9 + 16, BLUE, "payload request 251"),
            (22, ORANGE, "22"),
            (10 + 4 * 9, GREEN, "ACK 46"),
        ],
    ),
]
for y, (name, segs) in enumerate(segments):
    start = 0
    for width, color, label in segs:
        ax.barh(y, width, left=start, height=0.5, color=color)
        ax.text(start + width / 2, y, label, ha="center", va="center", fontsize=9, color="white")
        start += width
    ax.text(start + 5, y, f"{start} cycles = {start/1350:.3f} µs", va="center")
ax.set(
    yticks=[0, 1],
    yticklabels=["Read", "Write"],
    xlabel="Cycles",
    xlim=(0, 580),
    title="One 1 KiB packet · worker (1,2) ↔ bank 0 (0,11) · NoC0",
)
ax.invert_yaxis()
ax = axes[1]
for y, start in enumerate([5, 14, 23]):
    ax.broken_barh([(start, 17)], (y - 0.25, 0.5), facecolors=BLUE)
    ax.broken_barh([(start + 17, 17)], (y - 0.25, 0.5), facecolors=ORANGE)
    ax.text(start + 8.5, y, "A: 17", ha="center", va="center", color="white", fontsize=9)
    ax.text(start + 25.5, y, "B: 17", ha="center", va="center", color="white", fontsize=9)
ax.set(
    yticks=[0, 1, 2],
    yticklabels=["Link 1", "Link 2", "Link 3"],
    xlabel="Cycles",
    xlim=(0, 65),
    title="Illustrative two-packet stream: 17-cycle link reservation, 9-cycle hop propagation",
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.15)
fig.suptitle("Latency and resource occupancy are different quantities", fontsize=17, fontweight="bold")
fig.text(
    0.01,
    -0.03,
    "Top: exact model arithmetic with no queueing or read admission; write issue interval chosen as 40 cycles. Bottom: illustrative stream, not a captured trace.",
    fontsize=9,
)
save(fig, "packet_timing")
case = dict(cores=1, payload_bytes=1024, mode="noc0", banks="hotspot", batch=1, pages=1, iterations=1)
assert predict(topology, case, contention=False)["estimated_cycles"] == 428
assert (
    predict(topology, dict(case, traffic="write"), issue_cycles=(40, 40), contention=False)["estimated_cycles"] == 359
)

routes = {}
for core in [0, 31, 95, 109]:
    for noc in [0, 1]:
        for bank in range(8):
            a = topology["workers"][core]
            b = topology["banks" if noc == 0 else "banks_noc1"][bank]
            routes[f"{core}-{noc}-{bank}"] = {
                "src": a,
                "dst": b,
                "request": links(a, b, noc, topology["physical_grid"]),
                "response": links(b, a, noc, topology["physical_grid"]),
            }

manifest = {
    str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        *MODEL.glob("*.py"),
        *MODEL.glob("*.json"),
        *MODEL.glob("*.jsonl"),
        ROOT / "Data movement perf calculator.xlsx",
        ROOT / "tests/ttnn/unit_tests/operations/examples/test_noc_transport.py",
        Path("/tmp/noc-write-validation/phase2_measured.jsonl"),
    ]
    if p.exists()
}
(OUT / "source_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
payload = json.dumps(dict(rows=rows, routes=routes, topology=topology, batch=batch_data)).replace("</", "<\\/")

PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inside the data movement performance calculator</title>
<style>
:root{--ink:#21334b;--muted:#56667d;--blue:#2667c9;--green:#188476;--orange:#d47420;--line:#dce4ef}*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 28px 80px}h1{font-size:clamp(32px,5vw,54px);line-height:1.12;letter-spacing:-1.7px;max-width:900px;margin:18px 0}h2{font-size:28px;letter-spacing:-.6px;margin-top:0}h3{font-size:19px;margin:20px 0 8px}p{max-width:940px}a{color:var(--blue)}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:var(--blue);font-weight:750}.lede{font-size:20px;color:var(--muted)}section{padding:30px;background:white;border:1px solid var(--line);border-radius:16px;margin-top:24px}.cards,.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cols{grid-template-columns:1fr 1fr}.card{padding:18px;background:#f4f7fb;border-radius:12px}.card strong{display:block;font-size:30px}.small,small{font-size:13px;color:var(--muted)}.notice{border-left:4px solid var(--orange);padding:12px 18px;background:#fff6ec;margin:20px 0}.formula{padding:16px;background:#eef3fa;border-radius:10px;font:14px/1.8 ui-monospace,monospace;white-space:pre-wrap;overflow:auto}.flow{display:flex;gap:8px;align-items:stretch;margin:20px 0}.flow>div{flex:1;padding:16px 12px;background:#eef3fa;border-radius:10px;text-align:center;font-size:14px}.flow b{display:block;color:var(--blue)}.flow>span{align-self:center;color:var(--muted)}figure{margin:22px 0}figure img{width:100%;height:auto}figcaption{font-size:13px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f2f5fa}code{font-size:.9em}label{display:inline-flex;flex-direction:column;font-size:13px;gap:5px;margin:4px 16px 12px 0}select,input{font:inherit;padding:8px;border:1px solid #c7d2e1;border-radius:6px;background:white;color:var(--ink)}svg{max-width:100%;height:auto}.scroll{overflow:auto}.badge{border-radius:20px;background:#e7f2ef;padding:3px 9px;font-size:12px}details{margin:16px 0}summary{cursor:pointer;font-weight:650}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.toc{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;margin:24px 0}.negative{color:#a33748}.positive{color:#188476}@media(max-width:700px){main{padding:22px 12px}section{padding:20px 14px}.cards,.cols{grid-template-columns:1fr}.flow{flex-direction:column}.flow>span{transform:rotate(90deg)}h1{letter-spacing:-1px}}@media print{body{background:white}section{break-inside:avoid;border:0}main{max-width:none;padding:0}.toc{display:none}details{display:block}h1{font-size:34px}}
</style>
<main><div class="eyebrow">Source review · Blackhole snapshot · 23 October 2026</div>
<h1>How the data movement calculator predicts time</h1>
<p class="lede">It replays requests through directed NoC routes and shared resources, then takes the last completion time. Its useful foundation is a transport simulator; complete algorithm timing still needs a dependency and compute layer.</p>
<nav class="toc"><a href="#scope">Scope</a><a href="#engine">Event model</a><a href="#routes">Routes</a><a href="#batch">Batching</a><a href="#accuracy">Accuracy</a><a href="#review">Review findings</a><a href="#next">Algorithm model</a><a href="#sources">Evidence</a></nav>
<div class="cards"><div class="card"><span class="small">Single-core write development set</span><strong>1.35%</strong>median absolute prediction error · 24 cases</div><div class="card"><span class="small">Read probes, excluding 2 admission training rows</span><strong>5.66%</strong>median absolute prediction error · 22 cases</div><div class="card"><span class="small">Balanced multicore write holdout</span><strong>29.41%</strong>median absolute prediction error · 18 cases</div></div>
<p class="small">Recomputed from existing measurements; no new hardware benchmarks. Read comparisons reuse single-core issue calibration. These groups have different calibration and validation status and should not be pooled into one accuracy claim.</p>

<section id="scope"><h2>1. There are two generations of calculator here</h2>
<table><tr><th>Artifact</th><th>What it computes</th><th>Interpretation</th></tr>
<tr><td><code>Data movement perf calculator.xlsx</code></td><td>Serialized L1 transfer formulas, a batched L1-copy example, aggregate DRAM bounds, and historical queue snapshots.</td><td>Useful teaching and experiment record. Its queue predictions are saved values; changing spreadsheet parameters only recalculates the capacity bounds.</td></tr>
<tr><td><code>tools/perf_calculator/</code></td><td>DRAM read/write request simulation with explicit routes, issue spacing, batch completion, shared bank service, and optional measured read admission.</td><td>The current reusable implementation. It supports specific transport schedules, not arbitrary operation graphs.</td></tr></table>
<p>The captured configuration contains a <b>17 × 12 physical torus</b>, <b>110 usable workers</b> arranged as an 11 × 10 logical grid, eight allocator banks, and a 1350 MHz clock. Physical routing dimensions and usable worker dimensions serve different purposes.</p>
<p class="small">Workbook Sheet1 also retains a “120 Tensix cores” header. The current model uses the explicit 110-worker topology. Older workbook examples refer to source directories that are absent from this checkout; their embedded measurements remain inspectable but those harnesses are not reproducible from the referenced paths here.</p>
<div class="formula">Workbook serialized estimate (µs):
bytes / (effective_GBps × 1000) + (9 × round_trip_links + 4 × 5) / clock_MHz + residual_ns / 1000

Workbook batched L1-copy estimate (µs):
total_bytes / (effective_GBps × 1000) + ceil(pages / batch) × fixed_completion_latency_us</div>
<p>The workbook uses a measured 64 KiB effective transfer rate and a residual calibrated from a local 64-byte read. The Python model instead reserves individual packet resources. Those rates and residuals should not be stacked on top of its bank and link queues: that can count the same transport cost twice.</p></section>

<section id="engine"><h2>2. A discrete-event model, request by request</h2>
<div class="flow"><div><b>Describe the work</b>cores · pages · payload<br>batch · banks · NoC</div><span>→</span><div><b>Issue commands</b>per-core/per-NoC<br>measured spacing</div><span>→</span><div><b>Route packets</b>injection · links<br>ejection · queues</div><span>→</span><div><b>Serve the bank</b>read admission<br>shared byte service</div><span>→</span><div><b>Complete</b>response or ACK<br>next batch / finish</div></div>
<p>Events live in a time-ordered heap. For each resource, the simulator remembers when it is next free. A packet waits only when that resource is occupied. Different resources and independent cores can progress at the same time.</p>
<div class="formula">start = max(arrival, resource_free)
resource_free = start + occupancy
next_hop_arrival = start + propagation_delay

payload_packet_flits = 1 + ceil(payload_bytes / 64)
request_without_data_or_ACK_flits = 1
bank_occupancy = ceil(payload_bytes / 47.0805) cycles

duration_us = max(core_and_NoC_completion_cycles) / iterations / clock_MHz</div>
<p>Each NoC injection, directed link, and ejection resource is reserved for the packet's flit count, at one flit per cycle. Propagation adds 5 cycles at injection, 9 per link, and 5 at ejection. The packet tail adds <code>flits − 1</code> cycles once at the destination. Consecutive links overlap: multiplying the full packet serialization cost by the number of hops would be wrong.</p>
<table><tr><th>Step</th><th>Read</th><th>Non-posted write</th></tr><tr><td>Outbound packet</td><td>1-flit request</td><td>1 header + payload flits</td></tr><tr><td>Optional admission</td><td>Per-(NoC, bank), ~17.65 / ~16.85 cycles between requests</td><td>No characterized write admission</td></tr><tr><td>Shared bank reservation</td><td colspan="2">ceil(payload / 47.0805) cycles, shared between both NoCs and traffic directions</td></tr><tr><td>Return launch</td><td>131 cycles after bank service starts for 1 KiB; general delay = 126 + max(0, occupancy − payload_packet_flits)</td><td>22 cycles after bank service starts, even if the bank reservation is longer</td></tr><tr><td>Inbound packet</td><td>Header + data; completion at payload tail</td><td>1-flit ACK; completion at ACK arrival</td></tr></table>
<figure><img src="packet_timing.svg" alt="Read and write packet timing, followed by overlapping link reservations"><figcaption>Worked example checked against predict(): 428 read cycles and 359 write cycles. No queueing, no read-admission profile, and an illustrative 40-cycle write issue cost. Bank delay and occupancy are separate; ACK does not assert physical DRAM persistence.</figcaption></figure>
<p>A batch advances only after all its requests complete. In <code>split</code> mode each NoC owns half the batch and advances independently; there is no join between the two halves after each batch. All simulated cores start at time zero. Read request j starts at <code>j × issue_interval</code>; write request j starts at <code>(j + 1) × issue_interval</code>. That deliberate timing-origin asymmetry is part of the current implementation.</p>
<div class="notice"><b>Mixed does not mean copy.</b> Mixed mode runs independent reads on NoC0 and writes on NoC1. It does not require a write to wait for its source read, share a produced buffer, or wait for compute.</div>
<h3>Where the numbers come from</h3><table><tr><th>Input</th><th>Provenance in this checkout</th><th>Limit</th></tr><tr><td>9/5/5 cycles, 64-byte flits</td><td>Nominal architecture timings as recorded in local documentation and code</td><td>Not a measured arbitration or congestion model</td></tr><tr><td>126 read cycles, 22 write cycles, 47.0805 B/cycle</td><td>Retained measurement-derived simulator costs</td><td>Write delay and reused bank occupancy are provisional</td></tr><tr><td>Read admission intervals</td><td>Two 110-core, balanced, 128-byte probes: cycles / (110 × 64 / 8)</td><td>Physical resource unresolved; residual routing/barrier effects can be absorbed into the interval</td></tr><tr><td>Issue intervals</td><td>Single-core single-NoC ISSUE scopes; write CLI uses batch-16 cases</td><td>Effective interval can include backpressure and instrumentation</td></tr></table></section>

<section id="routes"><h2>3. Placement changes the route and the shared links</h2>
<p>NoC0 routes in increasing X, then increasing Y. NoC1 routes in decreasing Y, then decreasing X. Both wrap around the physical torus. The return packet uses the same NoC's routing rules, so it generally does not retrace the request path.</p>
<label>Worker index<select id="core"><option>0</option><option>31</option><option>95</option><option>109</option></select></label><label>NoC<select id="noc"><option value="0">NoC0 · +X then +Y</option><option value="1">NoC1 · −Y then −X</option></select></label><label>Allocator bank<select id="bank">__BANK_OPTIONS__</select></label>
<div class="legend"><span><i class="dot" style="background:#2667c9"></i>Request</span><span><i class="dot" style="background:#d47420"></i>Response / ACK</span><span>● Worker · ■ selected bank endpoint · dotted edges are torus wraps</span></div>
<div id="route"></div><div id="routeStats" class="formula"></div>
<p class="small">Routes come directly from routing.links() and topology_bh.json. Worker indexing is row-major. The bank endpoint can differ between NoCs; both still reserve the same logical bank resource. Arrow offsets separate request and return paths visually. Hover a link for its physical endpoints.</p>
<p>Balanced mapping uses <code>page % 8</code>; hotspot mapping targets bank 0. With the default 64 pages/core, every core begins at the same bank phase. Therefore balanced total bytes can still create bursts toward the same bank or link. Splitting traffic can reduce link pressure, but it does not double a bank's capacity.</p></section>

<section id="batch"><h2>4. Batching hides repeated completion latency</h2>
<p>For the historical 64 KiB far-source L1 copy, changing from one page per barrier to eight reduces the saved measured duration from <b>8.589 µs to 1.802 µs</b>, approximately <b>4.77× faster</b>. The bytes are unchanged; fewer serial completion waits make the difference.</p>
<figure><img src="batching.svg" alt="Measured and estimated near and far L1 copy duration for batch sizes 1 and 8"><figcaption>Workbook “L1 copy algorithm”, 32 pages × 2048 bytes. These are historical L1-copy results, not predictions from the current DRAM simulator.</figcaption></figure>
<label>Illustrative batch size <input id="batchInput" type="range" min="1" max="32" value="8"></label><div id="batchFormula" class="formula"></div>
<p class="small">The slider recomputes the workbook formula for the far-source 32-page copy. Only batch 1 and 8 have measurements in this view. Other settings are illustrative extrapolations. The formula omits issue limits and contention; the current Python simulator also rejects partial batches.</p></section>

<section id="accuracy"><h2>5. What the measured evidence supports</h2>
<figure><img src="accuracy.svg" alt="Estimated versus measured duration for reads, single-core writes, and multicore writes"><figcaption>Below the diagonal means underestimated duration. The gray band is ±10%, not a statistical confidence interval. Read admission training points are marked separately.</figcaption></figure>
<table><tr><th>Evidence group</th><th>Cases</th><th>Median absolute error</th><th>Maximum absolute error</th></tr><tr><td>Read comparisons excluding two admission training rows</td><td>22</td><td>5.66%</td><td>16.72%</td></tr><tr><td>Single-core writes, development data</td><td>24</td><td>1.35%</td><td>3.65%</td></tr><tr><td>Balanced multicore writes, holdout</td><td>18</td><td>29.41%</td><td>36.16%</td></tr><tr><td>Hotspot multicore writes, holdout</td><td>18</td><td>1.93%</td><td>9.65%</td></tr></table>
<p>The read maximum is the 32-core, 1 KiB split case: 4.918 µs estimated versus 5.905 µs measured. The read set covers balanced traffic only, two payload sizes, batch 16, and row-major core placement. Excluding the two admission-training rows does not make the remaining cases entirely independent: some supplied issue calibration.</p>
<figure><img src="write_scaling.svg" alt="Write duration versus active cores for balanced and hotspot mappings"><figcaption>In the 110-core 1 KiB hotspot, all three modes are within about 1%. In balanced traffic the same model is 29.8–34.1% too fast. The bank-limited regime works much better than the distributed transport regime.</figcaption></figure>
<h3>Explore all 84 recomputed comparisons</h3>
<label>Traffic<select id="traffic"><option value="write">Write</option><option value="read">Read</option><option value="all">Both</option></select></label><label>Payload<select id="payload"><option value="1024">1024 bytes</option><option value="128">128 bytes</option><option value="64">64 bytes</option><option value="4096">4096 bytes</option><option value="all">All</option></select></label><label>Banks<select id="banks"><option value="all">Both</option><option value="balanced">Balanced</option><option value="hotspot">Hotspot</option></select></label>
<div id="selectionStats" class="small"></div><div class="scroll"><table><thead><tr><th>Cores</th><th>NoC</th><th>Bytes / batch</th><th>Banks</th><th>Estimate µs</th><th>Measured µs</th><th>Error</th><th>Evidence</th></tr></thead><tbody id="results"></tbody></table></div>
<h3>Timing and validation boundaries</h3><p>The generic probe measures each core/RISC locally, divides its four-iteration TRANSPORT span by four, and takes the maximum local duration; trials are then summarized by the median. There is no synchronized cross-core start. The simulator instead starts all cores at time zero. These are related metrics, but they are not an independently established global operation makespan.</p>
<p>The generic harness checks the entire output exactly, including untouched hotspot pages. Pure-write source data is initialized in L1 outside timing. For read/mixed probes, post-scope verification writes can overlap slower workers and disturb their measured traffic. The older read-probe source harness is not retained alongside its 24 saved measurements.</p>
<p>The single-core write set informed the first-command issue-time correction, so its excellent agreement is development evidence. The saved multicore predictions were frozen before measurement. Their source hashes still match this checkout and the available raw holdout file, and replay reproduced all 36 saved predictions and measured values.</p></section>

<section id="review"><h2>6. Findings from the implementation review</h2>
<table><tr><th>Finding</th><th>Consequence</th><th>Recommended next step</th></tr>
<tr><td><b>Missing operation dependencies</b></td><td>No compute, circular-buffer capacity, producer/consumer relationships, semaphores, or output-stage dependencies. Summing or maximizing whole read/write predictions cannot generally model a kernel.</td><td>Build an operation event graph that invokes transport events when inputs and buffers are ready.</td></tr>
<tr><td><b>Unlimited downstream queues</b></td><td>A downstream stall never propagates upstream through finite credits. VCs, arbitration rules and outstanding-command limits are absent. This is a plausible family of explanations for balanced-write error, not an identified cause.</td><td>Measure one mechanism at a time with route, batch, issuer, and outstanding-depth sweeps; add independently supported resource constraints.</td></tr>
<tr><td><b>Parameters JSON is not loaded</b></td><td>9, 5, 64, 126, 22 and 47.0805 are hard-coded in noc_routing_model.py. Editing model_parameters.json does not change predictions. Its conversion sensitivity values are inactive.</td><td>Use one explicit parameter object, validate it, and include its hash in every estimate.</td></tr>
<tr><td><b>Read admission is an effective fit</b></td><td>Aggregate saturation time is assigned to a per-(NoC, bank) resource and reused across payload sizes. Physical placement and payload independence are hypotheses; adding explicit effects later could double count them.</td><td>Hold out payload, placement, bank mapping, batch and core-count families; refit only with declared training sets.</td></tr>
<tr><td><b>Issue and barrier approximations</b></td><td>First-read issue cost is omitted, write issue is charged, and barrier/loop instructions are not explicit. A uniform measured issue interval can already include some congestion.</td><td>Separate no-backpressure issue cost, issue throttling, and batch bookkeeping with component probes.</td></tr>
<tr><td><b>Limited input validation</b></td><td>Logical-coordinate bounds, duplicates, integer shapes and topology consistency are incompletely guarded. Linear indexing can silently map out-of-range x coordinates to another row, and negative coordinates use Python negative indexing.</td><td>Reject invalid coordinates and unsupported topology/architecture combinations before constructing routes.</td></tr>
<tr><td><b>Diagnostic gaps</b></td><td>waits is collected but not returned; only eight hottest links are exposed. No per-request timeline or per-core breakdown. Resource loads do not add up to elapsed time because resources overlap.</td><td>Expose trace events, per-resource wait/occupancy and critical-path attribution; keep capacity bounds separate from runtime.</td></tr>
<tr><td><b>Completion differs from occupancy drain</b></td><td>A write ACK starts 22 cycles after service begins; bank occupancy may continue after that. “bank_occupancy_lower_bound_us” is a capacity diagnostic, not an unconditional finite-workload bound on the ACK endpoint.</td><td>Specify completion endpoints and distinguish steady-state throughput bounds from final resource-drain time.</td></tr>
<tr><td><b>Deterministic ordering is not arbitration</b></td><td>Equal-time heap events are ordered by insertion serial, influenced by core/request launch order. The policy is reproducible but is not a characterized hardware arbiter.</td><td>Measure sensitivity to core/page ordering and characterize arbitration where it matters.</td></tr>
<tr><td><b>Coverage is narrow</b></td><td>One packet per request (1–16384 B), complete batches only, balanced or bank-0 hotspot, fixed placement semantics. No multicast, posted writes, dependent copy, L1 transport model, or validated mixed turnaround.</td><td>Declare supported regimes and require new validation for each added schedule family.</td></tr></table>
<p><code>shortlist.py</code> correctly deduplicates full configuration identities, keeps distinct schedules with equal estimates, and preserves supplied provenance. It does not generate workloads, implement kernels, validate correctness, or establish that the predicted ranking is correct. The caller must include all executable placement and scheduling choices.</p>
<p class="small">The 18 existing CPU tests passed. They check useful invariants: bytes, routing concentration, shared bank capacity, read-profile reproducibility, write completion semantics, and shortlist behavior. Passing these tests does not establish hardware prediction accuracy or cover the missing scheduling mechanisms above.</p></section>

<section id="next"><h2>7. Extending transport into an algorithm duration</h2>
<p>Represent the operation as events with explicit dependencies and finite resources. Each read, compute step, and write becomes ready only when its prerequisites are met. The scheduler then naturally models overlap, fill, drain, and backpressure.</p>
<div class="flow"><div><b>Read tile k</b>DRAM → L1<br>input buffer available</div><span>→</span><div><b>Compute tile k</b>read complete<br>compute engine available</div><span>→</span><div><b>Write tile k</b>output ready<br>writer available</div><span>→</span><div><b>Complete operation</b>required ACKs<br>final synchronization</div></div>
<div class="formula">Illustrative independent three-stage pipeline, n tiles:
serial time = n × (T_read + T_compute + T_write)
ideal overlapped time = T_read + T_compute + T_write
                       + (n − 1) × max(T_read, T_compute, T_write)</div>
<p>The ideal formula is a teaching bound under independent stage resources and sufficient buffering. Reads and writes here share banks and may share links, so their per-tile service times change when overlapped. For a real kernel, those activities must run in the same event simulation, with circular-buffer limits and dependencies.</p>
<ol><li><b>Make results reproducible:</b> unify runtime parameters, validate inputs, preserve raw measurements and hashes, and document exactly which completion is estimated.</li><li><b>Explain the balanced-write gap:</b> freeze predictions, sweep route overlap and outstanding depth, separate issue/flush/ACK phases, and test credit or VC hypotheses without fitting a universal slowdown multiplier.</li><li><b>Implement a minimal dependent copy:</b> read-complete → buffer-ready → write → ACK, with finite buffers. Validate serial and double-buffered variants.</li><li><b>Add measured compute stages:</b> then examine reader/compute/writer overlap and multiple core placements.</li><li><b>Validate ranking as well as duration:</b> compare the predicted shortlist with correct on-device implementations and measure the cost of selecting the wrong configuration.</li></ol>
<div class="notice">Practical use today: explain transport bottlenecks and prioritize experiments within measured regimes. General multicore algorithm duration still requires the missing scheduling layer and stronger validation of balanced write transport.</div></section>

<section id="sources"><h2>Evidence and reproducibility</h2><p>This review is grounded in the local source and saved measurements. Architecture constants are described as the implementation's inputs; no fresh hardware or external architecture validation was performed.</p>
<ul><li><a href="../../tools/perf_calculator/noc_routing_model.py">noc_routing_model.py</a>: predict(), claim(), flight(), request(), launch(), and validation CLI.</li><li><a href="../../tools/perf_calculator/routing.py">routing.py</a> and <a href="../../tools/perf_calculator/topology_bh.json">topology_bh.json</a>: routes and captured topology.</li><li><a href="../../tools/perf_calculator/read_service.py">read_service.py</a>, <a href="../../tools/perf_calculator/read_service_bh.json">read profile</a>, <a href="../../tools/perf_calculator/model_parameters.json">parameter record</a>.</li><li><a href="../../tools/perf_calculator/WRITE_MODEL.md">WRITE_MODEL.md</a>, <a href="../../tools/perf_calculator/README.md">calculator README</a>, <a href="../../tests/ttnn/unit_tests/operations/examples/test_noc_transport.py">generic hardware probe</a>.</li><li><a href="../../Data%20movement%20perf%20calculator.xlsx">Original workbook</a> · <a href="workbook_extract.json">extracted formulas and cached values</a>.</li><li><a href="read_validation.json">24 read replays</a> · <a href="write_recomputed.json">24 single-core write replays</a> · <a href="multicore_recomputed.json">36 multicore write replays</a> · <a href="comparisons.csv">all comparisons as CSV</a> · <a href="source_hashes.json">source SHA-256 manifest</a>.</li><li><a href="build_visuals.py">Report and figure generator</a>. Figures are available as SVG and PNG beside this report.</li></ul>
<details><summary>Commands to reproduce</summary><pre class="formula">python_env/bin/python -m unittest discover -s tools/perf_calculator -p 'test_*.py'
python_env/bin/python tools/perf_calculator/noc_routing_model.py \
  --topology tools/perf_calculator/topology_bh.json \
  --measurements tools/perf_calculator/read_probe_bh.jsonl \
  --read-service-profile tools/perf_calculator/read_service_bh.json \
  --output docs/perf_calculator_analysis/read_validation.json
python_env/bin/python tools/perf_calculator/validate_transport.py \
  --topology tools/perf_calculator/topology_bh.json \
  --measurements tools/perf_calculator/write_probe_bh.jsonl \
  --issue-measurements tools/perf_calculator/write_probe_bh.jsonl \
  --output docs/perf_calculator_analysis/write_recomputed.json
python_env/bin/python tools/perf_calculator/validate_transport.py \
  --topology tools/perf_calculator/topology_bh.json \
  --measurements /tmp/noc-write-validation/phase2_measured.jsonl \
  --issue-measurements tools/perf_calculator/write_probe_bh.jsonl \
  --output docs/perf_calculator_analysis/multicore_recomputed.json
python_env/bin/python docs/perf_calculator_analysis/build_visuals.py</pre><p class="small">The multicore raw file is available in this session under /tmp; preserve it for replay elsewhere. The saved write_validation_bh.json retains comparison values and its source hash. The visual report itself requires no network access.</p></details></section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
function renderRoute(){
 const core=+$('core').value,noc=+$('noc').value,bank=+$('bank').value,r=data.routes[`${core}-${noc}-${bank}`];
 const X=x=>50+x*50,Y=y=>35+y*34;
 let s='<svg viewBox="0 0 920 455" role="img" aria-label="Directed physical NoC request and return paths"><defs><marker id="req" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="#2667c9"/></marker><marker id="resp" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="#d47420"/></marker></defs>';
 for(let x=0;x<17;x++){s+=`<text x="${X(x)}" y="435" font-size="11" text-anchor="middle" fill="#687a91">${x}</text>`;for(let y=0;y<12;y++)s+=`<circle cx="${X(x)}" cy="${Y(y)}" r="3" fill="#d8e0eb"/>`;}
 for(let y=0;y<12;y++)s+=`<text x="15" y="${Y(y)+4}" font-size="11" fill="#687a91">${y}</text>`;
 for(const [kind,color,offset,marker] of [['request','#2667c9',-3,'req'],['response','#d47420',3,'resp']]){
  for(const edge of r[kind]){const [,n,x,y,u,v]=edge,a=X(x)+offset,b=Y(y)+offset,c=X(u)+offset,d=Y(v)+offset;
   const title=`${kind}: (${x},${y}) → (${u},${v})`;
   if(Math.abs(u-x)>1){let end=u<x?890:20,start=u<x?20:890;s+=`<path d="M${a},${b} L${end},${b} M${start},${d} L${c},${d}" fill="none" stroke="${color}" stroke-width="2.3" stroke-dasharray="4 3" marker-end="url(#${marker})"><title>${title} — torus wrap</title></path>`;}
   else if(Math.abs(v-y)>1){let end=v<y?426:8,start=v<y?8:426;s+=`<path d="M${a},${b} L${a},${end} M${c},${start} L${c},${d}" fill="none" stroke="${color}" stroke-width="2.3" stroke-dasharray="4 3" marker-end="url(#${marker})"><title>${title} — torus wrap</title></path>`;}
   else s+=`<path d="M${a},${b} L${c},${d}" fill="none" stroke="${color}" stroke-width="2.3" marker-end="url(#${marker})"><title>${title}</title></path>`;
  }
 }
 s+=`<circle cx="${X(r.src[0])}" cy="${Y(r.src[1])}" r="7" fill="#21334b"><title>Worker ${core}: ${r.src}</title></circle><rect x="${X(r.dst[0])-6}" y="${Y(r.dst[1])-6}" width="12" height="12" fill="#188476"><title>Bank ${bank}: ${r.dst}</title></rect></svg>`;
 $('route').innerHTML=s;let hops=r.request.length+r.response.length;
 $('routeStats').textContent=`Worker (${r.src}) → bank ${bank} (${r.dst})\nRequest: ${r.request.length} hops · return: ${r.response.length} hops · total: ${hops} hops\nRouter propagation: ${hops} × 9 / 1350 = ${(hops*9/1350).toFixed(3)} µs; four NIU crossings add ${(20/1350).toFixed(3)} µs.\nThese are propagation terms only: packet tails, issue, bank service and queueing are additional.`;
}
function renderTable(){
 const r=data.rows.filter(r=>($('traffic').value==='all'||r.traffic===$('traffic').value)&&($('payload').value==='all'||r.payload_bytes===+$('payload').value)&&($('banks').value==='all'||r.banks===$('banks').value));
 $('results').innerHTML=r.map(r=>`<tr><td>${r.cores}</td><td>${r.mode}</td><td>${r.payload_bytes} / ${r.batch}</td><td>${r.banks}</td><td>${r.estimated_us.toFixed(3)}</td><td>${r.measured_us.toFixed(3)}</td><td class="${r.error_percent<0?'negative':'positive'}">${r.error_percent.toFixed(2)}%</td><td>${r.traffic} · ${r.evidence}</td></tr>`).join('');
 $('selectionStats').textContent=`${r.length} cases shown. Error = (estimate / measured − 1) × 100%. All durations are per iteration, 64 requests/core.`;
}
function renderBatch(){let b=+$('batchInput').value,stream=65536/(82.072*1000),fixed=(29*9+20)/1350+32.72360666/1000,t=stream+Math.ceil(32/b)*fixed;$('batchFormula').textContent=`Batch ${b}: ${Math.ceil(32/b)} completion groups\n${stream.toFixed(3)} µs streaming + ${Math.ceil(32/b)} × ${fixed.toFixed(3)} µs fixed latency = ${t.toFixed(3)} µs\nFormula illustration only; no issue-rate or contention penalty.`;}
['core','noc','bank'].forEach(id=>$(id).addEventListener('change',renderRoute));['traffic','payload','banks'].forEach(id=>$(id).addEventListener('change',renderTable));$('batchInput').addEventListener('input',renderBatch);
renderRoute();renderTable();renderBatch();
</script></html>"""
PAGE = PAGE.replace("__BANK_OPTIONS__", "".join(f"<option>{b}</option>" for b in range(8))).replace("__DATA__", payload)
(OUT / "index.html").write_text(PAGE)
print(f"Created report, 4 SVG/PNG figures, and {len(rows)} comparison rows in {OUT}")
