#!/usr/bin/env python3
"""Build the self-contained KDA Tracy performance report.

Wall latency is the slowest device firmware span per replay. Graph timing
collapses each call position across devices (max for compute, mean for the
fused matmul/reduce-scatter collective). Both summarize replay sessions 2..11.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSVS = {
    "t640": Path(
        "/tmp/kda_perf_report_491dddde4fc_t640_r10/reports/2026_07_23_14_08_30/"
        "ops_perf_results_2026_07_23_14_08_30.csv"
    ),
    "t5120": Path(
        "/tmp/kda_perf_report_491dddde4fc_t5120_r10/reports/2026_07_23_14_09_48/"
        "ops_perf_results_2026_07_23_14_09_48.csv"
    ),
}
EXPECTED_SHA256 = {
    "t640": "30b5d9293e95a6c087025cbfa322eda4aabf03e11557bc50a8c2af9a88934cf1",
    "t5120": "a8762fd8d67bb78779f0689a0bda7caac20a567498dc3f8e96bfa99ccf96cba4",
}
EXPECTED_CALLS = {"t640": 30, "t5120": 35}
TOKENS = {"t640": 640, "t5120": 5120}
SESSION_IDS = list(range(2, 12))
DEVICE_IDS = list(range(8))
DEVICE_CLOCK_GHZ = 1.35
MRS_OP = "MatmulReduceScatterAsyncDeviceOperation"

BLOCKS = [
    {
        "id": "projection",
        "label": "fused input projection",
        "desc": "One TP-local matmul emits Q/K/V, decay rank, output gate, and beta.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [273, 278],
        "ranges": {"t640": [0, 1], "t5120": [0, 1]},
        "shape": "[1,T,2304] × [2304,2208] → [1,T,2208]",
        "distribution": "activations replicated; output columns grouped per TP rank",
    },
    {
        "id": "convolution",
        "label": "QKV causal convolution",
        "desc": "Carry concat, depthwise conv1d, layout conversion, and SiLU.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [163, 256],
        "ranges": {"t640": [1, 11], "t5120": [1, 16]},
        "shape": "[1,T,1536] + [1,3,1536] → [1,T,1536]",
        "distribution": "1536 TP-local channels; T=5120 uses two DRAM width slices",
    },
    {
        "id": "split",
        "label": "Q/K/V + auxiliary split",
        "desc": "Views the fused projection as Q, K, V, decay, output gate, and beta tensors.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [302, 321],
        "ranges": {"t640": [11, 17], "t5120": [16, 22]},
        "shape": "Q/K/V [1,T,512], decay [1,T,128], gate [1,T,512], beta [1,T,4]",
        "distribution": "four heads per chip; decay rank is replicated",
    },
    {
        "id": "decay",
        "label": "beta + decay gate",
        "desc": "Sigmoid beta, project decay rank, fused Softplus, scale, and promote to FP32.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [322, 353],
        "ranges": {"t640": [17, 20], "t5120": [22, 25]},
        "shape": "decay [1,T,128] → gate [1,T,512] FP32; beta [1,T,4]",
        "distribution": "TP-local four-head gate; rank-128 input replicated",
    },
    {
        "id": "layout",
        "label": "beta / recurrence layout",
        "desc": "Typecast, transpose, and reshape into the head-major chunk-kernel contract.",
        "file": "models/experimental/kimi_delta_attention/tt/recurrence.py",
        "lines": [115, 149],
        "ranges": {"t640": [20, 23], "t5120": [25, 28]},
        "shape": "beta [1,T,4] → head-major FP32 recurrence input",
        "distribution": "four local heads per chip",
    },
    {
        "id": "recurrence",
        "label": "chunk KDA recurrence",
        "desc": "Fused preparation and phased scan update the FP32 recurrent state.",
        "file": "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased.cpp",
        "lines": [69, 205],
        "ranges": {"t640": [23, 25], "t5120": [28, 30]},
        "shape": "Q/K/V/gate + state [1,4,128,128] → output + final state",
        "distribution": "prep/epilogue: 110 cores; scan: 16 cores, four V splits/head",
    },
    {
        "id": "epilogue",
        "label": "gated RMS epilogue",
        "desc": "Normalizes recurrence output and applies the SiLU output gate.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [364, 397],
        "ranges": {"t640": [25, 26], "t5120": [30, 31]},
        "shape": "[4,T,128] FP32 + gate [1,T,512] BF16 → [1,T,512] FP32",
        "distribution": "640 work items over 110 cores (90×6 + 20×5)",
    },
    {
        "id": "output",
        "label": "output projection + RS",
        "desc": "Fused row-parallel matmul overlaps an FP32 ring reduce-scatter, then clones its shared buffer.",
        "file": "models/demos/blackhole/qwen36/tt/tp_common.py",
        "lines": [434, 481],
        "ranges": {"t640": [26, 28], "t5120": [31, 33]},
        "shape": "[1,T,512] × [512,2304] → [1,T,288] per TP rank",
        "distribution": "8×8 matmul; two RS worker rows; ring TP8; two Ethernet links",
    },
    {
        "id": "state",
        "label": "state updates",
        "desc": "Copies recurrent and convolution carry outputs into persistent state.",
        "file": "models/experimental/kimi_delta_attention/tt/layer.py",
        "lines": [437, 445],
        "ranges": {"t640": [28, 30], "t5120": [33, 35]},
        "shape": "state [1,4,128,128] FP32; carry [1,3,1536] BF16",
        "distribution": "independent TP-local state",
    },
]

EDGES = [
    ["projection", "convolution", "QKV", "[1,T,1536]", "BF16", "TP-local channels"],
    ["projection", "split", "auxiliary projections", "[1,T,644]", "BF16", "mixed replicated/local"],
    ["convolution", "split", "activated QKV", "[1,T,1536]", "BF16", "four heads/chip"],
    ["split", "decay", "decay rank + beta", "[1,T,132]", "BF16", "rank replicated / beta local"],
    ["split", "layout", "Q/K/V", "3×[1,T,512]", "BF16", "four heads/chip"],
    ["decay", "layout", "gate + beta", "[1,T,516]", "FP32/BF16", "four heads/chip"],
    ["layout", "recurrence", "head-major inputs", "[4,T,128]", "FP32/BF16", "four heads/chip"],
    ["recurrence", "epilogue", "KDA output", "[4,T,128]", "FP32", "four heads/chip"],
    ["projection", "epilogue", "output gate", "[1,T,512]", "BF16", "four heads/chip"],
    ["epilogue", "output", "normalized output", "[1,T,512]", "FP32", "K-sharded"],
    ["output", "state", "layer output", "[1,T,288]", "FP32", "hidden TP8"],
    ["recurrence", "state", "final recurrent state", "[1,4,128,128]", "FP32", "TP-local"],
    ["convolution", "state", "new convolution carry", "[1,3,1536]", "BF16", "TP-local"],
]

COMMANDS = {
    scenario: (
        f"PERF_TRACE=1 PERF_SEQ={TOKENS[scenario]} PERF_REPS=10 "
        "python_env/bin/python3 -m tracy -p -r "
        f"-o /tmp/kda_perf_report_491dddde4fc_{scenario}_r10 "
        "--check-exit-code --op-support-count 10000 -t 5007 "
        '-a device_kernel_duration -m "pytest '
        'models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py -q -s"'
    )
    for scenario in TOKENS
}


@dataclass(frozen=True)
class Trace:
    csv: Path
    sha256: str
    rows: pd.DataFrame
    calls: list[dict[str, Any]]
    replay_spans_ns: list[float]
    replay_active_ns: list[float]
    block_timing_ns: dict[str, float]
    typical_device_ns: dict[str, float]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snippet(path: str, lines: list[int]) -> str:
    content = (REPO_ROOT / path).read_text().splitlines()
    start, end = lines
    return "\n".join(f"{number:4d}  {content[number - 1]}" for number in range(start, min(end, len(content)) + 1))


def short_op(op: str) -> str:
    for old, new in {"DeviceOperation": "", "ttnn::": "", "operations::": ""}.items():
        op = op.replace(old, new)
    return op


def parse_trace(scenario: str, csv: Path) -> Trace:
    sha256 = file_sha256(csv)
    if csv == DEFAULT_CSVS[scenario]:
        assert sha256 == EXPECTED_SHA256[scenario], (sha256, EXPECTED_SHA256[scenario])
    columns = [
        "OP CODE",
        "GLOBAL CALL COUNT",
        "DEVICE ID",
        "DEVICE ARCH",
        "DEVICE FW START CYCLE",
        "DEVICE FW END CYCLE",
        "DEVICE KERNEL DURATION [ns]",
        "METAL TRACE REPLAY SESSION ID",
        "CORE COUNT",
        "MATH FIDELITY",
        "PROGRAM HASH",
    ]
    rows = pd.read_csv(csv, usecols=columns)
    rows = rows[rows["METAL TRACE REPLAY SESSION ID"].isin(SESSION_IDS)].copy()
    rows["session"] = rows["METAL TRACE REPLAY SESSION ID"].astype(int)
    rows["device"] = rows["DEVICE ID"].astype(int)
    rows = rows.sort_values(["session", "device", "GLOBAL CALL COUNT"])
    rows["position"] = rows.groupby(["session", "device"]).cumcount()

    expected_calls = EXPECTED_CALLS[scenario]
    assert set(rows["session"]) == set(SESSION_IDS)
    assert set(rows["device"]) == set(DEVICE_IDS)
    assert rows.groupby(["session", "device"]).size().eq(expected_calls).all()
    assert not rows["DEVICE KERNEL DURATION [ns]"].isna().any()
    by_call = rows.groupby(["session", "position"])
    assert by_call["device"].nunique().eq(8).all()
    assert by_call["OP CODE"].nunique().eq(1).all()
    assert rows.groupby(["session", "device"])["OP CODE"].agg(tuple).nunique() == 1

    replay_spans_ns: list[float] = []
    replay_active_ns: list[float] = []
    collapsed_by_replay: dict[int, list[float]] = {}
    for session in SESSION_IDS:
        replay = rows[rows["session"] == session]
        device_spans = replay.groupby("device").apply(
            lambda group: (group["DEVICE FW END CYCLE"].max() - group["DEVICE FW START CYCLE"].min())
            / DEVICE_CLOCK_GHZ,
            include_groups=False,
        )
        replay_spans_ns.append(float(device_spans.max()))
        collapsed = []
        for position in range(expected_calls):
            call = replay[replay["position"] == position]
            durations = call["DEVICE KERNEL DURATION [ns]"]
            duration = durations.mean() if call["OP CODE"].iloc[0] == MRS_OP else durations.max()
            collapsed.append(float(duration))
        collapsed_by_replay[session] = collapsed
        replay_active_ns.append(sum(collapsed))

    calls: list[dict[str, Any]] = []
    canonical = rows[(rows["session"] == 2) & (rows["device"] == 0)]
    for position, (_, row) in enumerate(canonical.iterrows()):
        samples = pd.Series([collapsed_by_replay[session][position] for session in SESSION_IDS])
        calls.append(
            {
                "order": position + 1,
                "op": row["OP CODE"],
                "label": short_op(row["OP CODE"]),
                "duration_ns": float(samples.median()),
                "p10_ns": float(samples.quantile(0.1)),
                "p90_ns": float(samples.quantile(0.9)),
                "cores": int(row["CORE COUNT"]) if pd.notna(row["CORE COUNT"]) else None,
                "fidelity": row["MATH FIDELITY"] if pd.notna(row["MATH FIDELITY"]) else "n/a",
                "program_hash": str(row["PROGRAM HASH"]),
            }
        )

    block_timing_ns: dict[str, float] = {}
    for block in BLOCKS:
        start, end = block["ranges"][scenario]
        block_timing_ns[block["id"]] = sum(call["duration_ns"] for call in calls[start:end])
        for call in calls[start:end]:
            call["block"] = block["id"]
    assert len(calls) == sum(block["ranges"][scenario][1] - block["ranges"][scenario][0] for block in BLOCKS)
    assert all("block" in call for call in calls)
    assert round(sum(block_timing_ns.values())) == round(sum(call["duration_ns"] for call in calls))

    typical_device_ns = {}
    for needle in ("ChunkGatedDeltaRulePrep", "ChunkGatedDeltaRuleScan", "KdaGatedRmsNorm", MRS_OP):
        matching = rows[rows["OP CODE"].str.contains(needle, regex=False)]
        if not matching.empty:
            typical_device_ns[needle] = float(matching["DEVICE KERNEL DURATION [ns]"].median())
    return Trace(csv, sha256, rows, calls, replay_spans_ns, replay_active_ns, block_timing_ns, typical_device_ns)


def build_payload(traces: dict[str, Trace]) -> dict[str, Any]:
    blocks = []
    for block in BLOCKS:
        enriched = dict(block)
        enriched["snippet"] = source_snippet(block["file"], block["lines"])
        blocks.append(enriched)
    scenarios: dict[str, Any] = {}
    # ROOFLINE.md:321-323 establishes 67.594 GFLOP at T=640 for the
    # retained precomposed path; online work scales linearly with T.
    flops_per_token = 67_594_000_000 / 640
    peak_flops = 8 * 152.064e12
    for scenario, trace in traces.items():
        span = pd.Series(trace.replay_spans_ns)
        active = pd.Series(trace.replay_active_ns)
        tokens = TOKENS[scenario]
        executed_flops = tokens * flops_per_token
        latency_ns = float(span.median())
        mrs_ns = trace.typical_device_ns[MRS_OP]
        ccl_payload_bytes = tokens * 2304 * 4 * 7 / 8
        scenarios[scenario] = {
            "label": f"T={tokens:,}",
            "tokens": tokens,
            "latency_ns": latency_ns,
            "latency_min_ns": float(span.min()),
            "latency_max_ns": float(span.max()),
            "active_ns": float(active.median()),
            "graph_total_ns": sum(trace.block_timing_ns.values()),
            "replay_spans_ns": trace.replay_spans_ns,
            "replay_active_ns": trace.replay_active_ns,
            "calls": trace.calls,
            "block_timing_ns": trace.block_timing_ns,
            "typical_device_ns": trace.typical_device_ns,
            "compute_util": executed_flops / (latency_ns / 1e9) / peak_flops,
            "executed_flops": executed_flops,
            "ccl_payload_bytes": ccl_payload_bytes,
            "ccl_effective_gbps": ccl_payload_bytes / mrs_ns,
            "ccl_util": ccl_payload_bytes / mrs_ns / 100,
            "csv": str(trace.csv),
            "csv_sha256": trace.sha256,
            "csv_rows": len(trace.rows),
            "command": COMMANDS[scenario],
        }
    return {
        "title": "Kimi Linear KDA · performance dossier",
        "branch": "mvasilijevic/codex/kimi-linear-kda",
        "profile_commit": "491dddde4fc7c2c41f7cef9233c7ef8933578c77",
        "profile_tree_clean": True,
        "profile_date": "2026-07-23",
        "hardware": "LoudBox · 8× Blackhole · 1×8 mesh · FABRIC_1D",
        "software": "firmware 19.5.0 · KMD 2.4.1",
        "method": (
            "Median of ten measured Tracy replays (sessions 2–11). Wall time is the slowest-device "
            "firmware span. Graph calls use max duration across devices, except fused matmul/reduce-scatter "
            "uses the device mean. Session 1 is warm-up."
        ),
        "baseline": (
            "origin/main at merge-base 6a84cd727e2 contains no KDA layer, so no like-for-like mainline "
            "baseline exists. Historical branch traces are excluded from headline comparisons because "
            "they lack immutable run manifests."
        ),
        "blocks": blocks,
        "edges": EDGES,
        "scenarios": scenarios,
        "aspiration": {"compute_util": 0.60, "ccl_util": 0.40},
    }


def json_for_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def render_html(payload: dict[str, Any]) -> str:
    data = json_for_script(payload)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(payload["title"])}</title>
<style>
:root{{--paper:#f3efe5;--ink:#171714;--muted:#6b675e;--line:#c9c2b3;--accent:#d84825;--panel:#fffdf7;--cool:#1f6675}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
button{{font:inherit}}.wrap{{max-width:1500px;margin:auto;padding:30px}}header{{display:grid;grid-template-columns:1.35fr .65fr;gap:28px;border-top:7px solid var(--ink);padding-top:22px}}
.eyebrow,.kicker{{text-transform:uppercase;letter-spacing:.13em;font-size:11px;color:var(--accent);font-weight:800}}
h1{{font:800 clamp(38px,6vw,82px)/.92 Georgia,serif;margin:8px 0 16px;letter-spacing:-.045em}}.lede{{font:18px/1.45 Georgia,serif;max-width:760px}}
.meta{{border-left:1px solid var(--line);padding-left:22px}}.meta div{{padding:8px 0;border-bottom:1px solid var(--line)}}.meta b{{display:block;font-size:11px;text-transform:uppercase;color:var(--muted)}}
.controls{{position:sticky;top:0;z-index:9;background:rgba(243,239,229,.96);border-block:1px solid var(--ink);padding:10px 0;margin:28px 0;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}}
.seg{{display:flex;gap:4px}}.seg button{{border:1px solid var(--ink);background:transparent;padding:8px 12px;cursor:pointer}}.seg button[aria-pressed=true]{{background:var(--ink);color:var(--paper)}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}}.card{{background:var(--panel);padding:18px;min-height:145px}}
.card .value{{font:700 34px/1 Georgia,serif;margin:18px 0 7px}}.bar{{height:7px;background:#ded8ca;margin-top:14px}}.bar i{{display:block;height:100%;background:var(--accent)}}.bar.cool i{{background:var(--cool)}}
section{{margin-top:44px}}h2{{font:700 28px/1.1 Georgia,serif;border-bottom:2px solid var(--ink);padding-bottom:9px}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
.panel{{background:var(--panel);border:1px solid var(--line);padding:18px}}.spark{{width:100%;height:220px}}.spark polyline{{fill:none;stroke:var(--accent);stroke-width:3}}.spark circle{{fill:var(--panel);stroke:var(--ink);stroke-width:2}}.spark .axis{{stroke:var(--line);stroke-width:1}}.spark .grid{{stroke:#ded8ca;stroke-width:1;stroke-dasharray:3 4}}.spark .tick,.spark .axis-label{{fill:var(--muted);font-size:10px}}.spark .axis-label{{font-weight:800;letter-spacing:.06em;text-transform:uppercase}}.spark-point{{outline:none;cursor:crosshair}}.spark-point circle{{transition:r .12s,fill .12s}}.spark-point:hover circle,.spark-point:focus circle{{r:7px;fill:var(--accent)}}.spark-tip{{opacity:0;pointer-events:none;transition:opacity .12s}}.spark-tip rect{{fill:var(--ink);stroke:none}}.spark-tip text{{fill:var(--paper);font-size:10px;text-anchor:middle}}.spark-point:hover .spark-tip,.spark-point:focus .spark-tip{{opacity:1}}
.graph-shell{{position:relative;background:#141512;border:1px solid #141512;overflow:hidden;height:670px}}#graph{{width:100%;height:100%;cursor:grab}}#graph:active{{cursor:grabbing}}
.edge{{stroke:#817e76;stroke-width:1.4;fill:none}}.edge-label{{fill:#bdb8ad;font-size:10px}}.node rect{{fill:#f8f3e8;stroke:#171714;stroke-width:1.5}}.node text{{fill:#171714}}
.node .heat{{fill:var(--accent);stroke:none}}.node:hover rect{{stroke:var(--accent);stroke-width:3}}.opnode rect{{fill:#d8e6e7;stroke:#171714}}
.zoom{{position:absolute;right:12px;top:12px;display:flex;gap:4px}}.zoom button{{border:1px solid #eee8dc;background:#171714;color:#eee8dc;padding:7px 10px;cursor:pointer}}
.legend{{color:var(--muted);margin:10px 0 0}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}}
th{{position:sticky;top:0;z-index:2;background:var(--ink);color:var(--paper);cursor:pointer}}td.num,th.num{{text-align:right}}tr:hover td{{background:#f1e6d8}}.mono{{font-size:12px;word-break:break-all}}
.appendix details{{border-top:1px solid var(--line);padding:13px 0}}summary{{cursor:pointer;font-weight:800}}pre{{white-space:pre-wrap;background:#171714;color:#eee8dc;padding:15px;overflow:auto;font-size:11px}}
.call-grid{{max-height:460px;overflow:auto;border:1px solid var(--line)}}.drawer{{position:fixed;inset:0 0 0 auto;width:min(620px,94vw);background:var(--panel);z-index:20;border-left:3px solid var(--accent);padding:24px;overflow:auto;transform:translateX(105%);transition:transform .18s}}
.drawer.open{{transform:none}}.drawer-close{{float:right;border:1px solid var(--ink);background:none;padding:7px 10px;cursor:pointer}}.note{{padding:14px;border-left:5px solid var(--accent);background:#eee5d7}}
footer{{margin-top:55px;border-top:2px solid var(--ink);padding:18px 0;color:var(--muted)}}@media(max-width:900px){{header,.split{{grid-template-columns:1fr}}.cards{{grid-template-columns:1fr 1fr}}.meta{{border-left:0;padding-left:0}}}}
@media(max-width:560px){{.wrap{{padding:18px}}.cards{{grid-template-columns:1fr}}}}@media(prefers-reduced-motion:reduce){{.drawer{{transition:none}}}}
</style></head><body><main class="wrap">
<header><div><div class="eyebrow">Tracy-backed · source verified · TP8 Blackhole</div><h1>KDA<br>performance dossier</h1><p class="lede">A measured map of the Kimi Linear KDA layer: where time goes, how work is distributed, and how far the current implementation is from the 60% compute / 40% CCL aspiration.</p></div><div class="meta" id="meta"></div></header>
<div class="controls"><div class="seg" id="scenarioSeg"><button data-s="t640" aria-pressed="true">T=640</button><button data-s="t5120" aria-pressed="false">T=5,120</button></div><div class="seg" id="viewSeg"><button data-v="semantic" aria-pressed="true">semantic</button><button data-v="ops" aria-pressed="false">operations</button></div></div>
<section><div class="kicker">Measured outcome</div><div class="cards" id="cards"></div></section>
<section><h2>Replay stability and scaling</h2><div class="split"><div class="panel"><svg id="spark" class="spark" viewBox="0 0 680 220" role="img" aria-label="Measured wall latency by Tracy replay session"></svg><p class="legend">Ten measured replay spans. Axes show replay session and measured wall latency; hover or keyboard-focus a point for its exact value.</p></div><div class="panel" id="scaling"></div></div></section>
<section><h2>Source-verified dataflow</h2><p class="note">Node time is collapsed active kernel time, not wall time. Click a semantic node for its source proof; double-click to expand its exact Tracy calls. Drag to pan and use the controls to zoom.</p><div class="graph-shell"><svg id="graph" viewBox="0 0 1420 620"></svg><div class="zoom"><button data-z="in">+</button><button data-z="out">−</button><button data-z="reset">reset</button></div></div><p class="legend" id="graphLegend"></p></section>
<section><h2>Ordered operation ledger</h2><div class="call-grid"><table><thead><tr><th data-sort="order">#</th><th data-sort="label">Tracy operation</th><th data-sort="block">semantic block</th><th class="num" data-sort="duration_ns">median µs</th><th class="num" data-sort="p10_ns">p10</th><th class="num" data-sort="p90_ns">p90</th><th class="num" data-sort="cores">cores</th></tr></thead><tbody id="opRows"></tbody></table></div></section>
<section class="appendix"><h2>Evidence appendix</h2><details open><summary>Measurement provenance and method</summary><div id="provenance"></div></details><details><summary>Roofline / utilization interpretation</summary><div id="roofline"></div></details><details><summary>Full representative call tables</summary><div id="fullCalls"></div></details><details><summary>Source map</summary><div id="sources"></div></details></section>
<footer>Generated from immutable Tracy CSV hashes. No model weights are required: the perf harness uses deterministic random initialization.</footer></main>
<aside class="drawer" id="drawer"><button class="drawer-close" id="drawerClose">close</button><div id="drawerBody"></div></aside>
<script id="payload" type="application/json">{data}</script><script>
const P=JSON.parse(document.getElementById('payload').textContent);
let scenario='t640',view='semantic',sortKey='order',sortDir=1,scale=1,tx=0,ty=0,drag=null;
const expanded=new Set(),$=id=>document.getElementById(id),fmtNs=n=>(n/1000).toFixed(2)+' µs',pct=n=>(100*n).toFixed(1)+'%';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function sync(){{document.querySelectorAll('#scenarioSeg button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.s===scenario));document.querySelectorAll('#viewSeg button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.v===view));}}
function staticDraw(){{
 $('meta').innerHTML=`<div><b>profile commit</b>${{P.profile_commit.slice(0,12)}}</div><div><b>branch</b>${{esc(P.branch)}}</div><div><b>platform</b>${{esc(P.hardware)}}</div><div><b>software</b>${{esc(P.software)}}</div>`;
 $('sources').innerHTML=P.blocks.map(b=>`<details><summary>${{esc(b.label)}} · ${{esc(b.file)}}:${{b.lines[0]}}-${{b.lines[1]}}</summary><pre>${{esc(b.snippet)}}</pre></details>`).join('');
 $('fullCalls').innerHTML=Object.values(P.scenarios).map(s=>`<h3>${{s.label}}</h3><div class="call-grid"><table><thead><tr><th>#</th><th>op</th><th>block</th><th class="num">median µs</th><th>program hash</th></tr></thead><tbody>${{s.calls.map(c=>`<tr><td>${{c.order}}</td><td>${{esc(c.op)}}</td><td>${{esc(c.block)}}</td><td class="num">${{(c.duration_ns/1000).toFixed(3)}}</td><td class="mono">${{esc(c.program_hash)}}</td></tr>`).join('')}}</tbody></table></div>`).join('');
 $('provenance').innerHTML=Object.values(P.scenarios).map(s=>`<h3>${{s.label}}</h3><p><b>CSV</b> <span class="mono">${{esc(s.csv)}}</span><br><b>SHA-256</b> <span class="mono">${{s.csv_sha256}}</span><br><b>measured rows</b> ${{s.csv_rows.toLocaleString()}}</p><pre>${{esc(s.command)}}</pre>`).join('')+`<p><b>Profile tree:</b> commit ${{P.profile_commit}}, explicitly verified clean before both runs. The harness does not emit run_manifest.json, so the report records the equivalent command, commit, clean-tree assertion, CSV hash, hardware, and software metadata here.</p><p>${{esc(P.method)}}</p><p class="note">${{esc(P.baseline)}}</p>`;
}}
function drawCards(){{
 const s=P.scenarios[scenario],cards=[['wall latency',fmtNs(s.latency_ns),`${{fmtNs(s.latency_min_ns)}}–${{fmtNs(s.latency_max_ns)}} across ten replays`],['compute utilization',pct(s.compute_util),`executed ${{(s.executed_flops/1e9).toFixed(1)}} GFLOP / 8-chip HiFi4 peak`,s.compute_util/P.aspiration.compute_util,'cool'],['CCL utilization',pct(s.ccl_util),`${{s.ccl_effective_gbps.toFixed(1)}} GB/s effective payload / 100 GB/s target`,s.ccl_util/P.aspiration.ccl_util,''],['collapsed active',fmtNs(s.graph_total_ns),`${{s.calls.length}} calls · graph share denominator`]];
 $('cards').innerHTML=cards.map(c=>`<article class="card"><div class="kicker">${{c[0]}}</div><div class="value">${{c[1]}}</div><div>${{c[2]}}</div>${{c[3]!==undefined?`<div class="bar ${{c[4]||''}}"><i style="width:${{Math.min(100,c[3]*100)}}%"></i></div>`:''}}</article>`).join('');
}}
function drawSpark(){{
 const vals=P.scenarios[scenario].replay_spans_ns.map(v=>v/1000),rawLo=Math.min(...vals),rawHi=Math.max(...vals),pad=Math.max((rawHi-rawLo)*.15,.05),lo=rawLo-pad,hi=rawHi+pad,left=72,right=650,top=20,bottom=168,y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top),pts=vals.map((v,i)=>[left+i*(right-left)/(vals.length-1),y(v)]),ticks=[hi,(hi+lo)/2,lo];
 const yAxis=ticks.map(v=>`<line class="grid" x1="${{left}}" y1="${{y(v)}}" x2="${{right}}" y2="${{y(v)}}"/><text class="tick" x="${{left-8}}" y="${{y(v)+3}}" text-anchor="end">${{v.toFixed(2)}}</text>`).join('');
 const xAxis=pts.map((p,i)=>`<line class="axis" x1="${{p[0]}}" y1="${{bottom}}" x2="${{p[0]}}" y2="${{bottom+5}}"/><text class="tick" x="${{p[0]}}" y="${{bottom+18}}" text-anchor="middle">${{i+2}}</text>`).join('');
 const points=pts.map((p,i)=>{{const tipX=i===0?0:i===pts.length-1?-104:-52,tipY=p[1]<55?14:-36;return `<g class="spark-point" tabindex="0" role="img" aria-label="Replay ${{i+2}}, ${{vals[i].toFixed(3)}} microseconds"><circle cx="${{p[0]}}" cy="${{p[1]}}" r="5"/><g class="spark-tip" transform="translate(${{p[0]}},${{p[1]}})"><rect x="${{tipX}}" y="${{tipY}}" width="104" height="26" rx="2"/><text x="${{tipX+52}}" y="${{tipY+17}}">R${{i+2}} · ${{vals[i].toFixed(3)}} µs</text></g></g>`;}}).join('');
 $('spark').innerHTML=`${{yAxis}}<line class="axis" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{bottom}}"/><line class="axis" x1="${{left}}" y1="${{bottom}}" x2="${{right}}" y2="${{bottom}}"/>${{xAxis}}<text class="axis-label" x="${{(left+right)/2}}" y="208" text-anchor="middle">replay session</text><text class="axis-label" transform="translate(15 ${{(top+bottom)/2}}) rotate(-90)" text-anchor="middle">wall latency (µs)</text><polyline points="${{pts.map(p=>p.join(',')).join(' ')}}"/>${{points}}`;
 const a=P.scenarios.t640,b=P.scenarios.t5120;$('scaling').innerHTML=`<div class="kicker">8× token scaling</div><p style="font:700 42px/1 Georgia,serif">${{(b.latency_ns/a.latency_ns).toFixed(2)}}× latency</p><p>Work grows 8×; latency grows ${{(b.latency_ns/a.latency_ns).toFixed(2)}}×. Long-sequence throughput is ${{(b.tokens/(b.latency_ns/1e9)/1e6).toFixed(2)}} Mtok/s across the 8-chip layer.</p><p><b>Critical blocks at T=5,120:</b> ${{Object.entries(b.block_timing_ns).sort((x,y)=>y[1]-x[1]).slice(0,3).map(([id,n])=>P.blocks.find(z=>z.id===id).label+' '+fmtNs(n)).join(' · ')}}</p>`;
}}
function nodeMarkup(b,x,y,time,total){{const share=time/total,heat=Math.max(5,share*353);return `<g class="node" data-id="${{b.id}}" transform="translate(${{x}},${{y}})"><rect width="355" height="82"/><rect class="heat" x="1" y="75" width="${{heat}}" height="6"/><text x="12" y="22" font-weight="700">${{esc(b.label)}}</text><text x="12" y="42">${{fmtNs(time)}} · ${{pct(share)}}</text><text x="12" y="60" font-size="10">${{esc(b.distribution.slice(0,44))}}</text></g>`;}}
function drawGraph(){{
 const s=P.scenarios[scenario],svg=$('graph'),positions={{}};P.blocks.forEach((b,i)=>positions[b.id]={{x:45+(i%3)*455,y:40+Math.floor(i/3)*185}});let out=`<g id="viewport" transform="translate(${{tx}} ${{ty}}) scale(${{scale}})">`;
 P.edges.forEach(e=>{{const a=positions[e[0]],b=positions[e[1]];if(!a||!b)return;const x1=a.x+355,y1=a.y+43,x2=b.x,y2=b.y+43;out+=`<path class="edge" d="M${{x1}},${{y1}} C${{x1+50}},${{y1}} ${{x2-50}},${{y2}} ${{x2}},${{y2}}"/><text class="edge-label" x="${{(x1+x2)/2}}" y="${{(y1+y2)/2-5}}">${{esc(e[2])}}</text>`;}});
 if(view==='semantic'){{P.blocks.forEach(b=>{{const p=positions[b.id];out+=nodeMarkup(b,p.x,p.y,s.block_timing_ns[b.id],s.graph_total_ns);if(expanded.has(b.id))s.calls.filter(c=>c.block===b.id).forEach((c,j)=>{{out+=`<g class="opnode" transform="translate(${{p.x+18}},${{p.y+92+j*31}})"><rect width="320" height="26"/><text x="7" y="18">${{c.order}} · ${{esc(c.label.slice(0,31))}} · ${{fmtNs(c.duration_ns)}}</text></g>`;}});}});}}
 else{{s.calls.forEach((c,i)=>{{const col=i%5,row=Math.floor(i/5),x=28+col*275,y=25+row*76;out+=`<g class="opnode" transform="translate(${{x}},${{y}})"><rect width="248" height="55"/><text x="8" y="20">${{c.order}} · ${{esc(c.label.slice(0,28))}}</text><text x="8" y="40">${{fmtNs(c.duration_ns)}} · ${{esc(c.block)}}</text></g>`;}});}}
 svg.innerHTML=out+'</g>';svg.querySelectorAll('.node').forEach(n=>{{n.onclick=()=>openDrawer(n.dataset.id);n.ondblclick=()=>{{expanded.has(n.dataset.id)?expanded.delete(n.dataset.id):expanded.add(n.dataset.id);drawGraph();}};}});$('graphLegend').textContent=`${{s.label}} · node bars encode share of ${{fmtNs(s.graph_total_ns)}} collapsed active time · ${{view}} view`;
}}
function openDrawer(id){{const b=P.blocks.find(x=>x.id===id),s=P.scenarios[scenario],calls=s.calls.filter(c=>c.block===id);$('drawerBody').innerHTML=`<div class="kicker">${{fmtNs(s.block_timing_ns[id])}} · ${{pct(s.block_timing_ns[id]/s.graph_total_ns)}}</div><h2>${{esc(b.label)}}</h2><p>${{esc(b.desc)}}</p><p><b>tensor contract</b><br>${{esc(b.shape)}}</p><p><b>distribution</b><br>${{esc(b.distribution)}}</p><p><b>Tracy calls</b><br>${{calls.map(c=>c.order+' '+esc(c.label)+' '+fmtNs(c.duration_ns)).join('<br>')}}</p><p><b>${{esc(b.file)}}:${{b.lines[0]}}-${{b.lines[1]}}</b></p><pre>${{esc(b.snippet)}}</pre>`;$('drawer').classList.add('open');}}
function drawTable(){{const calls=[...P.scenarios[scenario].calls].sort((a,b)=>{{let x=a[sortKey]??0,y=b[sortKey]??0;return (typeof x==='string'?x.localeCompare(y):x-y)*sortDir;}});$('opRows').innerHTML=calls.map(c=>`<tr><td>${{c.order}}</td><td title="${{esc(c.op)}}">${{esc(c.label)}}</td><td>${{esc(c.block)}}</td><td class="num">${{(c.duration_ns/1000).toFixed(3)}}</td><td class="num">${{(c.p10_ns/1000).toFixed(3)}}</td><td class="num">${{(c.p90_ns/1000).toFixed(3)}}</td><td class="num">${{c.cores??'—'}}</td></tr>`).join('');}}
function drawRoofline(){{const s=P.scenarios[scenario],computeGap=P.aspiration.compute_util/s.compute_util,cclGap=P.aspiration.ccl_util/s.ccl_util;$('roofline').innerHTML=`<h3>${{s.label}}</h3><p>Executed-work estimate: <b>${{(s.executed_flops/1e9).toFixed(3)}} GFLOP</b>. Against the 8-chip HiFi4 peak of 1,216.512 TFLOP/s and measured wall latency, this is <b>${{pct(s.compute_util)}}</b> utilization. Reaching 60% requires about <b>${{computeGap.toFixed(1)}}×</b> the present useful-FLOP rate.</p><p>The fused output collective moves a minimum FP32 ring payload of <b>${{(s.ccl_payload_bytes/1e6).toFixed(2)}} MB</b>. Dividing by its typical-device Tracy median gives <b>${{s.ccl_effective_gbps.toFixed(1)}} GB/s</b>, or <b>${{pct(s.ccl_util)}}</b> of the 100 GB/s planning target; the 40% aspiration is ${{cclGap.toFixed(2)}}× away. This is a lower-bound payload model because Tracy exposes the fused kernel, not isolated link occupancy.</p><p>The source and trace together identify recurrence and fused output as the dominant optimization surfaces.</p>`;}}
function draw(){{sync();drawCards();drawSpark();drawGraph();drawTable();drawRoofline();}}
document.querySelectorAll('#scenarioSeg button').forEach(b=>b.onclick=()=>{{scenario=b.dataset.s;expanded.clear();scale=1;tx=0;ty=0;draw();}});document.querySelectorAll('#viewSeg button').forEach(b=>b.onclick=()=>{{view=b.dataset.v;expanded.clear();draw();}});
document.querySelectorAll('th[data-sort]').forEach(h=>h.onclick=()=>{{if(sortKey===h.dataset.sort)sortDir*=-1;else{{sortKey=h.dataset.sort;sortDir=1;}}drawTable();}});document.querySelectorAll('.zoom button').forEach(b=>b.onclick=()=>{{if(b.dataset.z==='in')scale*=1.18;else if(b.dataset.z==='out')scale/=1.18;else{{scale=1;tx=0;ty=0;}}drawGraph();}});
$('drawerClose').onclick=()=>$('drawer').classList.remove('open');const svg=$('graph');svg.onmousedown=e=>drag={{x:e.clientX,y:e.clientY,tx,ty}};window.onmousemove=e=>{{if(drag){{tx=drag.tx+e.clientX-drag.x;ty=drag.ty+e.clientY-drag.y;drawGraph();}}}};window.onmouseup=()=>drag=null;
staticDraw();draw();globalThis.__KDA_REPORT__={{P,draw,drawGraph,openDrawer,expanded,setScenario:s=>{{scenario=s;draw();}},setView:v=>{{view=v;draw();}},state:()=>({{scenario,view,scale,tx,ty}})}};
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t640-csv", type=Path, default=DEFAULT_CSVS["t640"])
    parser.add_argument("--t5120-csv", type=Path, default=DEFAULT_CSVS["t5120"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("kda_perf_report.html"))
    args = parser.parse_args()
    traces = {"t640": parse_trace("t640", args.t640_csv), "t5120": parse_trace("t5120", args.t5120_csv)}
    payload = build_payload(traces)
    args.output.write_text(render_html(payload) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    for scenario, result in payload["scenarios"].items():
        print(
            f"{scenario}: wall={result['latency_ns']/1000:.3f} us, active={result['graph_total_ns']/1000:.3f} us, compute={100*result['compute_util']:.2f}%, CCL={100*result['ccl_util']:.2f}%"
        )


if __name__ == "__main__":
    main()
