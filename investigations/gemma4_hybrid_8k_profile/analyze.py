# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attribute native program spans and export an inspectable device timeline."""

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [
    ("Global SDPA", "#d55d50"),
    ("Sliding-window SDPA", "#e7ab40"),
    ("Attention projection matmuls", "#447fbd"),
    ("MLP matmuls", "#5baec0"),
    ("TP collectives", "#9981bb"),
    ("KV updates", "#4b9a79"),
    ("Copies / layout / typecast", "#a88a65"),
    ("Other programs", "#8793a6"),
    ("Device gaps", "#d8dde5"),
]
MATMUL_NAMES = ["QKV projection", "Output projection", "MLP gate", "MLP up", "MLP down"]


def classify(sources, matmul_index):
    paths = " ".join(sources)
    if "ring_joint_sdpa.cpp" in paths:
        return int("halo" in paths)
    if "/matmul/" in paths:
        return 2 if matmul_index % 5 < 2 else 3
    if "update_cache.cpp" in paths or "update_padded_kv_cache.cpp" in paths:
        return 5
    if "/ccl/" in paths:
        return 4
    if "/data_movement/" in paths or "typecast" in paths or "nlp_" in paths:
        return 6
    return 7


def decode(report, row, capture, source_ids):
    profile = capture["profile"]
    chip = capture["chips"][str(profile["chip_id"])]
    programs = sorted(profile["programs"])
    assert programs[0][0] == chip["start"] and programs[-1][1] == chip["end"]
    assert len(capture["chips"]) == math.prod(report["mesh"])
    assert len({c["programs"] for c in capture["chips"].values()}) == 1
    assert len(programs) == chip["programs"]
    assert math.isclose(capture["device_us"], max(c["span_us"] for c in capture["chips"].values()))
    scale = chip["frequency_ghz"] * 1000
    categories = [0.0] * len(CATEGORIES)
    counts = [0] * len(CATEGORIES)
    events, attention = [], []
    sources = profile["kernel_sources"]
    matmul_index = sdpa_index = 0
    previous_end = chip["start"]
    request_count = len(row["request_lengths"])
    gaps_by_next_category = defaultdict(float)
    for start, end, source in programs:
        # Program spans are additive only when there is no overlap on this chip.
        assert end > start >= previous_end, "Overlapping programs need union-based attribution"
        paths = tuple(sorted(set(sources[str(source)])))
        sid = source_ids.setdefault(paths, len(source_ids))
        category = classify(paths, matmul_index)
        gap = (start - previous_end) / scale
        categories[8] += gap
        gaps_by_next_category[CATEGORIES[category][0]] += gap / 1000
        counts[8] += int(gap > 0)
        layer = request = phase = -1
        if category in (2, 3):
            layer, phase = divmod(matmul_index, 5)
            assert sdpa_index == (layer + (phase > 0)) * request_count
            matmul_index += 1
        elif category in (0, 1):
            layer, request = divmod(sdpa_index, request_count)
            assert matmul_index == layer * 5 + 1
            assert (category == 1) == (report["layer_types"][layer] == "sliding_attention")
            sdpa_index += 1
        duration = (end - start) / scale
        categories[category] += duration
        counts[category] += 1
        event = [round((start - chip["start"]) / scale, 3), round(duration, 3), category, sid, layer, request, phase]
        events.append(event)
        if category in (0, 1):
            attention.append(dict(layer=layer, request=request, category=category, ms=duration / 1000))
        previous_end = end
    assert matmul_index == report["layers"] * 5
    assert sdpa_index == report["layers"] * request_count
    assert math.isclose(sum(categories), capture["device_us"], abs_tol=1e-6)
    assert math.isclose(categories[8], chip["gap_us"], abs_tol=1e-6)
    per_request = []
    for request, tokens in enumerate(row["request_lengths"]):
        stats = dict(request=request, tokens=tokens)
        for category, name in ((0, "global"), (1, "sliding")):
            durations = [a["ms"] for a in attention if a["request"] == request and a["category"] == category]
            stats[name] = dict(count=len(durations), total_ms=sum(durations), mean_ms=sum(durations) / len(durations))
        per_request.append(stats)
    return dict(
        path=row["path"],
        prefix=row["prefix"],
        chip=profile["chip_id"],
        sample=capture["sample"],
        total_ms=capture["device_us"] / 1000,
        categories_ms=[v / 1000 for v in categories],
        counts=counts,
        gaps_by_next_category_ms=dict(gaps_by_next_category),
        per_request=per_request,
        events=events,
    )


def event_name(event):
    category, layer, request, phase = event[2], event[4], event[5], event[6]
    name = MATMUL_NAMES[phase] if phase >= 0 else CATEGORIES[category][0]
    if layer >= 0:
        name += f" | layer {layer + 1}"
    if request >= 0:
        name += f" | request {request + 1}"
    return name


def write_perfetto(root, timeline):
    events = []
    for pid, case in enumerate(timeline["cases"]):
        name = f"Prefix {case['prefix']:,} | {case['path']} | chip {case['chip']}"
        events += [
            dict(ph="M", pid=pid, tid=0, name="process_name", args=dict(name=name)),
            dict(ph="M", pid=pid, tid=0, name="thread_name", args=dict(name="Device programs")),
        ]
        for event in case["events"]:
            start, duration, category, source, layer, request, phase = event
            args = dict(kernel_sources=timeline["sources"][source], chip=case["chip"])
            if layer >= 0:
                args["layer"] = layer + 1
            if request >= 0:
                args.update(request=request + 1, tokens=case["per_request"][request]["tokens"])
            events.append(
                dict(
                    ph="X",
                    pid=pid,
                    tid=0,
                    name=event_name(event),
                    cat=CATEGORIES[category][0],
                    ts=start,
                    dur=duration,
                    args=args,
                )
            )
    trace = json.dumps(dict(traceEvents=events, displayTimeUnit="ms"), separators=(",", ":")).encode()
    (root / "trace.json.gz").write_bytes(gzip.compress(trace, mtime=0))


def plot(root, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "gemma4_hybrid_8k"}
    )
    fig, axes = plt.subplots(1, len(analysis["prefixes"]), figsize=(13, 5.8), layout="constrained")
    fig.suptitle("Where the hybrid prefill time goes", fontsize=20, fontweight="bold")
    for ax, prefix in zip(axes, analysis["prefixes"]):
        cases = [
            next(c for c in analysis["cases"] if c["prefix"] == prefix and c["path"] == path)
            for path in ("legacy", "hybrid")
        ]
        bottom = [0.0, 0.0]
        for category, (name, color) in enumerate(CATEGORIES):
            values = [c["categories_ms"][category] for c in cases]
            ax.bar([0, 1], values, bottom=bottom, width=0.56, color=color, label=name)
            for x, (value, base) in enumerate(zip(values, bottom)):
                if value > max(c["total_ms"] for c in cases) * 0.07:
                    ax.text(
                        x,
                        base + value / 2,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                    )
            bottom = [b + v for b, v in zip(bottom, values)]
        for x, case in enumerate(cases):
            ax.annotate(
                f"{case['total_ms']:.1f} ms",
                (x, case["total_ms"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
            )
        ax.set_title(f"Prefix per request: {prefix:,}", pad=16)
        ax.set_xticks([0, 1], ["Legacy\n1 × 8192", "Hybrid\n6 requests / 8192 tokens"])
        ax.set_ylabel("Total device time (ms)")
        ax.set_ylim(0, max(bottom) * 1.17)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=color, label=name) for name, color in CATEGORIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.savefig(root / "breakdown.png", dpi=160)
    svg = root / "breakdown.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def write_tables(root, analysis):
    text = [
        "# Recorded device profile",
        "",
        f"Full 60-layer Gemma4 31B, CP=8 / TP=4; revision `{analysis['git_revision']}`. "
        f"{analysis['repeats']} warmed samples per cell. Representative timeline is the median sample's longest-span chip.",
        "",
        "Legacy: one 8192-token request. Hybrid: 512 + 512 + 1024 + 1536 + 2048 + 2560. "
        "Every request has the selected prefix; KV history is synthetic. No correctness/PCC run is included.",
        "",
        "## Additive breakdown (ms)",
        "",
        "| Prefix | Component | Legacy | Hybrid | Added time |",
        "|---:|---|---:|---:|---:|",
    ]
    for prefix in analysis["prefixes"]:
        a, b = [
            next(c for c in analysis["cases"] if c["prefix"] == prefix and c["path"] == path)
            for path in ("legacy", "hybrid")
        ]
        for category, (name, _) in enumerate(CATEGORIES):
            x, y = a["categories_ms"][category], b["categories_ms"][category]
            text.append(f"| {prefix:,} | {name} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")
        text.append(
            f"| {prefix:,} | **Total** | **{a['total_ms']:.3f}** | **{b['total_ms']:.3f}** | **{b['total_ms']-a['total_ms']:+.3f}** |"
        )
    text += [
        "",
        "## Hybrid SDPA time per request",
        "",
        "Mean op time over the 10 global / 50 sliding-window layers in the representative replay. Both include internal CCL.",
        "",
        "| Prefix | Request | Tokens | Global ms/op | Global total ms | SWA ms/op | SWA total ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in analysis["cases"]:
        if case["path"] != "hybrid":
            continue
        for request in case["per_request"]:
            g, s = request["global"], request["sliding"]
            text.append(
                f"| {case['prefix']:,} | {request['request']+1} | {request['tokens']} | {g['mean_ms']:.3f} | {g['total_ms']:.3f} | {s['mean_ms']:.3f} | {s['total_ms']:.3f} |"
            )
    text += [
        "",
        "## Sample ranges",
        "",
        "| Path | Prefix | Min–max device ms | Programs per chip | Representative chip |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in analysis["cases"]:
        samples = case["sample_total_ms"]
        text.append(
            f"| {case['path']} | {case['prefix']:,} | {min(samples):.3f}–{max(samples):.3f} | {sum(case['counts'][:-1])} | {case['chip']} |"
        )
    text += [
        "",
        "All 20 samples have complete, non-overlapping program spans on the selected chip. Category totals plus gaps equal the measured device span. Matmul and SDPA counts/order are checked against the 60-layer model. Per-chip clocks are normalized independently; the tracks are separate replays, not simultaneous execution.",
        "",
    ]
    (root / "measurements.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    assert report["complete"] and report["profile"]
    assert report["layers"] == 60 and report["chunk_size"] == 8192
    assert report["requests"] == {"legacy": [8192], "hybrid": [512, 512, 1024, 1536, 2048, 2560]}
    assert report["prefixes"] == [0, 253952]
    assert Counter((r["path"], r["prefix"]) for r in report["results"]) == Counter(
        {(p, n): 1 for p in report["requests"] for n in report["prefixes"]}
    )
    sources, cases = {}, []
    for row in report["results"]:
        assert len(row["captures"]) == report["repeats"]
        samples = [decode(report, row, capture, sources) for capture in row["captures"]]
        representative = min(
            range(len(samples)), key=lambda i: abs(row["captures"][i]["device_us"] - row["device_us"]["median"])
        )
        case = samples[representative]
        assert math.isclose(case["total_ms"] * 1000, row["device_us"]["median"])
        case["sample_total_ms"] = [s["total_ms"] for s in samples]
        case["sample_categories_ms"] = [s["categories_ms"] for s in samples]
        cases.append(case)
    timeline = {
        k: report[k]
        for k in ("git_revision", "started_at", "requests", "prefixes", "repeats", "device_metric", "excluded")
    }
    timeline.update(categories=CATEGORIES, sources=list(sources), cases=cases, matmul_names=MATMUL_NAMES)
    root = args.results.parent
    (root / "timeline.json").write_text(json.dumps(timeline, separators=(",", ":")) + "\n")
    analysis = {**timeline, "cases": [{k: v for k, v in c.items() if k != "events"} for c in cases]}
    analysis.pop("sources")
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_perfetto(root, timeline)
    plot(root, analysis)
    write_tables(root, analysis)
    print(
        f"Validated {sum(len(r['captures']) for r in report['results'])} captures; wrote timeline, Perfetto trace, plots and tables to {root}"
    )


if __name__ == "__main__":
    main()
