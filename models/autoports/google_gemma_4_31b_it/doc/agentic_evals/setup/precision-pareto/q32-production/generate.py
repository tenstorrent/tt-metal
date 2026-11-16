import hashlib
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path("/home/mvasiljevic/tt-metal-gemma4-31b-pipeline")
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
out = D / "setup/precision-pareto/q32-production"
entries = [
    (
        "P19 Q32 down-BFP8",
        D / "setup/p19-q32-production-validation/acceptance/run/readiness_accuracy.json",
        D / "setup/p19-q32-production-validation/acceptance/run/performance_b1.json",
        D / "setup/p19-q32-production-validation/acceptance/run/runtime_policy.json",
        "protocol-rejected; diagnostic only",
        "P19 failed protocol controls; not selected. Acceptance run independently root-reviewed.",
    ),
    (
        "Full BF16/HiFi4 under Q32",
        D / "setup/full-q32-production-validation/full-bf16-broad-acceptance-candidate/run/readiness_accuracy.json",
        D / "setup/full-q32-production-validation/full-bf16-broad-acceptance-candidate/run/performance_b1.json",
        D / "setup/full-q32-production-validation/full-bf16-broad-acceptance-candidate/run/runtime_policy.json",
        "unselected diagnostic baseline; process exit status unknown",
        "Broad artifacts and context-boundary checks are present and root-reviewed; wrapper exited 143, actual child exit status is unrecoverable and remains null.",
    ),
]
rows = []
for label, accp, perf, polp, status, note in entries:
    acc = json.load(open(accp))
    perfj = json.load(open(perf))
    pol = json.load(open(polp))
    policy = pol.get("policy", pol)
    nat = acc["natural"][0]
    diag = perfj.get("diagnostics", {})
    assert acc["revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert perfj["source"] == "autoregressive" and (
        perfj["input_tokens"],
        perfj["generated_tokens"],
        perfj["batch_size"],
        perfj["concurrent_requests"],
    ) == (4096, 128, 1, 1)
    assert diag.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1" and diag.get("TT_METAL_DEVICE_PROFILER") == "0"
    row = {
        "label": label,
        "status": status,
        "status_note": note,
        "natural512": {
            "top1": nat["top1"],
            "top1_matches": nat["matches_top1"],
            "top5": nat["top5"],
            "top5_matches": nat["matches_top5"],
            "total": nat["total"],
        },
        "autoregressive_b1c1_4096_128": {
            "decode_tps": perfj["decode_tps"],
            "ttft_ms": perfj["ttft_ms"],
            "input_tokens": perfj["input_tokens"],
            "generated_tokens": perfj["generated_tokens"],
            "batch_size": perfj["batch_size"],
            "concurrent_requests": perfj["concurrent_requests"],
            "tracking": "1",
            "profiler": "0",
            "sample_decode_tps": [s["decode_tps"] for s in perfj["samples"]],
        },
        "runtime_policy": {
            k: policy.get(k)
            for k in [
                "config_id",
                "weight_groups",
                "compute_fidelities",
                "kv_cache_dtype",
                "ccl_dtype",
                "activation_dtype",
                "max_seq_len",
            ]
        },
        "sources": {
            str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in [accp, perf, polp]
        },
    }
    if label.startswith("P19"):
        terminal_paths = [D / "setup/p19-q32-production-validation/acceptance/root-review.json"]
    else:
        broad = D / "setup/full-q32-production-validation/full-bf16-broad-acceptance-candidate"
        terminal_paths = [
            broad / "root-review.json",
            broad / "completion.json",
            broad / "process-exit-observer.json",
            broad / "wrapper-interruption.json",
        ]
    row["terminal_evidence"] = {
        str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in terminal_paths
    }
    rows.append(row)
obj = {
    "schema_version": 1,
    "title": "Stage 12 Q32-production diagnostic comparison",
    "scope": "Two matched current-Q32 production-generation configurations only; natural 512-token accuracy vs actual autoregressive B1/C1 4096-input, 128-output decode rate. No vLLM rates or historical pre-Q32 datapoints.",
    "status": "Terminal broad artifacts and checks are root-reviewed, but the broad child process exit status is unknown (outer wrapper exit 143; no exit-zero inference). P19 is protocol-rejected; both points are unselected diagnostics.",
    "reference_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "instrumentation_requirements_verified": {"TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_DEVICE_PROFILER": "0"},
    "limitations": [
        "This compares diagnostic configurations and does not select a serving policy.",
        "Natural top-1/top-5 are agreement with the saved natural-token reference, not task correctness.",
        "Autoregressive single-request timing is not vLLM serving throughput.",
        "The full-BF16 broad row has all expected outputs and context-boundary checks, but child process exit status is unknown; do not claim a clean process exit.",
        "P19 was rejected for protocol behavior; charting it does not imply acceptance.",
    ],
    "rows": rows,
}
(out / "source.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def chart(metric, title, xlabel, path, threshold, xlim):
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=180)
    # P19 is protocol-rejected, so use neutral diagnostic colors and do not imply selection.
    colors = ["#d28a35", "#2878a5"]
    points = []
    for i, r in enumerate(rows):
        x = r["natural512"][metric]
        y = r["autoregressive_b1c1_4096_128"]["decode_tps"]
        points.append((x, y, i))
        ax.scatter([x], [y], s=105, color=colors[i], edgecolor="black", linewidth=0.6, zorder=3)
    # The non-dominated set is descriptive only: higher accuracy and rate are both preferred.
    frontier = []
    for x, y, i in points:
        if not any((xx >= x and yy >= y and (xx > x or yy > y)) for xx, yy, _ in points):
            frontier.append((x, y))
    frontier.sort()
    if len(frontier) > 1:
        ax.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            color="#667085",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label="non-dominated diagnostic frontier",
        )
    elif len(frontier) == 1:
        ax.scatter(
            [frontier[0][0]],
            [frontier[0][1]],
            s=210,
            facecolors="none",
            edgecolors="#667085",
            linewidth=1.2,
            zorder=2,
            label="single non-dominated diagnostic point",
        )
    ax.axvline(threshold, color="#555555", linestyle=":", linewidth=1.4, label=f"acceptance threshold {threshold:.2f}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    # Keep every callout within the axes even when both top-5 values are identical.
    offsets = [(-190, -22), (-190, 15)] if metric == "top5" else [(10, 14), (10, 14)]
    for i, (x, y, idx) in enumerate(points):
        r = rows[idx]
        txt = f"{r['label']}\n{r['natural512'][metric+'_matches']}/{r['natural512']['total']} {metric}; {y:.2f} tok/s\n{r['status']}"
        ax.annotate(
            txt,
            (x, y),
            xytext=offsets[idx],
            textcoords="offset points",
            fontsize=7.6,
            ha="left",
            va="bottom" if offsets[idx][1] > 0 else "top",
            bbox=dict(boxstyle="round,pad=.28", fc="white", ec=colors[idx], alpha=0.96),
            arrowprops=dict(arrowstyle="-", color=colors[idx], lw=0.8),
        )
    ax.set_title(title + "\nCurrent Q32 production diagnostics; not vLLM serving measurements", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Autoregressive diagnostic decode rate (tokens/s)")
    ax.set_xlim(*xlim)
    ymin = min(r["autoregressive_b1c1_4096_128"]["decode_tps"] for r in rows) - 0.45
    ymax = max(r["autoregressive_b1c1_4096_128"]["decode_tps"] for r in rows) + 0.65
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "AR benchmark: B1/C1, 4096 input, 128 output. P19 is protocol-rejected; no point is selected.\nFrontier is descriptive only. Tracking=1; profiler=0. Agreement is not task correctness.",
        fontsize=7.2,
        ha="left",
    )
    fig.tight_layout(rect=(0.12, 0.12, 0.99, 0.91))
    fig.savefig(out / path, bbox_inches="tight")
    plt.close(fig)


chart(
    "top1",
    "Natural 512-token top-1 agreement vs autoregressive rate",
    "Natural-token top-1 agreement",
    "top1_perf_pareto.png",
    0.90,
    (0.885, 0.965),
)
chart(
    "top5",
    "Natural 512-token top-5 agreement vs autoregressive rate",
    "Natural-token top-5 agreement",
    "top5_perf_pareto.png",
    0.98,
    (0.965, 1.015),
)
print("wrote", out)
