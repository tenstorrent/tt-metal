# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measured Galaxy sweep receipts and standalone plots; no device imports."""

import argparse
import csv
import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

INPUT_LENGTHS = (128, 8192, 32768, 55000, 131072, 262016)
CONCURRENCIES = (1, 2, 4, 8, 16)
MAX_CONTEXT = 262144
MAX_POOL_TOKENS = 1179648


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(replicas=1):
    if replicas not in (1, 8):
        raise ValueError("Sweep supports one TP4 replica or eight independent replicas")
    cells = []
    for length in INPUT_LENGTHS:
        for batch in CONCURRENCIES:
            pool_tokens = batch * ((length + 127 + 31) // 32 * 32)
            allowed = length + 127 <= MAX_CONTEXT and pool_tokens <= MAX_POOL_TOKENS
            cells.append(
                dict(
                    input_tokens=length,
                    batch_per_replica=batch,
                    concurrency=replicas * batch,
                    pool_tokens_per_replica=pool_tokens,
                    status="queued" if allowed else "capacity_guard",
                    reason=None if allowed else "Exceeds the current 1,179,648-token per-replica KV allocation guard",
                )
            )
    return dict(
        state="queued",
        created_at=utc_now(),
        replicas=replicas,
        chips=4 * replicas,
        output_tokens=128,
        warmup_runs=1,
        measured_runs=3,
        methodology=(
            "Full 64-layer native generator; fresh full prefills, no prefix reuse; fixed 128-token output "
            "including tokens beyond EOS. Warm measurements exclude weight loading and first-use compilation. "
            "TTFT includes request-state reset, prefill and first token readback. Decode uses device sampling "
            "and deferred history readback; HTTP and router overhead are not included. "
            "Whole-Galaxy throughput is measured only when replicas=8, never multiplied from TP4 results."
        ),
        capacity_note="Capacity-guard cells are not tested; they are not measured OOM failures.",
        cells=cells,
    )


def summarize(samples, *, concurrency, output_tokens):
    if not samples or output_tokens < 2 or concurrency < 1:
        raise ValueError("Expected positive concurrency, decode work and measured samples")
    for sample in samples:
        if sample["decode_s"] <= 0 or sample["elapsed_s"] <= 0 or len(sample["ttft_s"]) != concurrency:
            raise ValueError("Invalid measurement dimensions or duration")
        if sample["trace_captures"] != 0:
            raise ValueError("Measured window included trace capture; warmup is incomplete")
    decode = statistics.median(row["decode_s"] for row in samples)
    elapsed = statistics.median(row["elapsed_s"] for row in samples)
    ttfts = [latency for row in samples for latency in row["ttft_s"]]
    return dict(
        tokens_per_second_per_user=(output_tokens - 1) / decode,
        aggregate_decode_tokens_per_second=concurrency * (output_tokens - 1) / decode,
        aggregate_e2e_tokens_per_second=concurrency * output_tokens / elapsed,
        tpot_ms=1000 * decode / (output_tokens - 1),
        ttft_p50_s=statistics.median(ttfts),
        ttft_p90_s=statistics.quantiles(ttfts, n=10, method="inclusive")[8] if len(ttfts) > 1 else ttfts[0],
        decode_s=decode,
        elapsed_s=elapsed,
        samples=len(samples),
    )


def save_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = utc_now()
    temporary = directory / "sweep.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "sweep.json")


def render(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    directory = Path(directory)
    completed = [row for row in report["cells"] if row["status"] == "completed"]
    supported = sum(row["status"] != "capacity_guard" for row in report["cells"])
    colors = dict(zip(CONCURRENCIES, ("#2563eb", "#0d9488", "#a16207", "#9333ea", "#e11d48")))
    http = report.get("measurement_mode") == "http"
    metrics = (
        (
            "tokens_per_second_per_user",
            "Median client decode speed" if http else "Decode speed",
            "tokens/s/user",
            False,
        ),
        (
            "aggregate_e2e_tokens_per_second" if http else "aggregate_decode_tokens_per_second",
            "Aggregate end-to-end throughput" if http else "Aggregate decode throughput",
            "tokens/s",
            False,
        ),
        ("ttft_p50_s", "Time to first token", "seconds (log scale)", True),
    )
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
        for axis, (key, title, ylabel, log_y) in zip(axes, metrics):
            for batch in CONCURRENCIES:
                rows = sorted(
                    [row for row in completed if row["batch_per_replica"] == batch],
                    key=lambda row: row["input_tokens"],
                )
                if rows:
                    axis.plot(
                        [row["input_tokens"] for row in rows],
                        [row["summary"][key] for row in rows],
                        marker="o",
                        color=colors[batch],
                        linewidth=2,
                        label=f"C={batch * report['replicas']}",
                    )
            if key == "tokens_per_second_per_user":
                axis.axhline(40, color="#64748b", linestyle=":", linewidth=1, label="40 TSU planning target")
            axis.set_xscale("log", base=2)
            axis.set_xlim(100, MAX_CONTEXT * 1.15)
            axis.set_xticks(INPUT_LENGTHS)
            ticks = dict(zip(INPUT_LENGTHS, ("128", "8K", "32K", "55K", "128K", "~256K")))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ticks.get(x, str(int(x)))))
            axis.tick_params(axis="x", labelrotation=35)
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Input sequence length (tokens)")
            axis.set_ylabel(ylabel)
            if log_y and completed:
                axis.set_yscale("log")
            elif not log_y:
                axis.set_ylim(bottom=0)
            axis.grid(alpha=0.18)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, loc="best", fontsize=8)
            if not completed:
                axis.text(
                    0.5,
                    0.45,
                    "Sweep queued\nNo sweep measurements yet",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#475569",
                    fontsize=12,
                )
        figure.suptitle(
            f"Qwen3.8-27B · {report['replicas']} × TP4 · {len(completed)}/{supported} measured cells · "
            f"{report['state']}",
            fontsize=15,
            fontweight="bold",
        )
        for extension in ("png", "svg", "pdf"):
            figure.savefig(directory / f"sweep.{extension}", dpi=160)
        plt.close(figure)
    svg_path = directory / "sweep.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fields = list(
        dict.fromkeys(
            ["input_tokens", "concurrency", "batch_per_replica", "status"]
            + [m[0] for m in metrics]
            + ["aggregate_e2e_tokens_per_second", "ttft_p90_s", "tpot_ms", "reason"]
        )
    )
    with (directory / "sweep.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["cells"]:
            flat = {**row, **row.get("summary", {})}
            writer.writerow({key: flat.get(key, "") for key in fields})
    rows = []
    for row in report["cells"]:
        summary = row.get("summary", {})
        values = [f"{row['input_tokens']:,}", str(row["concurrency"]), row["status"]]
        values += [f"{summary[key]:.3f}" if key in summary else "—" for key, *_ in metrics]
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    baseline = report.get("baseline", {})
    baseline_note = ""
    if baseline:
        baseline_note = (
            "<aside><b>Existing sanity baseline, separate from this sweep:</b> "
            f"{baseline['input_tokens']} input tokens, C=1, "
            f"{baseline['tsu']:.2f} tokens/s/user, {baseline['ttft_ms']:.2f} ms TTFT. "
            "That test used traced short prefill; sweep points below measure the explicit native batch path.</aside>"
        )
    svg = (directory / "sweep.svg").read_text()
    svg = svg[svg.index("<svg") :]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Galaxy performance sweep</title>
<style>body{{font:16px system-ui;margin:36px auto;max-width:1450px;padding:0 24px;color:#172033;background:#f8fafc}}
h1{{font-size:30px}}p,aside{{line-height:1.6}}aside{{background:#e2e8f0;padding:16px;border-radius:8px}}
svg{{width:100%;height:auto;background:white;border-radius:12px;margin-top:20px}}
table{{border-collapse:collapse;width:100%;background:white;margin-top:24px}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #e2e8f0}}
th{{background:#e2e8f0}}a{{color:#2563eb;margin-right:18px}}small{{color:#475569}}button{{padding:8px 12px}}</style>
<h1>Qwen3.8-27B · input length × concurrency</h1>
<p><b>{report['replicas']} × TP4 ({report['chips']} chips)</b> · {len(completed)}/{supported} measured cells ·
status: <b>{html.escape(report['state'])}</b> · updated {html.escape(report.get('updated_at', ''))}</p>
<p>{html.escape(report['methodology'])}</p>{baseline_note}{svg}
<p><a href="sweep.png">PNG</a><a href="sweep.pdf">PDF</a><a href="sweep.svg">SVG</a>
<a href="sweep.csv">CSV</a><a href="sweep.json">Raw JSON</a></p>
<p><small>{html.escape(report['capacity_note'])} Near-256K uses 262,016 input tokens to leave room for 128 output tokens.
No predicted values are drawn as measurements. Lines connect completed measured points only.</small></p>
<button onclick="document.getElementById('results').hidden=!document.getElementById('results').hidden">Show / hide data table</button>
<table id="results"><thead><tr><th>ISL</th><th>Concurrency</th><th>Status</th><th>Tokens/s/user</th><th>{'End-to-end' if http else 'Decode'} tokens/s</th><th>p50 TTFT (s)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></html>"""
    (directory / "index.html").write_text(document + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--replicas", type=int, choices=(1, 8), default=1)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.init:
        if (args.output / "sweep.json").exists():
            parser.error("Refusing to replace an existing sweep")
        report = make_plan(args.replicas)
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            report["baseline"] = dict(
                input_tokens=len(baseline["prompt_tokens"]),
                tsu=baseline["second_run_perf"]["tokens_per_second"],
                ttft_ms=1000 * baseline["second_run_perf"]["ttft_s"],
            )
        save_report(report, args.output)
    else:
        report = json.loads((args.output / "sweep.json").read_text())
    render(report, args.output)


if __name__ == "__main__":
    main()
