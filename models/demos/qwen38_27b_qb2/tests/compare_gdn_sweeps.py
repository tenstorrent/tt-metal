# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Publish matched measured native/candidate sweeps without inventing missing cells."""

import argparse
import copy
import csv
import hashlib
import html
import json
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.sweep_recovery import normalized_configuration, resume_measurements


def compare(
    native_path,
    candidate_path,
    *,
    variants=("native", "single-step"),
    recurrence_policies=("native", "single_step"),
    require_same_output=False,
):
    if len(variants) != 2 or len(recurrence_policies) != 2:
        raise ValueError("Comparison requires two variants and policies")
    paths = [native_path, candidate_path]
    reports = [json.loads(path.read_text()) for path in paths]
    for report, path, variant in zip(reports, paths, variants):
        if (
            report.get("state") not in ("completed", "completed_with_oom")
            or report.get("cleanup_completed") is not True
            or report.get("recurrence_variant") != variant
        ):
            raise ValueError("Comparison requires clean terminal native/candidate sweeps")
        # Reuse the controller's raw-sample accounting, prompt, repeatability
        # and allocator-error verification. The original receipt is immutable.
        resume_measurements(copy.deepcopy(report), [path])
    native, candidate = reports
    for key in (
        "replicas",
        "chips",
        "batches",
        "input_lengths",
        "output_tokens",
        "warmup_runs",
        "measured_runs",
        "methodology",
    ):
        if native.get(key) != candidate.get(key):
            raise ValueError(f"Unmatched measurement protocol: {key}")
    for report in reports:
        if report["replicas"] != 1 or report["chips"] != 4:
            raise ValueError("This comparison artifact is scoped to one measured TP4 replica")
    sources = [{k: v for k, v in r["source_sha256"].items() if k != "effective_precision_override"} for r in reports]
    if sources[0] != sources[1]:
        raise ValueError("Unmatched model sources")
    policies = [
        {k: v for k, v in r["precision"].items() if k not in ("config_id", "decode_recurrence")} for r in reports
    ]
    if policies[0] != policies[1]:
        raise ValueError("Precision differs beyond recurrence selection")
    if tuple(r["precision"]["decode_recurrence"] for r in reports) != tuple(recurrence_policies):
        raise ValueError("Wrong recurrence policies")
    if normalized_configuration(native["configuration"]) != normalized_configuration(candidate["configuration"]):
        raise ValueError("Unmatched runtime settings")

    def indexed(report):
        cells = {(cell["input_tokens"], cell["batch_per_replica"]): cell for cell in report["cells"]}
        if len(cells) != len(report["cells"]):
            raise ValueError("Duplicate sweep cells")
        return cells

    left, right = indexed(native), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Unmatched sweep geometry")
    rows = []
    for (length, batch), a in sorted(left.items()):
        b = right[(length, batch)]
        row = dict(
            input_tokens=length, batch_per_replica=batch, native_status=a["status"], candidate_status=b["status"]
        )
        for label, cell in (("native", a), ("candidate", b)):
            if require_same_output and cell["status"] != "completed":
                raise ValueError("Output equivalence requires completed cells")
            if cell["status"] == "completed":
                for key in (
                    "aggregate_decode_tokens_per_second",
                    "tokens_per_second_per_user",
                    "aggregate_input_tokens_per_second",
                    "ttft_p50_s",
                ):
                    row[f"{label}_{key}"] = cell["summary"][key]
        if a["status"] == b["status"] == "completed":
            if a["prompt_sha256"] != b["prompt_sha256"] or a["concurrency"] != b["concurrency"]:
                raise ValueError("Unmatched prompts or concurrency")
            row["decode_uplift_percent"] = 100 * (
                b["summary"]["aggregate_decode_tokens_per_second"] / a["summary"]["aggregate_decode_tokens_per_second"]
                - 1
            )
            row["same_output_hash_as_native"] = (
                a["warmup"]["output_sha256_per_replica"] == b["warmup"]["output_sha256_per_replica"]
            )
            if require_same_output and not row["same_output_hash_as_native"]:
                raise ValueError("Full-model output differs between recurrence variants")
        rows.append(row)
    return dict(
        state="completed",
        scope="One TP4 replica, 64 layers; measured decode output throughput, not full-Galaxy or online-eval qualification",
        replicas=1,
        precision_change=False,
        promoted_to_serving=False,
        variant_labels=list(variants),
        output_equivalence_required=require_same_output,
        source_receipts=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
        methodology=native["methodology"],
        cells=rows,
    )


def render(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    output.mkdir(parents=True, exist_ok=False)
    rows = report["cells"]
    lengths = sorted({r["input_tokens"] for r in rows})
    batches = sorted({r["batch_per_replica"] for r in rows})
    cells = {(r["input_tokens"], r["batch_per_replica"]): r for r in rows}
    metrics = [
        "native_aggregate_decode_tokens_per_second",
        "candidate_aggregate_decode_tokens_per_second",
        "decode_uplift_percent",
    ]
    labels = report.get("variant_labels", ["native", "single-step"])
    titles = [f"{labels[0]} · output tok/s", f"{labels[1]} · output tok/s", "Candidate change vs control · %"]
    measured = sum("decode_uplift_percent" in row for row in rows)
    maximum = max(r.get(key, 0) for r in rows for key in metrics[:2])
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
        for axis, key, title in zip(axes, metrics, titles):
            matrix = np.full((len(lengths), len(batches)), np.nan)
            for y, length in enumerate(lengths):
                for x, batch in enumerate(batches):
                    matrix[y, x] = cells[(length, batch)].get(key, float("nan"))
            cmap = plt.get_cmap("RdBu" if key == metrics[-1] else "YlGnBu").copy()
            cmap.set_bad("#edf0f4")
            options = (
                dict(norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60))
                if key == metrics[-1]
                else dict(vmin=0, vmax=maximum)
            )
            image = axis.imshow(matrix, cmap=cmap, aspect="auto", **options)
            fig.colorbar(image, ax=axis, shrink=0.8)
            for y, length in enumerate(lengths):
                for x, batch in enumerate(batches):
                    value = matrix[y, x]
                    row = cells[(length, batch)]
                    if np.isnan(value):
                        status = row["candidate_status"] if key == metrics[1] else row["native_status"]
                        text = {"oom": "OOM", "capacity_guard": "guard", "implementation_guard": "B64\npending"}.get(
                            status, status
                        )
                        color = "#556070"
                    else:
                        text = f"{value:+.1f}" if key == metrics[-1] else f"{value:.1f}"
                        color = (
                            "white"
                            if (abs(value) > 38 if key == metrics[-1] else value > maximum * 0.62)
                            else "#102030"
                        )
                    axis.text(x, y, text, ha="center", va="center", color=color, fontsize=9)
            axis.set_xticks(range(len(batches)), batches)
            axis.set_yticks(
                range(len(lengths)),
                ["~256K" if n == 262016 else f"{n//1024}K" if n % 1024 == 0 else str(n) for n in lengths],
            )
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Users per TP4 replica")
            axis.set_ylabel("Input context")
        fig.suptitle(
            f"Qwen3.8-27B · matched full-model TP4 decode sweep\n{measured} matched measured cells · precision unchanged",
            fontsize=15,
        )
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"comparison.{extension}", dpi=170)
        plt.close(fig)
    svg_path = output / "comparison.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    svg = svg_path.read_text()
    svg = svg[svg.index("<svg") :]
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen matched GDN sweep</title>
<style>body{{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 24px;background:#f7f9fc;color:#172033}}p{{line-height:1.6}}svg{{width:100%;height:auto;background:white}}a{{margin-right:18px}}</style>
<h1>Qwen3.8-27B · {html.escape(labels[0])} vs {html.escape(labels[1])} GDN</h1>
<p>{html.escape(report['scope'])}. Model sources and all precision settings match except recurrence selection.</p>
{svg}<p><a href="comparison.png">PNG</a><a href="comparison.svg">SVG</a><a href="comparison.pdf">PDF</a><a href="comparison.csv">CSV</a><a href="comparison.json">JSON</a></p>
<p>{html.escape(report['methodology'])}</p>
<p>OOM and untested cells retain their explicit status. A guard is not a measured memory limit.
No measurements are imputed or multiplied into full-Galaxy throughput. The 128-token timing budget continues past EOS;
repeatability is not reference-evaluation qualification. Near-256K reserves space for those output tokens.</p>
<p><a href="../gdn-native-sweep-v4/index.html">Native input/output/TTFT graphs</a><a href="../gdn-candidate-sweep-v4/index.html">Candidate input/output/TTFT graphs</a></p>
</html>\n"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(compare(args.native, args.candidate), args.output)
