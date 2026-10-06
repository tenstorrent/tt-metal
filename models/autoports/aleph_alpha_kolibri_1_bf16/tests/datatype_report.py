# SPDX-License-Identifier: Apache-2.0
"""Aggregate measured full-model trials and draw their accuracy/performance frontier."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    rows = []
    for path in sorted((root / "candidates").glob("*/result.json")):
        result = json.loads(path.read_text())
        config = result["precision_config"]
        if result["status"] == "pass":
            assert len(result["runtime_summary"]["layers"]) == 50
            assert result["runtime_summary"]["capacity"] == 1048576
            assert result["accuracy"]["total"] == 100
        accuracy = result.get("accuracy", {})
        command = path.with_name("run.command.json")
        command = json.loads(command.read_text()) if command.exists() else {}
        rows.append(
            dict(
                config_id=result["config_id"],
                precision_config_path=str(root / "configs" / f"{result['config_id']}.json"),
                dtype_policy=dict(
                    mesh_policy={k: v for k, v in config["mesh_policy"].items() if "dtype" in k},
                    runtime=config["runtime"],
                    layer_exceptions=config["layer_exceptions"],
                ),
                compute_fidelity_policy={
                    k: v for k, v in config["mesh_policy"].items() if "fidelity" in k or "fp32" in k
                }
                | {k: v for k, v in config["runtime"].items() if "fidelity" in k or "fp32" in k},
                top1=accuracy.get("top1"),
                top5=accuracy.get("top5"),
                top100=accuracy.get("top100"),
                prefill=result.get("prefill"),
                token_count=accuracy.get("total"),
                ttft_ms=result.get("ttft_ms"),
                decode_t_s_u=result.get("decode_t_s_u"),
                decode_t_s_u_samples=[s["rows"][0]["decode_t/s/u"] for s in result.get("teacher_forcing_samples", [])],
                environment=command.get("environment", result["provenance"]["environment"]),
                measurement_regime=result.get("measurement_regime"),
                prefill_chunk_size=config["mesh_policy"]["prefill_chunk_size"],
                traced_prefill_buckets=sorted(
                    int(k.removeprefix("prefill_")) for k in result.get("traces", {}) if k.startswith("prefill_")
                ),
                construction_regime=result.get("construction_regime", "fresh generator through public build_generator"),
                trace_verified=bool(result.get("traces")),
                runtime_policy_evidence=str(path),
                command=command.get("command", result["provenance"]["command"]),
                git_head=result["provenance"]["git_head"],
                hardware=result["hardware"],
                mesh=result["mesh"],
                status=result["status"],
                error=result.get("error"),
                reference=result["reference"],
                reference_sha256=result["reference_sha256"],
            )
        )
    passing = [r for r in rows if r["status"] == "pass" and r["trace_verified"] and r["decode_t_s_u"]]
    if args.select:
        assert set(json.loads((root / "configs/matrix.json").read_text())) == {r["config_id"] for r in rows}
        assert all(r["status"] in ("pass", "accuracy-fail", "capacity-rejected") for r in rows)
        assert all(r["environment"].get("TT_METAL_TRACE_ALLOC_TRACKING") == "0" for r in passing)
    selected = max(passing, key=lambda r: r["decode_t_s_u"]) if passing else None
    for row in rows:
        row["selected"] = bool(selected and row["config_id"] == selected["config_id"])
    data = dict(
        thresholds=dict(top1=0.90, top5=0.98, top100=1.0),
        selection_metric="median warmed trace-verified teacher-forcing decode t/s/u",
        selected_config_id=selected["config_id"] if selected else None,
        final_selection=args.select,
        results=rows,
    )
    final_path = root / "selected_default/result.json"
    final_command_path = root / "selected_default/run.command.json"
    if args.select and final_path.exists() and final_command_path.exists():
        final = json.loads(final_path.read_text())
        final_command = json.loads(final_command_path.read_text())
        assert final["config_id"] == selected["config_id"] and final["status"] == "pass"
        assert final_command["exit_code"] == 0
        # Keep the handoff's serving comparison unambiguous. Public generate
        # samples provide TTFT; their readback decode rates remain in raw evidence.
        token_out = {k: v for k, v in final["token_out"].items() if k != "public_samples"}
        token_out["ttft_regime"] = "median of three warmed public generate calls, B1 P128 G128"
        token_out[
            "decode_regime"
        ] = "128 nonblocking model+split-sampler replays after warm first token; no per-token host updates/readbacks/synchronization"
        data["post_selection_token_out"] = dict(evidence="selected_default/result.json", **token_out)
        handoff = dict(
            schema_version=1,
            config_id=selected["config_id"],
            hardware=final["hardware"],
            mesh=final["mesh"],
            primary_later_comparison="post_selection_token_out",
            comparison_rule="Later serving reports must compare against this normal-default-path token-out benchmark, keeping teacher-forcing timing separate.",
            post_selection_token_out=data["post_selection_token_out"],
            teacher_forcing_selection={
                k: selected[k] for k in ("top1", "top5", "top100", "ttft_ms", "decode_t_s_u", "measurement_regime")
            },
            teacher_forcing_default_reproduction=dict(
                accuracy=final["accuracy"],
                ttft_ms=final["ttft_ms"],
                decode_t_s_u=final["decode_t_s_u"],
                evidence="selected_default/result.json",
            ),
        )
        (root / "perf_summary.json").write_text(json.dumps(handoff, indent=2) + "\n")
    (root / "sweep_results.json").write_text(json.dumps(data, indent=2) + "\n")
    if rows:
        with (root / "sweep_results.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows
            )
    if args.select:
        assert selected, "No passing trace-verified full-model candidate"
        measured_config = json.loads(Path(selected["runtime_policy_evidence"]).read_text())["precision_config"]
        assert json.loads(Path(selected["precision_config_path"]).read_text()) == measured_config
        (root / "selected_precision_config.json").write_text(json.dumps(measured_config, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "#ffffff",
        }
    )
    measured = [r for r in rows if r["trace_verified"] and r["decode_t_s_u"] and r["top1"] is not None]
    for metric, threshold in (("top1", 0.9), ("top5", 0.98)):
        fig, (ax, key) = plt.subplots(
            1, 2, figsize=(14, max(8, 1.5 + 0.22 * len(measured))), gridspec_kw={"width_ratios": [1.7, 1]}
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.13, wspace=0.25)
        key.axis("off")
        frontier = [
            r
            for r in measured
            if not any(
                o[metric] >= r[metric]
                and o["decode_t_s_u"] >= r["decode_t_s_u"]
                and (o[metric] > r[metric] or o["decode_t_s_u"] > r["decode_t_s_u"])
                for o in measured
            )
        ]
        frontier = sorted(frontier, key=lambda r: r[metric])
        ax.plot(
            [r[metric] * 100 for r in frontier],
            [r["decode_t_s_u"] for r in frontier],
            color="#348e9f",
            linewidth=1.8,
            marker="o",
            markersize=13,
            markeredgewidth=1.3,
            markerfacecolor="none",
            label="Pareto frontier",
            zorder=1,
        )
        ax.axvline(
            threshold * 100, color="#555e6b", linestyle=":", linewidth=1.6, label=f"Minimum {metric}: {threshold:.0%}"
        )
        for i, row in enumerate(measured):
            color = "#d22d3d" if row["selected"] else "#264f78" if row["status"] == "pass" else "#9ea6b2"
            ax.scatter(
                row[metric] * 100,
                row["decode_t_s_u"],
                s=160 if row["selected"] else 52,
                color=color,
                marker="*" if row["selected"] else "o",
                alpha=1.0 if row["selected"] else 0.6,
                zorder=4,
            )
            label = f"{i+1:02d}  {row['config_id']}"
            if row["selected"]:
                label += "  ★"
            key.text(
                0,
                0.92 - i * 0.84 / max(len(measured), 1),
                label,
                transform=key.transAxes,
                fontsize=8.5,
                color=color,
                fontweight="bold" if row["selected"] else "normal",
            )
        ax.margins(x=0.18, y=0.22)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # Label the frontier and meaningful controls without stretching dozens
        # of near-identical head/cache points across the entire plot height.
        annotated = {
            "baseline_bfp4_lofi",
            "bfp4_hifi2",
            "ccl_bfp8",
            "kv_bfp4",
            "projection_activation_bfp8",
            "residual_bfp8",
            *(row["config_id"] for row in frontier),
        }
        if measured:
            lowest_accuracy = min(row[metric] for row in measured)
            edge = max((row for row in measured if row[metric] == lowest_accuracy), key=lambda row: row["decode_t_s_u"])
            annotated.add(edge["config_id"])
        ordered = sorted(
            ((i, row) for i, row in enumerate(measured) if row["config_id"] in annotated),
            key=lambda pair: pair[1]["decode_t_s_u"],
        )
        gap = min(0.04, 0.86 / max(len(ordered), 1))
        labels = []
        for i, row in ordered:
            desired = (row["decode_t_s_u"] - ymin) / (ymax - ymin)
            labels.append(max(desired, labels[-1] + gap if labels else 0.03))
        if labels and labels[-1] > 0.97:
            labels[-1] = 0.97
            for i in range(len(labels) - 2, -1, -1):
                labels[i] = min(labels[i], labels[i + 1] - gap)
        for (i, row), y in zip(ordered, labels):
            color = "#d22d3d" if row["selected"] else "#4b5b70"
            ax.annotate(
                f"{i+1:02d}",
                (row[metric] * 100, row["decode_t_s_u"]),
                xytext=(row[metric] * 100 + 0.025 * (xmax - xmin), ymin + y * (ymax - ymin)),
                fontsize=8,
                color=color,
                va="center",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.35, lw=0.65),
            )
        ax.set(
            xlabel=f"{metric.replace('top','Top-')} accuracy (%)",
            ylabel="Traced teacher-forcing decode (tokens/s/user)",
        )
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        key.text(0, 1, "Evaluated full-model configurations", transform=key.transAxes, fontweight="bold", fontsize=11)
        key.text(
            0,
            -0.035,
            "Blue: pass   Gray: rejected   Red star: selected",
            transform=key.transAxes,
            fontsize=9,
            color="#586477",
        )
        fig.suptitle(
            f"Kolibri-1 · {metric.replace('top','Top-')} accuracy / decode performance",
            x=0.07,
            ha="left",
            fontsize=17,
            fontweight="bold",
            color="#183550",
        )
        fig.text(
            0.07,
            0.035,
            "AIME24 chat · B1 · 197 prompt + 100 continuation tokens · 50 layers · QB2 TP4 · 1M capacity",
            fontsize=10,
            color="#586477",
        )
        fig.text(
            0.07,
            0.012,
            "Final selection" if args.select else "Provisional results — sweep in progress",
            fontsize=9,
            color="#586477",
        )
        fig.savefig(root / f"{metric}_perf_pareto.png", dpi=170)
        plt.close(fig)
    print(
        json.dumps(
            dict(
                selected=data["selected_config_id"],
                rows=[{k: r[k] for k in ("config_id", "status", "top1", "top5", "decode_t_s_u")} for r in rows],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
