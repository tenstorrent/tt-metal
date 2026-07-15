# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Aggregate Gemma 4 full-model datatype candidates and draw readable Pareto charts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _row(result: dict) -> dict:
    runtime = result["runtime_policy_summary"]
    layer0 = runtime["layers"][0]
    accuracy = result["accuracy"]["teacher_forcing"]
    perf = result["performance"]
    weight_groups = {**layer0["weight_groups"], "lm_head": runtime["lm_head"]["weight_dtype"]}
    fidelities = {**layer0["compute_fidelities"], "lm_head": runtime["lm_head"]["math_fidelity"]}
    return {
        "config_id": result["config_id"],
        "precision_config": result["precision_config"],
        "weight_groups": weight_groups,
        "layer_exceptions": result["dtype_policy"].get("layer_exceptions", []),
        "compute_fidelities": fidelities,
        "activation_dtype": layer0["activation_dtype"],
        "residual_dtype": layer0["residual_dtype"],
        "ccl_dtype": layer0["ccl_dtype"],
        "kv_cache_dtype": layer0["kv_cache_dtype"],
        "logits_dtype": runtime["lm_head"]["logits_dtype"],
        "sampling_dtype": runtime["sampling"]["gather_values_dtype"],
        "runtime_consumption": {
            "physical_weight_dtypes": layer0.get("physical_weight_dtypes"),
            "qkv_split_input_dtype": layer0["qkv_split_input_dtype"],
            "cache_update_input_dtype": layer0["cache_update_input_dtype"],
            "mlp_packed_gate_up_in0_block_w": layer0["program_geometry"]["mlp_packed_gate_up_in0_block_w"],
            "physical_lm_head_weight_dtype": runtime["lm_head"].get("physical_weight_dtype"),
            "sampling": runtime["sampling"],
        },
        "top1": accuracy["top1"],
        "top5": accuracy["top5"],
        "top100": accuracy["top100"],
        "token_count": accuracy["total"],
        "ttft_ms": perf["ttft_ms"],
        "trace_verified_teacher_forcing_decode_t/s/u": perf["trace_verified_teacher_forcing_decode_t/s/u"],
        "teacher_forcing_e2e_t/s/u": perf["teacher_forcing_e2e_t/s/u"],
        "trace_verified": result["trace_verified"],
        "trace_counters": result["trace_counters"],
        "measurement_regime": perf["measurement_regime"],
        "command": result["command"],
        "git_commit": result["git_commit"],
        "hardware": result["hardware"],
        "mesh": result["mesh"],
        "reference": result["reference"],
        "pass": result["pass"],
        "status": result["status"],
    }


def _pareto(rows: list[dict], accuracy_key: str) -> list[dict]:
    perf_key = "trace_verified_teacher_forcing_decode_t/s/u"
    return sorted(
        [
            row
            for row in rows
            if not any(
                other[accuracy_key] >= row[accuracy_key]
                and other[perf_key] >= row[perf_key]
                and (other[accuracy_key] > row[accuracy_key] or other[perf_key] > row[perf_key])
                for other in rows
            )
        ],
        key=lambda row: row[accuracy_key],
    )


def _main_panel_rows(rows: list[dict], accuracy_key: str, threshold: float) -> list[dict]:
    """Keep the threshold decision region readable while the overview retains all rows."""

    accuracy_floor = max(0.0, threshold - 0.05)
    clustered = [row for row in rows if row[accuracy_key] >= accuracy_floor]
    return clustered or list(rows)


def _annotation_roles(
    rows: list[dict], selected: str, accuracy_key: str, threshold: float
) -> dict[str, tuple[dict, list[str]]]:
    """Select decision-relevant labels without recreating an unreadable label cloud."""

    perf_key = "trace_verified_teacher_forcing_decode_t/s/u"
    chosen = next(row for row in rows if row["config_id"] == selected)
    labels: dict[str, tuple[dict, list[str]]] = {}

    def add(row: dict, role: str) -> None:
        entry = labels.setdefault(row["config_id"], (row, []))
        if role not in entry[1]:
            entry[1].append(role)

    add(chosen, "selected")

    passing_alternatives = sorted(
        (row for row in rows if row["pass"] and row["config_id"] != selected),
        key=lambda row: (abs(chosen[perf_key] - row[perf_key]), -row[perf_key], row["config_id"]),
    )
    for row in passing_alternatives[:3]:
        add(row, "closest passing")

    for row in _pareto(rows, accuracy_key):
        add(row, "frontier")

    accuracy_failures = [row for row in rows if row[accuracy_key] < threshold]
    if accuracy_failures:
        add(max(accuracy_failures, key=lambda row: (row[accuracy_key], row[perf_key])), "closest accuracy failure")
        add(max(accuracy_failures, key=lambda row: row[perf_key]), "fastest accuracy failure")

    combined_gate_failures = [row for row in rows if not row["pass"]]
    if combined_gate_failures:
        add(
            min(
                combined_gate_failures,
                key=lambda row: (abs(row[accuracy_key] - threshold), -row[perf_key], row["config_id"]),
            ),
            "combined-gate failure",
        )
    return labels


def _spread_positions(values: list[float], lower: float, upper: float, gap: float) -> list[float]:
    """Return ordered label positions separated by at least ``gap`` when feasible."""

    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    placed: list[tuple[int, float]] = []
    for index, value in indexed:
        placed.append((index, max(value, placed[-1][1] + gap if placed else lower)))
    overflow = placed[-1][1] - upper
    if overflow > 0:
        placed = [(index, value - overflow) for index, value in placed]
    for position in range(len(placed) - 2, -1, -1):
        index, value = placed[position]
        value = min(value, placed[position + 1][1] - gap)
        placed[position] = (index, value)
    underflow = lower - placed[0][1]
    if underflow > 0:
        placed = [(index, value + underflow) for index, value in placed]
    result = [0.0] * len(values)
    for index, value in placed:
        result[index] = value
    return result


def _annotate_key_rows(
    ax,
    rows: list[dict],
    roles: dict[str, tuple[dict, list[str]]],
    accuracy_key: str,
    y_limits: tuple[float, float],
) -> list[str]:
    perf_key = "trace_verified_teacher_forcing_decode_t/s/u"
    visible = [roles[row["config_id"]] for row in rows if row["config_id"] in roles]
    visible.sort(key=lambda entry: (entry[0][perf_key], entry[0]["config_id"]), reverse=True)
    lanes = (visible[::2], visible[1::2])
    x_lower, x_upper = ax.get_xlim()
    x_span = x_upper - x_lower
    y_span = y_limits[1] - y_limits[0]
    annotated: list[str] = []
    for lane_index, lane in enumerate(lanes):
        positions = _spread_positions(
            [row[perf_key] for row, _ in lane],
            y_limits[0] + 0.05 * y_span,
            y_limits[1] - 0.05 * y_span,
            0.075 * y_span,
        )
        label_x = x_lower + (0.025 if lane_index == 0 else 0.975) * x_span
        horizontal_alignment = "left" if lane_index == 0 else "right"
        for (row, row_roles), label_y in zip(lane, positions):
            accuracy = 100 * row[accuracy_key]
            role = "/".join(row_roles)
            ax.annotate(
                f"{role}: {row['config_id']}\n{accuracy:.0f}%, {row[perf_key]:.2f} t/s/u",
                xy=(accuracy, row[perf_key]),
                xytext=(label_x, label_y),
                textcoords="data",
                ha=horizontal_alignment,
                va="center",
                fontsize=7.2,
                arrowprops={"arrowstyle": "-", "color": "#666", "linewidth": 0.7, "alpha": 0.75},
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#bbb", "alpha": 0.9},
                zorder=6,
            )
            annotated.append(row["config_id"])
    return annotated


def _draw_points(ax, rows: list[dict], frontier: list[dict], chosen: dict, accuracy_key: str, threshold: float) -> None:
    perf_key = "trace_verified_teacher_forcing_decode_t/s/u"
    passing = [row for row in rows if row["pass"]]
    rejected = [row for row in rows if not row["pass"]]
    if passing:
        ax.scatter(
            [100 * row[accuracy_key] for row in passing],
            [row[perf_key] for row in passing],
            s=72,
            color="#3178c6",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label="accuracy-gate passing",
            zorder=2,
        )
    if rejected:
        ax.scatter(
            [100 * row[accuracy_key] for row in rejected],
            [row[perf_key] for row in rejected],
            s=76,
            color="#d97706",
            marker="X",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            label="accuracy-gate rejected",
            zorder=2,
        )
    ax.plot(
        [100 * row[accuracy_key] for row in frontier],
        [row[perf_key] for row in frontier],
        color="#1b4332",
        linewidth=2.2,
        marker="o",
        label="Pareto frontier",
        zorder=3,
    )
    ax.scatter(
        [100 * chosen[accuracy_key]],
        [chosen[perf_key]],
        s=180,
        color="red",
        marker="*",
        edgecolor="#7f0000",
        linewidth=0.9,
        label="selected policy",
        zorder=5,
    )
    ax.axvline(
        100 * threshold,
        color="#555",
        linestyle=":",
        linewidth=2,
        label=f"minimum {100 * threshold:.0f}%",
    )


def _plot(rows: list[dict], selected: str, accuracy_key: str, threshold: float, output: Path) -> dict:
    perf_key = "trace_verified_teacher_forcing_decode_t/s/u"
    frontier = _pareto(rows, accuracy_key)
    chosen = next(row for row in rows if row["config_id"] == selected)
    main_rows = _main_panel_rows(rows, accuracy_key, threshold)
    excluded = [row for row in rows if row not in main_rows]
    roles = _annotation_roles(rows, selected, accuracy_key, threshold)

    fig, (ax, overview_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 8),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": [3.5, 1.15]},
    )
    _draw_points(ax, main_rows, frontier, chosen, accuracy_key, threshold)
    _draw_points(overview_ax, rows, frontier, chosen, accuracy_key, threshold)

    main_x_values = [100 * row[accuracy_key] for row in main_rows] + [100 * threshold]
    main_span = max(max(main_x_values) - min(main_x_values), 1.0)
    main_padding = max(0.35, 0.12 * main_span)
    ax.set_xlim(max(0.0, min(main_x_values) - main_padding), min(100.5, max(main_x_values) + main_padding))
    y_values = [row[perf_key] for row in rows]
    y_span = max(max(y_values) - min(y_values), 1.0)
    y_limits = (min(y_values) - 0.08 * y_span, max(y_values) + 0.08 * y_span)
    ax.set_ylim(*y_limits)
    annotated = _annotate_key_rows(ax, main_rows, roles, accuracy_key, y_limits)

    overview_x_values = [100 * row[accuracy_key] for row in rows] + [100 * threshold]
    overview_ax.set_xlim(max(0.0, min(overview_x_values) - 3.0), min(100.5, max(overview_x_values) + 3.0))
    overview_ax.set_title(f"All {len(rows)} policies\n(explicit outlier overview)", fontsize=10)
    if excluded:
        excluded_text = "Main-panel exclusions:\n" + "\n".join(
            f"{row['config_id']}: {100 * row[accuracy_key]:.0f}%, {row[perf_key]:.2f} t/s/u" for row in excluded
        )
        overview_ax.text(
            0.04,
            0.04,
            excluded_text,
            transform=overview_ax.transAxes,
            fontsize=7.2,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#999", "alpha": 0.92},
            zorder=7,
        )

    ax.set_xlabel(f"Full-model {accuracy_key.replace('top', 'top-')} accuracy (%)")
    ax.set_ylabel("Trace-verified teacher-forcing decode (tokens/s/user)")
    overview_ax.set_xlabel("Accuracy (%)")
    ax.set_title(
        f"Gemma 4 31B datatype sweep: {accuracy_key.replace('top', 'top-')} accuracy / performance\n"
        f"Decision-region zoom ({len(main_rows)}/{len(rows)} policies; overview contains every point)"
    )
    ax.grid(True, alpha=0.22)
    overview_ax.grid(True, alpha=0.22)
    handles, labels = overview_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "evaluated_count": len(rows),
        "main_count": len(main_rows),
        "overview_count": len(rows),
        "frontier_ids": [row["config_id"] for row in frontier],
        "selected_id": selected,
        "annotated_ids": annotated,
        "excluded_ids": [row["config_id"] for row in excluded],
        "threshold": threshold,
    }


def run(args: argparse.Namespace) -> None:
    files = sorted(args.candidates.glob("*.json"))
    rows = [_row(json.loads(path.read_text(encoding="utf-8"))) for path in files]
    if not rows:
        raise ValueError(f"no candidate JSON files found under {args.candidates}")
    selected = next((row for row in rows if row["config_id"] == args.selected), None)
    if selected is None:
        raise ValueError(f"selected config {args.selected!r} is not present")
    passing = [row for row in rows if row["pass"]]
    if not passing:
        raise ValueError("no evaluated full-model config passes the accuracy thresholds")
    fastest = max(passing, key=lambda row: row["trace_verified_teacher_forcing_decode_t/s/u"])
    if fastest["config_id"] != args.selected:
        raise ValueError(f"selected config is not fastest passing: {fastest['config_id']} is faster")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "google/gemma-4-31B",
        "thresholds": {"top1": args.min_top1, "top5": args.min_top5, "top100": args.min_top100},
        "selection_metric": "trace_verified_teacher_forcing_decode_t/s/u",
        "selected_config_id": args.selected,
        "results": rows,
    }
    (args.output_dir / "sweep_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_rows = []
    for row in rows:
        flat = dict(row)
        for key in (
            "weight_groups",
            "layer_exceptions",
            "compute_fidelities",
            "ccl_dtype",
            "runtime_consumption",
            "trace_counters",
        ):
            flat[key] = json.dumps(flat[key], sort_keys=True)
        csv_rows.append(flat)
    with (args.output_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(rows, args.selected, "top1", args.min_top1, args.output_dir / "top1_perf_pareto.png")
    _plot(rows, args.selected, "top5", args.min_top5, args.output_dir / "top5_perf_pareto.png")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--min-top1", type=float, default=0.90)
    parser.add_argument("--min-top5", type=float, default=0.98)
    parser.add_argument("--min-top100", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
