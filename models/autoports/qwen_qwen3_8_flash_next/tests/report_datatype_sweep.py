# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive sweep tables and pyplot Pareto charts from immutable runner records."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def wilson(matches, total):
    p, z = matches / total, 1.96
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - half, center + half]


def p1_decision(config_id, decisions):
    """Unknown is not a numerical failure and can never establish eligibility."""
    decision = decisions.get(config_id)
    if decision is None:
        return {"passed": None, "status": "not_scored", "reason": "P1 has not been scored"}
    result = dict(decision)
    if "passed" not in result or (result["passed"] is not None and type(result["passed"]) is not bool):
        raise ValueError(f"P1 decision {config_id} requires boolean or null passed")
    if not result.get("reason"):
        raise ValueError(f"P1 decision {config_id} requires a reason")
    result.setdefault(
        "status", "not_scored" if result["passed"] is None else ("pass" if result["passed"] else "rejected")
    )
    if result["passed"] is None and result["status"] not in ("not_scored", "not_scored_dominated"):
        raise ValueError(f"Unscored P1 decision {config_id} has an incompatible status")
    if result["passed"] is True and result["status"] not in ("pass", "pass_with_required_fit_limitation"):
        raise ValueError(f"Passing P1 decision {config_id} has an incompatible status")
    if result["passed"] is False and result["status"] not in ("rejected", "fail"):
        raise ValueError(f"Failed P1 decision {config_id} has an incompatible status")
    if result["passed"] is not None and not result.get("evidence"):
        raise ValueError(f"Scored P1 decision {config_id} requires evidence")
    return result


def capability_decision(config_id, decisions, base_dir):
    """Preserve unknown decisions; structured failed gates still reject."""
    decision = decisions.get(config_id)
    if decision is None:
        return {"passed": None, "reason": "Capability preservation has not been evaluated", "status": "not_evaluated"}
    if "passed" not in decision or (decision["passed"] is not None and type(decision["passed"]) is not bool):
        raise ValueError(f"Capability decision {config_id} requires boolean or null passed")
    if not decision.get("reason") or (decision["passed"] is not None and not decision.get("evidence")):
        raise ValueError(f"Capability decision {config_id} requires reason and evidence for scored outcomes")
    result = dict(decision)
    result["checked_gates"] = []
    for check in decision.get("checks", []):
        path = Path(check["artifact"])
        if not path.is_absolute():
            path = base_dir / path
        value = json.loads(path.read_text())
        for key in check["field"].split("."):
            value = value[key]
        if type(value) is not bool:
            raise ValueError(f"Capability gate {path}:{check['field']} must be boolean")
        result["checked_gates"].append({"artifact": str(path), "field": check["field"], "passed": value})
    result["passed"] = (
        False if any(check["passed"] is False for check in result["checked_gates"]) else decision["passed"]
    )
    result["status"] = {True: "pass", False: "rejected", None: "not_evaluated"}[result["passed"]]
    return result


def numerical_accuracy(runs, baseline_runs):
    """Score numerical rows independently of a later runtime/capture failure."""
    expected = {(run["reference"], run["runner"]) for run in baseline_runs if run.get("rows")}
    covered = {(run["reference"], run["runner"]) for run in runs if run.get("rows")}
    expected_totals = {}
    for run in baseline_runs:
        if run.get("rows"):
            key = (run["reference"], run["runner"])
            totals = tuple(row["total"] for row in run["rows"])
            if key in expected_totals and expected_totals[key] != totals:
                raise ValueError("Repeated baseline runs have incompatible scored token counts")
            expected_totals[key] = totals
    failures = []
    for run in runs:
        key = (run["reference"], run["runner"])
        if not run.get("rows"):
            failures.append({"reference": key[0], "runner": key[1], "reason": "no scored rows"})
        if key in expected_totals and tuple(row["total"] for row in run.get("rows", [])) != expected_totals[key]:
            failures.append(
                {"reference": key[0], "runner": key[1], "reason": "scored token counts differ from baseline"}
            )
        for index, row in enumerate(run.get("rows", [])):
            total = row.get("total", 0)
            passed = total > 0
            for metric, threshold in (("top1", 0.90), ("top5", 0.98), ("top100", 1.0)):
                count = row.get("matches_" + metric, -1)
                fraction = count / total if total > 0 else float("nan")
                passed = passed and 0 <= count <= total and fraction >= threshold
                passed = passed and math.isclose(fraction, row.get(metric, float("nan")), abs_tol=1e-12)
            if not passed:
                failures.append(
                    {
                        "reference": key[0],
                        "runner": key[1],
                        "row": index,
                        "reason": "accuracy threshold or scored-count consistency failed",
                    }
                )
    coverage = bool(expected) and covered == expected
    return {
        "passed": coverage and not failures,
        "coverage_complete": coverage,
        "missing_reference_runners": [list(key) for key in sorted(expected - covered)],
        "unexpected_reference_runners": [list(key) for key in sorted(covered - expected)],
        "row_failures": failures,
    }


def acceptance_status(*, accuracy, execution, p1, traced, capability):
    """Keep measured failures separate from gates that have not been evaluated."""
    gates = {
        "accuracy": accuracy,
        "execution": execution,
        "p1": p1,
        "untraced": traced,
        "capability": capability["passed"],
    }
    if any(value is not None and type(value) is not bool for value in gates.values()):
        raise ValueError("Gate outcomes must be boolean or null")
    failures = [name for name, passed in gates.items() if passed is False]
    pending = [name for name, passed in gates.items() if passed is None]
    if failures:
        status = (
            "fail_capability" if "capability" in failures else ("fail_execution" if "execution" in failures else "fail")
        )
    elif pending:
        status = "pending_p1" if pending == ["p1"] else "pending"
    else:
        status = "pass"
    return status, failures, pending


def validate_selection(records, selected_id, spread):
    """Unscored contenders cannot silently disappear from the fastest-policy proof."""
    if not math.isfinite(spread) or spread < 0:
        raise ValueError("Baseline spread must be finite and nonnegative")
    by_id = {row["config_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Aggregate compatible measurement inputs before selecting one row per config ID")
    selected = by_id.get(selected_id)
    if selected is None or not selected["eligible"]:
        raise ValueError("Selection must explicitly pass accuracy, execution, P1, trace and capability gates")
    eligible = [row for row in records if row["eligible"]]
    for row in records:
        speed = row["teacher_forcing_decode_t_s_u"]
        if row["trace_verified"] and (speed is None or not math.isfinite(speed) or speed <= 0):
            raise ValueError(f"Invalid traced speed for {row['config_id']}")
    fastest = max(eligible, key=lambda row: row["teacher_forcing_decode_t_s_u"])
    if selected["teacher_forcing_decode_t_s_u"] < fastest["teacher_forcing_decode_t_s_u"] - spread:
        raise ValueError("Selected config is slower than an eligible candidate beyond measured baseline noise")
    for row in records:
        decision = row["p1"]
        claimed_pruning = decision.get("status") == "not_scored_dominated"
        # A documented incompatible capability is already an explicit rejection;
        # arbitrary P1 reasons cannot manufacture that rejection.
        unresolved = not row["failed_gates"] and bool(row["pending_gates"])
        if not claimed_pruning and not unresolved:
            continue
        if not claimed_pruning:
            raise ValueError(f"Unresolved contender {row['config_id']} needs scoring or evidenced dominance")
        if decision["passed"] is not None or not decision.get("reason") or not decision.get("evidence"):
            raise ValueError(f"Dominance pruning for {row['config_id']} requires unscored P1, reason and evidence")
        comparator = by_id.get(decision.get("dominated_by"))
        if comparator is None or not comparator["eligible"]:
            raise ValueError(f"Dominance comparator for {row['config_id']} must be an eligible measured config")
        if (
            not row["trace_verified"]
            or comparator["teacher_forcing_decode_t_s_u"] - row["teacher_forcing_decode_t_s_u"] <= spread
        ):
            raise ValueError(f"Dominance pruning for {row['config_id']} must exceed baseline noise")
        # Every cross-process repeat participates in the mean: cite all of them,
        # rather than letting a first-path alias hide uncited measurements.
        cited = {resolve_evidence(path, decision.get("evidence_base_dir")) for path in decision["evidence"]}
        required = {
            str(Path(path).resolve())
            for item in (row, comparator)
            for path in item.get("evidence_paths", [item["evidence"]])
        }
        if not required <= cited:
            raise ValueError(f"Dominance pruning for {row['config_id']} must cite all measured result artifacts")
        row["p1"]["dominance_verified"] = {
            "comparator": comparator["config_id"],
            "speed_gap_t_s_u": comparator["teacher_forcing_decode_t_s_u"] - row["teacher_forcing_decode_t_s_u"],
            "baseline_spread_t_s_u": spread,
        }
        if not row["failed_gates"]:
            row["status"] = "not_selected_unscored"
    return fastest


def resolve_evidence(value, base_dir=None):
    """Resolve checkout-relative or decision-file-relative citations unambiguously."""
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return str(path.resolve())
    local, relative = path.resolve(), (Path(base_dir) / path).resolve()
    if local.exists() and relative.exists() and local != relative:
        raise ValueError(f"Ambiguous evidence citation {value}; use an absolute path")
    return str(relative if relative.exists() else local)


def measurement_groups(paths):
    """Group genuine repeated inputs without concealing failures or changing regimes.

    A singleton keeps historical behavior (including failed-attempt reporting).
    Repeated inputs require matching policy, environment, geometry and coverage;
    all processes and individual measurement rows must have completed cleanly.
    """
    groups, seen = {}, set()

    def coverage(raw):
        result = {}
        for run in raw.get("runs", []):
            identity = (run["reference"], run["runner"])
            counts = tuple((row["total"], row.get("k")) for row in run["rows"])
            if identity in result and result[identity] != counts:
                raise ValueError("Repeated rows have inconsistent reference token counts")
            result[identity] = counts
        return result

    def clean_repeat(raw, path, expected):
        if raw.get("completed") is not True or raw.get("passed") is not True or raw.get("error"):
            raise ValueError(f"Cannot aggregate failed/incomplete measurement {path}")
        sidecar = path.with_suffix(".provenance.json")
        if sidecar.is_file():
            provenance = json.loads(sidecar.read_text())
            if ("exit_code" in provenance and provenance["exit_code"] != 0) or (
                "status" in provenance and provenance["status"] != "finished"
            ):
                raise ValueError(f"Cannot aggregate unsuccessful or unfinished process provenance: {sidecar}")
        if "serial" not in raw.get("measurement_regime", "").lower():
            raise ValueError(f"Repeated measurements must declare a serial regime: {path}")
        for field in ("hardware", "mesh", "environment", "measurement_regime", "capacity", "build_kwargs"):
            if field not in raw:
                raise ValueError(f"Repeated measurement missing {field}: {path}")
        runs = raw.get("runs", [])
        if not runs or not expected:
            raise ValueError(f"Repeated measurement has no complete reference coverage: {path}")
        by_repeat = {}
        for run in runs:
            if run.get("passed") is not True or run.get("trace_verified") is not True or not run.get("rows"):
                raise ValueError(f"Cannot aggregate failed or unverified individual measurement: {path}")
            identity = (run["reference"], run["runner"])
            entries = by_repeat.setdefault(run.get("repeat", 0), set())
            if identity in entries:
                raise ValueError(f"Duplicate reference/path within one repeat: {path}")
            entries.add(identity)
            if run["runner"] == "run_teacher_forcing":
                counters = run.get("counter_delta", {})
                if (
                    counters.get("captures") != 0
                    or counters.get("releases") != 0
                    or counters.get("model_replays", 0) <= 0
                ):
                    raise ValueError(f"Repeated teacher forcing lacks clean model replay evidence: {path}")
                for row in run["rows"]:
                    speed = row.get("decode_t/s/u")
                    if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
                        raise ValueError(f"Invalid repeated teacher-forcing speed: {path}")
        if any(entries != set(expected) for entries in by_repeat.values()):
            raise ValueError(f"Partial reference coverage within a repeat: {path}")

    for path in paths:
        path = Path(path)
        canonical = str(path.resolve())
        if canonical in seen:
            raise ValueError(f"Duplicate input artifact (not an independent measurement): {path}")
        seen.add(canonical)
        raw = json.loads(path.read_text())
        policy = raw.get("runtime_precision", {}).get("requested")
        if policy is None:
            policy = json.loads(Path(raw["build_kwargs"]["precision_config"]).read_text())
        cid = policy["config_id"]
        groups.setdefault(cid, []).append((path, raw, policy))
    result = []
    for cid, entries in groups.items():
        first_path, first, policy = entries[0]
        if len(entries) > 1:
            expected_coverage = coverage(first)
            for entry_index, (path, raw, actual_policy) in enumerate(entries):
                if any(raw == previous[1] for previous in entries[:entry_index]):
                    raise ValueError(f"Duplicate report content is not an independent measurement: {path}")
                if actual_policy != policy:
                    raise ValueError(f"Policy mismatch across repeated config {cid}: {path}")
                for field in ("hardware", "mesh", "environment", "measurement_regime", "capacity"):
                    if raw.get(field) != first.get(field):
                        raise ValueError(f"Repeated config {cid} differs in {field}: {path}")
                build = {key: value for key, value in raw.get("build_kwargs", {}).items() if key != "precision_config"}
                first_build = {
                    key: value for key, value in first.get("build_kwargs", {}).items() if key != "precision_config"
                }
                if build != first_build:
                    raise ValueError(f"Repeated config {cid} differs in construction kwargs: {path}")
                if coverage(raw) != expected_coverage:
                    raise ValueError(f"Repeated config {cid} differs in reference coverage/counts: {path}")
                clean_repeat(raw, path, expected_coverage)
        sources, runs = [], []
        for path, raw, _ in entries:
            sidecar = path.with_suffix(".provenance.json")
            provenance = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            sources.append(
                {
                    "artifact": str(path.resolve()),
                    "command": raw["command"],
                    "git_head": raw["git_head"],
                    "build_kwargs": raw["build_kwargs"],
                    "hardware": raw["hardware"],
                    "mesh": raw["mesh"],
                    "environment": raw.get("environment"),
                    "measurement_regime": raw["measurement_regime"],
                    "capacity": raw.get("capacity"),
                    "completed": raw.get("completed"),
                    "passed": raw.get("passed"),
                    "provenance_artifact": str(sidecar.resolve()) if sidecar.is_file() else None,
                    "provenance_execution": {
                        key: provenance[key]
                        for key in ("status", "exit_code", "started_utc", "finished_utc", "elapsed_seconds", "log")
                        if key in provenance
                    },
                    "embedded_provenance": raw.get("source_provenance", raw.get("provenance")),
                }
            )
            runs.extend(
                {**run, "source_artifact": str(path.resolve()), "source_run_index": index}
                for index, run in enumerate(raw.get("runs", []))
            )
        result.append({"path": first_path, "raw": {**first, "runs": runs}, "policy": policy, "sources": sources})
    return result


def pareto_figure(records, metric, threshold, selected):
    """Plot true coordinates with collision-free short IDs and a readable key.

    Only labels move. The numbered key includes policies without valid timing,
    and IDs remain stable across the top-1 and top-5 views. This helper returns
    the figure so CPU fixtures can render without writing production results.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    point_ids = {cid: f"{index:02d}" for index, cid in enumerate(sorted(r["config_id"] for r in records), 1)}
    points = [r for r in records if r["trace_verified"] and r[metric] is not None]
    fig, (ax, key) = plt.subplots(
        1,
        2,
        figsize=(18, max(8.5, 1.8 + 0.32 * len(records))),
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.06},
        layout="constrained",
    )
    fig.set_facecolor("#fafbfe")
    ax.set_facecolor("#fafbfe")
    key.set_axis_off()
    frontier = []
    for row in sorted(points, key=lambda r: (r[metric], r["teacher_forcing_decode_t_s_u"]), reverse=True):
        if not frontier or row["teacher_forcing_decode_t_s_u"] > frontier[-1]["teacher_forcing_decode_t_s_u"]:
            frontier.append(row)
    frontier.sort(key=lambda r: r[metric])
    ax.plot(
        [100 * r[metric] for r in frontier],
        [r["teacher_forcing_decode_t_s_u"] for r in frontier],
        color="#527aa3",
        linewidth=1.6,
        marker="D",
        markersize=12,
        markerfacecolor="none",
        alpha=0.7,
        label="Measured frontier (includes rejected policies)",
    )

    def color(row):
        if row["selected"]:
            return "#d92332"
        if row["failed_gates"]:
            return "#a05b3e"
        return "#237c69" if row["status"] == "pass" else "#687786"

    # Paint the selected marker last so coincident policies cannot hide it.
    for row in sorted(points, key=lambda r: r["selected"]):
        ax.scatter(
            100 * row[metric],
            row["teacher_forcing_decode_t_s_u"],
            c=color(row),
            s=145 if row["selected"] else 55,
            edgecolors="white",
            linewidths=0.8,
            zorder=5 if row["selected"] else 4,
        )
    ax.axvline(100 * threshold, color="#574b69", linestyle=":", linewidth=1.5, label=f"Minimum: {100*threshold:g}%")
    ax.grid(alpha=0.15)
    ax.margins(x=0.10, y=0.12)
    ax.set(
        xlabel=f"Minimum {metric.replace('top', 'top-')} agreement across references and paths (%)",
        ylabel="Warmed traced teacher-forcing decode (tokens/s/user)",
    )
    legend = ax.legend(loc="lower left", frameon=True, facecolor="#fafbfe", edgecolor="none", fontsize=9)
    # A slow rejected policy can compress dozens of contenders into a few
    # pixels. Keep the complete overview and put that dense upper band in a
    # clearly labeled inset rather than stretching or jittering the data.
    detail, detail_ids = None, set()
    speeds = sorted(r["teacher_forcing_decode_t_s_u"] for r in points)
    if len(speeds) >= 10 and speeds[-1] > speeds[0]:
        split = max(range(1, len(speeds)), key=lambda i: speeds[i] - speeds[i - 1])
        if speeds[split] - speeds[split - 1] > 0.5 * (speeds[-1] - speeds[0]) and len(speeds) - split >= 10:
            clustered = [r for r in points if r["teacher_forcing_decode_t_s_u"] >= speeds[split]]
            detail_ids = {r["config_id"] for r in clustered}
            detail = ax.inset_axes([0.10, 0.30, 0.73, 0.50])
            detail.set_facecolor("#fafbfe")
            for row in sorted(clustered, key=lambda r: r["selected"]):
                detail.scatter(
                    100 * row[metric],
                    row["teacher_forcing_decode_t_s_u"],
                    c=color(row),
                    s=145 if row["selected"] else 55,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5 if row["selected"] else 4,
                )
            detail.plot(
                [100 * r[metric] for r in frontier],
                [r["teacher_forcing_decode_t_s_u"] for r in frontier],
                color="#527aa3",
                linewidth=1.6,
                marker="D",
                markersize=12,
                markerfacecolor="none",
                alpha=0.7,
            )
            xs = [100 * r[metric] for r in clustered]
            ys = [r["teacher_forcing_decode_t_s_u"] for r in clustered]
            xpad = max(0.12, (max(xs) - min(xs)) * 0.15)
            ypad = max(0.002, (max(ys) - min(ys)) * 0.15)
            detail.set_xlim(min(xs) - xpad, max(xs) + xpad)
            detail.set_ylim(min(ys) - ypad, max(ys) + ypad)
            detail.grid(alpha=0.15)
            detail.tick_params(labelsize=8)
            detail.ticklabel_format(axis="both", style="plain", useOffset=False)
            detail.set_title("Detail · clustered timings at their true coordinates", fontsize=10, pad=9)
    fig.suptitle(
        f"Qwen3.8-Flash-Next · precision / performance\nSelected: {selected} (red) · full48, TP4/EP4, batch1",
        fontsize=15,
    )
    key.text(0, 0.995, "POLICY KEY · ordered by measured speed", va="top", fontsize=11, fontweight="bold")
    key.text(
        0,
        0.968,
        "IDs match both charts; leaders move labels, never measurements.",
        va="top",
        fontsize=9,
        color="#52606d",
    )
    ordered = sorted(
        records,
        key=lambda r: (-(r["teacher_forcing_decode_t_s_u"] or 0), r["config_id"]),
    )
    step = min(0.075, 0.91 / max(1, len(ordered)))
    for index, row in enumerate(ordered):
        y = 0.925 - index * step
        label = f"{point_ids[row['config_id']]}  {row['config_id']}"
        key.text(
            0, y, label, va="top", fontsize=9.5, color=color(row), fontweight="bold" if row["selected"] else "normal"
        )
        if row["failed_gates"]:
            status = "Failed: " + ", ".join(row["failed_gates"])
        elif row["pending_gates"]:
            status = "Pending: " + ", ".join(row["pending_gates"])
        else:
            status = "Pass"
        if row["selected"]:
            status = "SELECTED · " + status
        speed = row["teacher_forcing_decode_t_s_u"]
        value = f"{100 * row[metric]:.3f}%" if row[metric] is not None else "n/a"
        timing = f"{speed:.6f} t/s/u" if row["trace_verified"] and speed is not None else "no valid traced timing"
        key.text(0.045, y - step * 0.44, f"{status} · {value} · {timing}", va="top", fontsize=8, color="#52606d")
    missing = len(records) - len(points)
    footnote = "All measured policies retain their exact coordinates; coincident points have separate ID labels."
    if missing:
        footnote += f"  {missing} attempt(s) lack valid timing; see the key and sweep_results.csv."
    fig.supxlabel(footnote, fontsize=9, color="#52606d")

    # Freeze layout before solving label positions in display pixels. Short IDs
    # are packed into free boxes near their point; leader lines disambiguate the
    # dense 100%-accuracy column without jittering or dropping any measurement.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family="DejaVu Sans", size=9)
    occupied_by_axes = {ax: [legend.get_window_extent(renderer).expanded(1.02, 1.05)]}
    if detail is not None:
        occupied_by_axes[ax].append(detail.get_window_extent(renderer).expanded(1.05, 1.1))
        occupied_by_axes[detail] = []
    for row in sorted(points, key=lambda r: (not r["selected"], -r["teacher_forcing_decode_t_s_u"], r["config_id"])):
        target_axes = detail if row["config_id"] in detail_ids else ax
        occupied = occupied_by_axes[target_axes]
        bounds = target_axes.get_window_extent(renderer)
        anchors = [
            target_axes.transData.transform((100 * r[metric], r["teacher_forcing_decode_t_s_u"])) for r in points
        ]
        label = point_ids[row["config_id"]]
        px, py = target_axes.transData.transform((100 * row[metric], row["teacher_forcing_decode_t_s_u"]))
        width, height, _ = renderer.get_text_width_height_descent(label, font, ismath=False)
        width, height = width + 10, height + 10
        candidates = []
        for radius in range(16, 241, 8):
            for degrees in (135, 45, 180, 0, 225, 315, 90, 270, 157, 23, 203, 337, 112, 68, 248, 292):
                angle = math.radians(degrees)
                x = px + radius * math.cos(angle)
                y = py + radius * math.sin(angle)
                candidates.append((x, y))
        # A larger grid is a deterministic escape hatch for very dense ties.
        candidates.extend(
            sorted(
                (
                    (x, y)
                    for x in range(int(bounds.x0 + width), int(bounds.x1 - width), max(1, int(width + 2)))
                    for y in range(int(bounds.y0 + height), int(bounds.y1 - height), max(1, int(height + 2)))
                ),
                key=lambda xy: (xy[0] - px) ** 2 + (xy[1] - py) ** 2,
            )
        )
        for x, y in candidates:
            box = Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
            if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
                continue
            if any(box.overlaps(other) for other in occupied) or any(box.contains(*anchor) for anchor in anchors):
                continue
            occupied.append(box)
            target_axes.annotate(
                label,
                (100 * row[metric], row["teacher_forcing_decode_t_s_u"]),
                xytext=target_axes.transAxes.inverted().transform((x, y)),
                textcoords="axes fraction",
                ha="center",
                va="center",
                fontsize=9,
                color=color(row),
                fontweight="bold" if row["selected"] else "normal",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "#fafbfe", "edgecolor": "none", "alpha": 0.95},
                arrowprops={"arrowstyle": "-", "color": color(row), "linewidth": 0.6, "alpha": 0.65},
                zorder=6,
            )
            break
        else:
            raise ValueError(f"No non-overlapping chart label placement for {row['config_id']}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--p1-decisions", type=Path, required=True)
    parser.add_argument(
        "--capability-decisions",
        type=Path,
        required=True,
        help="Per-config boolean/null capability decisions, reasons, evidence and optional checked JSON gates",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    p1 = json.loads(args.p1_decisions.read_text())
    capabilities = json.loads(args.capability_decisions.read_text())
    records = []
    for group in measurement_groups(args.inputs):
        path, raw, policy = group["path"], group["raw"], group["policy"]
        cid = policy["config_id"]
        runs = raw.get("runs", [])
        scored = [row for run in runs for row in run["rows"]]
        primary = [
            row
            for run in runs
            if run["runner"] == "run_teacher_forcing"
            and "aime24" in run["reference"]
            and run["trace_verified"]
            and run["counter_delta"]["captures"] == run["counter_delta"]["releases"] == 0
            for row in run["rows"]
        ]
        throughput = [r["decode_t/s/u"] for r in primary]
        capability = capability_decision(cid, capabilities, args.capability_decisions.parent)
        records.append(
            {
                "config_id": cid,
                "precision_config_path": raw["build_kwargs"]["precision_config"],
                "dtype_policy": {k: v for k, v in policy.items() if k != "compute_fidelities"},
                "compute_fidelity_policy": policy["compute_fidelities"],
                "top1": min((r["top1"] for r in scored), default=None),
                "top5": min((r["top5"] for r in scored), default=None),
                "top100": min((r["top100"] for r in scored), default=None),
                "ttft_ms": statistics.mean(r["ttft_ms"] for r in primary) if primary else None,
                "teacher_forcing_decode_t_s_u": statistics.mean(throughput) if throughput else None,
                "decode_repeat_range_t_s_u": [min(throughput), max(throughput)] if throughput else None,
                "trace_verified": bool(primary),
                "measurement_regime": raw["measurement_regime"],
                "command": raw["command"],
                "git_head": raw["git_head"],
                "hardware": raw["hardware"],
                "mesh": raw["mesh"],
                "accuracy_passed": None,
                "accuracy_evaluation": None,
                "execution_completed": raw.get("completed") is True,
                "runner_passed": raw["passed"],
                "p1": {**p1_decision(cid, p1), "evidence_base_dir": str(args.p1_decisions.parent.resolve())},
                "capability": capability,
                "capability_passed": capability["passed"],
                "failed_gates": [],
                "pending_gates": [],
                "eligible": False,
                "status": "pending",
                "selected": cid == args.selected,
                "evidence": str(path),
                "evidence_paths": [source["artifact"] for source in group["sources"]],
                "source_measurements": group["sources"],
                "measurement_input_count": len(group["sources"]),
                "aime_timed_sample_count": len(throughput),
                "error": raw.get("error"),
                "references": [
                    {**run, "top1_wilson95": [wilson(row["matches_top1"], row["total"]) for row in run["rows"]]}
                    for run in runs
                ],
            }
        )
    baseline = next(r for r in records if r["config_id"] == "inherited_e4_lofi")
    for record in records:
        accuracy = numerical_accuracy(record["references"], baseline["references"])
        record["accuracy_evaluation"] = accuracy
        record["accuracy_passed"] = accuracy["passed"]
        record["status"], record["failed_gates"], record["pending_gates"] = acceptance_status(
            accuracy=accuracy["passed"],
            execution=record["execution_completed"],
            p1=record["p1"]["passed"],
            traced=record["trace_verified"],
            capability=record["capability"],
        )
        record["eligible"] = record["status"] == "pass"
    low, high = baseline["decode_repeat_range_t_s_u"]
    spread = high - low
    fastest = validate_selection(records, args.selected, spread)
    summary = {
        "selected": args.selected,
        "capability_decisions_source": str(args.capability_decisions),
        "p1_decisions_source": str(args.p1_decisions),
        "eligibility_rule": "Numerical accuracy with full baseline reference coverage, completed execution, P1, traced performance and capability preservation must pass",
        "thresholds": {"top1": 0.90, "top5": 0.98, "top100": 1.0},
        "accuracy_scope": "minimum across every reference and prefill/teacher-forcing row",
        "ranking_metric": "warmed AIME24 chat100 traced teacher forcing; no capture/release inside timing",
        "repeat_aggregation": "One policy row; arithmetic mean of every valid serial AIME timing row across all compatible input processes, not a mean of process means",
        "baseline_repeat_spread_t_s_u": spread,
        "fastest_eligible_measured_config": fastest["config_id"],
        "tie_policy": "Inside the repeated baseline spread, prefer the simpler and safer policy",
        "results": records,
    }
    (root / "sweep_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = [k for k in records[0] if k not in ("references",)]
    with (root / "sweep_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {k: json.dumps(record[k]) if isinstance(record[k], (dict, list)) else record[k] for k in columns}
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, threshold in (("top1", 0.90), ("top5", 0.98)):
        fig = pareto_figure(records, metric, threshold, args.selected)
        fig.savefig(root / f"{metric}_perf_pareto.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
