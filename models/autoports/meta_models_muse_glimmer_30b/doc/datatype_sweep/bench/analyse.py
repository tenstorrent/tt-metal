# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect the sweep runs into results tables, a Pareto selection and two charts.

Reads every ``doc/datatype_sweep/runs/*.json`` written by ``sweep.py`` and emits:

* ``sweep_results.json`` -- one row per evaluated config, with the dtype and
  compute-fidelity policy, the accuracy triple, TTFT, the trace-verified
  teacher-forcing decode t/s/u, the measurement regime, the exact command, the
  hardware and mesh, and pass/fail against the acceptance bar;
* ``sweep_results.csv`` -- the same rows, flattened;
* ``top1_perf_pareto.png`` / ``top5_perf_pareto.png``.

Selection is mechanical: the fastest config, by trace-verified teacher-forcing
decode t/s/u, that satisfies both accuracy gates.  Ties inside the measured
spread are broken toward the simpler policy, and the tie rule is recorded in the
output rather than applied silently.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

OUT = ROOT / "doc/datatype_sweep"
RUNS = OUT / "runs"

#: ``$datatype-sweep`` default acceptance bar; no user override was given.
MIN_TOP1 = 0.90
MIN_TOP5 = 0.98
#: The readiness expectation carried forward from the optimized full model.
MIN_TOP100 = 1.00

# --- palette (validated categorical slots 1/3 + the mandated red for the pick)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
PASS_BLUE = "#2a78d6"
FAIL_AQUA = "#1baf7a"
SELECTED_RED = "#e34948"


def policy_summary(config: dict) -> dict:
    """The fields a reader needs to tell two candidates apart, flattened."""
    weights = config["weights"]
    fidelity = config["compute_fidelity"]

    def fid(phase: str) -> str:
        spec = fidelity[phase]
        by_role = spec.get("by_role") or {}
        if not by_role:
            return spec["default"]
        parts = ",".join(f"{role}={value}" for role, value in sorted(by_role.items()))
        return f"{spec['default']}({parts})"

    exceptions = config.get("layer_exceptions") or []
    return {
        "weight_attn": weights["attn_projections"]["dtype"],
        "weight_mlp_gate_up": weights["mlp_gate_up"]["dtype"],
        "weight_mlp_down": weights["mlp_down"]["dtype"],
        "weight_lm_head": weights["lm_head"]["dtype"],
        "lm_head_geometry": f"{weights['lm_head']['matmul']}/{weights['lm_head']['cores']}"
        f"/in0bw{weights['lm_head']['in0_block_w']}",
        "activation_dtype": config["activations"]["activation_dtype"],
        "residual_dtype": config["activations"]["residual_dtype"],
        "kv_cache_dtype": config["kv_cache"]["dtype"],
        "ccl_prefill_dtype": config["ccl"]["prefill_dtype"],
        "ccl_decode_dtype": config["ccl"]["decode_dtype"] or config["activations"]["activation_dtype"],
        "logits_dtype": config["logits"]["lm_head_output_dtype"],
        "decode_fidelity": fid("decode"),
        "prefill_fidelity": fid("prefill"),
        "lm_head_fidelity": fidelity["lm_head"]["fidelity"],
        "lm_head_fp32_acc": fidelity["lm_head"]["fp32_dest_acc_en"],
        "layer_exceptions": json.dumps(exceptions) if exceptions else "",
        # A candidate can differ from the baseline only in a companion setting --
        # `c19` is the baseline payload with the fractured prefill norm off -- so
        # leaving this out made two materially different configs print identical
        # policy rows in the table and the CSV.
        "decoder_overrides": json.dumps(config.get("decoder_overrides") or {}, sort_keys=True)
        if config.get("decoder_overrides")
        else "",
    }


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        config_path = REPO / run["config_path"]
        config = json.loads(config_path.read_text())
        teacher = run.get("teacher_forcing") or {}
        capacity = run.get("capacity") or {}
        row = {
            "config_id": run["config_id"],
            "description": run.get("description", ""),
            "precision_config_path": run["config_path"],
            "run_artifact": str(path.relative_to(REPO)),
            "status": run.get("status"),
            "error": run.get("error", ""),
            "policy": policy_summary(config),
            "propagation_problems": run.get("propagation_problems", []),
            "propagation_verified": run.get("status") == "ok" and not run.get("propagation_problems"),
            "accuracy": {
                "top1": teacher.get("top1"),
                "top5": teacher.get("top5"),
                "top100": teacher.get("top100"),
                "tokens": teacher.get("total_tokens"),
                "reference": run.get("reference"),
                "prefill_top1": (run.get("prefill_check") or {}).get("per_entry", [{}])[0].get("top1"),
                "prefill_top5": (run.get("prefill_check") or {}).get("per_entry", [{}])[0].get("top5"),
                "prefill_top100": (run.get("prefill_check") or {}).get("per_entry", [{}])[0].get("top100"),
                "stable_across_rounds": teacher.get("accuracy_stable"),
            },
            "performance": {
                "measurement_regime": (
                    "teacher-forcing decode, readiness run_teacher_forcing over "
                    f"{teacher.get('total_tokens')} generated tokens of {run.get('reference')}, "
                    "batch 1, prompt 204, traced decode (enable_trace=True), token restaged per step; "
                    f"median of {len(teacher.get('decode_tok_s_u_rounds') or [])} rounds"
                ),
                "teacher_forcing_decode_tok_s_u_median": teacher.get("decode_tok_s_u_median"),
                "teacher_forcing_decode_tok_s_u_max": teacher.get("decode_tok_s_u_max"),
                "teacher_forcing_decode_tok_s_u_min": teacher.get("decode_tok_s_u_min"),
                "teacher_forcing_decode_tok_s_u_rounds": teacher.get("decode_tok_s_u_rounds"),
                "teacher_forcing_traced": teacher.get("traced"),
                "trace_replays_per_round": teacher.get("trace_replays_per_round"),
                "ttft_ms_min": teacher.get("ttft_ms_min"),
                "ttft_ms_median": teacher.get("ttft_ms_median"),
                "ttft_ms_rounds": teacher.get("ttft_ms_rounds"),
                "ttft_regime": "readiness teacher-forcing prefill + first token, batch 1, prompt 204, eager prefill",
                "traced_logits_only_tok_s_u": (run.get("traced_logits_only") or {}).get("tok_s_u"),
                "traced_logits_only_rounds_ms": (run.get("traced_logits_only") or {}).get("rounds_ms"),
                "traced_logits_only_regime": (
                    "decode-trace replay alone, no sampling and no token readback, prompt 128, "
                    f"{(run.get('traced_logits_only') or {}).get('replays_per_round')} replays x 3 rounds, min"
                ),
            },
            "memory": {
                "per_device_kv_cache_bytes": capacity.get("per_device_kv_cache_bytes"),
                "per_device_total_bytes": capacity.get("per_device_total_bytes"),
                "per_device_dram_capacity_bytes": capacity.get("per_device_dram_capacity_bytes"),
                "per_device_kv_cache_bytes_per_block": capacity.get("per_device_kv_cache_bytes_per_block"),
                "supported_context": capacity.get("supported_context"),
            },
            "command": run.get("command"),
            "environment": run.get("environment"),
            "hardware": run.get("hardware"),
            "build_seconds": run.get("build_seconds"),
            "_logits_only_rounds_ms": (run.get("traced_logits_only") or {}).get("rounds_ms"),
        }
        gates = {
            "top1": row["accuracy"]["top1"] is not None and row["accuracy"]["top1"] >= MIN_TOP1,
            "top5": row["accuracy"]["top5"] is not None and row["accuracy"]["top5"] >= MIN_TOP5,
            "top100": row["accuracy"]["top100"] is not None and row["accuracy"]["top100"] >= MIN_TOP100,
            "propagation": row["propagation_verified"],
            "traced": bool(row["performance"]["teacher_forcing_traced"]),
        }
        row["gates"] = gates
        row["pass"] = all(gates.values())
        rows.append(row)
    return rows


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Non-dominated points of ``(accuracy, throughput, id)``; both maximised."""
    front = []
    for point in points:
        if not any(other[0] >= point[0] and other[1] >= point[1] and other[:2] != point[:2] for other in points):
            front.append(point)
    return sorted(front)


def cross_check_resolution(rows: list[dict]) -> float:
    """How finely the traced logits-only replay can separate two configs.

    Measured from the metric itself rather than assumed.  Each candidate's
    logits-only measurement is three 64-replay rounds, and the **third** is
    systematically 1.5-2.3 % slower than the first two in every candidate -- a
    property of sustained replay, not of the candidate -- so the metric takes the
    min and its real resolution is how well its two good rounds agree.  This
    returns the worst such agreement across all candidates, which is the smallest
    gap the metric is entitled to call a separation.
    """
    spreads = []
    for row in rows:
        rounds = sorted((row.get("_logits_only_rounds_ms") or []))
        if len(rounds) >= 2 and rounds[0] > 0:
            spreads.append(rounds[1] / rounds[0] - 1)
    return max(spreads) if spreads else 0.0


def _round_range(row: dict) -> tuple[float, float]:
    """The candidate's measured teacher-forcing spread, excluding round 0.

    Round 0 of every candidate is the first run after the 52-layer build: it pays
    program compilation and trace capture and lands 2-3 % low with a TTFT two to
    three times the steady-state one.  It is a warm-up, not a sample, and it is
    excluded from the spread for that reason -- the median over all rounds is
    unaffected by it either way at 11 rounds.
    """
    rounds = sorted((row["performance"]["teacher_forcing_decode_tok_s_u_rounds"] or [])[1:])
    if not rounds:
        median = row["performance"]["teacher_forcing_decode_tok_s_u_median"] or 0.0
        return median, median
    return rounds[0], rounds[-1]


def select(rows: list[dict], *, tie_tolerance: float) -> dict:
    """The fastest passing config, with the tie rule recorded rather than hidden.

    The ranking metric is the one ``$datatype-sweep`` mandates: median
    trace-verified teacher-forcing decode t/s/u.  Two candidates count as
    *separated* only when their measured round ranges do not overlap **and** they
    differ by more than ``tie_tolerance``; anything else is a tie, and the skill's
    tie rule then applies -- "if two configs are within measurement noise, prefer
    the simpler and safer one".  Safer is read as: does not regress full-model
    top-1 against the best top-1 in the tied set.  That ordering matters here
    because the fastest raw number and the best accuracy are not the same
    candidate, and the difference between them is inside the measurement.
    """
    passing = [r for r in rows if r["pass"]]
    if not passing:
        return {"config_id": None, "reason": "no evaluated config satisfied the accuracy gate"}
    ranked = sorted(passing, key=lambda r: -r["performance"]["teacher_forcing_decode_tok_s_u_median"])
    best = ranked[0]
    best_speed = best["performance"]["teacher_forcing_decode_tok_s_u_median"]
    best_low, best_high = _round_range(best)
    tied = [
        r
        for r in ranked
        if (best_speed - r["performance"]["teacher_forcing_decode_tok_s_u_median"]) / best_speed <= tie_tolerance
        or _round_range(r)[1] >= best_low  # overlapping measured ranges are not separated
    ]
    chosen = best
    reason = f"fastest passing config by median trace-verified teacher-forcing decode " f"({best_speed:.2f} t/s/u)"
    if len(tied) > 1:
        # A tie on the ranking metric is broken by the low-variance cross-check
        # first, and only then by simplicity.  Teacher forcing spends ~3.7 ms of
        # its ~26 ms step on host restaging, sampling and token readback, none of
        # which a dtype change touches, so a device-side win is diluted and can
        # land inside its round-to-round spread.  The traced logits-only replay
        # measures the same device work with none of that overhead, so when the
        # ranking metric cannot separate two configs, it is the evidence that can
        # -- and preferring the *faster* one there is what stops a real
        # lower-precision win being discarded as a tie.  Simplicity only decides
        # when both metrics are flat.
        baseline_row = next((r for r in rows if r["config_id"].startswith("c00")), None)
        baseline_policy = baseline_row["policy"] if baseline_row else {}

        def complexity(row: dict) -> int:
            return sum(1 for key, value in row["policy"].items() if baseline_policy.get(key) != value)

        def cross_check(row: dict) -> float:
            return row["performance"]["traced_logits_only_tok_s_u"] or 0.0

        # Safer first: inside a tie on the ranking metric, a candidate that gives
        # up measured full-model top-1 is not preferred over one that does not.
        best_top1 = max(r["accuracy"]["top1"] for r in tied)
        safe = [r for r in tied if r["accuracy"]["top1"] >= best_top1]
        dropped_for_accuracy = [
            f"{r['config_id']} ({r['performance']['teacher_forcing_decode_tok_s_u_median']:.3f} t/s/u, "
            f"top-1 {r['accuracy']['top1']:.3f})"
            for r in tied
            if r not in safe
        ]
        # The separation that matters is between the *top two* of the surviving
        # set, not the spread across all of it.  Round 3 of the stage review found
        # this measuring the latter: with a slow-but-accurate candidate in the set
        # the spread is large while the two candidates actually being chosen
        # between are indistinguishable, so the rule declared itself decisive on a
        # margin it had not looked at.
        # The cross-check separates on **its own** measured resolution, not on the
        # ranking metric's tie band -- 0.5 % would swallow real 0.4 % differences
        # in a metric whose good rounds agree to 0.04 %.  Everything the
        # cross-check does not separate from its best is then a genuine tie, and
        # simplicity decides among *those* rather than across the whole surviving
        # set (round 3 of the stage review found the earlier version applying
        # simplicity to candidates the cross-check had in fact separated).
        resolution = cross_check_resolution(safe)
        ranked_by_cross = sorted(safe, key=cross_check, reverse=True)
        cross_best = ranked_by_cross[0]
        indistinguishable = [
            r
            for r in ranked_by_cross
            if (cross_check(cross_best) - cross_check(r)) / max(cross_check(cross_best), 1e-9) <= resolution
        ]
        chosen = min(
            indistinguishable,
            key=lambda r: (complexity(r), -r["performance"]["teacher_forcing_decode_tok_s_u_median"]),
        )
        separated = [r for r in ranked_by_cross if r not in indistinguishable]
        if len(indistinguishable) == 1:
            rule = (
                f"tie broken by the traced logits-only cross-check, which separates every other "
                f"accuracy-neutral candidate from {chosen['config_id']} by more than its own measured "
                f"resolution of {resolution:.3%}: {chosen['config_id']} at {cross_check(chosen):.3f} t/s/u, "
                f"next {separated[0]['config_id']} at {cross_check(separated[0]):.3f}"
            )
        else:
            rule = (
                f"the traced logits-only cross-check separates {len(separated)} of the accuracy-neutral "
                f"candidates from the best but not "
                + ", ".join(f"{r['config_id']} ({cross_check(r):.3f})" for r in indistinguishable)
                + f", which agree to within its own measured resolution of {resolution:.3%}; among those "
                f"the decision is the simplest policy: {chosen['config_id']}"
            )
        if dropped_for_accuracy:
            rule += (
                ". Faster-by-median but not separated, and rejected for a measured full-model top-1 "
                f"regression to below {best_top1:.3f}: " + ", ".join(dropped_for_accuracy)
            )
        reason = (
            f"{len(tied)} passing configs are not separated from the fastest ({best_speed:.2f} t/s/u) "
            f"by the ranking metric -- within {tie_tolerance:.1%} of its median, or with an "
            f"overlapping measured round range, or both: "
            + ", ".join(
                f"{r['config_id']}={r['performance']['teacher_forcing_decode_tok_s_u_median']:.2f}" for r in tied
            )
            + f". {rule}"
        )
    return {
        "config_id": chosen["config_id"],
        "reason": reason,
        "tie_rule": (
            "candidates whose measured teacher-forcing round ranges overlap the fastest candidate's, "
            "or which are within the tolerance of its median, are not separated by the ranking metric. "
            "Within that set, in order: (1) prefer no regression in measured full-model top-1 -- the "
            "safer half of the skill's tie rule; (2) drop everything the traced logits-only cross-check "
            "separates from its best by more than that metric's own measured resolution, which is how "
            "well its two good rounds agree across all candidates rather than a fixed band; (3) among "
            "what remains -- genuinely indistinguishable on both metrics -- take the simplest policy, "
            "counted as departures from the carried-forward baseline."
        ),
        "tied_within_tolerance": [r["config_id"] for r in tied],
        "tied_round_ranges": {r["config_id"]: [round(v, 3) for v in _round_range(r)] for r in tied},
        "cross_check_metric": "traced logits-only decode t/s/u (decode-trace replay alone)",
        "cross_check_resolution": locals().get("resolution"),
        "cross_check_indistinguishable_from_best": [r["config_id"] for r in locals().get("indistinguishable", [])],
        "cross_check_tok_s_u": {r["config_id"]: r["performance"]["traced_logits_only_tok_s_u"] for r in tied},
        "tie_tolerance": tie_tolerance,
        "ranked": [
            {
                "config_id": r["config_id"],
                "decode_tok_s_u_median": r["performance"]["teacher_forcing_decode_tok_s_u_median"],
                "decode_tok_s_u_rounds": r["performance"]["teacher_forcing_decode_tok_s_u_rounds"],
                "traced_logits_only_tok_s_u": r["performance"]["traced_logits_only_tok_s_u"],
                "top1": r["accuracy"]["top1"],
                "top5": r["accuracy"]["top5"],
            }
            for r in ranked
        ],
        "rejected": [
            {
                "config_id": r["config_id"],
                "status": r["status"],
                "failed_gates": [name for name, ok in r["gates"].items() if not ok],
                "top1": r["accuracy"]["top1"],
                "top5": r["accuracy"]["top5"],
                "top100": r["accuracy"]["top100"],
                "decode_tok_s_u_median": r["performance"]["teacher_forcing_decode_tok_s_u_median"],
                "traced_logits_only_tok_s_u": r["performance"]["traced_logits_only_tok_s_u"],
                "error": r["error"],
            }
            for r in rows
            if not r["pass"]
        ],
    }


def write_selected_artifact(rows: list[dict], selection: dict, thresholds: dict) -> None:
    """Copy the winning candidate artifact into place with a fresh provenance block.

    The provenance used to be assembled by hand at selection time, which meant it
    described the candidate matrix *as it was then*: round 2 of the stage review
    found it still quoting a nine-config tied set after the matrix had grown. It
    is written here instead, from the same ``selection`` this run computed, so it
    cannot describe a different sweep from the one in ``sweep_results.json``.
    """
    selected_id = selection["config_id"]
    if selected_id is None:
        return
    row = next(r for r in rows if r["config_id"] == selected_id)
    config = json.loads((REPO / row["precision_config_path"]).read_text())
    config["selected"] = True
    config["provenance"] = {
        "stage": "datatype_sweep",
        "selected_by": "doc/datatype_sweep/bench/analyse.py",
        "selection_reason": selection["reason"],
        "tie_rule": selection["tie_rule"],
        "tied_within_tolerance": selection["tied_within_tolerance"],
        "sweep_results": "doc/datatype_sweep/sweep_results.json",
        "run_artifact": row["run_artifact"],
        "command": row["command"],
        "thresholds": thresholds,
        "accuracy": row["accuracy"],
        "teacher_forcing_decode_tok_s_u_median": row["performance"]["teacher_forcing_decode_tok_s_u_median"],
        "teacher_forcing_measurement_regime": row["performance"]["measurement_regime"],
        "traced_logits_only_tok_s_u": row["performance"]["traced_logits_only_tok_s_u"],
        "hardware": row["hardware"],
        "environment": row["environment"],
        "baseline_config_id": "c00-baseline-attn8-mlp4-kv8-lofi",
        "note": (
            "This file is a required input to tt/generator.py::build_generator. To return to the "
            "carried-forward optimized-full-model policy, copy "
            "doc/datatype_sweep/configs/c00-baseline-attn8-mlp4-kv8-lofi.json over it; nothing else "
            "has to change. The post-selection token-out numbers are in "
            "doc/datatype_sweep/evidence_perf.json and are the ones later reports and vLLM "
            "comparisons should use."
        ),
    }
    (OUT / "selected_precision_config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_tables(rows: list[dict], selection: dict, thresholds: dict) -> None:
    payload = {
        "schema_version": 1,
        "hf_model": "meta-models/Muse-Glimmer-30B",
        "stage": "datatype_sweep",
        "thresholds": thresholds,
        "ranking_metric": (
            "trace-verified teacher-forcing decode t/s/u (median of the per-config rounds). "
            "Eager or untraced decode is not used anywhere in this stage."
        ),
        "selected": selection,
        "configs": rows,
    }
    (OUT / "sweep_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    columns = [
        "config_id",
        "status",
        "pass",
        "propagation_verified",
        "weight_attn",
        "weight_mlp_gate_up",
        "weight_mlp_down",
        "weight_lm_head",
        "lm_head_geometry",
        "kv_cache_dtype",
        "activation_dtype",
        "residual_dtype",
        "ccl_prefill_dtype",
        "ccl_decode_dtype",
        "logits_dtype",
        "decode_fidelity",
        "prefill_fidelity",
        "lm_head_fidelity",
        "lm_head_fp32_acc",
        "layer_exceptions",
        "decoder_overrides",
        "top1",
        "top5",
        "top100",
        "tokens",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "ttft_ms_min",
        "teacher_forcing_decode_tok_s_u_median",
        "teacher_forcing_decode_tok_s_u_rounds",
        "teacher_forcing_traced",
        "traced_logits_only_tok_s_u",
        "measurement_regime",
        "per_device_kv_cache_bytes",
        "supported_context",
        "hardware",
        "mesh_shape",
        "branch",
        "commit",
        "command",
        "error",
    ]
    with (OUT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            hardware = row.get("hardware") or {}
            environment = row.get("environment") or {}
            flat = {
                "config_id": row["config_id"],
                "status": row["status"],
                "pass": row["pass"],
                "propagation_verified": row["propagation_verified"],
                **row["policy"],
                **{k: row["accuracy"].get(k) for k in ("top1", "top5", "top100", "tokens")},
                **{k: row["accuracy"].get(k) for k in ("prefill_top1", "prefill_top5", "prefill_top100")},
                "ttft_ms_min": row["performance"]["ttft_ms_min"],
                "teacher_forcing_decode_tok_s_u_median": row["performance"]["teacher_forcing_decode_tok_s_u_median"],
                "teacher_forcing_decode_tok_s_u_rounds": json.dumps(
                    row["performance"]["teacher_forcing_decode_tok_s_u_rounds"]
                ),
                "teacher_forcing_traced": row["performance"]["teacher_forcing_traced"],
                "traced_logits_only_tok_s_u": row["performance"]["traced_logits_only_tok_s_u"],
                "measurement_regime": row["performance"]["measurement_regime"],
                "per_device_kv_cache_bytes": row["memory"]["per_device_kv_cache_bytes"],
                "supported_context": row["memory"]["supported_context"],
                "hardware": f"{hardware.get('num_devices')}x{hardware.get('arch')}",
                "mesh_shape": json.dumps(hardware.get("mesh_shape")),
                "branch": environment.get("branch"),
                "commit": environment.get("commit"),
                "command": row["command"],
                "error": row["error"],
            }
            writer.writerow({key: flat.get(key) for key in columns})


def _frame(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)


def _place_labels(ax, subset, selected_id: str, *, fontsize: float = 8.5, min_gap_px: float = 15.0) -> None:
    """Direct labels, stacked in y order so a dense column stays readable.

    Every point is labelled -- there are sixteen of them and the whole question is
    which is which -- but a label is only allowed to sit on its mark when nothing
    else is within ``min_gap_px``; otherwise it slides down and grows a leader.
    """
    ax.figure.canvas.draw()
    to_display = ax.transData.transform
    to_data = ax.transData.inverted().transform
    placed: list[float] = []
    for x, y, config_id in sorted(subset, key=lambda p: -p[1]):
        px, py = to_display((x, y))
        at = py + 4.0
        for previous in placed:
            if abs(at - previous) < min_gap_px:
                at = previous - min_gap_px
        placed.append(at)
        label_x, label_y = to_data((px + 11.0, at))
        is_selected = config_id == selected_id
        if abs(at - py) > 7.0:
            ax.plot([x, label_x], [y, label_y], color=INK_MUTED, linewidth=0.7, zorder=3, solid_capstyle="round")
        ax.annotate(
            config_id.split("-", 1)[0],
            (label_x, label_y),
            fontsize=fontsize,
            va="center",
            ha="left",
            color=INK if is_selected else INK_2,
            fontweight="600" if is_selected else "normal",
            zorder=6,
        )


def plot(rows: list[dict], selection: dict, *, metric: str, threshold: float, path: pathlib.Path, label: str) -> None:
    """Two panels: every evaluated config, and a zoom on the contended band.

    One panel cannot do both jobs.  The accuracy gate has to be visible, which
    means the x-axis has to reach it; and the fidelity candidates are 25 % below
    the rest, which means the y-axis has to reach them.  On those two ranges the
    candidates the selection is actually made between -- a ~1 % band of decode
    throughput at one or two distinct accuracies -- collapse into a smear.  So the
    left panel is the whole field with the gate on it, and the right panel is that
    band at its own scale, with the same marks, the same frontier and the same
    colours.  Nothing is dropped from either.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluated = [
        r
        for r in rows
        if r["accuracy"][metric] is not None and r["performance"]["teacher_forcing_decode_tok_s_u_median"] is not None
    ]
    if not evaluated:
        raise SystemExit("nothing to plot: no config has both an accuracy and a decode measurement")

    points = [
        (r["accuracy"][metric] * 100.0, r["performance"]["teacher_forcing_decode_tok_s_u_median"], r["config_id"])
        for r in evaluated
    ]
    passing = {r["config_id"]: r["pass"] for r in evaluated}
    front = pareto_front(points)
    selected_id = selection["config_id"]

    top = max(y for _, y, _ in points)
    contended = [p for p in points if p[1] >= top * 0.985]
    show_zoom = 2 <= len(contended) < len(points)

    fig, axes = plt.subplots(1, 2 if show_zoom else 1, figsize=(13.6, 6.4) if show_zoom else (9.6, 6.4), dpi=200)
    fig.set_facecolor(SURFACE)
    ax_all = axes[0] if show_zoom else axes
    ax_zoom = axes[1] if show_zoom else None

    for ax in (ax_all, ax_zoom):
        if ax is None:
            continue
        subset = points if ax is ax_all else contended
        xs = [p[0] for p in subset] + ([threshold * 100.0] if ax is ax_all else [])
        ys = [p[1] for p in subset]
        xpad = max((max(xs) - min(xs)) * 0.10, 0.30)
        ypad = max((max(ys) - min(ys)) * 0.16, 0.03)
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad * 2.6)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

        if len(front) > 1:
            ax.plot([p[0] for p in front], [p[1] for p in front], color=INK_MUTED, linewidth=2.0, zorder=2)
        elif front:
            # A frontier can legitimately be one point -- on the top-5 chart every
            # candidate scores 1.000, so only the fastest is non-dominated.  Draw
            # it as a ring rather than leaving the legend entry pointing at
            # nothing.
            ax.plot(
                [front[0][0]],
                [front[0][1]],
                marker="o",
                markersize=20,
                markerfacecolor="none",
                markeredgecolor=INK_MUTED,
                markeredgewidth=1.6,
                linestyle="none",
                zorder=2,
            )
        for x, y, config_id in subset:
            is_selected = config_id == selected_id
            colour = SELECTED_RED if is_selected else (PASS_BLUE if passing[config_id] else FAIL_AQUA)
            ax.plot(
                [x],
                [y],
                marker="o" if passing[config_id] else "X",
                markersize=13 if is_selected else 9,
                color=colour,
                markeredgecolor=SURFACE,
                markeredgewidth=2.0,
                linestyle="none",
                zorder=5 if is_selected else 4,
            )
        ax.axvline(threshold * 100.0, color=INK_2, linestyle=(0, (2, 3)), linewidth=1.6, zorder=3)
        _frame(ax)
        # The right margin exists to hold the direct labels; an accuracy axis must
        # not advertise ticks above 100 % to buy it.
        ceiling = min(ax.get_xlim()[1], 100.0) + 1e-9
        ax.set_xticks([t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ceiling])

    if ax_zoom is None:
        _place_labels(ax_all, points, selected_id)
    else:
        # Label the outliers on the full panel and leave the contended cluster to
        # the zoom, rather than stacking sixteen labels into a column that has to
        # be read twice.
        contended_ids = {p[2] for p in contended}
        _place_labels(
            ax_all,
            [p for p in points if p[2] not in contended_ids],
            selected_id,
            min_gap_px=26.0,
        )
        _place_labels(ax_zoom, contended, selected_id, fontsize=9.0, min_gap_px=16.0)
        cx = sum(p[0] for p in contended) / len(contended)
        cy = max(p[1] for p in contended)
        ax_all.annotate(
            f"{len(contended)} candidates within 1.5 % of the fastest\n(a plotting band, not the "
            f"tie band) -- see the zoom",
            (cx, cy),
            textcoords="offset points",
            xytext=(0, 26),
            fontsize=8.5,
            color=INK_2,
            ha="center",
            va="bottom",
            zorder=6,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.7, "shrinkA": 2, "shrinkB": 6},
        )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=PASS_BLUE, markersize=9, label="passes the bar"),
        plt.Line2D([], [], marker="X", linestyle="none", color=FAIL_AQUA, markersize=9, label="fails the bar"),
        plt.Line2D([], [], marker="o", linestyle="none", color=SELECTED_RED, markersize=12, label="selected"),
        plt.Line2D(
            [],
            [],
            color=INK_MUTED,
            linewidth=2.0,
            marker="o" if len(front) == 1 else None,
            markersize=14,
            markerfacecolor="none",
            markeredgecolor=INK_MUTED,
            label="Pareto frontier" + (" (one point: every config ties on this axis)" if len(front) == 1 else ""),
        ),
        plt.Line2D(
            [], [], color=INK_2, linestyle=(0, (2, 3)), linewidth=1.6, label=f"min {label} = {threshold*100:.0f}%"
        ),
    ]
    legend = ax_all.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=1.0,
        edgecolor="#e6e5e0",
        facecolor=SURFACE,
        fontsize=9,
        labelcolor=INK_2,
    )
    legend.get_frame().set_linewidth(0.8)

    ax_all.set_ylabel("trace-verified teacher-forcing decode (t/s/u)", color=INK_2, fontsize=10, labelpad=8)
    zoom_caption = (
        f"zoom: the contended band (the {threshold*100:.0f} % gate is off-scale left; every point here clears it)"
    )
    for ax, caption in ((ax_all, "every evaluated config"), (ax_zoom, zoom_caption)):
        if ax is None:
            continue
        ax.set_xlabel(f"full-model {label} accuracy (%)", color=INK_2, fontsize=10, labelpad=8)
        ax.set_title(caption, color=INK_2, fontsize=10, loc="left", pad=8)

    fig.suptitle(
        f"Precision candidates: {label} against traced decode throughput",
        color=INK,
        fontsize=13.5,
        fontweight="600",
        x=0.008,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.008,
        0.925,
        "meta-models/Muse-Glimmer-30B - 52-layer full model on 4x Blackhole (1x4 mesh)\n"
        "AIME24 chat reference, 100 generated tokens, batch 1",
        color=INK_2,
        fontsize=9,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tie-tolerance", type=float, default=0.005, help="fractional decode tie band")
    parser.add_argument(
        "--keep-selected-artifact",
        action="store_true",
        help="do not rewrite selected_precision_config.json (for inspecting a what-if selection)",
    )
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"no run artifacts under {RUNS}")
    thresholds = {
        "top1_min": MIN_TOP1,
        "top5_min": MIN_TOP5,
        "top100_min": MIN_TOP100,
        "source": (
            "$datatype-sweep defaults (top-1 >= 90 %, top-5 >= 98 %); top-100 held at the readiness "
            "expectation the optimized full model already meets (1.000). No user override was given."
        ),
    }
    selection = select(rows, tie_tolerance=args.tie_tolerance)
    write_tables(rows, selection, thresholds)
    if not args.keep_selected_artifact:
        write_selected_artifact(rows, selection, thresholds)
    plot(rows, selection, metric="top1", threshold=MIN_TOP1, path=OUT / "top1_perf_pareto.png", label="top-1")
    plot(rows, selection, metric="top5", threshold=MIN_TOP5, path=OUT / "top5_perf_pareto.png", label="top-5")
    print(json.dumps(selection, indent=2))
    for row in rows:
        print(
            f"{row['config_id']:<34} pass={row['pass']!s:<5} "
            f"top1={row['accuracy']['top1']} top5={row['accuracy']['top5']} "
            f"decode={row['performance']['teacher_forcing_decode_tok_s_u_median']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
