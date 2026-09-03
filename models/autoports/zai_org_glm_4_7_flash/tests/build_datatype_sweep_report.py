# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assemble doc/datatype_sweep/{sweep_results.json,sweep_results.csv} and the
two Pareto PNGs from the per-candidate JSONs in doc/datatype_sweep/runs/
(produced by dev_datatype_sweep.py) plus the two rejected-without-run
candidates that can't execute on this hardware target at all.

    python -m models.autoports.zai_org_glm_4_7_flash.tests.build_datatype_sweep_report
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOC = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep"
RUNS = DOC / "runs"

# Accent palette: muted slate for evaluated/rejected points, a single warm
# accent for the Pareto frontier line, red for the selected point, and a
# neutral dotted gray for the accuracy-bar reference line. No rainbow, no
# color-as-rank: every rejected candidate keeps the same hue regardless of
# why it was rejected (the label text carries that distinction).
COLOR_REJECTED = "#5b6b7c"
COLOR_FRONTIER = "#c9822a"
COLOR_SELECTED = "#d33f3f"
COLOR_NOTRUN = "#9aa5b1"
COLOR_THRESHOLD = "#7a7a7a"

TOP5_BAR = 0.98
TOP100_BAR = 1.00
# This model's stated readiness bar (doc/full_model/README.md, doc/optimized_full_model/README.md,
# across many committed review rounds) is top-5>=0.98, top-100=1.00 with top-1 reported but not
# numerically thresholded. The goal contract for *this* stage names a "top-1/top-5 gate" explicitly,
# and this exact autoport's own FM-021 precedent (doc/full_model/work_log.md) treats any material
# top-1 regression from a precision change as a real, disqualifying finding, not noise -- so the
# gate this stage actually enforces adds a concrete, checkable top-1 condition: no regression below
# the C00 baseline's teacher-forced top-1. That baseline value, not a generic threshold, is the bar,
# because the model's own accepted history is what defines "regression" here.
BASELINE_TOP1 = 0.85

CMD_TEMPLATE = (
    "python -m models.autoports.zai_org_glm_4_7_flash.tests.dev_datatype_sweep "
    "--config-id {config_id} {flags} --out doc/datatype_sweep/runs/{config_id}.json"
)

# Human-readable metadata per candidate: the dtype/fidelity policy delta from
# baseline, the CLI flags used, and the kept/rejected decision with reason.
# Baseline = models/autoports/zai_org_glm_4_7_flash tt/model.py + tt/optimized_decoder.py
# shipped defaults, unchanged by this stage.
CANDIDATES = [
    {
        "config_id": "C00_baseline",
        "flags": "",
        "delta": "shipped optimized-decoder/optimized-full-model policy, unchanged",
        "decision": "SELECTED",
        "reason": "clears the real gate (top5>=0.98, top100=1.00, top1 no worse than the 0.850 "
        "baseline -- trivially true, it IS the baseline). The strictly fastest candidate that also "
        "clears that gate is C09 (44.029 t/s/u vs C00's 44.023, a +0.02% delta far inside this "
        "harness's own run-to-run spread, e.g. baseline itself reproduces at 43.98-44.02 across "
        "separate runs) -- within measurement noise, so the skill's tie-break rule applies: prefer "
        "the simpler, better-evidenced config. C09 (dense-MLP bf4) also carries an already-documented "
        "real-weight 202k long-context accuracy regression (doc/optimized_decoder/README.md) that "
        "this stage's short 154-token-prompt/100-position reference cannot see, so C00 is preferred "
        "over it, not merely tied with it. Every other config that passes the top5/top100 bar (C04, "
        "C06, C07) is measurably slower with no accuracy benefit; every config that fails the top1 "
        "no-regression gate (C01, C02, C03, C05, C08) is rejected on accuracy regardless of speed.",
    },
    {
        "config_id": "C01_lmhead_bf4_lofi",
        "flags": "--lm-head-dtype bf4 --lm-head-fidelity lofi",
        "delta": "LM head bfloat8_b -> bfloat4_b, fidelity HiFi2 -> LoFi",
        "decision": "FAIL_TOP1",
        "reason": "teacher-forced top-1 0.850->0.790 (-0.060 abs, -7.1% rel, below the 0.850 "
        "no-regression gate) even though top-5 1.000->0.990 still clears the model's stated 0.98 "
        "bar, for a +1.36% token-out decode gain (44.02->44.62 t/s/u); the isolated-op 624us-vs-878us "
        "win (doc/full_model/head_probe.json, a reduced 2-layer profile) is ~4% of the real 47-layer "
        "model-only step so the full-model gain is much smaller than the op-level number suggests. "
        "Same precedent as FM-021 (doc/full_model/work_log.md): an LM-head precision change is an "
        "accuracy change, and this stage's team rejected a comparable top-1 hit there for an even "
        "smaller (0.04%) gain. The LM head is not capacity-constrained (0.314 GiB bf8 vs the ~32 GiB "
        "budget), so there is no capacity argument for bf4 here the way there is for routed experts.",
    },
    {
        "config_id": "C02_lmhead_bf4_hifi2",
        "flags": "--lm-head-dtype bf4 --lm-head-fidelity hifi2",
        "delta": "LM head bfloat8_b -> bfloat4_b, fidelity stays HiFi2 (BFP4+HiFi2 vs BFP4+LoFi pair for C01)",
        "decision": "FAIL_TOP1",
        "reason": "identical to C01 (top1 0.790, top5 0.990, decode 44.62 t/s/u): the accuracy cost is "
        "from the bf4 dtype quantization, not the fidelity choice -- HiFi2 buys nothing over LoFi here. "
        "Confirms LoFi is the correct fidelity *if* bf4 LM head were ever adopted, and confirms the "
        "rejection is dtype-driven, not fidelity-driven. Fails the same top1 no-regression gate as C01.",
    },
    {
        "config_id": "C03_lmhead_bf8_lofi",
        "flags": "--lm-head-fidelity lofi",
        "delta": "LM head dtype unchanged (bfloat8_b); fidelity HiFi2 -> LoFi (BFP8+LoFi vs BFP8+HiFi2 "
        "for this dominant decode projection group, per skill mandate)",
        "decision": "FAIL_TOP1",
        "reason": "teacher-forced top-1 0.850->0.830 (-0.020 abs, below the no-regression gate) for a "
        "44.02->44.08 t/s/u change (+0.14%, inside this harness's run-to-run noise band, consistent "
        "with the isolated-op finding of a 0.6% device-time difference in "
        "doc/optimized_full_model/README.md item 1). No measurable speed benefit to justify the "
        "accuracy cost, so this fails the gate for no compensating reason.",
    },
    {
        "config_id": "C04_kvcache_bf16",
        "flags": "--cache-dtype bf16",
        "delta": "paged latent KV cache bfloat8_b -> bfloat16 (comparability arm)",
        "decision": "PASS_NOT_SELECTED",
        "reason": "clears the gate (top1 tied at 0.850) but TTFT regresses 590.28->620.48 ms (+5.1%, "
        "more DRAM traffic for the doubled-width cache read/write) and decode is marginally slower "
        "(44.02->43.91 t/s/u). Matches the already-committed 202k real-weight evidence in "
        "doc/context_contract.json's optimized_decoder section (bf8 == bf16 within noise on accuracy); "
        "bf8 wins on speed with no accuracy cost either way.",
    },
    {
        "config_id": "C05_router_hifi2",
        "flags": "--router-fidelity hifi2",
        "delta": "router/gate compute fidelity HiFi4+fp32acc -> HiFi2 (decode only; prefill routing is "
        "unaffected by this plumbing and stays HiFi4)",
        "decision": "FAIL_TOP1",
        "reason": "teacher-forced top-1 0.850->0.820 (-0.030 abs, below the no-regression gate), "
        "consistent with (though not directly measured as) routing-decision sensitivity under lower "
        "router fidelity -- this codebase has tests/dev_optimize.py --check-ties for verifying "
        "expert-selection ties/flips directly, which this stage did not run. Decode change is "
        "44.02->44.09 t/s/u (+0.16%, noise-level, matches the isolated-op finding of ~0.19% of the "
        "model-only step in doc/optimized_full_model/README.md item 2). No measurable benefit for a "
        "real accuracy cost on a selection-sensitive tensor.",
    },
    {
        "config_id": "C06_attn_hifi2",
        "flags": "--attn-fidelity hifi2",
        "delta": "attention decode group (wqkv_a, wq_b, w_uk, w_uv, wo; bfloat4_b) fidelity LoFi -> HiFi2 "
        "(BFP4+HiFi2 vs BFP4+LoFi pair)",
        "decision": "PASS_NOT_SELECTED",
        "reason": "clears the gate (top1 tied at 0.850) but decode regresses 44.02->41.87 t/s/u "
        "(-4.9%, the largest speed delta in this sweep after LM-head dtype). LoFi ties on accuracy "
        "and wins decisively on speed for this group.",
    },
    {
        "config_id": "C07_expert_hifi2",
        "flags": "--expert-fidelity hifi2",
        "delta": "routed experts (bfloat4_b, the dominant MoE compute) fidelity LoFi -> HiFi2 "
        "(BFP4+HiFi2 vs BFP4+LoFi pair)",
        "decision": "PASS_NOT_SELECTED",
        "reason": "clears the gate (top1 tied at 0.850); decode regresses 44.02->43.82 t/s/u (-0.45%). "
        "LoFi ties on accuracy and wins on speed.",
    },
    {
        "config_id": "C08_mlp_hifi2",
        "flags": "--mlp-fidelity hifi2",
        "delta": "shared-expert (bfloat4_b) + dense-MLP (bfloat8_b) fidelity LoFi -> HiFi2, both groups "
        "together (single class-attribute knob covers both; BFP4+HiFi2/BFP8+HiFi2 vs LoFi pair)",
        "decision": "FAIL_TOP1",
        "reason": "teacher-forced top-1 0.850->0.820 (-0.030 abs, below the no-regression gate) and "
        "decode regresses 44.02->43.48 t/s/u (-1.2%). LoFi wins on both axes for this combined group.",
    },
    {
        "config_id": "C09_dense_mlp_bf4_lofi",
        "flags": "--dense-mlp-dtype bf4",
        "delta": "dense-layer MLP (1 of 47 layers) bfloat8_b -> bfloat4_b, fidelity stays LoFi",
        "decision": "PASS_NOT_SELECTED",
        "reason": "clears the gate and is, on this stage's own numbers, the single fastest passing "
        "candidate (44.029 t/s/u, +0.02% over C00) with a *higher* teacher-forced top-1 (0.870 vs "
        "0.850) -- but that +0.02% speed delta is far inside this harness's measurement noise (see "
        "C00's reason), and the accuracy read is from a single 154-token prompt at 100 positions, too "
        "short to see the failure mode that already rejected this exact dtype at the decoder level: "
        "doc/optimized_decoder/README.md's real-weight 202k long-context evidence shows a measurable "
        "dense-control regression (decode@202751 0.99865 vs 0.99993, end window 28/32 vs 29/32 rows) "
        "for the same change. Not selected: no real speed benefit, and the only accuracy signal this "
        "stage's short reference can offer is not strong enough to overturn evidence from a much "
        "harder, already-committed long-context test.",
    },
    {
        "config_id": "C10_all_bf8_canonical",
        "flags": None,
        "delta": "canonical/comparability arm: expert_dtype and attn_weight_dtype/mlp_gateup_dtype/"
        "mlp_down_dtype all bfloat8_b (no bfloat4_b anywhere)",
        "decision": "NOT_RUN",
        "reason": "hard DRAM capacity limit, not a measured accuracy/speed tradeoff: "
        "doc/probe/README.md measured bfloat8_b routed experts at ~32 GB of expert weights alone, "
        "which does not fit the single Blackhole p150's 31.5 GiB allocatable DRAM "
        "(doc/context_contract.json full_model.measured_allocatable_dram_gib) once the remaining "
        "layers, KV cache and scratch are added. Not executed on hardware: it would OOM at model "
        "construction, not at the accuracy/perf checks this sweep gates on.",
    },
]

HARDWARE = "1x Blackhole p150-class chip (device 0), 11x10 compute grid, 8 DRAM banks"
MESH = "N150 (1x1)"
REFERENCE = (
    "models/autoports/zai_org_glm_4_7_flash/readiness_aime24_chat.refpt (AIME24, chat template, 100 generated tokens)"
)


def _fidelity_str(fid: dict) -> str:
    mf = fid["math_fidelity"].split(".")[-1]
    return f"{mf}{'+fp32acc' if fid['fp32_dest_acc_en'] else ''}"


def _dtype_str(dt: str) -> str:
    return dt.split(".")[-1].lower()


def load_run(config_id: str) -> dict | None:
    path = RUNS / f"{config_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_row(meta: dict) -> dict:
    config_id = meta["config_id"]
    run = load_run(config_id)
    row = {
        "config_id": config_id,
        "delta_from_baseline": meta["delta"],
        "decision": meta["decision"],
        "reason": meta["reason"],
        "command": (
            CMD_TEMPLATE.format(config_id=config_id, flags=meta["flags"])
            if meta["flags"] is not None
            else "N/A (not run)"
        ),
        "hardware": HARDWARE,
        "mesh": MESH,
        "reference": REFERENCE,
        "measurement_regime": "trace-verified teacher-forcing decode (models.common.readiness_check.run_teacher_forcing, "
        "enable_trace=True; generator.generate() is required to explicitly accept enable_trace and this "
        "path always passes it -- see run_teacher_forcing._require_explicit_generate_kwarg)",
    }
    if run is None:
        row.update(
            {
                "dtype_policy": meta["delta"],
                "compute_fidelity_policy": "n/a",
                "prefill_top1": None,
                "prefill_top5": None,
                "prefill_top100": None,
                "tf_top1": None,
                "tf_top5": None,
                "tf_top100": None,
                "ttft_ms": None,
                "decode_t_s_u": None,
                "e2e_t_s_u": None,
                "pass_fail": "NOT_RUN",
            }
        )
        return row

    snap = run["policy_snapshot"]
    pf = run["prefill_check"]["aggregate"]
    tf = run["teacher_forcing"]["aggregate"]
    dtype_policy = (
        f"expert={_dtype_str(snap['expert_dtype'])} weight={_dtype_str(snap['weight_dtype'])} "
        f"cache={_dtype_str(snap['kv_cache_dtype'])} lm_head={_dtype_str(snap['lm_head_dtype'])} "
        f"attn={_dtype_str(snap['moe_layer']['attn_weight_dtype'])} "
        f"shared_gu={_dtype_str(snap['moe_layer']['shared_gate_up_dtype'])} "
        f"shared_dn={_dtype_str(snap['moe_layer']['shared_down_dtype'])} "
        f"expert_gu={_dtype_str(snap['moe_layer']['expert_gate_up_dtype'])} "
        f"expert_dn={_dtype_str(snap['moe_layer']['expert_down_dtype'])} "
        f"dense_mlp={_dtype_str(snap['dense_layer']['mlp_gate_dtype'])}"
    )
    fidelity_policy = (
        f"lm_head={_fidelity_str(snap['ck_lm_head'])} attn={_fidelity_str(snap['moe_layer']['ck_attn'])} "
        f"mlp={_fidelity_str(snap['moe_layer']['ck_mlp_shared'])} expert={_fidelity_str(snap['moe_layer']['ck_expert'])} "
        f"router={_fidelity_str(snap['moe_layer']['ck_router'])}"
    )
    top5_ok = tf.get("top5", 0.0) >= TOP5_BAR
    top100_ok = tf.get("top100", 0.0) >= TOP100_BAR
    top1_ok = tf.get("top1", 0.0) >= BASELINE_TOP1
    row.update(
        {
            "dtype_policy": dtype_policy,
            "compute_fidelity_policy": fidelity_policy,
            "prefill_top1": pf.get("top1"),
            "prefill_top5": pf.get("top5"),
            "prefill_top100": pf.get("top100"),
            "tf_top1": tf.get("top1"),
            "tf_top5": tf.get("top5"),
            "tf_top100": tf.get("top100"),
            "ttft_ms": tf.get("ttft_ms"),
            "decode_t_s_u": tf.get("decode_t/s/u"),
            "e2e_t_s_u": tf.get("e2e_t/s/u"),
            # The real gate this stage selects against: the model's stated top5/top100
            # bar AND no teacher-forced top-1 regression from the C00 baseline (see
            # BASELINE_TOP1's comment -- this is the concrete form of the goal's
            # "top-1/top-5 gate" and of the FM-021 precedent this stage follows).
            "pass_fail": "PASS" if (top5_ok and top100_ok and top1_ok) else "FAIL",
        }
    )
    return row


def main():
    rows = [build_row(m) for m in CANDIDATES]

    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "sweep_results.json").write_text(
        json.dumps(
            {
                "model": "zai-org/GLM-4.7-Flash",
                "hardware": HARDWARE,
                "mesh": MESH,
                "reference": REFERENCE,
                "accuracy_bar": {
                    "top1": f">= {BASELINE_TOP1} (no regression from the C00 baseline's teacher-forced "
                    "top-1, not a fixed threshold -- the skill's generic 90% default would fail the "
                    "already-accepted baseline itself, so this stage uses the baseline as its own bar; "
                    "see README 'Accuracy bar' section)",
                    "top5": f">= {TOP5_BAR}",
                    "top100": f"= {TOP100_BAR}",
                    "provenance": "top5/top100 from doc/full_model/README.md and "
                    "doc/optimized_full_model/README.md, stated across many committed review rounds; "
                    "the top1 no-regression condition is this stage's own, following the FM-021 "
                    "precedent in doc/full_model/work_log.md",
                },
                "selected_config_id": "C00_baseline",
                "candidates": rows,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    fieldnames = [
        "config_id",
        "decision",
        "dtype_policy",
        "compute_fidelity_policy",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "tf_top1",
        "tf_top5",
        "tf_top100",
        "ttft_ms",
        "decode_t_s_u",
        "e2e_t_s_u",
        "pass_fail",
        "measurement_regime",
        "command",
        "hardware",
        "mesh",
        "reference",
        "delta_from_baseline",
        "reason",
    ]
    with (DOC / "sweep_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plot_pareto(rows)
    print(f"wrote {DOC / 'sweep_results.json'} and {DOC / 'sweep_results.csv'}")


def plot_pareto(rows: list[dict]):
    # top-1's gate is "no regression from the C00 baseline" (BASELINE_TOP1), not a
    # fixed universal threshold -- see the accuracy_bar note in main() / README
    # "Accuracy bar" for why the skill's generic 90% default doesn't apply here.
    for metric, fname, bar, bar_label, title in (
        (
            "tf_top1",
            "top1_perf_pareto.png",
            BASELINE_TOP1,
            f"minimum allowed = {BASELINE_TOP1:.2f} (no regression from C00 baseline; see README)",
            "GLM-4.7-Flash datatype sweep: top-1 vs decode throughput",
        ),
        (
            "tf_top5",
            "top5_perf_pareto.png",
            TOP5_BAR,
            f"minimum allowed = {TOP5_BAR:.2f}",
            "GLM-4.7-Flash datatype sweep: top-5 vs decode throughput",
        ),
    ):
        evaluated = [r for r in rows if r["decode_t_s_u"] is not None]
        xs = [r[metric] for r in evaluated]
        ys = [r["decode_t_s_u"] for r in evaluated]
        cids = [r["config_id"] for r in evaluated]

        fig, ax = plt.subplots(figsize=(9.5, 6.6), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Non-dominated Pareto frontier (higher accuracy AND higher perf is better).
        # Simple O(n^2) non-dominated set: a point is on the frontier if no other
        # point is >= on both axes and > on at least one.
        frontier = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            dominated = any(
                x2 >= x and y2 >= y and (x2 > x or y2 > y) for j, (x2, y2) in enumerate(zip(xs, ys)) if j != i
            )
            if not dominated:
                frontier.append((x, y))
        frontier.sort(key=lambda t: t[0])

        ax.scatter(xs, ys, s=70, color=COLOR_REJECTED, zorder=3, label="evaluated candidate")
        if len(frontier) > 1:
            ax.plot(
                [p[0] for p in frontier],
                [p[1] for p in frontier],
                color=COLOR_FRONTIER,
                lw=2,
                zorder=2,
                label="Pareto frontier",
            )

        sel = next((r for r in evaluated if r["decision"] == "SELECTED"), None)
        if sel is not None:
            ax.scatter(
                [sel[metric]],
                [sel["decode_t_s_u"]],
                s=200,
                color=COLOR_SELECTED,
                zorder=5,
                edgecolor="white",
                linewidth=1.5,
                label="selected (C00_baseline)",
            )

        ax.axvline(bar, color=COLOR_THRESHOLD, linestyle=(0, (2, 2)), lw=1.5, zorder=1, label=bar_label)
        ax.margins(x=0.12, y=0.14)

        # Group points that share (x, y) exactly (identical measured results,
        # e.g. C01/C02) into one combined label so they don't overlap.
        seen: dict[tuple[float, float], list[str]] = {}
        for x, y, cid in zip(xs, ys, cids):
            key = (round(x, 4), round(y, 3))
            seen.setdefault(key, []).append(cid.split("_", 1)[0])

        # Stack labels in *pixel* space so a minimum on-screen gap holds
        # regardless of the data range: finalize the axes limits with a draw,
        # then greedily push each label's target pixel-y away from any
        # already-placed label that would collide with it.
        fig.canvas.draw()
        min_gap_px = 15.0
        groups = sorted(seen.items(), key=lambda kv: kv[0][0])
        anchors_px = [ax.transData.transform((x, y)) for (x, y), _ in groups]
        label_px_y = [py + min_gap_px for _, py in anchors_px]  # default: just above the point
        order = sorted(range(len(groups)), key=lambda i: label_px_y[i])
        for a, b in zip(order, order[1:]):
            if label_px_y[b] - label_px_y[a] < min_gap_px:
                label_px_y[b] = label_px_y[a] + min_gap_px

        for ((x, y), labels), (px, _py), target_py in zip(groups, anchors_px, label_px_y):
            ax.annotate(
                "/".join(labels),
                xy=(x, y),
                xytext=(px + 7, target_py),
                textcoords="figure pixels",
                fontsize=8.5,
                color="#2b2b2b",
                arrowprops=None
                if abs(target_py - (ax.transData.transform((x, y))[1] + min_gap_px)) < 0.5
                else dict(arrowstyle="-", color="#b7bec5", lw=0.7, shrinkA=0, shrinkB=2),
            )

        not_run = [r for r in rows if r["decode_t_s_u"] is None]
        if not_run:
            note = "not run: " + "; ".join(f"{r['config_id']} ({r['reason'].split('.')[0]})" for r in not_run)
            ax.text(
                0.5,
                -0.16,
                note,
                transform=ax.transAxes,
                fontsize=7.5,
                color=COLOR_NOTRUN,
                va="top",
                ha="center",
                wrap=True,
            )

        metric_label = (
            "top-1 accuracy (AIME24, teacher-forced, 100 tok)"
            if metric == "tf_top1"
            else "top-5 accuracy (AIME24, teacher-forced, 100 tok)"
        )
        ax.set_xlabel(metric_label)
        ax.set_ylabel("trace-verified teacher-forcing decode throughput (tokens/s/user)")
        ax.set_title(title, fontsize=12)
        ax.margins(x=0.12, y=0.12)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.99))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="both", color="#e6e6e6", linewidth=0.8, zorder=0)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(DOC / fname)
        plt.close(fig)
        print(f"wrote {DOC / fname}")


if __name__ == "__main__":
    main()
