# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pareto analysis, selection, and the two required charts.

The selection rule, in one sentence
-----------------------------------

**Rank on traced teacher-forcing decode t/s/u among configs that clear the
formal accuracy gate (top-5 >= 0.98, top-100 == 1.000), and select the fastest
one that also (a) stays within 1 top-1 point of the shipped default and (b)
beats the default by more than the measured run-to-run band; if nothing does,
select the default.**

Clauses (a) and (b) are **additional to the goal's formal gate and are a
deliberate judgment call**, called out here and in the README rather than
applied quietly. The reasoning:

*(a) the top-1 floor.* The formal gate binds only on top-5 and top-100, and both
sit at 1.000 with maximum margin, so a purely mechanical "fastest passing
config" rule is free to spend unlimited top-1. It does: ``R04_qkv_bfp4`` spends
top-1 0.990 -> 0.960 -- three points of first-token agreement -- to buy +0.33%
decode, and still passes. Stacking two or three such rows would ship a
materially worse model for well under one percent of throughput. Top-1 is the
metric that tracks what a user actually sees, so it gets a floor even though the
contract does not demand one. The floor is set at one point below the shipped
0.990 (i.e. >= 0.980), which is the resolution of this reference: 100 tokens, so
one point is one token.

*(b) the noise band.* ``probes/repeats.py`` re-runs identical configs and
measures the spread of the ranking metric. Any "win" smaller than that band is
not a win, it is the same number measured twice. Ranking strictly on the point
estimate would draw the frontier through noise.

Ties, and anything inside the band, resolve to the **default** -- a candidate
must actually beat the shipped policy to displace it, not merely equal it.

*(c) the band between candidates, added during the stage-07 review.* The band
applies to candidate-vs-candidate comparisons for exactly the reason it applies
to candidate-vs-default ones. Eligible rows within one band of the fastest are
**tied**, and the tie is broken the way the governing datatype-sweep skill says
to break it: *"If two configs are within measurement noise, prefer the simpler
and safer one."* Simplicity is counted mechanically -- how many dtype or
fidelity fields the config moves off the shipped default, with block widths
excluded because a block width is a scheduling choice and bit-identical by
construction. Top-1, then decode, then TTFT order whatever simplicity leaves
tied.

Without this the rule would rank on differences it has just finished calling
unmeasurable: the row that forced the issue, ``R26``, leads ``R25`` by 0.09% --
a quarter of the band -- while taking two attention weight tensors to bfp4 that
``R25`` leaves untouched. ``R25`` changes **no** dtype and no fidelity at all,
so the skill's tie rule picks it whether or not the accompanying one-point top-1
gap means anything. This clause was written after seeing that row, which is
disclosed rather than smoothed over; what it does not do is change any
*eligibility* verdict, only the ordering among rows already eligible.

The two charts
--------------

``top1_perf_pareto.png`` and ``top5_perf_pareto.png``: accuracy on x, traced
decode t/s/u on y, every evaluated config plotted, the Pareto frontier through
the non-dominated points, the selected config in red, and a vertical dotted line
at the minimum allowed accuracy. Both also draw the **measured noise band** as a
horizontal ribbon around the default, so a reader can see directly which
apparent gains are inside the measurement's own resolution.

Each chart's dotted line is **its own axis's threshold**: the formal gate
(top-5 >= 0.98) on the top-5 chart, and the selection rule's top-1 floor
(baseline - 0.01 = 0.980) on the top-1 chart. The two happen to be the same
number, which is why the top-1 line was originally drawn at the gate value and
captioned as non-binding -- it was right by coincidence. The floor *is*
enforced (see :func:`select`, and the ``R04_qkv_bfp4`` entry in
``selection_reasons.json``), so it is labelled as the floor.

The top-5 frontier is a single point, because every evaluated config scores
top-5 1.000 and only the fastest is non-dominated. It is drawn as a ringed
marker with its own legend entry rather than omitted, which is what a
line-only frontier would do to it. Pass/fail uses marker shape as well as
colour, so the charts never depend on colour alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent

MIN_TOP5 = 0.98
MIN_TOP100 = 1.0
#: Judgment call, not the formal gate. See the module docstring.
TOP1_FLOOR_POINTS = 0.01
BASELINE_ID = "R00_default"

# From references/palette.md.
C_PASS = "#2a78d6"
C_FAIL = "#52514e"
C_SELECTED = "#d03b3b"
C_FRONTIER = "#4a3aa7"
C_BAND = "#9ec5f4"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e3e2df"
SURFACE = "#fcfcfb"


def measured(rows):
    return [r for r in rows if r.get("status") == "ok" and r.get("decode_tps_user")]


def pareto_front(points, xkey="_x", ykey="_y"):
    front = []
    for p in points:
        ax, ay = p[xkey], p[ykey]
        if not any((q[xkey] >= ax and q[ykey] >= ay) and (q[xkey] > ax or q[ykey] > ay) for q in points if q is not p):
            front.append(p)
    return sorted(front, key=lambda p: p[xkey])


def enrich(rows, band_pct: float, baseline: dict) -> None:
    """Add the derived, first-class columns every reader needs."""
    b_top1 = baseline["top1"]
    b_dec = baseline["decode_tps_user"]
    for r in rows:
        if r.get("status") != "ok":
            r["top1_cost_points"] = None
            r["decode_gain_pct"] = None
            r["gain_exceeds_noise_band"] = None
            r["within_top1_floor"] = None
            continue
        r["top1_cost_points"] = round(b_top1 - r["top1"], 4)
        r["decode_gain_pct"] = round((r["decode_tps_user"] - b_dec) / b_dec * 100, 3)
        r["gain_exceeds_noise_band"] = bool(r["decode_gain_pct"] > band_pct)
        r["within_top1_floor"] = bool(r["top1"] >= b_top1 - TOP1_FLOOR_POINTS)
        r["noise_band_pct"] = band_pct
        r["selection_rule"] = (
            f"gate(top5>={MIN_TOP5}, top100=={MIN_TOP100}) AND top1 >= {b_top1 - TOP1_FLOOR_POINTS:.3f} "
            f"AND decode gain > {band_pct:.3f}% (measured run-to-run band); among the eligible, "
            f"rows within {band_pct:.3f}% of the fastest are tied and the tie breaks on the "
            f"simpler and safer config (fewest dtype/fidelity fields moved off the default), "
            f"then top-1, then decode, then TTFT"
        )


#: Fields whose value changes the *numerics* of the model, as opposed to how the
#: same arithmetic is scheduled. ``experts_*_in0_block_w`` is deliberately not
#: here: a block width partitions a matmul's inner dimension and is bit-identical
#: by construction, which is exactly why ``R25`` is the simplest eligible row
#: despite being two fields away from the default.
def numerical_changes(row: dict, baseline: dict) -> list[str]:
    """Which dtype/fidelity fields this config moves off the shipped default.

    This is the mechanical reading of the governing skill's tie rule -- "if two
    configs are within measurement noise, prefer the simpler and safer one".
    Fewer numerical changes is simpler (less of the model's arithmetic differs
    from the policy that has already been through five stages of validation) and
    safer (every changed field is a field whose accuracy effect rests on this
    sweep's 100-token reference alone).
    """
    a, b = baseline.get("precision_config") or {}, row.get("precision_config") or {}
    return sorted(k for k in a if (k.endswith("_dtype") or k.endswith("_fidelity")) and a[k] != b.get(k))


def select(rows, band_pct: float, baseline: dict):
    """Apply the stated rule. Returns (selected_row, eligible, rejected_reasons)."""
    reasons = {}
    eligible = []
    for r in measured(rows):
        if r["config_id"] == BASELINE_ID:
            continue
        if not r["pass"]:
            reasons[r["config_id"]] = "fails the formal accuracy gate"
        elif not r["within_top1_floor"]:
            reasons[r["config_id"]] = (
                f"clears the formal gate but costs {r['top1_cost_points']:.3f} top-1 "
                f"(floor is {TOP1_FLOOR_POINTS:.3f}) for {r['decode_gain_pct']:+.2f}% decode"
            )
        elif r["decode_gain_pct"] < -band_pct:
            reasons[r["config_id"]] = (
                f"slower than the default by {-r['decode_gain_pct']:.2f}%, "
                f"beyond the +/-{band_pct:.2f}% run-to-run band -- a real regression"
            )
        elif not r["gain_exceeds_noise_band"]:
            reasons[r["config_id"]] = (
                f"{r['decode_gain_pct']:+.2f}% decode is inside the measured "
                f"+/-{band_pct:.2f}% run-to-run band -- indistinguishable from the default"
            )
        else:
            eligible.append(r)
    if not eligible:
        return baseline, eligible, reasons

    # -- the band applies BETWEEN candidates too, not only against the default --
    #
    # Added during the stage-07 review, and added *after* seeing the row it
    # decides, which is stated here rather than buried: ``R26`` (attention bfp4
    # on top of the selected widths) measured 43.58 t/s/u against ``R25``'s
    # 43.54 -- a 0.09% lead, roughly a quarter of the measured 0.368% band --
    # while costing a top-1 point that ``R25`` does not.
    #
    # Ranking eligible rows on the raw point estimate would hand the selection
    # to ``R26`` on a difference the same rule declares unmeasurable two clauses
    # earlier ("any win below the band is the same number measured twice", and
    # "ties, and anything inside the band, resolve to the default"). Applying
    # the band when comparing a candidate to the default but *not* when
    # comparing two candidates is the inconsistency, not the fix for it.
    #
    # So: every eligible row within one band of the fastest is **tied** with it,
    # and the tie is broken on the rule the governing datatype-sweep skill states
    # for exactly this situation -- "if two configs are within measurement noise,
    # prefer the simpler and safer one" -- measured as the count of dtype and
    # fidelity fields moved off the shipped default (``numerical_changes``).
    # Top-1, then decode, then TTFT order whatever that leaves tied.
    #
    # Leading on simplicity rather than on top-1 matters here. Teacher-forcing
    # top-1 is deterministic per config, so ``R25``'s 0.990 x3 against ``R26``'s
    # 0.980 x3 is the same single token re-observed, not three pieces of
    # evidence -- and this document argues twice over (clause (a), limitation 3)
    # that one point of top-1 on a 100-token reference is not signal. A tiebreak
    # that leans on it would be leaning on the axis the stage has already
    # declared unresolvable. The simplicity rule does not depend on that
    # judgment at all: ``R25`` moves no dtype and no fidelity, ``R26`` moves two
    # attention weight tensors to bfp4, so the skill's own rule selects ``R25``
    # whether or not the top-1 point means anything. Top-1 is kept as the
    # secondary ordering, and it happens to agree.
    fastest = max(eligible, key=lambda r: r["decode_tps_user"])
    tied = [
        r
        for r in eligible
        if (fastest["decode_tps_user"] - r["decode_tps_user"]) / fastest["decode_tps_user"] * 100 <= band_pct
    ]
    n_changes = {r["config_id"]: len(numerical_changes(r, baseline)) for r in tied}
    best = max(
        tied,
        key=lambda r: (
            -n_changes[r["config_id"]],
            r["top1"],
            round(r["decode_tps_user"], 2),
            -(r["ttft_ms"] or 0),
        ),
    )
    for r in tied:
        if r is best:
            continue
        lead = (r["decode_tps_user"] - best["decode_tps_user"]) / best["decode_tps_user"] * 100
        gap = "leads" if lead > 0 else "trails"
        changed = numerical_changes(r, baseline)
        reasons[r["config_id"]] = (
            f"eligible, and the closest thing to a rival the sweep produced: it {gap} the selected "
            f"{best['config_id']} by {abs(lead):.2f}% ({r['decode_tps_user']:.2f} vs "
            f"{best['decode_tps_user']:.2f} t/s/u), which is inside the {band_pct:.2f}% run-to-run "
            f"band -- so the two are tied on the ranking metric. The tie breaks on the simpler and "
            f"safer config: it moves {len(changed)} dtype/fidelity field(s) off the default "
            f"({', '.join(changed) or 'none'}) against {len(numerical_changes(best, baseline))} for "
            f"{best['config_id']}. Top-1 agrees as the secondary ordering, {r['top1']:.3f} against "
            f"{best['top1']:.3f}"
        )
    return best, eligible, reasons


def chart(rows, metric: str, path: Path, selected: dict, baseline: dict, band_pct: float) -> None:
    pts = measured(rows)
    for p in pts:
        p["_x"] = p[metric]
        p["_y"] = p["decode_tps_user"]
    front = pareto_front(pts)

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # the measured run-to-run band around the default: everything inside this
    # ribbon is the same speed as the default, as far as this rig can tell
    b = baseline["decode_tps_user"]
    half = b * band_pct / 100.0
    ax.axhspan(
        b - half,
        b + half,
        color=C_BAND,
        alpha=0.35,
        zorder=0,
        label=f"run-to-run band on the default (+/-{band_pct:.2f}%)",
    )

    # The top-5 frontier is a SINGLE point -- every evaluated config sits at
    # top-5 1.000, so only the fastest one is non-dominated. Guarding on
    # ``len(front) > 1`` silently dropped both the line and the legend entry
    # from that chart, which read as "this chart has no frontier" rather than as
    # the finding it is. A one-point frontier is drawn as a marker instead.
    if len(front) > 1:
        ax.plot(
            [p["_x"] for p in front],
            [p["_y"] for p in front],
            color=C_FRONTIER,
            lw=2,
            zorder=2,
            label="Pareto frontier",
            solid_capstyle="round",
        )
    elif front:
        ax.scatter(
            [front[0]["_x"]],
            [front[0]["_y"]],
            s=420,
            facecolors="none",
            edgecolors=C_FRONTIER,
            linewidths=2,
            zorder=2,
            label="Pareto frontier (a single point: every config scores 1.000 here)",
        )

    ok = [p for p in pts if p.get("pass") and p is not selected]
    bad = [p for p in pts if not p.get("pass")]
    if ok:
        ax.scatter(
            [p["_x"] for p in ok],
            [p["_y"] for p in ok],
            s=70,
            c=C_PASS,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
            label="clears accuracy gate",
        )
    if bad:
        ax.scatter(
            [p["_x"] for p in bad],
            [p["_y"] for p in bad],
            s=85,
            c=C_FAIL,
            marker="x",
            linewidths=2,
            zorder=3,
            label="fails accuracy gate",
        )
    ax.scatter(
        [selected["_x"]],
        [selected["_y"]],
        s=240,
        c=C_SELECTED,
        zorder=5,
        edgecolors=SURFACE,
        linewidths=2.5,
        label=f"selected: {selected['config_id']}",
    )

    # Each axis gets ITS OWN threshold. These are numerically equal (0.98) and
    # the top-1 line was previously drawn at MIN_TOP5 and labelled "shown for
    # reference: the gate binds on top-5, not top-1" -- but the selection rule
    # in ``select()`` does enforce ``top1 >= baseline - TOP1_FLOOR_POINTS``, and
    # ``selection_reasons.json`` records rejections on exactly that basis. The
    # line was right by coincidence and captioned as if it were not binding.
    top1_floor = round(baseline["top1"] - TOP1_FLOOR_POINTS, 3)
    threshold = MIN_TOP5 if metric == "top5" else top1_floor
    ax.axvline(threshold, color=C_INK2, ls=":", lw=1.8, zorder=1)
    gate_label = (
        f"formal accuracy gate (top-5 >= {MIN_TOP5:.2f})"
        if metric == "top5"
        else f"top-1 floor ({top1_floor:.2f}) -- one point below the\ndefault; enforced by the selection rule,\nnot by the formal gate"
    )
    # anchored to the TOP of the axes: the legend lives at lower-left and a
    # bottom-anchored rotated label collides with it on the top-5 chart, where
    # every point shares x = 1.000 and the gate line stands alone on the left.
    # Left of the line on the top-1 chart: the fastest row sits *on* the floor,
    # so its point label lands to the right of the line and the two collide.
    side = 1 if metric == "top5" else -1
    ax.annotate(
        gate_label,
        xy=(threshold, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5 * side, -8),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="left" if side > 0 else "right",
        fontsize=8.5,
        color=C_INK2,
    )

    # Label frontier points. On the top-5 chart the top-1 cost is invisible on
    # the axes, so it is spelled out -- that is the whole point of the pair.
    for p in front:
        if p is selected:
            continue
        cost = p.get("top1_cost_points") or 0.0
        tag = p["config_id"]
        if metric == "top5" and cost > 0:
            tag += f"\n(top-1 -{cost:.3f})"
        ax.annotate(tag, (p["_x"], p["_y"]), textcoords="offset points", xytext=(8, 4), fontsize=7.5, color=C_INK2)
    ax.annotate(
        f"{selected['config_id']}\n{selected['_y']:.2f} t/s/u, {metric}={selected['_x']:.3f}",
        (selected["_x"], selected["_y"]),
        textcoords="offset points",
        xytext=(12, -26),
        fontsize=9,
        color=C_INK,
        fontweight="bold",
    )

    metric_name = {"top1": "top-1", "top5": "top-5"}[metric]
    ax.set_xlabel(f"{metric_name} token accuracy vs AIME24 chat reference", fontsize=10, color=C_INK2)
    ax.set_ylabel("traced teacher-forcing decode (t/s/u)", fontsize=10, color=C_INK2)
    ax.set_title(
        f"Qwen3-Coder-30B-A3B stage 07 datatype sweep - {metric_name} vs traced decode\n"
        "48 layers, batch 1, 1x4 Blackhole P300_X2, FABRIC_1D_RING",
        fontsize=12,
        color=C_INK,
        loc="left",
        pad=14,
    )
    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    xs = [p["_x"] for p in pts] + [threshold]
    pad = max(0.004, (max(xs) - min(xs)) * 0.14)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}  ({len(pts)} configs, {len(front)} on the frontier)")


CSV_COLUMNS = [
    "config_id",
    "group",
    "delta_from_default",
    "dtype_policy",
    "compute_fidelity_policy",
    "top1",
    "top5",
    "top100",
    "top1_cost_points",
    "ttft_ms",
    "decode_tps_user",
    "decode_gain_pct",
    "noise_band_pct",
    "gain_exceeds_noise_band",
    "within_top1_floor",
    "pass",
    "gate",
    "selection_rule",
    "selected",
    "on_pareto_front_top1",
    "on_pareto_front_top5",
    "measurement_regime",
    "hardware",
    "mesh",
    "block_width_resolved",
    "blocker_op",
    "blocker_info",
    "command",
    "log",
    "rationale",
]


#: What ``R19_kv_bfp8`` measured **before** the prefill cache writer was fixed.
#:
#: The first run of this row scored top-1 / top-5 / top-100 all at 0.010 --
#: chance -- with decode 28.86 t/s/u and TTFT 8108 ms. That was never a verdict
#: on the dtype: ``bfloat4_b`` expert weights are far more aggressive and hold
#: top-5 at 1.000 in this same sweep, and a pure dtype reduction should be
#: *faster*, not 32% slower. ``probes/kv_bfp8_diagnosis.py`` found the cause at
#: the op level and ``tt/functional_decoder.match_cache_dtype`` closed it, so
#: both KV rows below are now real measurements and are ranked like any other.
#: Kept as data because "a documented field of the precision config silently
#: filled the KV cache with NaN" is the more useful half of this row's story.
KV_PRIOR_DEFECT = {
    "when": "first measurement, before tt/functional_decoder.match_cache_dtype existed",
    "top1": 0.010,
    "top5": 0.010,
    "top100": 0.010,
    "decode_tps_user": 28.86,
    "ttft_ms": 8108.0,
    "cause": (
        "paged_fill_cache accepts a bfloat16 input into a bfloat8_b cache -- its input "
        "validation is a permissive OR that a mismatch satisfies -- and then writes NaN. "
        "The cache was NaN from the first prefill write, which is why top-100 sat at "
        "chance from token zero."
    ),
    "fix": (
        "cast K/V to the cache tensor's own dtype at the paged_fill_cache and fill_cache "
        "sites (tt/functional_decoder.match_cache_dtype). The paged_update_cache sites are "
        "deliberately NOT cast: that op requires a FLOAT32/BFLOAT16 update, converts into "
        "the cache itself, and rejects a block-float input outright "
        "(paged_update_cache_device_operation.cpp:296)."
    ),
    "evidence": "probes/kv_bfp8_diagnosis.json -- both writers, six cache/input combinations",
}


def annotate_kv(rows) -> None:
    """Attach the pre-fix history and the op-level diagnosis to the KV rows.

    They are **not** reclassified any more. Both are ordinary measured rows that
    rank on the same metric as everything else; what is attached is the record
    of how they got here, so a reader of ``sweep_results.json`` alone can see
    that this row's first number was an integration defect rather than a dtype
    verdict, and what closed it.
    """
    diag_path = HERE / "kv_bfp8_diagnosis.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else None
    for row in rows:
        if row["config_id"] not in ("R19_kv_bfp8", "R28_kv_bfp8_bw64_24"):
            continue
        row["prior_measurement_invalid"] = KV_PRIOR_DEFECT
        row["diagnosis_evidence"] = diag


def main():
    rows = json.loads((SWEEP_DIR / "sweep_results.json").read_text())
    annotate_kv(rows)
    baseline = next(r for r in rows if r["config_id"] == BASELINE_ID)

    # --- the measured band -----------------------------------------------
    rep_path = SWEEP_DIR / "repeats.json"
    band_pct = 0.0
    repeats = {}
    if rep_path.exists():
        repeats = json.loads(rep_path.read_text())
        bands = [v["decode_spread_pct"] for v in repeats.values() if v.get("n", 0) > 1]
        band_pct = max(bands) if bands else 0.0
        print(f"measured run-to-run band: {band_pct:.3f}%  from {[(k, v['n']) for k, v in repeats.items()]}")
    else:
        print("WARNING: no repeats.json -- noise band unmeasured, treating as 0%")

    enrich(rows, band_pct, baseline)
    sel, eligible, reasons = select(rows, band_pct, baseline)

    # frontier membership, as data
    pts = measured(rows)
    for metric, key in (("top1", "on_pareto_front_top1"), ("top5", "on_pareto_front_top5")):
        for p in pts:
            p["_x"], p["_y"] = p[metric], p["decode_tps_user"]
        front = {id(p) for p in pareto_front(pts)}
        for p in pts:
            p[key] = id(p) in front
    for r in rows:
        r["selected"] = r["config_id"] == sel["config_id"]
        r.pop("_x", None)
        r.pop("_y", None)

    print(f"\nSELECTED: {sel['config_id']}  {sel['delta_from_default']}")
    print(f"  decode {sel['decode_tps_user']} t/s/u  top1={sel['top1']} top5={sel['top5']} top100={sel['top100']}")
    print(f"  eligible under the stated rule: {[r['config_id'] for r in eligible] or 'none -> default retained'}")

    for metric, name in (("top1", "top1_perf_pareto.png"), ("top5", "top5_perf_pareto.png")):
        chart(rows, metric, SWEEP_DIR / name, sel, baseline, band_pct)
    for r in rows:
        r.pop("_x", None)
        r.pop("_y", None)

    (SWEEP_DIR / "selected_precision_config.json").write_text(
        json.dumps(sel["precision_config"], indent=2, sort_keys=True) + "\n"
    )
    (SWEEP_DIR / "sweep_results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (SWEEP_DIR / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -(r.get("decode_tps_user") or 0)):
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})
    (SWEEP_DIR / "selection_reasons.json").write_text(
        json.dumps(
            {
                "selected": sel["config_id"],
                "selection_rule": rows[0].get("selection_rule"),
                "noise_band_pct": band_pct,
                "top1_floor_points": TOP1_FLOOR_POINTS,
                "baseline_top1": baseline["top1"],
                "eligible": [r["config_id"] for r in eligible],
                "rejected": reasons,
            },
            indent=2,
        )
        + "\n"
    )

    hdr = f"{'config':28s} {'top1':>6s} {'d.top1':>7s} {'top5':>6s} {'t100':>6s} {'t/s/u':>8s} {'gain%':>7s} {'TTFT':>8s} pass"
    print("\n" + hdr)
    for r in sorted(measured(rows), key=lambda r: -r["decode_tps_user"]):
        print(
            f"{r['config_id']:28s} {r['top1']:6.3f} {r['top1_cost_points']:+7.3f} {r['top5']:6.3f} "
            f"{r['top100']:6.3f} {r['decode_tps_user']:8.2f} {r['decode_gain_pct']:+7.2f} "
            f"{r['ttft_ms']:8.1f} {r['pass']}"
        )
    for r in rows:
        if r.get("status") == "blocked":
            print(f"{r['config_id']:28s}  BLOCKED  {r['blocker_op']}: {r.get('blocker_info')}")
    print("\nrejected under the stated rule:")
    for cid, why in reasons.items():
        print(f"  {cid:28s} {why}")


if __name__ == "__main__":
    main()
