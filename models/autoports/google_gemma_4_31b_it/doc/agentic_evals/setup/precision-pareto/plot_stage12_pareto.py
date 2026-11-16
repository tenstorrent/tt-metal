#!/usr/bin/env python3
"""Build candidate Pareto charts from matching stage-12 natural-512 accuracy evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[7]
D = REPO / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/precision-pareto"
CATALOG = D / "setup/precision-accuracy-candidates.json"
MEASUREMENT_SCOPE = "Historical measurements before the HiFi4 Q32 correction; not post-fix validation."
GEOMETRY_NOTE = (
    "The precision rows use policy-specific historical Q geometry; Q size is not uniform "
    "across candidates, and this plot does not claim every point used Q128."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nondominated(rows, x_key, y_key):
    # Keep points for which no evaluated candidate has both >= accuracy and >= speed,
    # with one strict improvement. Return in increasing accuracy order for plotting.
    out = []
    for r in rows:
        x, y = r[x_key], r[y_key]
        dominated = any(
            (q[x_key] >= x and q[y_key] >= y) and (q[x_key] > x or q[y_key] > y) for q in rows if q is not r
        )
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: (r[x_key], r[y_key]))


def compact_policy(p):
    wg = p.get("weight_groups", {})
    return (
        f"attn={wg.get('attention','?')}; mlp={wg.get('mlp','?')}; "
        f"down={wg.get('down','?')}; head={wg.get('lm_head','?')}; "
        f"KV={p.get('kv_cache_dtype','?')}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text())
    rows, ref_hashes, mismatches = [], set(), []
    for c in catalog["rows"]:
        evid = D / c["evidence"]
        actual = json.loads(evid.read_text())
        nat = actual.get("natural") or []
        if len(nat) != 1:
            continue
        n = nat[0]
        ref_path = Path(actual["natural_reference"])
        if not ref_path.is_file():
            raise RuntimeError(f"missing natural reference: {ref_path}")
        ref_hash = sha(ref_path)
        ref_hashes.add(ref_hash)
        total = n.get("total")
        if total != 512 or n.get("decode_tokens") != 511.0:
            continue
        traces = actual.get("traces") or {}
        if not traces.get("model") or not traces.get("sampling"):
            raise RuntimeError(f"missing trace IDs in {evid}")
        if not n.get("decode_t/s/u") or not n.get("decode_elapsed_s"):
            raise RuntimeError(f"missing traced decode speed in {evid}")
        policy_path = evid.parent / "runtime_policy.json"
        policy_data = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        policy = policy_data.get("policy", c.get("policy", {}))
        if policy.get("config_id") and c["id"] != policy["config_id"]:
            mismatches.append({"catalog_id": c["id"], "runtime_config_id": policy["config_id"], "evidence": str(evid)})
        natural_top1 = n["matches_top1"] / total
        natural_top5 = n["matches_top5"] / total
        passed = natural_top1 >= 0.90 and natural_top5 >= 0.98
        rows.append(
            {
                "id": c["id"],
                "label": c["id"].replace("stage12_", "").replace("_accuracy", ""),
                "round": (
                    f"P{re.search(r'_p(8|9|1[0-9]|20)$', c['id']).group(1)}"
                    if re.search(r"_p(8|9|1[0-9]|20)$", c["id"])
                    else ("Full BF16" if "diagnostic" in c["id"] else "candidate")
                ),
                "evidence_path": str(evid.relative_to(REPO)),
                "evidence_sha256": sha(evid),
                "runtime_policy_path": str(policy_path.relative_to(REPO)) if policy_path.is_file() else None,
                "runtime_policy_sha256": sha(policy_path) if policy_path.is_file() else None,
                "policy": policy,
                "policy_compact": compact_policy(policy),
                "natural_reference_path": str(ref_path),
                "natural_reference_sha256": ref_hash,
                "tokens": total,
                "top1_matches": n["matches_top1"],
                "top5_matches": n["matches_top5"],
                "top100_matches": n.get("matches_top100"),
                "top1": natural_top1,
                "top5": natural_top5,
                "top100": n.get("matches_top100", 0) / total,
                "trace_decode_tps_user": n["decode_t/s/u"],
                "decode_elapsed_s": n["decode_elapsed_s"],
                "ttft_ms": n.get("ttft_ms"),
                "model_trace_id": traces["model"],
                "sampling_trace_id": traces["sampling"],
                "accuracy_pass_90_98": passed,
                "catalog_accuracy_pass": c.get("accuracy_pass"),
                "candidate_status": "candidate_pending_root_selection"
                if "p19" in c["id"]
                else ("accuracy_pass" if passed else "below_accuracy_bar"),
                "measurement_scope": MEASUREMENT_SCOPE,
                "geometry_note": GEOMETRY_NOTE,
            }
        )
    if len(ref_hashes) != 1:
        raise RuntimeError(f"natural references are not byte-identical: {sorted(ref_hashes)}")
    if mismatches:
        raise RuntimeError(f"catalog/runtime policy ids disagree: {mismatches}")
    if len(rows) < 10:
        raise RuntimeError(f"unexpectedly few compatible rows: {len(rows)}")

    summary = {
        "status": "candidate_pareto_only_no_selection",
        "selection": None,
        "selection_note": "No point is colored as selected; P19 remains a candidate pending root decision.",
        "measurement_scope": MEASUREMENT_SCOPE,
        "geometry_note": GEOMETRY_NOTE,
        "accuracy_reference": {
            "tokens": 512,
            "reference_sha256": next(iter(ref_hashes)),
            "reference_path": rows[0]["natural_reference_path"],
        },
        "performance_metric": "natural-reference traced teacher-forcing decode_t/s/u (higher is better)",
        "thresholds": {"top1": 0.90, "top5": 0.98},
        "trace_validation": "Every plotted row has model and sampling trace IDs in readiness_accuracy.json; natural reply has 511 decode tokens.",
        "source_catalog": str(CATALOG.relative_to(REPO)),
        "source_catalog_sha256": sha(CATALOG),
        "rows": rows,
    }
    (OUT / "stage12-natural512-pareto.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "round",
        "id",
        "candidate_status",
        "top1",
        "top1_matches",
        "top5",
        "top5_matches",
        "top100",
        "top100_matches",
        "tokens",
        "trace_decode_tps_user",
        "decode_elapsed_s",
        "ttft_ms",
        "policy_compact",
        "measurement_scope",
        "geometry_note",
        "evidence_path",
        "evidence_sha256",
    ]
    with (OUT / "stage12-natural512-pareto.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)
    with (OUT / "stage12-natural512-pareto.md").open("w") as f:
        f.write("# Stage-12 natural-512 precision candidates\n\n")
        f.write(f"> **Scope:** {MEASUREMENT_SCOPE} {GEOMETRY_NOTE}\n\n")
        f.write("![Top-1 accuracy and traced decode performance Pareto plot](top1_perf_pareto.png)\n\n")
        f.write("![Top-5 accuracy and traced decode performance Pareto plot](top5_perf_pareto.png)\n\n")
        f.write(
            "Download the [compact CSV data](stage12-natural512-pareto.csv) or [machine-readable summary](stage12-natural512-pareto.json).\n\n"
        )
        f.write(
            "Performance is traced teacher-forcing decode tokens/s/user on the same 512-token natural-text reference. These are accuracy-round diagnostic rates, not vLLM serving rates. No config is currently marked selected; P19 is still pending root selection.\n\n"
        )
        f.write(
            "| Candidate | Top-1 | Top-5 | Top-100 | Traced decode t/s/u | Policy | Status |\n|---|---:|---:|---:|---:|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['round']} | {r['top1']:.4%} ({r['top1_matches']}/512) | {r['top5']:.4%} ({r['top5_matches']}/512) | {r['top100']:.4%} | {r['trace_decode_tps_user']:.3f} | {r['policy_compact']} | {r['candidate_status']} |\n"
            )
        f.write(
            "\nAcceptance thresholds shown for reference: top-1 ≥90%, top-5 ≥98%. The natural-reference scores are the comparability filter used here; this table does not assert other readiness gates have passed.\n"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    palette = {
        "below_accuracy_bar": "#8B98A5",
        "accuracy_pass": "#318C72",
        "candidate_pending_root_selection": "#E08B35",
    }
    round_labels = {r["id"]: r["round"] for r in rows}
    specs = [
        ("top1", 0.90, "Top-1 accuracy", "top1_perf_pareto.png"),
        ("top5", 0.98, "Top-5 accuracy", "top5_perf_pareto.png"),
    ]
    for key, threshold, xlabel, filename in specs:
        fig, ax = plt.subplots(figsize=(13.4, 7.6), constrained_layout=True)
        frontier = nondominated(rows, key, "trace_decode_tps_user")
        ax.plot(
            [r[key] * 100 for r in frontier],
            [r["trace_decode_tps_user"] for r in frontier],
            color="#34495E",
            lw=1.7,
            ls=(0, (4, 3)),
            label="Evaluated Pareto frontier",
            zorder=1,
        )
        label_offsets = {
            "P8": (-30, -22),
            "P9": (-44, 14),
            "P10": (-42, 22),
            "P11": (12, 26),
            "P12": (8, -15),
            "P13": (-48, 12),
            "P14": (8, 10),
            "P15": (8, -16),
            "P16": (-50, -18),
            "P17": (10, 14),
            "P18": (-45, -16),
            "P19": (10, -16),
            "P20": (10, 12),
            "Full BF16": (-54, -18),
        }
        legend_seen = set()
        for r in rows:
            status = r["candidate_status"]
            label = round_labels.get(r["id"], r["round"])
            legend_label = None
            if status == "candidate_pending_root_selection" and status not in legend_seen:
                legend_label = "P19 candidate (pending root)"
            elif status == "accuracy_pass" and status not in legend_seen:
                legend_label = "Accuracy bar pass"
            if legend_label:
                legend_seen.add(status)
            ax.scatter(
                r[key] * 100,
                r["trace_decode_tps_user"],
                s=105 if status == "candidate_pending_root_selection" else 78,
                marker="D" if status == "candidate_pending_root_selection" else "o",
                color=palette[status],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=legend_label,
            )
            dx, dy = label_offsets.get(label, (8, 8))
            ax.annotate(
                label,
                (r[key] * 100, r["trace_decode_tps_user"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#243447",
                arrowprops=dict(arrowstyle="-", color="#83909B", lw=0.5) if abs(dx) > 35 or abs(dy) > 18 else None,
            )
        ax.axvline(
            threshold * 100, color="#BD3D35", lw=1.4, ls=(0, (2, 2)), label=f"Accuracy threshold {threshold:.0%}"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Traced teacher-forcing decode (tokens/s/user)")
        ax.set_title(
            f"Gemma 4 31B stage-12 precision candidates — natural reference, 512 tokens\n{MEASUREMENT_SCOPE}\nSame reference SHA-256: {next(iter(ref_hashes))[:16]}…"
        )
        ax.grid(True, color="#DCE3E8", linewidth=0.7, alpha=0.8)
        ax.set_xlim(
            min(threshold * 100 - 1, min(r[key] * 100 for r in rows) - 0.5), max(r[key] * 100 for r in rows) + 1.8
        )
        ax.set_ylim(
            min(r["trace_decode_tps_user"] for r in rows) - 1.0, max(r["trace_decode_tps_user"] for r in rows) + 1.7
        )
        handles, labels = ax.get_legend_handles_labels()
        uniq = {l: h for h, l in zip(handles, labels) if l}
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), frameon=False, loc=("upper left" if key == "top5" else "upper right"))
        ax.text(
            0.99,
            0.015,
            "No selected config marked; P19 remains a pending candidate.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#6B7780",
        )
        fig.savefig(OUT / filename, dpi=190, facecolor="white")
        plt.close(fig)

    (OUT / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "row_count": len(rows),
                "all_natural_references_identical": True,
                "natural_reference_sha256": next(iter(ref_hashes), None),
                "all_rows_512_tokens": all(r["tokens"] == 512 for r in rows),
                "all_rows_traced": all(r["model_trace_id"] and r["sampling_trace_id"] for r in rows),
                "catalog_runtime_policy_id_mismatches": mismatches,
                "candidate_rows": sum(r["candidate_status"] == "candidate_pending_root_selection" for r in rows),
                "selected_config": None,
                "measurement_scope": MEASUREMENT_SCOPE,
                "geometry_note": GEOMETRY_NOTE,
                "generated_files": [
                    "stage12-natural512-pareto.json",
                    "stage12-natural512-pareto.csv",
                    "stage12-natural512-pareto.md",
                    "top1_perf_pareto.png",
                    "top5_perf_pareto.png",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "refs": len(ref_hashes),
                "out": str(OUT),
                "statuses": {s: sum(r["candidate_status"] == s for r in rows) for s in palette},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
