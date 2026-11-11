# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""report.md for one work directory: what was calibrated, what the model predicts, what the device measured."""

from __future__ import annotations

import json
from pathlib import Path


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def write_report(workdir: Path, plan: str = "bal128_s1") -> Path:
    spec = _read(workdir / "spec.json") or {}
    cal = _read(workdir / "calibration" / "summary.json")
    scr = _read(workdir / "screen.json")
    ab = _read(workdir / "ab" / "summary.json")
    lines = [f"# Expert placement report: {spec.get('name', '?')}", ""]
    lines += [
        f"- Experts / top-k: {spec.get('num_experts')} / {spec.get('top_k')}; mesh {spec.get('rows')}x{spec.get('groups')} "
        f"({spec.get('num_experts', 0) // max(1, spec.get('rows', 1) * spec.get('groups', 1))} experts per chip); transport model `{spec.get('transport')}`",
        f"- Plan: `{plan}`",
        "",
    ]
    if cal:
        main = [p for p in cal["plans"] if p != "identity" and not p.startswith(("lodo_", "indom_", "lolo_"))]
        lines += [
            "## Calibration (modeled, held-out test sets)",
            "",
            f"{len(cal['layers'])} MoE layers, {cal['steps']} annealing steps, constraint `{cal['constraint']}`, {cal['seconds']} s.",
            "",
            "| Test set | original utilization | " + " | ".join(main) + " |",
            "|---|---|" + "---|" * len(main),
        ]
        for t in cal["tests"]:
            r = cal["peak_change_pct"][t]
            lines.append(
                f"| {t} | {cal['utilization_identity'][t]:.3f} | " + " | ".join(f"{r[p]:+.1f}%" for p in main) + " |"
            )
        lines += [
            "",
            "Values: change of the busiest chip's routed-expert work vs the original layout (negative is better).",
            "",
        ]
        gen = [
            (t[len("domain_") :], t)
            for t in cal["tests"]
            if t.startswith("domain_") and f"lodo_{t[len('domain_'):]}" in cal["plans"]
        ]
        if gen and plan in cal["plans"]:
            lines += [
                "### Generalization to unseen domains",
                "",
                f"| Domain | {plan} (seen) | left out (lodo) | only this domain (indom) |",
                "|---|---|---|---|",
            ]
            for d, t in gen:
                r = cal["peak_change_pct"][t]
                lines.append(
                    f"| {d} | {r[plan]:+.1f}% | {r[f'lodo_{d}']:+.1f}% | {r.get(f'indom_{d}', float('nan')):+.1f}% |"
                )
            lines.append("")
    if scr:
        lines += ["## Screen", ""]
        if "verdict" in scr:
            lines.append(
                f"- Verdict: **{'go' if scr['verdict']['go'] else 'no-go'}**: {'; '.join(scr['verdict']['reasons'])}"
            )
        if "expected" in scr:
            e = scr["expected"]
            lines.append(
                f"- Modeled {e['modeled_pct']:+.1f}%; routed-expert share {e['routed_expert_share'] if e['routed_expert_share'] is not None else 'not profiled'}; "
                f"expected on device roughly {e['expected_pct_range'][0]:+.2f}% to {e['expected_pct_range'][1]:+.2f}%"
            )
        for name, r in scr.get("sequences", {}).items():
            extra = f", modeled {r['modeled_peak_change_pct']}" if "modeled_peak_change_pct" in r else ""
            lines.append(
                f"- `{name}`: original utilization {r['utilization']:.3f} (busiest chip {r['busiest_over_mean']:.2f}x the mean){extra}"
            )
        lines.append("")
    if ab:
        lines += [
            "## On-device A/B",
            "",
            "| Workload | original runs (s) | placed runs (s) | latency change | separated | digests identical |",
            "|---|---|---|---|---|---|",
        ]
        for w, s in ab["sides"].items():
            lines.append(
                f"| {w} | {', '.join(f'{x:.3f}' for x in s['O_runs_s'])} | {', '.join(f'{x:.3f}' for x in s['B_runs_s'])} | {s['latency_change_pct']:+.2f}% | "
                f"{'yes' if s['separated'] else 'no'} | {s.get('cache_digests_identical', 'n/a')} |"
            )
        g = ab["gate"]
        lines += ["", f"**Gate: {'PASS' if g['passed'] else 'FAIL'}**: {'; '.join(g['reasons'])}", ""]
    prof = sorted((workdir / "profile").glob("profile-*/measurement.json")) if (workdir / "profile").exists() else []
    if prof:
        lines += [
            "## Stage profile (sync-bracketed; shares, not absolute times)",
            "",
            "| Side | MoE share | routed-expert share |",
            "|---|---|---|",
        ]
        for p in prof:
            m = json.loads(p.read_text())
            if m.get("status") == "complete":
                lines.append(
                    f"| {p.parent.name} | {m['profile']['moe_share']:.1%} | {m['profile']['routed_expert_share']:.1%} |"
                )
        lines.append("")
    out = workdir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
