"""Read-only visibility CLIs for the per-model state files:
``view-evidence`` and ``view-skips``.

These complement the existing ``tackle-skipped --dry-run`` (which
prints the skip-list scoped to Phase 2 routing) by giving a plain
read-only view of what the tool currently believes about a model.

No mutations — purely diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cmd_view_evidence(args) -> int:
    """Pretty-print the enriched hot_cold.json evidence record.

    Shows per-component: kind, frequency, cpu_latency_ms, cpu_latency_pct,
    ops_count, io_bytes, compute_density, affinity_score, and the
    classifier's evidence reasons.
    """
    from ..overlay_manager import load_hot_cold_evidence

    model_id = args.model_id
    evidence = load_hot_cold_evidence(model_id)
    if not evidence:
        print(f"no hot_cold.json on file for `{model_id}`")
        print(f"run `python -m scripts.tt_hw_planner profile-cold {model_id}` first.")
        return 1

    print(f"evidence for {model_id}  ({len(evidence)} components)")
    print()

    multi_mode_count = sum(1 for e in evidence.values() if isinstance(e.get("modes"), dict))
    if multi_mode_count:
        print(
            f"  ({multi_mode_count} components have per-workload-mode evidence; "
            f"top-level kind is union-of-evidence)"
        )
        print()

    print(
        f"  {'component':<35} {'kind':<10} {'freq':>5} {'cpu_ms':>8} {'pct':>7}  "
        f"{'density':>10} {'aff':>5} {'modes':<25}  evidence"
    )
    print(f"  {'-'*35} {'-'*10} {'-'*5} {'-'*8} {'-'*7}  {'-'*10} {'-'*5} {'-'*25}  {'-'*40}")
    hot_count = cold_count = unknown_count = 0
    for name in sorted(evidence.keys()):
        e = evidence[name]
        kind = str(e.get("kind", "?"))
        if kind == "HOT":
            hot_count += 1
        elif kind == "COLD":
            cold_count += 1
        else:
            unknown_count += 1
        freq = e.get("frequency")
        lat_ms = e.get("cpu_latency_ms")
        lat_pct = e.get("cpu_latency_pct")
        dens = e.get("compute_density")
        aff = e.get("affinity_score")
        reasons = e.get("evidence", [])

        freq_s = "-" if freq is None else f"{freq:.2f}"
        lat_ms_s = "-" if lat_ms is None else f"{lat_ms:.2f}"
        lat_pct_s = "-" if lat_pct is None else f"{lat_pct:.2f}%"
        dens_s = "-" if (not dens or dens == 0) else f"{dens:.2e}"
        aff_s = "-" if aff is None else f"{aff:+d}"

        # Multi-mode column: "image:HOT, video:HOT" (or empty if single-mode)
        modes = e.get("modes")
        if isinstance(modes, dict) and modes:
            parts = [f"{m}:{(d.get('kind') or '?')[:4]}" for m, d in sorted(modes.items())]
            modes_s = ", ".join(parts)[:25]
        else:
            modes_s = "(default)"

        why = "; ".join(str(r) for r in reasons)[:60]
        print(
            f"  {name:<35} {kind:<10} {freq_s:>5} {lat_ms_s:>8} {lat_pct_s:>7}  "
            f"{dens_s:>10} {aff_s:>5} {modes_s:<25}  {why}"
        )
    print()
    print(f"  HOT: {hot_count}, COLD: {cold_count}, UNKNOWN: {unknown_count}")
    return 0


def cmd_view_skips(args) -> int:
    """Pretty-print the skip-list with category, reason, captured timestamp,
    and per-category remedy hint. Plain read-only — no Phase 2 routing
    side effects."""
    from ..overlay_manager import load_persistent_skips

    model_id = args.model_id
    skips = load_persistent_skips(model_id)
    if not skips:
        print(f"no skip-list entries for `{model_id}` — clean slate.")
        return 0

    # Group by category for clearer display.
    by_category: dict = {}
    for name, entry in skips.items():
        cat = (entry.get("category") or "UNKNOWN").upper()
        by_category.setdefault(cat, []).append((name, entry))

    print(f"skip-list for {model_id}  ({len(skips)} entries across {len(by_category)} categories)")
    print()

    _CATEGORY_REMEDY = {
        "COLD": "workload-cold; CPU is correct. No action needed.",
        "KERNEL_MISSING": "TTNN op verified missing. Wait for TTNN to ship the op.",
        "CONSTRAINT_MISMATCH": "TTNN op exists but failed for dtype/layout/shape; " "wait for TTNN to extend support.",
        "TOOL_BUG": "scaffolder produced bad inputs; fix the tool, then " "`overlay-clear-skips --category TOOL_BUG`.",
        "HF_ERROR": "HF reference errored; check transformers version + model availability.",
        "ITERATION_BUDGET": "ran out of attempts; retry with bigger --auto-max-iters.",
        "AGENT_STUCK": "agent didn't engage; consider decompose or different model.",
        "UNKNOWN": "(no category — legacy entry; treat as COLD until reclassified)",
    }

    for cat in sorted(by_category.keys()):
        entries = by_category[cat]
        print(f"  [{cat}]  ({len(entries)} component(s))")
        remedy = _CATEGORY_REMEDY.get(cat)
        if remedy:
            print(f"    → {remedy}")
        for name, entry in sorted(entries, key=lambda x: x[0]):
            reason = (entry.get("reason") or "")[:90]
            retries = entry.get("retry_count")
            extras = []
            if retries is not None:
                extras.append(f"retries={retries}")
            extras_s = f" [{', '.join(extras)}]" if extras else ""
            print(f"    {name:<35}{extras_s}  {reason}")
        print()
    return 0
