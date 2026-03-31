#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config hash preflight check.

Verifies that the branch's config_hash computation in ``generic_ops_tracer.py``
produces identical hashes for a reconstructed master JSON.

Hash source of truth used by this preflight:
    model_tracer.generic_ops_tracer.recompute_config_hashes()
        -> _compute_config_hash()

If hashes differ, the variation comes from the *current branch's* hash
implementation in generic_ops_tracer (normalization + hash inputs), not from an
independent algorithm in this script. In other words, this preflight compares:
    stored hashes in input JSON
    vs
    hashes recomputed by generic_ops_tracer on this checkout

Usage:
    python model_tracer/config_hash_preflight.py <master.json>
    python model_tracer/config_hash_preflight.py <master.json> --allow-partial
    python model_tracer/config_hash_preflight.py <master.json> --report report.txt

Exit codes:
    0  All hashes match, or partial changes with --allow-partial
    1  100% of hashes changed (always fatal), or partial without --allow-partial
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from model_tracer.generic_ops_tracer import recompute_config_hashes


def run_preflight(json_path, allow_partial=False):
    """Compare stored config hashes against recomputed ones.

    Returns (total, changed_entries, decision).
    """
    with open(json_path, "r") as f:
        before = json.load(f)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copyfile(json_path, tmp_path)
    recompute_config_hashes(str(tmp_path))

    with open(tmp_path, "r") as f:
        after = json.load(f)
    tmp_path.unlink()

    changed_entries = []
    total = 0
    for op_name, op_data in before.get("operations", {}).items():
        before_cfgs = op_data.get("configurations", [])
        after_cfgs = after.get("operations", {}).get(op_name, {}).get("configurations", [])
        for i, (bc, ac) in enumerate(zip(before_cfgs, after_cfgs)):
            total += 1
            old = bc.get("config_hash")
            new = ac.get("config_hash")
            if old != new:
                changed_entries.append(
                    {
                        "operation": op_name,
                        "config_id": bc.get("config_id", i),
                        "old_hash": old,
                        "new_hash": new,
                    }
                )

    changed = len(changed_entries)
    all_changed = total > 0 and changed == total

    if all_changed:
        decision = "fail_all_changed"
    elif changed > 0 and not allow_partial:
        decision = "fail_partial_changed"
    elif changed > 0:
        decision = "continue_partial_changed"
    else:
        decision = "pass"

    return total, changed_entries, decision


def format_report(json_path, total, changed_entries, decision, allow_partial):
    """Format a human-readable preflight report."""
    changed = len(changed_entries)
    pct = (changed / total * 100.0) if total else 0.0
    lines = [
        "Config hash preflight",
        "  Hash source:     model_tracer.generic_ops_tracer.recompute_config_hashes()",
        "  Hash method:     _compute_config_hash()",
        "  Variation means: branch generic_ops_tracer hash logic differs from stored hashes",
        f"  JSON:            {json_path}",
        f"  Configs:         {total}",
        f"  Changed:         {changed} ({pct:.1f}%)",
        f"  Allow partial:   {allow_partial}",
        f"  Decision:        {decision}",
    ]
    if changed_entries:
        lines.append("")
        lines.append("Changed hashes:")
        for e in changed_entries:
            lines.append(f"  {e['operation']} config_id={e['config_id']}: " f"{e['old_hash']} -> {e['new_hash']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("json_path", type=Path, help="Reconstructed master JSON to check")
    parser.add_argument("--allow-partial", action="store_true", help="Continue when some (but not all) hashes changed")
    parser.add_argument("--report", type=Path, help="Write report to file (in addition to stdout)")
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        return 1

    total, changed_entries, decision = run_preflight(args.json_path, args.allow_partial)
    report = format_report(args.json_path, total, changed_entries, decision, args.allow_partial)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")

    print("")
    print("=" * 60)
    print(report, end="")
    print("=" * 60)

    return 1 if decision.startswith("fail") else 0


if __name__ == "__main__":
    sys.exit(main())
