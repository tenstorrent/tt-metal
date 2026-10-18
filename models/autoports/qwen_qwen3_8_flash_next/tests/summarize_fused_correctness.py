# SPDX-License-Identifier: Apache-2.0
"""Preserve row-level FP64 comparisons, including regressions hidden by minima."""
import argparse
import hashlib
import json
from pathlib import Path

KEY = ("layer_idx", "batch", "mode", "position", "length")


def read(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "pcc" in value and "layer_idx" in value:
            rows.append(value)
    if not rows:
        raise ValueError(f"No correctness rows in {path}")
    assert all(row["pcc"] >= row["threshold"] for row in rows)
    assert all(row["pcc_method"] == "two_pass_centered_fp64" for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, selected = read(args.baseline), read(args.selected)
    base = {tuple(r.get(k) for k in KEY): r for r in baseline}
    comparisons = []
    for key, row in {tuple(r.get(k) for k in KEY): r for r in selected}.items():
        if key in base:
            old = base[key]
            comparisons.append(
                dict(
                    case=dict(zip(KEY, key)),
                    baseline_pcc=old["pcc"],
                    fused_pcc=row["pcc"],
                    pcc_delta=row["pcc"] - old["pcc"],
                    baseline_rmse=old["rmse"],
                    fused_rmse=row["rmse"],
                    rmse_ratio=row["rmse"] / old["rmse"] if old["rmse"] else None,
                )
            )
    result = dict(
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline, args.selected)},
        layers={
            str(layer): dict(
                checks=sum(r["layer_idx"] == layer for r in selected),
                minimum_pcc=min(r["pcc"] for r in selected if r["layer_idx"] == layer),
            )
            for layer in sorted({r["layer_idx"] for r in selected})
        },
        comparisons=sorted(comparisons, key=lambda r: r["pcc_delta"]),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(layers=result["layers"], matched_cases=len(comparisons), worst_pcc_deltas=result["comparisons"][:5]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
