# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build reviewable candidate tables from saved measurements; no device access."""

import csv
import json
import statistics
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            continue
        for result in data:
            if not isinstance(result, dict) or "timings" not in result:
                continue
            row = {"artifact": path.name}
            for key in (
                "implementation",
                "layer",
                "length",
                "batch",
                "activations",
                "prefill_pcc",
                "traced_decode_pcc",
                "trace_wait",
            ):
                row[key] = result.get(key)
            for key in ("warmed_prefill_ms", "traced_decode_ms"):
                values = result["timings"].get(key, [])
                row[key] = statistics.median(values) if values else None
            row["policy"] = json.dumps(result.get("policy", {}), sort_keys=True)
            rows.append(row)
    if rows:
        with (root / "candidate_results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    projections = []
    for path in sorted(root.glob("projection*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            continue
        for row in data:
            if "readers" not in row:
                continue
            keys = (
                "layer",
                "name",
                "readers",
                "cores",
                "block",
                "k",
                "n",
                "weight_dtype",
                "fidelity",
                "status",
                "pcc_quantized_control",
                "traced_us",
                "error",
            )
            projections.append({"artifact": path.name} | {key: row.get(key) for key in keys})
    if projections:
        with (root / "projection_results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=projections[0])
            writer.writeheader()
            writer.writerows(projections)


if __name__ == "__main__":
    summarize(Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder")
