# SPDX-License-Identifier: Apache-2.0
"""Regenerate compact Stage5 candidate and final performance tables."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def main():
    baseline = json.loads((DOC / "before_0_128.json").read_text())["policy"]
    rows = []
    for p in sorted(DOC.glob("*.json")):
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "latency_ms" not in data:
            continue
        rows.append(
            dict(
                artifact=p.name,
                layer=data["layer"],
                tokens=data["tokens"],
                prefill_ms=data["latency_ms"]["prefill"],
                decode_ms=data["latency_ms"]["decode"],
                minimum_pcc=min(data["pcc"].values()) if data["pcc"] else None,
                repetitions=data["repetitions"],
                changed_policy=json.dumps(
                    {k: v for k, v in data["policy"].items() if baseline.get(k) != v}, sort_keys=True
                ),
                options=json.dumps(data.get("options", data.get("split", {})), sort_keys=True),
            )
        )
    with (DOC / "candidate_index.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Candidate measurements",
        "",
        "All latency rows below are complete TP4 layers unless the artifact explicitly says baseline. See job JSON and command/provenance logs for policy and adapted candidate source. PCC uses accepted real-weight reference outputs.",
        "",
        "| Artifact | Prefill ms | Traced decode ms | Minimum PCC | Replays |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["minimum_pcc"] is not None:
            lines.append(
                f"| {r['artifact']} | {r['prefill_ms']:.6f} | {r['decode_ms']:.6f} | {r['minimum_pcc']:.8f} | {r['repetitions']} |"
            )
    (DOC / "candidate_measurements.md").write_text("\n".join(lines) + "\n")
    final = []
    for layer in (0, 4):
        for tokens in (128, 8193):
            p = DOC / f"final_{layer}_{tokens}.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text())
            b = json.loads((DOC / f"before_{layer}_{tokens}.json").read_text())
            final.append(
                dict(
                    layer=layer,
                    tokens=tokens,
                    before_ms=b["latency_ms"],
                    after_ms=a["latency_ms"],
                    pcc=a["pcc"],
                    physical_chunks=a["physical_chunks"],
                    speedup={k: b["latency_ms"][k] / a["latency_ms"][k] for k in ("prefill", "decode")},
                )
            )
    (DOC / "performance_summary.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
