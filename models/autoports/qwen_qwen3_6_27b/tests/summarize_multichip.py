# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only final benchmark table, with raw artifact links and explicit units."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final_perf")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    doc = root / "doc/multichip_decoder"
    records = []
    for path in sorted(doc.glob(f"{args.tag}_sharded_l*_b*_s*.json")):
        if path.name.endswith(".command.json"):
            continue
        run = json.loads(path.read_text())
        baseline_path = path.with_name(path.name.replace("_sharded_", "_baseline_"))
        base = json.loads(baseline_path.read_text())
        assert run["pass"] and run["reference_metadata"] == base["reference_metadata"]
        assert run["source_sha256"] == hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest()
        assert not run["timing"]["tracking_enabled"] and not base["timing"]["tracking_enabled"]
        a, b = base["timing"], run["timing"]
        records.append(
            dict(
                kind="GDN" if run["layer"] == 0 else "Full attention",
                batch=run["batch"],
                length=run["length"],
                baseline_prefill_ms=a["prefill_ms"],
                tp4_prefill_ms=b["prefill_ms"],
                prefill_speedup=a["prefill_ms"] / b["prefill_ms"],
                prefill_efficiency=a["prefill_ms"] / b["prefill_ms"] / 4,
                baseline_decode_ms=a["decode_trace_ms"],
                tp4_decode_ms=b["decode_trace_ms"],
                decode_speedup=a["decode_trace_ms"] / b["decode_trace_ms"],
                decode_efficiency=a["decode_trace_ms"] / b["decode_trace_ms"] / 4,
                prefill_pcc=run["prefill"],
                decode_pcc=run["decode"],
                artifact=path.name,
                baseline_artifact=baseline_path.name,
            )
        )
    assert records
    with (doc / "performance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "decoder_sha256": hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        "regime": "serial, warmed trace, no watcher/profiler/allocation tracking; 100 queued replays",
        "units": "milliseconds",
        "efficiency_definition": "single-chip latency / TP4 latency / 4",
        "policy_note": "Speedup includes TP4 sharding, compact local kernels and PCC-validated LoFi attention; the frozen single-chip GDN baseline uses HiFi4 attention.",
        "rows": records,
    }
    (doc / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "| Layer | B | S | Prefill single → TP4 ms | Decode single → TP4 ms | Prefill speedup / efficiency | Decode speedup / efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['kind']} | {r['batch']} | {r['length']} | {r['baseline_prefill_ms']:.3f} → {r['tp4_prefill_ms']:.3f} | {r['baseline_decode_ms']:.3f} → {r['tp4_decode_ms']:.3f} | {r['prefill_speedup']:.2f}× / {r['prefill_efficiency']:.1%} | {r['decode_speedup']:.2f}× / {r['decode_efficiency']:.1%} |"
        )
    lines += ["", report["policy_note"]]
    (doc / "performance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
