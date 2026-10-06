# SPDX-License-Identifier: Apache-2.0
"""Build compact candidate and same-run profiler accounting from raw artifacts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"


def main():
    candidates = []
    for path in sorted(ROOT.glob("**/profile_*.json")):
        data = json.loads(path.read_text())
        if "host_elapsed_ms" not in data:
            continue
        candidates.append(
            dict(
                artifact=str(path.relative_to(ROOT)),
                layer=data["layer"],
                prefill_tokens=data.get("prefill_tokens", 128),
                public_api=data.get("public_api", False),
                prefill_ms=data["host_elapsed_ms"]["prefill"],
                decode_ms=data["host_elapsed_ms"]["decode"],
                min_pcc=min(row["pcc"] for row in data["pcc"]),
                decoder_hash=data["provenance"]["source_sha256"]["tt/optimized_decoder.py"],
            )
        )
    (ROOT / "candidate_index.json").write_text(json.dumps(candidates, indent=2) + "\n")
    with (ROOT / "candidate_index.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    summaries = []
    for layer in (0, 4):
        directory = ROOT / f"tracy/layer_{layer}"
        profile = json.loads((ROOT / f"tracy_final_{layer}/profile_{layer}.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())["decode"]
        host_ms = profile["host_elapsed_ms"]["decode"]
        rows = list(csv.DictReader((directory / "decode_perf_report.csv").open()))
        raw = {
            int(float(r["GLOBAL CALL COUNT"])): r
            for r in csv.DictReader((directory / "decode_ops.csv").open())
            if r.get("GLOBAL CALL COUNT")
        }
        repetitions = profile["repetitions"]
        first = rows[: len(rows) // repetitions]
        ledger = []
        for row in first:
            original = raw[int(float(row["Global Call Count"]))]
            ledger.append(dict(row, attributes=original["ATTRIBUTES"]))
        (directory / "decode_operation_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        router_dtype = profile["provenance"]["resolved_policy"].get("router_dtype", "float32")
        tile_bytes = {"float32": 4096, "bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}
        weights = 34652160 + 80 * 12 * tile_bytes[router_dtype]
        kv = 278528
        floor_ms = (weights + kv) / 512e9 * 1000
        summaries.append(
            dict(
                layer=layer,
                source=str(directory.relative_to(ROOT)),
                host_source=f"tracy_final_{layer}/profile_{layer}.json",
                decode_host_ms=host_ms,
                decode_device_kernel_ms=summary["kernel_ms"],
                decode_device_gap_ms=summary["gap_ms"],
                residual_host_ms=host_ms - summary["kernel_ms"] - summary["gap_ms"],
                decode_ops=summary["ops"],
                modeled_weight_bytes=weights,
                modeled_kv_read_bytes=kv,
                modeled_bandwidth_floor_ms=floor_ms,
                modeled_fraction_of_peak=floor_ms / host_ms,
                peak_gb_s=512,
                note="Stored tile headers included; excludes norm/page/layout/write traffic. Modeled effective fraction is not a measured DRAM counter.",
            )
        )
    (ROOT / "performance_accounting.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
