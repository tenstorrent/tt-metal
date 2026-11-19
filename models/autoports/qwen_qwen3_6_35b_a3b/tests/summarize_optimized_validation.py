# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build compact current-result correctness and matched latency tables."""

import csv
import json
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/optimized_decoder"


def main():
    rows = []
    for layer in (0, 3):
        for length in (127, 128, 129, 256, 257, 4095, 4096, 4097):
            suffix = str(layer) if length == 256 else f"l{layer}_s{length}"
            baseline = json.loads((DOC / f"baseline_{suffix}.json").read_text())
            final = json.loads((DOC / f"final_{suffix}.json").read_text())
            assert baseline["length"] == final["length"] == length
            assert baseline["context"] == final["context"]
            # Mirror the inherited start_pos=0 planner, including singleton
            # full-attention tails; ceil(S/256) undercounts these calls.
            sizes, offset = [], 0
            while offset < length:
                remaining = length - offset
                if layer == 0:
                    n = min(64, remaining)
                else:
                    n = min(256, remaining // 32 * 32) if offset % 256 == 0 and remaining >= 32 else 1
                sizes.append(n)
                offset += n
            assert max(sizes) == baseline["prefill_chunk_size"] == final["prefill_chunk_size"]
            rows.append(
                dict(
                    layer=layer,
                    kind=final["kind"],
                    logical_length=length,
                    context=final["context"],
                    physical_chunk_max=max(sizes),
                    chunks=len(sizes),
                    logical_call_sizes=json.dumps(sizes),
                    singleton_calls=sizes.count(1),
                    baseline_prefill_ms=baseline["prefill_warmed_ms"],
                    optimized_prefill_ms=final["prefill_warmed_ms"],
                    prefill_improvement_percent=100 * (1 - final["prefill_warmed_ms"] / baseline["prefill_warmed_ms"]),
                    baseline_decode_ms=baseline["decode_traced_warmed_ms"],
                    optimized_decode_ms=final["decode_traced_warmed_ms"],
                    baseline_prefill_pcc=baseline["prefill_pcc"],
                    optimized_prefill_pcc=final["prefill_pcc"],
                    baseline_decode_pcc=baseline["decode_traced_pcc"],
                    optimized_decode_pcc=final["decode_traced_pcc"],
                )
            )
    with (DOC / "latency_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stress = []
    for layer in (0, 3, 34):
        for prefix in ("stress", "stress_extended"):
            path = DOC / f"{prefix}_{layer}.json"
            d = json.loads(path.read_text())
            assert d["gate_pass"]
            stress.append(
                dict(
                    layer=layer,
                    decode_length=d["decode_length"],
                    repeats=d["repeats"],
                    prefill_pcc=d["prefill_pcc"],
                    decode_whole_pcc=d.get("decode_whole_pcc"),
                    continuation_whole_pcc=d.get("continuation_whole_pcc"),
                    minimum_decode_row_pcc=min(r["pcc"] for r in d["results"]),
                    minimum_continuation_row_pcc=min(r["pcc"] for r in d["continuation_results"]),
                    gate_scope=d.get("gate_scope", "strict per-row .995"),
                    diagnostic_rows=d.get("row_diagnostics", []),
                    bit_equal=d["repeated_bit_equal"],
                    evidence=path.name,
                )
            )
    natural = []
    for layer in (0, 3, 34):
        d = json.loads((DOC / f"natural_final_{layer}.json").read_text())
        natural.append(
            dict(
                layer=layer,
                prefill_pcc=d["prefill_pcc"],
                decode_pcc=d["decode_traced_pcc"],
                activation_source=d["activation_source"],
            )
        )
    (DOC / "validation_summary.json").write_text(
        json.dumps(
            dict(
                latency_comparison="latency_comparison.csv",
                stress=stress,
                natural_activations=natural,
                watcher=json.loads((DOC / "watcher_audit.json").read_text()),
            ),
            indent=2,
        )
        + "\n"
    )
    print("VALIDATION_SUMMARY_WRITTEN")


if __name__ == "__main__":
    main()
