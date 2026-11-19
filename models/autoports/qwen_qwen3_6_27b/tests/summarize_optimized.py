# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only ledger of measured decoder candidates; failed trials stay visible."""

import csv
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).parents[1] / "doc/optimized_decoder"
    rows = []
    for artifact in sorted(root.glob("*.json")):
        data = json.loads(artifact.read_text())
        if not isinstance(data, dict) or "warmed_prefill_wall_ms" not in data:
            continue
        equivalence = data.get("fused_equivalence") or data.get("functional_equivalence") or {}
        reference = data.get("equivalence_reference", "functional")
        hf_state = [v for values in data.get("hf_state_pcc", {}).values() for v in values]
        functional = [v for k, v in equivalence.items() if k != "state_pcc"]
        state = equivalence.get("state_pcc", [])
        p = data["provenance"]
        bars = [data["prefill_pcc"], data["decode_pcc"], data["updated_input_pcc"], *functional, *state, *hf_state]
        rows.append(
            {
                "artifact": artifact.name,
                "decoder_sha256": p["decoder_source_sha256"],
                "implementation": p["decoder_implementation"],
                "layer": data["layer"],
                "batch": data["batch"],
                "length": data["length"],
                "compact_io": data.get("compact_io", False),
                "prefill_pcc": data["prefill_pcc"],
                "decode_pcc": data["decode_pcc"],
                "equivalence_reference": reference,
                "minimum_baseline_pcc": min(functional) if functional else None,
                "minimum_hf_state_pcc": min(hf_state) if hf_state else None,
                "minimum_state_pcc": min(state) if state else None,
                "short_gate_pass": min(bars) >= 0.995
                and data["repeated_bit_equal"]
                and data["prefill_repeated_bit_equal"]
                and data["updated_input_changed_output"],
                "warmed_prefill_ms": data["warmed_prefill_wall_ms"],
                "traced_decode_median_ms": statistics.median(data["trace_replay_wall_ms"]),
                "repetitions": len(data["trace_replay_wall_ms"]),
                "tracking": p["environment"].get("TT_METAL_TRACE_ALLOC_TRACKING"),
                "profiler": p["environment"].get("TT_METAL_DEVICE_PROFILER"),
                "watcher": p["environment"].get("TT_METAL_WATCHER"),
                "policy": json.dumps(p.get("optimization_policy", {}), sort_keys=True),
                "argv": json.dumps(p["command_argv"]),
            }
        )
    if rows:
        with (root / "candidate_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} measured rows; short_gate_pass does not imply context/stress/stage acceptance")
    sequences = []
    for artifact in sorted(root.glob("*.json")):
        data = json.loads(artifact.read_text())
        if not isinstance(data, dict) or not all(
            k in data for k in ("minimum_per_user_pcc", "prefill_length", "trace_wall_median_ms")
        ):
            continue
        p = data["provenance"]
        sequences.append(
            {
                "artifact": artifact.name,
                "decoder_sha256": p["decoder_source_sha256"],
                "implementation": p["decoder_implementation"],
                "layer": data["layer"],
                "batch": data["batch"],
                "prefill_length": data["prefill_length"],
                "steps": data["steps"],
                "compact_io": data.get("compact_io", False),
                "prefill_pcc": data["prefill_pcc"],
                "minimum_per_user_pcc": data["minimum_per_user_pcc"],
                "sequence_gate_pass": min(data["prefill_pcc"], data["minimum_per_user_pcc"]) >= 0.995,
                "traced_decode_median_ms": data["trace_wall_median_ms"],
                "tracking": p["environment"].get("TT_METAL_TRACE_ALLOC_TRACKING"),
                "policy": json.dumps(p.get("optimization_policy", {}), sort_keys=True),
                "argv": json.dumps(p["command_argv"]),
            }
        )
    if sequences:
        with (root / "sequence_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=sequences[0])
            writer.writeheader()
            writer.writerows(sequences)
    print(f"{len(sequences)} sequence rows; compare equal batch, context, steps, tracking and I/O layouts")


if __name__ == "__main__":
    main()
