# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write compact local telemetry from existing final-path evidence only."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[4]
packet = (
    repo
    / "bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/1a0f19aa-a217-42b9-acac-1a58fe787838.json"
)
data = json.loads(packet.with_suffix(".template.json").read_text())
data["selected_mesh"] = [1, 4]
runtime_hash = hashlib.sha256((root.parents[1] / "tt/multichip_decoder.py").read_bytes()).hexdigest()
for short, kind in [("sliding", "sliding_attention"), ("full", "full_attention")]:
    profile = root / ("profile_final_" + short) / "whole_layer.json"
    measured = root / ("final_" + short + ".json")
    if profile.exists() and json.loads((profile.parent / "run.json").read_text())["runtime_sha256"] == runtime_hash:
        source = json.loads(profile.read_text())
        assert source["target_workload_measured"] and source["selected_mesh"] == [1, 4]
        keys = [
            "prefill_device_us",
            "decode_device_us",
            "prefill_flops_pct",
            "decode_dram_pct",
            "prefill_useful_flops",
            "decode_dram_bytes",
            "peak_flops_per_s",
            "peak_dram_bytes_per_s",
            "peak_basis",
        ]
        data["layer_types"][kind] = {key: source[key] for key in keys}
        data["layer_types"][kind].update(source="single-layer traced TP4", evidence=str(profile.relative_to(repo)))
    else:
        data["layer_types"][kind] = dict(
            prefill_device_us=None, decode_device_us=None, prefill_flops_pct=None, decode_dram_pct=None
        )
        data["missing_reasons"][
            kind
        ] = "Final-default whole-layer device profile not yet available; host timings are not device measurements."
    if measured.exists() and json.loads(measured.read_text())["runtime_sha256"] == runtime_hash:
        source = json.loads(measured.read_text())
        assert source["length"] == 4096 and source["steps"] == 128 and source["trace"] and source["passed"]
        for phase, value, count in [("prefill", source["pcc"][0], 1), ("decode", min(source["pcc"][1:]), 128)]:
            data["accuracy"].append(
                dict(
                    name="minimum_output_pcc_vs_TP1",
                    value=value,
                    unit="PCC",
                    phase=phase,
                    layer_type=kind,
                    completed_samples=count,
                    dataset_samples=None,
                    scope="subset",
                    evidence=str(measured.relative_to(repo)),
                )
            )
data["missing_reasons"][
    "accuracy.dataset_samples"
] = "Layer fixture comparisons cover one prefill and128 decode positions per kind; no benchmark-dataset population is defined."
data["notes"] = [
    "Implementation/evidence commit dc254cbde8997f0ec8f8f42bccb5aaf1f0e8d69e; independent stage_review.md clean-pass.",
    "Decoder stage only; real TP4 active-expert execution. No full-model or serving result.",
    "Rooflines use complete firmware layer windows including gaps, four-ASIC theoretical peaks, useful active8 FLOPs and estimated DRAM transfers, not controller counters.",
    "33-token adjacent sliding stack minimum PCC0.995049733 has limited margin. Capacity reservations do not execute the complete model allocation order.",
    "Stack accuracy, batch32, nonaligned lengths,262144 context and Watcher pass. Independent stage-review clean-pass; see optimized_multichip_decoder/stage_review.md.",
]
packet.write_text(json.dumps(data, indent=2) + "\n")
print(packet)
