# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write the required local packet from existing evidence; never run measurements."""
import argparse
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parent
PACKET = Path(
    "/workspace/tt-metal/bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/a5411fee-0441-4640-a0ff-c23fc77b76a6.json"
)
FIELDS = (
    "prefill_device_us",
    "decode_device_us",
    "prefill_flops_pct",
    "decode_dram_pct",
    "prefill_useful_flops",
    "decode_dram_bytes",
    "peak_flops_per_s",
    "peak_dram_bytes_per_s",
    "peak_basis",
)


def relative(path):
    return str(path.relative_to(Path("/workspace/tt-metal")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--commit", action="append", default=[])
    args = parser.parse_args()
    result = json.loads(PACKET.with_suffix(".template.json").read_text())
    assert result["attempt_id"] == "a5411fee-0441-4640-a0ff-c23fc77b76a6"
    result["selected_mesh"] = dict(
        shape=[1, 1], physical_device_ids=[3], architecture="blackhole", participating_asics=1
    )
    version = next(
        candidate
        for candidate in ("v8", "v7", "v6", "v5")
        if all((DOC / f"tracy/actual_optimized_{candidate}_layer{layer}/whole_layer.json").exists() for layer in (0, 5))
    )
    accuracy_version = next(
        v
        for v in ("v8", "v7", "v6")
        if all((DOC / f"validated_{v}_headline_layer{layer}.json").exists() for layer in (0, 5))
    )
    for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
        path = DOC / f"tracy/actual_optimized_{version}_layer{layer}/whole_layer.json"
        if path.exists():
            measured = json.loads(path.read_text())
            assert all(measured["workload"][k] == v for k, v in result["workload"].items())
            result["layer_types"][kind] = {key: measured[key] for key in FIELDS}
            result["layer_types"][kind].update(source="single_layer_traced_teacher_forcing", evidence=relative(path))
        else:
            result["missing_reasons"][kind] = "Matching whole-layer target measurement unavailable."
        path = DOC / f"validated_{accuracy_version}_headline_layer{layer}.json"
        if path.exists():
            report = json.loads(path.read_text())
            assert report["passed"] and report["decode"]["passed"] and report["decode"]["steps"] == 128
            for phase, name, value, count in [
                ("prefill", "HF aggregate PCC", report["pcc"], 4096),
                ("decode", "Minimum per-position HF PCC", report["decode"]["min_pcc"], 128),
            ]:
                result["accuracy"].append(
                    dict(
                        name=name,
                        value=value,
                        unit="PCC",
                        phase=phase,
                        layer_type=kind,
                        completed_samples=count,
                        dataset_samples=count,
                        scope="full",
                        evidence=relative(path),
                    )
                )
    runtime = hashlib.sha256((DOC.parent.parent / "tt/optimized_decoder.py").read_bytes()).hexdigest()
    result["notes"] = [
        f"Selected single-layer runtime sha256 {runtime}; performance uses {version} native measurements. No full-model/token-out throughput claim.",
        "Whole-layer device windows include all operations and internal gaps; decode averages128 positions. Host timing is not substituted. Accuracy sample counts denote token rows in the recorded fixture.",
        "Useful FLOPs count8 active experts/token. Native DRAM estimates include BFP exponent storage and rounded KV reads; exclude extra per-core rereads, NoC and profiler traffic. See whole_layer.json assumptions.",
        "Common one-ASIC theoretical LoFi peak with mixed runtime fidelity. Sliding expert gate BFP8, full gate BFP4; both down BFP4/LoFi. Attention/cache BFP8 with selective higher fidelity.",
        "Evidence archives, when present, reconstruct original report paths byte-for-byte using doc/optimized_decoder/evidence_archives.json before CPU audits.",
        "Context262144 preserved; maximum-context HF checks cover291 selected rows. B32 correctness uses short contexts. Inherited unchanged branches, v6 repairs and full-only v8 prefill placement/blocking are mapped in source_delta_v8.json.",
    ]
    if not args.complete:
        result["missing_reasons"][
            "stage_completion"
        ] = "Independent review, final evidence packaging and local checkpoint completion remain in progress."
    else:
        assert version == "v8"
        review = (DOC / "STAGE_REVIEW.md").read_text()
        assert "Verdict: clean-pass" in review
        assert args.commit
    if version != "v8":
        result["missing_reasons"][
            "selected_runtime_performance"
        ] = f"Values are completed {version} target measurements; final v8 profile is pending. No v8 device latency is inferred."
    if args.commit:
        result["notes"].append("Local tt-metal commits: " + ", ".join(args.commit) + ". Never pushed.")
    PACKET.write_text(json.dumps(result, indent=2) + "\n")
    print(PACKET)


if __name__ == "__main__":
    main()
