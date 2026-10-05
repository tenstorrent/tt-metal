"""Summarize reduced full-model signposts without mixing device clock epochs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-split-size", type=int, default=16384)
    parser.add_argument("--head-workers", type=int, default=1)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.ops_csv.open()))
    windows = {}
    active = None
    for row in rows:
        code = row["OP CODE"]
        if row["OP TYPE"] == "signpost":
            active = None if code.endswith("_END") else code.removeprefix("PERF_").lower()
            continue
        if active and row.get("DEVICE FW START CYCLE") and row.get("DEVICE FW END CYCLE"):
            windows.setdefault(active, []).append(row)
    all_rows = [row for window in windows.values() for row in window]
    frequencies = [
        (float(row["DEVICE FW END CYCLE"]) - float(row["DEVICE FW START CYCLE"]))
        / float(row["DEVICE FW DURATION [ns]"])
        * 1000
        for row in all_rows
        if float(row["DEVICE FW DURATION [ns]"]) > 0
    ]
    frequency = round(statistics.median(frequencies))
    assert frequency == 1350, frequencies
    summary = {
        "source": str(args.ops_csv),
        "frequency_mhz": frequency,
        "boundary": "maximum complete per-rank FW span; no cross-device clock subtraction",
    }
    for name, window in windows.items():
        devices = {}
        for rank in sorted({row["DEVICE ID"] for row in window}):
            rank_rows = [row for row in window if row["DEVICE ID"] == rank]
            start = min(int(row["DEVICE FW START CYCLE"]) for row in rank_rows)
            end = max(int(row["DEVICE FW END CYCLE"]) for row in rank_rows)
            devices[rank] = {
                "start_cycle": start,
                "end_cycle": end,
                "device_us": (end - start) / frequency,
                "ops": len(rank_rows),
            }
        summary[name] = {"device_us": max(row["device_us"] for row in devices.values()), "per_device": devices}
    fields = [
        "OP CODE",
        "MATH FIDELITY",
        "INPUT_0_DATATYPE",
        "INPUT_1_DATATYPE",
        "INPUT_1_Y_PAD[LOGICAL]",
        "INPUT_1_X_PAD[LOGICAL]",
        "DEVICE KERNEL DURATION [ns]",
        "ATTRIBUTES",
    ]
    model_rows = [row for row in windows["model"] if row["DEVICE ID"] == "0"]
    heads = [
        row
        for row in model_rows
        if row["OP CODE"] == "MatmulDeviceOperation"
        and row["INPUT_1_X_PAD[LOGICAL]"] == f"{args.head_split_size}[{args.head_split_size}]"
    ]
    assert len(heads) == 65536 // args.head_split_size
    for row in heads:
        assert row["MATH FIDELITY"] == "HiFi2" and row["INPUT_1_DATATYPE"] == "BFLOAT8_B"
        assert f"num_workers_per_dram_bank={args.head_workers}" in row["ATTRIBUTES"]
    policy = {
        "frequency_mhz_from_fw_cycles_and_ns": frequency,
        "ops_csv": str(args.ops_csv),
        "head_matmuls": [{key: row[key] for key in fields} for row in heads],
        "decoder_matmuls": [
            {key: row[key] for key in fields}
            for row in model_rows
            if row["OP CODE"] == "MatmulDeviceOperation" and row not in heads
        ],
        "cache_update": [
            {key: row[key] for key in ["OP CODE", "INPUT_0_DATATYPE", "INPUT_1_DATATYPE", "OUTPUT_0_DATATYPE"]}
            for row in model_rows
            if "UpdateCache" in row["OP CODE"]
        ],
    }
    # This profile deliberately includes selected layers0/1, one of each MLP
    # precision class. Assert measured rows rather than constructor metadata.
    decoder = policy["decoder_matmuls"]
    assert len(decoder) == 10
    expected_shapes = [(4096, 1536), (1024, 4096), (4096, 3072), (4096, 3072), (3072, 4096)]
    for index, row in enumerate(decoder):
        k, n = expected_shapes[index % 5]
        low_precision = index in (7, 8)
        assert row["INPUT_1_Y_PAD[LOGICAL]"] == f"{k}[{k}]"
        assert row["INPUT_1_X_PAD[LOGICAL]"] == f"{n}[{n}]"
        assert row["INPUT_1_DATATYPE"] == ("BFLOAT4_B" if low_precision else "BFLOAT8_B")
        assert row["MATH FIDELITY"] == ("LoFi" if low_precision else "HiFi2")
    for row in decoder + policy["head_matmuls"]:
        assert row["INPUT_0_DATATYPE"] == "BFLOAT16"
        assert "fp32_dest_acc_en=1" in row["ATTRIBUTES"]
    assert len(policy["cache_update"]) == 2
    for row in policy["cache_update"]:
        assert row["INPUT_0_DATATYPE"] == row["OUTPUT_0_DATATYPE"] == "BFLOAT8_B"
        assert row["INPUT_1_DATATYPE"] == "BFLOAT16"
    policy["policy_assertions"] = {
        "pass": True,
        "scope": "Measured layers0/1:10 decoder matmuls,8 head splits,2 fused cache updates; dtype/fidelity/FP32 assertions are enforced by this analyzer.",
    }
    args.output_dir.mkdir(exist_ok=True)
    for name, value in [("spans.json", summary), ("runtime_policy.json", policy)]:
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({name: summary[name]["device_us"] for name in windows}), flush=True)


if __name__ == "__main__":
    main()
