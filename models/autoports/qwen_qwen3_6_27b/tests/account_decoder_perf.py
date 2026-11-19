"""Reconcile one signposted decoder profile, its host timing, and stored bytes.

This is a bandwidth lower bound, not a claim that SFPU recurrence is free.
Unused setup weight copies and transient activation traffic are excluded.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

TILE_BYTES = {"BFLOAT8_B": 1088, "BFLOAT4_B": 576, "BFLOAT16": 2048, "FLOAT32": 4096}


def padded_shape(row, prefix):
    return [int(row[f"{prefix}_{axis}_PAD[LOGICAL]"].split("[")[0]) for axis in "WZYX"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--recurrent-dtype", choices=["BFLOAT16", "FLOAT32"], default="BFLOAT16")
    parser.add_argument("--kv-dtype", choices=["BFLOAT8_B", "BFLOAT16"], default="BFLOAT8_B")
    parser.add_argument("--bandwidth-gbs", type=float, default=512.0)
    args = parser.parse_args()
    metrics = json.loads((args.profile / "metrics.json").read_text())
    result = json.loads(Path(metrics["correctness_result"]).read_text())
    with (args.profile / "ops.csv").open() as stream:
        raw = {row["GLOBAL CALL COUNT"]: row for row in csv.DictReader(stream)}
    with (args.profile / "decode_perf_report.csv").open() as stream:
        decode = list(csv.DictReader(stream))
    weights = []
    for row in decode:
        if "Matmul" not in row["OP Code"]:
            continue
        op = raw[row["Global Call Count"]]
        shape = padded_shape(op, "INPUT_1")
        dtype = op["INPUT_1_DATATYPE"]
        assert op["INPUT_1_LAYOUT"] == "TILE"
        assert math.prod(shape) % 1024 == 0
        weights.append(
            {
                "global_call_count": row["Global Call Count"],
                "physical_shape": shape,
                "dtype": dtype,
                "fidelity": op["MATH FIDELITY"],
                "memory": op["INPUT_1_MEMORY"],
                "bytes": math.prod(shape) // 1024 * TILE_BYTES[dtype],
            }
        )
    assert weights, "No measured decode matmul rows"
    batch = result["batch"]
    config = json.loads((Path(__file__).parents[1] / "doc/hf_config.json").read_text())["text_config"]
    linear = config["layer_types"][result["layer"]] == "linear_attention"
    if linear:
        recurrent = batch * 48 * 128 * 128 // 1024 * TILE_BYTES[args.recurrent_dtype]
        cache = {
            "recurrent_read": recurrent,
            "recurrent_write": recurrent,
            "convolution_history_read": batch * 3 * 10240 * 2,
            "convolution_history_write": batch * 3 * 10240 * 2,
        }
    else:
        context = metrics["decode_context"]
        # Minimum unique K/V payload; padded SDPA blocks and repeated GQA reads
        # cannot inflate the roofline used for the optimization stop rule.
        cache = {
            "logical_kv_read": batch * 2 * context * 4 * 256 * TILE_BYTES[args.kv_dtype] / 1024,
            "logical_kv_append": batch * 2 * 4 * 256 * TILE_BYTES[args.kv_dtype] / 1024,
        }
    total = sum(row["bytes"] for row in weights) + sum(cache.values())
    roofline_ms = total / (args.bandwidth_gbs * 1e6)
    wall = statistics.median(result["trace_replay_wall_ms"])
    device = metrics["decode"]["device_kernel_sum_us"] / 1000
    gaps = metrics["decode"]["op_gap_sum_us"] / 1000
    output = {
        "provenance": result["provenance"],
        "layer_kind": "linear_attention" if linear else "full_attention",
        "batch": batch,
        "compact_io": result.get("compact_io", False),
        "decode_context": metrics["decode_context"],
        "bandwidth_GB_per_second": args.bandwidth_gbs,
        "bandwidth_scope": "one assigned Blackhole chip, decimal GB/s",
        "measured_weight_reads": weights,
        "cache_bytes": cache,
        "minimum_bytes_per_decode_step": total,
        "roofline_ms": roofline_ms,
        "device_kernel_ms": device,
        "device_gap_ms": gaps,
        "host_trace_median_ms_same_run": wall,
        "host_minus_kernel_and_gap_ms": wall - device - gaps,
        "roofline_fraction_of_host": roofline_ms / wall,
        "roofline_fraction_of_device_window": roofline_ms / (device + gaps),
        "limitations": [
            "Bandwidth-only lower bound; SFPU/compute cost is additional.",
            "Profiler enabled; use tracking/profiler-off rows for candidate ranking.",
            "Unused setup weight copies and transient activations excluded.",
            "BFP tile exponent storage is included.",
        ],
    }
    target = args.profile / "accounting.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(
        target,
        f"roofline={roofline_ms:.6f} ms device={device+gaps:.6f} ms host={wall:.6f} ms fraction={roofline_ms/wall:.3%}",
    )


if __name__ == "__main__":
    main()
