"""Whole-layer rank spans and explicit traffic estimates; host standard library only."""

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
PEAK_FLOPS = 120 * 4096 * 1.35e9
PEAK_DRAM = 512e9


def window_rows(rows, start, end):
    starts = [i for i, r in enumerate(rows) if r["OP CODE"] == start and r["OP TYPE"] == "signpost"]
    ends = [i for i, r in enumerate(rows) if r["OP CODE"] == end and r["OP TYPE"] == "signpost"]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(f"Missing, repeated, or inverted window: {start}/{end}")
    ops = [r for r in rows[starts[0] + 1 : ends[0]] if r.get("DEVICE FW START CYCLE", "").strip()]
    if not ops:
        raise ValueError(f"No device timestamps: {start}")
    return ops


def window(rows, start, end, mhz):
    ops = window_rows(rows, start, end)
    spans = {}
    for device in sorted({r["DEVICE ID"] for r in ops}):
        dr = [r for r in ops if r["DEVICE ID"] == device]
        lo = min(int(r["DEVICE FW START CYCLE"]) for r in dr)
        hi = max(int(r["DEVICE FW END CYCLE"]) for r in dr)
        if hi <= lo or mhz <= 0:
            raise ValueError(f"Non-positive span or frequency: {start}, device {device}")
        spans[device] = {"start_cycle": lo, "end_cycle": hi, "device_us": (hi - lo) / mhz, "ops": len(dr)}
    # Device clocks need not share an epoch. Include every op and inter-op gap
    # within each rank, then select the slowest complete rank for this token.
    return {"device_us": max(d["device_us"] for d in spans.values()), "per_device": spans}


def tensor(row, port):
    if not row.get(port + "_MEMORY"):
        return None
    dims = [row[port + "_" + axis + "_PAD[LOGICAL]"] for axis in "WZYX"]
    return {
        "padded_shape": [int(v.split("[")[0]) for v in dims],
        "dtype": row[port + "_DATATYPE"].lower(),
        "memory": row[port + "_MEMORY"],
        "layout": row[port + "_LAYOUT"],
    }


def tensor_bytes(spec):
    elements = math.prod(spec["padded_shape"])
    dtype = spec["dtype"].lower()
    tile_bytes = {"bfloat4_b": 576, "bfloat8_b": 1088}
    if dtype in tile_bytes:
        return math.ceil(elements / 1024) * tile_bytes[dtype]
    sizes = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4, "uint32": 4, "uint16": 2, "uint8": 1}
    if dtype not in sizes:
        raise ValueError(f"Unknown traffic-estimate dtype: {dtype}")
    return elements * sizes[dtype]


def weight_key(spec):
    return (tuple(spec["padded_shape"][-2:]), spec["dtype"].lower())


def decode_traffic(ops, allocations, batch, context, devices):
    """Sum per-rank estimates; never count entire cache ports as accesses."""
    per_rank, weight_ledger, details = {}, {}, {}
    for device in devices:
        parts = defaultdict(int)
        observed_weights, records = [], []
        for row in (r for r in ops if r["DEVICE ID"] == device):
            code = row["OP CODE"]
            if code in (
                "AllGatherMatmulAsyncDeviceOperation",
                "MatmulReduceScatterAsyncDeviceOperation",
                "AllGatherMinimalMatmulAsyncOp",
                "AllGatherMinimalMatmulAsyncDeviceOperation",
            ):
                raise ValueError(
                    "Fused collective traffic needs its own port ledger; do not reuse the unfused estimate"
                )
            ports = {
                f"{kind}_{i}": tensor(row, f"{kind}_{i}")
                for kind, count in (("INPUT", 6), ("OUTPUT", 3))
                for i in range(count)
            }
            excluded = set()
            if "Matmul" in code:
                spec = ports["INPUT_1"]
                if spec is None or "_DRAM_" not in spec["memory"]:
                    raise ValueError(f"Expected allocated DRAM weight in {code}")
                observed_weights.append(spec)
                excluded.add("INPUT_1")
            if code == "SdpaDecodeDeviceOperation":
                for name in ("INPUT_1", "INPUT_2"):
                    spec = ports[name]
                    if spec is None:
                        raise ValueError("Missing SDPA cache port")
                    shape = spec["padded_shape"]
                    read = dict(spec, padded_shape=[batch, shape[1], math.ceil(context / 128) * 128, shape[3]])
                    parts["kv_attention_read_bytes"] += tensor_bytes(read)
                    excluded.add(name)
                # CORE COUNT includes idle grid cores: explicit conservative bound.
                for name, category in (("INPUT_4", "page_table_read_bytes"), ("INPUT_3", "position_read_bytes")):
                    spec = ports[name]
                    if spec and "_DRAM_" in spec["memory"]:
                        parts[category] += tensor_bytes(spec) * int(row["CORE COUNT"])
                        excluded.add(name)
            elif code == "PagedFusedUpdateCacheDeviceOperation":
                for name in ("INPUT_0", "INPUT_2"):
                    spec = ports[name]
                    shape = spec["padded_shape"]
                    tile_row = dict(spec, padded_shape=[batch, shape[1], 32, shape[3]])
                    amount = tensor_bytes(tile_row)
                    parts["kv_update_tile_read_bytes"] += amount
                    parts["kv_update_tile_write_bytes"] += amount
                excluded.update(("INPUT_0", "INPUT_2", "OUTPUT_0", "OUTPUT_1"))
                for name, category in (("INPUT_5", "page_table_read_bytes"), ("INPUT_4", "position_read_bytes")):
                    spec = ports[name]
                    if spec and "_DRAM_" in spec["memory"]:
                        parts[category] += tensor_bytes(spec) * int(row["CORE COUNT"])
                        excluded.add(name)
            elif "Cache" in code or "Sdpa" in code or "SDPA" in code:
                raise ValueError(f"Unmodeled decode cache/attention operation: {code}")
            elif code == "ReduceScatterMinimalAsyncDeviceOperation":
                # Outputs 0 and 2 are scratch; output 1 is the result. Only count
                # scratch belonging to an executed op, not unused resident buffers.
                for name in ("OUTPUT_0", "OUTPUT_2"):
                    spec = ports[name]
                    if spec and "_DRAM_" in spec["memory"]:
                        parts["ccl_intermediate_read_write_bytes"] += 2 * tensor_bytes(spec)
                        excluded.add(name)
            for name, spec in ports.items():
                if spec is not None and name not in excluded and "_DRAM_" in spec["memory"]:
                    amount = tensor_bytes(spec)
                    parts["other_recorded_dram_port_bytes"] += amount
                    records.append({"op": code, "port": name, "bytes": amount, **spec})
        recorded = [
            dict(padded_shape=shape, dtype=dtype, role=role)
            for role, phase, shape, dtype in allocations
            if phase == "decode"
        ]
        if recorded:
            if Counter(map(weight_key, observed_weights)) - Counter(map(weight_key, recorded)):
                raise ValueError(f"Executed decode weight ports exceed allocation ledger on device {device}")
            weights = observed_weights
            basis = "Executed padded weight ports matched against accuracy.json decode allocations; unused resident alternatives excluded"
        else:
            weights, basis = observed_weights, "executed padded weight ports; OptimizedDecoder default five projections"
            if len(weights) != 5:
                raise ValueError("Expected five baseline decode projections")
        parts["allocated_weight_read_bytes"] = sum(tensor_bytes(w) for w in weights)
        per_rank[device] = dict(parts)
        weight_ledger[device] = {"basis": basis, "weights": weights, "bytes": parts["allocated_weight_read_bytes"]}
        details[device] = records
    categories = sorted({key for parts in per_rank.values() for key in parts})
    aggregate = {key: sum(parts.get(key, 0) for parts in per_rank.values()) for key in categories}
    return {
        "aggregate_bytes": sum(aggregate.values()),
        "aggregate_breakdown_bytes": aggregate,
        "per_device_breakdown_bytes": per_rank,
        "weight_allocations": weight_ledger,
        "other_dram_port_ledger": details,
    }


def prefill_useful_flops(seq, batch):
    h, intermediate, heads, kv_heads, dim = 4096, 12288, 32, 8, 128
    params = h * ((heads + 2 * kv_heads) * dim + h) + 3 * h * intermediate
    pairs = seq * (seq + 1) // 2
    parts = {
        "dense_projection_flops": 2 * seq * params,
        "causal_qk_av_flops": 4 * heads * dim * pairs,
        "causal_softmax_flops": 5 * heads * pairs,
        "approximate_elementwise_flops": seq * (10 * h + 4 * (heads + kv_heads) * dim + 6 * intermediate),
    }
    return {key: value * batch for key, value in parts.items()}


def build_result(rows, mhz, accuracy):
    seq, steps, batch = (accuracy[k] for k in ("seq", "steps", "batch"))
    if (seq, steps, batch) != (4096, 128, 1) or accuracy.get("command_args", {}).get("stack", 1) != 1:
        raise ValueError("This roofline contract requires one layer, 4096 input, 128 decode, batch/concurrency 1")
    result = {
        "schema_version": 2,
        "workload": {"input_tokens": seq, "output_tokens": steps, "batch": batch, "concurrency": 1, "layers": 1},
        "clock_mhz": mhz,
        "paths": {},
    }
    useful = prefill_useful_flops(seq, batch)
    for label, count in (("baseline", 1), ("multichip", 4)):
        prefix = label.upper() + "_PERF_"
        pre = window(rows, prefix + "PREFILL", prefix + "PREFILL_END", mhz)
        decode = [window(rows, f"{prefix}DECODE_{i:03d}", f"{prefix}DECODE_{i:03d}_END", mhz) for i in range(steps)]
        devices = sorted(pre["per_device"])
        if len(devices) != count or any(sorted(w["per_device"]) != devices for w in decode):
            raise ValueError(f"Expected the same {count} participating devices in all {label} windows")
        mean_us = statistics.mean(w["device_us"] for w in decode)
        allocations = accuracy["measurements"][label].get("weight_allocations", [])
        if label == "multichip" and not allocations:
            raise ValueError("Multichip decode needs actual weight_allocations from accuracy.json")
        traffic = []
        for i in range(steps):
            ops = window_rows(rows, f"{prefix}DECODE_{i:03d}", f"{prefix}DECODE_{i:03d}_END")
            traffic.append(decode_traffic(ops, allocations, batch, seq + i + 1, devices))
        breakdown = {
            key: statistics.mean(t["aggregate_breakdown_bytes"].get(key, 0) for t in traffic)
            for key in sorted({key for t in traffic for key in t["aggregate_breakdown_bytes"]})
        }
        dram_bytes = sum(breakdown.values())
        result["paths"][label] = {
            "prefill": pre,
            "decode": decode,
            "decode_device_mean_us": mean_us,
            "layer_roofline": {
                "source": "layer",
                "participating_devices": devices,
                "participating_device_count": count,
                "prefill_useful_flops": sum(useful.values()),
                "prefill_flops_breakdown": useful,
                "prefill_flops_pct": 100 * sum(useful.values()) / (count * PEAK_FLOPS * pre["device_us"] / 1e6),
                "decode_estimated_dram_bytes_per_step": dram_bytes,
                "decode_dram_pct": 100 * dram_bytes / (count * PEAK_DRAM * mean_us / 1e6),
                "decode_aggregate_breakdown_mean_bytes": breakdown,
                "decode_aggregate_bytes_by_step": [t["aggregate_bytes"] for t in traffic],
                "decode_step_000_traffic_ledger": traffic[0],
                "aggregate_peak_flops_per_s": count * PEAK_FLOPS,
                "aggregate_peak_dram_bytes_per_s": count * PEAK_DRAM,
            },
        }
        burst_replays = accuracy["measurements"][label].get("stationary_burst_replays", 0)
        if burst_replays:
            burst = window(rows, prefix + "BURST", prefix + "BURST_END", mhz)
            result["paths"][label]["stationary_burst"] = {
                **burst,
                "replays": burst_replays,
                "device_us_per_replay": burst["device_us"] / burst_replays,
                "host_us_per_replay": accuracy["measurements"][label]["stationary_burst_host_us"],
                "scope": "Additional same-input last-position replays; dispatch-amortization control, not generated tokens",
            }
    a, b = (result["paths"][label] for label in ("baseline", "multichip"))
    result["speedup"] = {
        "prefill": a["prefill"]["device_us"] / b["prefill"]["device_us"],
        "decode": a["decode_device_mean_us"] / b["decode_device_mean_us"],
    }
    result["efficiency"] = {key: value / 4 for key, value in result["speedup"].items()}
    result["timing_basis"] = (
        "Per rank: (max DEVICE FW END CYCLE - min DEVICE FW START CYCLE)/CHIP_FREQ[MHz], "
        "including all layer ops and gaps. Use max rank span per window; decode averages all 128 maxima."
    )
    result["peak_basis"] = {
        "nominal_tensix_per_chip": 120,
        "available_workers_per_chip": 110,
        "flops_per_cycle_per_core_lofi": 4096,
        "nominal_clock_hz": 1.35e9,
        "dram_bytes_per_s_per_chip": PEAK_DRAM,
        "description": "Nominal whole-chip P300c LoFi peak, including reserved cores; mixed-fidelity ops remain in layer time.",
    }
    result["estimate_assumptions"] = [
        "FLOP%=100*aggregate useful FLOPs/(participating devices*per-chip peak FLOP/s*whole-layer seconds). "
        "BW%=100*mean aggregate estimated DRAM bytes/(participating devices*512 GB/s*mean whole-layer decode seconds).",
        "Useful prefill work counts each model projection once across ranks, causal QK/AV pairs only, five softmax "
        "operations per valid score, and approximate norm/residual/RoPE/SiLU scalar work; padded work is excluded. "
        "Prefill fusion changes operation grouping but not this useful-work numerator; prefill timing accepts every op type.",
        "Read each executed allocated decode weight once per rank, preserving duplicate gate/up entries and physical "
        "padding; BFP4 tiles are 576 bytes, BFP8 tiles 1088 bytes. Prefill-only and unused resident allocations are excluded.",
        "Each local KV head is read once across its split-K group, reused across four Q heads; include the complete final "
        "128-token block. Cache updates read and write one 32-token tile row per K/V head. Whole-cache metadata is excluded.",
        "SDPA page-table and position reads multiply port bytes by recorded CORE COUNT (110 in the first capture): "
        "a conservative bound including idle grid cores, not a count of active reader transactions. Cache-update metadata "
        "uses its recorded core count. DRAM alignment and controller transaction amplification are not modeled.",
        "Other materialized DRAM input/output ports are counted once at recorded padded size, including reshape, input, "
        "RoPE and attention output traffic; L1 ports are excluded. This is a graph estimate, not measured hardware counters.",
        "For each executed reduce-scatter, estimate one read/write pass of each reported DRAM intermediate allocation "
        "(outputs 0 and 2); allocation capacity is a proxy for scratch traffic, not a measured occupancy or transaction count. "
        "Pure-L1 all-gather contributes no DRAM bytes. Fabric/NoC byte traffic is excluded.",
        "No utilization is clamped and no per-op utilization percentages are averaged. A byte numerator for four devices "
        "is divided by four-device capacity while useful full-layer FLOPs are not multiplied by four.",
    ]
    return result


def file_provenance(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--accuracy", type=Path, help="Defaults to directory/accuracy.json")
    parser.add_argument("--ops-csv", type=Path, help="Required if directory has more than one profiler CSV")
    parser.add_argument("--capture-command", help="Exact profiling invocation to retain as provenance; never executed")
    args = parser.parse_args()
    root = args.directory
    candidates = list(root.glob("reports/*/ops_perf_results_*.csv"))
    if args.ops_csv is None and len(candidates) != 1:
        raise ValueError(f"Expected exactly one profiler CSV, found {len(candidates)}; select --ops-csv")
    source = args.ops_csv or candidates[0]
    accuracy_path = args.accuracy or root / "accuracy.json"
    accuracy = json.loads(accuracy_path.read_text())
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    device_log = root / ".logs/profile_log_device.csv"
    with device_log.open() as stream:
        header = "".join(next(stream) for _ in range(3))
    match = re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([\d.]+)", header)
    if not match:
        raise ValueError("Device log does not identify CHIP_FREQ[MHz]")
    result = build_result(rows, float(match.group(1)), accuracy)
    result["source_csv"] = str(source)
    report_source = root / "rank0_ops.csv"
    with report_source.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(r for r in rows if r["DEVICE ID"] == "0" or r["OP TYPE"] == "signpost")
    commands = []
    for label in ("baseline", "multichip"):
        for phase, start in (
            ("prefill", label.upper() + "_PERF_PREFILL"),
            ("decode", label.upper() + "_PERF_DECODE_000"),
        ):
            base = [
                "tt-perf-report",
                str(report_source),
                "--start-signpost",
                start,
                "--end-signpost",
                start + "_END",
                "--no-summary",
            ]
            for suffix, extra in (
                ("txt", []),
                ("console.log", ["--csv", str(root / f"{label}_{phase}_perf_report.csv")]),
            ):
                command = base + extra
                commands.append(command)
                with (root / f"{label}_{phase}_perf_report.{suffix}").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    sources = [
        Path(__file__),
        MODEL_ROOT / "tt/optimized_decoder.py",
        MODEL_ROOT / "tt/multichip_decoder.py",
        MODEL_ROOT / "tests/run_multichip.py",
        MODEL_ROOT / "tests/multichip_candidates.py",
        MODEL_ROOT / "tests/optimized_multichip_candidates.py",
        MODEL_ROOT / "tt/collective_buffers.py",
    ]
    result["provenance"] = {
        "analysis_command": shlex.join([sys.executable, *sys.argv]),
        "capture_command_as_supplied": args.capture_command,
        "workload_command_args_at_capture": accuracy.get("command_args"),
        "capture_source_sha256": {
            key: accuracy.get(key)
            for key in ("implementation_sha256", "runner_sha256", "candidate_sha256", "stage5_candidate_sha256")
        },
        "analysis_time_sources": [file_provenance(path) for path in sources],
        "ops_csv": file_provenance(source),
        "accuracy_json": file_provenance(accuracy_path),
        "rank0_table_csv": file_provenance(report_source),
        "device_log": {
            "path": str(device_log.resolve()),
            "size_bytes": device_log.stat().st_size,
            "header": header,
            "header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        },
        "table_commands": commands,
        "table_artifacts": [
            file_provenance(root / f"{label}_{phase}_perf_report.{suffix}")
            for label in ("baseline", "multichip")
            for phase in ("prefill", "decode")
            for suffix in ("txt", "csv", "console.log")
        ],
        "note": "Capture hashes are authoritative for the measured implementation; analysis-time source hashes may differ "
        "for historical captures. Missing historical command/hash fields are null. Rank0 tables are diagnostic; "
        "roofline timing and bytes use every participating rank. Device-log hash covers its header only.",
    }
    (root / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            label: {
                "prefill_us": path["prefill"]["device_us"],
                "decode_us": path["decode_device_mean_us"],
                "prefill_flops_pct": path["layer_roofline"]["prefill_flops_pct"],
                "decode_dram_pct": path["layer_roofline"]["decode_dram_pct"],
            }
            for label, path in result["paths"].items()
        }
    )
    print("speedup", result["speedup"])


if __name__ == "__main__":
    main()
