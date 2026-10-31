"""Whole-layer device windows and explicit useful-work roofline estimates."""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def window(rows):
    device = [r for r in rows if r.get("DEVICE FW START CYCLE") and r.get("DEVICE FW END CYCLE")]
    if not device:
        raise ValueError("No device timestamps; host time cannot replace device time")
    ids = {r["DEVICE ID"] for r in device}
    if len(ids) > 1:
        assert len(ids) == 4, ids
        measured = [window([r for r in device if r["DEVICE ID"] == chip]) for chip in ids]
        return max(r[0] for r in measured), statistics.median(r[1] for r in measured), sum(r[2] for r in measured)
    scales = [
        float(r["DEVICE FW DURATION [ns]"]) / (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
        for r in device
        if float(r["DEVICE FW END CYCLE"]) > float(r["DEVICE FW START CYCLE"])
    ]
    ns_per_cycle = statistics.median(scales)
    elapsed = (
        (max(float(r["DEVICE FW END CYCLE"]) for r in device) - min(float(r["DEVICE FW START CYCLE"]) for r in device))
        * ns_per_cycle
        / 1000
    )
    return elapsed, ns_per_cycle, len(device)


def extract(rows, start, end):
    begin = next(i for i, r in enumerate(rows) if r["OP CODE"] == start)
    finish = next(i for i, r in enumerate(rows[begin + 1 :], begin + 1) if r["OP CODE"] == end)
    return rows[begin + 1 : finish]


def main(args):
    rows = list(csv.DictReader(Path(args.csv).open()))
    pre = extract(rows, "PERF_PREFILL", "PERF_PREFILL_END")
    dec = extract(rows, "PERF_DECODE", "PERF_DECODE_END")
    pre_us, cycle_ns, pre_ops = window(pre)
    sessions = defaultdict(list)
    for row in dec:
        if row.get("DEVICE FW START CYCLE"):
            key = (row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])
            assert all(key), "Decode window contains untraced device work"
            sessions[key].append(row)
    assert len(sessions) == 128, f"Expected 128 decoded steps, found {len(sessions)}"
    sets = [{r["GLOBAL CALL COUNT"] for r in group} for group in sessions.values()]
    assert all(ids == sets[0] for ids in sets), "Trace replay op-ID sets differ"
    host_data = Path(args.trace_host_data).read_text()
    capture = host_data.split("TT_METAL_TRACE_BEGIN:", 1)[1].split("TT_METAL_TRACE_END:")[0]
    captured_ids = set()
    for line in capture.splitlines():
        if "TT_DNN_DEVICE_OP:" in line:
            match = re.search(r", (\d+)`;", line)
            assert match, "Expected warmed cached-op capture metadata"
            captured_ids.add(match.group(1))
    assert captured_ids, "Capture metadata missing"
    assert captured_ids == sets[0], "Replay op IDs differ from the complete captured body"
    windows = [window(group)[0] for group in sessions.values()]
    decode_us = statistics.mean(windows)
    assert len({len(group) for group in sessions.values()}) == 1
    kind = args.layer_type
    sliding = kind == "sliding_attention"
    h, i, nq, d, nk = 5376, 21504, 32, 256 if sliding else 512, 16 if sliding else 4
    q, k = nq * d, nk * d
    b, s = args.batch, 4096
    projection_params = h * (2 * q + (2 if sliding else 1) * k + 3 * i)
    pairs = s * 1024 - 1024 * 1023 // 2 if sliding else s * (s + 1) // 2
    useful_flops = b * (2 * s * projection_params + 4 * q * pairs)
    # Norm/RoPE/activation FLOPs are omitted from useful compute (a small lower bound).
    weight_bytes = None  # Runtime rows below supply actual stored projection dtypes.
    live_tokens = 1024 if sliding else statistics.mean(range(4097, 4225))
    decode_sample = next(iter(sessions.values()))
    sdpa = next(r for r in decode_sample if "SdpaDecode" in r["OP CODE"])
    widths = {
        "BFLOAT16": 2,
        "FLOAT32": 4,
        "INT32": 4,
        "UINT32": 4,
        "UINT16": 2,
        "UINT8": 1,
        "BFLOAT8_B": 1088 / 1024,
        "BFLOAT4_B": 576 / 1024,
    }
    cache_width = widths[sdpa["INPUT_1_DATATYPE"]]
    kv_read_bytes = 2 * cache_width * b * k * live_tokens

    # Estimate physical DRAM I/O from every operation in one complete replay.
    # Each ordinary DRAM operand/output is transferred once; sharded L1 tensors
    # are excluded. Sparse cache operations are modeled by active pages/tiles,
    # rather than incorrectly charging the entire reserved cache on each step.
    def tensor_bytes(row, prefix):
        if "DRAM" not in row.get(prefix + "_MEMORY", ""):
            return 0
        dims = [row.get(prefix + "_" + d + "_PAD[LOGICAL]", "") for d in "WZYX"]
        if not all(dims):
            return 0
        sizes = [int(v.split("[")[0]) for v in dims]
        dtype = row[prefix + "_DATATYPE"]
        width = widths[dtype]
        if row["OP CODE"] == "BinaryNgDeviceOperation":
            # BinaryNg reuses programs across volumes, so profiler hash-cached
            # shapes can describe prefill. Decode residual/MLP rows are B<=32;
            # shared KV norm has B head groups and a broadcast gamma operand.
            if sizes[-1] in (h, i, h // 4, i // 4, 5632, 6144):
                return 32 * sizes[-1] * width
            if sizes[-1] == d:
                return (1 if prefix == "INPUT_1" else b) * 32 * d * width
            raise ValueError("Unknown binary decode geometry")
        return math.prod(sizes) * width

    io_by_op = {}
    for row in next(iter(sessions.values())):
        op = row["OP CODE"]
        if "UpdateCache" in op:
            continue
        prefixes = ["INPUT_" + str(j) for j in range(5)] + ["OUTPUT_0"]
        if "SdpaDecode" in op:
            prefixes = [v for v in prefixes if v not in ("INPUT_1", "INPUT_2")]
        io_by_op[op] = io_by_op.get(op, 0) + sum(tensor_bytes(row, v) for v in prefixes)
    cache_update_bytes = 4 * b * cache_width * k * 32
    page_bytes = 4 * b * (132 * 4 + 8)
    dram_bytes = sum(io_by_op.values()) + kv_read_bytes + cache_update_bytes + page_bytes
    weight_bytes = sum(tensor_bytes(row, "INPUT_1") for row in decode_sample if "Matmul" in row["OP CODE"])
    peak_flops = 4 * 120 * 4096 * 1.35e9
    matmul_rows = []
    role_shapes = {
        (5376, 4096): "qkv_local",
        (5376, 4608): "qk_local",
        (2048, 5632): "o_local",
        (4096, 5632): "o_local",
        (5376, 5632): "gate_or_up_local",
        (5632, 5632): "down_local",
        (5376, 5376): "gate_or_up_unpadded",
    }
    for row in decode_sample:
        if "Matmul" not in row["OP CODE"]:
            continue
        shape = [int(row["INPUT_1_" + axis + "_PAD[LOGICAL]"].split("[")[0]) for axis in "YX"]
        duration = float(row["DEVICE KERNEL DURATION [ns]"])
        matmul_rows.append(
            dict(
                op=row["OP CODE"],
                role=role_shapes.get(tuple(shape), "unknown"),
                weight_shape=shape,
                input_dtype=row.get("INPUT_0_DATATYPE"),
                weight_dtype=row.get("INPUT_1_DATATYPE"),
                output_dtype=row.get("OUTPUT_0_DATATYPE"),
                input_memory=row.get("INPUT_0_MEMORY"),
                weight_memory=row.get("INPUT_1_MEMORY"),
                output_memory=row.get("OUTPUT_0_MEMORY"),
                fidelity=row.get("MATH FIDELITY"),
                first_replay_duration_us=duration / 1000,
                fraction_of_whole_layer_pct=100 * duration / 1000 / windows[0],
                attributes=row.get("ATTRIBUTES"),
            )
        )
    if args.expected_weight_dtype:
        assert matmul_rows and all(r["weight_dtype"] == args.expected_weight_dtype for r in matmul_rows), matmul_rows
    if args.expected_fidelity:
        assert all(str(r["fidelity"]).lower() == args.expected_fidelity.lower() for r in matmul_rows), matmul_rows
    prefill_matmuls = {}
    for row in pre:
        if "Matmul" not in row["OP CODE"]:
            continue
        shape = [int(row["INPUT_1_" + axis + "_PAD[LOGICAL]"].split("[")[0]) for axis in "YX"]
        key = (tuple(shape), row.get("INPUT_0_DATATYPE"), row.get("OUTPUT_0_DATATYPE"))
        if key not in prefill_matmuls:
            prefill_matmuls[key] = dict(
                role=role_shapes.get(tuple(shape), "unknown"),
                weight_shape=shape,
                input_dtype=row.get("INPUT_0_DATATYPE"),
                weight_dtype=row.get("INPUT_1_DATATYPE"),
                output_dtype=row.get("OUTPUT_0_DATATYPE"),
                fidelity=row.get("MATH FIDELITY"),
                attributes=row.get("ATTRIBUTES"),
                count=0,
                kernel_us=0.0,
            )
        prefill_matmuls[key]["count"] += 1
        prefill_matmuls[key]["kernel_us"] += float(row["DEVICE KERNEL DURATION [ns]"]) / 1000
    if args.expected_weight_dtype:
        assert prefill_matmuls and all(
            r["weight_dtype"] == args.expected_weight_dtype for r in prefill_matmuls.values()
        )
    if args.expected_fidelity:
        assert all(str(r["fidelity"]).lower() == args.expected_fidelity.lower() for r in prefill_matmuls.values())
    ccl_rows = []
    ccl_by_device = defaultdict(list)
    for row in decode_sample:
        if "AllGather" in row["OP CODE"] or "ReduceScatter" in row["OP CODE"]:
            ccl_by_device[row["DEVICE ID"]].append(row)
    assert len(ccl_by_device) == 4
    assert {len(v) for v in ccl_by_device.values()} == {4}
    for group in zip(*ccl_by_device.values()):
        row = group[0]
        assert len({r["OP CODE"] for r in group}) == 1
        ccl_rows.append(
            dict(
                op=row["OP CODE"],
                global_call_counts=[r["GLOBAL CALL COUNT"] for r in group],
                payload_dtype=row.get("INPUT_0_DATATYPE"),
                input_memory=row.get("INPUT_0_MEMORY"),
                output_memory=row.get("OUTPUT_0_MEMORY"),
                axis=3,
                topology="Ring",
                num_links=2,
                num_workers_per_link=2,
                num_buffers_per_channel=1,
                chunks_per_sync="factory shape heuristic; see ccl_report.md",
                persistent_output=True,
                persistent_intermediate=(
                    "yes, RS helper-created staging and penult"
                    if "ReduceScatter" in row["OP CODE"]
                    else "not exposed by AG API"
                ),
                max_device_kernel_us=max(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in group),
                attributes=row.get("ATTRIBUTES"),
                decision="kept; persistent and topology controls measured",
            )
        )
    peak_dram = 4 * 512e9
    result = dict(
        ccl_runtime_rows=ccl_rows,
        layer_type=kind,
        batch=b,
        workload=dict(input_tokens=4096, output_tokens=128, concurrency=1 if b == 1 else b),
        prefill_device_us=pre_us,
        decode_device_us=decode_us,
        prefill_useful_flops=useful_flops,
        decode_dram_bytes=dram_bytes,
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * useful_flops / (peak_flops * pre_us / 1e6),
        decode_dram_pct=100 * dram_bytes / (peak_dram * decode_us / 1e6),
        peak_basis="Four Blackhole ASICs in the P300_X2 mesh, each: 120 cores, 4096 ops/cycle, 1.35GHz, LoFi peak; 512GB/s. Mixed-fidelity layer useful work is divided by the whole-device LoFi theoretical peak, not a per-op average. Official p300/QuietBox2 specs and tt-perf-report ArchitectureSpec.",
        source="warmed_layer_tracy",
        evidence=str(Path(args.csv)),
        device_window_basis="Maximum per-device FW end minus FW start across ALL layer ops, including gaps, for all4 participating devices. Chip clocks are not assumed synchronized. Decode is mean of 128 complete per-replay windows; gaps between separate replays excluded.",
        ns_per_cycle=cycle_ns,
        prefill_device_ops=pre_ops,
        decode_replay_count=len(sessions),
        captured_body_op_count=len(captured_ids),
        replay_op_ids_match_capture=True,
        decode_ops_per_replay=len(next(iter(sessions.values()))),
        decode_min_device_us=min(windows),
        decode_max_device_us=max(windows),
        assumptions="Dtypes and fidelity are taken from runtime rows; BF16 norms/residuals, dense active work. BFP tile sizes include exponent headers: BFP4 576 and BFP8 1088 bytes per 1024 elements. Useful FLOPs count all linear projections and causal QK/PV, omit small elementwise work. Estimated DRAM: physical operand/output bytes for every ordinary DRAM op once across all4 devices, active KV once, tile cache updates and page metadata; L1 operands excluded. Includes tile padding, excludes redundant per-core reads and cache reuse. BinaryNg hash-cached prefill shapes are replaced by actual decode B<=32 residual/MLP and KV-norm geometry. Weight reference bytes are informational and not added twice. Percentages are estimates, not counters.",
        matmul_runtime_rows=matmul_rows,
        prefill_matmul_runtime_rows=list(prefill_matmuls.values()),
        prefill_sdpa_runtime_rows=[
            dict(
                op=r["OP CODE"],
                device_kernel_us=float(r["DEVICE KERNEL DURATION [ns]"]) / 1000,
                attributes=r.get("ATTRIBUTES"),
            )
            for r in pre
            if r["OP CODE"] == "SDPAOperation"
        ],
        decode_sdpa_attributes=sdpa.get("ATTRIBUTES"),
        kv_cache_dtype=sdpa["INPUT_1_DATATYPE"],
        bandwidth_lower_bound_us=(weight_bytes + kv_read_bytes) / peak_dram * 1e6,
        bandwidth_lower_bound_basis="One read of projection weights plus active K/V at theoretical DRAM peak; excludes other ops, communication, and compute. Not a latency prediction.",
        aggregate_four_device_op_kernel_sum_us=statistics.mean(
            sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in group) / 1000 for group in sessions.values()
        ),
        dram_components=dict(
            weight_parameter_bytes_reference_only=weight_bytes,
            active_kv=kv_read_bytes,
            ordinary_op_physical_io=sum(io_by_op.values()),
            ordinary_io_by_op=io_by_op,
            cache_tile_update=cache_update_bytes,
            page_table=page_bytes,
        ),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--trace-host-data", required=True)
    p.add_argument("--layer-type", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-weight-dtype")
    p.add_argument("--expected-fidelity")
    main(p.parse_args())
