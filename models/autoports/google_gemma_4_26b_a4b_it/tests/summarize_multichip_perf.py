# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer device windows and explicit TP4 roofline accounting."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.summarize_perf import (
    is_native_sdpa_decode,
    native_sdpa_cache_reads,
    number,
    sparse_weight_groups,
    tensor_shape,
    traffic,
    useful_prefill_flops,
)


def tp4_cache_reads(row, position, layer_type, read_chunk_override=None):
    """Reuse the validated read-span calculation with this rank's local heads."""
    heads, reference_heads, width = (2, 8, 256) if layer_type == "sliding_attention" else (1, 2, 512)
    if tensor_shape(row, "INPUT_0", logical=True) != (1, 1, 4, width):
        raise ValueError("TP4 SDPA accounting requires four local query heads")
    reference = dict(row)
    reference["INPUT_0_Y_PAD[LOGICAL]"] = "32[16]"
    for prefix in ("INPUT_1", "INPUT_2"):
        shape = tensor_shape(row, prefix)
        if len(shape) != 4 or shape[1:] != (heads, 32, width):
            raise ValueError(f"Unexpected TP4 native {prefix} cache shape: {shape}")
        reference[f"{prefix}_Z_PAD[LOGICAL]"] = str(reference_heads)
    counts, basis = native_sdpa_cache_reads(reference, position, layer_type, read_chunk_override)
    divisor = reference_heads // heads
    counts = {prefix: count // divisor for prefix, count in counts.items()}
    for key in ("kv_dram_bytes", "allocated_kv_bytes"):
        basis[key] /= divisor
    basis["local_kv_heads"] = heads
    basis["local_query_heads"] = 4
    return counts, basis


def collective_traffic(row):
    """Exclude RS scratch allocations from the declared logical-I/O estimate."""
    if "reducescatterminimalasync" not in row["OP CODE"].lower():
        return traffic(row), 0
    if not tensor_shape(row, "OUTPUT_1"):
        raise ValueError("Minimal reduce-scatter metadata must expose its actual OUTPUT_1")
    # Native outputs are [intermediate, reduced_output, optional_penult_scratch].
    # Allocated scratch size is not a write count; internal CCL passes are excluded.
    adjusted = dict(row)
    for key in row:
        if key.startswith("OUTPUT_") and key.endswith("_MEMORY") and key != "OUTPUT_1_MEMORY":
            adjusted[key] = ""
    counted = traffic(adjusted)
    return counted, traffic(row) - counted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--layer-type", required=True, choices=["sliding_attention", "full_attention"])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--decode-start-position", type=int, default=4096)
    parser.add_argument("--native-sdpa-read-chunk-size", type=int)
    parser.add_argument("--precision-policy", help="Measured runtime precision/fidelity description")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.decode_start_position < 0:
        parser.error("Positive decode steps and a nonnegative start position are required")
    phase = None
    windows = defaultdict(list)
    for row in csv.DictReader(args.csv.open()):
        code = row["OP CODE"]
        if code in ("PERF_PREFILL", "PERF_DECODE"):
            phase = code.removeprefix("PERF_").lower()
        elif code in ("PERF_PREFILL_END", "PERF_DECODE_END"):
            phase = None
        elif phase and row["OP TYPE"] == "tt_dnn_device":
            assert number(row, "DEVICE FW START CYCLE") is not None
            windows[phase].append(row)
    summary = {}
    replay_rows = []
    native_reads = []
    sparse_reads = []
    collective_examples = []
    for phase, rows in windows.items():
        groups = defaultdict(list)
        for row in rows:
            session = row["METAL TRACE REPLAY SESSION ID"] if phase == "decode" else "prefill"
            groups[session, row["DEVICE ID"]].append(row)
        session_order = sorted(
            {key[0] for key in groups},
            key=lambda key: min(
                number(r, "DEVICE FW START CYCLE") for (s, _), rs in groups.items() if s == key for r in rs
            ),
        )
        assert len(session_order) == (args.steps if phase == "decode" else 1)
        if phase == "decode":
            assert all(session not in ("", "-", "nan") for session in session_order), "Missing trace session ID"
            assert len({len(ops) for ops in groups.values()}) == 1, "Incomplete trace operation records"
        replay_times = []
        replay_bytes = []
        replay_ccl_bytes = []
        replay_scratch_bytes = []
        for index, session in enumerate(session_order):
            device_times = []
            transfers = 0
            ccl_transfers = 0
            scratch_bytes = 0
            for (s, device), ops in groups.items():
                if s != session:
                    continue
                ratios = [
                    number(r, "DEVICE FW DURATION [ns]")
                    / (number(r, "DEVICE FW END CYCLE") - number(r, "DEVICE FW START CYCLE"))
                    for r in ops
                    if number(r, "DEVICE FW END CYCLE") > number(r, "DEVICE FW START CYCLE")
                ]
                ratio = statistics.median(ratios)
                duration = (
                    (
                        max(number(r, "DEVICE FW END CYCLE") for r in ops)
                        - min(number(r, "DEVICE FW START CYCLE") for r in ops)
                    )
                    * ratio
                    / 1000
                )
                device_times.append(duration)
                replay_rows.append(
                    dict(phase=phase, session=session, device=device, device_us=duration, native_ops=len(ops))
                )
                if phase == "decode":
                    sdpa_count = 0
                    for row in ops:
                        counts = None
                        if is_native_sdpa_decode(row):
                            counts, basis = tp4_cache_reads(
                                row,
                                args.decode_start_position + index,
                                args.layer_type,
                                args.native_sdpa_read_chunk_size,
                            )
                            native_reads.append(dict(session=session, device=device, **basis))
                            sdpa_count += 1
                        code = row["OP CODE"].lower()
                        if "sparsematmul" in code:
                            sparse = sparse_weight_groups(row)
                            assert sparse["active"] == 8, "Model decode must use exactly eight active experts"
                            if index == 0:
                                sparse_reads.append(dict(device=device, **sparse))
                        if "reducescatter" in code or "allgather" in code:
                            counted, excluded = collective_traffic(row)
                            ccl_transfers += counted
                            scratch_bytes += excluded
                            if index == 0:
                                collective_examples.append(
                                    dict(
                                        device=device,
                                        op=row["OP CODE"],
                                        input_shape=tensor_shape(row, "INPUT_0"),
                                        input_dtype=row["INPUT_0_DATATYPE"],
                                        estimated_io_bytes=counted,
                                        excluded_scratch_allocation_bytes=excluded,
                                    )
                                )
                            transfers += counted
                        else:
                            transfers += traffic(row, counts)
                    assert sdpa_count == 1, "Expected one native SDPA per device and decode replay"
            assert len(device_times) == 4
            replay_times.append(max(device_times))
            replay_bytes.append(transfers)
            replay_ccl_bytes.append(ccl_transfers)
            replay_scratch_bytes.append(scratch_bytes)
        summary[phase] = dict(
            device_us=statistics.mean(replay_times),
            samples=len(replay_times),
            min_device_us=min(replay_times),
            max_device_us=max(replay_times),
        )
        if phase == "decode":
            summary[phase]["estimated_dram_bytes"] = statistics.mean(replay_bytes)
            summary[phase]["estimated_collective_io_bytes"] = statistics.mean(replay_ccl_bytes)
            summary[phase]["excluded_collective_scratch_allocation_bytes"] = statistics.mean(replay_scratch_bytes)
    assert set(summary) == {"prefill", "decode"}
    flop_terms = useful_prefill_flops(args.layer_type)
    flops = sum(flop_terms.values())
    peak_flops = 4 * 120 * 4096 * 1.35e9
    peak_dram = 4 * 512e9
    result = dict(
        layer_type=args.layer_type,
        selected_mesh=[1, 4],
        source=str(args.csv),
        workload=dict(
            input_tokens=4096,
            output_tokens=args.steps,
            batch=1,
            concurrency=1,
            decode_start_position=args.decode_start_position,
        ),
        target_workload_measured=args.steps == 128 and args.decode_start_position == 4096,
        whole_layer_windows=summary,
        prefill_device_us=summary["prefill"]["device_us"],
        decode_device_us=summary["decode"]["device_us"],
        prefill_useful_flops=flops,
        useful_flops_terms=flop_terms,
        precision_policy=args.precision_policy,
        sparse_weight_reads=sparse_reads,
        collective_examples=collective_examples,
        native_sdpa_cache_reads=native_reads,
        decode_dram_bytes=summary["decode"]["estimated_dram_bytes"],
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * flops / (peak_flops * summary["prefill"]["device_us"] / 1e6),
        decode_dram_pct=100
        * summary["decode"]["estimated_dram_bytes"]
        / (peak_dram * summary["decode"]["device_us"] / 1e6),
        peak_basis="Four P300 Blackhole ASICs, theoretical120 Tensix/ASIC,4096 FLOP/core/cycle at1.35GHz LoFi reference;512GB/s DRAM/ASIC. Runtime mixes fidelities.",
        assumptions=[
            "Each device window spans first firmware start to final firmware end of the entire layer, including gaps. Mesh latency uses the maximum per-device span; clocks are not assumed cross-device synchronized. This excludes host input refresh/readback between replays.",
            "Useful work uses4096 logical tokens and8 selected experts; padding, inactive prefill union work, normalization/transcendentals excluded from numerator only. All their device time remains included.",
            "DRAM estimate sums native operand reads/writes across all four devices with indexed active-expert metadata, BFP tile exponent bytes, selected embedding/slice rows and paged writes. SDPA uses validated local heads and metadata-derived chunk-rounded reads. Minimal reduce-scatter counts INPUT_0 and its real OUTPUT_1; scratch OUTPUT_0/OUTPUT_2 allocations are excluded, not treated as writes. Grouped CCL payloads and attention dtype follow recorded shapes and dtypes. Extra per-core reads, internal CCL scratch passes/NoC and profiler traffic excluded; not controller counters.",
            "No utilization value is clamped. These rooflines are estimates, not averaged per-op percentages.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".windows.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
