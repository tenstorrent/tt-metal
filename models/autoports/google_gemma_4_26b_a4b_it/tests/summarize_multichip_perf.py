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
    number,
    tensor_shape,
    traffic,
    useful_prefill_flops,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--layer-type", required=True, choices=["sliding_attention", "full_attention"])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        replay_times = []
        replay_bytes = []
        for index, session in enumerate(session_order):
            device_times = []
            transfers = 0
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
                    for row in ops:
                        counts = None
                        if is_native_sdpa_decode(row):
                            position = 4096 + index
                            end = ((position + 1 + 127) // 128) * 128
                            start = (
                                max(0, position + 1 - 1024) // 128 * 128
                                if args.layer_type == "sliding_attention"
                                else 0
                            )
                            counts = {}
                            for prefix in ("INPUT_1", "INPUT_2"):
                                shape = tensor_shape(row, prefix)
                                counts[prefix] = shape[1] * (end - start) * shape[-1]
                        transfers += traffic(row, counts)
            assert len(device_times) == 4
            replay_times.append(max(device_times))
            replay_bytes.append(transfers)
        summary[phase] = dict(
            device_us=statistics.mean(replay_times),
            samples=len(replay_times),
            min_device_us=min(replay_times),
            max_device_us=max(replay_times),
        )
        if phase == "decode":
            summary[phase]["estimated_dram_bytes"] = statistics.mean(replay_bytes)
    assert set(summary) == {"prefill", "decode"}
    flops = sum(useful_prefill_flops(args.layer_type).values())
    peak_flops = 4 * 120 * 4096 * 1.35e9
    peak_dram = 4 * 512e9
    result = dict(
        layer_type=args.layer_type,
        selected_mesh=[1, 4],
        source=str(args.csv),
        workload=dict(input_tokens=4096, output_tokens=args.steps, batch=1, concurrency=1),
        target_workload_measured=args.steps == 128,
        whole_layer_windows=summary,
        prefill_device_us=summary["prefill"]["device_us"],
        decode_device_us=summary["decode"]["device_us"],
        prefill_useful_flops=flops,
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
            "DRAM estimate sums native operand reads/writes across all four devices with indexed active-expert metadata, BFP tile exponent bytes, selected embedding/slice rows and paged writes. SDPA uses local cache heads and source-derived128-token rounded reads. Extra per-core reads, internal CCL passes/NoC and profiler traffic excluded; not controller counters.",
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
