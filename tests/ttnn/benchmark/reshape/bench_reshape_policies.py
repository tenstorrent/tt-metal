# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay harvested reshape workloads on legacy and new ``ttnn.reshape``: bit-exactness, host time, device time.

Each workload (see ``capture_reshape_workloads.py``) is rebuilt with its exact input spec and request and run on
both implementations through the pinned ``reshape_legacy`` / ``reshape_new`` entry points. Results go to a CSV
with one row per workload and a markdown summary sorted by regression::

    python tests/ttnn/benchmark/reshape/bench_reshape_policies.py reshape_workloads.json -o reshape_bench/

Device kernel time is measured when the device profiler is on (set before the device opens)::

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
        python tests/ttnn/benchmark/reshape/bench_reshape_policies.py reshape_workloads.json

Multi-device workloads are skipped: the replay runs on one device.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

import ttnn
from ttnn._ttnn.operations.data_movement import reshape_legacy, reshape_new

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reshape_workloads import (  # noqa: E402
    describe,
    is_multi_device,
    load_workloads,
    make_input,
    make_torch_input,
    request_arguments,
)

KERNEL_DURATION_ANALYSIS = "DEVICE KERNEL DURATION [ns]"
BLOCK_FLOAT_DTYPES = ("BFLOAT8_B", "BFLOAT4_B")

CSV_FIELDS = [
    "key",
    "workload",
    "count",
    "legacy",
    "bitwise_equal",
    "new_matches_input",
    "host_new_us",
    "host_legacy_us",
    "host_ratio",
    "device_new_ns",
    "device_legacy_ns",
    "device_ratio",
    "error",
]


def profiler_enabled():
    return all(
        os.environ.get(name) == "1"
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_MID_RUN_DUMP", "TT_METAL_PROFILER_CPP_POST_PROCESS")
    )


def as_bytes(tensor):
    """The logical elements as raw bytes, so NaN payloads and signed zeros compare exactly."""
    return ttnn.to_torch(tensor).contiguous().view(-1).view(torch.uint8)


def run(implementation, tensor, entry):
    args, kwargs = request_arguments(entry)
    return implementation(tensor, *args, **kwargs)


def host_time_us(implementation, tensor, entry, device, iterations):
    run(implementation, tensor, entry)
    if device is not None:
        ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iterations):
        run(implementation, tensor, entry)
    if device is not None:
        ttnn.synchronize_device(device)
    return (time.perf_counter() - start) * 1e6 / iterations


def device_time_ns(implementation, tensor, entry, device, iterations):
    """Kernel time per call, summed over every program the call launches."""
    run(implementation, tensor, entry)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    for _ in range(iterations):
        run(implementation, tensor, entry)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    total = 0
    for programs in ttnn.get_latest_programs_perf_data().values():
        for program in programs:
            result = program.program_analyses_results.get(KERNEL_DURATION_ANALYSIS)
            if result is not None:
                total += result.duration
    return total / iterations


def ratio(new, legacy):
    return new / legacy if new is not None and legacy else None


def bench_workload(entry, device, args, measure_device):
    row = {"key": entry["key"][:12], "workload": describe(entry), "count": entry["count"], "legacy": "ok"}
    on_device = entry["input"]["storage"] == "device"
    target_device = device if on_device else None
    try:
        torch_input = make_torch_input(entry)
        tensor = make_input(entry, device, torch_input)
        new_output = run(reshape_new, tensor, entry)
    except Exception as error:  # the row records the failure; the replay goes on
        row["error"] = f"new: {error}".splitlines()[0]
        return row

    if entry["input"]["dtype"] not in BLOCK_FLOAT_DTYPES:
        row["new_matches_input"] = torch.equal(as_bytes(new_output), as_bytes(tensor))

    legacy_output = None
    try:
        legacy_output = run(reshape_legacy, tensor, entry)
    except Exception as error:
        row["legacy"] = "unsupported"
        row["error"] = f"legacy: {error}".splitlines()[0]
    if legacy_output is not None:
        row["bitwise_equal"] = torch.equal(as_bytes(new_output), as_bytes(legacy_output))
    del new_output, legacy_output

    # New first, so warmup effects do not favor it.
    row["host_new_us"] = host_time_us(reshape_new, tensor, entry, target_device, args.iterations)
    if row["legacy"] == "ok":
        row["host_legacy_us"] = host_time_us(reshape_legacy, tensor, entry, target_device, args.iterations)
    row["host_ratio"] = ratio(row["host_new_us"], row.get("host_legacy_us"))

    if measure_device and on_device:
        row["device_new_ns"] = device_time_ns(reshape_new, tensor, entry, device, args.device_iterations)
        if row["legacy"] == "ok":
            row["device_legacy_ns"] = device_time_ns(reshape_legacy, tensor, entry, device, args.device_iterations)
        row["device_ratio"] = ratio(row["device_new_ns"], row.get("device_legacy_ns"))
    return row


def _format(value):
    if isinstance(value, float):
        return f"{value:.3f}" if value < 10 else f"{value:.1f}"
    return "" if value is None else str(value)


def write_summary(rows, path, measure_device):
    failures = [r for r in rows if r.get("bitwise_equal") is False or r.get("new_matches_input") is False]
    errors = [r for r in rows if r.get("error") and r["legacy"] == "ok"]
    sort_key = "device_ratio" if measure_device else "host_ratio"
    timed = sorted((r for r in rows if r.get(sort_key) is not None), key=lambda r: -r[sort_key])
    total_weight = sum(r["count"] for r in timed) or 1
    weighted = sum(r[sort_key] * r["count"] for r in timed) / total_weight if timed else None

    lines = [
        "# Reshape: new versus legacy",
        "",
        f"- workloads: {len(rows)}; bit mismatches: {len(failures)}; new-path errors: {len(errors)}",
        f"- legacy unsupported: {sum(r['legacy'] == 'unsupported' for r in rows)}",
        f"- call-weighted {sort_key.replace('_', ' ')} (new / legacy, lower is better): {_format(weighted)}",
        "",
    ]
    if failures or errors:
        lines += [
            "## Failures",
            "",
            "| key | workload | bitwise equal | matches input | error |",
            "|---|---|---|---|---|",
        ]
        for r in failures + errors:
            lines.append(
                f"| {r['key']} | {r['workload']} | {_format(r.get('bitwise_equal'))} | "
                f"{_format(r.get('new_matches_input'))} | {_format(r.get('error'))} |"
            )
        lines.append("")
    lines += [
        f"## Timing, worst {sort_key.replace('_', ' ')} first",
        "",
        "| key | workload | calls | host new us | host legacy us | device new ns | device legacy ns | ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in timed:
        lines.append(
            f"| {r['key']} | {r['workload']} | {r['count']} | {_format(r.get('host_new_us'))} | "
            f"{_format(r.get('host_legacy_us'))} | {_format(r.get('device_new_ns'))} | "
            f"{_format(r.get('device_legacy_ns'))} | {_format(r[sort_key])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workloads", type=Path, help="workload file from capture_reshape_workloads.py")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("reshape_bench"))
    parser.add_argument("--iterations", type=int, default=200, help="host-timed calls per implementation")
    parser.add_argument("--device-iterations", type=int, default=10, help="profiled calls per implementation")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="replay only the N most frequent workloads")
    args = parser.parse_args()

    workloads = [w for w in load_workloads(args.workloads) if not is_multi_device(w)][: args.limit]
    measure_device = profiler_enabled()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for index, entry in enumerate(workloads):
            row = bench_workload(entry, device, args, measure_device)
            rows.append(row)
            print(
                f"[{index + 1}/{len(workloads)}] {row['workload']}: legacy {row['legacy']}, "
                f"bitwise {_format(row.get('bitwise_equal'))}, host ratio {_format(row.get('host_ratio'))}, "
                f"device ratio {_format(row.get('device_ratio'))} {row.get('error', '')}"
            )
    finally:
        ttnn.close_device(device)

    csv_path = args.output_dir / "reshape_policies.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in CSV_FIELDS})
    summary_path = args.output_dir / "reshape_policies.md"
    write_summary(rows, summary_path, measure_device)
    print(f"wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
