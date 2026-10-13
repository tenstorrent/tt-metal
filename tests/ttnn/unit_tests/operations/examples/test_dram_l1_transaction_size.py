# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated transaction sweep; timing is reported, never a pass/fail threshold."""

import os

for key, value in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_LOGGER_LEVEL": "error",
}.items():
    os.environ.setdefault(key, value)

import datetime
import json
import math
import re
import socket
import statistics
import subprocess
from pathlib import Path
import torch
import ttnn
from ttnn.operations.examples.dram_l1_transaction_size.program_descriptor_with_inline_kernels import (
    VARIANTS,
    assignments,
    core_layout,
    create_program_descriptor,
    output_memory,
)


def read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    entries = [p for programs in (ttnn.get_latest_programs_perf_data() or {}).values() for p in programs]
    entries = [p for p in entries if "DEVICE KERNEL DURATION [ns]" in p.program_analyses_results]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one profiled program, got {len(entries)}")
    p = entries[0]
    return float(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration), p.core_count


def format_report(report):
    lines = [
        "# DRAM to resident L1 transaction-size measurements",
        "",
        f"box={report['host']} · arch={report['arch']} · clock={report['clock_mhz']} MHz · {report['date']} · git={report['git']}",
        f"shape={report['shape']} BF16 · cores={report['cores']} · placement={report['logical_cores']} · NoC policy={report['noc_policy']}",
        f"Metric: DEVICE KERNEL DURATION [ns]; {report['trials']} trials, median and population std; kernel-iters={report['kernel_iters']}. Warmup discarded.",
        "Input is DRAM interleaved, one row per page. Output remains in L1. One final read barrier per core per iteration.",
        "Uniform L1 allocation has padding after short shards; padding is checked unchanged and is not read from DRAM.",
        "",
        "| Cores | Transaction B | Requests / largest shard | Expected µs | Measured µs | Std µs | Measured / expected | Useful GB/s | Correct | Noise >3% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in report["results"]:
        pred = f"{r['expected_us']:.3f}" if r["expected_us"] is not None else "n/a"
        ratio = f"{r['ratio']:.2f}×" if r["ratio"] is not None else "n/a"
        lines.append(
            f"| {report['cores']} | {r['transaction_bytes']} | {r['transactions_max_core']} | {pred} | {r['median_us']:.3f} | {r['std_us']:.3f} | {ratio} | {r['gbps']:.2f} | yes | {'yes' if r['noise'] else 'no'} |"
        )
    lines += [
        "",
        "Expected values are the supplied model snapshot, unchanged by measurement. Device timings include kernel entry/exit, address generation, read issue, backpressure, contention, and final completion. They exclude host transfers and correctness checks. No performance assertion is made.",
        "",
    ]
    return "\n".join(lines)


def run_case(device, *, case_override=None, cores_override=None, variants_override=None, report_path=None, quiet=False):
    case_path = os.environ.get("DRAM_L1_MODEL_JSON")
    case = json.loads(Path(case_path).read_text()) if case_path else {}
    if case_override is not None:
        case = case_override
    shape = [
        int(x)
        for x in os.environ.get("DRAM_L1_SHAPE", ",".join(map(str, case.get("shape", [1, 1, 4096, 1024])))).split(",")
    ]
    cores = int(os.environ.get("DRAM_L1_CORES", case.get("cores", 7)))
    if cores_override is not None:
        cores = cores_override
    trials = int(os.environ.get("DRAM_L1_TRIALS", "1"))
    iters = int(os.environ.get("DRAM_L1_KERNEL_ITERS", "1"))
    profile_zones = os.environ.get("DRAM_L1_PROFILE_ZONES", "0") == "1"
    if profile_zones and iters > 32:
        raise ValueError("Use <=32 iterations to keep three profiling zones/iteration within the marker budget")
    assert len(shape) == 4 and min(shape) > 0 and trials > 0 and iters > 0
    rows, width = math.prod(shape[:3]), shape[3]
    shard_rows = math.ceil(rows / cores)
    predictions = {int(p["transaction_bytes"]): p["expected_us"] for p in case.get("predictions", [])}
    if case and (shape != case["shape"] or cores != case["cores"]):
        raise ValueError("Model snapshot shape/core count must match the measured workload")
    selected = os.environ.get("DRAM_L1_VARIANTS")
    variants = selected.split(",") if selected else [v for v in VARIANTS if width * 2 % int(v.split("_")[1]) == 0]
    if variants_override is not None:
        variants = variants_override
    # Seeded random BF16 values detect wrong row/bank/offset mappings; sentinel detects unwritten output.
    gen = torch.Generator().manual_seed(4217)
    host = torch.randn((rows, width), generator=gen).to(torch.bfloat16)
    src = ttnn.from_torch(
        host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert src.buffer_num_pages() == rows and src.buffer_aligned_page_size() == width * 2
    expected = torch.full((shard_rows * cores, width), -123.0, dtype=torch.bfloat16)
    for i, (start, count) in enumerate(assignments(rows, cores)):
        expected[i * shard_rows : i * shard_rows + count] = host[start : start + count]
    report = dict(
        host=socket.gethostname(),
        arch=str(device.arch()),
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        shape=shape,
        cores=cores,
        logical_cores=core_layout(device, cores),
        noc_policy="runtime preferred read NoC via ReaderConfigDescriptor (NoC0 in this build)",
        clock_mhz=None,
        trials=trials,
        kernel_iters=iters,
        profile_zones=profile_zones,
        row_bytes=width * 2,
        allocated_l1_bytes_per_core=shard_rows * width * 2,
        model=case,
        results=[],
    )
    for variant in variants:
        sentinel = torch.full_like(expected, -123.0)
        dst = ttnn.from_torch(
            sentinel,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory(device, rows, width, cores),
        )
        descriptor = create_program_descriptor(
            src, dst, variant=variant, cores=cores, kernel_iters=iters, profile_zones=profile_zones
        )

        assert all(isinstance(k.config, ttnn.ReaderConfigDescriptor) for k in descriptor.kernels)

        def run():
            ttnn.generic_op([src, dst], descriptor)

        run()
        torch.testing.assert_close(ttnn.to_torch(dst), expected, rtol=0, atol=0)
        ttnn.ReadDeviceProfiler(device)  # discard setup and correctness traffic
        run()
        read_ns(device)  # discard first measured launch
        samples = []
        for _ in range(trials):
            run()
            ns, actual_cores = read_ns(device)
            assert actual_cores == cores, (actual_cores, cores)
            samples.append(ns / iters / 1000)
        torch.testing.assert_close(ttnn.to_torch(dst), expected, rtol=0, atol=0)
        txn = int(variant.split("_")[1])
        median = statistics.median(samples)
        std = statistics.pstdev(samples)
        prediction = predictions.get(txn)
        report["results"].append(
            dict(
                variant=variant,
                transaction_bytes=txn,
                transactions_max_core=shard_rows * width * 2 // txn,
                expected_us=prediction,
                median_us=median,
                std_us=std,
                ratio=median / prediction if prediction else None,
                gbps=rows * width * 2 / (median * 1000),
                samples_us=samples,
                correct=True,
                noise=std / median > 0.03,
            )
        )
        print(f"{variant}: {median:.3f} us; exact correctness PASS", flush=True)
        ttnn.deallocate(dst)
        ttnn.ReadDeviceProfiler(device)
    # Metadata only: timing itself comes directly from the in-process profiler API.
    metadata = Path("generated/profiler/.logs/profile_log_device.csv")
    if metadata.exists():
        with metadata.open() as stream:
            first_line = stream.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", first_line)
        if match:
            report["clock_mhz"] = int(match.group(1))
    text = format_report(report)
    if not quiet:
        print(text, flush=True)
    out = Path(os.environ.get("DRAM_L1_REPORT", "ttnn/ttnn/operations/examples/dram_l1_transaction_size/report.md"))
    if report_path is not None:
        out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Append architecture-stamped reports instead of destroying older measurements.
    with out.open("a") as stream:
        stream.write(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    ttnn.deallocate(src)
    return report


def test_dram_l1_transaction_size(device):
    run_case(device)


def test_dram_l1_core_sweep(device):
    import copy

    import pytest

    if not os.environ.get("DRAM_L1_MODEL_JSON"):
        pytest.skip("Core sweep requires a model snapshot; use CLI --sweep-cores --model-json")
    case = json.loads(Path(os.environ["DRAM_L1_MODEL_JSON"]).read_text())
    if not case.get("core_predictions"):
        pytest.skip("Snapshot has no core-count predictions; export a core-sweep snapshot")
    requested = os.environ.get("DRAM_L1_CORE_COUNTS", "all")
    grid = device.compute_with_storage_grid_size()
    limit = grid.x * grid.y
    entries = {int(p["cores"]): p for p in case["core_predictions"]}
    counts = sorted(entries) if requested == "all" else sorted(set(map(int, requested.split(","))))
    unavailable = [n for n in counts if n > limit]
    counts = [n for n in counts if n <= limit]
    if not counts or any(n not in entries for n in counts):
        raise ValueError("Core sweep requires model-feasible counts and matching predictions")
    out = Path(os.environ.get("DRAM_L1_REPORT", "ttnn/ttnn/operations/examples/dram_l1_transaction_size/core_sweep.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        shape=case["shape"],
        transaction_bytes=2048,
        grid=[grid.x, grid.y],
        unavailable_core_counts=unavailable,
        results=[],
    )
    print(f"Core sweep: {counts}; grid={grid.x}x{grid.y}; unavailable={unavailable}", flush=True)
    for n in counts:
        local = copy.deepcopy(case)
        local["cores"] = n
        local["predictions"] = [entries[n]]
        local.pop("core_predictions", None)
        report = run_case(
            device,
            case_override=local,
            cores_override=n,
            variants_override=["txn_2048"],
            report_path=out.parent / f"core_sweep_cases/cores_{n:03}.md",
            quiet=True,
        )
        result["host"] = report["host"]
        result["arch"] = report["arch"]
        result["clock_mhz"] = report["clock_mhz"]
        result["date"] = report["date"]
        result["trials"] = report["trials"]
        result["kernel_iters"] = report["kernel_iters"]
        result["noc_policy"] = report["noc_policy"]
        result["git"] = report["git"]
        result["results"].append(
            dict(
                cores=n,
                logical_cores=report["logical_cores"],
                allocated_l1_bytes_per_core=report["allocated_l1_bytes_per_core"],
                **report["results"][0],
            )
        )
        out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"cores={n}: {report['results'][0]['median_us']:.3f} us", flush=True)
    best = min(result["results"], key=lambda r: r["median_us"])
    result["best"] = best
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    text = [
        "# DRAM to resident L1: core-count sweep",
        "",
        f"{result['host']} | {result['arch']} | {result['clock_mhz']} MHz | {result['date']}",
        f"BF16 {result['shape']}, 2048 B transactions, preferred read NoC, row-major placement, one barrier/core.",
        f"{result['trials']} trials; kernel-iters={result['kernel_iters']}; device kernel duration; warmup discarded.",
        f"Available grid: {grid.x}x{grid.y}; counts outside grid: {unavailable}.",
        f"Fastest sampled median: {best['cores']} cores, {best['median_us']:.3f} us, {best['gbps']:.2f} GB/s.",
        "",
        "| Cores | L1 B/core | Expected us | Measured us | Std us | GB/s | Correct | Noise >3% |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in result["results"]:
        text.append(
            f"| {r['cores']} | {r['allocated_l1_bytes_per_core']} | {r['expected_us']:.3f} | {r['median_us']:.3f} | {r['std_us']:.3f} | {r['gbps']:.2f} | yes | {'yes' if r['noise'] else 'no'} |"
        )
    text += [
        "",
        "This sweeps prefixes of one fixed row-major placement rule; it does not optimize physical core placement. Correctness, including untouched shard padding, passed for every count. Predictions are the unchanged model snapshot. Close minima should be treated as a plateau, not a unique optimum.",
        "",
    ]
    with out.open("a") as stream:
        stream.write("\n".join(text))
    print("\n".join(text), flush=True)
