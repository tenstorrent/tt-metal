# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling for row unicast versus multicast."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import datetime
import json
import socket
import statistics
import subprocess
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.row_unicast_vs_mcast import (
    VARIANTS,
    build_row_layout,
    create_program_descriptor,
    create_sharded_memory_config,
    row_all_gather,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_REPORT_PATH = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/row_unicast_vs_mcast/report.md"


def _selected_variants():
    selected = tuple(part for part in os.environ.get("ROW_BCAST_VARIANTS", ",".join(VARIANTS)).split(",") if part)
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown ROW_BCAST_VARIANTS: {sorted(unknown)}")
    return selected


def _num_rows(device):
    grid = device.compute_with_storage_grid_size()
    return int(os.environ.get("ROW_BCAST_ROWS", grid.y))


def _row_widths(device):
    grid = device.compute_with_storage_grid_size()
    if raw := os.environ.get("ROW_BCAST_WIDTHS"):
        widths = tuple(int(value) for value in raw.split(","))
    else:
        widths = tuple(dict.fromkeys(width for width in (2, 4, 8, grid.x) if width <= grid.x))
    for width in widths:
        build_row_layout(device, 1, width)
    return widths


def _make_input(device, num_rows, row_width, num_tiles):
    layout = build_row_layout(device, num_rows, row_width)
    payload_width = num_tiles * ttnn.TILE_SIZE
    state_width = row_width * payload_width
    shape = (layout.num_cores * ttnn.TILE_SIZE, state_width)
    values = torch.zeros(shape, dtype=torch.float32)

    for core_index in range(layout.num_cores):
        row_start = core_index * ttnn.TILE_SIZE
        pattern = (torch.arange(ttnn.TILE_SIZE * payload_width).reshape(ttnn.TILE_SIZE, payload_width) % 97) / 128
        slot_start = (core_index % row_width) * payload_width
        values[
            row_start : row_start + ttnn.TILE_SIZE,
            slot_start : slot_start + payload_width,
        ] = (
            pattern + core_index
        )

    quantized = values.to(torch.bfloat16).to(torch.float32)
    expected = torch.empty_like(quantized)
    for y in range(num_rows):
        row_payload = torch.cat(
            [
                quantized[
                    (y * row_width + x) * ttnn.TILE_SIZE : (y * row_width + x + 1) * ttnn.TILE_SIZE,
                    x * payload_width : (x + 1) * payload_width,
                ]
                for x in range(row_width)
            ],
            dim=1,
        )
        for x in range(row_width):
            core_index = y * row_width + x
            expected[core_index * ttnn.TILE_SIZE : (core_index + 1) * ttnn.TILE_SIZE] = row_payload

    tt_input = ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(device, num_rows, row_width, row_width * num_tiles),
    )
    return tt_input, expected


def _run_checked(tt_input, expected, variant, num_rows, row_width, num_tiles, num_writes, kernel_iters=1):
    output = row_all_gather(
        tt_input,
        variant=variant,
        num_rows=num_rows,
        row_width=row_width,
        num_tiles=num_tiles,
        num_writes=num_writes,
        kernel_iters=kernel_iters,
    )
    torch.testing.assert_close(ttnn.to_torch(output).to(torch.float32), expected)
    return output


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total = 0.0
    found = False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    _read_kernel_ns(device)

    samples = {variant: [] for variant in runners}
    for trial in range(trials + 1):
        for variant, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"device profiler produced no duration for {variant}"
            if trial > 0:
                samples[variant].append(duration / kernel_iters)
    return samples


def _clock_mhz():
    if value := os.environ.get("ROW_BCAST_CLOCK_MHZ"):
        return value
    try:
        result = subprocess.run(["tt-smi", "-s"], check=True, capture_output=True, text=True, timeout=10)
        devices = json.loads(result.stdout)["device_info"]
        return devices[0]["telemetry"]["aiclk"].strip()
    except (KeyError, OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return "unknown"


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _format_report(results, *, box, arch, clock_mhz, git_sha, trials, kernel_iters, num_rows):
    iteration_mode = "per-launch latency" if kernel_iters == 1 else "steady-state"
    lines = [
        "# Row unicast versus multicast — device report",
        "",
        f"box={box}  arch={arch}  clock={clock_mhz}MHz  date={datetime.date.today().isoformat()}  git={git_sha}",
        f"N={trials} (median)  kernel-iters={kernel_iters} ({iteration_mode})",
        "",
        "Each hardware row runs one ordered round per sender. The payload starts in its sender's L1",
        "slot and is delivered only to peers. Total payload bytes stay fixed as the dispatch count changes.",
        "",
        "| Cores | Placement | Payload | Dispatches | Bytes/write | Method | NoC calls/sender | Median ns/exchange | Std / median | vs unicast |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row_width, num_tiles, num_writes, samples in results:
        baseline = statistics.median(samples["unicast"]) if "unicast" in samples else None
        row_word = "row" if num_rows == 1 else "rows"
        for variant, values in samples.items():
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            noise = 100.0 * std / median if median else float("nan")
            noise_text = f"{noise:.1f}%" + (" (noisy)" if noise >= 5.0 else "")
            ratio_text = f"{baseline / median:.2f}x" if baseline is not None else "n/a"
            noc_calls = num_writes * (row_width - 1) if variant == "unicast" else num_writes
            lines.append(
                f"| {num_rows * row_width} | {num_rows} {row_word} × {row_width} cores | "
                f"{num_tiles * 2048} B | {num_writes} | {num_tiles * 2048 // num_writes} B | "
                f"{variant} | {noc_calls} | {median:.1f} | {noise_text} | {ratio_text} |"
            )
    return "\n".join(lines) + "\n"


def _write_report(report_path, report, arch):
    """Preserve prior architecture blocks when re-profiling on another device."""
    if not report_path.exists():
        report_path.write_text(report)
        return

    existing = report_path.read_text()
    arch_lines = [line for line in existing.splitlines() if line.startswith("box=") and " arch=" in line]
    has_this_arch = any(f"arch={arch}" in line for line in arch_lines)
    has_other_arch = any(f"arch={arch}" not in line for line in arch_lines)
    if arch_lines and (not has_this_arch or has_other_arch):
        report_path.write_text(existing.rstrip() + "\n\n---\n\n" + report)
    else:
        report_path.write_text(report)


def test_row_unicast_vs_mcast_correctness(device):
    grid = device.compute_with_storage_grid_size()
    num_rows = min(2, grid.y)
    for row_width in _row_widths(device):
        for num_writes in (1, 2, 4, 8, 16, 32):
            for variant in VARIANTS:
                tt_input, expected = _make_input(device, num_rows, row_width, num_tiles=2)
                _run_checked(
                    tt_input,
                    expected,
                    variant,
                    num_rows,
                    row_width,
                    num_tiles=2,
                    num_writes=num_writes,
                    kernel_iters=3,
                )


def test_row_unicast_vs_mcast_device_perf(device):
    """Correctness-gate both transports, then report device time without a perf assertion."""
    trials = int(os.environ.get("ROW_BCAST_TRIALS", "5"))
    kernel_iters = int(os.environ.get("ROW_BCAST_KERNEL_ITERS", "100"))
    num_tiles_values = tuple(int(value) for value in os.environ.get("ROW_BCAST_TILES", "1,4,16").split(","))
    num_writes_values = tuple(int(value) for value in os.environ.get("ROW_BCAST_WRITES", "1,2,4,8,16,32").split(","))
    num_rows = _num_rows(device)
    variants = _selected_variants()
    results = []

    for row_width in _row_widths(device):
        for num_tiles in num_tiles_values:
            for num_writes in num_writes_values:
                states = {}
                for variant in variants:
                    state, expected = _make_input(device, num_rows, row_width, num_tiles)
                    states[variant] = _run_checked(
                        state,
                        expected,
                        variant,
                        num_rows,
                        row_width,
                        num_tiles,
                        num_writes,
                    )
                descriptors = {
                    variant: create_program_descriptor(
                        states[variant],
                        states[variant],
                        variant=variant,
                        num_rows=num_rows,
                        row_width=row_width,
                        num_tiles=num_tiles,
                        num_writes=num_writes,
                        kernel_iters=kernel_iters,
                    )
                    for variant in variants
                }
                runners = {
                    variant: (
                        lambda state=states[variant], descriptor=descriptors[variant]: ttnn.generic_op(
                            [state, state],
                            descriptor,
                        )
                    )
                    for variant in variants
                }
                results.append((row_width, num_tiles, num_writes, _measure(device, runners, trials, kernel_iters)))

    report = _format_report(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        clock_mhz=_clock_mhz(),
        git_sha=_git_sha(),
        trials=trials,
        kernel_iters=kernel_iters,
        num_rows=num_rows,
    )
    logger.info("\n" + report)
    report_path = Path(os.environ.get("ROW_BCAST_REPORT", _REPORT_PATH))
    _write_report(report_path, report, os.environ.get("ARCH_NAME", str(device.arch())))
    logger.info(f"[row_unicast_vs_mcast] wrote {report_path}")
