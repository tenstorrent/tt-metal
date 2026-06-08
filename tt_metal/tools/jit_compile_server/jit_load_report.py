#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO


JIT_SERVER_LOG_PATTERN = re.compile(
    r"\[jit_server addr=(?P<addr>\S+) ts=(?P<ts>\d+)\] "
    r"count=(?P<count>\d+) "
    r"total_compile_time_ms=(?P<total_compile_time_ms>\d+) "
    r"queued=(?P<queued>\d+) "
    r"inflight=(?P<inflight>\d+) "
    r"peak_inflight=(?P<peak_inflight>\d+) "
    r"bytes_in=(?P<bytes_in>\d+) "
    r"bytes_out=(?P<bytes_out>\d+)"
)


@dataclass(frozen=True)
class JitServerSample:
    addr: str
    ts: int
    count: int
    total_compile_time_ms: int
    queued: int
    inflight: int
    peak_inflight: int
    bytes_in: int
    bytes_out: int


@dataclass(frozen=True)
class ServerLoadReport:
    addr: str
    count: int
    total_compile_time_ms: int
    peak_inflight: int
    bytes_in: int
    bytes_out: int
    throughput_compiles_per_sec: float | None


@dataclass(frozen=True)
class ImbalanceStats:
    max: float
    min: float
    mean: float
    stddev: float
    max_min_ratio: float | None
    coefficient_of_variation: float


def parse_jit_server_line(line: str) -> JitServerSample | None:
    match = JIT_SERVER_LOG_PATTERN.search(line)
    if match is None:
        return None

    groups = match.groupdict()
    return JitServerSample(
        addr=groups["addr"],
        ts=int(groups["ts"]),
        count=int(groups["count"]),
        total_compile_time_ms=int(groups["total_compile_time_ms"]),
        queued=int(groups["queued"]),
        inflight=int(groups["inflight"]),
        peak_inflight=int(groups["peak_inflight"]),
        bytes_in=int(groups["bytes_in"]),
        bytes_out=int(groups["bytes_out"]),
    )


def parse_jit_server_samples(lines: Iterable[str]) -> list[JitServerSample]:
    samples: list[JitServerSample] = []
    for line in lines:
        sample = parse_jit_server_line(line)
        if sample is not None:
            samples.append(sample)
    return samples


def group_samples_by_addr(samples: Iterable[JitServerSample]) -> dict[str, list[JitServerSample]]:
    grouped: dict[str, list[JitServerSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.addr, []).append(sample)
    for grouped_samples in grouped.values():
        grouped_samples.sort(key=lambda sample: sample.ts)
    return grouped


def _non_negative_delta(last: int, first: int) -> int:
    return max(0, last - first)


def compute_server_report(samples: Sequence[JitServerSample], window_seconds: float | None) -> ServerLoadReport:
    if not samples:
        raise ValueError("compute_server_report() requires at least one sample")

    last = samples[-1]
    if window_seconds is None:
        return ServerLoadReport(
            addr=last.addr,
            count=last.count,
            total_compile_time_ms=last.total_compile_time_ms,
            peak_inflight=last.peak_inflight,
            bytes_in=last.bytes_in,
            bytes_out=last.bytes_out,
            throughput_compiles_per_sec=None,
        )

    window_ms = int(window_seconds * 1000.0)
    cutoff = last.ts - window_ms
    window_samples = [sample for sample in samples if sample.ts >= cutoff]
    first = window_samples[0] if window_samples else last
    peak_inflight = max((sample.peak_inflight for sample in window_samples), default=last.peak_inflight)

    elapsed_ms = max(0, last.ts - first.ts)
    count_delta = _non_negative_delta(last.count, first.count)
    throughput = 0.0 if elapsed_ms == 0 else count_delta / (elapsed_ms / 1000.0)

    return ServerLoadReport(
        addr=last.addr,
        count=count_delta,
        total_compile_time_ms=_non_negative_delta(last.total_compile_time_ms, first.total_compile_time_ms),
        peak_inflight=peak_inflight,
        bytes_in=_non_negative_delta(last.bytes_in, first.bytes_in),
        bytes_out=_non_negative_delta(last.bytes_out, first.bytes_out),
        throughput_compiles_per_sec=throughput,
    )


def compute_imbalance_stats(values: Sequence[int | float]) -> ImbalanceStats:
    if not values:
        return ImbalanceStats(
            max=0.0,
            min=0.0,
            mean=0.0,
            stddev=0.0,
            max_min_ratio=1.0,
            coefficient_of_variation=0.0,
        )

    numeric_values = [float(value) for value in values]
    max_value = max(numeric_values)
    min_value = min(numeric_values)
    mean_value = sum(numeric_values) / len(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / len(numeric_values)
    stddev_value = math.sqrt(variance)
    max_min_ratio = (
        None if (min_value == 0.0 and max_value > 0.0) else (1.0 if min_value == 0.0 else max_value / min_value)
    )
    coefficient_of_variation = 0.0 if mean_value == 0.0 else stddev_value / mean_value

    return ImbalanceStats(
        max=max_value,
        min=min_value,
        mean=mean_value,
        stddev=stddev_value,
        max_min_ratio=max_min_ratio,
        coefficient_of_variation=coefficient_of_variation,
    )


def build_report(samples: Sequence[JitServerSample], window_seconds: float | None) -> dict:
    grouped = group_samples_by_addr(samples)
    server_reports = [
        compute_server_report(grouped_samples, window_seconds) for _, grouped_samples in sorted(grouped.items())
    ]

    imbalance_summary = {
        "count": asdict(compute_imbalance_stats([server.count for server in server_reports])),
        "total_compile_time_ms": asdict(
            compute_imbalance_stats([server.total_compile_time_ms for server in server_reports])
        ),
    }

    return {
        "window_seconds": window_seconds,
        "servers": [asdict(server_report) for server_report in server_reports],
        "imbalance_summary": imbalance_summary,
    }


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "undefined"
    return _format_float(value)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    row_lines = ["  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def render_human_report(report: dict, window_seconds: float | None) -> str:
    lines: list[str] = []

    servers = report["servers"]
    if not servers:
        return "No jit_server samples found."

    lines.append("Per-server metrics:")
    headers = ["addr", "count", "total_compile_time_ms", "peak_inflight", "bytes_in", "bytes_out"]
    if window_seconds is not None:
        headers.append("throughput_compiles_per_sec")

    rows: list[list[str]] = []
    for server in servers:
        row = [
            server["addr"],
            str(server["count"]),
            str(server["total_compile_time_ms"]),
            str(server["peak_inflight"]),
            str(server["bytes_in"]),
            str(server["bytes_out"]),
        ]
        if window_seconds is not None:
            throughput = server["throughput_compiles_per_sec"]
            row.append(_format_float(throughput) if throughput is not None else "n/a")
        rows.append(row)

    lines.append(_render_table(headers, rows))
    lines.append("")
    lines.append("Imbalance summary:")

    imbalance = report["imbalance_summary"]
    summary_headers = ["metric", "max", "min", "mean", "stddev", "max/min_ratio", "coefficient_of_variation"]
    summary_rows: list[list[str]] = []
    for metric_name in ("count", "total_compile_time_ms"):
        stats = imbalance[metric_name]
        summary_rows.append(
            [
                metric_name,
                _format_float(stats["max"]),
                _format_float(stats["min"]),
                _format_float(stats["mean"]),
                _format_float(stats["stddev"]),
                _format_ratio(stats["max_min_ratio"]),
                _format_float(stats["coefficient_of_variation"]),
            ]
        )

    lines.append(_render_table(summary_headers, summary_rows))
    lines.append("")
    lines.append("Stat definitions:")
    lines.append("max/min/mean: largest, smallest, and average value across servers.")
    lines.append("stddev: spread of server values around the mean (higher means more imbalance).")
    lines.append("max/min_ratio: skew between busiest and least busy server (1.0 means perfectly balanced).")
    lines.append("coefficient_of_variation: normalized spread (stddev/mean), useful across different scales.")
    return "\n".join(lines)


def _read_samples_from_path(path: Path) -> list[JitServerSample]:
    with path.open("r", encoding="utf-8") as handle:
        return parse_jit_server_samples(handle)


def _read_samples_from_stream(stream: TextIO) -> list[JitServerSample]:
    return parse_jit_server_samples(stream)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report per-server and cross-server JIT load imbalance.")
    parser.add_argument("logs", nargs="*", help="Path(s) to log files. Reads stdin when omitted.")
    parser.add_argument("--window", type=float, default=None, help="Window size in seconds for delta reporting.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.window is not None and args.window <= 0.0:
        raise ValueError("--window must be > 0")

    samples: list[JitServerSample] = []
    if args.logs:
        for path in args.logs:
            samples.extend(_read_samples_from_path(Path(path)))
    else:
        samples.extend(_read_samples_from_stream(sys.stdin))

    report = build_report(samples, args.window)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human_report(report, args.window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
