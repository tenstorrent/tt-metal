#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
JIT compile server load report tool.

Quick workflow for collecting and analyzing logs:

1) Enable periodic metrics on each compile server process:
   export TT_METAL_JIT_SERVER_LOG_INTERVAL_MS=1000

2) Start the server and capture logs (stdout/stderr), for example:
   tt_metal/tools/jit_compile_server/jit_compile_server > /tmp/jit_server_a.log 2>&1

3) Repeat per server (one log file per server is easiest to attribute):
   tt_metal/tools/jit_compile_server/jit_compile_server > /tmp/jit_server_b.log 2>&1

4) Run this report script:
   python tt_metal/tools/jit_compile_server/jit_load_report.py /tmp/jit_server_a.log /tmp/jit_server_b.log

5) Optional rolling-window view:
   python tt_metal/tools/jit_compile_server/jit_load_report.py --window 30 /tmp/jit_server_*.log
   - --window N means "use only the last N seconds ending at each server's latest sample".
   - In window mode, count/dedup_hits/total_compile_time_ms/bytes_in/bytes_out are deltas
     over that N-second window, and throughput is count_delta / elapsed_seconds.
   - peak_inflight is the max value observed inside that window.

6) Optional machine-readable output:
   python tt_metal/tools/jit_compile_server/jit_load_report.py --json /tmp/jit_server_*.log

Notes:
- The script parses both "[jit_server ...]" periodic metric lines and "compile ...:" lines.
- unique_kernel_hashes metrics are inferred from hash-like tokens in compile kernel names.
- Without --window, metrics are lifetime totals from each server's most recent sample.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


JIT_SERVER_LOG_PATTERN = re.compile(
    r"\[jit_server addr=(?P<addr>\S+) ts=(?P<ts>\d+)\] "
    r"count=(?P<count>\d+) "
    r"(?:dedup_hit[s]?=(?P<dedup_hits>\d+) )?"
    r"total_compile_time_ms=(?P<total_compile_time_ms>\d+) "
    r"queued=(?P<queued>\d+) "
    r"inflight=(?P<inflight>\d+) "
    r"peak_inflight=(?P<peak_inflight>\d+) "
    r"bytes_in=(?P<bytes_in>\d+) "
    r"bytes_out=(?P<bytes_out>\d+)"
)
COMPILE_LOG_PATTERN = re.compile(r"compile (?P<kernel_name>\S+): targets=\d+ genfiles=\d+ outstanding=\d+")
KERNEL_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{16,}")

# When reading from stdin there is no per-file path to attribute samples to, and per-line
# log prefixes (e.g. tt-logger timestamps) make prefix-based server inference unstable.
# Use a single constant id so all stdin samples aggregate into one coherent server, and so
# jit_server samples and compile samples share the same server_id for unique_kernel_hashes.
STDIN_SOURCE_ID = "<stdin>"


@dataclass(frozen=True)
class JitServerSample:
    server_id: str
    addr: str
    ts: int
    count: int
    dedup_hits: int
    total_compile_time_ms: int
    queued: int
    inflight: int
    peak_inflight: int
    bytes_in: int
    bytes_out: int


@dataclass(frozen=True)
class KernelCompileSample:
    server_id: str
    kernel_name: str
    kernel_hash: str | None


@dataclass(frozen=True)
class ServerLoadReport:
    server: str
    addr: str
    count: int
    dedup_hits: int
    total_compile_time_ms: int
    peak_inflight: int
    bytes_in: int
    bytes_out: int
    unique_kernel_hashes: int
    throughput_compiles_per_sec: float | None


@dataclass(frozen=True)
class ImbalanceStats:
    max: float
    min: float
    mean: float
    stddev: float
    max_min_ratio: float | None
    coefficient_of_variation: float


def _infer_server_id(line: str, addr: str, source_id: str | None) -> str:
    if source_id is not None:
        return source_id

    prefix = line.split("[jit_server", maxsplit=1)[0].strip()
    if prefix:
        return " ".join(prefix.split())
    return addr


def parse_jit_server_line(line: str, source_id: str | None = None) -> JitServerSample | None:
    match = JIT_SERVER_LOG_PATTERN.search(line)
    if match is None:
        return None

    groups = match.groupdict()
    addr = groups["addr"]
    return JitServerSample(
        server_id=_infer_server_id(line, addr, source_id),
        addr=addr,
        ts=int(groups["ts"]),
        count=int(groups["count"]),
        dedup_hits=0 if groups["dedup_hits"] is None else int(groups["dedup_hits"]),
        total_compile_time_ms=int(groups["total_compile_time_ms"]),
        queued=int(groups["queued"]),
        inflight=int(groups["inflight"]),
        peak_inflight=int(groups["peak_inflight"]),
        bytes_in=int(groups["bytes_in"]),
        bytes_out=int(groups["bytes_out"]),
    )


def parse_jit_server_samples(lines: Iterable[str], source_id: str | None = None) -> list[JitServerSample]:
    samples: list[JitServerSample] = []
    for line in lines:
        sample = parse_jit_server_line(line, source_id=source_id)
        if sample is not None:
            samples.append(sample)
    return samples


def _extract_kernel_hash(kernel_name: str) -> str | None:
    components = [component for component in kernel_name.split("/") if component]
    for component in reversed(components):
        if KERNEL_HASH_PATTERN.fullmatch(component):
            return component.lower()

    for component in reversed(components):
        match = KERNEL_HASH_PATTERN.search(component)
        if match is not None:
            return match.group(0).lower()
    return None


def _infer_compile_server_id(line: str, source_id: str | None) -> str:
    if source_id is not None:
        return source_id

    prefix = line.split("compile ", maxsplit=1)[0].strip()
    if prefix:
        return " ".join(prefix.split())
    return "unknown"


def parse_kernel_compile_line(line: str, source_id: str | None = None) -> KernelCompileSample | None:
    match = COMPILE_LOG_PATTERN.search(line)
    if match is None:
        return None

    kernel_name = match.group("kernel_name")
    return KernelCompileSample(
        server_id=_infer_compile_server_id(line, source_id),
        kernel_name=kernel_name,
        kernel_hash=_extract_kernel_hash(kernel_name),
    )


def parse_kernel_compile_samples(lines: Iterable[str], source_id: str | None = None) -> list[KernelCompileSample]:
    samples: list[KernelCompileSample] = []
    for line in lines:
        sample = parse_kernel_compile_line(line, source_id=source_id)
        if sample is not None:
            samples.append(sample)
    return samples


def group_samples_by_server(samples: Iterable[JitServerSample]) -> dict[str, list[JitServerSample]]:
    grouped: dict[str, list[JitServerSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.server_id, []).append(sample)
    for grouped_samples in grouped.values():
        grouped_samples.sort(key=lambda sample: sample.ts)
    return grouped


def _non_negative_delta(last: int, first: int) -> int:
    return max(0, last - first)


def compute_server_report(
    samples: Sequence[JitServerSample], window_seconds: float | None, unique_kernel_hashes: int
) -> ServerLoadReport:
    if not samples:
        raise ValueError("compute_server_report() requires at least one sample")

    last = samples[-1]
    if window_seconds is None:
        return ServerLoadReport(
            server=last.server_id,
            addr=last.addr,
            count=last.count,
            dedup_hits=last.dedup_hits,
            total_compile_time_ms=last.total_compile_time_ms,
            peak_inflight=last.peak_inflight,
            bytes_in=last.bytes_in,
            bytes_out=last.bytes_out,
            unique_kernel_hashes=unique_kernel_hashes,
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
        server=last.server_id,
        addr=last.addr,
        count=count_delta,
        dedup_hits=_non_negative_delta(last.dedup_hits, first.dedup_hits),
        total_compile_time_ms=_non_negative_delta(last.total_compile_time_ms, first.total_compile_time_ms),
        peak_inflight=peak_inflight,
        bytes_in=_non_negative_delta(last.bytes_in, first.bytes_in),
        bytes_out=_non_negative_delta(last.bytes_out, first.bytes_out),
        unique_kernel_hashes=unique_kernel_hashes,
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


def _build_kernel_hashes_by_server(samples: Iterable[KernelCompileSample]) -> dict[str, set[str]]:
    hashes_by_server: dict[str, set[str]] = {}
    for sample in samples:
        if sample.kernel_hash is None:
            continue
        hashes_by_server.setdefault(sample.server_id, set()).add(sample.kernel_hash)
    return hashes_by_server


def build_report(
    samples: Sequence[JitServerSample],
    window_seconds: float | None,
    kernel_compile_samples: Sequence[KernelCompileSample] | None = None,
) -> dict:
    if kernel_compile_samples is None:
        kernel_compile_samples = []
    grouped = group_samples_by_server(samples)
    kernel_hashes_by_server = _build_kernel_hashes_by_server(kernel_compile_samples)
    server_reports = [
        compute_server_report(
            grouped_samples,
            window_seconds,
            unique_kernel_hashes=len(kernel_hashes_by_server.get(server_id, set())),
        )
        for server_id, grouped_samples in sorted(grouped.items())
    ]

    imbalance_summary = {
        "count": asdict(compute_imbalance_stats([server.count for server in server_reports])),
        "dedup_hits": asdict(compute_imbalance_stats([server.dedup_hits for server in server_reports])),
        "unique_kernel_hashes": asdict(
            compute_imbalance_stats([server.unique_kernel_hashes for server in server_reports])
        ),
        "total_compile_time_ms": asdict(
            compute_imbalance_stats([server.total_compile_time_ms for server in server_reports])
        ),
    }

    return {
        "window_seconds": window_seconds,
        "servers": [asdict(server_report) for server_report in server_reports],
        "unique_kernel_hashes_total": len(
            {hash_value for hashes in kernel_hashes_by_server.values() for hash_value in hashes}
        ),
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
    headers = [
        "server",
        "addr",
        "count",
        "dedup_hits",
        "total_compile_time_ms",
        "peak_inflight",
        "bytes_in",
        "bytes_out",
        "unique_kernel_hashes",
    ]
    if window_seconds is not None:
        headers.append("throughput_compiles_per_sec")

    rows: list[list[str]] = []
    for server in servers:
        row = [
            server["server"],
            server["addr"],
            str(server["count"]),
            str(server["dedup_hits"]),
            str(server["total_compile_time_ms"]),
            str(server["peak_inflight"]),
            str(server["bytes_in"]),
            str(server["bytes_out"]),
            str(server["unique_kernel_hashes"]),
        ]
        if window_seconds is not None:
            throughput = server["throughput_compiles_per_sec"]
            row.append(_format_float(throughput) if throughput is not None else "n/a")
        rows.append(row)

    lines.append(_render_table(headers, rows))
    lines.append("")
    if window_seconds is not None:
        lines.append(
            "Window mode: count/dedup_hits/total_compile_time_ms/bytes_in/bytes_out are per-server deltas "
            f"over the last {window_seconds:g}s ending at each server's latest sample; "
            "peak_inflight is the max seen in that window."
        )
        lines.append("")
    lines.append("Imbalance summary:")

    imbalance = report["imbalance_summary"]
    summary_headers = ["metric", "max", "min", "mean", "stddev", "max/min_ratio", "coefficient_of_variation"]
    summary_rows: list[list[str]] = []
    for metric_name in ("count", "dedup_hits", "unique_kernel_hashes", "total_compile_time_ms"):
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
    lines.append(f"Unique kernel hashes (global): {report['unique_kernel_hashes_total']}")
    lines.append("")
    lines.append("Stat definitions:")
    lines.append("count: completed compile RPCs (includes dedup hits and on-disk cache hits).")
    lines.append("dedup_hits: requests served by in-flight deduplication rather than owning a compile.")
    lines.append("unique_kernel_hashes: distinct kernel hashes observed in compile kernel names per server.")
    lines.append("max/min/mean: largest, smallest, and average value across servers.")
    lines.append("stddev: spread of server values around the mean (higher means more imbalance).")
    lines.append("max/min_ratio: skew between busiest and least busy server (1.0 means perfectly balanced).")
    lines.append("coefficient_of_variation: normalized spread (stddev/mean), useful across different scales.")
    return "\n".join(lines)


def _read_samples_from_path(path: Path) -> list[JitServerSample]:
    with path.open("r", encoding="utf-8") as handle:
        return parse_jit_server_samples(handle, source_id=str(path))


def _read_samples_from_stream(lines: Iterable[str], source_id: str | None = None) -> list[JitServerSample]:
    return parse_jit_server_samples(lines, source_id=source_id)


def _read_kernel_samples_from_path(path: Path) -> list[KernelCompileSample]:
    with path.open("r", encoding="utf-8") as handle:
        return parse_kernel_compile_samples(handle, source_id=str(path))


def _read_kernel_samples_from_stream(lines: Iterable[str], source_id: str | None = None) -> list[KernelCompileSample]:
    return parse_kernel_compile_samples(lines, source_id=source_id)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report per-server and cross-server JIT load imbalance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Collection instructions:\n"
            "  1) Set TT_METAL_JIT_SERVER_LOG_INTERVAL_MS (for example 1000).\n"
            "  2) Capture each compile server's logs to a file.\n"
            "  3) Run this script on one or more log files.\n\n"
            "Window semantics (--window N):\n"
            "  - Uses samples from the last N seconds ending at each server's latest timestamp.\n"
            "  - Reports deltas for count/dedup_hits/total_compile_time_ms/bytes_in/bytes_out.\n"
            "  - Reports peak_inflight as the max value seen within that window.\n"
            "  - Adds throughput_compiles_per_sec = count_delta / elapsed_seconds.\n\n"
            "Examples:\n"
            "  python jit_load_report.py /tmp/jit_server_a.log /tmp/jit_server_b.log\n"
            "  python jit_load_report.py --window 60 /tmp/jit_server_*.log\n"
            "  tail -f /tmp/jit_server_a.log | python jit_load_report.py --window 30\n"
            "  python jit_load_report.py --json /tmp/jit_server_*.log\n"
        ),
    )
    parser.add_argument("logs", nargs="*", help="Path(s) to log files. Reads stdin when omitted.")
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help=(
            "Rolling window size in seconds. When set, report per-server deltas over the last N seconds "
            "(count/dedup_hits/compile_time/bytes) and throughput; peak_inflight becomes window max."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.window is not None and args.window <= 0.0:
        raise ValueError("--window must be > 0")

    samples: list[JitServerSample] = []
    kernel_samples: list[KernelCompileSample] = []
    if args.logs:
        for path in args.logs:
            samples.extend(_read_samples_from_path(Path(path)))
            kernel_samples.extend(_read_kernel_samples_from_path(Path(path)))
    else:
        stdin_lines = list(sys.stdin)
        samples.extend(_read_samples_from_stream(stdin_lines, source_id=STDIN_SOURCE_ID))
        kernel_samples.extend(_read_kernel_samples_from_stream(stdin_lines, source_id=STDIN_SOURCE_ID))

    report = build_report(samples, args.window, kernel_compile_samples=kernel_samples)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human_report(report, args.window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
