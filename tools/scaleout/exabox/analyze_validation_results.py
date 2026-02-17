#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze validation output logs for health status and errors."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Constants
EXPECTED_ITERATIONS = 50
TIMEOUT_ESCALATION_THRESHOLD = 10.0  # 10% failure rate
SUCCESS_RATE_EXCELLENT = 90.0  # Green success rate color threshold
SUCCESS_RATE_GOOD = 70.0  # Yellow success rate color threshold
SUCCESS_RATE_STABLE = 80.0  # "Ready for workloads" threshold in recommendations
MAX_DISPLAY_FILES = 10
MAX_HISTOGRAM_ENTRIES = 20
MAX_ERROR_MESSAGES = 15
MAX_MATCHED_LINES = 3
MAX_LINE_PREVIEW = 200
MAX_ERROR_PREVIEW = 100
TIMELINE_ROW_SIZE = 25


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in ["GREEN", "RED", "YELLOW", "BLUE", "CYAN", "BOLD", "NC"]:
            setattr(cls, attr, "")


def clean_line(line: str) -> str:
    """Remove MPI stdout/stderr prefixes from log line."""
    for prefix in ["<stdout>:", "<stderr>:"]:
        if prefix in line:
            line = line.split(prefix)[-1]
    return line.strip()


# Category definitions with display info: (pattern, color_name, label, case_insensitive?)
# - pattern: regex pattern to match
# - color_name: name of Colors attribute (resolved at runtime for --no-color support)
# - label: human-readable label for display
# - case_insensitive: optional 4th element, True for case-insensitive matching (default: False)
# Only report specific failure modes; everything else is inconclusive
CATEGORIES = {
    # Success
    "healthy": (r"All Detected Links are healthy", "GREEN", "Healthy links"),
    # Link health (traffic/data tests)
    "unhealthy": (r"Found Unhealthy Links|FAULTY LINKS REPORT", "RED", "Unhealthy links"),
    # Connectivity issues
    "missing_ports": (
        r"missing port/cable connections",
        "BLUE",
        "Missing port connections",
    ),
    "missing_channels": (
        r"missing channel connections",
        "BLUE",
        "Missing channel connections",
    ),
    "extra_connections": (
        r"extra port/cable connections|extra channel connections",
        "YELLOW",
        "Extra connections",
    ),
    # Timeouts and failures
    "workload_timeout": (
        r"Workload execution timed out|cluster is not in a healthy state",
        "YELLOW",
        "Workload timeout",
    ),
    "dram_failure": (r"DRAM training failed|gddr issue", "RED", "DRAM training failures", True),
    "arc_timeout": (r"ARC.*[Tt]imeout|ARC message timed out", "YELLOW", "ARC timeout"),
    "aiclk_timeout": (
        r"Waiting for AICLK value to settle failed on timeout",
        "RED",
        "AICLK timeout",
    ),
    # Connectivity issues
    "mpi_error": (r"PRTE has lost communication|MPI_ABORT", "RED", "MPI error"),
    "ssh_error": (r"Permission denied \(publickey\)", "YELLOW", "SSH error"),
}


def get_color(name: str) -> str:
    """Get color code by name, respecting --no-color flag."""
    return getattr(Colors, name, "")


# Patterns for detecting other issues (will be mapped to inconclusive)
# These are detected but not reported as separate categories - they indicate
# issues that require manual log review and triage
DETECTION_PATTERNS = {
    "unrecoverable": r"Encountered unrecoverable state",
    "validation_failed": r"Cluster validation failed",
    "device_error": r"Error starting devices|Error details:",
    "discovery_failed": r"Physical Discovery.*failed",
    "crc_error": r"CRC Error",
    "uncorrected_cw": r"Uncorrected CW",
    "data_mismatch": r"Data Mismatch",
    "stack_trace": r"TT_FATAL|TT_THROW|std::runtime_error",
    "truncated": r"Sending traffic across detected links",
}

# Compile patterns with explicit case-sensitivity from 4th tuple element (default: case-sensitive)
PATTERNS = {k: re.compile(v[0], re.IGNORECASE if (len(v) > 3 and v[3]) else 0) for k, v in CATEGORIES.items()}
# Detection patterns are all case-sensitive
DETECTION_PATTERNS_COMPILED = {k: re.compile(v) for k, v in DETECTION_PATTERNS.items()}

# Patterns for parsing structured data
PORT_PATTERN = re.compile(
    r"PhysicalPortEndpoint\{hostname='([^']+)',[^}]*tray_id=(\d+), port_type=(\w+), port_id=(\d+)\}"
)
CHANNEL_PATTERN = re.compile(
    r"PhysicalChannelEndpoint\{hostname='([^']+)', tray_id=(\d+), asic_channel=AsicChannel\{asic_location=(\d+), channel_id=(\d+)\}\}"
)


@dataclass
class FaultyLink:
    host: str
    tray: int
    asic: int
    channel: int
    port_id: int
    port_type: str
    retrains: int = 0
    crc_errors: int = 0
    uncorrected_cw: int = 0
    mismatch_words: int = 0
    failure_type: str = ""


@dataclass
class LogMetadata:
    iteration: int = 0
    hosts: list[str] = field(default_factory=list)
    chips_found: int = 0
    traffic_iters: int = 0
    pkt_size: int = 0
    data_size: int = 0


@dataclass
class LogAnalysis:
    filepath: str
    categories: list[str] = field(default_factory=list)
    faulty_links: list[FaultyLink] = field(default_factory=list)
    missing_connections: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    matched_lines: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    metadata: LogMetadata = field(default_factory=LogMetadata)
    content: str = ""  # Store for later use


def _shorten_log_name(log_name: str) -> str:
    """Shorten log filename for display."""
    return log_name.replace("cluster_validation_iteration_", "iter_").replace(".log", "")


def parse_int(s: str) -> int:
    """Parse int, handling hex prefix (0x...)."""
    try:
        return int(s, 16) if s.startswith("0x") else int(s)
    except (ValueError, AttributeError):
        return 0


def _has_qsfp_connections(analyses: list[LogAnalysis]) -> bool:
    """Check if any missing port connections are QSFP."""
    # PORT_PATTERN captures: (hostname, tray_id, port_type, port_id)
    # Index 2 is port_type
    for a in analyses:
        if "missing_ports" not in a.categories:
            continue
        for conn in a.missing_connections:
            if conn[0] == "port":
                # conn[1] and conn[2] are tuples from PORT_PATTERN
                for endpoint in (conn[1], conn[2]):
                    if len(endpoint) > 2 and "QSFP" in str(endpoint[2]).upper():
                        return True
    return False


def parse_faulty_links(content: str) -> list[FaultyLink]:
    """Parse FAULTY LINKS REPORT section."""
    # Column indices after parsing (accounting for potential unique_id insertion)
    COL_HOST = 0
    COL_TRAY = 1
    COL_ASIC = 2
    COL_CHANNEL = 3
    COL_PORT_ID = 4
    COL_PORT_TYPE = 5
    COL_UNIQUE_ID = 6  # May be inserted if port_type is glued to unique_id
    COL_RETRAINS = 7
    COL_CRC = 8
    COL_UNCORRECTED_CW = 10  # CorrectedCW at 9 is skipped (not used)
    COL_MISMATCH_START = 11  # MismatchWords may be missing, starts here or later

    links = []
    in_report = header_seen = False
    # Split once and reuse
    lines = content.split("\n")

    for line in lines:
        c = clean_line(line)
        if "FAULTY LINKS REPORT" in line:
            in_report, header_seen = True, False
            continue
        if not in_report:
            continue
        if c.startswith(("╚", "═", "-")) or "Total Faulty" in c or not c:
            continue
        if "Host" in c and "Tray" in c:
            header_seen = True
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}", c):
            break

        if header_seen and re.match(r"^[a-zA-Z][\w\-]+", c):
            parts = c.split()
            # Minimum required columns: Host, Tray, ASIC, Ch, PortID, PortType, UniqueID, Retrains, CRC, CorrectedCW, UncorrectedCW
            # That's 11 columns minimum. If unique_id is glued to port_type, we'll need to split it.
            if len(parts) < 11:
                continue
            try:
                # Handle port_type glued to unique_id (e.g., "TRACE0x...")
                # This shifts all subsequent indices, so we need to rebuild parts
                port_type_raw = parts[COL_PORT_TYPE]
                if "0x" in port_type_raw and not port_type_raw.startswith("0x"):
                    idx = port_type_raw.find("0x")
                    port_type = port_type_raw[:idx]
                    unique_id = port_type_raw[idx:]
                    # Rebuild parts with unique_id inserted at correct position
                    parts = parts[:COL_UNIQUE_ID] + [unique_id] + parts[COL_UNIQUE_ID:]
                else:
                    port_type = port_type_raw

                # After potential rebuild, verify we have enough columns for all accesses
                # We need at least: Host(0), Tray(1), ASIC(2), Ch(3), PortID(4), PortType(5),
                # UniqueID(6), Retrains(7), CRC(8), CorrectedCW(9), UncorrectedCW(10)
                if len(parts) <= COL_UNCORRECTED_CW:
                    continue

                # Find where failure type starts (first non-numeric after UncorrectedCW)
                # MismatchWords column is often missing - detect if parts[11] is numeric or failure type
                ft_start = COL_MISMATCH_START
                mismatch = 0
                for i in range(COL_MISMATCH_START, min(COL_MISMATCH_START + 2, len(parts))):
                    val = parts[i] if i < len(parts) else ""
                    if val.isdigit() or (val.startswith("0x") and len(val) > 2):
                        mismatch = parse_int(val)
                        ft_start = i + 1
                    else:
                        break

                # Capture failure type (until we hit packet size like "64 B" or "64B")
                failure_parts = []
                for i in range(ft_start, min(ft_start + 10, len(parts))):
                    p = parts[i]
                    # Stop at packet size indicators
                    if p == "B" or re.match(r"^\d+$", p):
                        break
                    # Clean up concatenated strings like "Mismatch64"
                    if re.match(r"^[A-Za-z]+\d+$", p):
                        p = re.sub(r"\d+$", "", p)
                    failure_parts.append(p)
                failure_type = " ".join(failure_parts)
                # If failure type mentions "Mismatch" but count is 0, set to 1 (indicates mismatch occurred)
                if mismatch == 0 and "Mismatch" in failure_type:
                    mismatch = 1

                links.append(
                    FaultyLink(
                        host=parts[COL_HOST],
                        tray=int(parts[COL_TRAY]),
                        asic=int(parts[COL_ASIC]),
                        channel=int(parts[COL_CHANNEL]),
                        port_id=int(parts[COL_PORT_ID]),
                        port_type=port_type,
                        retrains=parse_int(parts[COL_RETRAINS]) if len(parts) > COL_RETRAINS else 0,
                        crc_errors=parse_int(parts[COL_CRC]) if len(parts) > COL_CRC else 0,
                        uncorrected_cw=parse_int(parts[COL_UNCORRECTED_CW]) if len(parts) > COL_UNCORRECTED_CW else 0,
                        mismatch_words=mismatch,
                        failure_type=failure_type,
                    )
                )
            except (ValueError, IndexError) as e:
                # Log parsing errors while keeping script robust
                print(f"Warning: Failed to parse faulty link line: {c[:100]}... ({e})", file=sys.stderr)
    return links


def parse_missing_connections(content: str) -> list[tuple[str, tuple, tuple]]:
    """Parse missing port/channel connections."""
    connections = []
    mode = None  # "port" or "channel"
    # Split once and reuse
    lines = content.split("\n")

    for line in lines:
        c = clean_line(line)
        # Use cleaned line consistently for all pattern matching
        if "missing port/cable" in c or "Port Connections found in FSD" in c:
            mode = "port"
        elif "missing channel" in c or "Channel Connections found in FSD" in c:
            mode = "channel"
        elif mode == "port" and "PhysicalPortEndpoint" in c:
            matches = PORT_PATTERN.findall(c)
            if len(matches) == 2:
                connections.append(("port", matches[0], matches[1]))
        elif mode == "channel" and "PhysicalChannelEndpoint" in c:
            matches = CHANNEL_PATTERN.findall(c)
            if len(matches) == 2:
                connections.append(("channel", matches[0], matches[1]))
        elif mode and c and not c.startswith("-") and "Physical" not in c:
            if "Connections" not in c and "Total" not in c:
                mode = None
    return connections


def parse_metadata(content: str, filepath: str) -> LogMetadata:
    """Extract metadata from log content."""
    m = LogMetadata()
    # Iteration from filename
    match = re.search(r"iteration_(\d+)", os.path.basename(filepath))
    if match:
        m.iteration = int(match.group(1))

    # Split once and reuse
    lines = content.split("\n")
    for line in lines:
        if "Iteration:" in line:
            match = re.search(r"Iteration:\s*(\d+)", line)
            if match:
                m.iteration = int(match.group(1))
        elif "Detected Hosts:" in line:
            match = re.search(r"Detected Hosts:\s*(.+?)(?:\s*\(|$)", line)
            if match:
                hosts = [h.strip() for h in match.group(1).split(",") if h.strip()]
                m.hosts = [h for h in hosts if re.match(r"^[a-zA-Z][\w\-]*$", h)]
        elif "chips found" in line:
            match = re.search(r"All (\d+) chips found", line)
            if match:
                m.chips_found = int(match.group(1))
        elif "Sending traffic" in line:
            for pat, attr in [
                (r"Num Iterations:\s*(\d+)", "traffic_iters"),
                (r"Packet Size.*?:\s*(\d+)", "pkt_size"),
                (r"Data Size.*?:\s*(\d+)", "data_size"),
            ]:
                match = re.search(pat, line)
                if match:
                    setattr(m, attr, int(match.group(1)))
    return m


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a single log file."""
    result = LogAnalysis(filepath=filepath)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            result.content = f.read()
    except (OSError, IOError, PermissionError) as e:
        print(f"Warning: Failed to read {filepath}: {e}", file=sys.stderr)
        return result

    result.metadata = parse_metadata(result.content, filepath)
    # Split once and reuse
    lines = result.content.split("\n")

    # Match patterns for reported categories
    for cat, pattern in PATTERNS.items():
        matched = [(i + 1, clean_line(line)[:MAX_LINE_PREVIEW]) for i, line in enumerate(lines) if pattern.search(line)]
        if matched:
            result.categories.append(cat)
            result.matched_lines[cat] = matched

    # Parse structured data
    if "unhealthy" in result.categories:
        result.faulty_links = parse_faulty_links(result.content)
    if "missing_ports" in result.categories or "missing_channels" in result.categories:
        result.missing_connections = parse_missing_connections(result.content)

    # If no reported categories found, check for other issues and mark as inconclusive
    if not result.categories:
        # Check for other issues that should be mapped to inconclusive
        # Use any() with generator for early exit
        if any(pattern.search(line) for pattern in DETECTION_PATTERNS_COMPILED.values() for line in lines):
            result.categories.append("inconclusive")

    return result


def aggregate_stats(analyses: list[LogAnalysis]) -> tuple[dict[tuple, dict[str, int]], dict[str, dict[str, int]]]:
    """Aggregate link and host statistics."""
    link_stats = defaultdict(lambda: {"count": 0, "retrains": 0, "crc": 0, "mismatch": 0})
    host_stats = defaultdict(lambda: {"links": 0, "missing": 0})

    for a in analyses:
        for link in a.faulty_links:
            key = (link.host, link.tray, link.asic, link.channel, link.port_type)
            link_stats[key]["count"] += 1
            link_stats[key]["retrains"] += link.retrains
            link_stats[key]["crc"] += link.crc_errors
            link_stats[key]["mismatch"] += link.mismatch_words
            host_stats[link.host]["links"] += 1
        for conn in a.missing_connections:
            # conn format: (type, endpoint1, endpoint2) where endpoints are tuples from regex
            # Validate structure before accessing
            if len(conn) >= 3 and len(conn[1]) > 0 and len(conn[2]) > 0:
                host_stats[conn[1][0]]["missing"] += 1
                host_stats[conn[2][0]]["missing"] += 1
    return dict(link_stats), dict(host_stats)


def get_cluster_info(analyses: list[LogAnalysis]) -> dict[str, Any]:
    """Extract cluster configuration from analyses."""
    hosts, chips, config = set(), 0, None
    for a in analyses:
        hosts.update(a.metadata.hosts)
        chips = max(chips, a.metadata.chips_found)
        if a.metadata.traffic_iters and not config:
            config = a.metadata
    return {"hosts": sorted(hosts), "chips": chips, "config": config}


def print_summary(analyses: list[LogAnalysis], show_files: bool = True) -> None:
    """Print analysis summary."""
    total = len(analyses)
    cat_counts = defaultdict(list)
    for a in analyses:
        for cat in a.categories:
            cat_counts[cat].append(a.filepath)

    print("=" * 50)
    print("Validation Results Analysis")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")

    # Cluster info
    info = get_cluster_info(analyses)
    if info["hosts"] or info["chips"]:
        print(f"{Colors.CYAN}Cluster Configuration:{Colors.NC}")
        if info["hosts"]:
            print(f"  Hosts: {', '.join(info['hosts'])}")
        if info["chips"]:
            if info["hosts"]:
                print(f"  Chips per host: {info['chips']}, Total: {info['chips'] * len(info['hosts'])}")
            else:
                print(f"  Chips found: {info['chips']}")
        if info["config"]:
            c = info["config"]
            print(f"  Traffic: {c.traffic_iters} iters, {c.pkt_size}B pkts, {c.data_size}B data")
        print()

    print(f"Total log files: {total}")
    if total < EXPECTED_ITERATIONS:
        print(f"{Colors.YELLOW}Warning: Expected {EXPECTED_ITERATIONS} iterations, found {total}{Colors.NC}")
    print()

    # Category breakdown - only show reported categories
    # Maintain original order for consistency with existing reports
    display_order = [
        "healthy",
        "unhealthy",
        "missing_ports",
        "missing_channels",
        "extra_connections",
        "workload_timeout",
        "dram_failure",
        "arc_timeout",
        "aiclk_timeout",
        "mpi_error",
        "ssh_error",
        "inconclusive",
    ]
    for cat in display_order:
        if cat not in CATEGORIES and cat != "inconclusive":
            continue
        if cat == "inconclusive":
            color = Colors.CYAN
            label = "Inconclusive"
        else:
            color = get_color(CATEGORIES[cat][1])
            label = CATEGORIES[cat][2]
        count = len(cat_counts.get(cat, []))
        print(f"{color}{label}:{Colors.NC} {count} / {total}")
        if show_files and 0 < count <= MAX_DISPLAY_FILES:
            for f in cat_counts[cat]:
                print(f"    {os.path.basename(f)}")
    print()

    # Success rate
    healthy = len(cat_counts.get("healthy", []))
    if total > 0:
        rate = healthy / total * 100
        color = (
            Colors.GREEN
            if rate >= SUCCESS_RATE_EXCELLENT
            else Colors.YELLOW
            if rate >= SUCCESS_RATE_GOOD
            else Colors.RED
        )
        print(f"{Colors.BOLD}Success Rate:{Colors.NC} {color}{rate:.1f}%{Colors.NC}\n")


def print_details(analyses: list[LogAnalysis]) -> None:
    """Print detailed faulty links and missing connections with proper formatting."""
    # Faulty links with table formatting
    all_links = [(os.path.basename(a.filepath), l) for a in analyses for l in a.faulty_links]
    if all_links:
        print("=" * 120)
        print("Faulty Links Detail")
        print("=" * 120 + "\n")

        # Calculate dynamic column widths
        host_w = max(len("Host"), max(len(l.host) for _, l in all_links))
        type_w = max(len("Type"), max(len(l.port_type) for _, l in all_links))

        # Header
        print(
            f"{'Log':<25}  {'Host':<{host_w}}  {'Tray':>4}  {'ASIC':>4}  {'Ch':>2}  "
            f"{'Type':<{type_w}}  {'Port':>4}  {'Retrains':>8}  {'CRC':>8}  {'Uncorr':>6}  {'Mismatch':>8}"
        )
        print("-" * 120)

        for log, l in all_links:
            short = _shorten_log_name(log)
            print(
                f"{short:<25}  {l.host:<{host_w}}  {l.tray:>4}  {l.asic:>4}  {l.channel:>2}  "
                f"{l.port_type:<{type_w}}  {l.port_id:>4}  {l.retrains:>8}  {l.crc_errors:>8}  "
                f"{l.uncorrected_cw:>6}  {l.mismatch_words:>8}"
            )
        print()

        # Failure type breakdown
        failure_types: dict[str, int] = {}
        for _, l in all_links:
            failure_types[l.failure_type] = failure_types.get(l.failure_type, 0) + 1
        if failure_types:
            print("Failure Type Breakdown:")
            for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
                print(f"  {count:>3}x  {ftype}")
            print()

    # Missing connections grouped by type
    all_missing = [(os.path.basename(a.filepath), c) for a in analyses for c in a.missing_connections]
    if all_missing:
        print("=" * 100)
        print("Missing Connections")
        print("=" * 100 + "\n")

        # Group by connection type
        port_conns = [(log, c) for log, c in all_missing if c[0] == "port"]
        chan_conns = [(log, c) for log, c in all_missing if c[0] == "channel"]

        if port_conns:
            print(f"Port/Cable Connections ({len(port_conns)}):")
            for log, (_, ep1, ep2) in port_conns:
                short = _shorten_log_name(log)
                print(f"  {short}: {ep1[0]} tray {ep1[1]} {ep1[2]} <-> {ep2[0]} tray {ep2[1]} {ep2[2]}")
            print()

        if chan_conns:
            print(f"Channel Connections ({len(chan_conns)}):")
            for log, (_, ep1, ep2) in chan_conns:
                short = _shorten_log_name(log)
                print(
                    f"  {short}: {ep1[0]} tray {ep1[1]} ASIC {ep1[2]} ch {ep1[3]} <-> "
                    f"{ep2[0]} tray {ep2[1]} ASIC {ep2[2]} ch {ep2[3]}"
                )
            print()

        # Host pair summary
        host_pairs: dict[tuple, int] = {}
        for _, (_, ep1, ep2) in all_missing:
            pair = tuple(sorted([ep1[0], ep2[0]]))
            host_pairs[pair] = host_pairs.get(pair, 0) + 1
        if host_pairs:
            print("Affected Host Pairs:")
            for (h1, h2), count in sorted(host_pairs.items(), key=lambda x: -x[1]):
                print(f"  {count:>3}x  {h1} <-> {h2}")
            print()


def print_link_histogram(analyses: list[LogAnalysis]) -> None:
    """Print histogram of failing links by frequency."""
    link_stats, _ = aggregate_stats(analyses)
    if not link_stats:
        return

    print("=" * 100)
    print("Faulty Link Histogram (by frequency)")
    print("=" * 100 + "\n")

    sorted_links = sorted(link_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:MAX_HISTOGRAM_ENTRIES]
    if not sorted_links:
        return

    # Dynamic column widths
    # Key format: (host, tray, asic, channel, port_type)
    host_w = max(len("Host"), max(len(k[0]) for k, _ in sorted_links))
    type_w = max(len("Type"), max(len(k[4]) for k, _ in sorted_links))

    # Header
    header = f"{'Host':<{host_w}}  {'Tray':>4}  {'ASIC':>4}  {'Ch':>2}  {'Type':<{type_w}}  {'Fails':>5}  {'Retrains':>8}  {'CRC Err':>8}  {'Mismatch':>8}"
    print(header)
    print("-" * len(header))

    for (host, tray, asic, channel, port_type), stats in sorted_links:
        print(
            f"{host:<{host_w}}  {tray:>4}  {asic:>4}  {channel:>2}  {port_type:<{type_w}}  "
            f"{stats['count']:>5}  {stats['retrains']:>8}  {stats['crc']:>8}  {stats['mismatch']:>8}"
        )

    if len(link_stats) > MAX_HISTOGRAM_ENTRIES:
        print(f"... and {len(link_stats) - MAX_HISTOGRAM_ENTRIES} more")
    print()


def print_verbose(analyses: list[LogAnalysis]) -> None:
    """Print verbose output with matched log lines as evidence."""
    print("=" * 50)
    print("Matched Log Lines (Evidence)")
    print("=" * 50 + "\n")

    for a in analyses:
        if not a.matched_lines:
            continue
        log_name = os.path.basename(a.filepath)
        printed_header = False

        for cat, matches in a.matched_lines.items():
            if cat == "healthy":
                continue
            if not printed_header:
                print(f"{Colors.BOLD}{log_name}{Colors.NC}")
                printed_header = True

            # Get label from CATEGORIES or use category name
            label = CATEGORIES.get(cat, (None, None, cat))[2] if cat in CATEGORIES else cat
            print(f"  {Colors.CYAN}{label}:{Colors.NC}")
            for line_num, content in matches[:MAX_MATCHED_LINES]:
                preview_len = MAX_ERROR_PREVIEW
                print(f"    L{line_num}: {content[:preview_len]}{'...' if len(content) > preview_len else ''}")

        if printed_header:
            print()


def print_host_summary(analyses: list[LogAnalysis]) -> None:
    """Print per-host failure summary."""
    _, host_stats = aggregate_stats(analyses)
    if not host_stats:
        return
    print("=" * 50 + "\nPer-Host Summary\n" + "=" * 50 + "\n")
    print(f"{'Host':<30} {'Links':>8} {'Missing':>8}")
    print("-" * 48)
    for host, s in sorted(host_stats.items(), key=lambda x: -x[1]["links"]):
        print(f"{host:<30} {s['links']:>8} {s['missing']:>8}")
    print()


# Recommendation generator registry - makes it easy to add new category recommendations
# Each function receives: (category_counts, total, analyses) and returns list of recommendation strings
RECOMMENDATION_GENERATORS = {}


def register_recommendation(category: str):
    """Decorator to register a recommendation generator for a category."""

    def decorator(func):
        RECOMMENDATION_GENERATORS[category] = func
        return func

    return decorator


@register_recommendation("missing_ports")
@register_recommendation("missing_channels")
def _recommend_missing_connections(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for missing connections."""
    if not (cats.get("missing_ports") or cats.get("missing_channels")):
        return []

    port_count, chan_count = cats.get("missing_ports", 0), cats.get("missing_channels", 0)

    # Check if missing connections are transient (appear in some but not all logs)
    # Count unique logs that have ANY missing connection issue (port OR channel)
    logs_with_missing = sum(
        1 for a in analyses if "missing_ports" in a.categories or "missing_channels" in a.categories
    )
    is_transient = logs_with_missing < total

    # Check if any missing connections are QSFP
    has_qsfp = _has_qsfp_connections(analyses)

    # Build recommendation message
    msg_parts = [f"- {Colors.BLUE}Missing connections:{Colors.NC}"]

    if port_count and chan_count:
        msg_parts.append(f"Port ({port_count} logs) + Channel ({chan_count} logs).")
    elif port_count:
        msg_parts.append(f"Port connections ({port_count} logs).")
    elif chan_count:
        msg_parts.append(f"Channel connections ({chan_count} logs).")

    if is_transient:
        msg_parts.append(
            "Missing connections are transient (tests fail some of the time) - " "Factory System Descriptor is correct."
        )

    if has_qsfp:
        msg_parts.append("If missing connections are over QSFP, check cable seating.")

    msg_parts.append("Report to cluster installation team/Syseng for triage.")

    return [" ".join(msg_parts)]


@register_recommendation("extra_connections")
def _recommend_extra_connections(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for extra connections."""
    if not cats.get("extra_connections"):
        return []
    return [
        f"- {Colors.YELLOW}Extra connections:{Colors.NC} Unexpected links found. Verify correct FSD file for your topology."
    ]


@register_recommendation("workload_timeout")
def _recommend_workload_timeout(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for workload timeout."""
    if not cats.get("workload_timeout") or total <= 0:
        return []
    timeout_rate = cats["workload_timeout"] / total * 100
    msg = (
        f"- {Colors.YELLOW}Workload timeout:{Colors.NC} Traffic tests timed out ({cats['workload_timeout']} "
        f"occurrence(s), {timeout_rate:.1f}% of runs). This indicates an ethernet issue. "
        "Document timeout occurrences. "
    )
    if timeout_rate >= TIMEOUT_ESCALATION_THRESHOLD:
        msg += "High failure rate - escalate to Syseng."
    else:
        msg += "Report statistics to Syseng."
    return [msg]


@register_recommendation("dram_failure")
def _recommend_dram_failure(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for DRAM failures."""
    if not cats.get("dram_failure"):
        return []
    return [f"- {Colors.RED}DRAM training failures:{Colors.NC} Hardware issue. Report to Syseng."]


@register_recommendation("arc_timeout")
def _recommend_arc_timeout(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for ARC timeout."""
    if not cats.get("arc_timeout"):
        return []
    return [f"- {Colors.YELLOW}ARC timeout:{Colors.NC} ARC communication issues detected. Report statistics to Syseng."]


@register_recommendation("aiclk_timeout")
def _recommend_aiclk_timeout(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for AICLK timeout."""
    if not cats.get("aiclk_timeout"):
        return []
    count = cats["aiclk_timeout"]
    return [
        f"- {Colors.RED}AICLK timeout:{Colors.NC} AICLK failed to settle ({count} occurrence(s)). "
        f"Could indicate bad Firmware or Hardware state. Escalate to Systems Engineering."
    ]


@register_recommendation("mpi_error")
def _recommend_mpi_error(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for MPI errors."""
    if not cats.get("mpi_error"):
        return []
    return [f"- {Colors.RED}MPI error:{Colors.NC} Lost connection between hosts. Check SSH agent and network."]


@register_recommendation("ssh_error")
def _recommend_ssh_error(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for SSH errors."""
    if not cats.get("ssh_error"):
        return []
    return [f"- {Colors.YELLOW}SSH errors:{Colors.NC} Authentication failed. Ensure ssh-agent running and keys added."]


@register_recommendation("unhealthy")
def _recommend_unhealthy_links(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for unhealthy links."""
    if not cats.get("unhealthy"):
        return []
    return [
        f"- {Colors.RED}Unhealthy links:{Colors.NC} Link failures detected ({cats['unhealthy']} occurrence(s)). Review the failure histogram and timeline for details. Report to Syseng."
    ]


@register_recommendation("inconclusive")
def _recommend_inconclusive(cats: dict, total: int, analyses: list[LogAnalysis]) -> list[str]:
    """Generate recommendations for inconclusive results."""
    if not cats.get("inconclusive"):
        return []
    return [
        f"- {Colors.CYAN}Inconclusive results:{Colors.NC} {cats['inconclusive']} log(s) contain issues outside the reported categories. Review log files manually for triage."
    ]


def print_recommendations(analyses: list[LogAnalysis]) -> None:
    """Print actionable recommendations based on analysis."""
    cat_counts = defaultdict(int)
    for a in analyses:
        for c in a.categories:
            cat_counts[c] += 1
    total = len(analyses)
    if not total:
        return

    print("=" * 50 + "\nRecommendations\n" + "=" * 50 + "\n")
    recs = []

    # Use registry pattern to generate recommendations
    # Process categories in a consistent order
    processed_generators = set()
    category_order = [
        "missing_ports",
        "missing_channels",
        "extra_connections",
        "workload_timeout",
        "dram_failure",
        "arc_timeout",
        "aiclk_timeout",
        "mpi_error",
        "ssh_error",
        "unhealthy",
        "inconclusive",
    ]

    for cat in category_order:
        if cat in RECOMMENDATION_GENERATORS:
            generator = RECOMMENDATION_GENERATORS[cat]
            # Avoid calling the same generator twice (e.g., missing_ports and missing_channels)
            if generator not in processed_generators:
                recs.extend(generator(cat_counts, total, analyses))
                processed_generators.add(generator)

    # Special handling for discovery_failed (not a category, but needs recommendation)
    discovery_failed_count = 0
    for a in analyses:
        # Check if discovery_failed pattern was detected (even if mapped to inconclusive)
        if DETECTION_PATTERNS_COMPILED["discovery_failed"].search(a.content):
            discovery_failed_count += 1

    if discovery_failed_count > 0:
        recs.append(
            f"- {Colors.RED}Discovery failed:{Colors.NC} Physical discovery found no chips ({discovery_failed_count} "
            f"occurrence(s)). This is a critical red flag. Notify Syseng immediately."
        )

    # Overall health assessment
    if total > 0:
        rate = cat_counts["healthy"] / total * 100
        if rate >= 100:
            recs.append(f"- {Colors.GREEN}Cluster is healthy.{Colors.NC} Ready for workloads.")
        elif rate >= SUCCESS_RATE_STABLE:
            recs.append(f"- {Colors.YELLOW}Cluster looks stable ({rate:.0f}%).{Colors.NC} Ready for workloads.")
        elif rate > 0:
            recs.append(
                f"- {Colors.RED}Low success rate ({rate:.0f}%).{Colors.NC} Investigate failure patterns before proceeding."
            )
    else:
        recs.append(f"- {Colors.YELLOW}No validation logs analyzed.{Colors.NC} Unable to determine cluster health.")

    for r in recs:
        print(r)
    print()


def print_timeline(analyses: list[LogAnalysis]) -> None:
    """Print visual iteration timeline with specific category indicators."""
    if not analyses:
        return
    print("=" * 50 + "\nIteration Timeline\n" + "=" * 50 + "\n")

    sorted_a = sorted(analyses, key=lambda a: a.metadata.iteration)
    icons = []
    for a in sorted_a:
        # Priority order: show most critical issues first
        # "healthy" is lowest priority - only shown if no issues detected
        # This ensures issues aren't hidden by a passing health check
        if "dram_failure" in a.categories:
            icons.append(("!", Colors.RED))
        elif "unhealthy" in a.categories:
            icons.append(("✗", Colors.RED))
        elif "mpi_error" in a.categories:
            icons.append(("M", Colors.RED))
        elif "workload_timeout" in a.categories:
            icons.append(("⏱", Colors.YELLOW))
        elif "arc_timeout" in a.categories:
            icons.append(("A", Colors.YELLOW))
        elif "aiclk_timeout" in a.categories:
            icons.append(("C", Colors.RED))
        elif "ssh_error" in a.categories:
            icons.append(("S", Colors.YELLOW))
        elif "missing_ports" in a.categories:
            icons.append(("○", Colors.BLUE))
        elif "missing_channels" in a.categories:
            icons.append(("○", Colors.BLUE))
        elif "extra_connections" in a.categories:
            icons.append(("+", Colors.YELLOW))
        elif "inconclusive" in a.categories:
            icons.append(("?", Colors.CYAN))
        elif "healthy" in a.categories:
            icons.append(("✓", Colors.GREEN))
        else:
            icons.append(("~", Colors.YELLOW))

    for start in range(0, len(icons), TIMELINE_ROW_SIZE):
        row = icons[start : start + TIMELINE_ROW_SIZE]
        print("  " + " ".join(f"{start + i + 1:2d}" for i in range(len(row))))
        print("  " + " ".join(f"{color}{icon}{Colors.NC} " for icon, color in row) + "\n")

    print(
        f"Legend: {Colors.GREEN}✓{Colors.NC}=healthy "
        f"{Colors.RED}✗{Colors.NC}=unhealthy "
        f"{Colors.BLUE}○{Colors.NC}=missing "
        f"{Colors.YELLOW}+{Colors.NC}=extra "
        f"{Colors.YELLOW}⏱{Colors.NC}=timeout "
        f"{Colors.RED}!{Colors.NC}=dram "
        f"{Colors.YELLOW}A{Colors.NC}=arc "
        f"{Colors.RED}C{Colors.NC}=aiclk "
        f"{Colors.RED}M{Colors.NC}=mpi "
        f"{Colors.YELLOW}S{Colors.NC}=ssh "
        f"{Colors.CYAN}?{Colors.NC}=inconclusive "
        f"{Colors.YELLOW}~{Colors.NC}=other\n"
    )


def print_errors(analyses: list[LogAnalysis]) -> None:
    """Print unique error messages."""
    error_re = re.compile(r"(TT_THROW|TT_FATAL|Error:|RuntimeError)", re.IGNORECASE)
    errors = defaultdict(list)

    for a in analyses:
        seen = set()
        # Split once per analysis
        lines = a.content.split("\n")
        for line in lines:
            c = clean_line(line)
            # Skip if no error keywords found
            if not error_re.search(c) or "Found Unhealthy" in c:
                continue
            # Skip noisy stderr lines (addresses/warnings) unless they have clear error indicators
            if "<stderr>:" in line and ("0x" in line or "usually indicates" in line):
                if not re.search(r"(Error|Failed|fatal|exception)", c, re.IGNORECASE):
                    continue
            # Normalize: remove timestamps, file paths, MPI prefixes
            msg = re.sub(r"^\d{4}-\d{2}-\d{2}.*?\|\s*", "", c)
            msg = re.sub(r"^\[\d+,\d+\]<\w+>:\s*", "", msg)
            msg = re.sub(r"\s*\([^)]+\.(cpp|hpp):\d+\)\s*$", "", msg).strip()
            if len(msg) > 15 and msg not in seen:
                seen.add(msg)
                errors[msg].append(os.path.basename(a.filepath))

    if not errors:
        return
    print("=" * 50 + "\nUnique Error Messages\n" + "=" * 50 + "\n")
    for msg, files in sorted(errors.items(), key=lambda x: -len(x[1]))[:MAX_ERROR_MESSAGES]:
        print(
            f"{Colors.RED}[{len(files)}x]{Colors.NC} {msg[:MAX_ERROR_PREVIEW]}{'...' if len(msg) > MAX_ERROR_PREVIEW else ''}"
        )
    print()


def output_json(analyses: list[LogAnalysis]):
    """Output JSON results."""
    info = get_cluster_info(analyses)
    cats = defaultdict(list)
    for a in analyses:
        for c in a.categories:
            cats[c].append(os.path.basename(a.filepath))

    healthy = len(cats.get("healthy", []))
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(analyses),
        "cluster_info": {
            "hosts": info["hosts"],
            "chips_per_host": info["chips"],
            "total_chips": info["chips"] * len(info["hosts"]) if info["hosts"] else info["chips"],
        },
        "categories": dict(cats),
        "summary": {"healthy_count": healthy, "success_rate": healthy / len(analyses) * 100 if analyses else 0},
    }
    print(json.dumps(result, indent=2))


def plot_link_histogram(analyses: list[LogAnalysis], output_dir: str) -> str | None:
    """Generate bar chart of top failing links by frequency."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping histogram plot", file=sys.stderr)
        return None

    link_stats, _ = aggregate_stats(analyses)
    if not link_stats:
        print("No faulty links to plot", file=sys.stderr)
        return None

    # Sort by count and take top entries
    sorted_links = sorted(link_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:MAX_HISTOGRAM_ENTRIES]

    if not sorted_links:
        return None

    # Prepare data
    labels = [f"{host}:T{tray}:A{asic}:Ch{ch}:{ptype}" for (host, tray, asic, ch, ptype), _ in sorted_links]
    counts = [stats["count"] for _, stats in sorted_links]
    crc_errors = [stats["crc"] for _, stats in sorted_links]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))

    y_pos = range(len(labels))
    ax.barh(y_pos, counts, color="steelblue", edgecolor="black", label="Occurrences")

    # Add CRC error overlay if present
    if any(crc_errors):
        ax.barh(y_pos, crc_errors, color="salmon", edgecolor="darkred", alpha=0.7, label="CRC Errors")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Top items at top
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(labels)} Failing Links\n(Total logs analyzed: {len(analyses)})")
    ax.legend(loc="lower right")

    # Add count labels on bars
    for i, (count, crc) in enumerate(zip(counts, crc_errors)):
        ax.text(count + 0.1, i, str(count), va="center", fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "faulty_links_histogram.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_host_tray_heatmap(analyses: list[LogAnalysis], output_dir: str) -> str | None:
    """Generate heatmap of failures by host and tray."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping heatmap plot", file=sys.stderr)
        return None

    # Aggregate by host and tray
    host_tray_counts: dict[tuple[str, int], int] = defaultdict(int)
    for a in analyses:
        for link in a.faulty_links:
            host_tray_counts[(link.host, link.tray)] += 1

    if not host_tray_counts:
        print("No faulty links to plot in heatmap", file=sys.stderr)
        return None

    # Get unique hosts and trays
    hosts = sorted(set(h for h, _ in host_tray_counts.keys()))
    trays = sorted(set(t for _, t in host_tray_counts.keys()))

    if not hosts or not trays:
        return None

    # Build matrix
    matrix = [[host_tray_counts.get((h, t), 0) for t in trays] for h in hosts]

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(trays) * 0.8), max(6, len(hosts) * 0.5)))

    # Create heatmap
    cmap = plt.cm.YlOrRd  # Yellow-Orange-Red colormap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Failure Count", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(range(len(trays)))
    ax.set_yticks(range(len(hosts)))
    ax.set_xticklabels([f"Tray {t}" for t in trays], fontsize=9)
    ax.set_yticklabels(hosts, fontsize=9)

    # Add text annotations
    for i, host in enumerate(hosts):
        for j, tray in enumerate(trays):
            value = matrix[i][j]
            if value > 0:
                text_color = "white" if value > max(max(row) for row in matrix) / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title(f"Link Failures by Host and Tray\n(Total logs: {len(analyses)})")
    ax.set_xlabel("Tray")
    ax.set_ylabel("Host")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "host_tray_heatmap.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_plots(analyses: list[LogAnalysis], output_dir: str) -> None:
    """Generate all plots and save to output directory."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib", file=sys.stderr)
        return

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 50}\nGenerating Plots\n{'=' * 50}\n")

    # Generate histogram
    hist_path = plot_link_histogram(analyses, output_dir)
    if hist_path:
        print(f"  Saved: {hist_path}")

    # Generate heatmap
    heatmap_path = plot_host_tray_heatmap(analyses, output_dir)
    if heatmap_path:
        print(f"  Saved: {heatmap_path}")

    if not hist_path and not heatmap_path:
        print("  No plots generated (no faulty link data)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze validation logs.")
    parser.add_argument("directory", nargs="?", default="validation_output", help="Log directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--all", action="store_true", help="Show all details")
    parser.add_argument("--histogram", action="store_true", help="Show link histogram")
    parser.add_argument("--hosts", action="store_true", help="Show host summary")
    parser.add_argument("--timeline", action="store_true", help="Show timeline")
    parser.add_argument("--errors", action="store_true", help="Show error messages")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    parser.add_argument("--plot-dir", type=str, default=".", help="Directory to save plots (default: current dir)")

    args = parser.parse_args()
    if args.no_color or args.json:
        Colors.disable()

    log_dir = Path(args.directory)
    if not log_dir.is_dir():
        print(f"Error: {args.directory} not found", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files in {args.directory}", file=sys.stderr)
        sys.exit(1)

    analyses = [analyze_log_file(str(f)) for f in log_files]

    if args.json:
        output_json(analyses)
    else:
        print_summary(analyses)
        # Always show recommendations - they're actionable and concise
        print_recommendations(analyses)
        if args.all:
            print_details(analyses)
        if args.histogram:
            print_link_histogram(analyses)
        if args.hosts or args.all:
            print_host_summary(analyses)
        if args.timeline or args.all:
            print_timeline(analyses)
        if args.errors or args.all:
            print_errors(analyses)
        if args.verbose:
            print_verbose(analyses)
        if args.plot:
            generate_plots(analyses, args.plot_dir)


if __name__ == "__main__":
    main()
