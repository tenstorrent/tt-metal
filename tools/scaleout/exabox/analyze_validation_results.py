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

SCRIPT_DIR = Path(__file__).parent.resolve()

# Troubleshooting section titles (stable references)
TROUBLESHOOTING_SECTIONS = {
    "tensix_stall": "Tensix Stall Issue",
    "gddr_issue": "GDDR Issue on Chip",
    "fw_mismatch": "UMD Firmware Version Mismatch",
    "missing_connections": "QSFP Connections Missing Between Hosts",
    "data_mismatch": "Data Mismatch During Traffic Tests",
    "ssh_agent": "SSH Agent Forwarding for MPI",
}


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    NC = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in ["GREEN", "RED", "YELLOW", "BLUE", "CYAN", "MAGENTA", "BOLD", "NC"]:
            setattr(cls, attr, "")


def ts_ref(section: str) -> str:
    """Generate troubleshooting reference using section title."""
    title = TROUBLESHOOTING_SECTIONS.get(section, section)
    return f"'{title}' in TROUBLESHOOTING.md"


def clean_line(line: str) -> str:
    """Remove MPI stdout/stderr prefixes from log line."""
    for prefix in ["<stdout>:", "<stderr>:"]:
        if prefix in line:
            line = line.split(prefix)[-1]
    return line.strip()


# Category definitions with display info: (pattern, color, label)
CATEGORIES = {
    # Success
    "healthy": (r"All Detected Links are healthy", Colors.GREEN, "Healthy links"),
    # Critical failures
    "unhealthy": (r"Found Unhealthy Links|FAULTY LINKS REPORT", Colors.RED, "Unhealthy links"),
    "unrecoverable": (r"Encountered unrecoverable state", Colors.RED, "Unrecoverable state"),
    "validation_failed": (r"Cluster validation failed", Colors.RED, "Validation failed"),
    "dram_failure": (r"DRAM training failed|gddr issue", Colors.RED, "DRAM/GDDR failures"),
    "pcie_error": (r"PCIe error|AER:.*aer_status|\[Hardware Error\]", Colors.RED, "PCIe errors"),
    "device_error": (r"Error starting devices|Error details:", Colors.RED, "Device startup errors"),
    # Timeouts
    "timeout": (r"Timeout \(\d+ ms\) waiting for physical cores", Colors.YELLOW, "Physical core timeout"),
    "workload_timeout": (
        r"Workload execution timed out|cluster is not in a healthy state",
        Colors.YELLOW,
        "Workload timeout",
    ),
    "arc_timeout": (r"ARC.*[Tt]imeout|ARC message timed out", Colors.YELLOW, "ARC timeout"),
    # Connectivity
    "missing_ports": (
        r"missing port/cable connections|Port Connections found in FSD but missing",
        Colors.BLUE,
        "Missing port connections",
    ),
    "missing_channels": (
        r"missing channel connections|Channel Connections found in FSD but missing",
        Colors.BLUE,
        "Missing channel connections",
    ),
    "extra_connections": (
        r"extra port/cable connections|Connections found in GSD but not in FSD",
        Colors.YELLOW,
        "Extra connections",
    ),
    "discovery_failed": (r"Physical Discovery.*failed|Discovery Complete.*0 chips", Colors.RED, "Discovery failed"),
    # Link health
    "link_retrain": (r"Retrain.*detected|Link retrains detected", Colors.YELLOW, "Link retrains"),
    "crc_error": (r"CRC Error|crc_error_count > 0", Colors.YELLOW, "CRC errors"),
    "uncorrected_cw": (r"Uncorrected CW|uncorrected_codeword", Colors.YELLOW, "Uncorrected codewords"),
    "data_mismatch": (r"Data Mismatch|mismatched_words|num_mismatched", Colors.RED, "Data mismatch"),
    # Other
    "fw_mismatch": (r"FW Bundle version mismatch|ERISC FW version.*mismatch", Colors.MAGENTA, "Firmware mismatch"),
    "stack_trace": (r"TT_FATAL|TT_THROW|std::runtime_error", Colors.RED, "Stack trace/crash"),
    "mpi_error": (r"PRTE has lost communication|MPI_ABORT", Colors.RED, "MPI error"),
    "ssh_error": (r"Permission denied \(publickey\)", Colors.YELLOW, "SSH error"),
    "truncated": (r"Sending traffic across detected links", Colors.YELLOW, "Truncated run"),
}

PATTERNS = {k: re.compile(v[0], re.IGNORECASE if "gddr" in v[0].lower() else 0) for k, v in CATEGORIES.items()}

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
    hosts: list = field(default_factory=list)
    chips_found: int = 0
    traffic_iters: int = 0
    pkt_size: int = 0
    data_size: int = 0


@dataclass
class LogAnalysis:
    filepath: str
    categories: list = field(default_factory=list)
    faulty_links: list = field(default_factory=list)
    missing_connections: list = field(default_factory=list)
    matched_lines: dict = field(default_factory=dict)
    metadata: LogMetadata = field(default_factory=LogMetadata)
    content: str = ""  # Store for later use


def parse_int(s: str, base: int = 10) -> int:
    """Parse int, handling hex prefix."""
    try:
        return int(s, 16) if s.startswith("0x") else int(s, base)
    except (ValueError, AttributeError):
        return 0


def parse_faulty_links(content: str) -> list[FaultyLink]:
    """Parse FAULTY LINKS REPORT section."""
    links = []
    in_report = header_seen = False

    for line in content.split("\n"):
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
            if len(parts) >= 11:
                try:
                    # Handle port_type glued to unique_id (e.g., "TRACE0x...")
                    # This shifts all subsequent indices, so we need to rebuild parts
                    port_type_raw = parts[5]
                    if "0x" in port_type_raw and not port_type_raw.startswith("0x"):
                        idx = port_type_raw.find("0x")
                        port_type = port_type_raw[:idx]
                        unique_id = port_type_raw[idx:]
                        # Rebuild parts with unique_id inserted at correct position
                        parts = parts[:6] + [unique_id] + parts[6:]
                    else:
                        port_type = port_type_raw

                    # Columns: Host(0) Tray(1) ASIC(2) Ch(3) PortID(4) PortType(5) UniqueID(6)
                    #          Retrains(7) CRC(8) CorrectedCW(9) UncorrectedCW(10) [MismatchWords] FailureType...
                    # Note: MismatchWords column is often missing - detect if parts[11] is numeric or failure type
                    # Find where failure type starts (first non-numeric after parts[10])
                    ft_start = 11
                    mismatch = 0
                    for i in range(11, min(13, len(parts))):
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
                            host=parts[0],
                            tray=int(parts[1]),
                            asic=int(parts[2]),
                            channel=int(parts[3]),
                            port_id=int(parts[4]),
                            port_type=port_type,
                            retrains=parse_int(parts[7]) if len(parts) > 7 else 0,
                            crc_errors=parse_int(parts[8]) if len(parts) > 8 else 0,
                            uncorrected_cw=parse_int(parts[10]) if len(parts) > 10 else 0,
                            mismatch_words=mismatch,
                            failure_type=failure_type,
                        )
                    )
                except (ValueError, IndexError):
                    pass
    return links


def parse_missing_connections(content: str) -> list[tuple]:
    """Parse missing port/channel connections."""
    connections = []
    mode = None  # "port" or "channel"

    for line in content.split("\n"):
        c = clean_line(line)
        if "missing port/cable" in line or "Port Connections found in FSD" in line:
            mode = "port"
        elif "missing channel" in line or "Channel Connections found in FSD" in line:
            mode = "channel"
        elif mode == "port" and "PhysicalPortEndpoint" in c:
            matches = PORT_PATTERN.findall(line)
            if len(matches) == 2:
                connections.append(("port", matches[0], matches[1]))
        elif mode == "channel" and "PhysicalChannelEndpoint" in c:
            matches = CHANNEL_PATTERN.findall(line)
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

    for line in content.split("\n"):
        if "Iteration:" in line:
            match = re.search(r"Iteration:\s*(\d+)", line)
            if match:
                m.iteration = int(match.group(1))
        elif "Detected Hosts:" in line:
            match = re.search(r"Detected Hosts:\s*(.+?)(?:\s*\(|$)", line)
            if match:
                hosts = [h.strip() for h in match.group(1).split(",") if h.strip()]
                m.hosts = [h for h in hosts if re.match(r"^[a-zA-Z][\w\-]+$", h)]
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
    except Exception:
        return result

    result.metadata = parse_metadata(result.content, filepath)
    lines = result.content.split("\n")

    # Match patterns
    for cat, pattern in PATTERNS.items():
        matched = [(i + 1, clean_line(line)[:200]) for i, line in enumerate(lines) if pattern.search(line)]
        if matched:
            result.categories.append(cat)
            result.matched_lines[cat] = matched

    # Parse structured data
    if "unhealthy" in result.categories:
        result.faulty_links = parse_faulty_links(result.content)
    if "missing_ports" in result.categories or "missing_channels" in result.categories:
        result.missing_connections = parse_missing_connections(result.content)

    # Handle truncated detection
    has_result = "healthy" in result.categories or "unhealthy" in result.categories
    if "truncated" in result.categories and has_result:
        result.categories.remove("truncated")
    if not [c for c in result.categories if c != "truncated"]:
        result.categories.append("indeterminate")

    return result


def aggregate_stats(analyses: list[LogAnalysis]) -> tuple[dict, dict]:
    """Aggregate link and host statistics."""
    link_stats = defaultdict(lambda: {"count": 0, "retrains": 0, "crc": 0, "mismatch": 0})
    host_stats = defaultdict(lambda: {"links": 0, "missing": 0})

    for a in analyses:
        for link in a.faulty_links:
            key = (link.host, link.tray, link.channel, link.port_type)
            link_stats[key]["count"] += 1
            link_stats[key]["retrains"] += link.retrains
            link_stats[key]["crc"] += link.crc_errors
            link_stats[key]["mismatch"] += link.mismatch_words
            host_stats[link.host]["links"] += 1
        for conn in a.missing_connections:
            host_stats[conn[1][0]]["missing"] += 1
            host_stats[conn[2][0]]["missing"] += 1
    return dict(link_stats), dict(host_stats)


def get_cluster_info(analyses: list[LogAnalysis]) -> dict:
    """Extract cluster configuration from analyses."""
    hosts, chips, config = set(), 0, None
    for a in analyses:
        hosts.update(a.metadata.hosts)
        chips = max(chips, a.metadata.chips_found)
        if a.metadata.traffic_iters and not config:
            config = a.metadata
    return {"hosts": sorted(hosts), "chips": chips, "config": config}


def print_summary(analyses: list[LogAnalysis], show_files: bool = True):
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
            print(f"  Chips per host: {info['chips']}, Total: {info['chips'] * len(info['hosts'])}")
        if info["config"]:
            c = info["config"]
            print(f"  Traffic: {c.traffic_iters} iters, {c.pkt_size}B pkts, {c.data_size}B data")
        print()

    print(f"Total log files: {total}")
    if total < 50:
        print(f"{Colors.YELLOW}Warning: Expected 50 iterations, found {total}{Colors.NC}")
    print()

    # Category breakdown
    display_order = [
        "healthy",
        "unhealthy",
        "unrecoverable",
        "validation_failed",
        "dram_failure",
        "pcie_error",
        "device_error",
        "timeout",
        "workload_timeout",
        "arc_timeout",
        "missing_ports",
        "missing_channels",
        "extra_connections",
        "discovery_failed",
        "link_retrain",
        "crc_error",
        "uncorrected_cw",
        "data_mismatch",
        "fw_mismatch",
        "stack_trace",
        "mpi_error",
        "ssh_error",
        "truncated",
        "indeterminate",
    ]
    for cat in display_order:
        if cat not in CATEGORIES and cat != "indeterminate":
            continue
        color = CATEGORIES.get(cat, (None, Colors.CYAN, "Indeterminate"))[1]
        label = CATEGORIES.get(cat, (None, None, "Indeterminate"))[2]
        count = len(cat_counts.get(cat, []))
        print(f"{color}{label}:{Colors.NC} {count} / {total}")
        if show_files and 0 < count <= 10:
            for f in cat_counts[cat]:
                print(f"    {os.path.basename(f)}")
    print()

    # Success rate
    healthy = len(cat_counts.get("healthy", []))
    if total > 0:
        rate = healthy / total * 100
        color = Colors.GREEN if rate >= 90 else Colors.YELLOW if rate >= 70 else Colors.RED
        print(f"{Colors.BOLD}Success Rate:{Colors.NC} {color}{rate:.1f}%{Colors.NC}\n")


def print_details(analyses: list[LogAnalysis]):
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
            short = log.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
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
                short = log.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
                print(f"  {short}: {ep1[0]} tray {ep1[1]} {ep1[2]} <-> {ep2[0]} tray {ep2[1]} {ep2[2]}")
            print()

        if chan_conns:
            print(f"Channel Connections ({len(chan_conns)}):")
            for log, (_, ep1, ep2) in chan_conns:
                short = log.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
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


def print_link_histogram(analyses: list[LogAnalysis]):
    """Print histogram of failing links by frequency."""
    link_stats, _ = aggregate_stats(analyses)
    if not link_stats:
        return

    print("=" * 100)
    print("Faulty Link Histogram (by frequency)")
    print("=" * 100 + "\n")

    sorted_links = sorted(link_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:20]
    if not sorted_links:
        return

    # Dynamic column widths
    host_w = max(len("Host"), max(len(k[0]) for k, _ in sorted_links))
    type_w = max(len("Type"), max(len(k[3]) for k, _ in sorted_links))

    # Header
    header = f"{'Host':<{host_w}}  {'Tray':>4}  {'Ch':>2}  {'Type':<{type_w}}  {'Fails':>5}  {'Retrains':>8}  {'CRC Err':>8}  {'Mismatch':>8}"
    print(header)
    print("-" * len(header))

    for (host, tray, channel, port_type), stats in sorted_links:
        print(
            f"{host:<{host_w}}  {tray:>4}  {channel:>2}  {port_type:<{type_w}}  "
            f"{stats['count']:>5}  {stats['retrains']:>8}  {stats['crc']:>8}  {stats['mismatch']:>8}"
        )

    if len(link_stats) > 20:
        print(f"... and {len(link_stats) - 20} more")
    print()


def print_verbose(analyses: list[LogAnalysis]):
    """Print verbose output with matched log lines as evidence."""
    print("=" * 50)
    print("Matched Log Lines (Evidence)")
    print("=" * 50 + "\n")

    category_labels = {cat: info[2] for cat, info in CATEGORIES.items()}

    for a in analyses:
        if not a.matched_lines:
            continue
        log_name = os.path.basename(a.filepath)
        printed_header = False

        for cat, matches in a.matched_lines.items():
            if cat in ["healthy", "truncated"]:
                continue
            if not printed_header:
                print(f"{Colors.BOLD}{log_name}{Colors.NC}")
                printed_header = True

            label = category_labels.get(cat, cat)
            print(f"  {Colors.CYAN}{label}:{Colors.NC}")
            for line_num, content in matches[:3]:  # Show up to 3 matches per category
                print(f"    L{line_num}: {content[:120]}{'...' if len(content) > 120 else ''}")

        if printed_header:
            print()


def print_host_summary(analyses: list[LogAnalysis]):
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


def print_recommendations(analyses: list[LogAnalysis]):
    """Print actionable recommendations based on analysis."""
    cats = defaultdict(int)
    for a in analyses:
        for c in a.categories:
            cats[c] += 1
    total = len(analyses)
    if not total:
        return

    print("=" * 50 + "\nRecommendations\n" + "=" * 50 + "\n")
    recs = []

    # Timeout issues
    if cats["timeout"]:
        recs.append(
            f"- {Colors.YELLOW}Timeout issues:{Colors.NC} Power cycle the cluster. See {ts_ref('tensix_stall')}"
        )

    # Missing connections
    if cats["missing_ports"] or cats["missing_channels"]:
        port_count, chan_count = cats["missing_ports"], cats["missing_channels"]
        if port_count and chan_count:
            recs.append(
                f"- {Colors.BLUE}Missing connections:{Colors.NC} Port ({port_count} logs) + "
                f"Channel ({chan_count} logs). Check cables. See {ts_ref('missing_connections')}"
            )
        else:
            recs.append(
                f"- {Colors.BLUE}Missing connections:{Colors.NC} Check cable seating. "
                f"Verify correct FSD. See {ts_ref('missing_connections')}"
            )

    # PCIe errors
    if cats["pcie_error"]:
        recs.append(
            f"- {Colors.RED}PCIe errors:{Colors.NC} Hardware issue. Machine may have rebooted. "
            "Check dmesg/syslog. May need power cycle."
        )

    # ARC timeout
    if cats["arc_timeout"]:
        recs.append(f"- {Colors.YELLOW}ARC timeout:{Colors.NC} Try tt-smi reset. If persists, power cycle needed.")

    # Link retrains
    if cats["link_retrain"]:
        recs.append(f"- {Colors.YELLOW}Link retrains:{Colors.NC} Some links unstable. Check cables on affected trays.")

    # Workload timeout
    if cats["workload_timeout"]:
        recs.append(
            f"- {Colors.RED}Workload timeout:{Colors.NC} Cluster hung during traffic test. "
            "Power cycle required. Check for GDDR/hardware issues."
        )

    # Device startup error
    if cats["device_error"]:
        recs.append(
            f"- {Colors.RED}Device startup error:{Colors.NC} Failed to initialize devices. "
            "Try tt-smi reset. If persists, power cycle."
        )

    # Unrecoverable/validation failure
    if cats["unrecoverable"] or cats["validation_failed"]:
        recs.append(
            f"- {Colors.RED}Unrecoverable/validation failure:{Colors.NC} "
            "Cluster in bad state. Full power cycle required."
        )

    # Discovery failed
    if cats["discovery_failed"]:
        recs.append(
            f"- {Colors.RED}Discovery failed:{Colors.NC} Physical discovery found no chips. "
            "Check PCIe connections, tt-smi status. May need reboot."
        )

    # CRC/uncorrected codeword errors
    if cats["crc_error"] or cats["uncorrected_cw"]:
        recs.append(
            f"- {Colors.YELLOW}CRC/codeword errors:{Colors.NC} Link quality issues. "
            "Check cable seating. For Wormhole: may indicate bad cable."
        )

    # DRAM failures
    if cats["dram_failure"]:
        recs.append(
            f"- {Colors.RED}DRAM failures:{Colors.NC} Hardware issue. Contact syseng. See {ts_ref('gddr_issue')}"
        )

    # Unhealthy links - check for repeat offenders
    if cats["unhealthy"]:
        link_stats, _ = aggregate_stats(analyses)
        repeats = [(k, v) for k, v in link_stats.items() if v["count"] >= 3]
        if repeats:
            recs.append(
                f"- {Colors.RED}Repeated link failures:{Colors.NC} {len(repeats)} channel(s) failing 3+ times. "
                "Likely bad cable. Check histogram."
            )
        else:
            recs.append(f"- {Colors.YELLOW}Scattered link failures:{Colors.NC} Try power cycle. Check cables.")

    # Firmware mismatch
    if cats["fw_mismatch"]:
        recs.append(
            f"- {Colors.MAGENTA}Firmware mismatch:{Colors.NC} UMD may ignore some links. See {ts_ref('fw_mismatch')}"
        )

    # Stack trace/crash
    if cats["stack_trace"]:
        recs.append(
            f"- {Colors.RED}Stack trace/crash:{Colors.NC} Review full log for root cause. "
            "May indicate software bug or hardware issue."
        )

    # MPI/SSH errors
    if cats["mpi_error"]:
        recs.append(
            f"- {Colors.RED}MPI error:{Colors.NC} Lost connection between hosts. "
            f"Check SSH agent and network. See {ts_ref('ssh_agent')}"
        )
    if cats["ssh_error"]:
        recs.append(
            f"- {Colors.YELLOW}SSH errors:{Colors.NC} Authentication failed. "
            f"Ensure ssh-agent running and keys added. See {ts_ref('ssh_agent')}"
        )

    # Overall health assessment
    rate = cats["healthy"] / total * 100
    if rate >= 100:
        recs.append(f"- {Colors.GREEN}Cluster is healthy.{Colors.NC} Ready for workloads.")
    elif rate >= 80:
        recs.append(f"- {Colors.YELLOW}Cluster looks stable ({rate:.0f}%).{Colors.NC} Ready for workloads.")
    elif rate > 0:
        recs.append(
            f"- {Colors.RED}Low success rate ({rate:.0f}%).{Colors.NC} Investigate failure patterns before proceeding."
        )

    for r in recs:
        print(r)
    print()


def print_timeline(analyses: list[LogAnalysis]):
    """Print visual iteration timeline."""
    if not analyses:
        return
    print("=" * 50 + "\nIteration Timeline\n" + "=" * 50 + "\n")

    sorted_a = sorted(analyses, key=lambda a: a.metadata.iteration)
    icons = []
    for a in sorted_a:
        if "healthy" in a.categories:
            icons.append(("✓", Colors.GREEN))
        elif "unhealthy" in a.categories or "workload_timeout" in a.categories:
            icons.append(("✗", Colors.RED))
        elif "missing_ports" in a.categories or "missing_channels" in a.categories:
            icons.append(("○", Colors.BLUE))
        elif "dram_failure" in a.categories or "pcie_error" in a.categories:
            icons.append(("!", Colors.RED))
        elif "indeterminate" in a.categories:
            icons.append(("?", Colors.CYAN))
        else:
            icons.append(("~", Colors.YELLOW))

    for start in range(0, len(icons), 25):
        row = icons[start : start + 25]
        print("  " + " ".join(f"{start + i + 1:2d}" for i in range(len(row))))
        print("  " + " ".join(f"{c}{i}{Colors.NC} " for i, c in row) + "\n")

    print(
        f"Legend: {Colors.GREEN}✓{Colors.NC}=ok {Colors.RED}✗{Colors.NC}=fail "
        f"{Colors.BLUE}○{Colors.NC}=missing {Colors.RED}!{Colors.NC}=hw {Colors.CYAN}?{Colors.NC}=unknown\n"
    )


def print_errors(analyses: list[LogAnalysis]):
    """Print unique error messages."""
    error_re = re.compile(r"(TT_THROW|TT_FATAL|Error:|RuntimeError)", re.IGNORECASE)
    errors = defaultdict(list)

    for a in analyses:
        seen = set()
        for line in a.content.split("\n"):
            if "<stderr>:" in line and ("0x" in line or "usually indicates" in line):
                continue
            c = clean_line(line)
            if error_re.search(c) and "Found Unhealthy" not in c:
                # Normalize
                msg = re.sub(r"^\d{4}-\d{2}-\d{2}.*?\|\s*", "", c)
                msg = re.sub(r"\s*\([^)]+\.(cpp|hpp):\d+\)\s*$", "", msg)
                msg = re.sub(r"^\[\d+,\d+\]<\w+>:\s*", "", msg).strip()
                if len(msg) > 15 and msg not in seen:
                    seen.add(msg)
                    errors[msg].append(os.path.basename(a.filepath))

    if not errors:
        return
    print("=" * 50 + "\nUnique Error Messages\n" + "=" * 50 + "\n")
    for msg, files in sorted(errors.items(), key=lambda x: -len(x[1]))[:15]:
        display = msg[:100] + "..." if len(msg) > 100 else msg
        print(f"{Colors.RED}[{len(files)}x]{Colors.NC} {display}")
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
        if args.all:
            print_details(analyses)
        if args.histogram:
            print_link_histogram(analyses)
        if args.hosts or args.all:
            print_host_summary(analyses)
        if args.all:
            print_recommendations(analyses)
        if args.timeline or args.all:
            print_timeline(analyses)
        if args.errors or args.all:
            print_errors(analyses)
        if args.verbose:
            print_verbose(analyses)


if __name__ == "__main__":
    main()
