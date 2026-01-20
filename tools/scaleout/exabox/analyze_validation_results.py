#!/usr/bin/env python3
"""
Analyze validation output logs for health status and errors.

Usage:
    ./analyze_validation_results.py [directory]
    ./analyze_validation_results.py validation_output/
    ./analyze_validation_results.py validation_output/ --json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Get script directory for relative file references
SCRIPT_DIR = Path(__file__).parent.resolve()
TROUBLESHOOTING_FILE = SCRIPT_DIR / "TROUBLESHOOTING.md"


# ANSI color codes
class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color

    @classmethod
    def disable(cls):
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = ""
        cls.CYAN = cls.MAGENTA = cls.BOLD = cls.NC = ""


# Section line numbers in TROUBLESHOOTING.md (update if file changes)
TROUBLESHOOTING_SECTIONS = {
    "tensix_stall": 288,
    "gddr_issue": 304,
    "fw_mismatch": 347,
    "missing_connections": 391,  # QSFP Connections Missing Between Hosts
    "data_mismatch": 501,  # Data Mismatch During Traffic Tests
    "ssh_agent": 126,
    "permission_denied": 148,
}


def troubleshooting_link(section: str, label: str) -> str:
    """Generate a clickable file:line reference for terminal."""
    if not TROUBLESHOOTING_FILE.exists():
        return f"TROUBLESHOOTING.md '{label}'"

    line = TROUBLESHOOTING_SECTIONS.get(section, 1)
    # Format: path:line - clickable in VS Code/Cursor terminals
    return f"{TROUBLESHOOTING_FILE}:{line}"


@dataclass
class FaultyLink:
    """Represents a single faulty link from the FAULTY LINKS REPORT."""

    host: str
    tray: int
    asic: int
    channel: int
    port_id: int
    port_type: str
    unique_id: str
    retrains: int
    crc_errors: int
    corrected_cw: int
    uncorrected_cw: int
    mismatch_words: int
    failure_type: str
    pkt_size: str
    data_size: str


@dataclass
class LogMetadata:
    """Metadata extracted from log file."""

    iteration: int = 0
    timestamp: str = ""
    detected_hosts: list = field(default_factory=list)
    chips_found: int = 0
    num_iterations: int = 0
    packet_size_bytes: int = 0
    data_size_bytes: int = 0
    discovery_complete: bool = False
    validation_complete: bool = False


@dataclass
class LogAnalysis:
    """Analysis result for a single log file."""

    filepath: str
    categories: list = field(default_factory=list)
    faulty_links: list = field(default_factory=list)
    missing_connections: list = field(default_factory=list)
    extra_connections: list = field(default_factory=list)
    error_messages: list = field(default_factory=list)
    matched_lines: dict = field(default_factory=dict)  # category -> list of matching lines
    metadata: LogMetadata = field(default_factory=LogMetadata)


# Pattern definitions - matching C++ validation tool output
PATTERNS = {
    # Success indicators
    "healthy": re.compile(r"All Detected Links are healthy"),
    # Failure indicators
    "unhealthy": re.compile(r"Found Unhealthy Links|FAULTY LINKS REPORT"),
    "timeout": re.compile(r"Timeout \(\d+ ms\) waiting for physical cores to finish"),
    "workload_timeout": re.compile(
        r"Workload execution timed out|" r"cluster is not in a healthy state|" r"ERROR:.*timed out after \d+ seconds"
    ),
    # Missing connection patterns (both formats from C++ tool)
    "missing_connections": re.compile(
        r"Physical Discovery found (\d+) missing port/cable connections|"
        r"Port Connections found in FSD but missing in GSD"
    ),
    "missing_channels": re.compile(
        r"Physical Discovery found (\d+) missing channel connections|"
        r"Channel Connections found in FSD but missing in GSD"
    ),
    "extra_connections": re.compile(
        r"Physical Discovery found (\d+) extra port/cable connections|"
        r"extra port/cable connections|"
        r"Connections found in GSD but not in FSD"
    ),
    # Hardware issues
    "dram_failure": re.compile(r"DRAM training failed|gddr issue", re.IGNORECASE),
    "pcie_error": re.compile(r"PCIe error|AER:.*aer_status|\[Hardware Error\]"),
    "arc_timeout": re.compile(r"ARC.*[Tt]imeout|timeout.*ARC|ARC message timed out"),
    "device_startup_error": re.compile(r"Error starting devices|Error details:"),
    # Firmware and compatibility
    "fw_mismatch": re.compile(
        r"Firmware bundle version .* is newer than the latest fully tested version|"
        r"FW Bundle version mismatch|"
        r"ERISC FW version.*mismatch"
    ),
    # Link health indicators (from C++ link_unhealthy logic)
    "link_retrain": re.compile(r"Retrain.*detected|retrain_count.*increasing|Link retrains detected"),
    "crc_error": re.compile(r"CRC Error|crc_error_count > 0"),
    "uncorrected_cw": re.compile(r"Uncorrected CW|uncorrected_codeword"),
    "data_mismatch": re.compile(r"Data Mismatch|mismatched_words|num_mismatched"),
    # Discovery and validation status
    "discovery_failed": re.compile(r"Physical Discovery.*failed|Discovery Complete.*0 chips"),
    "unrecoverable_state": re.compile(r"Encountered unrecoverable state|unrecoverable.*state"),
    "validation_failed": re.compile(r"Cluster validation failed|validation.*failed", re.IGNORECASE),
    # Stack trace/crash indicators
    "stack_trace": re.compile(r"<stderr>:.*\[\s*\d+\].*0x[a-fA-F0-9]+|TT_FATAL|TT_THROW|std::runtime_error"),
    # MPI/communication errors
    "mpi_error": re.compile(r"PRTE has lost communication|MPI_ABORT|mpi.*error", re.IGNORECASE),
    "ssh_error": re.compile(r"Permission denied \(publickey\)|ssh.*connection.*refused", re.IGNORECASE),
    # Truncated/incomplete runs (started but didn't finish)
    "truncated": re.compile(r"Sending traffic across detected links"),  # If present without result = truncated
}


def parse_faulty_link_line(line: str) -> Optional[FaultyLink]:
    """Parse a single faulty link line using flexible token-based parsing."""
    # Remove MPI stdout prefix if present
    if "<stdout>:" in line:
        line = line.split("<stdout>:")[-1]

    line = line.strip()
    if not line or line.startswith("-"):
        return None

    # Check if line starts with a hostname pattern (e.g., bh-glx-c02u08)
    if not re.match(r"^[a-zA-Z][\w\-]+", line):
        return None

    # Split and extract fixed-position fields
    parts = line.split()
    if len(parts) < 12:
        return None

    try:
        host = parts[0]
        tray = int(parts[1])
        asic = int(parts[2])
        channel = int(parts[3])
        port_id = int(parts[4])

        # Handle case where port_type and unique_id are jammed together (no space)
        # e.g., "LINKING_BOARD_10xbc48eb85986cf88c8" should be "LINKING_BOARD_1" + "0xbc48eb85986cf88c8"
        port_type_raw = parts[5]
        if "0x" in port_type_raw and not port_type_raw.startswith("0x"):
            idx = port_type_raw.find("0x")
            port_type = port_type_raw[:idx]
            unique_id = port_type_raw[idx:]
            # Shift remaining parts since we consumed one less token
            parts = parts[:6] + [unique_id] + parts[6:]
        else:
            port_type = port_type_raw
            unique_id = parts[6]
        retrains = int(parts[7], 16) if parts[7].startswith("0x") else int(parts[7])
        crc_errors = int(parts[8], 16) if parts[8].startswith("0x") else int(parts[8])

        # Remaining fields vary - find where failure type starts (contains letters after numbers)
        # Look for data size at end (pattern: number followed by B)
        data_size = ""
        pkt_size = ""

        # Find "B" markers from end
        for i in range(len(parts) - 1, 8, -1):
            if parts[i] == "B":
                if not data_size:
                    data_size = parts[i - 1] + " B"
                elif not pkt_size:
                    pkt_size = parts[i - 1] + " B"
                    break

        # Find numerical fields between port info and failure type
        # Fields 9-11ish are: Corrected CW, Uncorrected CW, Mismatch Words
        corrected_cw = 0
        uncorrected_cw = 0
        mismatch_words = 0
        failure_type = ""

        # Parse corrected CW (could be hex or decimal)
        if len(parts) > 9:
            val = parts[9]
            if val.startswith("0x"):
                corrected_cw = int(val, 16)
            elif val.isdigit():
                corrected_cw = int(val)

        # Parse uncorrected CW
        if len(parts) > 10:
            val = parts[10]
            if val.isdigit():
                uncorrected_cw = int(val)
            elif val.startswith("0x"):
                uncorrected_cw = int(val, 16)

        # Determine where failure type starts
        # It starts after numeric fields (look for non-numeric, non-hex value)
        failure_start = 11
        for i in range(11, min(14, len(parts))):
            val = parts[i]
            if val.isdigit():
                if i == 11:
                    mismatch_words = int(val)
                failure_start = i + 1
            elif val.startswith("0x"):
                failure_start = i + 1
            else:
                # Found start of failure type
                failure_start = i
                break
        failure_end = len(parts)

        # Find the position of last "B" to exclude size fields
        for i in range(len(parts) - 1, 0, -1):
            if parts[i] == "B":
                failure_end = i - 1
                # Check for second B (pkt size)
                for j in range(i - 2, 0, -1):
                    if parts[j] == "B":
                        failure_end = j - 1
                        break
                break

        # Handle case where failure type is jammed with pkt size (e.g., "Mismatch64")
        failure_parts = parts[failure_start : failure_end + 1] if failure_start <= failure_end else []
        if failure_parts:
            # Check last part for number suffix (pkt size jammed in)
            last = failure_parts[-1] if failure_parts else ""
            match = re.match(r"^(.+?)(\d+)$", last)
            if match and not last.startswith("0x"):
                failure_parts[-1] = match.group(1)
                if not pkt_size:
                    pkt_size = match.group(2) + " B"

        failure_type = " ".join(failure_parts).strip()

        return FaultyLink(
            host=host,
            tray=tray,
            asic=asic,
            channel=channel,
            port_id=port_id,
            port_type=port_type,
            unique_id=unique_id,
            retrains=retrains,
            crc_errors=crc_errors,
            corrected_cw=corrected_cw,
            uncorrected_cw=uncorrected_cw,
            mismatch_words=mismatch_words,
            failure_type=failure_type if failure_type else "Unknown",
            pkt_size=pkt_size,
            data_size=data_size,
        )
    except (ValueError, IndexError):
        return None


# Pattern for missing/extra port connection endpoints (non-greedy to match both endpoints)
PORT_CONNECTION_PATTERN = re.compile(
    r"PhysicalPortEndpoint\{hostname='([^']+)',[^}]*tray_id=(\d+), port_type=(\w+), port_id=(\d+)\}"
)

# Pattern for missing channel connection endpoints
CHANNEL_CONNECTION_PATTERN = re.compile(
    r"PhysicalChannelEndpoint\{hostname='([^']+)', tray_id=(\d+), asic_channel=AsicChannel\{asic_location=(\d+), channel_id=(\d+)\}\}"
)


def parse_faulty_links_report(content: str) -> list[FaultyLink]:
    """Parse the FAULTY LINKS REPORT section from log content."""
    links = []
    in_report = False
    found_header = False
    empty_after_data = False

    for line in content.split("\n"):
        clean = line.split("<stdout>:")[-1].strip() if "<stdout>:" in line else line.strip()

        # Detect start of report
        if "FAULTY LINKS REPORT" in line:
            in_report = True
            found_header = False
            empty_after_data = False
            continue

        if not in_report:
            continue

        # Skip box drawing characters and "Total Faulty Links"
        if clean.startswith("╚") or clean.startswith("═") or "Total Faulty Links" in clean:
            continue

        # Skip header row
        if "Host" in clean and "Tray" in clean and "ASIC" in clean:
            found_header = True
            continue

        # Skip separator lines
        if clean.startswith("-") and len(clean) > 10:
            continue

        # Empty line handling
        if clean == "":
            if found_header and links:
                # Empty line after we've parsed some data = end of section
                empty_after_data = True
            continue

        # End of report on timestamp line
        if re.match(r"^\d{4}-\d{2}-\d{2}", clean):
            in_report = False
            continue

        # If we've seen empty line after data and now see non-empty, we're done
        if empty_after_data:
            in_report = False
            continue

        # Try to parse data line
        if found_header:
            link = parse_faulty_link_line(line)
            if link:
                links.append(link)

    return links


def parse_missing_connections(content: str) -> list[tuple]:
    """Parse missing connection endpoints from log content (both port and channel types).

    Handles multiple formats:
    - "Physical Discovery found X missing port/cable connections:"
    - "Port Connections found in FSD but missing in GSD (X connections):"
    - "Physical Discovery found X missing channel connections:"
    - "Channel Connections found in FSD but missing in GSD (X connections):"
    """
    connections = []
    in_missing_port = False
    in_missing_channel = False

    for line in content.split("\n"):
        # Remove MPI stdout prefix if present
        clean_line = line.split("<stdout>:")[-1].strip() if "<stdout>:" in line else line.strip()

        # Detect start of missing port sections (multiple formats)
        if "missing port/cable connections" in line or "Port Connections found in FSD but missing in GSD" in line:
            in_missing_port = True
            in_missing_channel = False
            continue

        # Detect start of missing channel sections (multiple formats)
        if "missing channel connections" in line or "Channel Connections found in FSD but missing in GSD" in line:
            in_missing_channel = True
            in_missing_port = False
            continue

        # Parse port connections
        if in_missing_port:
            if "PhysicalPortEndpoint" in clean_line:
                matches = PORT_CONNECTION_PATTERN.findall(line)
                if len(matches) == 2:
                    # Format: (hostname, tray_id, port_type, port_id)
                    connections.append(("port", matches[0], matches[1]))
            elif clean_line and not clean_line.startswith("-") and "PhysicalPortEndpoint" not in clean_line:
                # End of section - but check for section headers
                if "Channel Connections" in clean_line or "missing channel" in clean_line.lower():
                    in_missing_port = False
                    in_missing_channel = True
                elif clean_line and not clean_line.startswith("Total"):
                    in_missing_port = False

        # Parse channel connections
        if in_missing_channel:
            if "PhysicalChannelEndpoint" in clean_line:
                matches = CHANNEL_CONNECTION_PATTERN.findall(line)
                if len(matches) == 2:
                    # Format: (hostname, tray_id, asic_location, channel_id)
                    connections.append(("channel", matches[0], matches[1]))
            elif clean_line and not clean_line.startswith("-") and "PhysicalChannelEndpoint" not in clean_line:
                # End of section - but check for section headers
                if "Port Connections" in clean_line or "missing port" in clean_line.lower():
                    in_missing_channel = False
                    in_missing_port = True
                elif clean_line and not clean_line.startswith("Total"):
                    in_missing_channel = False

    return connections


def parse_log_metadata(content: str, filepath: str) -> LogMetadata:
    """Extract metadata from log file content."""
    metadata = LogMetadata()

    # Extract iteration number from filename
    basename = os.path.basename(filepath)
    iter_match = re.search(r"iteration_(\d+)", basename)
    if iter_match:
        metadata.iteration = int(iter_match.group(1))

    # Parse line by line for metadata
    for line in content.split("\n"):
        clean = line.split("<stdout>:")[-1].strip() if "<stdout>:" in line else line.strip()

        # Iteration header from script
        if "Iteration:" in line:
            match = re.search(r"Iteration:\s*(\d+)", line)
            if match:
                metadata.iteration = int(match.group(1))

        # Timestamp from script header
        if "Timestamp:" in line and not metadata.timestamp:
            match = re.search(r"Timestamp:\s*(.+)", line)
            if match:
                metadata.timestamp = match.group(1).strip()

        # Detected Hosts from validation tool
        if "Detected Hosts:" in line:
            match = re.search(r"Detected Hosts:\s*(.+?)(?:\s*\(|$)", line)
            if match:
                hosts_str = match.group(1).strip()
                # Split by comma, strip whitespace, filter empty and non-hostname entries
                hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]
                # Only keep entries that look like hostnames (contain letters and dashes)
                metadata.detected_hosts = [h for h in hosts if re.match(r"^[a-zA-Z][\w\-]+$", h)]

        # Chips found from tt-smi
        if "chips found" in line:
            match = re.search(r"All (\d+) chips found", line)
            if match:
                metadata.chips_found = int(match.group(1))

        # Traffic parameters
        if "Sending traffic" in line or "Sweeping traffic" in line:
            iter_match = re.search(r"Num Iterations:\s*(\d+)", line)
            if iter_match:
                metadata.num_iterations = int(iter_match.group(1))
            pkt_match = re.search(r"Packet Size.*?:\s*(\d+)", line)
            if pkt_match:
                metadata.packet_size_bytes = int(pkt_match.group(1))
            data_match = re.search(r"Data Size.*?:\s*(\d+)", line)
            if data_match:
                metadata.data_size_bytes = int(data_match.group(1))

        # Discovery status
        if "Physical Discovery Complete" in line:
            metadata.discovery_complete = True

        # Validation status
        if "Validation Complete" in line:
            metadata.validation_complete = True

    return metadata


def analyze_log_file(filepath: str) -> LogAnalysis:
    """Analyze a single log file and return categorized results."""
    result = LogAnalysis(filepath=filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception as e:
        result.error_messages.append(f"Failed to read file: {e}")
        return result

    # Extract metadata first
    result.metadata = parse_log_metadata(content, filepath)

    # Check each pattern and capture matched lines
    for category, pattern in PATTERNS.items():
        matched = []
        for i, line in enumerate(lines):
            if pattern.search(line):
                # Capture the matched line with line number
                clean_line = line.split("<stdout>:")[-1].strip() if "<stdout>:" in line else line.strip()
                matched.append((i + 1, clean_line[:200]))  # line number, truncated content
        if matched:
            result.categories.append(category)
            result.matched_lines[category] = matched

    # Parse faulty links if unhealthy
    if "unhealthy" in result.categories:
        result.faulty_links = parse_faulty_links_report(content)

    # Parse missing connections (check both patterns)
    if "missing_connections" in result.categories or "missing_channels" in result.categories:
        result.missing_connections = parse_missing_connections(content)

    # Check for truncated runs - has traffic start but no result
    has_result = "healthy" in result.categories or "unhealthy" in result.categories
    if "truncated" in result.categories and not has_result:
        # Keep truncated, remove from indeterminate candidates
        pass
    elif "truncated" in result.categories and has_result:
        # Has result, so not truncated - remove the truncated category
        result.categories.remove("truncated")

    # If no meaningful categories matched, mark as indeterminate
    non_truncated_cats = [c for c in result.categories if c != "truncated"]
    if not non_truncated_cats:
        result.categories.append("indeterminate")

    return result


def aggregate_link_stats(analyses: list[LogAnalysis]) -> dict:
    """Aggregate faulty link statistics across all logs."""
    stats = defaultdict(
        lambda: {
            "count": 0,
            "retrains_total": 0,
            "crc_errors_total": 0,
            "mismatch_words_total": 0,
            "failure_types": defaultdict(int),
        }
    )

    for analysis in analyses:
        for link in analysis.faulty_links:
            key = (link.host, link.tray, link.channel, link.port_type)
            stats[key]["count"] += 1
            stats[key]["retrains_total"] += link.retrains
            stats[key]["crc_errors_total"] += link.crc_errors
            stats[key]["mismatch_words_total"] += link.mismatch_words
            stats[key]["failure_types"][link.failure_type] += 1

    return dict(stats)


def aggregate_host_stats(analyses: list[LogAnalysis]) -> dict:
    """Aggregate failure statistics per host."""
    stats = defaultdict(lambda: {"faulty_link_count": 0, "missing_count": 0})

    for analysis in analyses:
        for link in analysis.faulty_links:
            stats[link.host]["faulty_link_count"] += 1

        for conn in analysis.missing_connections:
            # conn format: (type, endpoint1, endpoint2)
            ep1, ep2 = conn[1], conn[2]
            stats[ep1[0]]["missing_count"] += 1
            stats[ep2[0]]["missing_count"] += 1

    return dict(stats)


def print_cluster_info(analyses: list[LogAnalysis]):
    """Print cluster configuration extracted from logs."""
    if not analyses:
        return

    # Find first log with meaningful metadata
    hosts = set()
    chips_found = 0
    traffic_config = None

    for a in analyses:
        if a.metadata.detected_hosts:
            hosts.update(a.metadata.detected_hosts)
        if a.metadata.chips_found > chips_found:
            chips_found = a.metadata.chips_found
        if a.metadata.num_iterations > 0 and not traffic_config:
            traffic_config = a.metadata

    if hosts or chips_found or traffic_config:
        print(f"{Colors.CYAN}Cluster Configuration:{Colors.NC}")
        if hosts:
            print(f"  Hosts: {', '.join(sorted(hosts))}")
        if chips_found:
            print(f"  Chips per host: {chips_found}")
            print(f"  Total chips: {chips_found * len(hosts) if hosts else chips_found}")
        if traffic_config:
            print(
                f"  Traffic config: {traffic_config.num_iterations} iterations, "
                f"{traffic_config.packet_size_bytes}B packets, "
                f"{traffic_config.data_size_bytes}B data"
            )
        print()


def print_summary(analyses: list[LogAnalysis], show_files: bool = True):
    """Print summary of all analyses."""
    total = len(analyses)

    # Count by category
    category_counts = defaultdict(list)
    for a in analyses:
        for cat in a.categories:
            category_counts[cat].append(a.filepath)

    print("=" * 50)
    print("Validation Results Analysis")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print()

    # Print cluster info first
    print_cluster_info(analyses)

    print(f"Total log files analyzed: {total}")
    if total < 50:
        print(f"{Colors.YELLOW}Warning: Expected 50 iterations, found {total}{Colors.NC}")
    print()

    # Category breakdown - grouped by severity
    category_display = [
        # Success
        ("healthy", Colors.GREEN, "Healthy links"),
        # Critical failures
        ("unhealthy", Colors.RED, "Unhealthy links"),
        ("unrecoverable_state", Colors.RED, "Unrecoverable state"),
        ("validation_failed", Colors.RED, "Validation failed"),
        ("dram_failure", Colors.RED, "DRAM/GDDR failures"),
        ("pcie_error", Colors.RED, "PCIe errors"),
        ("device_startup_error", Colors.RED, "Device startup errors"),
        # Timeout issues
        ("timeout", Colors.YELLOW, "Physical core timeout"),
        ("workload_timeout", Colors.YELLOW, "Workload timeout"),
        ("arc_timeout", Colors.YELLOW, "ARC timeout"),
        # Connectivity issues
        ("missing_connections", Colors.BLUE, "Missing port connections"),
        ("missing_channels", Colors.BLUE, "Missing channel connections"),
        ("extra_connections", Colors.YELLOW, "Extra connections"),
        ("discovery_failed", Colors.RED, "Discovery failed"),
        # Link health
        ("link_retrain", Colors.YELLOW, "Link retrains"),
        ("crc_error", Colors.YELLOW, "CRC errors"),
        ("uncorrected_cw", Colors.YELLOW, "Uncorrected codewords"),
        ("data_mismatch", Colors.RED, "Data mismatch"),
        # Warnings
        ("fw_mismatch", Colors.MAGENTA, "Firmware mismatch"),
        # Communication/infrastructure
        ("stack_trace", Colors.RED, "Stack trace/crash"),
        ("mpi_error", Colors.RED, "MPI communication error"),
        ("ssh_error", Colors.YELLOW, "SSH connection error"),
        # Incomplete
        ("truncated", Colors.YELLOW, "Truncated (started, no result)"),
        # Unknown
        ("indeterminate", Colors.CYAN, "Indeterminate/empty"),
    ]

    for cat_key, color, label in category_display:
        count = len(category_counts.get(cat_key, []))
        print(f"{color}{label}:{Colors.NC} {count} / {total}")
        if show_files and count > 0 and count <= 10:
            for f in category_counts[cat_key]:
                print(f"    {os.path.basename(f)}")

    print()

    # Success rate
    healthy_count = len(category_counts.get("healthy", []))
    if total > 0:
        success_rate = (healthy_count / total) * 100
        color = Colors.GREEN if success_rate >= 90 else Colors.YELLOW if success_rate >= 70 else Colors.RED
        print(f"{Colors.BOLD}Success Rate:{Colors.NC} {color}{success_rate:.1f}%{Colors.NC}")
    print()


def print_link_histogram(analyses: list[LogAnalysis]):
    """Print histogram of failing links."""
    link_stats = aggregate_link_stats(analyses)

    if not link_stats:
        return

    print("=" * 50)
    print("Faulty Link Histogram (by frequency)")
    print("=" * 50)
    print()

    # Sort by failure count descending
    sorted_links = sorted(link_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    top_links = sorted_links[:20]

    # Calculate dynamic column widths
    host_w = max(len("Host"), max((len(k[0]) for k, _ in top_links), default=4))
    type_w = max(len("Type"), max((len(k[3]) for k, _ in top_links), default=4))

    # Print header
    header = (
        f"{'Host':<{host_w}}  {'Tray':>4}  {'Ch':>2}  {'Type':<{type_w}}  {'Fails':>5}  {'Retrains':>8}  {'CRC Err':>8}"
    )
    print(header)
    print("-" * len(header))

    # Print data rows
    for (host, tray, channel, port_type), stats in top_links:
        print(
            f"{host:<{host_w}}  {tray:>4}  {channel:>2}  {port_type:<{type_w}}  "
            f"{stats['count']:>5}  {stats['retrains_total']:>8}  {stats['crc_errors_total']:>8}"
        )

    if len(sorted_links) > 20:
        print(f"... and {len(sorted_links) - 20} more")
    print()


def print_faulty_links_detail(analyses: list[LogAnalysis]):
    """Print detailed faulty links info from all unhealthy logs."""
    all_links = []
    for a in analyses:
        for link in a.faulty_links:
            all_links.append((os.path.basename(a.filepath), link))

    if not all_links:
        return

    print("=" * 50)
    print("Faulty Links Detail")
    print("=" * 50)
    print()

    # Calculate column widths
    host_w = max(len("Host"), max(len(link.host) for _, link in all_links))
    type_w = max(len("Type"), max(len(link.port_type) for _, link in all_links))

    # Header
    print(
        f"{'Log':<25}  {'Host':<{host_w}}  {'Tray':>4}  {'ASIC':>4}  {'Ch':>2}  "
        f"{'Type':<{type_w}}  {'Port':>4}  {'Retrains':>8}  {'CRC':>8}  {'Uncorr':>6}  {'Mismatch':>8}"
    )
    print("-" * 120)

    for log_name, link in all_links:
        short_log = log_name.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
        print(
            f"{short_log:<25}  {link.host:<{host_w}}  {link.tray:>4}  {link.asic:>4}  {link.channel:>2}  "
            f"{link.port_type:<{type_w}}  {link.port_id:>4}  {link.retrains:>8}  {link.crc_errors:>8}  "
            f"{link.uncorrected_cw:>6}  {link.mismatch_words:>8}"
        )

    print()

    # Also show failure type breakdown with recommendations
    failure_types = defaultdict(int)
    for _, link in all_links:
        failure_types[link.failure_type] += 1

    if failure_types:
        print("Failure Type Breakdown:")
        for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
            print(f"  {count:>3}x  {ftype}")

        # Add recommendation based on failure types
        print()
        if any("Mismatch" in ft or "mismatch" in ft for ft in failure_types):
            link = troubleshooting_link("data_mismatch", "Data Mismatch During Traffic Tests")
            print(
                f"  {Colors.YELLOW}Tip:{Colors.NC} Same link failing repeatedly = bad cable. Scattered = try power cycle."
            )
            print(f"       See {link}")
        print()


def print_missing_connections_detail(analyses: list[LogAnalysis]):
    """Print detailed missing connections info."""
    all_missing = []
    for a in analyses:
        for conn in a.missing_connections:
            all_missing.append((os.path.basename(a.filepath), conn))

    if not all_missing:
        return

    print("=" * 50)
    print("Missing Connections Detail")
    print("=" * 50)
    print()

    for log_name, conn_data in all_missing:
        short_log = log_name.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
        conn_type = conn_data[0]
        endpoint1 = conn_data[1]
        endpoint2 = conn_data[2]

        print(f"  {short_log} ({conn_type}):")
        if conn_type == "port":
            # endpoint format: (hostname, tray_id, port_type, port_id)
            host1, tray1, ptype1, pid1 = endpoint1
            host2, tray2, ptype2, pid2 = endpoint2
            print(f"    {host1} tray {tray1} {ptype1} port {pid1}")
            print(f"    <--> {host2} tray {tray2} {ptype2} port {pid2}")
        else:  # channel
            # endpoint format: (hostname, tray_id, asic_location, channel_id)
            host1, tray1, asic1, ch1 = endpoint1
            host2, tray2, asic2, ch2 = endpoint2
            print(f"    {host1} tray {tray1} ASIC {asic1} channel {ch1}")
            print(f"    <--> {host2} tray {tray2} ASIC {asic2} channel {ch2}")
        print()

    # Summary by host pair
    host_pairs = defaultdict(int)
    for _, conn_data in all_missing:
        ep1, ep2 = conn_data[1], conn_data[2]
        pair = tuple(sorted([ep1[0], ep2[0]]))
        host_pairs[pair] += 1

    if len(host_pairs) > 1 or len(all_missing) > 2:
        print("Missing connections by host pair:")
        for (h1, h2), count in sorted(host_pairs.items(), key=lambda x: -x[1]):
            if h1 == h2:
                print(f"  {count}x  {h1} (internal)")
            else:
                print(f"  {count}x  {h1} <-> {h2}")
        print()


def print_host_summary(analyses: list[LogAnalysis]):
    """Print per-host failure summary."""
    host_stats = aggregate_host_stats(analyses)

    if not host_stats:
        return

    print("=" * 50)
    print("Per-Host Failure Summary")
    print("=" * 50)
    print()

    sorted_hosts = sorted(host_stats.items(), key=lambda x: x[1]["faulty_link_count"], reverse=True)

    print(f"{'Host':<30} {'Faulty Links':>12} {'Missing Conn':>12}")
    print("-" * 56)

    for host, stats in sorted_hosts:
        print(f"{host:<30} {stats['faulty_link_count']:>12} {stats['missing_count']:>12}")
    print()


def print_recommendations(analyses: list[LogAnalysis]):
    """Print actionable recommendations based on analysis."""
    category_counts = defaultdict(int)
    for a in analyses:
        for cat in a.categories:
            category_counts[cat] += 1

    total = len(analyses)
    if total == 0:
        return

    print("=" * 50)
    print("Recommendations")
    print("=" * 50)
    print()

    recommendations = []

    if category_counts["timeout"] > 0:
        link = troubleshooting_link("tensix_stall", "Tensix Stall Issue")
        recommendations.append(
            f"- {Colors.YELLOW}Timeout issues detected:{Colors.NC} Power cycle the cluster. " f"See {link}"
        )

    if category_counts["missing_connections"] > 0 or category_counts["missing_channels"] > 0:
        link = troubleshooting_link("missing_connections", "Missing Connections")
        port_count = category_counts["missing_connections"]
        channel_count = category_counts["missing_channels"]
        if port_count and channel_count:
            recommendations.append(
                f"- {Colors.BLUE}Missing connections:{Colors.NC} Port ({port_count} logs) + Channel ({channel_count} logs). "
                f"Check cable seating. See {link}"
            )
        else:
            recommendations.append(
                f"- {Colors.BLUE}Missing connections:{Colors.NC} Check cable seating. "
                f"Verify correct FSD file. See {link}"
            )

    if category_counts["pcie_error"] > 0:
        recommendations.append(
            f"- {Colors.RED}PCIe errors detected:{Colors.NC} Hardware issue. Machine may have rebooted. "
            "Check dmesg/syslog for details. May need power cycle."
        )

    if category_counts["arc_timeout"] > 0:
        recommendations.append(
            f"- {Colors.YELLOW}ARC timeout:{Colors.NC} Try tt-smi reset. If persists, power cycle may be needed."
        )

    if category_counts["link_retrain"] > 0:
        recommendations.append(
            f"- {Colors.YELLOW}Link retrains:{Colors.NC} Some links are unstable. "
            "Check cables and connections on affected trays."
        )

    if category_counts["workload_timeout"] > 0:
        recommendations.append(
            f"- {Colors.RED}Workload timeout:{Colors.NC} Cluster hung during traffic test. "
            "Power cycle required. Check for GDDR/hardware issues."
        )

    if category_counts["device_startup_error"] > 0:
        recommendations.append(
            f"- {Colors.RED}Device startup error:{Colors.NC} Failed to initialize devices. "
            "Try tt-smi reset. If persists, power cycle and check tt-smi status."
        )

    if category_counts["unrecoverable_state"] > 0 or category_counts["validation_failed"] > 0:
        recommendations.append(
            f"- {Colors.RED}Unrecoverable/validation failure:{Colors.NC} "
            "Cluster is in bad state. Full power cycle required. Contact syseng if issue persists."
        )

    if category_counts["discovery_failed"] > 0:
        recommendations.append(
            f"- {Colors.RED}Discovery failed:{Colors.NC} Physical discovery found no chips. "
            "Check PCIe connections, tt-smi status. May need reboot."
        )

    if category_counts["data_mismatch"] > 0 and category_counts["unhealthy"] == 0:
        # Data mismatch without being flagged as unhealthy - might be transient
        link = troubleshooting_link("data_mismatch", "Data Mismatch During Traffic Tests")
        recommendations.append(f"- {Colors.YELLOW}Data mismatch detected:{Colors.NC} See {link}")

    if category_counts["crc_error"] > 0 or category_counts["uncorrected_cw"] > 0:
        recommendations.append(
            f"- {Colors.YELLOW}CRC/codeword errors:{Colors.NC} Link quality issues. "
            "Check cable seating. For Wormhole: may indicate bad cable."
        )

    if category_counts["dram_failure"] > 0:
        link = troubleshooting_link("gddr_issue", "GDDR Issue")
        recommendations.append(
            f"- {Colors.RED}DRAM failures:{Colors.NC} Hardware issue. " f"Contact syseng. See {link}"
        )

    if category_counts["unhealthy"] > 0:
        # Check if same links fail repeatedly
        link_stats = aggregate_link_stats(analyses)
        repeat_offenders = [(k, v) for k, v in link_stats.items() if v["count"] >= 3]

        if repeat_offenders:
            recommendations.append(
                f"- {Colors.RED}Repeated link failures:{Colors.NC} Same channel failing multiple times. "
                "Likely bad cable. Check histogram above."
            )
        else:
            recommendations.append(
                f"- {Colors.YELLOW}Scattered link failures:{Colors.NC} " "Try power cycle. If persists, check cables."
            )

    if category_counts["fw_mismatch"] > 0:
        link = troubleshooting_link("fw_mismatch", "UMD Firmware Version Mismatch")
        recommendations.append(
            f"- {Colors.MAGENTA}Firmware mismatch:{Colors.NC} " f"UMD may ignore some links. See {link}"
        )

    if category_counts["stack_trace"] > 0:
        recommendations.append(
            f"- {Colors.RED}Stack trace/crash detected:{Colors.NC} "
            "Review full log for root cause. May indicate software bug or hardware issue."
        )

    if category_counts["mpi_error"] > 0:
        link = troubleshooting_link("ssh_agent", "SSH Agent Forwarding")
        recommendations.append(
            f"- {Colors.RED}MPI communication error:{Colors.NC} Lost connection between hosts. "
            f"Check SSH agent forwarding and network. See {link}"
        )

    if category_counts["ssh_error"] > 0:
        link = troubleshooting_link("ssh_agent", "SSH Agent Forwarding")
        recommendations.append(
            f"- {Colors.YELLOW}SSH errors:{Colors.NC} Authentication failed. "
            f"Ensure ssh-agent running and keys added. See {link}"
        )

    healthy_rate = category_counts["healthy"] / total * 100
    if healthy_rate >= 100:
        recommendations.append(f"- {Colors.GREEN}Cluster is healthy.{Colors.NC} Ready for workloads.")
    elif healthy_rate >= 80:
        recommendations.append(f"- {Colors.YELLOW}Cluster looks stable.{Colors.NC} Ready for workloads.")
    else:
        recommendations.append(
            f"- {Colors.RED}Low success rate ({healthy_rate:.0f}%).{Colors.NC} "
            "Investigate failure patterns before proceeding."
        )

    for rec in recommendations:
        print(rec)
    print()


def print_verbose(analyses: list[LogAnalysis]):
    """Print matched log lines for each detected issue."""
    print("=" * 50)
    print("Matched Log Lines (Evidence)")
    print("=" * 50)
    print()

    # Group by category
    category_labels = {
        "healthy": "Healthy",
        "unhealthy": "Unhealthy Links",
        "unrecoverable_state": "Unrecoverable State",
        "validation_failed": "Validation Failed",
        "timeout": "Physical Core Timeout",
        "workload_timeout": "Workload Timeout",
        "missing_connections": "Missing Port Connections",
        "missing_channels": "Missing Channel Connections",
        "extra_connections": "Extra Connections",
        "discovery_failed": "Discovery Failed",
        "dram_failure": "DRAM/GDDR Failures",
        "pcie_error": "PCIe Errors",
        "arc_timeout": "ARC Timeout",
        "device_startup_error": "Device Startup Error",
        "fw_mismatch": "Firmware Mismatch",
        "link_retrain": "Link Retrains",
        "crc_error": "CRC Errors",
        "uncorrected_cw": "Uncorrected Codewords",
        "data_mismatch": "Data Mismatch",
        "stack_trace": "Stack Trace/Crash",
        "mpi_error": "MPI Communication Error",
        "ssh_error": "SSH Connection Error",
    }

    for category, label in category_labels.items():
        matches_found = False
        for a in analyses:
            if category in a.matched_lines:
                if not matches_found:
                    print(f"{Colors.BOLD}--- {label} ---{Colors.NC}")
                    matches_found = True

                filename = os.path.basename(a.filepath)
                for line_num, content in a.matched_lines[category][:3]:  # Show max 3 per file
                    # Make the path clickable: filepath:line
                    print(f"  {a.filepath}:{line_num}")
                    print(f"    {content}")

                # For unhealthy links, also show the faulty link details
                if category == "unhealthy" and a.faulty_links:
                    print(f"  {Colors.RED}Faulty Links:{Colors.NC}")
                    for link in a.faulty_links:
                        print(
                            f"    {link.host} | Tray {link.tray} ASIC {link.asic} Ch {link.channel} | "
                            f"{link.port_type} Port {link.port_id}"
                        )
                        print(
                            f"      Retrains: {link.retrains}, CRC Err: {link.crc_errors}, "
                            f"Uncorrected CW: {link.uncorrected_cw}, Mismatch: {link.mismatch_words}"
                        )
                        print(f"      Failure: {link.failure_type}")

        if matches_found:
            print()


def extract_unique_errors(analyses: list[LogAnalysis]) -> dict:
    """Extract unique error messages from logs."""
    error_pattern = re.compile(r"(TT_THROW|TT_FATAL|Error:|failed|exception|RuntimeError)", re.IGNORECASE)
    unique_errors = defaultdict(list)  # message -> list of files

    for a in analyses:
        try:
            with open(a.filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

        file_errors = set()  # Track unique errors per file
        for line in content.split("\n"):
            # Skip stderr stack traces and MPI noise
            if "<stderr>:" in line:
                if "0x" in line and "[" in line:  # Stack traces
                    continue
                if "usually indicates" in line or "failure of the peer" in line:
                    continue

            clean = line.split("<stdout>:")[-1].strip() if "<stdout>:" in line else line.strip()
            clean = clean.split("<stderr>:")[-1].strip() if "<stderr>:" in line else clean

            if error_pattern.search(clean):
                # Skip generic unhealthy message (we already track this)
                if "Found Unhealthy Links" in clean:
                    continue

                # Normalize the error message
                # Remove timestamps and log prefixes
                msg = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s*\|[^|]+\|[^|]+\|\s*", "", clean)
                msg = re.sub(r"\s*\([^)]+\.(cpp|hpp|h):\d+\)\s*$", "", msg)  # Remove file:line suffix
                msg = re.sub(r"^\[\d+,\d+\]<\w+>:\s*", "", msg)  # Remove MPI rank prefix
                msg = re.sub(r"^what\(\):\s*", "", msg)  # Remove what(): prefix
                msg = msg.strip()

                # Skip very short or duplicate messages
                if msg and len(msg) > 15 and msg not in file_errors:
                    file_errors.add(msg)
                    if os.path.basename(a.filepath) not in unique_errors[msg]:
                        unique_errors[msg].append(os.path.basename(a.filepath))

    return unique_errors


def print_unique_errors(analyses: list[LogAnalysis]):
    """Print unique error messages from logs."""
    errors = extract_unique_errors(analyses)

    if not errors:
        print("No error messages found.")
        print()
        return

    print("=" * 50)
    print("Unique Error Messages")
    print("=" * 50)
    print()

    # Sort by frequency (most common first)
    sorted_errors = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)

    for msg, files in sorted_errors[:20]:  # Limit to top 20
        count = len(files)
        # Truncate long messages
        display_msg = msg[:100] + "..." if len(msg) > 100 else msg
        print(f"{Colors.RED}[{count}x]{Colors.NC} {display_msg}")

    if len(sorted_errors) > 20:
        print(f"\n... and {len(sorted_errors) - 20} more unique error types")
    print()


def print_timeline(analyses: list[LogAnalysis]):
    """Print visual timeline of iteration results."""
    if not analyses:
        return

    print("=" * 50)
    print("Iteration Timeline")
    print("=" * 50)
    print()

    # Sort by iteration number
    sorted_analyses = sorted(analyses, key=lambda a: a.metadata.iteration)

    # Determine what icons to use
    icons = []
    for a in sorted_analyses:
        if "healthy" in a.categories:
            icons.append(("✓", Colors.GREEN))
        elif "unhealthy" in a.categories or "workload_timeout" in a.categories:
            icons.append(("✗", Colors.RED))
        elif "missing_connections" in a.categories or "missing_channels" in a.categories:
            icons.append(("○", Colors.BLUE))
        elif "dram_failure" in a.categories or "pcie_error" in a.categories:
            icons.append(("!", Colors.RED))
        elif "indeterminate" in a.categories:
            icons.append(("?", Colors.CYAN))
        else:
            icons.append(("~", Colors.YELLOW))

    # Print timeline in rows of 25
    row_size = 25
    for row_start in range(0, len(icons), row_size):
        row_end = min(row_start + row_size, len(icons))
        row_icons = icons[row_start:row_end]

        # Print iteration numbers
        nums = " ".join(f"{row_start + i + 1:2d}" for i in range(len(row_icons)))
        print(f"  {nums}")

        # Print status icons
        statuses = " ".join(f"{color}{icon}{Colors.NC} " for icon, color in row_icons)
        print(f"  {statuses}")
        print()

    # Legend
    print(
        f"Legend: {Colors.GREEN}✓{Colors.NC}=healthy  "
        f"{Colors.RED}✗{Colors.NC}=unhealthy  "
        f"{Colors.BLUE}○{Colors.NC}=missing conn  "
        f"{Colors.RED}!{Colors.NC}=hardware  "
        f"{Colors.YELLOW}~{Colors.NC}=other  "
        f"{Colors.CYAN}?{Colors.NC}=indeterminate"
    )
    print()

    # Find patterns - consecutive failures
    consecutive_failures = []
    current_streak = []
    for i, a in enumerate(sorted_analyses):
        if "healthy" not in a.categories:
            current_streak.append(i + 1)
        else:
            if len(current_streak) >= 3:
                consecutive_failures.append(current_streak.copy())
            current_streak = []
    if len(current_streak) >= 3:
        consecutive_failures.append(current_streak)

    if consecutive_failures:
        print(f"{Colors.YELLOW}Warning: Consecutive failure streaks detected:{Colors.NC}")
        for streak in consecutive_failures:
            print(f"  Iterations {streak[0]}-{streak[-1]} ({len(streak)} consecutive failures)")
        print()


def output_json(analyses: list[LogAnalysis]):
    """Output results as JSON."""
    # Extract cluster info
    hosts = set()
    chips_found = 0
    traffic_config = None
    for a in analyses:
        if a.metadata.detected_hosts:
            hosts.update(a.metadata.detected_hosts)
        if a.metadata.chips_found > chips_found:
            chips_found = a.metadata.chips_found
        if a.metadata.num_iterations > 0 and not traffic_config:
            traffic_config = a.metadata

    result = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(analyses),
        "cluster_info": {
            "hosts": sorted(hosts) if hosts else [],
            "chips_per_host": chips_found,
            "total_chips": chips_found * len(hosts) if hosts else chips_found,
            "traffic_iterations": traffic_config.num_iterations if traffic_config else 0,
            "packet_size_bytes": traffic_config.packet_size_bytes if traffic_config else 0,
            "data_size_bytes": traffic_config.data_size_bytes if traffic_config else 0,
        },
        "categories": defaultdict(list),
        "link_stats": {},
        "host_stats": {},
    }

    for a in analyses:
        for cat in a.categories:
            result["categories"][cat].append(os.path.basename(a.filepath))

    # Convert link stats
    link_stats = aggregate_link_stats(analyses)
    for (host, tray, channel, port_type), stats in link_stats.items():
        key = f"{host}:tray{tray}:ch{channel}:{port_type}"
        result["link_stats"][key] = {
            "count": stats["count"],
            "retrains_total": stats["retrains_total"],
            "crc_errors_total": stats["crc_errors_total"],
            "mismatch_words_total": stats["mismatch_words_total"],
            "failure_types": dict(stats["failure_types"]),
        }

    result["host_stats"] = aggregate_host_stats(analyses)

    # Calculate summary
    healthy = len(result["categories"].get("healthy", []))
    result["summary"] = {
        "healthy_count": healthy,
        "success_rate": (healthy / len(analyses) * 100) if analyses else 0,
    }

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Analyze validation output logs for health status and errors.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="validation_output",
        help="Directory containing validation log files (default: validation_output/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Show faulty link histogram",
    )
    parser.add_argument(
        "--hosts",
        action="store_true",
        help="Show per-host failure summary",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all details (histogram, hosts, recommendations)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show matched log lines for each detected issue",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Show iteration timeline with pass/fail status",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Show unique error messages across all logs",
    )

    args = parser.parse_args()

    if args.no_color or args.json:
        Colors.disable()

    # Validate directory
    validation_dir = Path(args.directory)
    if not validation_dir.is_dir():
        print(f"Error: Directory {args.directory} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all log files
    log_files = sorted(validation_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Analyze each file
    analyses = [analyze_log_file(str(f)) for f in log_files]

    # Output
    if args.json:
        output_json(analyses)
    else:
        print_summary(analyses)

        if args.histogram and not args.all:
            print_link_histogram(analyses)

        if args.all:
            print_faulty_links_detail(analyses)  # More detailed than histogram
            print_missing_connections_detail(analyses)

        if args.hosts or args.all:
            print_host_summary(analyses)

        if args.all:
            print_recommendations(analyses)

        if args.timeline or args.all:
            print_timeline(analyses)

        if args.errors or args.all:
            print_unique_errors(analyses)

        if args.verbose:
            print_verbose(analyses)


if __name__ == "__main__":
    main()
