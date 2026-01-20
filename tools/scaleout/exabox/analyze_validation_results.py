#!/usr/bin/env python3
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
TROUBLESHOOTING_FILE = SCRIPT_DIR / "TROUBLESHOOTING.md"

# Troubleshooting section line numbers (update if file changes)
TROUBLESHOOTING_LINES = {
    "tensix_stall": 288,
    "gddr_issue": 304,
    "fw_mismatch": 347,
    "missing_connections": 391,
    "data_mismatch": 501,
    "ssh_agent": 126,
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


def ts_link(section: str) -> str:
    """Generate clickable troubleshooting link."""
    if TROUBLESHOOTING_FILE.exists():
        return f"{TROUBLESHOOTING_FILE}:{TROUBLESHOOTING_LINES.get(section, 1)}"
    return "TROUBLESHOOTING.md"


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
            if len(parts) >= 12:
                try:
                    # Handle port_type glued to unique_id (e.g., "TRACE0x...")
                    pt = parts[5]
                    if "0x" in pt and not pt.startswith("0x"):
                        pt = pt[: pt.find("0x")]

                    links.append(
                        FaultyLink(
                            host=parts[0],
                            tray=int(parts[1]),
                            asic=int(parts[2]),
                            channel=int(parts[3]),
                            port_id=int(parts[4]),
                            port_type=pt,
                            retrains=parse_int(parts[7]),
                            crc_errors=parse_int(parts[8]),
                            uncorrected_cw=parse_int(parts[10]) if len(parts) > 10 else 0,
                            mismatch_words=parse_int(parts[11]) if len(parts) > 11 else 0,
                            failure_type=" ".join(parts[12:15]) if len(parts) > 12 else "",
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
    """Print detailed faulty links and missing connections."""
    # Faulty links
    all_links = [(os.path.basename(a.filepath), l) for a in analyses for l in a.faulty_links]
    if all_links:
        print("=" * 50 + "\nFaulty Links Detail\n" + "=" * 50 + "\n")
        for log, l in all_links:
            short = log.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
            print(
                f"  {short}: {l.host} tray {l.tray} ASIC {l.asic} ch {l.channel} "
                f"| CRC:{l.crc_errors} Uncorr:{l.uncorrected_cw} Mismatch:{l.mismatch_words}"
            )
        print()

    # Missing connections
    all_missing = [(os.path.basename(a.filepath), c) for a in analyses for c in a.missing_connections]
    if all_missing:
        print("=" * 50 + "\nMissing Connections\n" + "=" * 50 + "\n")
        for log, (ctype, ep1, ep2) in all_missing:
            short = log.replace("cluster_validation_iteration_", "iter_").replace(".log", "")
            if ctype == "port":
                print(f"  {short}: {ep1[0]} tray {ep1[1]} {ep1[2]} <-> {ep2[0]} tray {ep2[1]} {ep2[2]}")
            else:
                print(
                    f"  {short}: {ep1[0]} tray {ep1[1]} ASIC {ep1[2]} ch {ep1[3]} <-> "
                    f"{ep2[0]} tray {ep2[1]} ASIC {ep2[2]} ch {ep2[3]}"
                )
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
    """Print actionable recommendations."""
    cats = defaultdict(int)
    for a in analyses:
        for c in a.categories:
            cats[c] += 1
    total = len(analyses)
    if not total:
        return

    print("=" * 50 + "\nRecommendations\n" + "=" * 50 + "\n")
    recs = []

    if cats["timeout"]:
        recs.append(f"- {Colors.YELLOW}Timeout:{Colors.NC} Power cycle cluster. See {ts_link('tensix_stall')}")
    if cats["missing_ports"] or cats["missing_channels"]:
        recs.append(
            f"- {Colors.BLUE}Missing connections:{Colors.NC} Check cables. See {ts_link('missing_connections')}"
        )
    if cats["pcie_error"]:
        recs.append(f"- {Colors.RED}PCIe errors:{Colors.NC} Check dmesg. May need power cycle.")
    if cats["workload_timeout"]:
        recs.append(f"- {Colors.RED}Workload timeout:{Colors.NC} Power cycle required.")
    if cats["device_error"]:
        recs.append(f"- {Colors.RED}Device error:{Colors.NC} Try tt-smi reset.")
    if cats["dram_failure"]:
        recs.append(f"- {Colors.RED}DRAM failure:{Colors.NC} Hardware issue. See {ts_link('gddr_issue')}")
    if cats["unhealthy"]:
        link_stats, _ = aggregate_stats(analyses)
        repeats = sum(1 for s in link_stats.values() if s["count"] >= 3)
        if repeats:
            recs.append(f"- {Colors.RED}Repeated failures:{Colors.NC} {repeats} links failing 3+ times. Bad cable?")
        else:
            recs.append(f"- {Colors.YELLOW}Scattered failures:{Colors.NC} Try power cycle.")
    if cats["mpi_error"] or cats["ssh_error"]:
        recs.append(f"- {Colors.RED}SSH/MPI error:{Colors.NC} Check agent forwarding. See {ts_link('ssh_agent')}")

    rate = cats["healthy"] / total * 100
    if rate >= 90:
        recs.append(f"- {Colors.GREEN}Cluster healthy.{Colors.NC}")
    elif rate < 70:
        recs.append(f"- {Colors.RED}Low success rate ({rate:.0f}%).{Colors.NC} Investigate before proceeding.")

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
            link_stats, _ = aggregate_stats(analyses)
            if link_stats:
                print("=" * 50 + "\nLink Histogram\n" + "=" * 50 + "\n")
                for (h, t, c, pt), s in sorted(link_stats.items(), key=lambda x: -x[1]["count"])[:15]:
                    print(f"  {h} tray {t} ch {c} {pt}: {s['count']}x")
                print()
        if args.hosts or args.all:
            print_host_summary(analyses)
        if args.all:
            print_recommendations(analyses)
        if args.timeline or args.all:
            print_timeline(analyses)
        if args.errors or args.all:
            print_errors(analyses)
        if args.verbose:
            print("=" * 50 + "\nMatched Lines\n" + "=" * 50 + "\n")
            for a in analyses:
                for cat, matches in a.matched_lines.items():
                    if cat not in ["healthy", "truncated"]:
                        for ln, content in matches[:2]:
                            print(f"  {a.filepath}:{ln}: {content[:100]}")


if __name__ == "__main__":
    main()
