#!/usr/bin/env python3
"""
Parse test_system_health output and generate a markdown connectivity diagram.

Usage:
    ./build/test/tt_metal/tt_fabric/test_system_health | python parse_system_health.py
    python parse_system_health.py < output.txt
    python parse_system_health.py output.txt
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
from loguru import logger


def parse_system_health_output(text: str) -> Dict[int, Dict]:
    """Parse test_system_health output and extract chip connectivity."""
    chips = {}
    current_chip = None

    lines = text.strip().split("\n")

    for line in lines:
        # Parse chip header: "Chip: 3 PCIe: 2 Unique ID: 824632441ec0"
        chip_match = re.match(r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)", line)
        if chip_match:
            chip_id = int(chip_match.group(1))
            pcie_id = int(chip_match.group(2))
            unique_id = chip_match.group(3)
            current_chip = chip_id
            chips[chip_id] = {"pcie": pcie_id, "unique_id": unique_id, "connections": [], "down_links": []}
            continue

        # Parse ethernet channel: "eth channel 4 core (x=0,y=4) link UP (QSFP), connected to Chip 0 core (x=0,y=4)"
        if current_chip is not None and "eth channel" in line:
            eth_match = re.search(r"eth channel (\d+) core \(x=(\d+),y=(\d+)\) link (UP|DOWN)", line)
            if eth_match:
                channel = int(eth_match.group(1))
                x, y = int(eth_match.group(2)), int(eth_match.group(3))
                status = eth_match.group(4)

                if status == "UP":
                    # Check if connected to another chip
                    conn_match = re.search(r"connected to Chip (\d+) core", line)
                    if conn_match:
                        connected_chip = int(conn_match.group(1))
                        chips[current_chip]["connections"].append(
                            {"channel": channel, "core": (x, y), "connected_to": connected_chip}
                        )
                elif status == "DOWN":
                    chips[current_chip]["down_links"].append(channel)

    return chips


def build_connectivity_map(chips: Dict[int, Dict]) -> Dict[Tuple[int, int], List[int]]:
    """Build a map of chip pairs to their connecting channels."""
    connectivity = defaultdict(set)

    for chip_id, chip_data in chips.items():
        for conn in chip_data["connections"]:
            # Create sorted tuple to avoid duplicates (0,1) and (1,0)
            pair = tuple(sorted([chip_id, conn["connected_to"]]))
            # Use set to avoid counting bidirectional links twice
            connectivity[pair].add(conn["channel"])

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in connectivity.items()}


def generate_markdown_report(chips: Dict[int, Dict], connectivity: Dict[Tuple[int, int], List[int]]) -> str:
    """Generate markdown report with connectivity diagram."""
    md = []

    md.append("## Chip Connectivity Analysis\n")

    # Summary
    md.append("### Connection Summary:")
    for pair, channels in sorted(connectivity.items()):
        chip1, chip2 = pair
        link_count = len(channels)
        md.append(
            f"- **Chip {chip1} ↔ Chip {chip2}**: {link_count} links (eth channels {','.join(map(str, channels))})"
        )

    md.append("")

    # Total active links
    total_links = sum(len(channels) for channels in connectivity.values())
    md.append(f"**Total Active Links**: {total_links}\n")

    # Down links
    md.append("### Down/Unconnected Links:")
    has_down = False
    for chip_id, chip_data in sorted(chips.items()):
        if chip_data["down_links"]:
            has_down = True
            md.append(f"- Chip {chip_id}: channels {', '.join(map(str, sorted(chip_data['down_links'])))}")
    if not has_down:
        md.append("- None (all links operational)")
    md.append("")

    # Generate topology diagram
    md.append("### Topology Diagram:\n")
    md.append(generate_mermaid_diagram(chips, connectivity))
    md.append("")

    # Chip details
    md.append("### Chip Details:\n")
    for chip_id in sorted(chips.keys()):
        chip = chips[chip_id]
        md.append(f"**Chip {chip_id}** (PCIe: {chip['pcie']}, ID: {chip['unique_id']})")
        md.append(f"  - Active links: {len(chip['connections'])}")
        md.append(f"  - Down links: {len(chip['down_links'])}")
        md.append("")

    return "\n".join(md)


def generate_mermaid_diagram(chips: Dict[int, Dict], connectivity: Dict[Tuple[int, int], List[int]]) -> str:
    """Generate a Mermaid diagram for chip connectivity."""
    lines = []

    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    %% Chip nodes")

    # Define nodes with chip information
    for chip_id in sorted(chips.keys()):
        chip = chips[chip_id]
        has_down = len(chip["down_links"]) > 0

        # Create node label with chip info
        label = f"Chip {chip_id}<br/>PCIe: {chip['pcie']}<br/>ID: {chip['unique_id'][:8]}..."

        # Add styling based on whether there are down links
        if has_down:
            lines.append(f'    C{chip_id}["{label}"]:::warning')
        else:
            lines.append(f'    C{chip_id}["{label}"]:::healthy')

    lines.append("")
    lines.append("    %% Connections")

    # Add edges for connections
    for (chip1, chip2), channels in sorted(connectivity.items()):
        link_count = len(channels)
        channel_str = ",".join(map(str, channels))
        # Use thick lines for more links
        if link_count >= 4:
            lines.append(f"    C{chip1} <==> |{link_count} links<br/>ch: {channel_str}| C{chip2}")
        else:
            lines.append(f"    C{chip1} <--> |{link_count} links<br/>ch: {channel_str}| C{chip2}")

    lines.append("")
    lines.append("    %% Styling")
    lines.append("    classDef healthy fill:#90EE90,stroke:#2E8B57,stroke-width:2px,color:#000")
    lines.append("    classDef warning fill:#FFD700,stroke:#FF8C00,stroke-width:2px,color:#000")
    lines.append("```")

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Read input
    if len(sys.argv) > 1:
        # Read from file
        filepath = sys.argv[1]
        logger.info(f"Reading from file: {filepath}")
        with open(filepath, "r") as f:
            text = f.read()
    else:
        # Read from stdin
        logger.info("Reading from stdin...")
        text = sys.stdin.read()

    if not text.strip():
        logger.error("No input provided")
        sys.exit(1)

    # Parse the output
    logger.info("Parsing system health output...")
    chips = parse_system_health_output(text)

    if not chips:
        logger.error("No chip data found in input")
        sys.exit(1)

    logger.info(f"Found {len(chips)} chips")

    # Build connectivity map
    connectivity = build_connectivity_map(chips)
    logger.info(f"Found {len(connectivity)} chip-to-chip connections")

    # Generate markdown report
    markdown = generate_markdown_report(chips, connectivity)

    # Output
    print(markdown)


if __name__ == "__main__":
    main()
