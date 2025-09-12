#!/usr/bin/env python3
"""
TT Excellence Script for Fabric Router Debugging

This script analyzes compiled outputs and generated outputs to debug the system.
It looks at watcher.log and kernel_elf_paths.txt to build a view of the current
state of fabric router cores and their program counter locations.
"""

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Check if tt-exalens module is available

import ttexalens
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_init import init_ttexalens


class FabricRouterDebugger:
    def __init__(self, watcher_log_path: str, kernel_elf_paths_path: str):
        self.watcher_log_path = Path(watcher_log_path)
        self.kernel_elf_paths_path = Path(kernel_elf_paths_path)
        self.fabric_routers = {}  # device_id -> core_info
        self.kernel_elf_map = {}  # kernel_id -> elf_path
        self.pc_data = defaultdict(list)  # core_key -> [pc_samples]

    def parse_kernel_elf_paths(self):
        """Parse kernel_elf_paths.txt to map kernel IDs to ELF files."""
        print("Parsing kernel ELF paths...")

        with open(self.kernel_elf_paths_path, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line and "fabric_erisc_router" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        kernel_id = int(parts[0].strip())
                        elf_path = parts[1].strip()
                        self.kernel_elf_map[kernel_id] = elf_path
                        # print(f"  Found fabric router kernel {kernel_id}: {elf_path}")

        # print(f"Found {len(self.kernel_elf_map)} fabric router kernel mappings")

    def parse_watcher_log(self):
        """Parse watcher.log to find active fabric router cores."""
        print("Parsing watcher log for fabric router cores...")

        # Pattern to match fabric router cores (acteth cores with fabric_erisc_router kernel IDs)
        fabric_core_pattern = re.compile(
            r"Device (\d+) acteth core\(x=\s*(\d+),y=\s*(\d+)\) virtual\(x=\s*(\d+),y=\s*(\d+)\):.*k_ids:\s*(\d+)\|"
        )

        # Keep track of the latest entry for each core
        latest_entries = {}

        with open(self.watcher_log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                match = fabric_core_pattern.search(line)
                if match:
                    device_id = int(match.group(1))
                    core_x = int(match.group(2))
                    core_y = int(match.group(3))
                    virtual_x = int(match.group(4))
                    virtual_y = int(match.group(5))
                    kernel_id = int(match.group(6))

                    # Check if this kernel_id corresponds to a fabric router
                    if kernel_id in self.kernel_elf_map:
                        core_key = f"Device{device_id}_Core({core_x},{core_y})"
                        latest_entries[core_key] = {
                            "device_id": device_id,
                            "core_x": core_x,
                            "core_y": core_y,
                            "virtual_x": virtual_x,
                            "virtual_y": virtual_y,
                            "kernel_id": kernel_id,
                            "elf_path": self.kernel_elf_map[kernel_id],
                            "line_number": line_num,
                        }

        self.fabric_routers = latest_entries
        # print(f"Found {len(self.fabric_routers)} active fabric router cores:")
        # for core_key, info in self.fabric_routers.items():
        #     print(f"  {core_key}: kernel_id={info['kernel_id']}, elf={info['elf_path']}")

    def collect_pc_data(self, num_samples: int = 5) -> bool:
        """Collect PC data for each fabric router using tt-exalens module.

        Returns:
            bool: True if PC data collection was successful, False otherwise
        """
        print(f"Collecting PC data for {len(self.fabric_routers)} fabric router cores...")

        # Limit to first 3 cores for debugging
        cores_to_process = list(self.fabric_routers.items())[:3]

        # Initialize tt-exalens context
        try:
            print("  Initializing tt-exalens context...")
            context = init_ttexalens(use_noc1=False)
            print("    tt-exalens context initialized successfully")

        except Exception as e:
            print(f"ERROR: Failed to initialize tt-exalens context: {e}")
            print("This usually means no TT hardware is connected or accessible.")
            print("tt-exalens requires actual TT hardware or simulation environment.")
            return False

        for core_key, core_info in cores_to_process:
            device_id = core_info["device_id"]
            core_x = core_info["core_x"]
            core_y = core_info["core_y"]

            print(f"  Collecting PC data for {core_key}...")
            print(f"    Using active core location: Device {device_id}, Core ({core_x},{core_y})")

            # Get device
            device = context.devices[device_id]
            print(f"    Accessing device {device_id}, core ({core_x},{core_y})")

            # Get all eth block locations and find the one matching our coordinates
            eth_locations = device.get_block_locations("eth")
            target_location = None

            for location in eth_locations:
                eth_x = 0
                eth_y = int(location.to_str("logical").split(",")[1])
                if eth_x == core_x and eth_y == core_y:
                    target_location = location
                    break
            assert target_location is not None

            # Get the block and risc debug interface
            block = device.get_block(target_location)

            # Try to get PC from the risc debug interface
            # Try different RISC names as different eth block types may use different names
            risc_names_to_try = ["erisc0", "erisc1", "ncrisc", "trisc0", "trisc1", "trisc2"]
            pc_value = None

            num_samples = 100
            pc_set = set()
            for sample in range(num_samples):
                for risc_name in risc_names_to_try:
                    try:
                        risc_debug = block.get_risc_debug(risc_name)
                        pc_value = risc_debug.get_pc()
                        break
                    except Exception as e:
                        continue

                pc_set.add(f"0x{pc_value:x}")

            self.pc_data[core_key] = list(sorted(pc_set))

        return True

    def resolve_pc_to_source(self):
        """Resolve PC addresses to source code locations using ELF files."""
        print("Resolving PC addresses to source code locations...")

        for core_key, core_info in self.fabric_routers.items():
            if core_key not in self.pc_data:
                continue

            elf_path = core_info["elf_path"]
            print(f"  Resolving PCs for {core_key} using {elf_path}")

            # Use addr2line or objdump to resolve PC to source
            resolved_locations = []

            for pc in self.pc_data[core_key]:
                try:
                    # Use addr2line to get source location
                    cmd = ["addr2line", "-e", elf_path, "-f", "-C", pc]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        if len(lines) >= 2:
                            function = lines[0]
                            location = lines[1]
                            resolved_locations.append({"pc": pc, "function": function, "location": location})
                        else:
                            resolved_locations.append({"pc": pc, "function": "unknown", "location": "unknown"})
                    else:
                        resolved_locations.append(
                            {"pc": pc, "function": "error", "location": f"addr2line error: {result.stderr}"}
                        )

                except Exception as e:
                    print(f"  Exception: {e}")
                    resolved_locations.append({"pc": pc, "function": "error", "location": f"Exception: {e}"})

            # Store resolved locations
            core_info["resolved_locations"] = resolved_locations

    def generate_summary(self):
        """Generate a summary report of all captured PCs."""
        print("\n" + "=" * 80)
        print("FABRIC ROUTER DEBUG SUMMARY")
        print("=" * 80)

        if not self.fabric_routers:
            print("No fabric router cores found!")
            return

        for core_key, core_info in self.fabric_routers.items():
            print(f"\nCore: {core_key}")
            print(f"  Device ID: {core_info['device_id']}")
            print(f"  Active Core Location: ({core_info['core_x']}, {core_info['core_y']})")
            print(f"  Virtual Core Location: ({core_info['virtual_x']}, {core_info['virtual_y']})")
            print(f"  Kernel ID: {core_info['kernel_id']}")
            print(f"  ELF Path: {core_info['elf_path']}")

            if core_key in self.pc_data:
                print(f"  PC Samples ({len(self.pc_data[core_key])}):")
                for i, pc in enumerate(self.pc_data[core_key], 1):
                    print(f"    Sample {i}: {pc}")

            if "resolved_locations" in core_info:
                print(f"  Resolved Locations:")
                for i, loc in enumerate(core_info["resolved_locations"], 1):
                    print(f"    Sample {i}: {loc['pc']}")
                    print(f"      Function: {loc['function']}")
                    print(f"      Location: {loc['location']}")
            else:
                print("  No resolved locations available")

        print("\n" + "=" * 80)
        print(f"Total fabric router cores analyzed: {len(self.fabric_routers)}")
        print("=" * 80)


def main():
    """Main function to run the fabric router debugger."""
    if len(sys.argv) != 3:
        print("Usage: python fabric_router_debug.py <watcher_log_path> <kernel_elf_paths_path>")
        sys.exit(1)

    watcher_log_path = sys.argv[1]
    kernel_elf_paths_path = sys.argv[2]

    debugger = FabricRouterDebugger(watcher_log_path, kernel_elf_paths_path)

    try:
        # Parse the kernel ELF paths first
        debugger.parse_kernel_elf_paths()

        # Parse the watcher log to find fabric router cores
        debugger.parse_watcher_log()

        if not debugger.fabric_routers:
            print("No active fabric router cores found!")
            return

        # Collect PC data for each fabric router
        pc_data_collected = debugger.collect_pc_data(num_samples=5)

        # Only continue if PC data was successfully collected
        if pc_data_collected:
            # Resolve PC addresses to source code locations
            debugger.resolve_pc_to_source()

            # Generate summary report
            debugger.generate_summary()
        else:
            print("Skipping PC resolution and summary generation due to tt-exalens initialization failure.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
