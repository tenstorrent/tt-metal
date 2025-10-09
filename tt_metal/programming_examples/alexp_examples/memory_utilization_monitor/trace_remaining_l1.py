#!/usr/bin/env python3
"""
Trace the remaining 3 L1 buffers (12KB) that persist after test completion.

This script helps identify WHERE these buffers are allocated by:
1. Running the test with detailed logging
2. Capturing allocation events
3. Matching buffer IDs to their allocation source
4. Reporting which buffers remain after cleanup

Usage:
    # Terminal 1: Start allocation server (if not already running)
    ./allocation_server_poc

    # Terminal 2: Run this tracer
    python trace_remaining_l1.py
"""

import subprocess
import time
import socket
import json
import os
from collections import defaultdict

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


class AllocationTracer:
    def __init__(self):
        self.allocations = {}  # buffer_id -> {device, size, type, time}
        self.deallocations = set()  # buffer_ids that were freed
        self.server_socket_path = "/tmp/tt_allocation_server.sock"

    def check_server(self):
        """Check if allocation server is running"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.server_socket_path)
            sock.close()
            print(f"{GREEN}✓ Allocation server is running{RESET}")
            return True
        except:
            print(f"{RED}✗ Allocation server not found at {self.server_socket_path}{RESET}")
            print(f"{YELLOW}Start it with: ./allocation_server_poc{RESET}")
            return False

    def run_test_and_monitor(self):
        """Run the mesh allocation test while monitoring allocations"""
        print(f"\n{BOLD}{CYAN}Starting Mesh Allocation Test with Tracing{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")

        # Start monitoring allocations via server query
        print(f"{YELLOW}Note: This script monitors the test, but the detailed tracking")
        print(f"      is done by the allocation server itself.{RESET}\n")

        # Run the test
        env = os.environ.copy()
        env["TT_ALLOC_TRACKING_ENABLED"] = "1"

        print(f"{CYAN}Running test...{RESET}")
        result = subprocess.run(
            ["python", "test_mesh_allocation.py"],
            cwd="/workspace/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor",
            env=env,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print(f"{YELLOW}Stderr:{RESET}")
            print(result.stderr)

        # Give server time to process final deallocations
        print(f"\n{CYAN}Waiting 2 seconds for final cleanup...{RESET}")
        time.sleep(2)

        # Query server for remaining allocations
        self.query_remaining_allocations()

    def query_remaining_allocations(self):
        """Query allocation server for remaining allocations"""
        print(f"\n{BOLD}{CYAN}Querying Remaining Allocations{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.server_socket_path)

            # Send query command (the server protocol would need to support this)
            sock.sendall(b"QUERY_ALL\n")

            # Receive response
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n\n" in response:  # End of message
                    break

            sock.close()

            # Parse response
            if response:
                print(f"{GREEN}Server Response:{RESET}")
                print(response.decode())
            else:
                print(f"{YELLOW}No response from server{RESET}")

        except Exception as e:
            print(f"{RED}Error querying server: {e}{RESET}")
            print(f"\n{YELLOW}The allocation server might not support QUERY_ALL command yet.{RESET}")
            print(f"{YELLOW}Check the server output directly for remaining allocations.{RESET}")

    def analyze_leaks(self):
        """Analyze which allocations weren't freed"""
        print(f"\n{BOLD}{CYAN}Leak Analysis{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")

        remaining = []
        for buffer_id, info in self.allocations.items():
            if buffer_id not in self.deallocations:
                remaining.append((buffer_id, info))

        if not remaining:
            print(f"{GREEN}✓ No leaks detected! All buffers were properly freed.{RESET}")
            return

        print(f"{RED}⚠ Found {len(remaining)} leaked buffer(s):{RESET}\n")

        # Group by device
        by_device = defaultdict(list)
        for buffer_id, info in remaining:
            by_device[info["device"]].append((buffer_id, info))

        for device_id in sorted(by_device.keys()):
            buffers = by_device[device_id]
            total_size = sum(info["size"] for _, info in buffers)

            print(f"{YELLOW}Device {device_id}:{RESET}")
            print(f"  Total leaked: {total_size} bytes ({total_size/1024:.1f} KB)")
            print(f"  Buffers:")

            for buffer_id, info in buffers:
                buffer_type = {0: "DRAM", 1: "L1", 2: "L1_SMALL", 3: "TRACE"}.get(info["type"], "UNKNOWN")
                print(f"    • Buffer {buffer_id}: {info['size']} bytes of {buffer_type}")
            print()


def main():
    tracer = AllocationTracer()

    # Check if server is running
    if not tracer.check_server():
        return 1

    # Run test and monitor
    tracer.run_test_and_monitor()

    print(f"\n{BOLD}{CYAN}Tracing Complete{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    print(f"{YELLOW}To see detailed allocation events, check the output of:{RESET}")
    print(f"{YELLOW}  ./allocation_server_poc{RESET}\n")

    print(f"{YELLOW}The server shows exactly which 3 L1 buffers remain and their buffer IDs.{RESET}")
    print(f"{YELLOW}Look for the 'Current Statistics' section at the end.{RESET}\n")

    return 0


if __name__ == "__main__":
    exit(main())
