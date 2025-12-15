#!/usr/bin/env python3
"""
Scrape metrics endpoint at high frequency until debug heartbeat values are found.
Looking for txHeartbeat=42 and rxHeartbeat=69.
"""

import requests
import time
import sys
from datetime import datetime

# Configuration
METRICS_URL = "http://localhost:8080/api/metrics"
POLL_INTERVAL = 0.01  # seconds - very high frequency
OUTPUT_FILE = f"heartbeat_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Debug values we're looking for
TX_HEARTBEAT_DEBUG_VALUE = 42
RX_HEARTBEAT_DEBUG_VALUE = 24

# Spinner characters
SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def check_for_debug_heartbeats(metrics_text):
    """
    Check if txHeartbeat=42 and rxHeartbeat=69 are present.
    Returns (found_tx, found_rx, matching_lines) tuple.
    """
    found_tx = False
    found_rx = False
    matching_lines = []

    for line in metrics_text.split("\n"):
        # Skip TYPE and HELP lines
        if line.startswith("#"):
            continue

        # Look for heartbeat metrics
        if "txHeartbeat{" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    value = int(float(parts[-2]))  # Second to last is the value
                    if value == TX_HEARTBEAT_DEBUG_VALUE:
                        found_tx = True
                        matching_lines.append(("TX", line))
                except (ValueError, IndexError):
                    continue

        elif "rxHeartbeat{" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    value = int(float(parts[-2]))  # Second to last is the value
                    if value == RX_HEARTBEAT_DEBUG_VALUE:
                        found_rx = True
                        matching_lines.append(("RX", line))
                except (ValueError, IndexError):
                    continue

    return found_tx, found_rx, matching_lines


def main():
    print(f"🔍 Monitoring {METRICS_URL} for debug heartbeat values...")
    print(f"   Looking for: txHeartbeat={TX_HEARTBEAT_DEBUG_VALUE}, rxHeartbeat={RX_HEARTBEAT_DEBUG_VALUE}")
    print(f"   Polling every {POLL_INTERVAL}s")
    print(f"   Press Ctrl+C to stop\n")

    spinner_idx = 0
    attempts = 0
    found_tx_ever = False
    found_rx_ever = False

    try:
        while True:
            attempts += 1

            # Show spinner with status
            status = ""
            if found_tx_ever:
                status += " TX✓"
            if found_rx_ever:
                status += " RX✓"

            sys.stdout.write(f"\r{SPINNER[spinner_idx]} Checking... (attempt {attempts}){status}")
            sys.stdout.flush()
            spinner_idx = (spinner_idx + 1) % len(SPINNER)

            try:
                # Fetch metrics
                response = requests.get(METRICS_URL, timeout=2)

                if response.status_code == 200:
                    metrics_text = response.text

                    # Check for debug heartbeat values
                    found_tx, found_rx, matching_lines = check_for_debug_heartbeats(metrics_text)

                    # Update our tracking
                    if found_tx:
                        found_tx_ever = True
                    if found_rx:
                        found_rx_ever = True

                    # If we found both, we're done!
                    if found_tx and found_rx:
                        # Clear spinner line
                        sys.stdout.write("\r" + " " * 80 + "\r")

                        # Save to file
                        with open(OUTPUT_FILE, "w") as f:
                            f.write(f"# Captured at: {datetime.now().isoformat()}\n")
                            f.write(f"# After {attempts} attempts\n")
                            f.write(f"# Found txHeartbeat={TX_HEARTBEAT_DEBUG_VALUE}: {found_tx}\n")
                            f.write(f"# Found rxHeartbeat={RX_HEARTBEAT_DEBUG_VALUE}: {found_rx}\n\n")
                            f.write(metrics_text)

                        print(f"✅ Success! Debug heartbeat values detected!")
                        print(f"📁 Saved to: {OUTPUT_FILE}")
                        print(f"🔢 Attempts: {attempts}")

                        # Print the matching metrics
                        print(f"\n💓 Found {len(matching_lines)} matching heartbeat metrics:")
                        for hb_type, line in matching_lines:
                            print(f"   [{hb_type}] {line}")

                        return 0

                    # If we found one but not the other, show partial success
                    elif found_tx and not found_rx_ever:
                        sys.stdout.write(
                            f"\r{SPINNER[spinner_idx]} Found TX={TX_HEARTBEAT_DEBUG_VALUE} ✓ | Still searching for RX={RX_HEARTBEAT_DEBUG_VALUE}... (attempt {attempts})"
                        )
                        sys.stdout.flush()
                    elif found_rx and not found_tx_ever:
                        sys.stdout.write(
                            f"\r{SPINNER[spinner_idx]} Found RX={RX_HEARTBEAT_DEBUG_VALUE} ✓ | Still searching for TX={TX_HEARTBEAT_DEBUG_VALUE}... (attempt {attempts})"
                        )
                        sys.stdout.flush()

                else:
                    sys.stdout.write(f"\r⚠️  HTTP {response.status_code} - retrying...")
                    sys.stdout.flush()

            except requests.exceptions.RequestException as e:
                sys.stdout.write(f"\r❌ Connection error - retrying...")
                sys.stdout.flush()

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 80 + "\r")
        print("\n\n⏹️  Monitoring stopped by user")
        print(f"   Total attempts: {attempts}")
        if found_tx_ever:
            print(f"   ✓ Found txHeartbeat={TX_HEARTBEAT_DEBUG_VALUE}")
        else:
            print(f"   ✗ Never found txHeartbeat={TX_HEARTBEAT_DEBUG_VALUE}")
        if found_rx_ever:
            print(f"   ✓ Found rxHeartbeat={RX_HEARTBEAT_DEBUG_VALUE}")
        else:
            print(f"   ✗ Never found rxHeartbeat={RX_HEARTBEAT_DEBUG_VALUE}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
