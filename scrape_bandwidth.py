#!/usr/bin/env python3
"""
Scrape metrics endpoint at high frequency until non-zero bandwidth metrics are found.
"""

import requests
import time
import sys
from datetime import datetime

# Configuration
METRICS_URL = "http://localhost:8080/api/metrics"
POLL_INTERVAL = 0.05  # seconds - very high frequency
OUTPUT_FILE = f"bandwidth_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
DEBUG = False  # Set to True to see what metrics are being checked

# Spinner characters
SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def check_for_nonzero_bandwidth(metrics_text):
    """
    Check if any bandwidth or fabric metrics are non-zero.
    Returns (found, metric_type) tuple.
    """
    # List of metrics to check
    metric_patterns = [
        "rxBandwidthMBps{",
        "txBandwidthMBps{",
        "rxPeakBandwidthMBps{",
        "txPeakBandwidthMBps{",
    ]

    checked_count = 0
    for line in metrics_text.split("\n"):
        # Skip TYPE and HELP lines
        if line.startswith("#"):
            continue

        # Check each metric pattern
        for pattern in metric_patterns:
            if pattern in line:
                # Extract the value (last space-separated field before timestamp)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[-2])  # Second to last is the value
                        checked_count += 1
                        if DEBUG and checked_count <= 5:
                            print(f"\n[DEBUG] {pattern}: {value}")
                        if value > 0.0:
                            return (True, pattern.rstrip("{"))
                    except (ValueError, IndexError):
                        continue

    if DEBUG:
        print(f"\n[DEBUG] Checked {checked_count} metrics, all zero")
    return (False, None)


def main():
    print(f"🔍 Monitoring {METRICS_URL} for non-zero fabric metrics...")
    print(f"   Checking: rxBandwidthMBps, txBandwidthMBps, rxPeakBandwidthMBps, txPeakBandwidthMBps")
    print(f"   Polling every {POLL_INTERVAL}s")
    print(f"   Debug mode: {'ON' if DEBUG else 'OFF'} (set DEBUG=True in script to enable)")
    print(f"   Press Ctrl+C to stop\n")

    spinner_idx = 0
    attempts = 0

    try:
        while True:
            attempts += 1

            # Show spinner
            sys.stdout.write(f"\r{SPINNER[spinner_idx]} Checking... (attempt {attempts})")
            sys.stdout.flush()
            spinner_idx = (spinner_idx + 1) % len(SPINNER)

            try:
                # Fetch metrics
                response = requests.get(METRICS_URL, timeout=2)

                if response.status_code == 200:
                    metrics_text = response.text

                    # Check for non-zero bandwidth
                    found, metric_type = check_for_nonzero_bandwidth(metrics_text)
                    if found:
                        # Clear spinner line
                        sys.stdout.write("\r" + " " * 80 + "\r")

                        # Save to file
                        with open(OUTPUT_FILE, "w") as f:
                            f.write(f"# Captured at: {datetime.now().isoformat()}\n")
                            f.write(f"# After {attempts} attempts\n")
                            f.write(f"# Triggered by: {metric_type}\n\n")
                            f.write(metrics_text)

                        print(f"✅ Success! Non-zero {metric_type} detected!")
                        print(f"📁 Saved to: {OUTPUT_FILE}")
                        print(f"🔢 Attempts: {attempts}")

                        # Print a sample of non-zero metrics
                        print("\n📊 Sample of non-zero fabric metrics:")
                        count = 0
                        metric_patterns = [
                            "rxBandwidthMBps{",
                            "txBandwidthMBps{",
                            "rxPeakBandwidthMBps{",
                            "txPeakBandwidthMBps{",
                        ]
                        for line in metrics_text.split("\n"):
                            if any(pattern in line for pattern in metric_patterns) and not line.startswith("#"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        value = float(parts[-2])
                                        if value > 0.0:
                                            print(f"   {line}")
                                            count += 1
                                            if count >= 10:  # Show first 10
                                                break
                                    except (ValueError, IndexError):
                                        pass

                        return 0

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
        return 1


if __name__ == "__main__":
    sys.exit(main())
