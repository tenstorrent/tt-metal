#!/usr/bin/env python3
"""
Debug script for investigating bandwidth vs peak bandwidth counter inconsistency.

Issue: Regular bandwidth stays 0 while peak bandwidth shows non-zero values.
- Regular bandwidth uses: elapsed_cycles (total time)
- Peak bandwidth uses: elapsed_active_cycles (active time only)

This shouldn't happen - if elapsed_active_cycles increments, elapsed_cycles MUST also increment.

Usage:
    python3 debug_bandwidth_inconsistency.py [--url http://localhost:8080]
"""

import requests
import time
import sys
import argparse
from datetime import datetime
from collections import defaultdict

METRICS_URL = "http://localhost:8080/api/metrics"
POLL_INTERVAL = 5  # seconds - match telemetry update rate


def parse_prometheus_line(line):
    """Parse a Prometheus metric line into (metric_name, labels, value, timestamp)."""
    if line.startswith("#") or not line.strip():
        return None

    try:
        # Split metric{labels} value timestamp
        metric_part, rest = line.split("}", 1)
        metric_name, labels_str = metric_part.split("{", 1)
        parts = rest.strip().split()
        value = float(parts[0])
        timestamp = int(parts[1]) if len(parts) > 1 else None

        # Parse labels into dict
        labels = {}
        for label_pair in labels_str.split(","):
            if "=" in label_pair:
                key, val = label_pair.split("=", 1)
                labels[key.strip()] = val.strip().strip('"')

        return (metric_name, labels, value, timestamp)
    except Exception:
        return None


def get_metrics():
    """Fetch and parse metrics from endpoint."""
    try:
        response = requests.get(METRICS_URL, timeout=5)
        if response.status_code != 200:
            return None

        metrics = defaultdict(lambda: defaultdict(dict))

        for line in response.text.split("\n"):
            parsed = parse_prometheus_line(line)
            if not parsed:
                continue

            metric_name, labels, value, _ = parsed

            # Only interested in bandwidth metrics
            if "Bandwidth" not in metric_name:
                continue

            # Build endpoint key from labels
            chip = labels.get("chipId", labels.get("chip_id", "unknown"))
            channel = labels.get("channel", "unknown")
            endpoint = f"chip{chip}.channel{channel}"

            metrics[endpoint][metric_name] = value

        return dict(metrics)

    except Exception as e:
        print(f"Error fetching metrics: {e}", file=sys.stderr)
        return None


def analyze_metrics(current, previous):
    """Analyze metrics for inconsistencies."""
    inconsistencies = []
    active_endpoints = []

    for endpoint, values in sorted(current.items()):
        tx_bw = values.get("txBandwidthMBps", 0)
        tx_peak = values.get("txPeakBandwidthMBps", 0)
        rx_bw = values.get("rxBandwidthMBps", 0)
        rx_peak = values.get("rxPeakBandwidthMBps", 0)

        # Skip completely idle endpoints
        if all(v == 0 for v in [tx_bw, tx_peak, rx_bw, rx_peak]):
            continue

        active_endpoints.append(endpoint)

        # Detect TX inconsistency
        tx_issue = tx_bw == 0 and tx_peak > 0
        # Detect RX inconsistency
        rx_issue = rx_bw == 0 and rx_peak > 0

        if tx_issue or rx_issue:
            issue = {
                "endpoint": endpoint,
                "tx_bandwidth": tx_bw,
                "tx_peak": tx_peak,
                "rx_bandwidth": rx_bw,
                "rx_peak": rx_peak,
                "tx_inconsistent": tx_issue,
                "rx_inconsistent": rx_issue,
            }

            # Add deltas if we have previous data
            if previous and endpoint in previous:
                prev = previous[endpoint]
                issue["delta_tx_bw"] = tx_bw - prev.get("txBandwidthMBps", 0)
                issue["delta_tx_peak"] = tx_peak - prev.get("txPeakBandwidthMBps", 0)
                issue["delta_rx_bw"] = rx_bw - prev.get("rxBandwidthMBps", 0)
                issue["delta_rx_peak"] = rx_peak - prev.get("rxPeakBandwidthMBps", 0)

            inconsistencies.append(issue)

    return inconsistencies, active_endpoints


def main(args):
    global METRICS_URL
    METRICS_URL = args.url

    print("=" * 80)
    print("BANDWIDTH INCONSISTENCY DEBUG TOOL")
    print("=" * 80)
    print(f"\nMonitoring: {METRICS_URL}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"\nLooking for cases where:")
    print(f"  - Regular bandwidth = 0 BUT peak bandwidth > 0")
    print(f"  - This indicates elapsed_cycles=0 but elapsed_active_cycles>0 (BUG)")
    print(f"\nPress Ctrl+C to stop\n")

    previous = None
    sample_count = 0
    total_issues = 0

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current = get_metrics()

            if not current:
                print(f"[{timestamp}] Failed to fetch metrics")
                time.sleep(POLL_INTERVAL)
                continue

            sample_count += 1
            inconsistencies, active_endpoints = analyze_metrics(current, previous)

            print(f"\n{'='*80}")
            print(f"Sample #{sample_count} at {timestamp}")
            print(f"{'='*80}")
            print(f"Active endpoints: {len(active_endpoints)}")

            if inconsistencies:
                total_issues += len(inconsistencies)
                print(f"\n🔴 FOUND {len(inconsistencies)} INCONSISTENCIES:\n")

                for issue in inconsistencies:
                    print(f"Endpoint: {issue['endpoint']}")

                    if issue["tx_inconsistent"]:
                        print(
                            f"  TX: bandwidth={issue['tx_bandwidth']:.2f} MB/s, "
                            f"peak={issue['tx_peak']:.2f} MB/s ⚠️  BUG!"
                        )
                        if "delta_tx_bw" in issue:
                            print(
                                f"      Δbandwidth={issue['delta_tx_bw']:.2f}, " f"Δpeak={issue['delta_tx_peak']:.2f}"
                            )
                    else:
                        print(f"  TX: bandwidth={issue['tx_bandwidth']:.2f} MB/s, " f"peak={issue['tx_peak']:.2f} MB/s")

                    if issue["rx_inconsistent"]:
                        print(
                            f"  RX: bandwidth={issue['rx_bandwidth']:.2f} MB/s, "
                            f"peak={issue['rx_peak']:.2f} MB/s ⚠️  BUG!"
                        )
                        if "delta_rx_bw" in issue:
                            print(
                                f"      Δbandwidth={issue['delta_rx_bw']:.2f}, " f"Δpeak={issue['delta_rx_peak']:.2f}"
                            )
                    else:
                        print(f"  RX: bandwidth={issue['rx_bandwidth']:.2f} MB/s, " f"peak={issue['rx_peak']:.2f} MB/s")

                    print()

            else:
                print("✓ No inconsistencies detected")

                if args.verbose and active_endpoints:
                    print(f"\nActive endpoints ({len(active_endpoints)}):")
                    for endpoint in active_endpoints[:5]:  # Show first 5
                        values = current[endpoint]
                        print(f"  {endpoint}:")
                        print(
                            f"    TX: bw={values.get('txBandwidthMBps', 0):.2f}, "
                            f"peak={values.get('txPeakBandwidthMBps', 0):.2f}"
                        )
                        print(
                            f"    RX: bw={values.get('rxBandwidthMBps', 0):.2f}, "
                            f"peak={values.get('rxPeakBandwidthMBps', 0):.2f}"
                        )
                    if len(active_endpoints) > 5:
                        print(f"  ... and {len(active_endpoints) - 5} more")

            previous = current
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total samples: {sample_count}")
        print(f"Total inconsistencies found: {total_issues}")
        if total_issues > 0:
            print(f"\n⚠️  Inconsistencies detected! This is a bug that needs investigation.")
            print(f"Likely causes:")
            print(f"  1. Firmware not atomically updating counters")
            print(f"  2. Snapshot timing capturing mid-update")
            print(f"  3. Initialization issue (one counter starts before the other)")
            print(f"  4. dynamic_info availability mismatch")
        else:
            print("\n✓ No inconsistencies detected across all samples")
        return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug bandwidth vs peak bandwidth inconsistency")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/api/metrics",
        help="Metrics endpoint URL (default: http://localhost:8080/api/metrics)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output for healthy endpoints")

    args = parser.parse_args()

    # Check if server is reachable
    try:
        response = requests.get(args.url, timeout=2)
        if response.status_code != 200:
            print(f"Error: Metrics server returned HTTP {response.status_code}", file=sys.stderr)
            print(f"Make sure tt-smi server is running on {args.url}", file=sys.stderr)
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to {args.url}", file=sys.stderr)
        print(f"Make sure tt-smi server is running", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(main(args))
