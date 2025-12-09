#!/usr/bin/env python3
"""
Monitor telemetry metrics endpoint for health and reasonable values.
Checks for:
- Non-zero bandwidth metrics during active transfers
- Incrementing heartbeats
- Reasonable metric values (not suspiciously large)
"""

import requests
import time
import sys
from datetime import datetime
from collections import defaultdict

# Configuration
METRICS_URL = "http://localhost:8080/api/metrics"
POLL_INTERVAL = 1.0  # seconds
SAMPLE_COUNT = 10  # Number of samples to collect

# Thresholds
MAX_REASONABLE_BANDWIDTH_MBPS = 100000  # 100 GB/s - way above realistic
MAX_REASONABLE_HEARTBEAT = 10**12  # 1 trillion - way above realistic
MIN_HEARTBEAT_INCREMENT = 1  # Heartbeats should increment


def parse_metric_line(line):
    """Parse a Prometheus metric line into name, labels, value, timestamp."""
    if line.startswith("#") or not line.strip():
        return None

    try:
        # Format: metric_name{label1="value1",label2="value2"} value timestamp
        if "{" in line:
            metric_name = line[: line.index("{")]
            labels_end = line.index("}")
            labels_str = line[line.index("{") + 1 : labels_end]
            rest = line[labels_end + 1 :].strip().split()
        else:
            parts = line.split()
            metric_name = parts[0]
            labels_str = ""
            rest = parts[1:]

        value = float(rest[0]) if len(rest) > 0 else 0
        timestamp = int(rest[1]) if len(rest) > 1 else 0

        # Parse labels into dict
        labels = {}
        if labels_str:
            for label_pair in labels_str.split(","):
                if "=" in label_pair:
                    key, val = label_pair.split("=", 1)
                    labels[key.strip()] = val.strip('"')

        return {"name": metric_name, "labels": labels, "value": value, "timestamp": timestamp}
    except Exception as e:
        return None


def get_metrics():
    """Fetch and parse metrics from endpoint."""
    try:
        response = requests.get(METRICS_URL, timeout=5)
        if response.status_code == 200:
            metrics = []
            for line in response.text.split("\n"):
                parsed = parse_metric_line(line)
                if parsed:
                    metrics.append(parsed)
            return metrics
        return None
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None


def check_health(samples):
    """Analyze collected samples for health issues."""
    issues = []
    successes = []

    # Track bandwidth metrics
    bandwidth_metrics = defaultdict(list)
    heartbeat_metrics = defaultdict(list)

    for sample in samples:
        for metric in sample:
            if "Bandwidth" in metric["name"]:
                key = f"{metric['name']}[{metric['labels'].get('channel', 'unknown')}]"
                bandwidth_metrics[key].append(metric["value"])
            elif "Heartbeat" in metric["name"] or "heartbeat" in metric["name"]:
                key = f"{metric['name']}[{metric['labels'].get('channel', 'unknown')}]"
                heartbeat_metrics[key].append(metric["value"])

    # Check bandwidth metrics
    for key, values in bandwidth_metrics.items():
        non_zero = [v for v in values if v > 0]
        if non_zero:
            avg = sum(non_zero) / len(non_zero)
            max_val = max(non_zero)

            if max_val > MAX_REASONABLE_BANDWIDTH_MBPS:
                issues.append(f"⚠️  {key}: Suspiciously high bandwidth {max_val:.2f} MB/s")
            else:
                successes.append(f"✅ {key}: Active with avg {avg:.2f} MB/s (max {max_val:.2f})")

    # Check heartbeat metrics
    for key, values in heartbeat_metrics.items():
        if len(values) >= 2:
            # Check if incrementing
            increments = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            positive_increments = [inc for inc in increments if inc > 0]

            if values[-1] > MAX_REASONABLE_HEARTBEAT:
                issues.append(f"⚠️  {key}: Suspiciously high value {values[-1]}")
            elif len(positive_increments) >= len(increments) * 0.5:  # At least 50% incrementing
                avg_inc = sum(positive_increments) / len(positive_increments) if positive_increments else 0
                successes.append(f"✅ {key}: Incrementing (avg +{avg_inc:.0f}/sample)")
            else:
                issues.append(f"⚠️  {key}: Not incrementing consistently")

    return issues, successes


def main():
    print(f"🔍 Monitoring telemetry health at {METRICS_URL}")
    print(f"   Collecting {SAMPLE_COUNT} samples at {POLL_INTERVAL}s intervals")
    print(f"   Press Ctrl+C to stop\n")

    try:
        samples = []

        # Collect samples
        for i in range(SAMPLE_COUNT):
            sys.stdout.write(f"\r📊 Collecting sample {i+1}/{SAMPLE_COUNT}...")
            sys.stdout.flush()

            metrics = get_metrics()
            if metrics:
                samples.append(metrics)

            if i < SAMPLE_COUNT - 1:
                time.sleep(POLL_INTERVAL)

        sys.stdout.write("\r" + " " * 80 + "\r")
        print(f"✅ Collected {len(samples)} samples\n")

        # Analyze
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)

        issues, successes = check_health(samples)

        if successes:
            print("\n✅ HEALTHY METRICS:")
            for success in successes[:20]:  # Show first 20
                print(f"   {success}")
            if len(successes) > 20:
                print(f"   ... and {len(successes) - 20} more")

        if issues:
            print(f"\n⚠️  ISSUES DETECTED ({len(issues)}):")
            for issue in issues[:20]:  # Show first 20
                print(f"   {issue}")
            if len(issues) > 20:
                print(f"   ... and {len(issues) - 20} more")

        if not successes and not issues:
            print("\n⚠️  No active telemetry detected!")
            print("   - Are you running a fabric test?")
            print("   - Is TT_METAL_FABRIC_TELEMETRY=1 set?")

        print("\n" + "=" * 80)

        # Summary
        total_checks = len(successes) + len(issues)
        health_pct = (len(successes) / total_checks * 100) if total_checks > 0 else 0

        print(f"\nHEALTH SCORE: {health_pct:.1f}% ({len(successes)}/{total_checks} checks passed)")

        return 0 if len(issues) == 0 else 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
