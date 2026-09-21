#!/usr/bin/env python3
"""Post runner-failure scanner health alerts to Slack."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner_failure_post_slack import (
    load_report_json,
    post_slack_message,
    slack_config_from_channel,
    slack_escape,
    slack_link,
    workflow_run_url_from_env,
)


DEFAULT_FAILURE_RATE_THRESHOLD = 0.10


@dataclass(frozen=True)
class ScanHealth:
    attempts: int
    successes: int
    failures: int
    failure_rate: float
    failure_statuses: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post a Slack alert when runner-failure log downloads are unhealthy.")
    parser.add_argument(
        "--scan-report-json",
        type=Path,
        required=True,
        help="Path to a completed runner-failure scan report JSON.",
    )
    parser.add_argument(
        "--slack-channel",
        default=os.environ.get("RUNNER_FAILURE_SCAN_SLACK_CHANNEL"),
        help="Slack channel ID for runner-failure scan health alerts.",
    )
    parser.add_argument(
        "--failure-rate-threshold",
        type=float,
        default=DEFAULT_FAILURE_RATE_THRESHOLD,
        help="Alert when the failed log download fraction is above this value (default: 0.10).",
    )
    return parser.parse_args()


def nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def scan_health_from_report(report: dict[str, Any]) -> ScanHealth:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    results = report.get("scan_results") if isinstance(report.get("scan_results"), list) else []

    attempts = nonnegative_int(counts.get("log_download_attempts"), nonnegative_int(counts.get("jobs_to_scan")))
    result_successes = sum(1 for result in results if isinstance(result, dict) and result.get("log_checked") is True)
    successes = nonnegative_int(counts.get("log_download_successes"), result_successes)
    failures = nonnegative_int(counts.get("log_download_failures"), max(attempts - successes, 0))
    failure_rate = failures / attempts if attempts else 0.0

    failure_statuses: Counter[str] = Counter()
    for result in results:
        if not isinstance(result, dict) or result.get("log_checked") is True:
            continue
        status = " ".join(str(result.get("log_status") or "unknown error").split())
        failure_statuses[status] += 1

    missing_results = max(failures - sum(failure_statuses.values()), 0)
    if missing_results:
        failure_statuses["job scan did not return a result"] += missing_results

    return ScanHealth(
        attempts=attempts,
        successes=successes,
        failures=failures,
        failure_rate=failure_rate,
        failure_statuses=failure_statuses,
    )


def should_post_health_alert(health: ScanHealth, threshold: float) -> bool:
    return health.attempts > 0 and health.failure_rate > threshold


def compact_status(value: str, max_length: int = 180) -> str:
    value = value.replace("`", "'")
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def format_scan_health_alert(health: ScanHealth, threshold: float, workflow_run_url: str) -> str:
    text = (
        "Runner failure scan health alert: "
        f"{health.failures} of {health.attempts} job logs ({health.failure_rate:.1%}) could not be downloaded, "
        f"above the {threshold:.1%} threshold. Runner-failure detection may be incomplete."
    )
    if workflow_run_url:
        text += f" {slack_link(workflow_run_url, 'View workflow run')}."

    if health.failure_statuses:
        status_summary = "; ".join(
            f"{count}x {slack_escape(compact_status(status))}"
            for status, count in health.failure_statuses.most_common(3)
        )
        text += f" Most common errors: {status_summary}."
    return text


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.failure_rate_threshold <= 1.0:
        print("--failure-rate-threshold must be between 0 and 1.", file=sys.stderr)
        return 1

    try:
        report = load_report_json(args.scan_report_json)
        health = scan_health_from_report(report)
        print(
            f"Log download health: {health.successes}/{health.attempts} succeeded; "
            f"{health.failures} failed ({health.failure_rate:.1%})."
        )
        if not should_post_health_alert(health, args.failure_rate_threshold):
            return 0

        slack_config = slack_config_from_channel(args.slack_channel)
        if slack_config is None:
            raise RuntimeError("Log download failure rate exceeded the threshold, but no Slack channel is configured.")
        post_slack_message(
            slack_config,
            format_scan_health_alert(
                health,
                args.failure_rate_threshold,
                workflow_run_url_from_env(),
            ),
        )
        print("Posted Slack runner-failure scan health alert.")
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
