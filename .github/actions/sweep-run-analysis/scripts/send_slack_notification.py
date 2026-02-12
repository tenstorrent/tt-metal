#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build Slack Block Kit message and send notification"""

import json
import os
import sys
import time
from typing import Optional

import requests

# Configuration
RESULTS_FILE = os.environ.get("RESULTS_FILE", "sweep_results.json")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")  # Alternative: OAuth bot token
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "")  # Required with bot token
CONCLUSION = os.environ.get("CONCLUSION", "success")  # success, failure, cancelled
GITHUB_RUN_ID = os.environ.get("GITHUB_RUN_ID", "")
SUPERSET_BASE_URL = "https://superset.tenstorrent.com/superset/dashboard/lead-models-sweep-run/"
GITHUB_ACTIONS_URL = f"https://github.com/tenstorrent/tt-metal/actions/runs/{GITHUB_RUN_ID}"

# Slack API endpoints
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds


def load_results() -> dict:
    """Load results from JSON file."""
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found: {RESULTS_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        return json.load(f)


def determine_status(results: dict, conclusion: str) -> tuple[str, str, str]:
    """Determine status emoji, color, and text based on results and conclusion.

    Returns: (emoji, status_text, header_suffix)
    """
    # Check for infrastructure failure or cancellation first
    if conclusion == "cancelled":
        return ":white_circle:", "Cancelled", ""

    if conclusion == "failure":
        # Check if we have any test results - if not, it's infrastructure failure
        if results["run_summary"]["test_count"] == 0:
            return ":red_circle:", "Failed", ""

    # Check for regressions
    has_pass_rate_regressions = len(results.get("pass_rate_regressions", [])) > 0
    has_perf_regressions = (
        len(results.get("perf_regressions_by_op", [])) > 0 or len(results.get("perf_regressions_by_test", [])) > 0
    )
    has_regressions = has_pass_rate_regressions or has_perf_regressions

    if has_regressions:
        return ":large_yellow_circle:", "Complete", " (regressions detected)"

    return ":large_green_circle:", "Complete", ""


def format_duration(ns) -> str:
    """Format nanoseconds to human-readable string."""
    if ns is None:
        return "N/A"
    ns = float(ns)  # Handle Decimal/string from JSON
    us = ns / 1000
    if us < 1000:
        return f"{us:.1f}μs"
    ms = us / 1000
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms/1000:.2f}s"


def build_header_block(emoji: str, status_text: str, suffix: str) -> dict:
    """Build the header block."""
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{emoji} Lead Models Sweeps {status_text}{suffix}",
            "emoji": True,
        },
    }


def build_context_block(results: dict) -> dict:
    """Build the context block with architecture, commit, branch."""
    summary = results["run_summary"]

    commit_link = ""
    if summary.get("git_sha"):
        commit_link = f"<https://github.com/tenstorrent/tt-metal/commit/{summary['git_sha']}|{summary['git_sha'][:7]}>"
    else:
        commit_link = "unknown"

    elements = [
        {
            "type": "mrkdwn",
            "text": f"*Branch:* {summary.get('git_branch', 'unknown')} | *Commit:* {commit_link}",
        }
    ]

    return {"type": "context", "elements": elements}


def build_testing_mode_note_block() -> dict:
    """Build the testing mode note shown below the context block."""
    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "_Still in preview/test mode.",
            }
        ],
    }


def build_overall_results_block(results: dict) -> dict:
    """Build the overall results section."""
    summary = results["run_summary"]

    pass_rate_text = f"{summary['pass_pct']}%"
    if results["comparison_available"] and summary.get("prev_pass_pct") is not None:
        if summary["pass_pct"] == summary["prev_pass_pct"]:
            pass_rate_text += " (no change)"
        else:
            pass_rate_text += f" (was {summary['prev_pass_pct']}%)"

    text = f"*Overall Results:* {summary['test_count']:,} tests | {pass_rate_text} pass rate"

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }


def build_pass_rate_regressions_block(regressions: list[dict]) -> list[dict]:
    """Build the pass rate regressions section."""
    if not regressions:
        return []

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*:warning: Pass Rate Regressions*"},
        }
    ]

    # Build table
    lines = ["```"]
    lines.append(f"{'Module':<35} {'Prev':>8} {'Now':>8} {'Delta':>8}")
    for reg in regressions[:10]:  # Limit to top 10
        module = reg["module"][:35]
        prev = f"{float(reg['prev']):.1f}%" if reg.get("prev") is not None else "N/A"
        current = f"{float(reg['current']):.1f}%" if reg.get("current") is not None else "N/A"
        delta = f"{float(reg['delta']):+.1f}%" if reg.get("delta") is not None else "N/A"
        lines.append(f"{module:<35} {prev:>8} {current:>8} {delta:>8}")
    lines.append("```")

    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        }
    )

    return blocks


def build_perf_regressions_block(op_regressions: list[dict], test_regressions: list[dict]) -> list[dict]:
    """Build the performance regressions section."""
    if not op_regressions and not test_regressions:
        return []

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*:stopwatch: Performance Regressions (>15%)*"},
        }
    ]

    # Operation-level regressions
    if op_regressions:
        lines = ["_By Operation (avg):_", "```"]
        lines.append(f"{'Operation':<30} {'Prev':>12} {'Now':>12} {'Change':>10}")
        for reg in op_regressions[:5]:  # Limit to top 5
            op_name = reg["op_name"][:30] if reg.get("op_name") else "unknown"
            prev = format_duration(reg.get("prev_ns"))
            current = format_duration(reg.get("current_ns"))
            change = f"+{float(reg['pct_change']):.1f}%" if reg.get("pct_change") is not None else "N/A"
            lines.append(f"{op_name:<30} {prev:>12} {current:>12} {change:>10}")
        lines.append("```")

        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(lines)},
            }
        )

    # Individual test regressions
    if test_regressions:
        lines = ["_Individual Tests:_", "```"]
        lines.append(f"{'Test':<50} {'Prev':>10} {'Now':>10} {'Change':>8}")
        for reg in test_regressions[:5]:  # Limit to top 5
            # Truncate test name and add model info
            test_name = reg.get("full_test_name", "unknown")
            model = reg.get("model_name", "")
            if model:
                display_name = f"{test_name[:35]}[{model[:10]}]"
            else:
                display_name = test_name[:50]
            prev = format_duration(reg.get("prev_ns"))
            current = format_duration(reg.get("current_ns"))
            change = f"+{float(reg['pct_change']):.1f}%" if reg.get("pct_change") is not None else "N/A"
            lines.append(f"{display_name:<50} {prev:>10} {current:>10} {change:>8}")
        lines.append("```")

        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(lines)},
            }
        )

    return blocks


def build_models_affected_block(models_affected: list[dict]) -> list[dict]:
    """Build the models affected section."""
    if not models_affected:
        return []

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*:robot_face: Models Affected*"},
        }
    ]

    lines = []
    for model in models_affected[:10]:  # Limit to top 10
        parts = []
        if model.get("new_failures"):
            parts.append(f"{model['new_failures']} new failures")
        if model.get("perf_regressions"):
            parts.append(f"{model['perf_regressions']} perf regressions")

        if parts:
            lines.append(f"- `{model['model_name']}` ({', '.join(parts)})")

    if lines:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(lines)},
            }
        )

    return blocks


def build_models_tested_block(models_tested: list[str]) -> list[dict]:
    """Build the models tested section (shown when no regressions)."""
    if not models_tested:
        return []

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:white_check_mark: Models Tested*\n{', '.join(models_tested)}",
            },
        }
    ]

    return blocks


def build_baseline_note_block() -> dict:
    """Build the baseline established note for first runs."""
    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "_Baseline established. Future runs will include regression analysis._",
            }
        ],
    }


def build_comparison_unavailable_note_block() -> dict:
    """Build the comparison unavailable note."""
    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "_Note: Comparison with previous run unavailable. Regression analysis not performed._",
            }
        ],
    }


def build_infrastructure_failure_block() -> list[dict]:
    """Build blocks for infrastructure failure."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Workflow failed before completion. Check GitHub Actions for details.\n\n{GITHUB_ACTIONS_URL}",
            },
        }
    ]


def build_cancelled_block() -> list[dict]:
    """Build blocks for cancelled workflow."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Workflow was cancelled before completion.\n\n{GITHUB_ACTIONS_URL}",
            },
        }
    ]


def build_superset_link_block(run_id: Optional[int]) -> dict:
    """Build the Superset dashboard link block.

    Uses gh_run_number (GITHUB_RUN_ID) since it's available immediately,
    whereas run_id (database PK) requires Airflow ingestion first.
    """
    if GITHUB_RUN_ID:
        # Use GitHub run ID - works immediately, dashboard shows data after ingestion
        url = f"{SUPERSET_BASE_URL}?gh_run_number={GITHUB_RUN_ID}"
        text = f"<{url}|View in Superset>"
    else:
        text = f"<{GITHUB_ACTIONS_URL}|View in GitHub Actions>"

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    }


def build_slack_message(results: dict, conclusion: str) -> dict:
    """Build the complete Slack message payload."""
    emoji, status_text, suffix = determine_status(results, conclusion)

    blocks = []

    # Header
    blocks.append(build_header_block(emoji, status_text, suffix))

    # Handle infrastructure failure
    if conclusion == "failure" and results["run_summary"]["test_count"] == 0:
        blocks.append(build_context_block(results))
        blocks.append(build_testing_mode_note_block())
        blocks.extend(build_infrastructure_failure_block())
        return {"blocks": blocks}

    # Handle cancellation
    if conclusion == "cancelled":
        blocks.append(build_context_block(results))
        blocks.append(build_testing_mode_note_block())
        blocks.extend(build_cancelled_block())
        return {"blocks": blocks}

    # Normal flow - add context and results
    blocks.append(build_context_block(results))
    blocks.append(build_testing_mode_note_block())
    blocks.append(build_overall_results_block(results))

    # Add divider
    blocks.append({"type": "divider"})

    # Check for regressions
    has_pass_rate_regressions = len(results.get("pass_rate_regressions", [])) > 0
    has_perf_regressions = (
        len(results.get("perf_regressions_by_op", [])) > 0 or len(results.get("perf_regressions_by_test", [])) > 0
    )
    has_regressions = has_pass_rate_regressions or has_perf_regressions

    # Regression sections
    if has_pass_rate_regressions:
        blocks.extend(build_pass_rate_regressions_block(results["pass_rate_regressions"]))

    if has_perf_regressions:
        blocks.extend(
            build_perf_regressions_block(
                results.get("perf_regressions_by_op", []),
                results.get("perf_regressions_by_test", []),
            )
        )

    # Models affected (if regressions) or Models tested (if no regressions)
    if results.get("models_affected"):
        blocks.extend(build_models_affected_block(results["models_affected"]))
    elif not has_regressions and results.get("models_tested"):
        blocks.extend(build_models_tested_block(results["models_tested"]))

    # Add notes for special cases
    if not results.get("comparison_available"):
        if results["run_summary"]["test_count"] > 0:
            # First run - baseline established
            blocks.append(build_baseline_note_block())

    # Superset link
    blocks.append({"type": "divider"})
    blocks.append(build_superset_link_block(results.get("run_id")))

    return {"blocks": blocks}


def send_slack_message_webhook(payload: dict) -> bool:
    """Send message using incoming webhook URL."""
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                SLACK_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                print("Slack notification sent successfully via webhook")
                return True

            print(f"Slack API returned status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2

    return False


def send_slack_message_bot_token(payload: dict, channel: str) -> bool:
    """Send message using bot token and chat.postMessage API."""
    # Add channel to payload for chat.postMessage
    api_payload = {"channel": channel, **payload}

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                SLACK_POST_MESSAGE_URL,
                json=api_payload,
                headers={
                    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

            result = response.json()

            if response.status_code == 200 and result.get("ok"):
                print(f"Slack notification sent successfully via bot token to {channel}")
                return True

            error = result.get("error", "unknown error")
            print(f"Slack API error: {error}")

            # Don't retry on authentication or channel errors
            if error in ["invalid_auth", "channel_not_found", "not_in_channel"]:
                print(f"ERROR: {error} - check SLACK_BOT_TOKEN and SLACK_CHANNEL")
                return False

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2

    return False


def send_slack_message(payload: dict) -> bool:
    """Send message to Slack using webhook or bot token."""
    # Check which authentication method is available
    if SLACK_WEBHOOK_URL:
        print("Using webhook URL for Slack notification")
        return send_slack_message_webhook(payload)

    if SLACK_BOT_TOKEN and SLACK_CHANNEL:
        print(f"Using bot token for Slack notification to {SLACK_CHANNEL}")
        return send_slack_message_bot_token(payload, SLACK_CHANNEL)

    # No authentication configured - print payload for debugging
    print("WARNING: No Slack authentication configured")
    print("Set either SLACK_WEBHOOK_URL or (SLACK_BOT_TOKEN + SLACK_CHANNEL)")
    print("\nWould have sent:")
    print(json.dumps(payload, indent=2))
    return True


def main():
    print(f"Loading results from {RESULTS_FILE}")
    results = load_results()

    print(f"Building Slack message (conclusion={CONCLUSION})")
    payload = build_slack_message(results, CONCLUSION)

    # Debug: print the payload
    print("Slack payload:")
    print(json.dumps(payload, indent=2))

    success = send_slack_message(payload)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
