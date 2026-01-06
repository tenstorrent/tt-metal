#!/usr/bin/env python3
"""
Script to analyze Slack export JSON and find tests that fail non-deterministically the most.
Looks for messages containing "the failure is clearly non-deterministic/infra noise"
and extracts job names and failure reasons.
"""

import json
import re
from collections import defaultdict
from typing import Dict, Tuple


def extract_job_name_from_reply(reply_text: str) -> str:
    """Extract job name from reply text.

    Looks for pattern: *Job:* workflow / job_name
    Example: *Job:* t3000-unit-tests / t3k n300 mesh llama3.2-11b-vision tests
    """
    # Pattern to match job name after *Job:*
    pattern = r"\*Job:\*\s*([^\n]+)"
    match = re.search(pattern, reply_text)

    if match:
        job_info = match.group(1).strip()
        # If it contains a slash, extract the part after the slash
        if "/" in job_info:
            parts = job_info.split("/", 1)
            if len(parts) == 2:
                return parts[1].strip()
        return job_info

    # Fallback: try to extract from parent message text
    return "Unknown Job"


def extract_failure_reason(text: str) -> str:
    """Extract the failure reason from the ND message.

    Extracts the text after "the failure is clearly non-deterministic/infra noise"
    (with or without colon) and before "*Workflow:" or end of text
    """
    # Look for the pattern - handle both with and without colon
    # Pattern 1: with colon
    pattern1 = r"the failure is clearly non-deterministic/infra noise:\s*(.+?)(?:\n\*Workflow:|$)"
    match = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)

    if match:
        reason = match.group(1).strip()
        # Clean up the reason - remove trailing newlines and normalize whitespace
        reason = re.sub(r"\s+", " ", reason)
        # Remove trailing periods/newlines
        reason = reason.rstrip(".\n")
        return reason

    # Pattern 2: without colon (e.g., "noise and would inevitably become Case 3")
    pattern2 = r"the failure is clearly non-deterministic/infra noise\s+(.+?)(?:\n\*Workflow:|$)"
    match = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)

    if match:
        reason = match.group(1).strip()
        # Clean up the reason - remove trailing newlines and normalize whitespace
        reason = re.sub(r"\s+", " ", reason)
        # Remove trailing periods/newlines
        reason = reason.rstrip(".\n")
        return reason

    return "Unknown reason"


def parse_message_and_replies(message: dict, job_failures: Dict[str, Dict[str, int]]):
    """Parse a message and its replies to find ND failures."""
    # Check replies for ND failure messages
    if "replies" not in message:
        return

    for reply in message.get("replies", []):
        reply_text = reply.get("text", "")

        # Check if this reply contains the ND failure message
        if "the failure is clearly non-deterministic/infra noise" in reply_text.lower():
            # Extract job name from reply text (it's in the reply itself)
            job_name = extract_job_name_from_reply(reply_text)

            # Extract failure reason from reply text first
            reason = extract_failure_reason(reply_text)

            # Also try to extract from blocks if text extraction failed
            if reason == "Unknown reason" and "blocks" in reply:
                reason = extract_reason_from_blocks(reply.get("blocks", []))

            # Store the failure
            if job_name not in job_failures:
                job_failures[job_name] = defaultdict(int)

            job_failures[job_name][reason] += 1


def extract_reason_from_blocks(blocks: list) -> str:
    """Extract failure reason from Slack blocks structure."""
    # Collect all text elements to reconstruct the full message
    text_parts = []
    found_nd_start = False
    nd_text = ""

    for block in blocks:
        if block.get("type") == "rich_text":
            elements = block.get("elements", [])
            for element in elements:
                if element.get("type") == "rich_text_section":
                    sub_elements = element.get("elements", [])
                    for sub_elem in sub_elements:
                        if sub_elem.get("type") == "text":
                            text = sub_elem.get("text", "")
                            if "the failure is clearly non-deterministic/infra noise" in text.lower():
                                found_nd_start = True
                                nd_text = text
                                # Try to extract reason from this text element
                                # Pattern 1: with colon
                                pattern1 = r"the failure is clearly non-deterministic/infra noise:\s*(.+?)(?:\n|$)"
                                match = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
                                if match:
                                    reason = match.group(1).strip()
                                    reason = re.sub(r"\s+", " ", reason)
                                    return reason.rstrip(".\n")
                                # Pattern 2: without colon
                                pattern2 = r"the failure is clearly non-deterministic/infra noise\s+(.+?)(?:\n|$)"
                                match = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
                                if match:
                                    reason = match.group(1).strip()
                                    reason = re.sub(r"\s+", " ", reason)
                                    return reason.rstrip(".\n")
                            elif found_nd_start:
                                # Continue collecting text until we hit *Workflow:*
                                if "*Workflow:" in text or "Workflow:" in text:
                                    break
                                text_parts.append(text)

    # If we collected parts, join them
    if text_parts:
        reason = " ".join(text_parts).strip()
        reason = re.sub(r"\s+", " ", reason)
        return reason.rstrip(".\n")

    # If we have the ND text but didn't extract, try to extract from it
    if nd_text:
        # Try without colon pattern
        pattern = r"the failure is clearly non-deterministic/infra noise\s+(.+?)(?:\n|$)"
        match = re.search(pattern, nd_text, re.DOTALL | re.IGNORECASE)
        if match:
            reason = match.group(1).strip()
            reason = re.sub(r"\s+", " ", reason)
            return reason.rstrip(".\n")

    return "Unknown reason"


def main():
    json_file = "/home/ubuntu/tt-metal/build_slack_export_with_threads.json"

    print(f"Loading JSON file: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} messages")

    # Dictionary to store: job_name -> {reason: count}
    job_failures = {}

    # Parse all messages
    print("Parsing messages for ND failures...")
    for message in data:
        parse_message_and_replies(message, job_failures)

    print(f"\nFound {len(job_failures)} unique jobs with ND failures")

    # Calculate total failures per job
    job_totals = {}
    for job_name, reasons in job_failures.items():
        job_totals[job_name] = sum(reasons.values())

    # Sort by total failures (descending)
    sorted_jobs = sorted(job_totals.items(), key=lambda x: x[1], reverse=True)

    # Print top 5 jobs
    print("\n" + "=" * 80)
    print("TOP 5 JOBS WITH MOST NON-DETERMINISTIC FAILURES")
    print("=" * 80)

    for i, (job_name, total_failures) in enumerate(sorted_jobs[:5], 1):
        print(f"\n{i}. {job_name}")
        print(f"   Total ND Failures: {total_failures}")
        print(f"   Unique Failure Reasons: {len(job_failures[job_name])}")
        print("\n   Failure Reasons Breakdown:")

        # Sort reasons by count (descending)
        sorted_reasons = sorted(job_failures[job_name].items(), key=lambda x: x[1], reverse=True)

        for reason, count in sorted_reasons:
            print(f"      • ({count}x) {reason}")

        print("-" * 80)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total unique jobs with ND failures: {len(job_failures)}")
    print(f"Total ND failure occurrences: {sum(job_totals.values())}")
    print(f"Average failures per job: {sum(job_totals.values()) / len(job_totals):.2f}")

    # Find most common failure reasons across all jobs
    all_reasons = defaultdict(int)
    for reasons_dict in job_failures.values():
        for reason, count in reasons_dict.items():
            all_reasons[reason] += count

    print(f"\nMost common failure reasons across all jobs:")
    sorted_all_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_all_reasons[:10]:
        print(f"  • ({count}x) {reason}")


if __name__ == "__main__":
    main()
