#!/usr/bin/env python3
"""
Combined script to fetch Slack messages, extract ND failures, and analyze them.
Fetches messages from Slack, filters to only ND failure replies, saves simplified JSON,
and outputs analysis to debug.log.
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Dict, List

# --- CONFIGURATION ---
SECRETS_FILE = "build_secrets.json"
START_DATE_STR = "January 1, 2026"
OUTPUT_FILE = "build_slack_export_with_threads.json"
DEBUG_LOG = "debug.log"


def load_secrets(secrets_file: str) -> dict:
    """Load secrets from JSON file."""
    if not os.path.exists(secrets_file):
        raise FileNotFoundError(
            f"Secrets file '{secrets_file}' not found. " f"Please create it with 'slack_token' and 'channel_id' fields."
        )

    with open(secrets_file, "r", encoding="utf-8") as f:
        secrets = json.load(f)

    if "slack_token" not in secrets or "channel_id" not in secrets:
        raise ValueError(f"Secrets file '{secrets_file}' must contain 'slack_token' and 'channel_id' fields.")

    return secrets


def get_unix_timestamp(date_string):
    """Convert date string to Unix timestamp."""
    dt = datetime.strptime(date_string, "%B %d, %Y")
    return time.mktime(dt.timetuple())


def format_date(timestamp: str) -> str:
    """Convert Unix timestamp to readable format like 'January 4th, 2026'."""
    try:
        ts_float = float(timestamp)
        dt = datetime.fromtimestamp(ts_float)

        day = dt.day
        # Add ordinal suffix (1st, 2nd, 3rd, 4th, etc.)
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        month_name = dt.strftime("%B")
        year = dt.year

        return f"{month_name} {day}{suffix}, {year}"
    except (ValueError, OSError):
        return "Unknown date"


def extract_job_name_from_reply(reply_text: str) -> str:
    """Extract job name from reply text."""
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

    return "Unknown Job"


def extract_failure_reason(text: str) -> str:
    """Extract the failure reason from the ND message."""
    # Find the position of "*Workflow:" or "\nWorkflow:" or "\n*Workflow:"
    workflow_match = re.search(r"\n?\*?Workflow:", text, re.IGNORECASE)
    if workflow_match:
        workflow_pos = workflow_match.start()
    else:
        workflow_pos = len(text)

    text_before_workflow = text[:workflow_pos]

    # Pattern 1: with colon
    pattern1 = r"(the failure is clearly non-deterministic/infra noise:\s*.+)"
    match = re.search(pattern1, text_before_workflow, re.DOTALL | re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        reason = re.sub(r"\s+", " ", reason)
        return reason.rstrip(".\n ")

    # Pattern 2: without colon
    pattern2 = r"(the failure is clearly non-deterministic/infra noise\s+.+)"
    match = re.search(pattern2, text_before_workflow, re.DOTALL | re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        reason = re.sub(r"\s+", " ", reason)
        return reason.rstrip(".\n ")

    return "Unknown reason"


def extract_reason_from_blocks(blocks: list) -> str:
    """Extract failure reason from Slack blocks structure."""
    full_text = ""
    found_nd_start = False

    for block in blocks:
        if block.get("type") == "rich_text":
            elements = block.get("elements", [])
            for element in elements:
                if element.get("type") == "rich_text_section":
                    sub_elements = element.get("elements", [])
                    for sub_elem in sub_elements:
                        if sub_elem.get("type") == "text":
                            text = sub_elem.get("text", "")
                            if (
                                "the failure is clearly non-deterministic/infra noise" in text.lower()
                                and not found_nd_start
                            ):
                                found_nd_start = True
                                full_text = ""

                            if found_nd_start:
                                if "Workflow:" in text or "*Workflow:" in text:
                                    if "Workflow:" in text:
                                        text = text.split("Workflow:")[0]
                                    elif "*Workflow:" in text:
                                        text = text.split("*Workflow:")[0]
                                    if text:
                                        full_text += text
                                    break
                                full_text += text

    if not found_nd_start or not full_text:
        return "Unknown reason"

    # Extract reason using patterns
    pattern1 = r"(the failure is clearly non-deterministic/infra noise:\s*.+)"
    match = re.search(pattern1, full_text, re.DOTALL | re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        reason = re.sub(r"\s+", " ", reason)
        return reason.rstrip(".\n ")

    pattern2 = r"(the failure is clearly non-deterministic/infra noise\s+.+)"
    match = re.search(pattern2, full_text, re.DOTALL | re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        reason = re.sub(r"\s+", " ", reason)
        return reason.rstrip(".\n ")

    return "Unknown reason"


def fetch_replies(client, channel_id, thread_ts):
    """Fetches all replies for a specific thread."""
    replies = []
    cursor = None
    while True:
        try:
            response = client.conversations_replies(channel=channel_id, ts=thread_ts, cursor=cursor, limit=1000)
            time.sleep(0.5)
            # Filter out the parent message
            batch = [msg for msg in response["messages"] if msg.get("ts") != thread_ts]
            replies.extend(batch)

            if response["has_more"]:
                cursor = response["response_metadata"]["next_cursor"]
            else:
                break
        except SlackApiError as e:
            print(f"Error fetching replies for thread {thread_ts}: {e}")
            break
    return replies


def extract_failure_message_from_blocks(blocks: list) -> str:
    """Extract FAILURE MESSAGE content from Slack blocks, including code blocks."""
    failure_message_parts = []
    found_failure_message_label = False
    collecting_content = False

    for block in blocks:
        if block.get("type") == "rich_text":
            elements = block.get("elements", [])
            for element in elements:
                if element.get("type") == "rich_text_section":
                    sub_elements = element.get("elements", [])
                    block_text = ""
                    for sub_elem in sub_elements:
                        if sub_elem.get("type") == "text":
                            block_text += sub_elem.get("text", "")
                        elif sub_elem.get("type") == "link":
                            url = sub_elem.get("url", "")
                            text = sub_elem.get("text", url)
                            block_text += f"<{url}|{text}>"

                    # Check if this block contains "FAILURE MESSAGE:"
                    if "FAILURE MESSAGE:" in block_text.upper():
                        found_failure_message_label = True
                        collecting_content = True
                        # Extract text after "FAILURE MESSAGE:" if any (not just dashes)
                        parts = re.split(r"FAILURE MESSAGE:\s*", block_text, flags=re.IGNORECASE)
                        if len(parts) > 1 and parts[1].strip() and parts[1].strip() not in ["---", "-"]:
                            failure_message_parts.append(parts[1].strip())
                    elif collecting_content:
                        # We're collecting content after FAILURE MESSAGE label
                        # Stop if we hit another field label
                        if any(
                            label in block_text.upper()
                            for label in [
                                "RELEVANT DEVELOPERS:",
                                "RELEVANT FILES:",
                                "NOTES:",
                                "DISCLAIMER:",
                                "COMMITS:",
                            ]
                        ):
                            collecting_content = False
                            break
                        # Skip empty lines and dash-only lines
                        if block_text.strip() and block_text.strip() not in ["---", "-", ""]:
                            failure_message_parts.append(block_text.strip())
                elif element.get("type") == "rich_text_preformatted":
                    # Code block content - collect if we're in FAILURE MESSAGE section
                    if found_failure_message_label and collecting_content:
                        preformatted_elements = element.get("elements", [])
                        code_content = ""
                        for pre_elem in preformatted_elements:
                            if pre_elem.get("type") == "text":
                                code_content += pre_elem.get("text", "")
                        if code_content.strip():
                            failure_message_parts.append(code_content.strip())
                            collecting_content = False  # Code block usually ends the section

    if failure_message_parts:
        # Join all parts with newlines
        result = "\n".join(part for part in failure_message_parts if part.strip())
        return result.strip()

    return ""


def extract_structured_info_from_reply(reply: dict) -> dict:
    """Extract structured information from any reply message."""
    reply_text = reply.get("text", "")

    # If blocks exist, reconstruct full text from blocks
    if "blocks" in reply:
        full_text = ""
        for block in reply.get("blocks", []):
            if block.get("type") == "rich_text":
                elements = block.get("elements", [])
                for element in elements:
                    if element.get("type") == "rich_text_section":
                        sub_elements = element.get("elements", [])
                        for sub_elem in sub_elements:
                            if sub_elem.get("type") == "text":
                                full_text += sub_elem.get("text", "")
                            elif sub_elem.get("type") == "link":
                                # Preserve links
                                url = sub_elem.get("url", "")
                                text = sub_elem.get("text", url)
                                full_text += f"<{url}|{text}>"
                    elif element.get("type") == "rich_text_preformatted":
                        # Handle code blocks (preformatted text)
                        preformatted_elements = element.get("elements", [])
                        for pre_elem in preformatted_elements:
                            if pre_elem.get("type") == "text":
                                full_text += pre_elem.get("text", "")
        if full_text:
            reply_text = full_text

    # Extract structured fields using regex patterns
    def extract_field(field_name: str, text: str) -> str:
        # Pattern to match field name followed by content until next field or end
        # Use DOTALL to match across newlines
        next_fields = r"(?:FULL REPORT|FAILING WORKFLOW|FAILING JOB|FAILING TEST|FAILING RUN|SCENARIO|FAILURE MESSAGE|RELEVANT DEVELOPERS|RELEVANT FILES|NOTES):"
        pattern = rf"{re.escape(field_name)}:\s*(.+?)(?=\n{next_fields}|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            # Remove leading/trailing dashes if the value is just dashes (empty code block)
            if value.strip() == "---" or value.strip() == "-" * len(value.strip()):
                # Look for actual content after the dashes or in code blocks
                # Try to find content that follows the pattern but isn't just dashes
                # This handles cases where FAILURE MESSAGE: is followed by dashes then actual content
                pass  # Will return empty if it's just dashes
            # Preserve markdown links but convert to readable format: text (url)
            value = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2 (\1)", value)
            # Also handle plain URLs
            value = re.sub(r"<([^>]+)>", r"\1", value)
            # If result is just dashes, return empty string
            if value.strip() in ["---", "-", "--"]:
                return ""
            return value
        return ""

    # Extract FULL REPORT link (extract URL but don't include in text)
    full_report_match = re.search(r"FULL REPORT:\s*<([^|>]+)\|([^>]+)>", reply_text, re.IGNORECASE)
    full_report_link = ""
    if full_report_match:
        full_report_link = full_report_match.group(1)

    # Extract other fields (order matters - extract NOTES last since it's at the end)
    failing_workflow = extract_field("FAILING WORKFLOW", reply_text)
    failing_job = extract_field("FAILING JOB", reply_text)
    failing_test = extract_field("FAILING TEST", reply_text)
    failing_run = extract_field("FAILING RUN", reply_text)
    scenario = extract_field("SCENARIO", reply_text)

    # Special handling for FAILURE MESSAGE - extract from blocks if available to get code block content
    failure_message = ""
    if "blocks" in reply:
        failure_message = extract_failure_message_from_blocks(reply.get("blocks", []))

    # Fall back to regex extraction if blocks didn't yield a result
    if not failure_message:
        failure_message = extract_field("FAILURE MESSAGE", reply_text)

    relevant_developers = extract_field("RELEVANT DEVELOPERS", reply_text)
    relevant_files = extract_field("RELEVANT FILES", reply_text)
    notes = extract_field("NOTES", reply_text)

    # Extract timestamp and format date
    timestamp = reply.get("ts", "")
    formatted_date = format_date(timestamp)

    # Split full_text by newlines into a list (preserve all lines including empty ones)
    full_text_lines = reply_text.split("\n")

    # Build structured result
    result = {
        "date": formatted_date,
        "timestamp": timestamp,
        "full_report_link": full_report_link,
        "failing_workflow": failing_workflow,
        "failing_job": failing_job,
        "failing_test": failing_test,
        "failing_run": failing_run,
        "scenario": scenario,
        "failure_message": failure_message,
        "relevant_developers": relevant_developers,
        "relevant_files": relevant_files,
        "notes": notes,
        "full_text": full_text_lines,  # List of strings, one per line
    }

    return result


def extract_nd_failure_from_reply(reply: dict) -> dict:
    """Extract ND failure information from a reply. Returns None if not an ND failure."""
    reply_text = reply.get("text", "")

    # Reconstruct full text from blocks if available (for scenario extraction)
    full_text_for_extraction = reply_text
    if "blocks" in reply:
        full_text = ""
        for block in reply.get("blocks", []):
            if block.get("type") == "rich_text":
                elements = block.get("elements", [])
                for element in elements:
                    if element.get("type") == "rich_text_section":
                        sub_elements = element.get("elements", [])
                        for sub_elem in sub_elements:
                            if sub_elem.get("type") == "text":
                                full_text += sub_elem.get("text", "")
                            elif sub_elem.get("type") == "link":
                                url = sub_elem.get("url", "")
                                text = sub_elem.get("text", url)
                                full_text += f"<{url}|{text}>"
        if full_text:
            full_text_for_extraction = full_text

    # Extract scenario field to check for "Failure likely outside tt-metal"
    def extract_field(field_name: str, text: str) -> str:
        next_fields = r"(?:FULL REPORT|FAILING WORKFLOW|FAILING JOB|FAILING TEST|FAILING RUN|SCENARIO|FAILURE MESSAGE|RELEVANT DEVELOPERS|RELEVANT FILES|NOTES):"
        pattern = rf"{re.escape(field_name)}:\s*(.+?)(?=\n{next_fields}|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            value = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2 (\1)", value)
            value = re.sub(r"<([^>]+)>", r"\1", value)
            return value
        return ""

    scenario = extract_field("SCENARIO", full_text_for_extraction)
    is_scenario_nd = "failure likely outside tt-metal" in scenario.lower() if scenario else False

    # Check if this reply contains the ND failure message (old type)
    is_text_nd = "the failure is clearly non-deterministic/infra noise" in reply_text.lower()

    # Must be one of the two ND failure types
    if not is_text_nd and not is_scenario_nd:
        return None

    # Extract job name - different logic for each type
    if is_scenario_nd:
        # For scenario-based ND failures, extract from FAILING JOB field
        failing_job = extract_field("FAILING JOB", full_text_for_extraction)
        if failing_job:
            # Extract job name from "workflow / job" format if present
            if "/" in failing_job:
                parts = failing_job.split("/", 1)
                if len(parts) == 2:
                    job_name = parts[1].strip()
                else:
                    job_name = failing_job.strip()
            else:
                job_name = failing_job.strip()
        else:
            job_name = "Unknown Job"
    else:
        # For text-based ND failures, use existing extraction
        job_name = extract_job_name_from_reply(reply_text)

    # Extract failure reason - different logic for each type
    if is_scenario_nd:
        # For scenario-based ND failures, use failure_message or notes
        failure_message = extract_field("FAILURE MESSAGE", full_text_for_extraction)
        notes = extract_field("NOTES", full_text_for_extraction)

        if failure_message:
            reason = failure_message
        elif notes:
            reason = notes
        else:
            reason = "Failure likely outside tt-metal"
    else:
        # For text-based ND failures, use existing extraction
        reason_from_text = extract_failure_reason(reply_text)
        reason_from_blocks = "Unknown reason"

        if "blocks" in reply:
            reason_from_blocks = extract_reason_from_blocks(reply.get("blocks", []))

        # Prefer the longer reason (more likely to be complete)
        if reason_from_blocks != "Unknown reason" and len(reason_from_blocks) > len(reason_from_text):
            reason = reason_from_blocks
        elif reason_from_text != "Unknown reason":
            reason = reason_from_text
        else:
            reason = reason_from_blocks

    # Extract timestamp and format date
    timestamp = reply.get("ts", "")
    formatted_date = format_date(timestamp)

    return {"job_name": job_name, "failure_reason": reason, "date": formatted_date, "timestamp": timestamp}


def download_and_extract_messages():
    """Download messages from Slack and extract structured information from all replies."""
    # Load secrets from file
    secrets = load_secrets(SECRETS_FILE)
    SLACK_TOKEN = secrets["slack_token"]
    CHANNEL_ID = secrets["channel_id"]

    # Delete output file if it already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Deleting existing {OUTPUT_FILE}...")
        os.remove(OUTPUT_FILE)

    client = WebClient(token=SLACK_TOKEN)
    oldest_timestamp = get_unix_timestamp(START_DATE_STR)

    all_messages = []
    cursor = None

    print(f"Fetching messages from {START_DATE_STR}...")

    # First pass: collect all messages without fetching replies
    while True:
        try:
            response = client.conversations_history(
                channel=CHANNEL_ID, oldest=str(oldest_timestamp), cursor=cursor, limit=1000
            )

            messages = response["messages"]
            all_messages.extend(messages)

            print(f"Retrieved {len(messages)} parent messages... (Total so far: {len(all_messages)})")

            if response["has_more"]:
                cursor = response["response_metadata"]["next_cursor"]
            else:
                break

        except SlackApiError as e:
            if e.response.status_code == 429:
                delay = int(e.response.headers.get("Retry-After", 10))
                print(f"Rate limited. Sleeping for {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Error fetching conversations: {e}")
                break

    # Count messages with replies
    messages_with_replies = [msg for msg in all_messages if "thread_ts" in msg and msg.get("reply_count", 0) > 0]
    total_with_replies = len(messages_with_replies)

    print(f"\nFound {total_with_replies} messages with replies. Fetching replies...")

    # Second pass: fetch replies and extract structured information
    all_replies_structured = []
    nd_failures = []
    current_index = 0

    for msg in all_messages:
        if "thread_ts" in msg and msg.get("reply_count", 0) > 0:
            current_index += 1
            print(f"  --> Fetching replies from notification {current_index}/{total_with_replies} (ts: {msg['ts']})...")
            replies = fetch_replies(client, CHANNEL_ID, msg["thread_ts"])

            # Extract structured information from all replies
            for reply in replies:
                structured_data = extract_structured_info_from_reply(reply)
                all_replies_structured.append(structured_data)

                # Also check for ND failures for analysis
                nd_failure_data = extract_nd_failure_from_reply(reply)
                if nd_failure_data:
                    nd_failures.append(nd_failure_data)

    # Save all replies in structured format
    print(f"\nFound {len(all_replies_structured)} total replies. Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_replies_structured, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(all_replies_structured)} structured reply records to {OUTPUT_FILE}")
    print(f"Found {len(nd_failures)} ND failure replies for analysis.")
    return nd_failures


def analyze_failures(nd_failures: List[dict], log_file: str):
    """Analyze ND failures and write results to log file."""
    # Build job_failures structure: job_name -> {reason: [dates]}
    job_failures = {}

    for failure in nd_failures:
        job_name = failure["job_name"]
        reason = failure["failure_reason"]
        date = failure["date"]

        if job_name not in job_failures:
            job_failures[job_name] = defaultdict(list)

        job_failures[job_name][reason].append(date)

    # Redirect output to log file
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"ND Failure Analysis\n")
        log.write(f"Total ND failure records: {len(nd_failures)}\n")
        log.write(f"Found {len(job_failures)} unique jobs with ND failures\n\n")

        # Calculate total failures per job
        job_totals = {}
        for job_name, reasons in job_failures.items():
            job_totals[job_name] = sum(len(dates) for dates in reasons.values())

        # Sort by total failures (descending)
        sorted_jobs = sorted(job_totals.items(), key=lambda x: x[1], reverse=True)

        # Print top 5 jobs
        log.write("=" * 80 + "\n")
        log.write("TOP 5 JOBS WITH MOST NON-DETERMINISTIC FAILURES\n")
        log.write("=" * 80 + "\n\n")

        for i, (job_name, total_failures) in enumerate(sorted_jobs[:5], 1):
            log.write(f"{i}. {job_name}\n")
            log.write(f"   Total ND Failures: {total_failures}\n")
            log.write(f"   Unique Failure Reasons: {len(job_failures[job_name])}\n")
            log.write("\n   Failure Reasons Breakdown:\n")

            # Sort reasons by count (descending)
            sorted_reasons = sorted(job_failures[job_name].items(), key=lambda x: len(x[1]), reverse=True)

            for reason, dates in sorted_reasons:
                count = len(dates)
                # Format dates for display
                if count == 1:
                    date_str = dates[0]
                elif count <= 5:
                    date_str = ", ".join(dates)
                else:
                    date_str = ", ".join(dates[:3]) + f", and {count - 3} more"

                log.write(f"      • ({count}x) {reason}\n")
                log.write(f"        Dates: {date_str}\n")

            log.write("-" * 80 + "\n")

        # Summary statistics
        log.write("\n" + "=" * 80 + "\n")
        log.write("SUMMARY STATISTICS\n")
        log.write("=" * 80 + "\n")
        log.write(f"Total unique jobs with ND failures: {len(job_failures)}\n")
        log.write(f"Total ND failure occurrences: {sum(job_totals.values())}\n")
        if len(job_totals) > 0:
            log.write(f"Average failures per job: {sum(job_totals.values()) / len(job_totals):.2f}\n")

        # Find most common failure reasons across all jobs
        all_reasons = defaultdict(int)
        for reasons_dict in job_failures.values():
            for reason, dates in reasons_dict.items():
                all_reasons[reason] += len(dates)

        log.write(f"\nMost common failure reasons across all jobs:\n")
        sorted_all_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_all_reasons[:10]:
            log.write(f"  • ({count}x) {reason}\n")

    print(f"Analysis complete! Results written to {log_file}")


def main():
    """Main function to fetch Slack data and analyze ND failures."""
    # Download messages and extract structured information from all replies
    # Also extract ND failures for analysis
    nd_failures = download_and_extract_messages()

    # Analyze failures and write to debug.log
    if nd_failures:
        analyze_failures(nd_failures, DEBUG_LOG)
    else:
        print("No ND failures found. Nothing to analyze.")


if __name__ == "__main__":
    main()
