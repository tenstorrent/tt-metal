#!/usr/bin/env python3
"""
Extract error messages from build_slack_export_with_threads.json

When EXTRACT_ALL_ERRORS is False (default):
- Only includes non-deterministic errors:
  - Errors from cancelled analysis runs (Auto-triage cancelled) that have "FAILURE MESSAGE:"
  - Errors with scenario "Failure likely outside tt-metal" that have "FAILURE MESSAGE:"

When EXTRACT_ALL_ERRORS is True:
- Extracts all error messages from "FAILURE MESSAGE:" field regardless of determinism

All errors are extracted from the "FAILURE MESSAGE:" field as-is, without any truncation or cleanup.
Entries without "FAILURE MESSAGE:" are skipped.
"""

import json
import sys
import re

# Set to True to extract all errors, False to only extract non-deterministic errors
EXTRACT_ALL_ERRORS = True


def is_non_deterministic(entry):
    """Check if an entry represents a non-deterministic error."""
    full_text = entry.get("full_text", [])
    scenario = entry.get("scenario", "")

    # Check if it's an auto-triage cancelled message
    if full_text and len(full_text) > 0 and full_text[0] == "Auto-triage cancelled:":
        return True

    # Check if scenario is "Failure likely outside tt-metal"
    if scenario == "Failure likely outside tt-metal":
        return True

    return False


def extract_error_message(entry):
    """Extract error message from failure_message field."""
    failure_message = entry.get("failure_message", "")

    # Return the failure message if it exists and is not empty or just dashes
    if failure_message and failure_message.strip() and failure_message.strip() not in ["---", "-"]:
        return failure_message.strip()

    return None


def extract_failing_run_url(entry):
    """Extract the URL from the failing_run field."""
    failing_run = entry.get("failing_run", "")

    if not failing_run:
        return None

    # Extract URL from parentheses: "Run #123 (description) (https://...)"
    # Pattern matches URLs in parentheses
    url_pattern = r"\(https?://[^\)]+\)"
    match = re.search(url_pattern, failing_run)

    if match:
        # Remove the parentheses
        url = match.group(0)[1:-1]  # Remove first and last character (parentheses)
        return url

    return None


def main():
    input_file = "build_slack_export_with_threads.json"
    output_file = "all_errors.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    errors = []
    skipped = 0

    for entry in data:
        # Filter based on EXTRACT_ALL_ERRORS flag
        if not EXTRACT_ALL_ERRORS:
            # Only process non-deterministic errors
            if not is_non_deterministic(entry):
                skipped += 1
                continue

        error_msg = extract_error_message(entry)
        if error_msg:
            # Extract failing run URL
            failing_run_url = extract_failing_run_url(entry)
            # Save as tuple: [error_message, failing_run_url]
            # Use None if URL not found
            errors.append([error_msg, failing_run_url])
        else:
            skipped += 1

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    mode_str = "all" if EXTRACT_ALL_ERRORS else "non-deterministic"
    print(f"Extracted {len(errors)} {mode_str} error messages")
    print(f"Skipped {skipped} entries")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
