#!/usr/bin/env python3
"""
Extract tt-triage output sections from GitHub Actions CI logs.

Parses triage-runs.csv, downloads job logs via `gh api`, extracts
the triage section from each, and saves them to triage_outputs/.
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = "tenstorrent/tt-metal"
CSV_PATH = Path(__file__).resolve().parents[2] / "triage-runs.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "triage_outputs"

# Triage output starts with the first script output (dump_configuration.py)
# and ends before docker cleanup / post-job steps.
TRIAGE_START_PATTERNS = [
    "dump_configuration.py:",
    "check_arc.py:",
    "check_noc_status.py:",
    "check_eth_status.py:",
]
TRIAGE_END_PATTERNS = [
    "Total reclaimed space:",
    "Cleaning up orphan processes",
    "##[group]Post job cleanup",
    "##[group]Run actions/",
    "Post job cleanup",
    "##[group]Complete job",
]


def parse_csv():
    """Parse CSV and return dict of job_id -> metadata (deduplicated by job URL)."""
    jobs = {}
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link = row["github_job_link"]
            m = re.search(r"runs/(\d+)/job/(\d+)", link)
            if not m:
                continue
            run_id, job_id = m.group(1), m.group(2)
            url = f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}"
            if job_id not in jobs:
                jobs[job_id] = {
                    "job_id": job_id,
                    "run_id": run_id,
                    "url": url,
                    "test_function": row["test_function"],
                    "host_name": row["host_name"],
                    "job_name": row["job_name"],
                    "test_start_ts": row["test_start_ts"],
                }
    return jobs


def download_job_log(job_id):
    """Download raw log for a job via gh api. Returns log text or None."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"  [ERROR] gh api failed for job {job_id}: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] job {job_id}")
        return None


def strip_ansi(text):
    """Remove ANSI escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def strip_timestamp(line):
    """Remove GitHub Actions timestamp prefix (e.g., 2026-04-14T05:51:22.9192425Z)."""
    return re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?", "", line)


def extract_triage_section(log_text):
    """Extract the triage output section from a full CI log.

    Handles two cases:
    1. Normal triage output: starts with dump_configuration.py or other script names
    2. Triage crash: tt-triage.py crashes with a traceback before producing script output
       (e.g., exalens init failure, UMD failure, ARC core failure)
    """
    lines = log_text.split("\n")
    start_idx = None
    end_idx = None

    # Find start: first line matching a triage script output
    for i, line in enumerate(lines):
        clean = strip_ansi(strip_timestamp(line))
        for pattern in TRIAGE_START_PATTERNS:
            if clean.strip() == pattern or clean.strip().startswith(pattern):
                start_idx = i
                break
        if start_idx is not None:
            break

    if start_idx is None:
        # Fallback: look for triage crash (traceback after tt-triage.py invocation)
        triage_invoke_idx = None
        for i, line in enumerate(lines):
            clean = strip_ansi(line)
            if "tt-triage.py" in clean and ("Executing command" in clean or "tools/tt-triage.py" in clean):
                triage_invoke_idx = i
        # Look for traceback/error after the last triage invocation
        if triage_invoke_idx is not None:
            for i in range(triage_invoke_idx, min(triage_invoke_idx + 200, len(lines))):
                clean = strip_ansi(lines[i])
                if "Traceback" in clean or "RuntimeError" in clean or "Error" in clean or "Exception" in clean:
                    # Capture from 2 lines before traceback start to give context
                    start_idx = max(triage_invoke_idx, i - 2)
                    break
        if start_idx is None:
            return None

    # Find end: first line after start matching an end pattern
    for i in range(start_idx + 1, len(lines)):
        clean = strip_ansi(strip_timestamp(lines[i]))
        for pattern in TRIAGE_END_PATTERNS:
            if pattern in clean:
                end_idx = i
                break
        if end_idx is not None:
            break

    if end_idx is None:
        # Take up to 3000 lines after start as fallback
        end_idx = min(start_idx + 3000, len(lines))

    # Clean up the extracted section
    section_lines = []
    for line in lines[start_idx:end_idx]:
        cleaned = strip_ansi(strip_timestamp(line))
        section_lines.append(cleaned)

    return "\n".join(section_lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Parsing CSV...")
    jobs = parse_csv()
    print(f"Found {len(jobs)} unique jobs")

    # Check which jobs already have output files (for resumability)
    existing = {f.stem for f in OUTPUT_DIR.glob("*.txt") if f.stem != "index"}
    remaining = {jid: meta for jid, meta in jobs.items() if jid not in existing}
    print(f"Already downloaded: {len(existing)}, remaining: {len(remaining)}")

    # Download and extract
    failed = []
    no_triage = []
    for i, (job_id, meta) in enumerate(remaining.items()):
        print(f"[{i+1}/{len(remaining)}] Job {job_id} ({meta['test_function']} on {meta['host_name']})")

        log_text = download_job_log(job_id)
        if log_text is None:
            failed.append(job_id)
            continue

        triage_section = extract_triage_section(log_text)
        if triage_section is None:
            print(f"  [NO TRIAGE] No triage section found")
            no_triage.append(job_id)
            # Save a marker file so we don't retry
            (OUTPUT_DIR / f"{job_id}.txt").write_text("[NO TRIAGE SECTION FOUND]\n")
            continue

        (OUTPUT_DIR / f"{job_id}.txt").write_text(triage_section)
        line_count = len(triage_section.split("\n"))
        print(f"  Extracted {line_count} lines")

        # Rate limit: ~1 request per second
        time.sleep(0.5)

    # Save index with ALL jobs (including previously downloaded)
    index = {}
    for job_id, meta in jobs.items():
        output_file = OUTPUT_DIR / f"{job_id}.txt"
        index[job_id] = {
            **meta,
            "has_output": output_file.exists() and output_file.read_text().strip() != "[NO TRIAGE SECTION FOUND]",
        }

    with open(OUTPUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone!")
    print(f"  Successful: {len(jobs) - len(failed) - len(no_triage)}")
    print(f"  No triage section: {len(no_triage)}")
    print(f"  Failed to download: {len(failed)}")
    if failed:
        print(f"  Failed job IDs: {failed}")


if __name__ == "__main__":
    main()
