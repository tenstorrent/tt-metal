# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repeat the shared prompt suite using the validated device top-k32 mode."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    baseline = args.events.read_text().splitlines()
    results = []
    for row in control["results"]:
        response = requests.post("http://localhost:8000/v1/chat/completions", json=row["request"], timeout=180)
        response.raise_for_status()
        value = response.json()
        results.append(
            dict(
                id=row["id"],
                request=row["request"],
                response=value,
                previous_stage_exact_text=value["choices"][0]["message"]["content"]
                == row["response"]["choices"][0]["message"]["content"],
            )
        )
    events = [json.loads(line) for line in args.events.read_text().splitlines()[len(baseline) :]]
    bad = [
        event
        for event in events
        if event["event"] == "host_sampling_compatibility" or event["event"] == "decode_submit" and not event["sample"]
    ]
    args.output.write_text(
        json.dumps(
            dict(
                control=str(args.control),
                sampling_mode=control["sampling_mode"],
                results=results,
                forbidden_events=bad,
                event_count=len(events),
                status="pass" if not bad else "fail",
            ),
            indent=2,
        )
        + "\n"
    )
    if bad:
        raise AssertionError("Device qualitative suite used host sampling")
    print(f"PASS: {len(results)} on-device qualitative responses")


if __name__ == "__main__":
    main()
