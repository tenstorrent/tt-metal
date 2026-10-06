# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attribute native C1 TTFT using existing host-only completion observations."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads(args.events.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    assert not observed["errors"], observed["errors"]
    assert not observed["pending"], observed["pending"]
    assert benchmark["max_concurrency"] == 1
    events = observed["events"]
    completions = {event["submission_id"]: event for event in events if event["event"] == "completion"}
    prefills = [event for event in events if event["event"] == "dispatch" and event["phase"] == "prefill"]
    rows = []
    for index, (start, ttft) in enumerate(zip(benchmark["start_times"], benchmark["ttfts"])):
        matches = [event for event in prefills if start * 1e9 <= event["timestamp_ns"] <= (start + ttft) * 1e9]
        assert len(matches) == 1, (index, matches)
        dispatch = matches[0]
        completion = completions[dispatch["submission_id"]]
        assert len(dispatch["request_ids"]) == 1
        request = dispatch["request_ids"][0]
        decodes = [event for event in events if event["phase"] == "decode" and event["request_ids"] == [request]]
        row = dict(
            index=index,
            request_id=request,
            ttft_ms=ttft * 1000,
            client_to_prefill_dispatch_ms=(dispatch["timestamp_ns"] / 1e9 - start) * 1000,
            runner_prefill_ms=(completion["timestamp_ns"] - dispatch["timestamp_ns"]) / 1e6,
            prefill_completion_to_client_token_ms=(start + ttft - completion["timestamp_ns"] / 1e9) * 1000,
            decode_completions=sum(event["event"] == "completion" for event in decodes),
            first_decode_dispatch_after_prefill_ms=min(
                (
                    (event["timestamp_ns"] - completion["timestamp_ns"]) / 1e6
                    for event in decodes
                    if event["event"] == "dispatch"
                ),
                default=None,
            ),
        )
        assert row["prefill_completion_to_client_token_ms"] >= 0, row
        rows.append(row)
    report = dict(
        scope="C1 host-clock decomposition; runner prefill includes host work and device wait, not pure device time",
        clock="time.perf_counter on the same host; native start_times and observer timestamp_ns",
        events=str(args.events),
        benchmark=str(args.benchmark),
        rows=rows,
        medians={
            key: statistics.median(row[key] for row in rows)
            for key in (
                "ttft_ms",
                "client_to_prefill_dispatch_ms",
                "runner_prefill_ms",
                "prefill_completion_to_client_token_ms",
            )
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
