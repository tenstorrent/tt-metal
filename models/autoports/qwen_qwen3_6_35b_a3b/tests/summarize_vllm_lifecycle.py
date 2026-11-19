# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Join HTTP timing with generator capture and adapter execution events."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text())["results"]
    begin = min(row["started"] for row in requests)
    end = max(row["started"] + row["duration"] for row in requests)
    events, captures = [], []
    for line in args.server_log.read_text().splitlines():
        for marker, target in (("QWEN_VLLM_EVENT ", events), ("QWEN_TRACE_EVENT ", captures)):
            if marker in line:
                target.append(json.loads(line.split(marker, 1)[1]))
    during = [event for event in events if begin <= event["monotonic"] <= end]
    prefills = [event for event in during if event["event"] == "prefill"]
    decodes = [event for event in during if event["event"] == "decode"]
    submitted = [event for event in during if event["event"] == "decode_submitted"]
    baseline = [event for event in events if event["monotonic"] < begin][-1]
    final = during[-1]
    overlaps = []
    for event in decodes:
        now = event["monotonic"]
        waiting = [row["length"] for row in requests if row["started"] < now < row["arrivals"][0]]
        emitting = [row["length"] for row in requests if row["arrivals"][0] < now < row["started"] + row["duration"]]
        if waiting and emitting:
            overlaps.append(dict(monotonic=now, waiting_for_prefill=waiting, emitting=emitting))
    itls = [value for row in requests for value in row["itl_ms"]]
    ttfts = [row["ttft_ms"] for row in requests]
    repeated = [row["response"]["token_ids"] for row in requests if row["length"] == 129]
    replay_delta = {
        key: final.get("trace_replays", {}).get(key, 0) - baseline.get("trace_replays", {}).get(key, 0)
        for key in final.get("trace_replays", {})
    }
    sampling_replay_delta = {
        key: count - baseline.get("sampling_trace_replays", {}).get(key, 0)
        for key, count in final.get("sampling_trace_replays", {}).items()
    }
    chunks = [
        dict(slot=slot, start=start, end=stop, monotonic=event["monotonic"])
        for event in prefills
        for slot, start, stop in zip(event["slots"], event["starts"], event["ends"])
    ]
    interleaved = []
    for slot in {chunk["slot"] for chunk in chunks}:
        owned = [chunk for chunk in chunks if chunk["slot"] == slot]
        for left, right in zip(owned, owned[1:]):
            if left["end"] == right["start"] and right["start"] > 0:
                count = sum(left["monotonic"] < event["monotonic"] < right["monotonic"] for event in decodes)
                if count:
                    interleaved.append(dict(previous=left, continuation=right, intervening_decode_submissions=count))
    page_only = [event for event in submitted if event["page_only"]]
    previous_replays = baseline.get("trace_replays", {})
    bucket_transitions = []
    sampling_transitions = []
    previous_sampling = baseline.get("sampling_trace_replays", {})
    model_buckets = {
        event["key"]: event["signature"]["physical_rows"] for event in captures if event["key"].startswith("model_")
    }
    bucket_selection_valid = True
    replay_accounting_valid = True
    for event in submitted:
        current = event["trace_replays"]
        executed = [key for key, count in current.items() if count > previous_replays.get(key, 0)]
        replay_accounting_valid &= len(executed) == 1 and all(
            current[key] - previous_replays.get(key, 0) == 1 for key in executed
        )
        if executed and (not bucket_transitions or bucket_transitions[-1]["key"] != executed[0]):
            bucket_transitions.append(dict(key=executed[0], monotonic=event["monotonic"]))
        extent = max(event.get("active_slots", ()), default=-1) + 1
        expected = next(
            (
                key
                for key in sorted(model_buckets, key=model_buckets.get)
                if key != "model_solo0" and model_buckets[key] >= extent
            ),
            None,
        )
        if "model_solo0" in model_buckets and event.get("active_slots") == [0]:
            expected = "model_solo0"
        bucket_selection_valid &= extent > 0 and executed == [expected]
        sampling = event.get("sampling_trace_replays", {})
        for key, count in sampling.items():
            if count > previous_sampling.get(key, 0) and (not sampling_transitions or sampling_transitions[-1] != key):
                sampling_transitions.append(key)
        previous_sampling = sampling
        previous_replays = current
    checks = {
        "all_requests_completed": bool(requests) and all(row["status"] == 200 for row in requests),
        "requested_output_lengths": all(
            len(row["response"]["token_ids"]) == row.get("requested_output_tokens", 96) for row in requests
        ),
        "greedy_repeat_identical": len(repeated) >= 2 and all(tokens == repeated[0] for tokens in repeated),
        "continuation_offsets": any(any(start > 0 for start in event["starts"]) for event in prefills),
        "concurrent_decode_during_prefill": bool(overlaps),
        "stable_trace_ids": baseline["traces"] == final["traces"],
        "no_recaptures": baseline["counters"]["captures"] == final["counters"]["captures"] == len(captures),
        "no_workload_releases": baseline["counters"]["releases"] == final["counters"]["releases"] == 0,
        "one_capture_per_signature": len({json.dumps(e["signature"], sort_keys=True) for e in captures})
        == len(captures),
        "device_decode_only": bool(decodes) and all(event["sampling"] == "device" for event in decodes),
    }
    if "model_2" in baseline["traces"]:
        checks["both_decode_buckets_executed"] = all(replay_delta.get(key, 0) > 0 for key in ("model_2", "model_fixed"))
        checks["all_prepared_decode_buckets_executed"] = all(
            replay_delta.get(key, 0) > 0 for key in baseline["traces"] if key.startswith("model_")
        )
        checks["one_model_replay_per_submission"] = bool(submitted) and replay_accounting_valid
        keys = [event["key"] for event in bucket_transitions]
        checks["returned_to_previous_decode_bucket"] = any(key in keys[:index] for index, key in enumerate(keys))
        checks["physical_bucket_matches_active_slots"] = bool(submitted) and bucket_selection_valid
        if "model_4" in model_buckets:
            checks["entered_and_left_B4"] = "model_4" in keys[1:-1]
        checks["overlapped_outputs_observed"] = any(event.get("pending_outputs", 0) >= 2 for event in submitted)
        checks["decode_between_continuation_chunks"] = bool(interleaved)
        checks["allocator_growth_observed"] = any(event.get("page_changes") for event in decodes)
        checks["page_only_preserves_token_position_inputs"] = bool(page_only) and all(
            event["counter_delta"]["token_refreshes"] == event["counter_delta"]["position_refreshes"] == 0
            for event in page_only
        )
    if any(row.get("top_k", 0) > 32 and row["sampled"] for row in requests):
        checks["both_sampling_widths_executed"] = all(
            sampling_replay_delta.get(key, 0) > 0 for key in ("sampling", "sampling_64")
        )
        checks["sampling_width_signatures"] = {
            event["key"]: event["signature"].get("native_draw_width")
            for event in captures
            if event["key"].startswith("sampling")
        } == {"sampling": 32, "sampling_64": 64}
        checks["returned_to_native32_sampling"] = any(
            key == "sampling" and "sampling_64" in sampling_transitions[:index]
            for index, key in enumerate(sampling_transitions)
        )
    result = dict(
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        request_source=str(args.requests),
        server_log=str(args.server_log),
        workload=[
            dict(length=row["length"], sampled=row["sampled"], output_tokens=len(row["response"]["token_ids"]))
            for row in requests
        ],
        capture_signatures=captures,
        baseline=baseline,
        last_submission=final,
        prefill_executions=prefills,
        concurrent_decode_events=overlaps,
        decode_submissions=len(decodes),
        model_trace_replay_delta=replay_delta,
        sampling_trace_replay_delta=sampling_replay_delta,
        bucket_transitions=bucket_transitions,
        sampling_transitions=sampling_transitions,
        maximum_pending_outputs=max((event.get("pending_outputs", 0) for event in submitted), default=0),
        chunk_interleaving=interleaved,
        page_only_updates=sum(bool(e.get("reload_page_table")) and not e.get("reload_inputs") for e in decodes),
        ttft_ms=dict(zip(("p50", "p95", "p99"), np.percentile(ttfts, [50, 95, 99]).tolist())),
        itl_ms=dict(zip(("p50", "p95", "p99", "max"), np.percentile(itls, [50, 95, 99, 100]).tolist())),
        elapsed_s=end - begin,
        completed_requests_per_s=len(requests) / (end - begin),
        note="First encounters included. decode_submitted counters include the submitted replay; older decode-only logs precede it. Eager prefill has no trace; model and sampler signatures are generator-owned. Shutdown releases are excluded.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks, ttft_ms=result["ttft_ms"])))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
