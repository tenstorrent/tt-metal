# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure host phase reducer for immutable serving completion events.

Each event has submission_id, event (dispatch/completion), timestamp_ns,
phase (prefill/decode), and request_ids. Metadata must match across the pair.
The caller supplies exact measured request IDs. Unrelated work is excluded only
outside the measured envelope. Initial queueing and final transport are outside
scope. Transition gaps belong to the following phase. No work estimates or
hardware/device timing are inferred.
"""

from collections import defaultdict


def reduce_phases(events, request_ids):
    """Validate immutable event pairs and return nonoverlapping host phases."""
    cohort = frozenset(request_ids)
    if not cohort or any(not isinstance(r, str) or not r for r in cohort):
        raise ValueError("measured request_ids must be nonempty strings")
    paired = defaultdict(dict)
    for event in events:
        sid = event["submission_id"]
        kind = event["event"]
        timestamp = event["timestamp_ns"]
        phase = event["phase"]
        ids = tuple(event["request_ids"])
        if not isinstance(sid, str) or not sid:
            raise ValueError("submission_id must be a nonempty string")
        if kind not in ("dispatch", "completion") or phase not in ("prefill", "decode"):
            raise ValueError("invalid event or phase")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(r, str) or not r for r in ids):
            raise ValueError("invalid request_ids")
        if kind in paired[sid]:
            raise ValueError(f"duplicate {kind}: {sid}")
        paired[sid][kind] = (timestamp, phase, ids)
    selected = []
    unrelated = []
    for sid, pair in paired.items():
        if set(pair) != {"dispatch", "completion"}:
            raise ValueError(f"missing dispatch or completion: {sid}")
        start, phase, ids = pair["dispatch"]
        end, end_phase, end_ids = pair["completion"]
        if (phase, ids) != (end_phase, end_ids):
            raise ValueError(f"mutated submission metadata: {sid}")
        if end <= start:
            raise ValueError(f"invalid interval: {sid}")
        record = dict(submission_id=sid, phase=phase, request_ids=list(ids), dispatch_ns=start, completion_ns=end)
        overlap = set(ids) & cohort
        if overlap and not set(ids) <= cohort:
            raise ValueError(f"mixed measured and unrelated requests: {sid}")
        (selected if overlap else unrelated).append(record)
    covered = {r for record in selected for r in record["request_ids"]}
    if covered != cohort:
        raise ValueError("measured request coverage mismatch")
    selected.sort(key=lambda record: (record["dispatch_ns"], record["submission_id"]))
    first = selected[0]["dispatch_ns"]
    last = max(record["completion_ns"] for record in selected)
    if any(record["dispatch_ns"] < last and record["completion_ns"] > first for record in unrelated):
        raise ValueError("unrelated work overlaps measured envelope")
    segments = []
    for record in selected:
        if segments and segments[-1]["phase"] == record["phase"]:
            segment = segments[-1]
            segment["end_ns"] = max(segment["end_ns"], record["completion_ns"])
            segment["submission_ids"].append(record["submission_id"])
        else:
            if segments and record["dispatch_ns"] < segments[-1]["end_ns"]:
                raise ValueError("cross-phase overlap cannot be attributed safely")
            segments.append(
                dict(
                    phase=record["phase"],
                    start_ns=segments[-1]["end_ns"] if segments else first,
                    first_dispatch_ns=record["dispatch_ns"],
                    end_ns=record["completion_ns"],
                    submission_ids=[record["submission_id"]],
                )
            )
    seconds = {phase: 0.0 for phase in ("prefill", "decode")}
    for segment in segments:
        segment["duration_ns"] = segment["end_ns"] - segment["start_ns"]
        seconds[segment["phase"]] += segment["duration_ns"] / 1e9
    return dict(
        schema_version=1,
        status="validated_host_phase_reduction",
        scope="first measured dispatch through final measured completion; transition gaps assigned to following phase",
        request_ids=sorted(cohort),
        first_dispatch_ns=first,
        last_completion_ns=last,
        elapsed_ns=last - first,
        excluded_submission_ids=sorted(record["submission_id"] for record in unrelated),
        submissions=selected,
        segments=segments,
        phase_seconds=seconds,
    )
