# SPDX-License-Identifier: Apache-2.0
"""Derive an auditable request-window view without dropping trace lifecycle events.

CPU/file-only utility. The source is never changed. Every retained JSONL record
is copied byte-for-byte; counters and timestamps are never synthesized or reset.
Only the five explicitly listed request/replay event types may be omitted outside
the selected window. Unknown events, preparation, captures, and all release or
retirement records survive regardless of timestamp. This is not a passing audit.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REQUEST_EVENTS = frozenset({"prefill_schedule", "prefill_chunk", "prefill_complete", "decode_submit", "replay"})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scope(source, client, *, start_time=None, end_time=None, start_byte=None, end_byte=None):
    """Return exact source records and a separate provenance manifest."""
    require(source and source.endswith(b"\n"), "Source must end with a complete newline-terminated record")
    requests = client.get("requests", [])
    require(requests, "Client must contain completed requests")
    require(client.get("completed_requests") == len(requests), "Incomplete client request summary")
    require(client.get("status") in ("pass", "failed"), "Client must preserve its completed pass/failed result")
    starts, ends = [r["start"] for r in requests], [r["end"] for r in requests]
    require(all(math.isfinite(t) for t in starts + ends), "Nonfinite request timestamp")
    require(all(a < b for a, b in zip(starts, ends)), "Invalid request timing")
    first, last = min(starts), max(ends)
    lower = first if start_time is None else start_time
    upper = last if end_time is None else end_time
    require(
        math.isfinite(lower) and math.isfinite(upper) and lower <= first < last <= upper,
        "Scope must include every completed request's entire time interval",
    )
    begin = 0 if start_byte is None else start_byte
    finish = len(source) if end_byte is None else end_byte
    require(0 <= begin <= finish <= len(source), "Invalid byte interval")
    require(begin == 0 or source[begin - 1 : begin] == b"\n", "Start byte must be a JSONL record boundary")
    require(finish == 0 or source[finish - 1 : finish] == b"\n", "End byte must be a JSONL record boundary")
    kept, omitted = Counter(), Counter()
    kept_spans, window_spans = [], []
    output = []
    offset = 0
    all_times, window_times = [], []
    pids = set()

    def extend_span(spans, line, left, right):
        if spans and spans[-1]["last_line"] == line - 1:
            spans[-1].update(last_line=line, end_byte=right)
        else:
            spans.append(dict(first_line=line, last_line=line, start_byte=left, end_byte=right))

    for line, raw in enumerate(source.splitlines(keepends=True), 1):
        event = json.loads(raw)
        kind, when = event["event"], event["time"]
        require(isinstance(kind, str) and math.isfinite(when), "Invalid event kind/time")
        require("pid" in event, "Missing event process identity")
        pids.add(event["pid"])
        right = offset + len(raw)
        in_time = lower <= when <= upper
        in_bytes = begin <= offset and right <= finish
        # Byte markers constrain which request records belong to the window;
        # never let them silently trim a request whose client timing is inside.
        if first <= when <= last:
            require(in_bytes, "Byte interval excludes an event within a completed request interval")
        in_window = in_time and in_bytes
        retain = kind not in REQUEST_EVENTS or in_window
        all_times.append(when)
        if in_window:
            window_times.append(when)
            extend_span(window_spans, line, offset, right)
        if retain:
            output.append(raw)
            kept[kind] += 1
            extend_span(kept_spans, line, offset, right)
        else:
            omitted[kind] += 1
        offset = right
    require(len(pids) == 1, "Source contains multiple server processes")
    require(window_times, "No events in selected request window")
    require(kept["prepared"] == 1, "Expected exactly one retained preparation baseline")
    result = b"".join(output)
    manifest = dict(
        status="scoped-not-a-correctness-verdict",
        source_sha256=sha(source),
        source_bytes=len(source),
        source_lines=sum(kept.values()) + sum(omitted.values()),
        source_time_bounds=[min(all_times), max(all_times)],
        source_pid=next(iter(pids)),
        client_status=client["status"],
        client_completed_requests=len(requests),
        client_time_bounds=[first, last],
        requested_time_bounds=[lower, upper],
        requested_byte_bounds=[begin, finish],
        window_time_bounds=[min(window_times), max(window_times)],
        window_line_byte_spans=window_spans,
        retained_line_byte_spans=kept_spans,
        retained_counts=dict(sorted(kept.items())),
        omitted_counts=dict(sorted(omitted.items())),
        retained_records=sum(kept.values()),
        omitted_records=sum(omitted.values()),
        output_sha256=sha(result),
        output_bytes=len(result),
        omission_allowlist=sorted(REQUEST_EVENTS),
        rule="Only listed request/replay types outside the window are omitted; every other record is retained byte-for-byte across the entire source snapshot.",
        source_lifetime_limit="Source hash covers the complete file read at collection; later appends require a new view/manifest. Teardown can be audited separately against the final full journal.",
    )
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "client", "output", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--start-byte", type=int)
    parser.add_argument("--end-byte", type=int)
    args = parser.parse_args()
    require(not args.output.exists() and not args.manifest.exists(), "Use fresh output/manifest paths")
    require(
        len({p.resolve() for p in (args.events, args.client, args.output, args.manifest)}) == 4,
        "Source, client, view and manifest paths must be distinct",
    )
    source = args.events.read_bytes()
    client_raw = args.client.read_bytes()
    view, report = scope(
        source,
        json.loads(client_raw),
        start_time=args.start_time,
        end_time=args.end_time,
        start_byte=args.start_byte,
        end_byte=args.end_byte,
    )
    # Do not attach a full-file hash to an input that changed while scoping.
    require(args.events.read_bytes() == source, "Source changed during scoping; retry while traffic is quiescent")
    report.update(
        source=str(args.events.resolve()),
        client=str(args.client.resolve()),
        client_sha256=sha(client_raw),
        output=str(args.output.resolve()),
        scoper_source_sha256=sha(Path(__file__).read_bytes()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(view)
    with args.manifest.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "retained_records", "omitted_records", "output_sha256")}))


if __name__ == "__main__":
    main()
