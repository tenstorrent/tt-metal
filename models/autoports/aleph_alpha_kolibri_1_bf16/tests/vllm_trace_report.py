# SPDX-License-Identifier: Apache-2.0
"""Validate recorded same-server trace lifetimes and scheduler input refreshes."""

import argparse
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("requests")
p.add_argument("--events", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
r = json.loads(Path(a.requests).read_text())
events = r["events"]
pids = {e["pid"] for e in events}
assert len(pids) == 1, pids
all_events = [json.loads(x) for x in Path(a.events).read_text().splitlines() if json.loads(x)["pid"] in pids]
captures = [x for x in all_events if x["event"] == "capture"]
counts = Counter(x["key"] for x in captures)
compiles = [x for x in all_events if x["event"] == "compile"]
assert set(counts) <= {x["key"] for x in compiles}, "Missing compilation timings"
assert all(v == 1 for v in counts.values()), counts
assert not any(x["event"] in ("capture", "shutdown_release", "retirement", "eviction") for x in events)
assert {129, 130, 131} <= {x["length"] for x in r["results"]}
chunks = [x for x in events if x["event"] == "prefill_chunk"]
assert len({x["start"] for x in chunks if x["start"] > 0}) >= 2
assert any(x["logical"] != x["physical"] for x in chunks)
decode = [x for x in events if x["event"] == "decode"]
assert any(x["reload_page_table"] and not x["reload_inputs"] for x in decode)
steady = 0
for prev, cur in zip(decode, decode[1:]):
    if cur["reload_inputs"] or cur["reload_page_table"]:
        continue
    # With no intervening prefill, unchanged decode must not copy any scheduler inputs.
    if any(prev["time"] < x["time"] < cur["time"] for x in events if x["event"] == "prefill"):
        continue
    for field in (
        "token_refreshes",
        "position_refreshes",
        "rope_refreshes",
        "page_table_refreshes",
        "sampling_parameter_refreshes",
    ):
        assert cur["counters"].get(field, 0) == prev["counters"].get(field, 0), (field, prev, cur)
    steady += 1
assert steady > 10, steady
# Match adjacent chunks in the same physical request slot by absolute end/start.
last_prefill = {}
interleaved = []
for event in events:
    if event["event"] != "prefill":
        continue
    for slot, start, end in zip(event["slots"], event["starts"], event["lengths"]):
        previous = last_prefill.get(slot)
        if start > 0 and previous is not None and previous["end"] == start:
            interleaved.extend(d for d in decode if previous["time"] < d["time"] < event["time"])
        last_prefill[slot] = dict(time=event["time"], end=end)
assert interleaved
report = dict(
    status="pass",
    worker_pid=next(iter(pids)),
    captures_per_signature=dict(counts),
    capture_details=captures,
    compile_details=compiles,
    workload_captures=0,
    workload_retirements=0,
    logical_lengths=[x["length"] for x in r["results"]],
    chunk_start_positions=sorted({x["start"] for x in chunks}),
    scheduler_start_positions=sorted({v for x in events if x["event"] == "prefill" for v in x["starts"]}),
    chunk_shapes=sorted({(x["start"], x["logical"], x["physical"]) for x in chunks}),
    steady_decode_steps_without_refresh=steady,
    page_only_refreshes=sum(x["reload_page_table"] and not x["reload_inputs"] for x in decode),
    interleaved_decode_steps=len(interleaved),
    largest_active_decode_rows=max(sum(p >= 0 for p in x["positions"]) for x in decode),
    ttft_percentiles_ms=r.get("ttft_percentiles_ms"),
    streaming_pause_percentiles_ms=r.get("streaming_pause_percentiles_ms"),
    completed_requests_per_second=r.get("completed_requests_per_second"),
)
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("capture_details", "compile_details", "chunk_shapes")}))
