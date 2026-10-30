"""Select one complete median-duration replay for readable native perf advice.

Headline metrics continue to use all 128 whole-layer windows. Exclude the
incoming inter-replay gap from this diagnostic table, preserving all internal
latencies and original timing/operation identifiers.
"""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tests.summarize_optimized_profile import extract, window


def select(source, output):
    with Path(source).open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    decode = extract(rows, "PERF_DECODE", "PERF_DECODE_END")
    sessions = defaultdict(list)
    for row in decode:
        if row.get("DEVICE FW START CYCLE"):
            sessions[(row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])].append(row)
    assert len(sessions) == 128
    durations = {key: window(group)[0] for key, group in sessions.items()}
    median = statistics.median(durations.values())
    key = min(durations, key=lambda k: abs(durations[k] - median))
    group = sessions[key]
    first = dict(group[0])
    boundary = {}
    for field in fields:
        if "LATENCY" in field and ("OP TO OP" in field or "OP-TO-OP" in field):
            boundary[field] = first[field]
            first[field] = "0"
    begin = next(row for row in rows if row["OP CODE"] == "PERF_DECODE")
    end = next(row for row in rows if row["OP CODE"] == "PERF_DECODE_END")
    with Path(output).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([begin, first, *group[1:], end])
    Path(str(output) + ".json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                trace_id=key[0],
                replay_session=key[1],
                device_us=durations[key],
                ops=len(group),
                selection="complete replay nearest median duration among128",
                incoming_boundary_latency_excluded=boundary,
                all_internal_timestamps_and_latencies_preserved=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    a = p.parse_args()
    select(a.source, a.output)
