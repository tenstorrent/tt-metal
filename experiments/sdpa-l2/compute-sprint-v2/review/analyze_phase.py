# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode sampled RISC issue markers; not an engine-retirement profiler."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

STEPS = ["ENTRY", "ACQUIRED", "COPY_INIT", "COPIED", "BROADCAST", "COMMITTED", "WAITED", "SFPU", "PACKED", "RELEASED"]

def analyze(path):
    batches = defaultdict(dict)
    kernels = defaultdict(dict)
    with path.open() as stream:
        metadata = next(stream).strip()
        for raw in csv.DictReader(stream):
            r = {k.strip():v.strip() for k,v in raw.items()}
            zone = r["zone name"]
            if zone.startswith("B_STATE_"):
                key = (r["RISC processor type"], int(r["data"]))
                batches[key][zone.removeprefix("B_STATE_")] = int(r["time[cycles since reset]"])
            if zone == "SDPA_RESIDENT" and r["RISC processor type"] == "TRISC_1":
                kernels[r["run host ID"]][r["type"]] = int(r["time[cycles since reset]"])
    groups = defaultdict(list)
    details = []
    for (risc,batch), events in sorted(batches.items()):
        assert set(events)==set(STEPS), (risc,batch,"missing markers",set(STEPS)-set(events))
        intervals = {a+"_to_"+b:events[b]-events[a] for a,b in zip(STEPS,STEPS[1:])}
        intervals["total"] = events["RELEASED"]-events["ENTRY"]
        kind = "denominator" if batch>=100 else "numerator"
        groups[(risc,kind)].append(intervals)
        details.append(dict(risc=risc,batch=batch,kind=kind,events=events,intervals=intervals))
    assert len(batches)==30, ("Expected 10 sampled batches on 3 RISCs",len(batches))
    summary = []
    for (risc,kind), samples in sorted(groups.items()):
        summary.append(dict(risc=risc,kind=kind,count=len(samples),median_cycles={
            key:statistics.median(s[key] for s in samples) for key in samples[0]}))
    return dict(metadata=metadata,warning="RISC issue timeline, not exact engine-retirement latency; markers perturb scheduling.",
                kernels={key:v["ZONE_END"]-v["ZONE_START"] for key,v in kernels.items()},summary=summary,details=details)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    args=p.parse_args()
    print(json.dumps(analyze(args.path),indent=2))
