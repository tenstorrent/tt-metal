"""Compare 32-bit perf-counter references with 64-bit TRISC1 kernel timestamps.

FPU/SFPU event counts are NOT automatically unwrapped. Validate them with a
shorter workload before interpreting a long run. Combined MATH is omitted.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict


def analyze(path):
    records = defaultdict(dict)
    with open(path) as source:
        header = source.readline()
        chip_cores = int(header.split("Max Compute Cores: ")[1])
        for raw in csv.DictReader(source):
            row = {key.strip(): value.strip() for key, value in raw.items()}
            key = (int(row["run host ID"]), int(row["core_x"]), int(row["core_y"]))
            if row["RISC processor type"] == "TRISC_1" and row["zone name"] == "TRISC-KERNEL":
                records[key][row["type"]] = int(row["time[cycles since reset]"])
            if row["timer_id"] == "9090":
                data = json.loads(row["meta data"].replace(";", ","))
                records[key][data["counter type"]] = data
    by_run = defaultdict(list)
    for (run, x, y), record in records.items():
        if "FPU_COUNTER" not in record:
            continue
        duration = record["ZONE_END"] - record["ZONE_START"]
        ref = record["FPU_COUNTER"]["ref cnt"]
        wraps = round((duration - ref) / 2**32)
        unwrapped_ref = ref + wraps * 2**32
        assert abs(unwrapped_ref - duration) < 10000, (run, x, y, unwrapped_ref, duration)
        record.update(duration=duration, reference_wraps=wraps, unwrapped_reference=unwrapped_ref)
        by_run[run].append(record)
    for run, cores in sorted(by_run.items()):
        span = max(core["ZONE_END"] for core in cores) - min(core["ZONE_START"] for core in cores)
        result = dict(path=path, run=run, active_cores=len(cores), chip_cores=chip_cores, span_cycles=span)
        result["reference_wraps"] = sorted(set(core["reference_wraps"] for core in cores))
        for counter in ("FPU_COUNTER", "SFPU_COUNTER"):
            values = [core[counter]["value"] for core in cores]
            percentages = [100 * core[counter]["value"] / core["unwrapped_reference"] for core in cores]
            result[counter] = dict(
                raw_count_sum=sum(values),
                raw_count_min=min(values),
                raw_count_max=max(values),
                full_chip_pct=100 * sum(values) / (chip_cores * span),
                active_core_median_pct=statistics.median(percentages),
                active_core_min_pct=min(percentages),
                active_core_max_pct=max(percentages),
            )
        print(json.dumps(result))


for path in sys.argv[1:]:
    analyze(path)
