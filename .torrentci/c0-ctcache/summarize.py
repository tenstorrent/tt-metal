#!/usr/bin/env python3
"""C0 (torrentci#3525): one line of counts from a ctcache-shim JSON log."""
import json
import statistics
import sys

records = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        line = line.strip()
        if line:
            records.append(json.loads(line))

tus = len(records)
no_digest = sum(1 for r in records if not r["digest"])
hits = sum(1 for r in records if r["hit"])
misses = tus - hits
stored = sum(1 for r in records if r["stored"])
failed = sum(1 for r in records if r["rc"] != 0)
in_ref = sum(1 for r in records if r.get("in_ref"))
hash_times = [r["hash_s"] for r in records if r["hash_s"] is not None]
hit_times = [r["total_s"] for r in records if r["hit"]]
miss_times = [r["total_s"] for r in records if not r["hit"]]


def p50(values):
    return round(statistics.median(values), 2) if values else 0


print(
    f"tus={tus} hits={hits} misses={misses} no_digest={no_digest} stored={stored} "
    f"tidy_failed={failed} in_ref={in_ref} hit_ratio={hits / tus if tus else 0:.3f} "
    f"hash_p50_s={p50(hash_times)} hit_p50_s={p50(hit_times)} miss_p50_s={p50(miss_times)} "
    f"hash_sum_s={round(sum(hash_times))} miss_sum_s={round(sum(miss_times))}"
)
