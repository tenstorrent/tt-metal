#!/usr/bin/env python3
# TEMPORARY (torrentci#2377 diagnostic): summarize a CCACHE_LOGFILE.
# Per invocation: wall time by result category. Per message template: total
# time until the next line of the same invocation (where the time goes).
import collections
import datetime
import re
import statistics
import sys

LINE = re.compile(r"^\[(\S+) (\d+)\s*\] (.*)$")
path = sys.argv[1]
procs = collections.defaultdict(list)
with open(path, errors="replace") as handle:
    for raw in handle:
        match = LINE.match(raw.rstrip("\n"))
        if not match:
            continue
        stamp, pid, message = match.groups()
        try:
            when = datetime.datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            continue
        procs[pid].append((when, message))


def template(message):
    message = re.sub(r"/\S+", "/P", message)
    message = re.sub(r"\b[0-9a-z]{16,}\b", "K", message)
    message = re.sub(r"\d+(\.\d+)?", "N", message)
    return message[:110]


def category(results):
    for name in ("direct_cache_hit", "preprocessed_cache_hit", "cache_miss"):
        if name in results:
            return name
    return "other:" + ",".join(sorted(results))[:60]


by_category = collections.defaultdict(list)
by_template = collections.defaultdict(lambda: [0, 0.0])
examples = collections.defaultdict(list)
for pid, lines in procs.items():
    lines.sort(key=lambda item: item[0])
    results = {m.split("Result: ", 1)[1].strip() for _, m in lines if "Result: " in m}
    if not results:
        continue
    cat = category(results)
    by_category[cat].append(lines[-1][0] - lines[0][0])
    if len(examples[cat]) < 2:
        examples[cat].append(lines)
    for (when, message), (after, _) in zip(lines, lines[1:]):
        entry = by_template[template(message)]
        entry[0] += 1
        entry[1] += after - when

print(f"invocations: {sum(len(v) for v in by_category.values())}")
print("category                       count   p50_s   p90_s   max_s   sum_s")
for cat, values in sorted(by_category.items(), key=lambda kv: -sum(kv[1])):
    values.sort()
    p90 = values[min(len(values) - 1, int(len(values) * 0.9))]
    print(f"{cat:30} {len(values):5} {statistics.median(values):7.3f} {p90:7.3f} {values[-1]:7.2f} {sum(values):7.1f}")
print("\ntop templates by time until the next line (count, sum_s, mean_ms):")
for name, (count, total) in sorted(by_template.items(), key=lambda kv: -kv[1][1])[:45]:
    print(f"{count:6} {total:8.1f} {1000 * total / count:8.1f}  {name}")
mismatch = collections.Counter()
for lines in procs.values():
    for _, message in lines:
        if message.startswith("Mismatch for "):
            mismatch[message.split(" hash ", 1)[0][13:]] += 1
print(f"\ndirect-mode include mismatches by file ({sum(mismatch.values())} total, {len(mismatch)} files):")
for name, count in mismatch.most_common(30):
    print(f"{count:6}  {name}")
for cat in ("preprocessed_cache_hit", "cache_miss"):
    for lines in examples.get(cat, [])[:1]:
        print(f"\nexample {cat}:")
        base = lines[0][0]
        for when, message in [item for item in lines if not item[1].startswith("Config:")][:150]:
            print(f"  +{when - base:7.3f} {message[:200]}")
