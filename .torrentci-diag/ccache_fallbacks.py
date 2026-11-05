#!/usr/bin/env python3
# TEMPORARY (torrentci#2377 diagnostic): attribute ccache direct-mode fallbacks.
# Usage: ccache_fallbacks.py LOGFILE OUT_EXCERPT
# Prints result counts and fallback reasons; writes the log lines of every
# non-direct-hit invocation (lock/config noise dropped) to OUT_EXCERPT.
import collections
import re
import sys

LINE = re.compile(r"^\[(\S+) (\d+)\s*\] (.*)$")
NOISE = re.compile(r"^(Config: |Acquir|Releas|Creating .*\.alive|Removing |Using default compression)")
log_path, out_path = sys.argv[1], sys.argv[2]
procs = collections.defaultdict(list)
with open(log_path, errors="replace") as handle:
    for raw in handle:
        match = LINE.match(raw.rstrip("\n"))
        if match:
            procs[match.group(2)].append(raw.rstrip("\n"))


def msg(line):
    return LINE.match(line).group(3)


def norm(path):
    path = re.sub(r"^/workspace/", "", path)
    path = re.sub(r"\.cpmcache/[^/]+/[0-9a-f]{6,}/", ".cpmcache/<pkg>/<hash>/", path)
    return path


results = collections.Counter()
reasons = collections.Counter()
first_mismatch = collections.Counter()
any_mismatch = collections.Counter()
sources = collections.defaultdict(list)
excerpt_lines = 0
with open(out_path, "w") as out:
    for pid, lines in procs.items():
        messages = [msg(line) for line in lines]
        res = {m.split("Result: ", 1)[1].strip() for m in messages if m.startswith("Result: ")}
        if "direct_cache_hit" in res:
            results["direct_cache_hit"] += 1
            continue
        if "preprocessed_cache_hit" in res:
            kind = "preprocessed_cache_hit"
        elif "cache_miss" in res:
            kind = "cache_miss"
        else:
            continue
        results[kind] += 1
        mkey = next((m.split(": ", 1)[1] for m in messages if m.startswith("Manifest key: ")), None)
        mismatches = [
            norm(m[len("Mismatch for ") :].split(" hash ", 1)[0].split(" ", 1)[0])
            for m in messages
            if m.startswith("Mismatch for ")
        ]
        disabled = [
            m
            for m in messages
            if re.search(
                r"disabling direct mode|Unsupported compiler option for direct mode|Read-only direct mode|Found (__DATE__|__TIME__|__TIMESTAMP__)|Include file .* too new",
                m,
                re.I,
            )
        ]
        overflow = [m for m in messages if m.startswith("More than ") or "Serialized manifest too large" in m]
        if disabled:
            reason = "direct mode disabled/skipped: " + re.sub(r"/\S+", "<path>", disabled[0])[:100]
        elif mkey is None:
            reason = "no manifest lookup"
        elif not any(m.startswith("Retrieved " + mkey) for m in messages):
            reason = "manifest not in storage"
        elif any(m.startswith("Got result key from manifest") for m in messages):
            reason = "manifest matched, result object missing"
        elif mismatches:
            reason = "manifest found, include mismatch"
            first_mismatch[mismatches[0]] += 1
            for path in set(mismatches):
                any_mismatch[path] += 1
        else:
            reason = "manifest found, no entry matched (no per-file detail logged)"
        if overflow:
            reason += " [manifest overflow logged]"
        reasons[(kind, reason)] += 1
        src = next((m.split(": ", 1)[1] for m in messages if m.startswith("Source file: ")), "?")
        if len(sources[(kind, reason)]) < 8:
            sources[(kind, reason)].append(norm(src))
        for line in lines:
            if not NOISE.match(msg(line)):
                out.write(line + "\n")
                excerpt_lines += 1

print("== ccache_fallbacks (torrentci#2377)")
print("invocations by result:", dict(results))
print("\nfallback/miss reasons (result, reason, count):")
for (kind, reason), count in reasons.most_common():
    print(f"{count:6}  {kind:24} {reason}")
    for src in sources[(kind, reason)]:
        print(f"          e.g. {src}")
print(f"\nfirst mismatched include per invocation ({sum(first_mismatch.values())}):")
for path, count in first_mismatch.most_common(25):
    print(f"{count:6}  {path}")
print(f"\nany mismatched include per invocation ({len(any_mismatch)} files):")
for path, count in any_mismatch.most_common(25):
    print(f"{count:6}  {path}")
print(f"\nexcerpt lines written: {excerpt_lines}")
