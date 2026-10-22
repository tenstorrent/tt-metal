# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Strict, device-free analysis of one isolated P/D request's evidence."""
import math
import re
from datetime import datetime, timezone


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not math.isfinite(start + end) or end < start:
            raise ValueError("invalid interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def interval_metrics(prefill, transfers):
    p, t = union(prefill), union(transfers)
    if not p or not t:
        raise ValueError("prefill and transfer evidence required")
    overlap = duration([(max(a, c), min(b, d)) for a, b in p for c, d in t if max(a, c) < min(b, d)])
    return dict(
        prefill_active_s=duration(p),
        prefill_span_s=p[-1][1] - p[0][0],
        transfer_sum_s=sum(b - a for a, b in transfers),
        transfer_active_s=duration(t),
        transfer_span_s=t[-1][1] - t[0][0],
        overlap_s=overlap,
        nonoverlap_prefill_s=duration(p) - overlap,
        nonoverlap_transfer_s=duration(t) - overlap,
        initial_prefill_s=duration([(a, min(b, t[0][0])) for a, b in p if a < t[0][0]]),
        final_transfer_tail_s=max(0, t[-1][1] - p[-1][1]),
        final_transfer_active_s=duration([(max(a, p[-1][1]), b) for a, b in t if b > p[-1][1]]),
    )


def log_epoch(line):
    stamp = re.search(r"\[(\d{4}-\d\d-\d\d[^]]+)\]", line)
    if not stamp:
        raise ValueError("migration timestamp missing: " + line)
    value = datetime.fromisoformat(stamp[1].replace("Z", "+00:00"))
    return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()


def expected_migration_positions(prefill_positions, prompt_tokens, block_size):
    # PrefixIndexer::last_reusable_boundary excludes the final prompt block.
    limit = (prompt_tokens - 1) // block_size * block_size
    return [(a, min(b, limit)) for a, b in prefill_positions if a < limit]


def migration_intervals(text, layers, expected_positions):
    """Validate EVERY command and completion in the saved request log window."""
    starts, ends, command_counts = {}, {}, {}
    faults = [
        line
        for line in text.splitlines()
        if re.search(r"successful=false|queue has no chunks|stays stale|condemn|reclaim", line, re.I)
    ]
    if faults:
        raise ValueError("migration fault in request window: " + "\n".join(faults[:10]))
    for line in text.splitlines():
        if "migrate command:" in line:
            match = re.search(r"migration_id=(\d+).*layers=\[(\d+),(\d+)\).*src_positions=\[(\d+),(\d+)\)", line)
            if not match:
                if re.search(r"migration_id=\d+: deferred reason=\w+", line):
                    continue
                raise ValueError("unparsed migrate command: " + line)
            mid, lb, le, begin, end = map(int, match.groups())
            for layer in range(lb, le):
                key = (mid, layer)
                if key in ends:
                    raise ValueError(f"migrate command after completion {key}")
                if key in starts and starts[key][1] != (begin, end):
                    raise ValueError(f"conflicting migrate command {key}")
                # Admission retries and local child dispatch repeat this line.
                # Retain the first timestamp, including queue/control-plane time.
                starts.setdefault(key, (log_epoch(line), (begin, end)))
                command_counts[key] = command_counts.get(key, 0) + 1
        elif re.search(r"migration_id=\d+ layer=\d+.*: completed successful=", line):
            match = re.search(
                r"migration_id=(\d+) layer=(\d+).*src_positions=\[(\d+),(\d+)\).*completed successful=true reason= releasing",
                line,
            )
            if not match:
                raise ValueError("unparsed migration completion: " + line)
            mid, layer, begin, end = map(int, match.groups())
            key = (mid, layer)
            if key in ends:
                raise ValueError(f"duplicate migration completion {key}")
            ends[key] = (log_epoch(line), (begin, end))
    if starts.keys() != ends.keys():
        raise ValueError(f"unmatched commands/completions: {len(starts)} starts, {len(ends)} ends")
    expected = {(tuple(position), layer) for position in expected_positions for layer in range(layers)}
    observed = [(position, layer) for (_, layer), (_, position) in starts.items()]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f"layer/range coverage mismatch: {len(observed)} vs expected {len(expected)}")
    result = []
    for key, (start, position) in starts.items():
        end, final_position = ends[key]
        if final_position != position or end < start:
            raise ValueError("migration completion range/time mismatch")
        result.append(
            dict(
                migration_id=key[0],
                layer=key[1],
                positions=position,
                start_epoch=start,
                end_epoch=end,
                duration_s=end - start,
                command_log_count=command_counts[key],
            )
        )
    return result


def metric(text, name, labels=None):
    samples = []
    for line in text.splitlines():
        match = re.fullmatch(r"([\w:]+)(\{.*?\})?\s+([-+\w.eE]+)(?:\s+\d+)?", line)
        if not match or match[1] != name:
            continue
        found = dict(re.findall(r'(\w+)="([^"\\]*)"', match[2] or ""))
        if labels and any(found.get(key) != val for key, val in labels.items()):
            continue
        samples.append(float(match[3]))
    if not samples or not all(math.isfinite(x) for x in samples):
        raise ValueError(f"missing/nonfinite metric {name} {labels or {}}")
    return sum(samples)


def counter_delta(before, after, name, labels=None):
    delta = metric(after, name, labels) - metric(before, name, labels)
    if delta < 0:
        raise ValueError("counter reset during request: " + name)
    return delta
