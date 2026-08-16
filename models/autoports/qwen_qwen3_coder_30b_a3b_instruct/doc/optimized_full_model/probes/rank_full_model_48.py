# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rank the ops of one verified 48-layer decode iteration, per-layer vs terminal.

`tt-perf-report`'s stacked table groups by op code plus in0 memory config, which
merges (for example) the LM head into a generic ``MatmulDeviceOperation
(in0:dram_interleaved)`` bucket and gives no per-layer / terminal split at all.
This produces the split ranking stage 06 needs, off the *same* windowed CSV that
`tt-perf-report` reads, so the two are reconcilable row for row.

The three regions are derived from the CSV, not transcribed:

* **terminal-pre** -- everything before the first layer's input norm: the token
  ``ttnn.embedding`` plus the two ``rope_decode_tables`` cos/sin gathers and
  their untilize/tilize/transpose tail;
* **the 48-layer stack** -- from the first layer's ``InterleavedToSharded``
  feeding its input norm through the residual add two rows after the **96th**
  ``AllGatherAsyncDeviceOperation`` (the layer's second all-reduce ends
  ``... AllGather -> Clone -> BinaryNg``, exactly as in stage 04's published
  window);
* **terminal-post** -- final norm, LM head, distributed argmax, feedback copy.

Times are ``DEVICE KERNEL DURATION``. Device 0 is the reported device and the
other three are printed alongside as a spread check.

    python rank_full_model_48.py /tmp/fm48_decode_window.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

LAYERS = 48


def shape(row, i):
    try:
        return "x".join(str(int(float(row[f"INPUT_{i}_{a}_PAD[LOGICAL]"].split("[")[0]))) for a in "WZYX")
    except (KeyError, ValueError):
        return ""


def label(row):
    """Op code plus just enough shape to tell two call sites apart."""
    code = row["OP CODE"].replace("DeviceOperation", "").replace("Operation", "")
    s0, s1 = shape(row, 0), shape(row, 1)
    if row["OP CODE"] in ("MatmulDeviceOperation", "SparseMatmulDeviceOperation"):
        return f"{code} {s0} @ {s1}"
    return f"{code} {s0}" if s0 else code


def regions(rows):
    ag = [i for i, r in enumerate(rows) if r["OP CODE"] == "AllGatherAsyncDeviceOperation"]
    assert len(ag) == 2 * LAYERS, f"{len(ag)} all-gathers, expected {2 * LAYERS}"
    end = ag[-1] + 2  # AllGather -> Clone -> BinaryNg(residual add)
    assert rows[end - 1]["OP CODE"] == "CloneOperation", rows[end - 1]["OP CODE"]
    assert rows[end]["OP CODE"] == "BinaryNgDeviceOperation", rows[end]["OP CODE"]
    # first layer starts at the InterleavedToSharded feeding its first LayerNorm
    first_ln = next(i for i, r in enumerate(rows) if r["OP CODE"] == "LayerNormDeviceOperation")
    start = first_ln - 1
    assert rows[start]["OP CODE"] == "InterleavedToShardedDeviceOperation", rows[start]["OP CODE"]
    return rows[:start], rows[start : end + 1], rows[end + 1 :]


def us(row):
    return float(row["DEVICE KERNEL DURATION [ns]"] or 0) / 1000.0


def report(name, rows, total_us, per_layer=None, top=None):
    agg = defaultdict(lambda: [0.0, 0, set()])
    for r in rows:
        e = agg[label(r)]
        e[0] += us(r)
        e[1] += 1
        e[2].add(int(r["CORE COUNT"]))
    section = sum(v[0] for v in agg.values())
    print(
        f"\n=== {name}: {section:,.1f} us over {len(rows)} ops "
        f"({100 * section / total_us:.2f}% of the {total_us:,.1f} us iteration) ==="
    )
    hdr = f"{'us':>9} {'%iter':>6} {'n':>5} {'us/call':>8} {'cores':>10}  op"
    if per_layer:
        hdr = f"{'us':>9} {'%iter':>6} {'n':>5} {'us/call':>8} {'us/layer':>9} {'cores':>10}  op"
    print(hdr)
    items = sorted(agg.items(), key=lambda kv: -kv[1][0])
    if top:
        items = items[:top]
    for op, (t, n, cores) in items:
        c = ",".join(str(x) for x in sorted(cores))
        if per_layer:
            print(f"{t:9.1f} {100*t/total_us:5.2f}% {n:5d} {t/n:8.3f} {t/LAYERS:9.3f} {c:>10}  {op}")
        else:
            print(f"{t:9.1f} {100*t/total_us:5.2f}% {n:5d} {t/n:8.3f} {c:>10}  {op}")
    return section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--top", type=int, default=0)
    args = parser.parse_args()

    allrows = list(csv.DictReader(args.csv.open()))
    by_device = defaultdict(list)
    for r in allrows:
        by_device[r["DEVICE ID"]].append(r)

    print("iteration device-kernel-time totals, all devices (spread check):")
    for d in sorted(by_device):
        pre, layers, post = regions(by_device[d])
        tp, tl, ts = (sum(us(r) for r in x) for x in (pre, layers, post))
        print(
            f"  device {d}: total {tp+tl+ts:9,.1f} us = pre {tp:7,.1f} + 48 layers {tl:9,.1f} "
            f"+ terminal {ts:7,.1f}   (per layer {tl/LAYERS:7.3f} us)"
        )

    rows = by_device["0"]
    pre, layers, post = regions(rows)
    total = sum(us(r) for r in rows)
    report("terminal-pre (embedding + rope cos/sin gather)", pre, total, top=args.top or None)
    report("48-layer stack", layers, total, per_layer=True, top=args.top or None)
    report("terminal-post (final norm, LM head, sampler, feedback)", post, total, top=args.top or None)


if __name__ == "__main__":
    main()
