# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Slice the LAST traced decode iteration out of a **48-layer** ops CSV.

Stage 03 published a window that straddled two decode iterations and invalidated
eight figures. Stage 05's `window_full_model.py` answered that with a *checked*
boundary rather than an eyeballed one; this is the 48-layer version of the same
check, with the counts that changed since stage 05 spelled out:

* the boundary **starts** at the first of the last three
  ``EmbeddingsDeviceOperation`` rows per device -- ``decode_hidden`` opens with
  exactly three embedding gathers per token (the token lookup plus the cos and
  sin rows ``rope_decode_tables`` reads), and nothing else in the graph uses the
  op;
* it **ends** at the last row of the file;
* and it must then hold, **per device**, exactly:

  | count | op | why |
  |---|---|---|
  | ``2 * layers`` | ``ReduceScatterMinimalAsyncDeviceOperation`` | two all-reduces per layer |
  | ``2 * layers`` | ``AllGatherAsyncDeviceOperation`` | the same two all-reduces. Stage 05's windower expected ``2 * layers + 1`` because the old sampler gathered the whole vocabulary with one ``AllGatherAsync``; that constant is stale twice over. The stage-06 distributed argmax gathers **4-wide** tensors, and at a gather dim of 4 (padded to a 32 tile) ``ttnn.all_gather`` takes its *composite* path, which is not ``AllGatherAsync`` at all -- see the next two rows. |
  | ``2`` | ``AllBroadcastDeviceOperation`` | half of the composite all-gather the distributed argmax's two 4-wide gathers (values, indices) decompose into |
  | ``2`` | ``ConcatDeviceOperation`` | the other half of the same two composite gathers |
  | ``1`` | ``GatherDeviceOperation`` | the distributed argmax's ``ttnn.gather`` of the local maximum |
  | ``layers`` | ``SdpaDecodeDeviceOperation`` | one attention per layer |
  | ``2 * layers`` | ``SparseMatmulDeviceOperation`` | the expert pair per layer |
  | ``2 * layers`` | ``PagedUpdateCacheDeviceOperation`` | K and V per layer |
  | ``3`` | ``EmbeddingsDeviceOperation`` | the boundary itself |
  | ``1`` | ``ArgMaxDeviceOperation`` | the sampler's per-die argmax |

  Six independent per-layer tallies rather than stage 05's two. A window that
  straddled a boundary would have to get *all* of them wrong in the same
  direction to pass, which is why the boundary is checkable and not eyeballed.

Rows are taken in **file order**, not sorted by ``HOST START TS``: inside a
replayed trace every op reports the trace launch's host timestamp, so sorting by
it interleaves consecutive replays and makes the boundary unfindable.

    python window_full_model_48.py /tmp/prof_fm48_dec/reports/*/ops_perf_results_*.csv \\
        --out /tmp/fm48_decode_window.csv --layers 48
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

LAYERS = 48
#: token lookup + the cos and sin rows ``rope_decode_tables`` gathers.
EMBEDDINGS_PER_STEP = 3


def expected_counts(layers: int) -> dict:
    return {
        "ReduceScatterMinimalAsyncDeviceOperation": 2 * layers,
        "AllGatherAsyncDeviceOperation": 2 * layers,
        # The distributed argmax's two 4-wide gathers take ttnn.all_gather's
        # composite path, which is AllBroadcast + Concat, not AllGatherAsync.
        "AllBroadcastDeviceOperation": 2,
        "ConcatDeviceOperation": 2,
        "GatherDeviceOperation": 1,
        "SdpaDecodeDeviceOperation": layers,
        "SparseMatmulDeviceOperation": 2 * layers,
        "PagedUpdateCacheDeviceOperation": 2 * layers,
        "EmbeddingsDeviceOperation": EMBEDDINGS_PER_STEP,
        "ArgMaxDeviceOperation": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--manifest", type=Path, help="write a cut-point manifest for the raw capture")
    parser.add_argument("--relaxed", action="store_true", help="report the tally mismatches instead of asserting")
    args = parser.parse_args()

    with args.csv.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)  # file order == dispatch order; see the module docstring

    devices = sorted({row["DEVICE ID"] for row in rows})
    keep: set[int] = set()
    for device in devices:
        indices = [i for i, r in enumerate(rows) if r["DEVICE ID"] == device]
        embeddings = [i for i in indices if rows[i]["OP CODE"] == "EmbeddingsDeviceOperation"]
        if len(embeddings) < EMBEDDINGS_PER_STEP:
            raise SystemExit(
                f"device {device}: {len(embeddings)} embedding rows, need {EMBEDDINGS_PER_STEP} "
                "for one decode step -- is this a decode capture?"
            )
        start = embeddings[-EMBEDDINGS_PER_STEP]
        keep.update(i for i in indices if i >= start)
    window = [rows[i] for i in sorted(keep)]

    per_device = Counter()
    for row in window:
        per_device[(row["DEVICE ID"], row["OP CODE"])] += 1

    problems = []
    for device in devices:
        for op, want in expected_counts(args.layers).items():
            got = per_device[(device, op)]
            status = "ok" if got == want else "MISMATCH"
            if got != want:
                problems.append(f"device {device}: {op} = {got}, expected {want}")
            print(f"  boundary check  device {device}  {op:<48} {got:>5} / {want:<5} {status}")
    if problems and not args.relaxed:
        raise SystemExit("boundary check failed:\n  " + "\n  ".join(problems))

    tally = Counter(row["OP CODE"] for row in window)
    print(f"window: {len(window)} of {len(rows)} rows, {len(devices)} devices, {args.layers} layers")
    for op, count in tally.most_common():
        print(f"  {count:>5}  {op}")

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window)
    print(f"wrote {args.out}")

    # The raw capture is ~139 MB and is not archived, so the published window's
    # *internal* consistency is checkable but its **cut point** was not: nothing
    # in the tree said how many rows were discarded or what they were. This
    # manifest is the cheap half of that -- the raw file's size and digest, the
    # per-device row counts before and after the cut, and a digest of the
    # discarded rows -- so a re-run that still has the capture can prove it cut
    # in the same place.
    if args.manifest:
        digest = hashlib.sha256()
        with args.csv.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        discarded = hashlib.sha256()
        for index, row in enumerate(rows):
            if index not in keep:
                discarded.update(repr(sorted(row.items())).encode())
        manifest = {
            "raw_capture": args.csv.name,
            "raw_bytes": args.csv.stat().st_size,
            "raw_sha256": digest.hexdigest(),
            "raw_rows": len(rows),
            "raw_rows_per_device": {d: sum(1 for r in rows if r["DEVICE ID"] == d) for d in devices},
            "window_rows": len(window),
            "window_rows_per_device": {d: sum(1 for r in window if r["DEVICE ID"] == d) for d in devices},
            "discarded_rows": len(rows) - len(window),
            "discarded_rows_sha256": discarded.hexdigest(),
            "cut_index_per_device": {d: min(i for i in sorted(keep) if rows[i]["DEVICE ID"] == d) for d in devices},
            "cut_rule": ("the first of the last three EmbeddingsDeviceOperation rows per device, " "in file order"),
        }
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
