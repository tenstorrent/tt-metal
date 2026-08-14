# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Slice the LAST traced decode iteration out of a full-model ops CSV.

``profile_full_model.py`` captures a whole process: the weight upload alone is
hundreds of ``Tilize``/``Typecast`` rows, and a `tt-perf-report` over the raw
file is dominated by them. Stage 04 solved the same problem with
``optimized_multichip_decoder/probes/window.py``, which re-derives the row range
from the CSV rather than trusting a transcribed constant. This is its full-model
equivalent.

The decode iteration boundary is checkable rather than eyeballed:

* it **starts** at the first of the last three ``EmbeddingsDeviceOperation``
  rows *per device*. ``decode_hidden`` opens with exactly three embedding
  gathers per token -- the token lookup, then the cos and sin rows that
  ``rope_decode_tables`` reads with ``ttnn.embedding`` -- so three per device is
  one decode step and nothing else in the graph uses the op;
* it **ends** at the last row of the file, because ``CopyDeviceOperation`` after
  ``ArgMaxDeviceOperation`` is the sampler writing ``tt_out_tok``;
* and it must contain, per device, exactly ``2 * layers``
  ``ReduceScatterMinimalAsyncDeviceOperation`` and ``2 * layers + 1``
  ``AllGatherAsyncDeviceOperation`` -- the layer's two all-reduces are each a
  reduce-scatter plus an all-gather, and the sampler adds one more gather of its
  own. A tally that does not match has straddled a boundary.

Rows are taken in **file order**, not sorted by ``HOST START TS``. Inside a
replayed trace every op reports the trace launch's host timestamp, so sorting by
it interleaves consecutive replays and makes the boundary unfindable; the file's
own order is the dispatch order.

    python .../probes/window_full_model.py /tmp/prof_fm_dec/reports/*/ops_perf_results_*.csv \\
        --out /tmp/fm_decode_window.csv
    tt-perf-report /tmp/fm_decode_window.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LAYERS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=LAYERS)
    args = parser.parse_args()

    with args.csv.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)  # file order == dispatch order; see the module docstring

    #: token lookup + the cos and sin rows ``rope_decode_tables`` gathers.
    EMBEDDINGS_PER_STEP = 3
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
    for device in devices:
        rs = per_device[(device, "ReduceScatterMinimalAsyncDeviceOperation")]
        ag = per_device[(device, "AllGatherAsyncDeviceOperation")]
        assert rs == 2 * args.layers, f"device {device}: {rs} reduce-scatters, expected {2 * args.layers}"
        assert ag == 2 * args.layers + 1, f"device {device}: {ag} all-gathers, expected {2 * args.layers + 1}"

    tally = Counter(row["OP CODE"] for row in window)
    print(f"window: {len(window)} of {len(rows)} rows, {len(devices)} devices, {args.layers} layers")
    for op, count in tally.most_common():
        print(f"  {count:>4}  {op}")

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
