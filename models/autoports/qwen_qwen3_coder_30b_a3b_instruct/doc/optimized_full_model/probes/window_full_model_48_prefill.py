# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Slice the LAST 48-layer **prefill** pass out of an ops CSV, and prove it is one pass.

Stage 03 published a window that straddled two iterations and invalidated eight
figures; every window in this project has been boundary-checked since. Prefill
needs a different check from decode, because it is eager rather than traced and
has no three-embedding opening to anchor on -- its rotary comes from a slice of
the precomputed tables, not from a per-token ``ttnn.embedding`` gather.

What anchors it instead is that ``prefill_hidden`` opens with **exactly one**
``EmbeddingsDeviceOperation`` per pass (the token lookup), and that
``profile_full_model_48_prefill.py`` runs the *same* prefill three times -- one
warm-up and two measured -- with no ``reset()`` in between. So:

* the window is from the **last** ``EmbeddingsDeviceOperation`` row to the end
  of the file, per device;
* and the block before it, from the second-to-last embedding to the last, must
  be the **identical sequence of op codes, row for row**. That is a strictly
  stronger boundary check than decode's ten tallies: a straddled boundary would
  have to reproduce several thousand op codes in the same order to pass;
* on top of which the per-layer tallies below are asserted per device, exactly
  as in ``window_full_model_48.py``, so a systematic mis-slice that happened to
  be self-similar would still be caught.

Rows are taken in **file order**, not sorted by ``HOST START TS`` -- the same
rule as the decode windower.

    python window_full_model_48_prefill.py \\
        /tmp/prof_fm48_pf/reports/*/ops_perf_results_*.csv \\
        --out /tmp/fm48_prefill_window.csv --layers 48
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LAYERS = 48
#: The prompt length ``profile_full_model_48_prefill.py`` captures at.
PROMPT_LEN = 128


#: ``functional_decoder.EXPERT_CHUNK_SIZE`` -- prefill's MoE walks the sequence
#: in 32-row blocks, so its ``SparseMatmul`` count is length-dependent where
#: decode's is not.
EXPERT_CHUNK_SIZE = 32


def expected_counts(layers: int, seq_len: int) -> dict:
    """Per-device op tallies for exactly one 48-layer prefill pass.

    Fourteen independent tallies, each with a reason. Prefill's collectives are
    the same two all-reduces per layer as decode's, but almost everything else
    differs, and each difference is a separate check:

    * the attention op is the full-sequence ``SDPAOperation``, not
      ``SdpaDecode``, once per layer;
    * the cache write is ``paged_fill_cache`` (K and V) rather than
      ``paged_update_cache``;
    * the rotary is ``RotaryEmbedding`` (the prefill spelling, Q and K per
      layer), not ``RotaryEmbeddingHf``;
    * the experts are **chunked**: ``moe_prefill_optimized`` pads the sequence
      to a multiple of ``EXPERT_CHUNK_SIZE`` and runs the gate/up and down
      ``SparseMatmul`` pair once per 32-row block, then ``Concat``s the blocks
      back -- so ``2 * layers * ceil(S/32)`` sparse matmuls and ``layers``
      concats, plus the two the sampler's composite all-gathers contribute;
    * there is exactly **one** token embedding for the whole pass (prefill's
      rotary reads a slice of the precomputed tables rather than gathering per
      position), which is also what anchors the window;
    * and the terminal path is the same distributed argmax decode uses --
      prefill samples the last row on device -- so ``ArgMax``, ``Gather`` and
      the two ``AllBroadcast`` halves of the 4-wide composite gathers appear
      once each per pass.
    """
    chunks = -(-seq_len // EXPERT_CHUNK_SIZE)
    return {
        "ReduceScatterMinimalAsyncDeviceOperation": 2 * layers,
        "AllGatherAsyncDeviceOperation": 2 * layers,
        "SDPAOperation": layers,
        "SparseMatmulDeviceOperation": 2 * layers * chunks,
        "PagedFillCacheDeviceOperation": 2 * layers,
        "RotaryEmbeddingDeviceOperation": 2 * layers,
        "TopKDeviceOperation": layers,
        "NlpCreateHeadsDeviceOperation": layers,
        "NLPConcatHeadsDeviceOperation": layers,
        "ConcatDeviceOperation": layers + 2,
        "EmbeddingsDeviceOperation": 1,
        "ArgMaxDeviceOperation": 1,
        "AllBroadcastDeviceOperation": 2,
        "GatherDeviceOperation": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--seq-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--relaxed", action="store_true", help="report the tally mismatches instead of asserting")
    args = parser.parse_args()

    with args.csv.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)  # file order == dispatch order

    devices = sorted({row["DEVICE ID"] for row in rows})
    keep: set[int] = set()
    problems: list[str] = []

    for device in devices:
        indices = [i for i, r in enumerate(rows) if r["DEVICE ID"] == device]
        embeddings = [i for i in indices if rows[i]["OP CODE"] == "EmbeddingsDeviceOperation"]
        if len(embeddings) < 2:
            raise SystemExit(
                f"device {device}: {len(embeddings)} embedding rows, need at least 2 "
                "(one warm-up pass and one measured) -- is this a prefill capture?"
            )
        window_start, previous_start = embeddings[-1], embeddings[-2]
        window = [i for i in indices if i >= window_start]
        previous = [i for i in indices if previous_start <= i < window_start]
        # The repeat check: the pass before this one must be the same program.
        window_codes = [rows[i]["OP CODE"] for i in window]
        previous_codes = [rows[i]["OP CODE"] for i in previous]
        if window_codes != previous_codes:
            first_difference = next(
                (k for k in range(min(len(window_codes), len(previous_codes))) if window_codes[k] != previous_codes[k]),
                min(len(window_codes), len(previous_codes)),
            )
            problems.append(
                f"device {device}: the published pass ({len(window_codes)} ops) is not the same program as "
                f"the one before it ({len(previous_codes)} ops); first difference at index {first_difference}"
            )
        else:
            print(f"  repeat check   device {device}  {len(window_codes)} ops, identical to the preceding pass  ok")
        keep.update(window)

    window_rows = [rows[i] for i in sorted(keep)]

    per_device = Counter()
    for row in window_rows:
        per_device[(row["DEVICE ID"], row["OP CODE"])] += 1
    for device in devices:
        for op, want in expected_counts(args.layers, args.seq_len).items():
            got = per_device[(device, op)]
            status = "ok" if got == want else "MISMATCH"
            if got != want:
                problems.append(f"device {device}: {op} = {got}, expected {want}")
            print(f"  boundary check  device {device}  {op:<48} {got:>5} / {want:<5} {status}")

    if problems and not args.relaxed:
        raise SystemExit("boundary check failed:\n  " + "\n  ".join(problems))

    tally = Counter(row["OP CODE"] for row in window_rows)
    print(f"window: {len(window_rows)} of {len(rows)} rows, {len(devices)} devices, {args.layers} layers")
    for op, count in tally.most_common():
        print(f"  {count:>5}  {op}")

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
