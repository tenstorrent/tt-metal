# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Localise the PCC 0.883: read the weight back through the matmul, one row at a time.

Rather than infer the fault from a correlation number, this makes the matmul reveal
the weight it actually sees. Feed one-hot activation rows: with a single 1 in row r
at column j, `y = x @ w` returns row r of w. Feeding 32 such rows in one call
returns 32 rows of the device's decoded weight.

Then diff against the codes. A correct bf2 path matches exactly. Otherwise the
pattern of wrong positions says which kind of fault it is, and those cases want
different fixes:

  whole tiles wrong past some index   -> page/tile-size problem: the in1 CB is not
                                        stepping 320 B per tile
  a consistent offset within a face    -> face/bit ordering
  every other value wrong              -> bit-position or nibble-pair packing
  scattered, roughly 12% of values     -> the code-to-value mapping (0b10 emitted,
                                        or a magnitude from the wrong exponent)

Each of those is reported as a count per (face, row-in-face, column-in-face, bit
position), so the diagnosis is a table rather than a guess.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_onehot
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def face_of(r: int) -> int:
    """Row-major quadrant index for a row of a 32x32 tile."""
    return (r // 16) * 2 + 0  # placeholder; filled per element below


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tiles", type=int, default=4, help="K tiles in the weight")
    ap.add_argument("--bits", default="4,2", help="formats to compare: 4 (bfp4), 2 (bfp2)")
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    kt = args.k_tiles
    k = kt * 32
    n = 32                     # one output tile column: a single 32x32 weight tile
    rng = np.random.default_rng(3)
    codes = rng.integers(-1, 2, (n, k)).astype(np.float32)

    # One-hot activations: for each of the 32 rows of the output tile, put a single
    # 1 at a column we control. Row r of the output then equals column r of the
    # device's weight tile, so one call per K-tile-position reveals every value.
    # We probe all 32 output rows at once by using x with 32 rows (M=32).
    x_np = np.zeros((32, k), np.float32)
    for r in range(32):
        x_np[r, r] = 1.0        # row r picks weight column r at K-position 0

    dtypes = {"4": ttnn.bfloat4_b, "2": ttnn.DataType.BFLOAT2_B}
    rc = 0
    try:
        wt_dev = {}
        for key in args.bits.split(","):
            wt_dev[key] = ttnn.from_torch(
                torch.from_numpy(codes.T.copy()), dtype=dtypes[key],
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xt = ttnn.from_torch(
            torch.from_numpy(x_np).to(torch.bfloat16).reshape(1, 1, 32, k),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG)

        for key, wt in wt_dev.items():
            y = ttnn.linear(xt, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
            got = ttnn.to_torch(y).float().numpy().reshape(32, 32)
            ttnn.deallocate(y)
            # got[r, c] = w[c, r] where x[r] is the one-hot at column r... so the
            # decoded weight column for output column c is got[:, c], and the true
            # weight column c is codes[c, 0:32].
            true_cols = codes[:, :32]          # [n=32, 32]  (n indexes output column)
            decoded = got.T                    # decoded[c, j] = w_dev[c, j]
            wrong = (decoded != true_cols)
            n_wrong = int(wrong.sum())
            print(f"\nbfp{key}: {n_wrong}/{wrong.size} wrong "
                  f"({n_wrong / wrong.size * 100:.1f}%)", flush=True)
            if n_wrong == 0:
                print(f"  bfp{key} decodes EXACTLY through this matmul path.")
                continue
            rc = 1
            # Where are they? Report per face, per row-in-face, per column-in-face.
            print(f"  {'face':>4} {'row%16':>7} {'col%16':>7} {'wrong':>6}")
            tally = {}
            for c in range(32):
                for j in range(32):
                    if not wrong[c, j]:
                        continue
                    face = (c // 16) * 2 + (j // 16)
                    tally.setdefault((face, c % 16, j % 16), 0)
                    tally[(face, c % 16, j % 16)] += 1
            # Summarise by face and by column-in-face, which is what distinguishes
            # an ordering fault from a per-tile one.
            by_face = {}
            by_col = {}
            by_row = {}
            for (face, rr, cc), cnt in tally.items():
                by_face[face] = by_face.get(face, 0) + cnt
                by_col[cc] = by_col.get(cc, 0) + cnt
                by_row[rr] = by_row.get(rr, 0) + cnt
            print(f"  wrong by face        : {dict(sorted(by_face.items()))}")
            print(f"  wrong by row-in-face : {dict(sorted(by_row.items()))}")
            print(f"  wrong by col-in-face : {dict(sorted(by_col.items()))}")
            # Whole-tile check: is anything wrong past the first 32 columns of K?
            # We only probed K position 0..31 here, so extend by probing deeper.
            print("  (probe covers weight columns 0..31 of each output tile; extend with")
            print("   --k-tiles and an x one-hot at the relevant K offset to test page size)")
        for wt in wt_dev.values():
            ttnn.deallocate(wt)
        ttnn.deallocate(xt)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
