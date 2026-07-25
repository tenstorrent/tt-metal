#!/usr/bin/env python3
"""Root-cause the perfect cross-position correlation in the device Gumbel draw (#48291).

max|r| == 1.0 across canvas positions means whole noise ROWS are identical, not merely
dependent. This probe characterises the duplication so the fix targets the mechanism:

* how many positions are exact duplicates of an earlier position;
* whether the duplication is periodic, and with what period (a period of 32 or a multiple
  points at TILE-granular reuse inside ``ttnn.rand``);
* whether it is present in the raw 2-D ``ttnn.rand((vocab, inner))`` draw itself, i.e. before
  the permute/reshape, which separates an RNG defect from a layout defect.
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/home/zni/tt-metal"))

import ttnn  # noqa: E402
from models.experimental.diffusion_gemma.tt import sampling as TS  # noqa: E402

CANVAS = 256
VOCAB = int(os.environ.get("PROBE_VOCAB", "16384"))


def dup_structure(rows: torch.Tensor, label: str) -> None:
    """Report exact-duplicate structure of ``[n, width]`` rows."""
    n = rows.shape[0]
    seen = {}
    first_owner = [-1] * n
    for index in range(n):
        key = hash(rows[index].numpy().tobytes())
        if key in seen:
            first_owner[index] = seen[key]
        else:
            seen[key] = index
    duplicates = [(i, first_owner[i]) for i in range(n) if first_owner[i] >= 0]
    offsets = sorted({i - owner for i, owner in duplicates})
    print(f"[{label}] rows={n} unique={len(seen)} duplicated={len(duplicates)}")
    if duplicates:
        print(f"[{label}] duplicate offsets (i - first_owner), up to 12: {offsets[:12]}")
        print(f"[{label}] first 8 duplicate pairs: {duplicates[:8]}")
        # A single dominant offset is the signature of periodic reuse.
        from collections import Counter

        common = Counter(i - owner for i, owner in duplicates).most_common(4)
        print(f"[{label}] most common offsets: {common}")


def periodicity(rows: torch.Tensor, label: str) -> None:
    """For candidate periods, what fraction of rows equal the row one period earlier?"""
    n = rows.shape[0]
    for period in (1, 2, 4, 8, 16, 32, 64, 128):
        if period >= n:
            continue
        matches = torch.all(rows[period:] == rows[:-period], dim=-1)
        frac = float(matches.float().mean())
        if frac > 0.0:
            print(f"[{label}] period {period:3d}: {frac:.4f} of rows identical to row-{period}")


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        # 1) the production path, as the model calls it
        noise = TS.sample_gumbel_noise_with_permuted_vocab((1, 1, CANVAS, VOCAB), device=device, seed=48291)
        host = ttnn.to_torch(noise).float().reshape(-1, VOCAB)[:CANVAS]
        noise.deallocate(True)
        print(f"=== production permuted-vocab Gumbel, shape {tuple(host.shape)} ===")
        dup_structure(host, "permuted")
        periodicity(host, "permuted")

        # 2) the RAW 2-D rand, before permute/reshape -- is the defect in rand or in the layout?
        raw = ttnn.rand(
            (VOCAB, CANVAS),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
            seed=48291,
        )
        raw_host = ttnn.to_torch(raw).float()  # [vocab, inner]
        raw.deallocate(True)
        print(f"\n=== raw ttnn.rand((vocab={VOCAB}, inner={CANVAS})), shape {tuple(raw_host.shape)} ===")
        # columns of the raw draw == canvas positions, so transpose to compare position rows
        columns = raw_host.t().contiguous()
        dup_structure(columns, "raw_columns")
        periodicity(columns, "raw_columns")
        # and the vocab axis of the raw draw, for symmetry
        dup_structure(raw_host[:512], "raw_rows_first512")

        # 3) a 1-D-ish control: does a single tall draw duplicate too?
        tall = ttnn.rand(
            (CANVAS, VOCAB),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
            seed=48291,
        )
        tall_host = ttnn.to_torch(tall).float()
        tall.deallocate(True)
        print(f"\n=== control ttnn.rand((canvas={CANVAS}, vocab={VOCAB})) -- vocab innermost ===")
        dup_structure(tall_host, "tall_rows")
        periodicity(tall_host, "tall_rows")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
