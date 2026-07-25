#!/usr/bin/env python3
"""Confirm the ttnn.rand tile-reuse pattern and test a DG-local workaround (#48291).

Hypothesis from the duplication scan: within every 32-row TILE, ``ttnn.rand`` fills only 24
distinct row streams and rows 24..31 repeat rows 0..7. Predicted duplicate set is therefore
exactly ``{i : i % 32 >= 24}`` -- 8 of every 32 rows, 64 of 256, offset 24, independent of the
other axis extent. That is what the scan measured at both vocab 16384 and 262144.

If the pattern holds, a DG-local workaround needs no new op: draw only the USABLE 24 columns per
tile and stitch enough tiles to cover the canvas. Two candidates are measured here:

  A. ``narrow``    -- one ``rand((vocab, 24))`` per chunk with a distinct seed, 11 chunks for 256.
  B. ``strided``   -- one big ``rand((vocab, 32*11))`` then keep columns with ``col % 32 < 24``.

Both are checked against the same IID metrics as the gate test: exact-duplicate count and
flat-logit winner multiplicity across the 256 canvas positions.
"""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/home/zni/tt-metal"))

import ttnn  # noqa: E402

CANVAS = 256
VOCAB = int(os.environ.get("PROBE_VOCAB", "16384"))
TILE = 32
USABLE = 24
SEED = 48291


def _rand(device, shape, seed):
    return ttnn.rand(shape, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, low=0.0, high=1.0, seed=seed)


def dup_report(positions: torch.Tensor, label: str) -> dict:
    """positions: [n, width]; report duplicate structure + flat-logit winner multiplicity."""
    n = positions.shape[0]
    seen, owner = {}, [-1] * n
    for i in range(n):
        key = hash(positions[i].numpy().tobytes())
        if key in seen:
            owner[i] = seen[key]
        else:
            seen[key] = i
    dups = [i for i in range(n) if owner[i] >= 0]
    offsets = Counter(i - owner[i] for i in dups).most_common(3)
    winners = positions.argmax(dim=-1)
    counts = torch.bincount(winners)
    stats = {
        "unique": len(seen),
        "duplicated": len(dups),
        "offsets": offsets,
        "distinct_winners": int((counts > 0).sum()),
        "max_multiplicity": int(counts.max()),
    }
    print(
        f"[{label}] unique={stats['unique']}/{n} duplicated={stats['duplicated']} "
        f"offsets={stats['offsets']} distinct_winners={stats['distinct_winners']}/{n} "
        f"max_multiplicity={stats['max_multiplicity']}"
    )
    return stats


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        # --- 1) confirm the predicted duplicate set exactly ---
        raw = _rand(device, (VOCAB, CANVAS), SEED)
        columns = ttnn.to_torch(raw).float().t().contiguous()
        raw.deallocate(True)
        seen, owner = {}, [-1] * CANVAS
        for i in range(CANVAS):
            key = hash(columns[i].numpy().tobytes())
            if key in seen:
                owner[i] = seen[key]
            else:
                seen[key] = i
        observed = {i for i in range(CANVAS) if owner[i] >= 0}
        predicted = {i for i in range(CANVAS) if i % TILE >= USABLE}
        print(f"=== tile-reuse hypothesis (vocab={VOCAB}) ===")
        print(f"[hypothesis] predicted duplicate set == observed: {observed == predicted}")
        print(f"[hypothesis] |observed|={len(observed)} |predicted|={len(predicted)}")
        if observed != predicted:
            print(f"[hypothesis] observed-only={sorted(observed - predicted)[:12]}")
            print(f"[hypothesis] predicted-only={sorted(predicted - observed)[:12]}")
        # does every duplicate map to i-24 within its own tile?
        in_tile = all(owner[i] == i - USABLE for i in sorted(observed))
        print(f"[hypothesis] every duplicate is exactly row i-{USABLE}: {in_tile}")
        print("\n=== baselines ===")
        dup_report(columns, "production")

        # --- 2) candidate A: narrow per-chunk draws, distinct seed each ---
        chunks = []
        num_chunks = -(-CANVAS // USABLE)
        for chunk_index in range(num_chunks):
            part = _rand(device, (VOCAB, USABLE), SEED + 7919 * (chunk_index + 1))
            chunks.append(ttnn.to_torch(part).float().t().contiguous())
            part.deallocate(True)
        narrow = torch.cat(chunks, dim=0)[:CANVAS]
        dup_report(narrow, f"A:narrow({num_chunks}x{USABLE})")

        # --- 3) candidate B: one wide draw, keep only the usable columns of each tile ---
        wide_cols = num_chunks * TILE
        wide = _rand(device, (VOCAB, wide_cols), SEED)
        wide_host = ttnn.to_torch(wide).float().t().contiguous()
        wide.deallocate(True)
        keep = [c for c in range(wide_cols) if c % TILE < USABLE][:CANVAS]
        dup_report(wide_host[keep], f"B:strided({wide_cols}->{CANVAS})")

        # --- 4) host IID control, same metrics ---
        generator = torch.Generator().manual_seed(SEED)
        host = torch.rand((CANVAS, VOCAB), generator=generator, dtype=torch.float32)
        dup_report(host, "host-IID")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
