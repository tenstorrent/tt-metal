# r04-b04-a01 result: 1.3666 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the parent's values (same bytes, just no write ack).
No hang, so the posted counters behave correctly under DM_DYNAMIC_NOC on both NoCs (default NoC1 + alt NoC0).
Parent r03-b02-a02 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.59 | 12.32 | -0.27 (-2.2%) |
| h4096 | 14.05 | 13.64 | -0.41 (-2.9%) |
| h6144 | 17.55 | 17.35 | -0.20 (-1.1%) |
| h7168 | 19.32 | 18.65 | -0.67 (-3.5%) |

Score 1.3666 vs 1.3333 (+2.5%). Every shape is faster, three of four clearly outside the ±1% noise band, and h6144
sits right at its edge. This is the new campaign best. I predicted -0.5..-1.5 µs per shape and got -0.2..-0.7 µs, so the
lower end of that range.

## Why (profiler evidence)
`drain.py` (r03-b03-a03's script) plus `dsum.py` (per-shape medians, measured calls, all chips/workers; output in
`drain_out.txt`). Times are µs from each worker's W_DRAIN start:

| shape (tiles) | drain dur parent -> this | first POST unpack | pack POST end | drain end - pack POST end |
|---|---|---|---|---|
| h3584 (28) | 3.50 -> 3.32 | 0.55 / 0.55 | 2.48 / 2.48 | 1.02 -> **0.84** |
| h4096 (32) | 4.06 -> 3.86 | 0.54 / 0.54 | 2.73 / 2.73 | 1.32 -> **1.12** |
| h6144 (48) | 5.75 -> 5.33 | 0.55 / 0.55 | 3.77 / 3.78 | 1.98 -> **1.55** |
| h7168 (56) | 6.11 -> 5.65 | 0.54 / 0.54 | 4.29 / 4.29 | 1.82 -> **1.36** |

- Compute is untouched: POST start and pack's POST end are identical to the parent. The whole gain is drain rate.
- Per-tile drain rate after the first POST tile: parent ~99-110 ns/tile, this node ~91-104 ns/tile, about 7% faster.
  The saving grows with tiles (0.18 µs at 28 tiles, 0.43-0.46 µs at 48-56), which is what a per-tile cost removal looks
  like. The kernel-end gain is a bit larger than the drain gain on h4096/h7168. That is the usual cross-call coupling
  (a shorter tail evens out the next call's launch skew), or noise; it is not separately measured.
- So the ack/response path for non-posted writes was a real but **small** part of the per-core cap: ~7-10 ns of the
  ~100 ns per 2 KB tile. The drain still trails POST (~64-68 ns/tile) by 0.8-1.6 µs. The remaining per-core cap is not
  issue overhead (r03-b02-a03), VC (r03-b03-a03), aggregate bandwidth (r03-b04-a02) or the write ack (this node).
  What is left: NIU injection / L1 read port per packet, or the path (link back-pressure toward the 2 DRAM columns).
  2 KB per ~95 ns is ~16 B/cycle at 1.35 GHz, far below one link's 64 B/cycle, which still points at path contention
  near the DRAM columns rather than the source NIU.

## Classification
win (+2.5% geomean over the best node, every shape faster, 3 of 4 outside noise). Caveat for production: posted writes
have no completion ack, so at kernel end the last output tiles may still be in flight. The kernel waits for them to
*leave* L1 (posted flush) but not to land. For a production version, end with a fence: e.g. make the last tile per
DRAM bank a non-posted write and barrier on it (NoC ordering to the same endpoint on the same VC), or do one
non-posted 4 B write per bank after the drain and barrier.

## What a child of this node should try next
1. **Stack with the gamma streaming of r03-b04-a03** (writer W_GAMMA block only, orthogonal to this drain change). It
   removed the dev-0 h7168 straggler loop (-0.8 µs h7168 on its lineage). Expected ~1.38-1.39 if a sibling hasn't already
   ported it. Whichever lands first, combine the two.
2. **Make the posted drain production-safe** with the per-bank fence above, and check it costs nothing (it should be
   ~1 ack round trip at the very end).
3. **The drain is still ~1.4 µs behind POST on wide shapes.** Since issue, VC, ack and aggregate are ruled out, test the
   path: measure per-core drain rate vs core position and destination bank column (r02-b02-a01 / r01-b03-a04 tables) with
   posted writes on. If the gradient follows distance to the x=0/x=9 DRAM columns, re-tune the NoC0 share per core
   (r02-b02-a01 #1) now that acks no longer ride the return path.
4. Larger write packets are the other untested knob: two consecutive output tiles that land in the same bank (out_idx and
   out_idx+8) are not contiguous, so that needs a different output layout; not available inside this op.
