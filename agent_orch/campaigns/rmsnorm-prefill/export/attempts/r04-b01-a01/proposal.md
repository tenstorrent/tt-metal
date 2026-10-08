# r04-b01-a01: port r03-b04-a03's streamed gamma (8-page sticky-trid chunks, pushed 2 chunks behind the issue front) onto the best node r03-b02-a02

## Motivation
r03-b04-a03 (1.3318) found a cross-call straggler loop: on dev 0 one worker (usually (2,2)) gets its whole-row gamma
after its PRE, so its x*gamma pre-pass (~3.3 µs at h7168) runs after the AG, its POST starts 2-2.5 µs late, it finishes
last, and launches ~1.4 µs late in the next call. Its late stick also gates F_COLLECT, and the AG couples all four chips.
Streaming gamma to compute in chunks removed the loop (straggler calls 9/10 -> 0/10 on dev0 h7168) and cut h7168 from
19.53 to 18.72 µs. r03-b04-a03's `late2_r03-b02-a02.txt` shows the best node r03-b02-a02 still has the straggler in
9/10 h7168 calls (last-med drain 1.58 µs). r03-b04-a03 reflection #1 names this port as the cheapest likely new best.

## Mechanism
Writer-only (`dit_rmsnorm_fused_worker_writer.cpp`, W_GAMMA block), identical to r03-b04-a03's diff:
- gamma face-row reads issued in chunks of NUM_DRAM_BANKS (8) pages, each chunk tagged with a sticky read trid 1..4;
- a chunk is barrier-waited (TXN_ID barrier on its trid) and `push_back`-ed to weight_cb 2 chunks behind the issue
  front, so compute's cumulative per-block `cb_weight.wait_front` can start x*gamma as soon as PRE ends;
- bank rotation within each chunk (start page = tile_row_start % 8);
- stick push polled between reads and chunks, as before; default read trid restored to 0 afterwards.
The r03-b02-a02 writer is byte-identical to r03-b04-a01's writer (the base of r03-b04-a03's diff), so the patch
applies unchanged. Compute, factory and the L1 stats scratch are untouched.

## Why this is not a repeat
It's a combination: r03-b04-a03's gamma streaming was measured only on the r03-b04-a01 lineage (3-pass row-0 SFPU
combine, DRAM stats scratch). r03-b02-a02 has the fused add_rsqrt combine and the L1 scratch but whole-row gamma. The
two touch disjoint code (writer W_GAMMA block vs compute combine / factory scratch) and attack different costs
(cross-call straggler vs post-AG fixed costs).

## Expected effect and risk
h7168 19.32 -> ~18.5-18.7 µs; h6144 maybe -0.1; h3584/h4096 unchanged (2-3 µs gamma slack). Score ~1.34-1.35.
Risk is low: the code ran valid on the sibling lineage. Trid use: the post-go stick reads use the default trid 0
(restored), the drain uses writes. If the straggler doesn't vanish, `late2.py` from r03-b04-a03 will show it.
