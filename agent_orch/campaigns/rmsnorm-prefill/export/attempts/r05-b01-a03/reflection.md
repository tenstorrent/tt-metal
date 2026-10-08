# r05-b01-a03 result: 1.4661 (ok)

## What happened vs expected
The result is valid on every shape on the first run. PCC is 0.9999985 and max_abs 0.0204-0.0237, the same as the
parent. Score 1.4661 vs the parent r05-b01-a02 at 1.4320 (+2.4% geomean). µs, chip mean:

| shape | 2-wave r05-b01-a01 | parent (4 waves, gated) | this (4 waves, un-gated) | vs parent |
|---|---|---|---|---|
| h3584 | 11.19 | 13.02 | 12.14 | -6.7% |
| h4096 | 12.08 | 13.28 | 12.94 | -2.6% |
| h6144 | 14.59 | 15.30 | 15.12 | -1.2% (≈ noise) |
| h7168 | 15.84 | 17.04 | 17.27 | +1.3% (≈ noise) |

The repair did what it was meant to: the pushes are now staggered per wave. But 4 waves still lose to the 2-wave
best (1.5696) on every shape, including the min-chip kernel time:

| shape | 2-wave min chip | this node min chip |
|---|---|---|
| h3584 | 9.65 | 10.63 |
| h7168 | 15.71 | 16.99 |

So host launch skew doesn't explain the loss. I expected ~13 µs at h7168 and got 17.3. The model's
"fixed ~2.6 µs AG chain per wave" premise is wrong (see below).

## Why (profiler evidence)
Analysis outputs (in `analysis/`):
- `wavesN_out.txt` from the parent's `wavesN.py <report> 4`.
- `fwd_out.txt` from r05-b01-a01's `fwd.py`.

All values are medians of per-call maxima, in µs from each chip's first worker start.

1. **The pushes are un-gated.**

   | | h3584 (this) | h3584 (parent) | h7168 (this) | h7168 (parent) |
   |---|---|---|---|---|
   | W_PUSH end, waves 0-3 | 2.54 / 3.13 / 3.91 / 4.77 | 4.40 / 4.36 / 4.44 / 4.81 | 3.78 / 4.48 / 5.79 / 7.22 | 3.64 / 6.68 / 6.92 / 7.21 |

   Each push is now read end + 0.7-1.1 µs.
   F_SEND#0..3 starts at h3584 are 3.07 / 3.43 / 4.09 / 4.88 (parent 4.65 / 4.94 / 5.17 / 5.41). The sends are
   staggered as designed.
2. **The gos did not follow, and the forwarder release is now the serial bottleneck.**
   - At h3584, F_SEND#0 ends at 3.34 but F_GO#0 starts at 6.46. That is 3.1 µs, mostly cross-chip launch skew: the
     min over chip-calls is 3.98, so in-sync chips take ~0.6-1.2 µs.
   - By the time the slowest chip's wave-0 packet lands, the later waves' packets have landed too. The forwarder
     then releases them back to back.
   - Each F_GO is ~0.52 µs: a write barrier plus 20 serial go incs. F_GO#k+1 starts right after F_GO#k ends.
   - The 4 gos therefore span 6.46 -> 9.14 at h3584 and 8.29 -> 11.25 at h7168. The 2-wave node spans only 1 gap.
   - Last-wave go is 9.21 at h3584, vs 7.86 for 2-wave wave B. The ~1.35 µs difference is the h3584 loss.
3. **The drain doesn't hide at h6144/h7168.**
   - Wave 0's drain starts at 9.37 at h7168. That is later than 2-wave wave A's (9.05), even though wave 0 pushed
     0.7 µs earlier: wave 0's go waits on skew.
   - The four drains then overlap and are DRAM-write aggregate-bound. The write window is 9.37 -> 16.98 = 7.6 µs,
     vs 9.05 -> 15.59 = 6.5 µs for 2 waves.
   - Per-core drain medians are 2.8-3.8 µs for 14 tiles. Wave 0's max end is 15.68, so it drains alongside wave 3.
   - Quarter-row drains from 80 cores are not faster in aggregate than half-row drains from 40.
4. **The reads are unchanged.** Last read ends at 3.93 / 6.18 µs, the same as the parent.

## Classification
win vs the parent (repair confirmed, +2.4%, h3584/h4096 well outside noise). The bug, the push gated by the streamed
gamma trid barrier, is fixed: the pushes and sends now step up per wave. But the 4-wave idea is weak as it stands:
- the cross-chip skew plus the serial forwarder release (~0.52 µs per wave) bunch the gos;
- the overlapping drains are aggregate-bound.

So 4 waves remain below the 2-wave best (1.4661 vs 1.5696) even with the fix.

## What a child of this node should try next
1. **Port this exact fix to the 2-wave best r05-b01-a01.** Its wave B push sits on W_GAMMA end: B push 7.24 vs read
   end 6.29 at h7168. The change is 6 writer lines (`while (!noc.is_read_trid_flushed(t)) poll_stick();` before the
   chunk barrier). It is the most likely new campaign best, at ~0.3-0.9 µs on B's chain.
2. **If 4 waves are pursued further, cut the serial forwarder release**, because the gos are now back to back. Two
   ways to do it:
   - drop the write barrier before the go incs (flush, then inc, r04-b03-a01 style);
   - release all waves whose data has landed in a single pass, with one inc per worker per pass.

   Each F_GO is 0.52 µs, so 4 waves pay ~2 µs of serial releases after the skew-gated wave 0.
3. **Don't add more waves past 2 without fixing the skew + release bunching.** The per-wave AG cost is dominated by
   cross-chip launch skew (send -> go median ~3.1 µs vs ~0.6-1.2 µs in sync). Waves only pipeline when the release is
   cheap. A per-shape S (S=2 for every campaign shape) is the safe default.
4. The drain at h6144/h7168 is DRAM-write aggregate-bound (~7 µs window for the full output). Fewer, longer drains
   are no worse than many short ones. Any further gain on the wide shapes must start the first drain earlier, not
   split the output finer.
