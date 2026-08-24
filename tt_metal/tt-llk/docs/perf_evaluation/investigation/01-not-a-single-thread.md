# 01. The jump does not come from any single thread

## Question

`L1_TO_L1` times the whole pipeline — unpack start to pack end. When one
configuration's measurement jumps by thousands of cycles, does any individual
thread jump too?

- If one does, that thread's own work is bistable and the pipeline inherits it.
- If none does, the time is lost in the interaction between threads.

## Method

Join the `L1_TO_L1` baseline against the isolates baseline on the full sweep
configuration, and compare each flagged configuration's pipeline movement against
its own movement in `UNPACK_ISOLATE`, `MATH_ISOLATE` and `PACK_ISOLATE`.

Both baselines are Wormhole, non speed-of-light, `TILE_LOOP` marker. 33,137
configurations matched.

## Result

Ten-run `L1_TO_L1` against five-run isolates, 42 flagged configurations:

| thread measured alone | moves >30 cycles | largest movement |
|---|--:|--:|
| `MATH_ISOLATE` | 0 of 42 | 18 cycles |
| `PACK_ISOLATE` | 3 of 42 | 1,287 cycles |
| `UNPACK_ISOLATE` | 9 of 42 | 200 cycles |

Against `L1_TO_L1` jumps with a **median of 4,086 cycles and a maximum of
13,965**.

The share of each pipeline jump explained by the largest single-thread movement:

| | |
|---|--:|
| 25th percentile | 0.0% |
| median | **0.2%** |
| 75th percentile | 1.7% |

Thirty-three of the forty-two have no thread moving at all.

The five-run `L1_TO_L1` set, matched to the isolates for run count, gives
0 of 27, 0 of 27, and 4 of 27.

## Conclusion

**For the large configurations — 118k to 218k cycles, jumps of 2,600 to 13,965 —
no thread accounts for more than 5% of the pipeline's movement, and usually none
of it.** The instability only exists when the three threads run together.

## What it does not say

**Five small configurations behave differently.** At 10k–19k cycles with jumps of
245–394 cycles, a thread sometimes moves as much as or more than the pipeline —
one shows a `PACK_ISOLATE` movement of 1,287 cycles against an `L1_TO_L1` jump of
246. Those are plausibly a separate, minor effect and should not be folded in.

**The isolate kernels are different binaries.** In `PACK_ISOLATE` the math thread
does only the synchronisation needed to keep the packer fed, not its real work.
So an isolate number is not that thread's share of the pipeline; it is that
thread under artificial conditions. "The packer is stable in isolate mode" means
*the packer is stable when math is not competing for DEST* — which is what you
would expect if the race needs both threads working.

**The run counts differ.** Ten runs for `L1_TO_L1`, five for the isolates, so the
isolates had fewer chances to reveal movement. The flips appear in one to four
runs out of ten, so five runs would usually catch one, but this is strong
evidence rather than airtight.

## Supporting numbers

`L1_TO_L1` runs 1.12x the slowest single thread at the median and 1.35x at the
third quartile. So 12–35% of pipeline time is not attributable to any thread —
fill, drain, and synchronisation. Comfortably large enough to hold a 2–6% jump.
