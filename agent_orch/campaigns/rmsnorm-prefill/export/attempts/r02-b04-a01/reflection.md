# r02-b04-a01 result: 0.8470 (ok)

## What happened vs expected
The run is valid, with accuracy bit-identical to the root (PCC 0.9999985, max_abs 0.0217-0.0243). It is much slower on
every shape. Root r01-b04-a04 in brackets: h3584 19.86 µs (14.17), h4096 21.33 (15.43), h6144 27.96 (19.08),
h7168 31.10 (20.98). That is +5.7 to +10.1 µs, a -30% regression. I expected -1.5 to -2.5 µs. The placement change
did engage: the workers are at virtual (6, 2..11) and (14, 2..11), and the forwarder is at (1, 2).

## Why (profiler evidence)
`percore.py` (in this node dir): `python3 percore.py $DREAM_HOME/rmsnorm-prefill/reports/r02-b04-a01 1 58`.
Device 1, h7168 call 58, µs from the first marker. Root per-core numbers come from the same script on
reports/r01-b04-a04.

| | root (rows y=2,3, x=1..14) | this node (cols x=6,14, y=2..11) |
|---|---|---|
| R_INPUT duration | 5.2-5.4 µs on all cores | 7.2-7.6 µs at y=2,3; 9.3-10.1 µs at y=4..11 |
| W_PUSH end - kernel start (AG start gate) | 6.6-8.6 | 8.6-12.5 |
| F_FABRIC + go (F_COLLECT end -> AG wait end) | ~2.7 | ~2.0 |
| drain duration (W_DRAIN start->end) | 6.6-9.2, rising with x | 5.8-7.1 at y=2,3,11; 8.4-15.0 at y=4..10, rising with y to y=9/10 |

- **The congestion gradient moved from x to y and got much worse.** With 10 workers stacked in one column, the
  input read and the drain both slow down with y. The worst cores are y=8-10, and y=11 is fast again. That is the
  same "through-traffic starves downstream injectors" signature the root showed along x (and r01-b03-a04 showed for
  NoC1 in reverse), but steeper. So the vertical links in a worker column are at least as much a shared hot spot as
  a worker row's horizontal links. BH routing probably puts one leg of every DRAM transaction in the worker's own
  column (or the DRAM column, entered close to it), so stacking workers in one column funnels all of them onto that
  column's links. Even the y=2,3 cores, which share a row with only one other worker, read ~2 µs slower than any
  root core. So the per-row horizontal link was never the only limit.
- Compute is unaffected. POST (AG wait end -> TRISC end) is ~5.8 µs, the same as the root. The AG itself is a bit
  shorter (~2.0 vs ~2.7 µs F_FABRIC + go). The whole loss is the slower read (AG start ~+3.9 µs, via the slowest
  W_PUSH at 26.75) and the slower drain (worst core finishes 9.6 µs after TRISC end vs ~3.7 µs in the root).
- Accuracy and the protocol are fine with arbitrary placement. The forwarder at (1,2) plus per-worker unicast
  go-sems work. No hang.

## Classification
flawed idea (as executed: column-stacked placement). Packing workers into whole columns is strictly worse than
packing them into rows on this chip. The broader question, whether a layout with at most ~2 workers per row AND
per column beats the row-major one, is still untested. It is weakly motivated now, because even the y=2,3 cores got
slower.

## What a child of this node should try next
1. **Don't retry column stacking.** If placement is revisited, use a diagonal or knight's-move layout (logical
   (x, y) = ((3*w) % grid_x, w % grid_y), checked for uniqueness) so no row or column holds more than 2 workers.
   Expect at best a small gain, and test it as one attempt only. It costs 2 kernel-config CoreRanges per worker
   (20 ranges), so watch dispatch.
2. Better levers for the root lineage are unchanged: the column split (r01-b03-a02/a03) ported onto r01-b04-a04,
   and the fixed PRE tail (reduce + transpose, see r01-b04-a04's reflection).
3. Generic lesson for this campaign: the drain and the read are limited by NoC link sharing among co-located cores,
   in BOTH dimensions. A future attempt should cut the bytes per shared link (fewer bytes, or traffic spread over
   both DRAM columns and both NoCs per core by destination) instead of moving cores around.
