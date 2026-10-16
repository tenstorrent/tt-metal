# r02-b04-a03 result: 1.5234 (ok), neutral vs r02-b04-a01 (1.5314, -0.5%, inside noise); +11.9% vs parent r02-b04-a02 (1.3612)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the baseline. That
includes kimi-k3, which now runs the new 10 x 1-row layout. So the 0-receiver weight paths (the
reader's existing guards plus the writer guard added here) are correct, and no core hangs as its
own sender. The host built and the kernels JIT-compiled on the first try.

Per shape vs r02-b04-a01, whose op code this node restores (noise ±1%):

| shape | r02-b04-a01 | this | Δ | plan |
|---|---|---|---|---|
| kimi-k3 | 1397.0 | 1365.0 | **-2.3%** | 10 x 1-row (was 2x5) |
| minimax-m3 | 388.8 | 383.2 | -1.4% | unchanged |
| dsv4-pro | 1131.8 | 1125.3 | -0.6% | unchanged |
| glm | 486.3 | 491.6 | +1.1% | unchanged |
| mistral | 192.6 | 194.3 | +0.9% | unchanged |
| dsv4-flash | 322.4 | 327.2 | +1.5% | unchanged |
| dsv3 | 555.7 | 564.6 | +1.6% | unchanged |
| kimi-k2.7 | 767.5 | 773.1 | +0.7% | unchanged |
| gpt-oss | 277.1 | 286.2 | +3.3% | unchanged |

- The 8 unchanged shapes run byte-identical schedules. The C++ planner matches the Python replay
  on 24k random cases and on the test routing, and kimi-k3 is the only shape whose plan
  changes. Their ±1-3% is the known group-tail lottery.
- I predicted -6..-9% on kimi-k3 and got -2.3%.

## Why (profiler evidence)
TRISC1 kernel end per logical row, mean of the 10 measured calls (/tmp/r02b04a01/rows.py):
- r02-b04-a01 (2x5): 1360 1360 1342 1342 1352 1352 1213 1213 1184 1183. These are 6/6/6/5/5
  experts at pcM 2, i.e. 12 tile-rows on the critical rows, ~112 µs per tile-row.
- This node (1-row): 1049 1319 1321 1348 1337 1324 1330 1298 1277 837. The LPT deal gives row 0
  {4,4} = 8 tile-rows, rows 1-8 {4,3,3} = 10 and row 9 {4,3} = 7. Row end is proportional to
  tile-rows at ~130 µs per tile-row.
- So the critical work fell 12 -> 10 tile-rows (-17%), as modelled, but each tile-row got ~16%
  slower. Total traffic is 557 MB / 1365 µs = 408 GB/s, the same rate as before (~400 GB/s).
  kimi-k3 is DRAM-bound with either layout. Ten single-row streams (110 self-reading senders)
  draw no more bandwidth than five 2-row groups with multicast. Removing phantom rows and
  per-expert fixed costs only helps where compute, not DRAM, is the limit.
- The ~2% that remains comes from a better-balanced tail (rows 1-8 end within ~50 µs of each
  other, and only rows 0/9 idle early) rather than from extra bandwidth.

## Classification
neutral (within noise overall). Small real win on kimi-k3 (-2.3%, outside its noise), capped by
DRAM (~410 GB/s). The mechanism is correct, general (8- and 10-row grids, TILE path excluded by
define) and harmless: the planner takes it only when strictly better under the cost model.

## What a child of this node should try next
1. **Keep or drop freely.** It is worth ~0.25% geomean through kimi-k3. On this lineage it is now
   r02-b04-a01 plus a 1-row layout (the parent's down-stream change is reverted).
2. **kimi-k3 is at the DRAM floor (~408 GB/s) in both layouts.** Only fewer bytes or higher DRAM
   efficiency will move it further. Neither more streams (this node) nor fewer, larger groups
   (r02-b01-a01, r02-b03-a01) raise the aggregate above ~410-455 GB/s.
3. **The group-tail lottery is a gate/up-phase delivery effect**, from analysis of r02-b04-a02's
   zones with /tmp/r02b04a03/lot.py:
   - Slow groups have long RE-GU (mistral 168-173 vs ~110 µs) and short DN.
   - Sometimes only ONE row of a 2-row group is slow (mistral call 7: y4 GU 130 vs y5 173). That
     row's own x stream is the suspect: ~96 x 0.6-1 KB stick reads per block per row, issued by
     the rotating owner.
   - On gpt-oss the pcM-4 2-row groups' GU is a steady 125 µs (compute-bound), while the pcM-3
     3-row groups take 115-164 µs. Those are delivery-bound while every group streams gate/up at
     once.
   - Candidate probes:
     - (a) Read x with fewer, larger requests: one read per stick spanning 2 K-blocks, with
       the RM staging doubled.
     - (b) Move the x stick reads off the reader RISC of weight-sender rows.
     - (c) Add sender-side DM zones (read wait vs ready wait) on one shape before changing more.
4. Activation hiding (PACK-thread SFPU) is bounded by dst half-sync: one SFPU step can overlap only
   one MATH dst-half cycle. Single-expert groups (gpt-oss, minimax, mistral) also have no
   independent FPU work to overlap with. Don't expect more than ~1% from it without a
   cross-expert DM reorder.
