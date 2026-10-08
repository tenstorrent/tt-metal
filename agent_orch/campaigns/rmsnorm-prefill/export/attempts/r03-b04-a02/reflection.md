# r03-b04-a02 result: 1.2440 (ok)

## What happened vs expected
The run is valid. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the same as the parent. So the wave slot mapping,
the per-wave DRAM page offsets and the 16-bit per-wave semaphore fields are all correct, and the run neither hung nor
desynced across the 13 calls per shape. But the node is slower on every shape. Parent r03-b04-a01 -> this node
(µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.89 | 13.76 | +0.87 (+6.7%) |
| h4096 | 14.20 | 14.78 | +0.58 (+4.1%) |
| h6144 | 17.83 | 18.78 | +0.95 (+5.3%) |
| h7168 | 19.53 | 20.73 | +1.20 (+6.1%) |

Score is 1.2440 vs 1.3131 (-5.3%), far outside the ±1% noise. I expected -1 to -3 µs and got +0.6 to +1.2 µs.

## Why (profiler evidence)
`waves.py` is in this dir: `python3 waves.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Wave B = cores with an
R_WAVEWAIT zone. Values are medians over measured calls x 4 chips of the per-call max over the wave's workers, in µs
from the first BRISC kernel start. The outputs are `waves_out.txt` (this node) and `waves_parent_out.txt`.

h7168, parent (all 20 workers) vs wave A / wave B:

| | parent | wave A | wave B |
|---|---|---|---|
| R_INPUT start -> end | 0.1 -> 6.38 | 0.1 -> 4.19 | 3.80 -> 7.87 |
| W_PUSH end (stick at forwarder) | 8.10 | 6.88 | 10.48 |
| fabric send -> go | 8.03 -> ~11.2 | 6.77 -> 9.12 | 10.37 -> 12.66 |
| AG wait end | 11.50 | 9.45 | 12.99 |
| TRISC end | 16.81 | 15.59 | 18.21 |
| drain start -> end | 12.20 -> 18.82 | 10.14 -> 18.15 | 13.79 -> 20.32 |

The pipeline itself works: wave B's read starts while A's tail lands, A's gather runs during B's read, and each wave's
gather takes ~2.3 µs from send to go. But four of the phases are **per-core-bound, not aggregate-bound**, so a wave of
10 cores takes much more than half the time of 20 cores:

1. **The read.** 10 cores with lookahead 8 read 1.12 MB in 4.1 µs, which is 273 GB/s. The parent's 20 cores ran at
   355 GB/s. Per-core rate went up (13.7 vs 8.9 tiles/µs), but not 2x. Both waves together take 7.9 µs vs 6.4 µs.
2. **PRE.** W_PUSH - R_INPUT grew from 1.7 µs to 2.6-2.7 µs in both waves. A wave's reads now arrive faster than PRE
   (DST-accumulated HiFi4 x*x, ~110 ns/tile) can consume them, so the stick is ready ~6.6-6.8 µs after the read starts,
   the compute time for 56 tiles. In the parent, read and PRE rates were matched, so the tail was shorter. Wave B's
   stick is at 10.48 vs the parent's 8.10. That +2.4 µs is the core of the loss.
3. **The drain is per-core-bound.** Wave A drained alone from 10.1 to 13.8 µs, with no other writer on the chip. It
   still ran at the parent's per-core rate (~7-8 tiles/µs, ~16 GB/s per core) and ended at 16.9-18.2. Same at h3584:
   28 tiles in 3.6-3.9 µs per core, both alone and with 20 concurrent writers. If the drain were aggregate-bound (DRAM
   or shared links), 10 lone writers would finish about 2x faster per core. output_cb holds 2 rows, so POST is not
   gated by the drain. TRISC ends 2-2.5 µs before the drain on every wave and every shape: the writer itself (or its
   per-core outstanding non-posted writes) is the limit.
4. So both waves' drain chains are as long as the parent's, and B's starts ~1.5 µs later. Kernel end is +1.2 µs at
   h7168. The same pattern holds on all shapes: h3584 B stick 6.53 vs parent 4.90, kernel 13.82 vs 12.62.

## Classification
flawed idea (at 20 workers). Staggering the cores in time only pays if each phase is aggregate-bound. Here PRE
(~110 ns/tile), the drain (~16 GB/s per core) and, partly, the read are per-core-bound, so 10 cores per wave stretch
each wave. The plumbing is correct and reusable:
- the forked forwarder (`dit_rmsnorm_wave_forwarder.cpp`) with send/release decoupled per wave;
- the 16-bit per-wave fields in arrival and out_ready (the fabric's fused inc carries a full 32-bit value);
- the reader start_sem gate.

## What a child of this node should try next
1. **Revert to the parent's single wave** (or keep `use_waves=false`). Don't retry waves on 20 cores.
2. **The drain is per-core-bound. That is this node's most useful finding, and the biggest lever left.** At h7168 the
   drain trails TRISC end by ~2 µs on every core, even with half the chip idle. Candidates:
   - Posted writes for the output tiles (`noc.async_write` with the posted option), with one posted flush before
     each `pop_front` and a final barrier. Non-posted writes keep an ack round trip in flight per transaction, and
     ~16 GB/s per core fits "a few KB outstanding x ~1 µs DRAM write latency".
   - Drop the per-block `async_writes_flushed()`. The output_cb holds 2 full rows, so pop cumulatively per row.
   - Issue 2-4 blocks ahead.
   Check with this node's `waves.py` (drain start/end vs TRISC end per core). The goal is drain end within ~0.3 µs of
   TRISC end.
3. **PRE is per-core compute-bound at ~110 ns/tile** (the HiFi4 ELWMUL x*x into DST). A lower fidelity for the x*x
   only (HiFi2/LoFi via the LLK init with explicit fidelity; untried since r01-b01-a04 suggested it) would shorten the
   PRE tail by up to ~1 µs once reads are fast. The parent's read and PRE are about matched, so pair it with the deeper
   read (this node's lookahead 8 lifted per-core read rate 54%).
4. Waves become worth a retry only with more cores than rows: e.g. a k=2 column split (40 workers, as two waves of
   20). Or after (2) and (3) make the per-core rates fast enough that the phases become aggregate-bound.
