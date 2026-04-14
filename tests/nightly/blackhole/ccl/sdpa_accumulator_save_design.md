# SDPA Ring Writer — Accumulator Save Design

## Problem

In ring SDPA with multiple Q chunks per core, intermediate accumulators (output,
running max, running sum) must round-trip through DRAM between ring iterations.
The writer kernel saves accumulators after each Q chunk's K-loop completes, and
restores them at the start of the next ring iteration.

The save happens right after compute signals the last K-chunk (`cb_signal`). At
this point, mcast synchronization has aligned all cores in the ring — they all
hit the save window at roughly the same time. When every core writes its full
output accumulator to DRAM simultaneously, the aggregate DRAM bank pressure
causes cross-NOC contention: writer NOC1 output writes compete with reader NOC0
K/V reads at the physical DRAM bank level. This propagates through the mcast
semaphore chain and causes WAIT-K spikes on downstream cores.

## Solution: Split Save with Deferred Flush

The save is split into two temporally separated halves:

1. **Immediate save** (after `cb_signal`): Write the first half of output rows
   to DRAM, pop them from `cb_out`. Leave the remaining rows in `cb_out`.

2. **Deferred flush** (next Q's processing window): After the read barrier for
   the next Q's restore completes, write the remaining rows from `cb_out` to
   DRAM and pop them.

The split point is `num_row_groups / 2` where `num_row_groups = Sq_chunk_t /
out_subblock_h`. This is purely derived from the Q chunk size and subblock
height — no hardcoded tile counts.

### Why halving works

The key insight is that the contention threshold depends on how many tiles all
cores write simultaneously. Halving the burst size halves the peak DRAM pressure
in each window. Additionally, the two windows are separated by a read barrier
(`noc_async_read_barrier()` inside `complete_restore`) whose completion time
varies per core (different DRAM addresses for different Q chunks). This provides
natural timing stagger across cores for the deferred half, preventing the
synchronized all-cores-at-once pressure that causes spikes.

### Stats deferral (unchanged)

The stats tiles (running max + running sum) were already deferred before this
change. They stay in their CBs during the save window and get flushed after the
deferred output flush, in a separate time window.

## Writer Flow Per Q Chunk

```
for each Q chunk q:

    1. complete_restore(Q[q])
       └─ noc_async_read_barrier()  ← natural timing stagger
       └─ cb_push_back(prev_out, max, sum)

    2. issue_restore_reads(Q[q+1])
       └─ NOC1 reads in flight (concurrent with steps 3-6)

    3. flush_deferred_output(Q[q-1])      ← second half of prev Q's output
       └─ write deferred rows from cb_out
       └─ cb_pop_front to free cb_out

    4. flush_deferred_stats(Q[q-1])       ← stats from prev Q
       └─ write max/sum tiles from CBs

    ─── K-loop runs (compute processes Q[q] across all K chunks) ───

    5. cb_wait_front(cb_signal)           ← compute signals last K chunk

    6. save_output_with_trid(Q[q])
       └─ write first half of rows (immediate_groups)
       └─ leave remaining rows in cb_out for step 3 of next Q
       └─ stats stay in CBs for step 4 of next Q
```

Steps 1-2 only run for ring_iter > 0 (first ring iteration has no previous
accumulators to restore). Step 6 only runs on non-last ring iterations (last
iteration writes final output with `write_out_row_by_row` + full barrier).

## CB Safety

`cb_out` is allocated at `Sq_chunk_t * vDHt` tiles. After the immediate save
pops the first half, the remaining half stays in `cb_out`. The deferred flush
(step 3) runs before the K-loop, so compute hasn't pushed any new output yet.
The flush pops the remaining rows, freeing `cb_out` completely before the K-loop
produces new output.

## TRID Correctness

Both the immediate and deferred output rows use the same TRID (transaction ID).
The per-TRID barrier scheme fires once per ring iteration to ensure all writes
for a given Q chunk complete before the next ring iteration reads from the same
addresses. Since the deferred flush happens well before the next ring iteration's
TRID barrier, correctness is preserved.

The three TRIDs (FIRST, INNER, LAST) correspond to Q[0], Q[1..N-2], Q[N-1]:
- Q[0]'s save uses TRID_FIRST, barrier fires when Q[N-1] starts next ring iter
- Q[1..N-2]'s saves use TRID_INNER, barrier fires when Q[0] starts next ring iter
- Q[N-1]'s save uses TRID_LAST, barrier fires when Q[N-2] starts next ring iter

This provides maximum flight time for each TRID's writes to complete.

## Single Q Chunk Case

When a core has only one Q chunk (`global_q_end - global_q_start == 1`),
accumulators persist in L1 across ring iterations — no save/restore to DRAM. The
split-save path is not entered. The single-Q path uses `write_out_row_by_row`
only on the last ring iteration (final output).
