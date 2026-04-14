# SDPA DRAM R/W Spike Fix: Deferred Stats Writes

## Problem

In ring joint SDPA with multicast (mcast) store-and-forward, the writer kernel's
accumulator save causes **16K–28K cycle WAIT-K spikes** on compute's
`cb_wait_front(cb_kt_in)`. The spike is absent in the unicast path.

### Root cause

The writer's `save_accumulators_with_trid` writes **54 tiles** to DRAM per Q chunk
per ring iteration on NOC1:

| Component | Tiles | Pattern |
|-----------|-------|---------|
| Output accumulator (cb_out) | 36 | Row-by-row, paced by compute |
| Max stats (cb_max_out) | 9 | Bulk, after all output rows |
| Sum stats (cb_sum_out) | 9 | Bulk, after all output rows |

The 36 paced output tiles alone do **not** spike (proven by Exp 2: 0/10 spikes,
54-cycle peak). The additional 18 bulk stats tiles push DRAM bank pressure past
the cross-NOC contention threshold.

Two contention sources combine:

1. **NOC1 self-contention**: Prefetched restore reads (NOC1 reads) overlap with
   save writes (NOC1 writes) — simultaneous read+write on the same NOC.

2. **Cross-NOC contention**: Writer NOC1 writes overlap with reader NOC0 reads
   at the physical DRAM bank level.

In mcast mode, all receiver cores get K simultaneously, so all cores' pipelines
are synchronized. All writers save accumulators at the same time, creating
**maximum aggregate DRAM bank pressure**. Any one core's writer slowdown
propagates through the mcast semaphore chain to stall all cores:

```
writer DRAM contention on core X
  → core X compute stalls on cb_push_back(cb_out)
  → core X reader can't advance to next K chunk
  → core X increments sender_semaphore late
  → injector's noc_semaphore_wait blocks longer
  → injector's mcast of next K chunk delayed
  → all receiver cores' cb_wait_front(cb_kt_in) spikes
```

In unicast mode, the chain propagation (core 0 → core 1 → core 2) naturally
staggers pipelines, so saves never overlap across all cores simultaneously.

### Why reorganizing reads/writes within the save window doesn't work

Several approaches were tested during investigation:

| Approach | Result | Why it fails |
|----------|--------|-------------|
| Interleave stats with output rows (6 tiles/burst) | 16K–21K spike | Total write volume unchanged (54 tiles); cross-NOC threshold is ~36 tiles regardless of burst size |
| Per-tile write barriers (1 tile at a time) | 37K–41K spike | Barriers add dead time, extending the save window and delaying the pipeline — spike worsens |
| Read barrier before save (serialize NOC1) | Tested in prior investigation: 29K spike | Eliminates NOC1 self-contention but not cross-NOC contention |
| Phased barriers (9 tiles per phase) | Tested in prior investigation: 31K spike | Even 9 tiles of stats writes exceed the cross-NOC threshold |

The cross-NOC contention threshold is very low: **any stats writes added to the
36-tile output writes push the aggregate over the limit**. The threshold is a
per-time-window aggregate across all cores, not per-burst.

---

## Solution: Deferred stats writes

### Core idea

Split the accumulator save into two time-separated operations:

1. **Save window** (after `cb_signal`): Write **output only** (36 tiles, paced).
   Stays below the cross-NOC contention threshold. Max/sum tiles remain in their
   circular buffers (`cb_max_out`, `cb_sum_out`), unpoppped.

2. **Next Q's processing window** (before the K-loop): Flush the held stats to
   DRAM, then pop the CBs. This happens during a different time window where the
   aggregate DRAM pressure is lower.

### Why it works

The flush is placed **after `issue_restore_reads` + `noc_async_read_barrier()`**.
This creates two effects:

1. **No NOC1 self-contention**: The read barrier ensures all restore reads
   complete before the flush writes begin. During the flush, only NOC1 writes
   are active — no concurrent NOC1 reads.

2. **Timing desynchronization**: The read barrier's latency depends on the
   specific DRAM bank addresses of the restore reads (which differ per core,
   per Q chunk). Each core's barrier completes at a slightly different time.
   This naturally staggers the flush writes across cores, preventing the
   synchronized all-cores-at-once pressure that causes the spike.

Combined, the flush's 18 stats tiles from each core arrive at DRAM at staggered
times, keeping the instantaneous aggregate write rate below the contention
threshold.

### CB safety

The stats CBs (`cb_max_out` = c_17, `cb_sum_out` = c_10) hold Q[q]'s stats
between save and flush without overflow:

- Compute only pushes to these CBs on the **last K-chunk** of each Q
  (`save_to_staging = is_last_k && !is_last_ring_iter && q_per_core > 1`).
- The flush completes during the **first few K-chunks** of Q[q+1] (~3K cycles
  vs ~90K+ cycles before the last K-chunk). Massive timing margin.
- On the **last ring iteration**, `save_to_staging = false`, so compute never
  pushes to these CBs. The stats from the previous ring iteration's last Q can
  sit safely until flushed at Q[0] of the last ring iter.

### TRID correctness

The deferred stats writes use the **same TRID** as the output writes for the
same Q chunk (TRID_FIRST / TRID_INNER / TRID_LAST). The existing per-TRID
barrier scheme at each ring iteration ensures all writes (output + deferred stats)
complete before the next ring iteration reads from the same addresses.

Since deferred stats are flushed during Q[q+1]'s processing — well before the
next ring iteration's TRID barrier fires — the TRID outstanding count is always
zero when the barrier checks it.

---

## Code changes

All changes are in `ring_joint_writer.cpp`.

### New: `DeferredStats` struct (line 182)

Stores the DRAM addressing metadata (batch, head, stats tile range, TRID) for
one deferred flush. Declared outside the ring iteration loop so deferred stats
from Q[N-1] of ring_iter R carry over to Q[0] of ring_iter R+1.

### New: `flush_deferred_stats()` (line 196)

Writes held max/sum tiles from CBs to DRAM and pops the CBs. Sets the deferred
TRID, issues all max+sum write tiles, flushes with TRID, resets TRID to 0, and
pops both CBs.

### Renamed: `save_accumulators_with_trid()` → `save_output_with_trid()`

Simplified to write **output only**. Unused parameters removed (stats_writer,
stats_tile_logical, nb, nq, Sq_chunk_t, stats ranges, sum_offset, cb_max_out,
cb_sum_out, stats_tile_bytes — 11 of 18 were dead). Second template parameter
(`TensorAccessorType`) also removed. Stats tiles stay in the CBs for the next
Q's `flush_deferred_stats` to drain.

### Modified: Q-loop in deferred norm path

Two flush points added:

1. **After restore reads**: For both ring_iter > 0 and ring_iter == 0.
   On ring_iter > 0, placed after `issue_restore_reads` with a preceding
   `noc_async_read_barrier()` that provides timing stagger. On ring_iter == 0
   (no restore reads in flight), the read barrier is a no-op.

2. **After save**: Sets `has_deferred_stats = true` and stores metadata in
   `deferred`.

### State variable lifetime

`has_deferred_stats` and `deferred` are declared before the ring iteration loop
(line 446). They persist across ring iterations:

```
Ring iter R, Q[2] (last Q): save → has_deferred_stats = true
Ring iter R+1, Q[0]:        flush Q[2]'s stats → has_deferred_stats = false
```

---

## Verification

### Spike test (AllGather disabled, check_kt_spike.py)

| Metric | Before | After |
|--------|--------|-------|
| Peak WAIT-K cycles | 16K–28K (10/10 spikes) | 54–56 (0/15 spikes) |

### Accuracy test (AllGather enabled)

```
test_ring_joint_attention_sdpa_accuracy[wan2_2_1xGLX-q288-k512]: PASSED
```

### Approaches tested during development

| Approach | Peak cycles | Spike rate | Notes |
|----------|-------------|------------|-------|
| Baseline (bulk stats in save) | 16K–28K | 10/10 | Original code |
| Output-only save (diagnostic) | 54 | 0/10 | Confirms threshold is 36 tiles |
| Interleaved stats with output | 16K–21K | 3/3 | 6 tiles/burst, total unchanged |
| Per-tile write barriers in save | 37K–41K | 3/3 | Dead time worsens pipeline |
| Deferred flush before restore reads | 10K–15K | 5/5 | Correct but still synchronized |
| Deferred flush after restore reads + read barrier | **54–56** | **0/15** | **Final fix** |

---

## Performance considerations

The `noc_async_read_barrier()` before the flush forces the prefetched restore
reads to complete early (instead of flying during the K-loop). This trades some
prefetch overlap for spike elimination. The restore read latency (~3–5K cycles)
becomes visible but is absorbed within the K-loop idle window (~100K cycles).
Net impact on total SDPA latency is negligible.

No L1 allocation changes are needed. No CB size changes. No program factory
changes. The fix is entirely within the writer kernel's dataflow scheduling.
