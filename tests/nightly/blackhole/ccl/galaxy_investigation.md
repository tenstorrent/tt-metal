# SDPA DRAM R/W Spike Investigation — Galaxy (32 devices)

## Overview

The deferred stats fix (branch `skrstic/sdpa-dram-r-w`) eliminates WAIT-K spikes
on a quiet box (4 devices, ring_size=4) but does NOT eliminate them on Galaxy
(32 devices, ring_size=8). Math utilization improves slightly (68.8% -> 69.7%)
but spikes persist.

This document describes experiments run on Galaxy to identify what is different
and find the minimal setup that eliminates spikes.

---

## Galaxy vs Quiet Box Configuration

| Parameter | Galaxy | Quiet Box |
|-----------|--------|-----------|
| Devices | 32 | 4 |
| Ring size | 8 | 4 |
| TP size | 4 | 1 |
| Parallel rings | 4 | 1 |
| Per-device seq | 9,472 | 9,472 |
| Total seq (per ring) | 75,776 | 37,888 |
| Gathered K/V tiles | 2,368 | 1,184 |
| K chunks per ring iter | 19 | 19 |
| Q chunks per head | 33 | 33 |
| Cores per device | 110 (11 cols × 10 rows) | 110 |
| Q per core | 3 | 3 |
| AllGather remote shards | 7 | 3 |

Model config: `wan2_2_1xGLX` (q=288, k=512, nhq=10, d=128, non-causal).

Per-device compute work is identical. The key hardware difference is ring_size
(8 vs 4) and AllGather volume (7 vs 3 remote shards written to gathered K/V DRAM).

---

## Part 1: Is the Root Cause the Same as Quiet Box?

On quiet box, the spike was caused by the writer's 18 stats tiles pushing total
DRAM writes above a contention threshold. The deferred fix moved those 18 tiles
to a different time window. On galaxy, we check if the same mechanism applies.

### Experiment 1: Deferred Fix Baseline per Ring Iteration

WAIT-K zone placed at specific ring iterations, one at a time. Full deferred-fix
code active (output-only save + deferred stats flush).

| # | Ring Iter | Peak (cycles) | Spikes | Rate | Writer Traffic |
|---|-----------|---------------|--------|------|----------------|
| 1A | 0 (first) | 8,816 | 3,520 | 0.6% | Save output only (no restore, no stats flush for Q[0..1]) |
| 1B | 1 | 54,847 | 2,755 | 0.5% | Full: save + restore reads + deferred flush |
| 1C | 3 (middle) | 22,479 | 3,647 | 0.6% | Full: save + restore reads + deferred flush |
| 1D | 7 (last) | 9,468 | 106 | 0.0% | Final output write + restore reads (no save) |

**Total zones per measurement**: 601,920 (32 devices × 110 cores × 3 Q × 57 op invocations).

Important: on quiet box, ring_iter 3 was the **last** iteration (ring_size=4). On
galaxy, ring_iter 3 is a **middle** iteration. The last is ring_iter 7.

Per-chip distribution at ring_iter 3: all 32 chips affected except 8, 9, 24.
Peaks range 13K–22K. Rates 0.1%–2.2%.

### Experiment 2: Remove Stats Writes

`flush_deferred_stats` stubbed to pop CBs without DRAM writes.
Output save (36 tiles) remains active. Ring_iter 3.

| Peak | Spikes | Rate |
|------|--------|------|
| 27,513 | 4,256 | 0.7% |

**Removing stats writes does NOT help.** All 32 chips affected.

On quiet box, removing stats writes eliminated spikes completely (0/10, 54 cycles).
This is the fundamental difference: on quiet box the threshold was between 36 and
54 tiles; on galaxy the threshold is **below 36 tiles**.

### Experiment 3: Remove ALL Save Writes

Both `save_output_with_trid` and `flush_deferred_stats` stubbed — CBs drained
without any DRAM writes. Restore reads remain active. Ring_iter 3.

| Peak | Spikes | Rate |
|------|--------|------|
| 54 | 0 | 0.0% |

**0 spikes.** Confirms that save writes are necessary for the spike. Restore
reads alone don't cause contention.

### Experiment 4: Output Save Only (No Stats, No Restore Reads)

`save_output_with_trid` active (36 tiles on NOC1 writes), `flush_deferred_stats`
stubbed, `issue_restore_reads` stubbed (reserve-only, no DRAM reads). Ring_iter 3.

| Peak | Spikes | Rate |
|------|--------|------|
| 5,709 | 42 | 0.0% |

Small spikes on 15/32 chips. **Output writes alone (no NOC1 reads in flight)
create cross-NOC contention with reader K/V reads on galaxy.**

On quiet box, the equivalent (Exp 6D: save writes, no restore reads) was 0/10.

### Experiment 5: AllGather Disabled

AllGather reader/writer kernels return immediately. `fused_op_receiver` sync
disabled. Full deferred-fix code otherwise. Ring_iter 3.

| Peak | Spikes | Rate |
|------|--------|------|
| 50,003 | 1,372 | 0.2% |

**Disabling AllGather makes spikes WORSE**, same pattern as quiet box (Exp 4).
AllGather sync provides pacing that reduces contention.

### Part 1 Conclusion

The root cause mechanism is the same (DRAM bank contention propagated via mcast
semaphore chain), but the **contention threshold is lower on galaxy**:

| | Quiet Box | Galaxy |
|--|-----------|--------|
| Output writes only (36 tiles) | 0/10 spikes | 42 spikes (5.7K peak) |
| Output + stats (54 tiles) | 10/10 spikes | spikes |
| Output + stats deferred | **0/15 spikes** | 3,647 spikes (22K peak) |

On quiet box, the contention source was **NOC1 self-contention** (save writes +
restore reads on the same NOC). The deferred fix eliminated this by separating
the read and write windows.

On galaxy, the contention source is **cross-NOC DRAM bank contention** (writer
NOC1 output writes + reader NOC0 K/V reads competing at the physical DRAM bank
level). Even without any NOC1 reads in flight, 36 tiles of output writes cause
spikes. The deferred fix doesn't address this because it only moved stats writes;
the 36-tile output writes remain in the save window.

**Why is the threshold lower?** AllGather fills a 2× larger gathered K/V buffer
(8 vs 4 ring devices), writing 7 remote shards to DRAM per device. This
background DRAM traffic raises the baseline bank pressure, lowering the headroom
for any additional writes from the SDPA writer.

---

## Part 2: Finding the Galaxy Threshold

### Output Row Sweep (No Restore Reads, No Stats)

`issue_restore_reads` and `flush_deferred_stats` stubbed. `save_output_with_trid`
modified to write only the first N rows (each row = 1 × 4 = 4 tiles, via
`out_subblock_h=1`). Remaining rows popped without DRAM writes. Ring_iter 3.

| Rows Written | Tiles | Peak | Spikes | Spike? |
|-------------|-------|------|--------|--------|
| 0 | 0 | 55 | 0 | NO |
| 1 | 4 | 54 | 0 | NO |
| 3 | 12 | 55 | 0 | NO |
| 5 | 20 | 55 | 0 | NO |
| 7 | 28 | 55 | 0 | NO |
| **8** | **32** | **2,283** | **3** | **YES** |
| 9 | 36 | 6,053 | 49 | YES |

**Cross-NOC threshold (output writes only, no concurrent NOC1 reads):** between
28 and 32 tiles (7–8 rows).

### Output Row Sweep WITH Restore Reads (No Stats)

`issue_restore_reads` restored to original (54 tiles of NOC1 reads in flight).
`flush_deferred_stats` still stubbed. Ring_iter 3.

| Rows Written | Tiles | Peak | Spikes | Spike? |
|-------------|-------|------|--------|--------|
| 5 | 20 | 55 | 0 | NO |
| **6** | **24** | **25,279** | **290** | **YES** |
| 7 | 28 | 16,131 | 1,048 | YES |
| 9 | 36 | 27,513 | 4,256 | YES |

**Threshold with concurrent restore reads:** between 20 and 24 tiles (5–6 rows).
Restore reads lower the threshold by ~8 tiles (one row group of 4 cols).

### Output Row Sweep WITH Restore Reads AND Deferred Stats (Full Pipeline)

`issue_restore_reads` and `flush_deferred_stats` both fully restored. Ring_iter 3.

| Rows Written | Tiles | Peak | Spikes | Spike? |
|-------------|-------|------|--------|--------|
| 5 | 20 | 56 | 0 | NO |
| **6** | **24** | **14,457** | **247** | **YES** |
| 9 | 36 | 22,479 | 3,647 | YES |

**Threshold with full pipeline:** same as restore-reads-only, between 20 and 24
tiles. The deferred stats flush does not change the threshold — the stats are
flushed during a different time window (before the K-loop), so they don't overlap
with the save.

### Can Barriers Help?

| Approach | Peak | Spikes | Rate |
|----------|------|--------|------|
| Barrier after row 5 (`write_barrier_with_trid`) | 22,948 | 5,425 | 0.90% |
| Barrier every 3 rows | 36,272 | 9,798 | 1.63% |

**Barriers make things worse.** The write barrier forces all writes to complete
before the next batch, adding dead time that extends the save window. Worse,
barriers create a **synchronization point**: all cores hit the barrier and resume
at the same time, creating a coordinated burst of writes in the second batch.
This is the opposite of what we want.

### Threshold Summary

```
                  Output Only         + Restore Reads       + Restore + Stats
                  (NOC1 writes only)  (+ NOC1 reads)        (full pipeline)
Threshold:        28–32 tiles         20–24 tiles            20–24 tiles
                  (7–8 rows)          (5–6 rows)             (5–6 rows)

Quiet box:        >36 tiles           >36 tiles              36–54 tiles
                  (no spikes)         (no spikes)            (deferred fix works)
```

Galaxy's threshold is roughly half of quiet box's. The 36-tile output save that
was safely below threshold on quiet box is well above threshold on galaxy.

---

## Part 3: Split-Save Fix

### Core Idea

Instead of writing all 9 output rows (36 tiles) during the save window, split
the save into two temporally separated phases:

1. **Immediate save** (after `cb_signal`): Write only the first 5 rows (20 tiles)
   to DRAM. This is below the full-pipeline threshold (20–24 tiles). Pop these
   rows from `cb_out`. Leave the remaining 4 rows (16 tiles) in `cb_out`.

2. **Deferred flush** (next Q's processing window): After the read barrier for
   the next Q's restore reads, write the remaining 4 rows (16 tiles) from `cb_out`
   to DRAM and pop them. This happens in a different time window where:
   - The read barrier completion time varies per core (different DRAM addresses
     for different Q chunks), providing **natural timing stagger** across cores.
   - The 16 deferred tiles are below the output-only threshold (28–32 tiles).

This is the same principle as the stats deferral, extended to cover part of the
output.

### CB Safety

`cb_out` is allocated at `Sq_chunk_t × vDHt = 36 tiles`. After the immediate
save pops 5 rows (20 tiles), 16 tiles remain occupied. The CB has 20 tiles free.

The deferred flush runs at the start of the next Q's processing — **before** the
K-loop. Compute only pushes new output to `cb_out` during the **last K chunk**
(after `cb_signal`). There are 18 K chunks before the last one, providing massive
timing margin for the flush to complete and free the CB.

### TRID Correctness

The deferred output rows use the **same TRID** as the immediate rows. The
existing per-TRID barrier scheme ensures all writes complete before the next ring
iteration reads from the same addresses. Since the deferred flush happens well
before the next ring iteration's TRID barrier, correctness is preserved.

### Implementation (all in `ring_joint_writer.cpp`)

#### 1. New constant: `SAVE_IMMEDIATE_ROWS` (line ~193)

```cpp
constexpr uint32_t SAVE_IMMEDIATE_ROWS = 5;
```

Max rows written during the save window. 5 rows × 4 cols = 20 tiles, which is
below the full-pipeline threshold of 20–24 tiles. This value was determined
empirically by the Part 2 sweep experiments.

#### 2. New struct: `DeferredOutput` (line ~189)

```cpp
struct DeferredOutput {
    Slice out_slice;        // DRAM addressing for this Q chunk's output
    uint32_t start_row;     // first deferred row index (e.g., 5)
    uint32_t end_seq_tile;  // padding boundary (0xFFFFFFFF for accumulators)
    uint32_t save_trid;     // same TRID as the immediate rows
    bool is_joint_q;        // selects out_generator vs joint_out_generator
};
```

Stores the metadata needed to write the deferred tail rows later. Declared as
file-scope globals `has_deferred_output` and `deferred_out` (alongside the
existing `has_deferred_stats` / `deferred` for stats).

#### 3. New function: `flush_deferred_output()` (line ~198)

```cpp
template <typename ReaderType>
void flush_deferred_output(
    const DeferredOutput& dout,
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const uint32_t cb_out,
    const uint32_t tile_bytes,
    const uint32_t sbh) {
    // ...
    noc_async_write_set_trid(dout.save_trid);
    for (uint32_t row = dout.start_row; row < out_rows; row += sbh) {
        cb_wait_front(cb_out, row_tiles);
        // ... write tiles via maybe_write_tile (same as write_out_row_by_row) ...
        cb_pop_front(cb_out, row_tiles);
    }
    noc_async_write_flushed_with_trid(dout.save_trid);
    noc_async_write_set_trid(0);
}
```

Same write logic as `write_out_row_by_row`, but starts from `dout.start_row`
instead of row 0. The rows are still sitting at the front of `cb_out` (never
popped during the save), so `cb_wait_front` returns immediately.

Uses the same TRID as the immediate rows — this is safe because the TRID barrier
for this Q chunk won't fire until the next ring iteration (by which time both
the immediate and deferred writes have long completed).

#### 4. Modified: `save_output_with_trid()` (line ~280)

Before (wrote all 9 rows):
```cpp
noc_async_write_set_trid(save_trid);
write_out_row_by_row(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes, sbh);
noc_async_write_flushed_with_trid(save_trid);
noc_async_write_set_trid(0);
```

After (writes first 5 rows, leaves 4 in cb_out):
```cpp
noc_async_write_set_trid(save_trid);
uint32_t rows_done = 0;
for (uint32_t rg = 0; rg < num_row_groups; ++rg) {
    if (rows_done >= SAVE_IMMEDIATE_ROWS) break;  // ← stop after 5 rows
    cb_wait_front(cb_out, row_tiles);
    // ... write tiles ...
    cb_pop_front(cb_out, row_tiles);
    rows_done += sbh;
}
noc_async_write_flushed_with_trid(save_trid);
noc_async_write_set_trid(0);

if (rows_done < out_rows) {
    has_deferred_output = true;
    deferred_out = {out_slice, rows_done, end_seq_tile, save_trid, false};
}
```

The loop breaks after writing `SAVE_IMMEDIATE_ROWS` rows. Rows 5–8 remain in
`cb_out` (pushed by compute but not yet popped by the writer). The `deferred_out`
struct captures everything needed to write them later.

#### 5. Modified: Q-loop flush point (line ~683)

Before:
```cpp
// Flush deferred stats from previous Q during the K-loop window.
if (has_deferred_stats) {
    flush_deferred_stats(...);
    has_deferred_stats = false;
}
```

After:
```cpp
// Flush deferred output tail rows from previous Q.
if (has_deferred_output) {
    flush_deferred_output(
        deferred_out,
        deferred_out.is_joint_q ? joint_out_generator : out_generator,
        cb_out, tile_bytes, out_subblock_h);
    has_deferred_output = false;
}

// Flush deferred stats from previous Q during the K-loop window.
if (has_deferred_stats) {
    flush_deferred_stats(...);
    has_deferred_stats = false;
}
```

The deferred output flush runs **before** the deferred stats flush, both placed
after `complete_restore` + `issue_restore_reads` (which includes a
`noc_async_read_barrier()` that provides the natural timing stagger).

#### 6. Patching `is_joint_q` at the call site (line ~733)

The `save_output_with_trid` function doesn't know whether the Q chunk uses the
joint output generator. The `is_joint_q` flag is patched right after the save:

```cpp
save_output_with_trid(...);
if (has_deferred_output) {
    deferred_out.is_joint_q = qi.is_joint_q;
}
```

For the WAN 2.2 config (L=0, no joint sequence), `is_joint_q` is always false.

### Timeline comparison

**Before (deferred stats only):**
```
Q[q] save window (after cb_signal):
  ├─ write 9 output rows (36 tiles, NOC1 writes)  ← EXCEEDS THRESHOLD
  └─ [stats stay in CB for later flush]

Q[q+1] flush window (after read_barrier):
  └─ flush stats (18 tiles, NOC1 writes)
```

**After (split save):**
```
Q[q] save window (after cb_signal):
  ├─ write 5 output rows (20 tiles, NOC1 writes)  ← BELOW THRESHOLD ✓
  └─ [rows 5-8 stay in cb_out, stats stay in CB]

Q[q+1] flush window (after read_barrier):
  ├─ flush 4 deferred output rows (16 tiles)       ← BELOW THRESHOLD ✓
  └─ flush stats (18 tiles)                         ← separate time window
```

The read barrier between the save window and flush window provides natural
timing desynchronization: each core's barrier completes at a different time
(depending on its DRAM addresses), so the deferred flushes are staggered across
cores. This prevents the synchronized all-cores-at-once pressure that causes
the spike.

### Results

| Run | Peak (cycles) | Spikes | Spike? |
|-----|---------------|--------|--------|
| 1 | 55 | 0 | NO |
| 2 | 54 | 0 | NO |
| 3 | 55 | 0 | NO |
| 4 | 56 | 0 | NO |

**0 spikes across 4 consecutive runs (2.4M total WAIT-K zones).** Peak is
54–56 cycles (same as the hardware baseline for `cb_wait_front` overhead).

Accuracy test: not yet validated (device reset issue during test run).

---

## Summary

### Why the deferred-stats-only fix fails on galaxy

```
Quiet Box (ring_size=4)              Galaxy (ring_size=8)
─────────────────────────            ──────────────────────
Contention type:                     Contention type:
  NOC1 self-contention                 Cross-NOC bank contention
  (save writes + restore reads)        (NOC1 writes + NOC0 reads at DRAM)

Threshold: 36–54 tiles               Threshold: 20–24 tiles (full pipeline)

Stats deferral (−18 tiles):          Stats deferral (−18 tiles):
  36 tiles → below threshold           36 tiles → STILL above threshold
  ✓ Fix works                          ✗ Fix insufficient
```

The deferred stats fix removes 18 tiles from the save window, bringing the total
from 54 to 36. On quiet box, 36 tiles is safely below the 36–54 tile threshold.
On galaxy, 36 tiles is still above the 20–24 tile threshold.

### Why the threshold is lower on galaxy

AllGather fills a 2× larger gathered buffer (8 vs 4 ring devices). Each device
receives 7 remote K/V shards written to DRAM. This background traffic raises the
baseline DRAM bank pressure, lowering the headroom for SDPA writer saves.

The AllGather sync also provides **pacing** — it gates SDPA ring iterations,
preventing all cores from hitting DRAM simultaneously. Disabling AllGather
makes spikes worse (50K peak), not better.

### The split-save fix

Split the 9-row output save into:
- **5 rows immediate** (20 tiles, below threshold) in the save window
- **4 rows deferred** (16 tiles, below threshold) in the next Q's flush window

Both halves are individually below the contention threshold. The read barrier
between provides natural timing stagger across cores (preventing synchronized
DRAM bursts). Result: **0 spikes in 4 consecutive runs.**

### Spike propagation path (unchanged from quiet box)

```
writer output save contention on core X
  → core X compute stalls on cb_push_back(cb_out)
  → core X reader can't advance to next K chunk
  → core X increments sender_semaphore late
  → injector core's noc_semaphore_wait blocks longer
  → injector's mcast of next K chunk delayed
  → all receiver cores' cb_wait_front(cb_kt_in) spikes
```

### Full Experiment Table

| # | Configuration | Ring Iter | Peak | Spikes | Rate |
|---|---------------|-----------|------|--------|------|
| 1A | Deferred fix baseline | 0 | 8,816 | 3,520 | 0.6% |
| 1B | Deferred fix baseline | 1 | 54,847 | 2,755 | 0.5% |
| 1C | Deferred fix baseline | 3 | 22,479 | 3,647 | 0.6% |
| 1D | Deferred fix baseline | 7 | 9,468 | 106 | 0.0% |
| 2 | Stats writes removed | 3 | 27,513 | 4,256 | 0.7% |
| 3 | ALL save writes removed | 3 | **54** | **0** | **0.0%** |
| 4 | Output save only (no stats/restore) | 3 | 5,709 | 42 | 0.0% |
| 5 | AllGather disabled | 3 | 50,003 | 1,372 | 0.2% |
| 6a | 1 row save only (4 tiles) | 3 | 54 | 0 | 0.0% |
| 6b | 3 rows save only (12 tiles) | 3 | 55 | 0 | 0.0% |
| 6c | 5 rows save only (20 tiles) | 3 | 55 | 0 | 0.0% |
| 6d | 7 rows save only (28 tiles) | 3 | 55 | 0 | 0.0% |
| 6e | 8 rows save only (32 tiles) | 3 | 2,283 | 3 | 0.0% |
| 6f | 9 rows save only (36 tiles) | 3 | 6,053 | 49 | 0.0% |
| 7a | 5 rows + restore (no stats) | 3 | 55 | 0 | 0.0% |
| 7b | 6 rows + restore (no stats) | 3 | 25,279 | 290 | 0.05% |
| 7c | 7 rows + restore (no stats) | 3 | 16,131 | 1,048 | 0.17% |
| 8a | 5 rows + restore + stats (full) | 3 | 56 | 0 | 0.0% |
| 8b | 6 rows + restore + stats (full) | 3 | 14,457 | 247 | 0.04% |
| 9a | 9 rows, barrier after row 5 | 3 | 22,948 | 5,425 | 0.90% |
| 9b | 9 rows, barrier every 3 rows | 3 | 36,272 | 9,798 | 1.63% |
| **10** | **Split save (5+4 deferred)** | **3** | **55** | **0** | **0.0%** |
