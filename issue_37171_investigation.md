# Issue #37171: `test_layer_norm_sharded_two_stage` Watcher Failure Investigation

## Summary

When `TT_METAL_WATCHER=1` is enabled, `test_layer_norm_sharded_two_stage` fails with a
`DebugAssertNCriscNOCReadsFlushedTripped` assert on core (0,0) BRISC, indicating the kernel
`reader_mcast_sender_unary_sharded_ln.cpp` completed with pending NOC read transactions.

**Without watcher:** test passes in ~1.08s.
**With watcher:** test aborts due to tripped assert.

## CONFIRMED Root Cause (2026-03-05)

**Non-posted multicast writes on Wormhole spuriously increment `NIU_MST_RD_RESP_RECEIVED`.**

DPRINT diagnostics confirmed:
```
post-read-barrier: HW_RD_RESP=40 SW_ISSUED=40    ← in sync after last read barrier
post-mcast-writes: HW_RD_RESP=42 SW_ISSUED=40    ← HW jumped +2, SW unchanged
MISMATCH! delta=2
```

Between these two measurements, the kernel issued **zero NOC reads** — only non-posted
multicast writes (`noc_async_write_multicast` with `linked=true`) and `noc_semaphore_set_multicast`.
Yet the hardware read-response counter incremented by 2, causing the firmware assert to trip.

The delta of 2 matches the number of multicast write loop iterations
(`num_all_to_all_workers_first_stage`), suggesting each iteration's write+semaphore pair
causes one spurious increment of the read response counter.

---

## Test Command

```bash
export TT_METAL_WATCHER=1  # omit for non-watcher run
pytest tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_two_stage \
  -k "dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=True-use_welford=True-h=128-w=256-num_cores_h=4-num_cores_w=2-block_ht=4-block_wt=1-subblock_wt=1"
```

---

## Hardware/Software Context

- **Device:** N150 (single-chip Wormhole)
- **NOC Mode:** `DM_DEDICATED_NOC` (BRISC uses NOC0, NCRISC uses NOC1)
- **Kernel mapping on each Tensix core:**
  - **BRISC (RISCV_0):** `reader_mcast_sender_unary_sharded_ln.cpp` — uses NOC0
  - **NCRISC (RISCV_1):** `writer_unary_sharded_ln.cpp` — uses NOC1
  - **TRISC0/1/2:** `layernorm_sharded_welford.cpp` (compute)
- **NOC assignment confirmed in:**
  - `layernorm_op_multi_core_sharded.cpp:402-406` — `reader_noc = NOC_0`, `writer_noc = NOC_1`
  - `sharded_layernorm_factory_helpers.cpp:787-790` — `noc_mode = DM_DEDICATED_NOC`

---

## The Watcher Assert

### Where it triggers

In `brisck.cc:82-91` (the BRISC firmware entry point for Wormhole), after `kernel_main()` returns:

```cpp
if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
    WAYPOINT("NKFW");
    ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
    ASSERT(ncrisc_noc_nonposted_writes_sent(NOC_INDEX), ...);
    ASSERT(ncrisc_noc_nonposted_atomics_flushed(NOC_INDEX), ...);
    ASSERT(ncrisc_noc_posted_writes_sent(NOC_INDEX), ...);
    WAYPOINT("NKFD");
}
```

### What `ncrisc_noc_reads_flushed` checks

In `noc_nonblocking_api.h:427-429`:

```cpp
inline bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}
```

This compares:
- `NIU_MST_RD_RESP_RECEIVED` — **hardware register** counting NOC read responses received
- `noc_reads_num_issued[noc]` — **software counter** incremented each time a NOC read is issued

The assert fires when these two values don't match, meaning
`NIU_MST_RD_RESP_RECEIVED != noc_reads_num_issued[NOC_0]` at kernel completion.

### Counter initialization

In `brisck.cc:66-68`, *before* `kernel_main()` is called:

```cpp
if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
    noc_local_state_init(NOC_INDEX);
}
```

`noc_local_state_init` (in `noc_nonblocking_api.h:661-674`) snapshots the current hardware
register value into the software counter:

```cpp
void noc_local_state_init(int noc) {
    uint32_t reads_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    // ... (also snapshots write counters)
    noc_reads_num_issued[noc] = reads_num_issued;
    // ...
}
```

This assumes the hardware and software counters start in sync. If they later diverge,
the assert will trip.

---

## Watcher Log Analysis

From `generated/watcher/watcher.log`, the assert log shows:

```
Device 0 worker core(x= 0,y= 0) virtual(x=18,y=18): NKFW, W, W, W, W
  rmsg:H0G|BNT h_id:1024 smsg:DDDD k_ids: 1| 3| 4| 4| 4
```

- `NKFW` = waypoint right before the reads-flushed assert (BRISC)
- `W` for all subordinates = they finished and are waiting
- `smsg:DDDD` = all subordinates done
- BRISC is at `NKFW`, meaning `kernel_main()` returned successfully and it's the
  firmware assert that tripped

---

## Kernel Code Analysis

### `reader_mcast_sender_unary_sharded_ln.cpp` (BRISC, NOC0)

The kernel's NOC operations follow this pattern:

1. **Lines 161-173:** Read partial results from remote cores via `noc_async_read_one_packet` → `noc_async_read_barrier()` ✓
2. **Lines 189-199:** (two-stage) Read combined results from remote cores → `noc_async_read_barrier()` ✓
3. **Lines 225-236:** Gather final results via `noc_async_read_one_packet`/`noc_async_read` → `noc_async_read_barrier()` ✓ (line 236, **last read barrier**)
4. **Lines 249-259:** Multicast writes via `noc_async_write_multicast` (with `linked=true`) and `noc_semaphore_set_multicast` → `noc_async_write_barrier()`

**After line 236, the kernel issues zero NOC reads.** Only non-posted multicast writes
and semaphore multicasts occur (lines 249-259), all on NOC0.

### `writer_unary_sharded_ln.cpp` (NCRISC, NOC1)

This kernel exclusively uses NOC1 for all reads/writes. It should not affect NOC0 counters.

---

## Why Naive `noc_async_read_barrier()` at End Caused a Hang

### The attempted fix

Adding `noc_async_read_barrier()` at the very end of `kernel_main()` in
`reader_mcast_sender_unary_sharded_ln.cpp`.

### What happened

The test **hung indefinitely** with watcher showing BRISC stuck at waypoint `NRBW`
(NOC read barrier wait). The barrier spins on:

```cpp
while (!ncrisc_noc_reads_flushed(noc));
// i.e., while (NIU_MST_RD_RESP_RECEIVED != noc_reads_num_issued[noc])
```

### Why this proves the counters are mismatched *in a specific direction*

- The existing `noc_async_read_barrier()` at line 236 **passed** — at that point,
  `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[NOC_0]`.
- Between line 236 and kernel exit, the kernel issues **zero** NOC reads, so
  `noc_reads_num_issued[NOC_0]` does not change.
- Yet at kernel exit, the firmware assert says `NIU_MST_RD_RESP_RECEIVED != noc_reads_num_issued[NOC_0]`.
- The added barrier at the end **hangs**, meaning `NIU_MST_RD_RESP_RECEIVED` never
  catches up to `noc_reads_num_issued[NOC_0]`.

This means `noc_reads_num_issued[NOC_0] > NIU_MST_RD_RESP_RECEIVED` — the software counter
is **ahead** of the hardware counter. But wait, the original assert fires because they're
not equal. Let me reconcile:

**Actually**, the barrier and the assert both check the same condition:
`NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]`. If the barrier hangs, the
hardware counter never equals the software counter. If the assert trips without hanging
the barrier... the assert fires once without spinning.

This means:
- **Without the added barrier:** The kernel finishes, firmware checks once, finds them
  not equal → assert trips.
- **With the added barrier:** The kernel reaches the barrier and spins forever because
  the condition is never satisfied → hang.

The direction of mismatch must be: **`noc_reads_num_issued[NOC_0] > NIU_MST_RD_RESP_RECEIVED`**
(software ahead of hardware), because if hardware were ahead, the barrier would immediately
pass (the hardware counter wraps and the equality check uses unsigned arithmetic that would
eventually match, or the hardware counter is already >= software counter).

Wait — actually both the barrier and the assert check `==` (strict equality). If `HW > SW`,
the barrier would also hang forever (waiting for `HW == SW` but HW is already past SW).

The key insight is: **something is causing the software counter (`noc_reads_num_issued[NOC_0]`)
to diverge from the hardware counter (`NIU_MST_RD_RESP_RECEIVED`) between the last
successful barrier (line 236) and kernel exit.**

---

## Root Cause: CONFIRMED

### Non-posted multicast write acks increment `NIU_MST_RD_RESP_RECEIVED` on Wormhole

DPRINT instrumentation was added to `reader_mcast_sender_unary_sharded_ln.cpp` to read
the hardware register `NIU_MST_RD_RESP_RECEIVED` and software counter `noc_reads_num_issued`
at two points:
1. Immediately after the last `noc_async_read_barrier()` (line 237)
2. Immediately after the multicast write loop completes (after all `noc_async_write_barrier()` calls)

**Results:**
```
post-read-barrier: HW_RD_RESP=40 SW_ISSUED=40    ← perfectly in sync
post-mcast-writes: HW_RD_RESP=42 SW_ISSUED=40    ← HW +2, SW unchanged
MISMATCH! delta=2
```

Between these two points, the kernel issues **zero NOC reads**. The only operations are:
- `noc_async_write_multicast(..., linked=true)` — non-posted multicast write
- `noc_semaphore_set_multicast(...)` — non-posted multicast semaphore set
- `noc_async_write_barrier()` — wait for write acks

Each loop iteration (there are 2, matching `num_all_to_all_workers_first_stage`) causes
`NIU_MST_RD_RESP_RECEIVED` to increment by 1, for a total delta of 2.

**This is a Wormhole NOC hardware behavior** where non-posted multicast write acknowledgements
also increment the read response counter. The software counter `noc_reads_num_issued` is
never incremented for writes, so they diverge.

### Why it only fails with watcher

Without watcher, the kernel completes and the firmware never checks the counters —
`noc_local_state_init` at the start of the next kernel re-snapshots both to their
current (matched) values. With watcher, the firmware checks `==` at `NKFW` and catches
the mismatch.

### Why adding `noc_async_read_barrier()` at the end caused a hang

The barrier spins on `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]`.
Since `HW (42) > SW (40)`, and nothing will ever increment `SW`, the barrier hangs forever.

---

## Key Files

| File | Role |
|------|------|
| `ttnn/.../reader_mcast_sender_unary_sharded_ln.cpp` | BRISC kernel (NOC0) — the failing kernel |
| `ttnn/.../writer_unary_sharded_ln.cpp` | NCRISC kernel (NOC1) — runs on same core |
| `ttnn/.../layernorm_sharded_welford.cpp` | TRISC compute kernel |
| `tt_metal/hw/firmware/src/tt-1xx/brisck.cc` | BRISC firmware — contains assert |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h` | NOC API — counter logic |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | High-level dataflow API — barriers |
| `tt_metal/hw/inc/internal/debug/sanitize.h` | Watcher sanitization macros |
| `ttnn/.../layernorm_op_multi_core_sharded.cpp` | Host-side kernel config (NOC assignment) |
| `ttnn/.../sharded_layernorm_factory_helpers.cpp` | Kernel descriptor setup (NOC mode) |

---

## Recommended Next Steps (updated after confirmation)

### 1. Determine the correct fix layer

There are several possible fix approaches. The right one depends on whether this is
considered a hardware quirk to work around, a firmware assert that's too strict, or
a kernel-level issue:

**Option A: Fix in the kernel (localized, lowest risk)**

Re-snapshot `noc_reads_num_issued` after the multicast writes to absorb the spurious
HW increments, then let the firmware assert pass naturally:

```cpp
// After the multicast write loop in reader_mcast_sender_unary_sharded_ln.cpp:
noc_reads_num_issued[noc_index] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
```

This is simple but only fixes THIS kernel. Any other kernel that does non-posted
multicast writes on Wormhole will have the same issue.

**Option B: Fix in the firmware assert (broader, medium risk)**

Change the `NKFW` assert in `brisck.cc` to re-snapshot before checking, or use `>=`:

```cpp
// Instead of:
ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
// Use:
ASSERT(NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_RD_RESP_RECEIVED) >= noc_reads_num_issued[NOC_INDEX],
       DebugAssertNCriscNOCReadsFlushedTripped);
```

This handles all kernels but weakens the assert (can't detect HW > SW mismatches
from other bugs). Needs discussion with firmware team.

**Option C: Fix in the NOC write API (broadest, highest risk)**

Make `ncrisc_noc_fast_write` (for non-posted multicast) also increment
`noc_reads_num_issued` to account for the HW behavior. This is the most
architecturally correct fix but needs confirmation from the hardware team about
exactly which write operations cause the increment and under what conditions.

### 2. Investigate scope: which write operations cause the increment?

The delta is 2 with 2 loop iterations. Each iteration does:
- `noc_async_write_multicast` (linked=true, non-posted)
- `noc_semaphore_set_multicast` (non-posted)

Need to determine:
- Is it the multicast write, the semaphore set, or both?
- Does `linked=true` matter?
- Does this happen for unicast non-posted writes too?
- Is this Wormhole-specific or also affects Blackhole?

### 3. Search for other affected kernels

Any kernel that:
1. Uses `DM_DEDICATED_NOC` mode
2. Issues non-posted multicast writes on its NOC
3. Has those writes as the last operations before kernel exit

...will have the same watcher assert failure. Search the codebase for other
`noc_async_write_multicast` calls in kernels to assess the scope.

### 4. Discuss with hardware team

Confirm whether `NIU_MST_RD_RESP_RECEIVED` incrementing on non-posted multicast
write acks is expected Wormhole behavior, a known errata, or a newly discovered quirk.

---

## Key Observations

1. The test **only fails with watcher enabled** — without watcher, it passes consistently.
2. The failing kernel's **internal read barriers all pass** (the kernel completes normally).
3. **CONFIRMED:** `NIU_MST_RD_RESP_RECEIVED` increments by exactly 2 during the multicast
   write phase, with zero NOC reads issued. `HW=40→42, SW=40→40`.
4. The delta of 2 matches `num_all_to_all_workers_first_stage` (the multicast loop iteration count).
5. Adding a `noc_async_read_barrier()` at the end **hangs forever** because `HW > SW` and
   `SW` will never catch up — this is NOT a "pending reads" problem, it's a spurious HW
   counter increment problem.
6. The only NOC0 operations between the last successful barrier and kernel exit are
   **non-posted multicast writes with `linked=true`** and **`noc_semaphore_set_multicast`**.
