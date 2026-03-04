# Issue #37171: `test_layer_norm_sharded_two_stage` Watcher Failure Investigation

## Summary

When `TT_METAL_WATCHER=1` is enabled, `test_layer_norm_sharded_two_stage` fails with a
`DebugAssertNCriscNOCReadsFlushedTripped` assert on core (0,0) BRISC, indicating the kernel
`reader_mcast_sender_unary_sharded_ln.cpp` completed with pending NOC read transactions.

**Without watcher:** test passes in ~1.08s.
**With watcher:** test aborts due to tripped assert.

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

## Root Cause Hypotheses

### Hypothesis 1: Watcher NOC Sanitization Overhead Creates a Timing Window

When `WATCHER_ENABLED` is defined, every NOC transaction goes through sanitization macros
(`DEBUG_SANITIZE_NOC_READ_TRANSACTION`, `DEBUG_SANITIZE_NOC_WRITE_TRANSACTION` in `sanitize.h`).
These add significant overhead:
- Address validation logic
- `debug_insert_delay()` calls for each transaction (configurable delays)
- Additional register reads for validation

This overhead **slows down the kernel execution significantly**. With `DM_DEDICATED_NOC` mode:

1. `noc_local_state_init` snapshots `NIU_MST_RD_RESP_RECEIVED` into `noc_reads_num_issued[NOC_0]`
2. The kernel runs and issues reads, incrementing `noc_reads_num_issued[NOC_0]` each time
3. Hardware `NIU_MST_RD_RESP_RECEIVED` increments as read responses arrive

**The sanitization overhead slows down software counter increments relative to hardware events.**
This shouldn't cause a mismatch at the *end* though, since the barrier at line 236 synchronizes them.

### Hypothesis 2: `ALIGN_LOCAL_CBS_TO_REMOTE_CBS` Pre-Kernel NOC Operations (Less Likely)

In `brisck.cc:72-74`, before `kernel_main()`:

```cpp
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
```

This macro, when defined, expands to code generated at `program.cpp:1022-1056` that calls
`align_local_cbs_to_remote_cb()`. However, examining the implementation in
`remote_circular_buffer.h:387-405`, this function only manipulates CB interface structs
(fifo pointers, sizes) — **it does NOT issue NOC reads**. So this is likely not the cause.

### Hypothesis 3: Watcher Sanitize NOC Address Check Interfering with NOC Read Counter (Most Likely)

Looking more carefully at the sanitize macros in `sanitize.h`:

```cpp
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_(noc_id, noc_a, worker_a, l, check_linked)
    debug_sanitize_noc_and_worker_addr(noc_id, noc_a, worker_a, l, ...);
    LOG_LEN(l);
    debug_insert_delay((uint8_t)TransactionRead);
```

The sanitize macro for reads is called **BEFORE** the actual NOC read command is issued
to hardware. However, the software counter `noc_reads_num_issued[noc]` is incremented
**AFTER** the hardware command is issued (`NOC_CMD_CTRL, NOC_CTRL_SEND_REQ`). The typical
flow in `ncrisc_noc_fast_read` is:

```
1. [WATCHER] sanitize addresses, insert delay    ← slow
2. Write NOC command registers
3. NOC_CMD_CTRL = SEND_REQ                        ← hardware starts read
4. noc_reads_num_issued[noc] += 1                  ← software counter update
```

If the sanitize delay (step 1) is significant and there's a race between the hardware
completing a prior read vs the software processing the next read, this could create a
**temporary** mismatch. But it would self-resolve after the barrier.

### Hypothesis 4: Non-Posted Multicast Write Response Aliasing (Primary Suspect)

After the last read barrier at line 236, the kernel issues:

```cpp
noc_async_write_multicast(
    l1_read_addr_ex_global,
    multicast_data_noc | l1_read_addr_ex_global,
    num_tiles_scaler * num_tiles_bytes,
    num_blocks - 1,
    true);  // linked = true
noc_semaphore_set_multicast(
    reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);
```

These are **non-posted** writes on NOC0. In the Wormhole NOC architecture, non-posted
writes expect acknowledgement responses. The `ncrisc_noc_fast_write` function increments
`noc_nonposted_writes_num_issued` and `noc_nonposted_writes_acked` — NOT `noc_reads_num_issued`.

**However**, there's a subtle possibility: on Wormhole, the NOC hardware may share some
response paths between read responses and write acknowledgements, especially for linked
multicast transactions. If the hardware's `NIU_MST_RD_RESP_RECEIVED` register gets
spuriously incremented by a write-ack response (a hardware quirk), this would cause
`NIU_MST_RD_RESP_RECEIVED > noc_reads_num_issued[NOC_0]`, and:

- The barrier at line 236 would pass (both equal at that point)
- After the multicast writes, `NIU_MST_RD_RESP_RECEIVED` would be spuriously incremented
- At kernel exit, `NIU_MST_RD_RESP_RECEIVED != noc_reads_num_issued[NOC_0]` → assert trips
- An added `noc_async_read_barrier()` at the end would hang forever because
  `NIU_MST_RD_RESP_RECEIVED > noc_reads_num_issued[NOC_0]` and nothing will increment
  `noc_reads_num_issued[NOC_0]` to catch up

**This hypothesis perfectly explains both the original assert AND the hang with the added barrier.**

### Hypothesis 5: Watcher Sanitize Overhead Changes Timing of Multicast Write Acks

A variant of Hypothesis 4: the multicast write acks are always arriving, but without watcher
the timing works out such that they arrive before `noc_local_state_init` of the *next*
kernel invocation (which re-snapshots the counters). With watcher's overhead, the kernel
runs slower, and the ack timing relative to the assert check changes.

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

## Recommended Next Steps

### 1. Verify counter mismatch direction (diagnostic)

Add temporary debug prints (or use `DPRINT`) right before the assert in `brisck.cc` to
print the actual values of `NIU_MST_RD_RESP_RECEIVED` and `noc_reads_num_issued[NOC_0]`
at the `NKFW` waypoint. This will tell us:
- Which direction the mismatch is in (HW > SW or SW > HW)
- By how much they differ

```cpp
// Temporary debug in brisck.cc at NKFW
WAYPOINT("NKFW");
DPRINT << "RD_RESP=" << NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_RD_RESP_RECEIVED)
       << " SW=" << noc_reads_num_issued[NOC_INDEX] << ENDL();
```

### 2. Verify multicast write ack hypothesis (diagnostic)

Add counter reads right after the last `noc_async_read_barrier()` at line 236 and again
after the multicast write loop to see if `NIU_MST_RD_RESP_RECEIVED` changes during the
write phase:

```cpp
// After line 236 in reader_mcast_sender_unary_sharded_ln.cpp:
noc_async_read_barrier();
uint32_t rd_resp_before = NOC_STATUS_READ_REG(0, NIU_MST_RD_RESP_RECEIVED);
uint32_t sw_before = noc_reads_num_issued[0];

// ... multicast write loop ...

noc_async_write_barrier();
uint32_t rd_resp_after = NOC_STATUS_READ_REG(0, NIU_MST_RD_RESP_RECEIVED);
// DPRINT both before/after values
```

If `rd_resp_after != rd_resp_before`, the multicast writes are affecting the read counter.

### 3. If multicast writes are the cause

The fix would be to call `noc_async_read_barrier()` at the end, but **first update the
software counter** to account for the spurious increments. Or, investigate whether the
`linked=true` flag on the multicast write is the root cause and whether removing it
(or using posted writes) avoids the issue.

### 4. Check if this is a known Wormhole NOC quirk

Search for similar issues or comments about `NIU_MST_RD_RESP_RECEIVED` being affected
by write transactions, especially linked multicasts. Check with hardware team if this
is expected behavior.

### 5. Alternative: Adjust the firmware assert

If the hardware register aliasing is confirmed, the firmware assert in `brisck.cc` may
need to account for this. For example, checking `>=` instead of `==`, or re-snapshotting
the counters after writes complete.

---

## Key Observations

1. The test **only fails with watcher enabled** — without watcher, it passes consistently.
2. The failing kernel's **internal read barriers all pass** (the kernel completes normally).
3. The mismatch appears **after the kernel's last read barrier** but before firmware checks.
4. Adding a read barrier at the end **hangs** rather than fixing the issue, proving the
   software and hardware counters are out of sync in a way that can never self-resolve.
5. The only NOC0 operations between the last successful barrier and kernel exit are
   **non-posted multicast writes with `linked=true`**.
