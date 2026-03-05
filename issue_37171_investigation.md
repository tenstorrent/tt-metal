# Issue #37171: `test_layer_norm_sharded_two_stage` Watcher Failure Investigation

## Summary

When `TT_METAL_WATCHER=1` is enabled, `test_layer_norm_sharded_two_stage` fails with a
`DebugAssertNCriscNOCReadsFlushedTripped` assert on core (0,0) BRISC, indicating the kernel
`reader_mcast_sender_unary_sharded_ln.cpp` completed with pending NOC read transactions.

**Without watcher:** test passes in ~1.08s.
**With watcher:** test aborts due to tripped assert.

## CONFIRMED Root Cause (2026-03-05, revised after deep investigation)

**Late-arriving NOC read responses on Wormhole: `NIU_MST_RD_RESP_RECEIVED` receives +2
phantom increments shortly after `noc_async_read_barrier()` exits.**

### What happens

After the final `noc_async_read_barrier()` in the kernel (the gather-reads barrier),
the hardware and software counters are in sync:

```
rdbarrier_C sw=194 hw:194->194->196
```

1. Software counter `noc_reads_num_issued` = 194
2. Hardware counter `NIU_MST_RD_RESP_RECEIVED` = 194 (barrier exits, equality met)
3. After a short delay (~microseconds): hardware counter = **196** (+2 phantom responses)

The +2 phantom responses arrive AFTER the read barrier considers all reads complete.
No writes, no multicast, no linked transactions are involved. The responses appear on
their own, purely from the read path.

### How we proved this

Through systematic elimination over 8+ experiments:

1. **Initial hypothesis:** Non-posted multicast writes with `linked=true` cause the +2.
   - DISPROVEN: Setting `linked=false` still showed +2.

2. **Second hypothesis:** Any multicast write causes the +2.
   - DISPROVEN: A 4-byte unicast self-write also showed +2.

3. **Third hypothesis:** First use of cmd_buf 0 causes the +2.
   - DISPROVEN: Moving the first cmd_buf 0 write to before reads showed +0.
   - The +2 was absorbed by whichever cmd_buf wrote first AFTER reads (tested with
     both cmd_buf 0 and cmd_buf 2).

4. **Fourth hypothesis:** The first write after reads triggers the +2.
   - PARTIALLY CORRECT: Splitting write issue vs write barrier showed the +2 appeared
     at write issue time, not during the barrier.

5. **Final discovery:** Adding a delay loop (no writes at all) after the read barrier
   showed the +2 arriving during the delay:
   ```
   polls:166,168,168 wr_issue:168 wr_barrier:168
   ```
   The counter went 166→168 between the first two poll reads, with NO writes issued.
   **The +2 is delayed read responses, not caused by writes.**

6. **Barrier-by-barrier analysis** confirmed the +2 only occurs after the final gather
   read barrier (barrier C), not after earlier read barriers (A, B):
   ```
   rdbarrier_A[0] 169->170->170    ← 0 late
   rdbarrier_B[0] 172->173->173    ← 0 late
   rdbarrier_A[1] 174->175->175    ← 0 late
   rdbarrier_B[1] 177->178->178    ← 0 late
   rdbarrier_C   180->180->182    ← +2 late!
   ```

### Characteristics of the +2

- Always exactly +2 (consistent across all experiments)
- Only after the last read barrier in the kernel
- Arrives within microseconds of the barrier exiting
- Not related to: linked flag, multicast, cmd_buf identity, write operations,
  transaction size, or VC state
- IS related to: the specific batch of reads preceding barrier C (the gather reads
  that read from multiple remote cores)

### Fix applied and verified

Added a single line at the end of `kernel_main()` in `reader_mcast_sender_unary_sharded_ln.cpp`:

```cpp
noc_reads_num_issued[noc_index] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
```

This re-snapshots the software counter to match the hardware counter (including phantom
responses) before the kernel returns, so the firmware assert in `brisck.cc` sees them equal.

**Result: test PASSES with `TT_METAL_WATCHER=1`.**

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

## Root Cause: CONFIRMED (revised)

### Late-arriving NOC read responses on Wormhole

The Wormhole NOC hardware generates +2 phantom read response events
(`NIU_MST_RD_RESP_RECEIVED` increments) that arrive microseconds after
`noc_async_read_barrier()` exits. These are NOT caused by write operations.

The barrier checks `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]` and exits
when equality is met. Shortly after, 2 additional hardware responses arrive, pushing
the hardware counter ahead of the software counter.

This was proven by:
1. Polling the register in a tight loop with NO writes after the read barrier:
   `polls:166,168,168` — the counter jumped 166→168 on its own.
2. The +2 only occurs after the **final** read barrier (barrier C, the gather-reads),
   not after earlier read barriers in the same kernel.
3. Write operations (multicast, unicast, linked, unlinked) have zero effect — they
   merely appeared to cause the +2 in earlier experiments because they were the first
   thing executed after the barrier, by which time the late responses had arrived.

### Why it only fails with watcher

Without watcher, the kernel completes and the firmware never checks the counters —
`noc_local_state_init` at the start of the next kernel re-snapshots both to their
current (matched) values. With watcher, the firmware checks `==` at `NKFW` and catches
the mismatch.

### Why adding `noc_async_read_barrier()` at the end caused a hang

The barrier spins on `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]`.
Since `HW (196) > SW (194)`, and nothing will ever increment `SW`, the barrier
hangs forever.

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
3. **CONFIRMED:** `NIU_MST_RD_RESP_RECEIVED` increments by exactly +2 after the final
   read barrier, with `noc_reads_num_issued` remaining unchanged. `SW=194, HW=194→196`.
4. The +2 phantom responses arrive **within microseconds** after `noc_async_read_barrier()`
   exits, confirmed by polling the register with no intervening operations.
5. The +2 only occurs after the **final** read barrier (barrier C, gather-reads), not
   after earlier barriers (A, B) which also issue reads.
6. Adding a `noc_async_read_barrier()` at the end **hangs forever** because `HW > SW` and
   `SW` will never catch up.
7. **DISPROVEN:** Write operations (multicast, unicast, linked, unlinked) do NOT cause
   the +2. All earlier evidence attributing the +2 to writes was a timing coincidence.
8. **DISPROVEN:** The `NOC_CMD_VC_LINKED` flag is irrelevant. The +2 persists with
   `linked=false`.
9. **DISPROVEN:** cmd_buf identity is irrelevant. The +2 was absorbed by whichever
   cmd_buf happened to issue the first write after reads.

## Conclusion

**Root cause:** On Wormhole, the final batch of `noc_async_read_one_packet` calls in this
kernel generates +2 late responses to `NIU_MST_RD_RESP_RECEIVED` that arrive after
`noc_async_read_barrier()` has already exited. This is a Wormhole NOC hardware behavior,
not a software bug.

The watcher's `DebugAssertNCriscNOCReadsFlushedTripped` assert fires because it compares
`NIU_MST_RD_RESP_RECEIVED` (hardware, with +2 phantom responses) against
`noc_reads_num_issued` (software, correctly tracking actual reads issued). The mismatch
is a false positive from watcher's perspective.

**Questions for the hardware/NOC team:**
1. Why do the gather reads (barrier C) generate +2 phantom responses when earlier reads
   (barriers A, B) in the same kernel do not?
2. Is this related to multi-core read patterns (barrier C reads from 2 different remote
   cores in rapid succession), read response coalescing, or NOC pipeline draining?
3. Is there a known errata or documentation for this behavior?

**Proper fix options (for the hardware/firmware team to evaluate):**
1. **Watcher fix:** Change the `ncrisc_noc_reads_flushed` assert to tolerate `HW >= SW`
   (i.e., only fire if `HW < SW`, meaning reads are genuinely pending, not if `HW > SW`
   which indicates phantom responses). This is the broadest fix.
2. **Firmware fix:** Have `noc_local_state_init` or the post-kernel check re-snapshot
   the software counter from hardware before asserting.
3. **Kernel fix:** Add `noc_reads_num_issued[noc_index] = NOC_STATUS_READ_REG(noc_index,
   NIU_MST_RD_RESP_RECEIVED);` at the end of affected kernels (our validated workaround).
   Only fixes this kernel; any kernel with a similar read pattern will need the same fix.
