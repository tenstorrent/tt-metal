<!--
SUMMARY: Investigation of non-MMIO sysmem_manager reset gap — stale cq_to_quiesced / prefetch_q_in_flight after relay-broken re-init
KEYWORDS: sysmem_manager, non-mmio, reset, cq_to_quiesced, prefetch_q_in_flight, relay-broken, FIX-AL
SOURCE: Code analysis of nsexton/0-racecondition-hunt worktree
SCOPE: system_memory_manager.cpp/hpp, device.cpp configure_command_queue_programs, non-MMIO init path
USE WHEN: Investigating stale CQ state after non-MMIO device re-initialization
-->

# Non-MMIO `sysmem_manager_->reset()` Gap Analysis

## Question

In `device.cpp::configure_command_queue_programs()` (line 195), `sysmem_manager_->reset(cq_id)` is called only inside the `if (this->is_mmio_capable())` guard. Non-MMIO devices never get their `sysmem_manager_->reset()` called during re-initialization. Can this leave `cq_to_quiesced`, `prefetch_q_in_flight`, or `prefetch_q_dev_fences` in a stale state?

## What `reset()` Does

File: `tt_metal/impl/dispatch/system_memory_manager.cpp`, line 347.

```
reset(cq_id):
  1. issue_fifo_wr_ptr     = start position (16B words)
  2. issue_fifo_wr_toggle  = false
  3. completion_fifo_rd_ptr = issue_fifo_limit
  4. completion_fifo_rd_toggle = false
  5. cq_to_quiesced[cq_id] = false          (atomic store, release)
  6. prefetch_q_in_flight[cq_id] = 0
  7. prefetch_q_dev_fences[cq_id] = HW register read (PREFETCH_Q_RD)
```

Step 7 is the critical difference from the constructor: `reset()` reads the **actual hardware fence register** to sync `dev_fences` with whatever firmware currently holds, whereas the constructor sets `dev_fences` to the sentinel value (prefetch_q_limit = base + entries * entry_size).

## Lifecycle Analysis

### Path 1: Initial device open (close + reopen)

1. `Device::close()` calls `sysmem_manager_.reset()` (unique_ptr reset — **destroys** the sysmem_manager).
2. `Device::init_command_queue_host()` creates a **new** `SystemMemoryManager` via `std::make_unique`.
3. Constructor initializes all fields clean:
   - `prefetch_q_in_flight(num_hw_cqs, 0)` — zero
   - `cq_to_quiesced = make_unique<atomic<bool>[]>(num_hw_cqs)` — value-initialized to `false`
   - `prefetch_q_dev_fences[cq_id] = prefetch_q_limit` (sentinel) — set in `init_dispatch_core_interfaces()` line 249
4. `init_command_queue_device_with_topology()` → `configure_command_queue_programs()`:
   - MMIO device: calls `reset()` inside the `is_mmio_capable()` guard (reads HW fence register)
   - Non-MMIO device: skips `reset()` — but sysmem_manager was just freshly constructed, all fields are already clean
5. For freshly-loaded dispatch firmware: HW fence register == sentinel (limit) == what the constructor set. **No mismatch.**

**Verdict: No stale state. Fresh construction provides clean initialization.**

### Path 2: Quiesce/restart cycle (fabric workers restarted, CQ dispatch NOT restarted)

1. `finish_and_reset_in_use()` drains CQ, then:
   - Sets `cq_to_quiesced = true` on **all devices** (including non-MMIO) via `set_quiesced(id_, true)`
   - Sets event counters: reference CQ gets `UINT32_MAX`, others get `0` / `UINT32_MAX`
2. `quiesce_and_restart_fabric_workers()` restarts fabric ERISC/MUX cores. **Does NOT restart CQ dispatch programs.** **Does NOT call `configure_command_queue_programs()`.**
3. sysmem_manager is **reused** (not destroyed/recreated).
4. When new workload starts: `mark_in_use()` clears `cq_to_quiesced = false` on **all devices**.
5. CQ dispatch firmware was never restarted, so `prefetch_q_dev_fences` still matches the HW fence register (no divergence).
6. `prefetch_q_in_flight` should be 0 because `finish_and_reset_in_use()` called `finish_nolock()` which waits for all in-flight work to complete.

**Verdict: No stale state in the normal quiesce path.** The `cq_to_quiesced` flag is properly managed by `finish_and_reset_in_use()` (sets true) and `mark_in_use()` (clears to false) without relying on `reset()`.

### Path 3: DispatchContext manual FD setup/teardown

1. `dispatch_context.cpp:77`: `init_command_queue_host()` called for **all** active devices — creates fresh sysmem_manager.
2. `dispatch_context.cpp:80`: `initialize_dispatch_firmware()` → `init_command_queue_device_with_topology()` for all devices.
3. Non-MMIO devices skip `reset()` in `configure_command_queue_programs()`, but sysmem_manager was just freshly constructed.

**Verdict: No stale state. Same as Path 1.**

## Answering the Three Specific Questions

### Q1: Can `cq_to_quiesced` short-circuit `wait_for_pending_events` in a new session?

**No, in current code.** There are two mechanisms that clear it:

1. **Close/reopen path**: sysmem_manager is destroyed and reconstructed (constructor value-initializes `cq_to_quiesced` to `false`).
2. **Quiesce path**: `mark_in_use()` clears `cq_to_quiesced = false` on all devices (including non-MMIO) before any new work is enqueued.

The `is_quiesced()` check at `distributed.cpp:51,55` and `mesh_buffer.cpp:362` will only short-circuit if `cq_to_quiesced` is `true`, which correctly indicates all prior work was drained. Once `mark_in_use()` runs (before new events are issued), it's cleared.

**However, there is a subtle concern**: if a non-MMIO device's sysmem_manager survives into a new session through some path that bypasses both destruction and `mark_in_use()`, the stale `true` would cause silent event skipping. This is currently not reachable but would be if a new init path were added without calling `mark_in_use()` or reconstructing the sysmem_manager.

### Q2: Can `prefetch_q_in_flight` underflow?

**No, in the normal paths.** The two protection mechanisms:

1. **Close/reopen**: Constructor sets `prefetch_q_in_flight(num_hw_cqs, 0)`.
2. **Quiesce**: `finish_nolock()` drains all in-flight entries (blocks until device catches up), so by the time `finish_and_reset_in_use()` returns, `prefetch_q_in_flight` should be 0 (or very close — the drain loop in `fetch_queue_reserve_back` at line ~730 reconciles it).

The underflow clamp at lines 739-745 (`consumed <= this->prefetch_q_in_flight[cq_id]`) is a safety net. It fires a `TT_ASSERT` but then clamps to 0 instead of wrapping. This could mask a bug but won't crash in release builds.

**Risk scenario**: If `finish_nolock()` throws during quiesce (e.g., dispatch timeout), the scope guard still sets `cq_to_quiesced = true`, but `prefetch_q_in_flight` may be non-zero. If the CQ dispatch firmware is then restarted (without reconstructing sysmem_manager), the stale `in_flight` count would cause the underflow clamp to fire on the first `fetch_queue_reserve_back` in the new session. This is a genuine edge case but requires a throw during `finish_nolock()` followed by continued use of the same CQ — which currently wouldn't happen because the throw propagates up and halts the session.

### Q3: Can `prefetch_q_dev_fences` mismatch after HW reset?

**This is the most interesting case.** The constructor sets `dev_fences` to the sentinel (limit), and `reset()` reads the actual HW register. If CQ dispatch firmware is freshly loaded, HW register == sentinel → no mismatch. If firmware was NOT reloaded (quiesce path), HW register == last consumed ptr → same as what `dev_fences` already holds (no drift because CQ was drained).

**Genuine gap**: If the dispatch firmware is reloaded on a non-MMIO device WITHOUT reconstructing the sysmem_manager, `dev_fences` would be stale (holding the old consumed ptr) while HW is at sentinel (limit). The `count_consumed()` function would then compute wrong deltas. **This path does not currently exist** — dispatch firmware reload always goes through `init_command_queue_device_with_topology()`, which is only called after `init_command_queue_host()` (fresh sysmem_manager).

## Is This a Genuine Gap?

**Mostly no, but with a defensive improvement opportunity.**

The current code is correct because:
1. Every path that reconstructs CQ dispatch also reconstructs the sysmem_manager (constructor provides clean state).
2. The quiesce path doesn't reconstruct CQ dispatch, so the sysmem_manager's existing state is still valid.
3. `cq_to_quiesced` is properly managed by `finish_and_reset_in_use()` / `mark_in_use()` without relying on `reset()`.

**The asymmetry is still a code quality concern**: the `is_mmio_capable()` guard around `reset()` creates an implicit assumption that non-MMIO devices always get a fresh sysmem_manager before `configure_command_queue_programs()`. If someone adds a path that calls `configure_command_queue_programs()` on a non-MMIO device with an existing sysmem_manager (e.g., a hot-restart of dispatch firmware), the stale state would be silently inherited.

## Recommended Defensive Fix

Make `reset()` unconditional in `configure_command_queue_programs()` — call it for ALL devices, not just MMIO:

```cpp
// In device.cpp::configure_command_queue_programs(), around line 195:
// Move the reset() call OUTSIDE the is_mmio_capable() guard:
for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
    this->sysmem_manager_->reset(cq_id);
}

if (this->is_mmio_capable()) {
    // ... existing hugepage pointer reset logic ...
}
```

**Cost**: One additional `read_core()` per CQ per non-MMIO device (reads the HW fence register through the relay). This is a single 4-byte read — negligible compared to the rest of init.

**Benefit**: Eliminates the implicit assumption and makes the code robust against future init path changes. Also syncs `dev_fences` with the actual hardware state instead of relying on the constructor's sentinel.

**Caveat for relay-broken devices**: `reset()` calls `read_core()` to read the HW fence register. For a non-MMIO device with a broken relay, this would hang/timeout. The fix should be guarded:

```cpp
for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
    // reset() reads HW fence via relay; skip for broken-relay devices
    // (constructor sentinel is acceptable for fresh firmware)
    if (!this->fabric_relay_path_broken_.load()) {
        this->sysmem_manager_->reset(cq_id);
    }
}
```

## Summary

| Field | Stale in close/reopen? | Stale in quiesce? | Stale in DispatchContext? |
|-------|----------------------|-------------------|--------------------------|
| `cq_to_quiesced` | No (constructor = false) | No (mark_in_use clears) | No (constructor = false) |
| `prefetch_q_in_flight` | No (constructor = 0) | No (finish drains) | No (constructor = 0) |
| `prefetch_q_dev_fences` | No (constructor = sentinel = HW) | No (CQ not restarted) | No (constructor = sentinel = HW) |

**No currently-reachable stale state.** The gap is theoretical but real: if a future code path reloads CQ dispatch firmware on a non-MMIO device without reconstructing its sysmem_manager, `dev_fences` will mismatch. The defensive fix (unconditional `reset()`) is low-cost and eliminates the hazard.
