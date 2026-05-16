<!--
SUMMARY: Complete dispatch path trace from ttnn::write_buffer to non-MMIO device 4, hang mechanism via dead ERISC routers, and proposed fixes for Options 2 and 3
KEYWORDS: dispatch, write_buffer, NOC, 0x880030060, ARC, T3K, remote device, ERISC, MUX, quiesce, hang, thread_pool, fd_mesh_command_queue, topology, FABRIC_MUX, PREFETCH_H, DISPATCH_D
SOURCE: Local code analysis of branch nsexton-0-racecondition-hunt
SCOPE: Full dispatch path for write_buffer + record_event to non-MMIO chips, thread pool hang mechanism, ERISC zero state root cause, Option 2/3 proposals
USE WHEN: Investigating T3K dispatch hangs on non-MMIO chips, ARC scratch register corruption, fabric relay dispatch issues, quiesce/restart behavior
-->

# Researcher C: Complete Dispatch Path, Hang Mechanism, and Fix Proposals

## 1. Complete Dispatch Path: ttnn::write_buffer to Device 4

### 1.1 Entry Point: ttnn::write_buffer

**File**: `ttnn/core/async_runtime.cpp:13-22`

```
ttnn::write_buffer(cq_id, dst_tensor, src_data)
  -> mesh_device->mesh_command_queue(*cq_id)
  -> for each device_tensor in get_device_tensors(dst):
       tt::tt_metal::copy_to_device(cq, src.at(i), device_tensors[i])
```

Each device tensor gets its own `copy_to_device` call. For a 4-device mesh, this iterates over devices 0, 1, 4, 5.

### 1.2 copy_to_device -> enqueue_write_tensor

**File**: `ttnn/core/tensor/tensor_ops.cpp:128-136`

```
copy_to_device(queue, src_bytes, mesh_tensor, region)
  -> enqueue_write_tensor(queue, src, mesh_tensor, region)
```

### 1.3 enqueue_write_shards_nolock (THE HANG POINT)

**File**: `tt_metal/distributed/mesh_command_queue_base.cpp:234-284`

```cpp
for (shard_idx : shard_data_transfers) {
    auto shard_coord = shard_data_transfers[shard_idx].shard_coord;
    dispatch_thread_pool_->enqueue(
        [&dispatch_lambda, shard_idx]() { dispatch_lambda(shard_idx); },
        mesh_device_->impl().get_device(shard_coord)->id());
}
dispatch_thread_pool_->wait();  // line 267 — HANGS HERE for device 4
```

The `dispatch_lambda` calls `write_shard_to_device()` for each shard.

### 1.4 write_shard_to_device

**File**: `tt_metal/distributed/fd_mesh_command_queue.cpp:585-618`

```cpp
void FDMeshCommandQueue::write_shard_to_device(...) {
    auto* target_device = mesh_device_->impl().get_device(shard_coord);
    auto& sysmem_manager = target_device->sysmem_manager();
    buffer_dispatch::write_to_device_buffer(sysmem_manager, ...);
}
```

Key insight: Each device has its **own** `SystemMemoryManager`. For non-MMIO device 4, the sysmem_manager writes to a hugepage region that is read by the MMIO peer's PREFETCH_H.

### 1.5 buffer_dispatch::write_to_device_buffer

**File**: `tt_metal/impl/buffers/dispatch.cpp:1125+`

Gets the device's sysmem_manager, then writes dispatch commands (WRITE_LINEAR or WRITE_PAGED) into the hugepage issue queue:
1. `issue_queue_reserve(cmd_size)` — blocks if issue queue is full (back-pressure from slow prefetcher)
2. Encodes CQ_DISPATCH_CMD_WRITE_LINEAR with target address, size, data
3. `issue_queue_push_back(cmd_size)` — advances write pointer
4. `fetch_queue_write(cmd_size)` — signals prefetcher via fetch queue doorbell

### 1.6 SystemMemoryManager -> Prefetcher

**File**: `tt_metal/impl/dispatch/system_memory_manager.cpp:790-820`

`fetch_queue_write` writes the command size to the prefetch queue slot, then pushes the device pointer. The prefetcher hardware/firmware polls this queue.

### 1.7 N300 2CQ Dispatch Topology (Remote Path)

**File**: `tt_metal/impl/dispatch/topology.cpp:159-180` (`two_chip_arch_2cq_fabric`)

For non-MMIO device 4, the dispatch topology is:

```
Host hugepage (device 4's sysmem_manager)
    |
    v
PREFETCH_H [node 4] (on MMIO chip, reads hugepage)
    |
    v
FABRIC_MUX [node 8] (Tensix MUX on MMIO chip, bridges to ETH)
    |
    v  (ETH link via ERISC routers)
    |
PREFETCH_D [node 9] (on device 4, receives from ETH)
    |
    v
DISPATCH_D [node 11] (on device 4, executes dispatch commands)
    |
    v
Device 4 L1/DRAM (write destination)
```

Topology node assignments:
- Nodes 0-3: PREFETCH_HD + DISPATCH_HD on MMIO chip (CQ0, CQ1) — for local MMIO chip
- Nodes 4-5: PREFETCH_H on MMIO chip — for remote chip (CQ0, CQ1)
- Nodes 6-7: DISPATCH_H on MMIO chip — host-side dispatch for remote
- Node 8: FABRIC_MUX — Tensix MUX bridging to ETH
- Nodes 9-10: PREFETCH_D on remote chip (CQ0, CQ1)
- Nodes 11-12: DISPATCH_D on remote chip (CQ0, CQ1)
- Node 13: RETURN_FABRIC_MUX — return path

### 1.8 ETH Fabric Relay

The FABRIC_MUX (Tensix kernel `tt_fabric_mux.cpp`) receives commands from PREFETCH_H and forwards them over the ETH link via ERISC router connections:

```cpp
// tt_fabric_mux.cpp — simplified flow
while (true) {
    if (fabric_connection.has_data()) {
        auto data = fabric_connection.read();
        erisc_connection.send(data);  // -> ETH link -> remote chip ERISC -> PREFETCH_D
    }
}
```

The ERISC routers on both ends of the ETH link handle the actual Ethernet packet framing, credit management, and flow control.

## 2. How issue_record_event_commands Reaches Non-MMIO Devices

**File**: `tt_metal/distributed/fd_mesh_command_queue.cpp:718-760`

```cpp
void FDMeshCommandQueue::enqueue_record_event_helper(MeshEvent& event, ...) {
    for_each_local(mesh_device_, event.device_range(), [&](const auto& coord) {
        dispatch_thread_pool_->enqueue(
            [&dispatch_lambda, coord]() { dispatch_lambda(coord); },
            mesh_device_->impl().get_device(coord)->id());
    });
    dispatch_thread_pool_->wait();  // line 757 — waits for ALL devices
}
```

The `dispatch_lambda` calls `event_dispatch::issue_record_event_commands()` which:

**File**: `tt_metal/impl/event/dispatch.cpp:42-201`

1. Encodes WAIT_STREAM commands (wait for worker completion)
2. Encodes WRITE_PACKED command (unicast event ID to dispatch cores)
3. Encodes WRITE_LINEAR_HOST command (notify host via completion queue)
4. Writes all to the device's hugepage issue queue via sysmem_manager
5. Signals prefetcher via fetch_queue_write

For device 4, these commands travel the same PREFETCH_H -> FABRIC_MUX -> ERISC -> PREFETCH_D -> DISPATCH_D path. The DISPATCH_D kernel processes the event commands on the remote chip.

**ETH fabric health dependency**: If device 4's ERISC routers are dead/zeroed, the FABRIC_MUX on the MMIO peer cannot relay packets. This causes:
- PREFETCH_H to stall (no ETH credits, can't send)
- Fetch queue to stop draining
- `issue_queue_reserve()` to block (issue queue full)
- Worker thread to block indefinitely
- `dispatch_thread_pool_->wait()` to hang

## 3. Dispatch Command Sequence on Device 4's Tensix

When things work correctly, DISPATCH_D on device 4 processes these commands in order:

For a write_buffer operation:
1. `CQ_DISPATCH_CMD_WRITE_LINEAR` — NOC write of tensor data to device 4 L1/DRAM
2. (possibly multiple WRITE_LINEAR for large tensors)

For record_event:
1. `CQ_DISPATCH_CMD_WAIT` (WAIT_STREAM) — spin on stream registers for worker completion
2. `CQ_DISPATCH_CMD_WRITE_PACKED` (FLAG_TYPE_EVENT) — unicast event ID to dispatch cores
3. `CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST` — write event to host completion queue

The DISPATCH_D kernel (`cq_dispatch.cpp`) compiles with `FABRIC_RELAY=1` for remote chips, enabling fabric relay code paths in `cq_relay.hpp`.

## 4. Why dispatch_thread_pool_->wait() Hangs

### Thread Pool Architecture

**File**: `tt_metal/impl/threading/thread_pool.cpp`

`DeviceBoundThreadPool` has one `NumaAwareExecutor` per physical device. Each executor has:
- A lock-free `TaskQueue` (ring buffer, 65536 slots)
- A single worker thread
- An atomic `task_counter_` for synchronization

`wait()` implementation:
```cpp
std::exception_ptr NumaAwareExecutor::wait() const {
    int current;
    while ((current = task_counter_.load(std::memory_order_acquire)) > 0) {
        task_counter_.wait(current, std::memory_order_relaxed);
    }
    return stored_exception_;
}
```

`DeviceBoundThreadPool::wait()` calls `worker->wait()` on **all** workers. If device 4's worker never completes, the entire wait blocks.

### The Hang Chain

```
1. enqueue_write_shards_nolock dispatches write_shard_to_device for device 4
2. Worker thread calls buffer_dispatch::write_to_device_buffer
3. write_to_device_buffer calls sysmem_manager.issue_queue_reserve(cmd_size)
4. issue_queue_reserve blocks because the issue queue is full
5. The issue queue is full because the fetch queue isn't draining
6. The fetch queue isn't draining because PREFETCH_H can't send to FABRIC_MUX
7. FABRIC_MUX can't send because the ERISC ETH link is dead
8. The ERISC ETH link is dead because device 4's ERISC routers show all-zero state
9. The worker thread is stuck in issue_queue_reserve → never completes
10. task_counter_ never reaches 0 → wait() spins forever
11. TT_METAL_OPERATION_TIMEOUT_SECONDS=5 fires → TT_THROW: TIMEOUT
```

Alternative hang point: Even if the write_buffer commands make it through, the subsequent `record_event` can hang at step 4 of the same chain.

## 5. Why Device 4 ERISC Routers Show Zero State

### The CI Evidence

From the CI job log (run 24641059431):
```
Device 0: chans 0-5 → "OTHER" states (0x3f803f80, 0x40004000, ...) — initialized
Device 1: chans 0-5 → similar non-zero "OTHER" states — initialized
Device 4: ALL channels → 0x00000000 — ZERO STATE
Device 5: similar non-zero states — initialized
```

### Three Hypotheses

**Hypothesis A: AllGather Teardown Clears ERISC State on Device 4**

AllGather uses ERISC channels for inter-device communication. When AllGather completes, its teardown sequence may clear ERISC router state on device 4 but not on other devices, due to:
- Asymmetric teardown ordering (device 4 tears down first as the "receiver" end)
- The TERMINATED status write at `tt_fabric_mux.cpp:295` (`status_ptr[0] = FabricMuxStatus::TERMINATED`) propagates to ERISC via `close_start()` + `close_finish()`, but device 4's ERISCs may complete teardown before others

**Hypothesis B: Dispatch ERISCs Were Never Initialized for Device 4**

The fixture uses `FabricTensixConfig::DISABLED`:
```cpp
// Only 1-arg SetFabricConfig(FABRIC_1D) is called
// FabricTensixConfig stays DISABLED
// This means no Tensix MUX workers are created
```

With no Tensix MUX, the dispatch topology for device 4 may not include ERISC relay channels at all. The ERISC channels that show zero state may be the **dispatch** ERISCs (separate from AllGather ERISCs). If dispatch ERISCs were never configured, they'd show zero from the start — but this wouldn't explain why the first iteration works.

**Hypothesis C (Most Likely): quiesce_devices() After AllGather Leaves ERISCs in Limbo**

The test flow is:
1. AllGather runs → uses ERISC channels for data movement
2. AllGather completes → ERISC channels still hold AllGather routing state
3. `quiesce_devices()` is called
4. `quiesce_internal()` waits for CQ completion, then calls `quiesce_and_restart_fabric_workers()` per device
5. BUT `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` causes immediate return → ALL Phases skipped
6. ERISC channels are left in whatever state AllGather left them
7. Some channels may have been cleared by AllGather teardown (device 4)
8. Other channels may still hold stale routing tables (devices 0, 1, 5)

The zero state on device 4 specifically could be because:
- AllGather teardown on device 4's ERISCs completes fully (writing zeros to router state)
- AllGather teardown on other devices' ERISCs only partially completes or leaves routing state intact
- The ordering depends on which device is the "last sender" vs "first receiver" in the AllGather ring

## 6. The 0x880030060 Address (Two Distinct Failure Modes)

### Failure Mode 1: The Currently Observed Hang (env var IS set)

With `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1`:
- Phases 1-4 are ALL skipped
- No L1 overwrite occurs (Phase 3 never runs)
- No BRISC corruption → no wild NOC writes to 0x880030060
- The hang is purely due to dead ERISC channels, not ARC corruption
- `0x880030060` writes are NOT the cause of THIS hang

### Failure Mode 2: The Race Condition (env var NOT set)

Without the env var, when quiesce_and_restart_fabric_workers() runs fully:
1. Phase 2: Polls Tensix MUX for TERMINATED, then calls `assert_risc_reset_at_core` to halt BRISC
2. **BUT**: The MUX writes TERMINATED *before* `close_finish()` completes (tt_fabric_mux.cpp:292-296):
   ```cpp
   fabric_connection.close_start();      // sends close to ERISC
   noc_async_write_barrier();
   noc_async_atomic_barrier();
   status_ptr[0] = TERMINATED;           // <-- host sees this
   fabric_connection.close_finish();     // <-- still running!
   ```
3. Phase 2 sees TERMINATED, proceeds to halt BRISC
4. **Race window**: Between TERMINATED write and BRISC halt, `close_finish()` may still be polling
5. Phase 3: `configure_fabric_cores()` overwrites L1 of ERISC cores
6. If BRISC is still executing close_finish() when L1 is overwritten, it runs garbage
7. Garbage instructions generate wild NOC writes, including to 0x880030060 (ARC_RESET_SCRATCH_ADDR)
8. ARC state corrupted → subsequent dispatch operations fail

The branch fixes this with `assert_risc_reset_at_core()` in Phase 2 (device.cpp:~590) to halt BRISC before L1 overwrite.

## 7. Option 2 Investigation: Why Device 4 ETH Routers Show Zero

### Proposed Investigation Steps

**Step 1**: Add health probe BEFORE AllGather to establish baseline
- **File**: `tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp`
- **Line**: ~270 (before AllGather call)
- Add `FabricERISCHealthProbe(mesh_device)` to see if device 4 ERISCs are initialized pre-AllGather

**Step 2**: Add health probe AFTER AllGather but BEFORE quiesce
- **File**: Same test file
- **Line**: ~275 (after AllGather returns, before quiesce_devices)
- This shows whether AllGather teardown is what zeroes device 4's ERISCs

**Step 3**: Inspect AllGather ERISC teardown ordering
- **File**: `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`
- **Lines 459-503**: `close_start()` and `close_finish()` — check if device 4 is the first to complete teardown
- Look for asymmetric behavior in the AllGather ring topology

**Step 4**: Check if dispatch ERISCs vs AllGather ERISCs are separate
- The health probe may be reading dispatch ERISC channels, while AllGather uses different ERISC channels
- Need to verify which ERISC channels the health probe reads vs which AllGather uses

### Key Code References

- Health probe: `device.cpp` — search for `FabricERISCHealthProbe` or `FABRIC_HEALTH_PROBE`
- ERISC channel assignment: `tt_metal/fabric/impl/fabric_manager.cpp` — how channels are allocated to dispatch vs CCL
- AllGather teardown: `ttnn/cpp/ttnn/operations/ccl/all_gather/` — teardown sequence

## 8. Option 3 Proposal: Let Phase 2.5 + Phase 3 Run

### The Fix

**File**: `tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp`
**Line 96**: Remove or disable the env var:

```cpp
// BEFORE (causes hang by skipping ERISC re-initialization):
setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", 0);

// AFTER (let Phase 2.5 + Phase 3 run):
// setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", 0);  // REMOVED
```

### What This Enables

With the env var removed, `quiesce_and_restart_fabric_workers()` will run:

1. **Phase 1**: Send IMMEDIATELY_TERMINATE to Tensix MUX — **SKIPPED** (FabricTensixConfig::DISABLED, no MUX workers)
2. **Phase 2**: Poll MUX for TERMINATED + halt BRISC — **SKIPPED** (no MUX workers)
3. **Phase 2.5**: Send TERMINATE to each active ERISC channel, wait for response — **RUNS**
   - `device.cpp:603-710`
   - Reads each ERISC channel's sync address
   - If status == 0 or TERMINATED → skip (already clean)
   - Otherwise → send TERMINATE, poll for TERMINATED (500ms timeout)
4. **Phase 3**: `configure_fabric_cores()` + re-launch ETH cores — **RUNS**
   - `device.cpp:712-773`
   - Overwrites ERISC L1 with fresh firmware
   - Re-launches ERISC cores
5. **Phase 4**: Poll MUX for READY_FOR_TRAFFIC — **SKIPPED** (no MUX workers)

### Why This Should Fix the Hang

After quiesce_devices() with Phase 2.5 + Phase 3:
- Device 4's ERISC channels are properly terminated (Phase 2.5)
- Device 4's ERISC channels are re-initialized with fresh firmware (Phase 3)
- When "Enqueue dummy ops" runs, the ERISC relay path is alive
- PREFETCH_H -> FABRIC_MUX -> ERISC -> PREFETCH_D -> DISPATCH_D works
- write_buffer and record_event complete normally

### Remaining Risk: Phase 3 L1 Overwrite Race

Even with the env var removed, the Phase 2/3 L1 overwrite race (failure mode 2 from section 6) may still occur. The branch already has fixes for this:
- Phase 2: `assert_risc_reset_at_core()` to halt BRISC before L1 overwrite
- Phase 2.5: Global deadline for ERISC teardown poll
- These fixes must also be present for Option 3 to work reliably

### Additional Fix: Per-Device Quiesce Ordering

**File**: `tt_metal/distributed/mesh_device.cpp:1433-1459`

`quiesce_internal()` iterates devices and calls `quiesce_and_restart_fabric_workers()` per device. The ordering may matter:
- If MMIO devices are quiesced before non-MMIO devices, the ERISC relay may be torn down before the non-MMIO device can complete its quiesce
- Consider quiescing non-MMIO devices first, or all devices simultaneously

## 9. Summary of Findings

### Root Cause of the Hang

`TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` prevents ERISC re-initialization after AllGather. Device 4's ERISC routers end up in a zero/dead state (likely from AllGather teardown completing more aggressively on that device). Subsequent dispatch operations to device 4 via the ETH fabric relay cannot proceed because the ERISC link is dead.

### The 0x880030060 Issue is Separate

The ARC scratch register writes (0x880030060) are a **different** failure mode that occurs when the env var is NOT set and the Phase 2/3 L1 overwrite race triggers. In the currently observed hang, Phase 3 never runs, so no L1 overwrite occurs and no wild NOC writes to 0x880030060 happen.

### Recommended Fixes

1. **Immediate**: Remove `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` from the test fixture (line 96) — lets Phase 2.5 + Phase 3 re-initialize ERISCs between iterations
2. **Required companion**: Ensure the branch's Phase 2 BRISC halt fix and Phase 2.5 ERISC teardown timeout are present to avoid the L1 overwrite race
3. **Investigation**: Add health probes before/after AllGather to confirm device 4 ERISC zeroing is from AllGather teardown
4. **Long-term**: Consider whether `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART` should exist at all, or if quiesce restart should always run for multi-device configurations
