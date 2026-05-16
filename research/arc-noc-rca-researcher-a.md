<!-- SUMMARY: Deep analysis of NOC address 0x880030060 (ARC reset scratch), close_finish race condition, device 4 zero-state root cause, and non-MMIO dispatch path failure in T3K quiesce hang
KEYWORDS: 0x880030060, ARC_RESET_SCRATCH, NOC address, Wormhole, quiesce, ERISC, BRISC, L1 overwrite, close_finish, race condition, T3K hang, fabric, device 4, non-MMIO, dispatch relay
SOURCE: local codebase analysis (nsexton-0-racecondition-hunt branch + Wormhole NOC headers)
SCOPE: NOC address decoding, close_finish/TERMINATED race, device 4 zero-state, non-MMIO dispatch failure, ERISC reset limitations
USE WHEN: investigating T3K hangs after AllGather/quiesce, NOC address decoding, ARC scratch register corruption, non-MMIO device dispatch failures -->

# ARC NOC RCA - Researcher A Findings

## 1. Decoding 0x880030060 in the Wormhole NOC Address Space

### NOC Address Format (Wormhole)

Wormhole uses a 48-bit NOC address:
- **Bits [47:42]**: 6-bit Y coordinate
- **Bits [41:36]**: 6-bit X coordinate
- **Bits [35:0]**: 36-bit local address

Defined in `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h`:
```c
// Line 233-234
#define NOC_ADDR_LOCAL_BITS 36
#define NOC_ADDR_NODE_ID_BITS 6

// Line 256-258
#define NOC_XY_ADDR(x, y, addr) \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | ((uint64_t)(addr)))
```

### Decoding 0x880030060

`0x880030060` fits in 36 bits (the local address portion), meaning **X=0, Y=0** (coordinate bits are all zero). The target is the NOC node at (0,0), which on Wormhole is the ARC/PCIe subsystem.

**Bit 35 is SET** (`0x800000000`). On Wormhole, bit 35 activates the **NOC-to-AXI bridge** (PCIe/ARC address space), as shown by:
```c
// noc_parameters.h:276-277
#define NOC_XY_PCIE_ENCODING(x, y) \
    ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET)) | 0x800000000)
```

Decomposition:
```
0x880030060 = 0x800000000 (bit 35: AXI bridge flag)
            | 0x080000000 (ARC memory region base)
            | 0x000030060 (ARC_RESET_SCRATCH0 offset)
```

### What Is ARC_RESET_SCRATCH0?

ARC_RESET_SCRATCH0 is the ARC firmware's **postcode/scratch register** at AXI offset 0x80030060. The ARC processor writes status codes here during boot and operation. Host diagnostics read it to determine firmware state:
- `tools/triage/device_info.py:43` reads it for postcode diagnostics
- `tools/triage/check_arc.py:109` uses it for ARC health checks
- `device.cpp:581` names it explicitly: `"writes to ARC_RESET_SCRATCH_ADDR (0x880030060)"`

**Writing garbage to this register corrupts ARC firmware state**, potentially breaking ARC's ability to manage ERISC initialization, ETH link configuration, and device reset sequences.

## 2. Conditions Under Which BRISC Writes to 0x880030060

The address is **NEVER** written intentionally by Tensix firmware. It appears as a side effect of **L1 memory corruption while BRISC is still executing**.

### The Corruption Mechanism

From `device.cpp:576-581`:
```cpp
// The CCL MUX kernel writes TERMINATED *before* close_finish() completes (close_finish
// spins on worker_teardown_addr waiting for the ERISC ACK). If we proceed directly
// to Phase 3's ConfigureDeviceWithProgram, the still-running BRISC executes whatever
// instructions now reside in its overwritten L1, generating invalid NOC traffic -- including
// writes to ARC_RESET_SCRATCH_ADDR (0x880030060) -- that corrupt ERISC or ARC state.
```

The sequence:
1. Phase 2 polls MUX status, sees TERMINATED
2. Phase 3 immediately calls `ConfigureDeviceWithProgram()` which overwrites L1
3. BRISC is **still running** `close_finish()` -- it's spinning on `worker_teardown_addr`
4. BRISC's instruction fetch now reads the new L1 content as machine code
5. The corrupted instruction stream generates random NOC write transactions
6. Some of these writes happen to target 0x880030060

The address is not deterministic -- it depends on whatever data the new firmware image contains at the instruction pointer's current offset. But 0x880030060 was specifically observed in CI failures, and corrupting ARC scratch is especially damaging.

### Two Affected Code Paths

**Path A: Legacy MUX kernel** (`tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:282-303`):
```cpp
fabric_connection.close_start();       // line 295
noc_async_write_barrier();             // line 296
noc_async_atomic_barrier();            // line 297
status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;  // line 299 -- BEFORE close_finish!
fabric_connection.close_finish();      // line 301 -- BRISC STILL RUNNING HERE
```

**Path B: Router MUX extension** (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp:390-396`):
```cpp
fabric_connection.close();  // line 392 -- sequential close_start + close_finish
noc_async_write_barrier();
noc_async_atomic_barrier();
status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;  // line 396 -- AFTER close
```

Path B is safe (writes TERMINATED after close completes). Path A has the race.

## 3. The close_finish() / TERMINATED Race Condition

### close_finish() Implementation

From `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:485-504`:
```cpp
void close_finish() {
    constexpr uint32_t kCloseFinishMaxIter = 5'000'000;
    for (uint32_t i = 0; i < kCloseFinishMaxIter; ++i) {
        if (*this->worker_teardown_addr == 1) { break; }
        invalidate_l1_cache();
    }
    noc_async_write_barrier(get_fabric_worker_noc());
    *(this->worker_teardown_addr) = 0;
}
```

`close_finish()` spins up to 5 million iterations waiting for the ERISC to acknowledge teardown by writing `1` to `worker_teardown_addr`. This is a **long-running poll**.

### The Race Window

```
Timeline for Legacy MUX (Path A):

T0: close_start() sends teardown request to ERISC
T1: noc_async_write_barrier() completes
T2: TERMINATED written to status_ptr        <-- Host Phase 2 sees this
T3: close_finish() begins polling            <-- BRISC still running!
T4: ... BRISC spins for up to 5M iterations ...
T5: ERISC ACK arrives, close_finish() completes

Host Phase 2 (between T2 and T5):
  - Reads TERMINATED from status_ptr
  - Proceeds to Phase 3
  - Phase 3 overwrites L1
  - BRISC (at T3-T4) is now executing corrupted instructions
```

The fix on the branch adds `assert_risc_reset_at_core()` in Phase 2 after seeing TERMINATED, which hard-halts the BRISC before L1 is overwritten.

## 4. Why Device 4 Shows All-Zero ERISC Router State After Quiesce

### The Device Map

```
mesh[0,0] = chip 0 (N300, MMIO, left)
mesh[0,1] = chip 1 (N300, MMIO, right)
mesh[0,2] = chip 4 (N300, non-MMIO, far left)   <-- ALL ZEROS
mesh[0,3] = chip 5 (N300, non-MMIO, far right)
```

### Root Cause Chain

**Step 1: `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` skips everything.**

From `device.cpp:429-438` (the top of `quiesce_and_restart_fabric_workers()`):
```cpp
if (tt::parse_env("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", false)) {
    log_debug(LogMetal, "Skipping quiesce and restart fabric workers");
    return;  // ALL phases skipped
}
```

The fixture sets this env var in `test_ccl_multi_cq_multi_device.cpp:73`:
```cpp
setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", 0);
```

This means **Phases 1, 2, 2.5, 3, and 4 are ALL skipped** for every device.

**Step 2: AllGather teardown terminates ERISC routers.**

When AllGather completes, its internal teardown sends TERMINATE signals to ERISC fabric channels. This is part of the CCL operation cleanup, independent of quiesce_and_restart.

**Step 3: Without restart, ERISCs stay dead.**

Since Phase 2.5 (ERISC termination acknowledgment) and Phase 3 (re-configure and re-launch) are skipped, the ERISCs on device 4 that were terminated by AllGather's cleanup are never restarted. Their L1 state reads as all zeros because:
- The ERISC firmware has completed its cleanup and halted
- L1 memory retains the post-termination state (zeroed status registers)
- No new firmware is loaded to reinitialize them

**Step 4: Why device 4 specifically?**

Device 4 is the **non-MMIO far N300**, accessed exclusively through the ETH fabric. The other devices show non-zero states because:
- Devices 0 and 1 (MMIO) have direct host access -- their ERISCs may be independently initialized
- Device 5's ERISCs may have different termination timing or routing that preserves some state
- Device 4's ERISCs were likely the first to fully terminate and drain, leaving clean zeros

The probe output from CI job 72045468285:
```
Device 0: chans 0-5 -> "OTHER" states (0x3f803f80, 0x40004000, 0x40404040) -- initialized
Device 1: chans 0-5 -> similar non-zero "OTHER" states -- initialized
Device 4: ALL channels -> 0x00000000 -- ZERO STATE (never restarted after AllGather teardown)
Device 5: similar non-zero states -- initialized
```

## 5. Non-MMIO Dispatch Path and Why "Enqueue Dummy Ops" Hangs

### The Full Dispatch Path for Non-MMIO Device 4

When `ttnn::write_buffer` targets device 4, the call chain is:

1. **`ttnn::write_buffer()`** (`ttnn/core/async_runtime.cpp:13-22`)
   - Iterates device tensors, calls `copy_to_device()` for each shard

2. **`FDMeshCommandQueue::write_shard_to_device()`** (`tt_metal/distributed/fd_mesh_command_queue.cpp:585-618`)
   - Calls `buffer_dispatch::write_to_device_buffer()` for the specific device

3. **`dispatch_thread_pool_->wait()`** (`tt_metal/distributed/mesh_command_queue_base.cpp:267`)
   - Blocks until all dispatch threads complete their work

4. **Device-side dispatch** (for non-MMIO device 4):
   - Host writes commands to the MMIO device's (device 0 or 1) issue queue
   - The **prefetcher** on the MMIO device reads commands
   - **CQRelayClient** (`tt_metal/impl/dispatch/kernels/cq_relay.hpp`) relays commands via fabric
   - The relay calls `wait_for_fabric_endpoint_ready()` (line 89) and `fabric_client_connect()` (line 90)
   - These calls send data through the **fabric MUX** to **ERISC routers**
   - ERISC routers forward packets over the **ETH link** to device 4's ERISC
   - Device 4's ERISC delivers commands to its local dispatcher

### Where It Hangs

When device 4's ERISC routers are dead (all-zero state), the relay path breaks at the ERISC router hop. The CQRelayClient's `wait_for_fabric_endpoint_ready()` is an **unbounded spin**:

```cpp
// cq_relay.hpp:89
tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x, mux_y, mux_status_address, local_mux_status_address);
```

If the MUX or ERISC endpoint never becomes ready (because it was never restarted after AllGather teardown), this spin never terminates. The host-side `dispatch_thread_pool_->wait()` blocks indefinitely until `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` fires:
```
TT_THROW: TIMEOUT: device timeout, the device is unrecoverable
```

### The Event Recording Angle

The branch also adds detailed logging to `issue_record_event_commands()` (`tt_metal/impl/event/dispatch.cpp:89-123`) with paired "reserve begin" / "reserve ok" log markers. If the hang occurs during event recording rather than buffer write:
- A hang **before** "reserve ok" = host-side issue queue full (device dispatcher not draining prior commands)
- A hang **after** "reserve ok" = device-side WAIT command stuck (workers never completed, or ERISC fabric broken)

Both point to the same root cause: device 4's fabric path is broken because ERISCs were never restarted.

## 6. ERISC Reset Limitations on Wormhole

### Why ERISC Cannot Be Hard-Reset on WH

From `device.cpp:604-616`:
```cpp
// Phase 2.5 — Terminate ERISC Fabric Channels
// On Wormhole, asserting RISC reset on an ERISC core tears down the
// ETH PHY link. For non-MMIO devices (accessed via ETH fabric), this
// would sever the only communication path to the device.
```

On Wormhole architecture:
- ERISC cores manage the Ethernet PHY directly
- `assert_risc_reset_at_core()` on an ERISC halts the RISC **and tears down the ETH PHY link**
- For non-MMIO devices (like device 4), the ETH link is the **only** communication path
- Tearing down ETH PHY = losing the device entirely until full board reset

This is why Phase 2.5 uses **software termination** (send TERMINATE signal, poll for TERMINATED response) instead of hardware reset for ERISCs. The software approach:
1. Reads ERISC router sync address
2. If status == 0 or TERMINATED: skip (already clean)
3. Otherwise: write TERMINATE signal, poll for TERMINATED acknowledgment

This preserves the ETH link while stopping the ERISC firmware.

### Contrast with Tensix BRISC

Tensix BRISC cores CAN be hard-reset (`assert_risc_reset_at_core`) because:
- They don't manage any PHY or external link
- Resetting them only affects local compute, not communication fabric
- Phase 2 uses this for MUX BRISCs after seeing TERMINATED

## 7. The Bug Summary

### Causal Chain

```
1. AllGather runs on 1x4 mesh (devices 0, 1, 4, 5)
2. AllGather teardown sends TERMINATE to ERISC fabric channels
3. quiesce_devices() is called
4. quiesce_and_restart_fabric_workers() returns IMMEDIATELY due to env var
5. ERISC routers on device 4 remain dead (never restarted)
6. "Enqueue dummy ops" calls ttnn::write_buffer targeting device 4
7. Dispatch relay tries to send commands via dead ERISC fabric
8. CQRelayClient spins indefinitely waiting for fabric endpoint
9. 5-second timeout fires: "device timeout, the device is unrecoverable"
```

### Why the Env Var Exists

`TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` was likely added as a performance optimization or workaround for other quiesce-related issues. The test fixture (`MultiCQFabricMeshDevice2x4Fixture`) sets it because:
- Without it, the quiesce restart might trigger the BRISC/ERISC L1 corruption race (the very bug being investigated)
- The fixture also sets `TT_METAL_FABRIC_HEALTH_PROBE=1` to diagnose the state

### The Fix Path

The branch's changes to `quiesce_and_restart_fabric_workers()` aim to make the full restart safe:
1. **Phase 2**: After TERMINATED, hard-halt BRISC with `assert_risc_reset_at_core()` before L1 overwrite
2. **Phase 2.5**: Software-terminate ERISCs with proper polling before L1 overwrite
3. **Phase 3**: Only then overwrite L1 and re-launch

With these fixes, removing the `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART` env var should allow quiesce+restart to work correctly, keeping ERISC routers alive for subsequent operations.

## 8. File Reference Index

| File | Lines | What |
|------|-------|------|
| `tt_metal/impl/device/device.cpp` | 429-438 | Env var early-exit in quiesce_and_restart |
| `tt_metal/impl/device/device.cpp` | 483-510 | Phase 1: IMMEDIATELY_TERMINATE to MUX |
| `tt_metal/impl/device/device.cpp` | 521-601 | Phase 2: Poll TERMINATED, halt BRISC |
| `tt_metal/impl/device/device.cpp` | 576-581 | **Critical comment**: 0x880030060 corruption |
| `tt_metal/impl/device/device.cpp` | 603-710 | Phase 2.5: ERISC software termination |
| `tt_metal/impl/device/device.cpp` | 712-773 | Phase 3: Re-configure and re-launch |
| `tt_metal/impl/device/device.cpp` | 785-819 | Phase 4: Wait READY_FOR_TRAFFIC |
| `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` | 282-303 | Legacy MUX: TERMINATED before close_finish (THE RACE) |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp` | 390-396 | Router MUX: TERMINATED after close (safe) |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | 485-504 | close_finish() implementation (5M iteration poll) |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | 48-62 | EDMStatus enum (TERMINATED=0xA4B4C4D4) |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` | 233-258, 276-277 | NOC address format, PCIe encoding |
| `tt_metal/impl/event/dispatch.cpp` | 42-201 | issue_record_event_commands (event dispatch) |
| `tt_metal/impl/dispatch/kernels/cq_relay.hpp` | 69-113 | CQRelayClient::init (fabric relay for non-MMIO) |
| `tt_metal/distributed/mesh_device.cpp` | 1433-1471 | quiesce_internal/quiesce_devices flow |
| `tt_metal/distributed/mesh_command_queue_base.cpp` | 234-267 | enqueue_write_shards_nolock (hang location) |
| `tt_metal/distributed/fd_mesh_command_queue.cpp` | 585-618 | write_shard_to_device |
| `ttnn/core/async_runtime.cpp` | 13-22 | ttnn::write_buffer entry point |
| `tests/.../test_ccl_multi_cq_multi_device.cpp` | 73, 83-95, 284-306 | Fixture setup, bug comment, hang section |

All paths are relative to `/workspace/group/worktrees/nsexton-0-racecondition-hunt/`.
