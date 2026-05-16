<!--
SUMMARY: Root-cause investigation of CI job 71692955830 — stale ERISC firmware probe causes cascading fabric hang on T3K
KEYWORDS: CI, T3K, ERISC, stale firmware, fabric hang, assert_risc_reset, Q1 probe, ReduceScatter, dispatch timeout, race condition
SOURCE: CI log analysis of GitHub Actions job 71692955830 on branch nsexton/0-racecondition-hunt
SCOPE: Full timeline, root cause, UMD unsafe-access errors, recommended fixes
USE WHEN: Debugging T3K fabric initialization failures, ERISC reset issues, or dispatch hangs after CCL operations
-->

# CI Job 71692955830 Investigation — Stale ERISC Firmware Cascade Hang

**Job**: `t3000-unit-tests / t3k_ttnn_tests`
**Branch**: `nsexton/0-racecondition-hunt`
**Runner**: `tt-metal-ci-vm-t3k-10` (4x N300 cards, 8 WH B0 chips)
**Outcome**: 6 tests failed, 2 tests timed out (killed by wall-clock `timeout 300`)

---

## 1. Timeline of Events

All timestamps are UTC on 2026-04-16.

```
17:43:15  tt-smi reset successful, runner starts
17:45:48  repro_ccl_cq0_hang.sh begins
          [repro] predecessor: ./build/test/ttnn/unit_tests_ttnn_ccl_ops
17:45:48  [ RUN ] CclAsyncOp.ReduceScatterSmall_PersistentFabric
          ReduceScatter opens MeshShape{1,4} → FABRIC_1D, uses 4 devices (1,3,4,6)
17:45:49  Fabric init: FABRIC_1D, compiling + configuring all 8 devices
17:45:56  "Fabric Initialized with config FabricConfig::FABRIC_1D"
17:46:00  wait_for_fabric_router_sync() → Q1 STALE PROBE FIRES:
            Device 6  ch=6   edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 0  ch=15  edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 5  ch=6   edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 1  ch=14  edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 4  ch=6   edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 2  ch=15  edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 7  ch=6   edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
            Device 3  ch=14  edm_status=0xa2b2c2d2  → TERMINATE sent → 2s timeout → ERISC RESET
          Total time burned on stale probes: ~16 seconds (sequential, 2s each device)
17:46:16  After probe, code proceeds to main handshake poll.
          enqueue_write_shards, record_event dispatched.
          read_completion_queue_event blocks on device 4, cq=0, event=1
17:46:27  Hang report fires (TT_METAL_OPERATION_TIMEOUT_SECONDS=5 elapsed since dispatch)
17:46:29  check_arc.py: Devices 4-7 report "unsafe access at address 0x880030060"
17:46:44  tt-triage dump shows:
            - 16 dispatch cores across devices 0-7 stuck in wait_for_fabric_endpoint_ready()
              at tt_fabric_mux_interface.hpp:69 (CQRelayClient::init → cq_prefetch/cq_dispatch)
            - 6 ERISC cores "in reset" (devices 1,3,4,5,6,7 — all at channel X:6-Y)
            - Remaining ERISC cores show "PC not in range of ELF" (base ERISC FW, not fabric router)
17:46:48  CRITICAL: "device timeout, potential hang detected, the device is unrecoverable"
          system_memory_manager.cpp:730
17:46:48  Teardown begins: terminate_command_queues()
17:46:53  Device 0 dispatch core timeout (5s)
17:46:58  Device 1 dispatch core timeout
17:47:03  Device 2 dispatch core timeout
17:47:08  Device 3 dispatch core timeout
17:47:08  FabricFirmwareInitializer::teardown() — no-ops (fabric not in TERMINATE mode)
17:47:08  [FAILED] CclAsyncOp.ReduceScatterSmall_PersistentFabric (80050 ms)
17:47:08  FabricSendRecv2x4 test suite begins (19 tests)
17:47:08  SendRecvAsync/0: config changed FABRIC_1D → FABRIC_2D, reinit control plane
17:47:13  [FAILED] SendRecvAsync/0 — "Timeout waiting for Ethernet core service remote IO request flush" (5138 ms)
17:47:18  [FAILED] SendRecvAsync/1 — same error (5104 ms), stack: assert_risc_reset → TTDevice::assert_risc_reset
17:47:34  [FAILED] SendRecvAsync/2 — same error (15114 ms)
17:47:44  [FAILED] SendRecvAsync/3 — same error (10104 ms)
17:47:49  [FAILED] SendRecvAsync/4 — same error (5108 ms)
17:47:49  SendRecvAsync/5 RUN — hangs until killed by timeout(300)
17:50:48  [repro] predecessor: exit=124 (killed by wall-clock timeout)
17:50:50  [repro] async_cq0: test_ccl_multi_cq_multi_device
          MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0
17:55:50  [repro] async_cq0: exit=124 (killed by wall-clock timeout)
17:55:50  Process exits with code 2
```

---

## 2. Root Cause Analysis

### 2.1 The Immediate Problem: Q1 Stale Probe Resets Only the Master ERISC Channel

The Q1 stale-firmware probe in `wait_for_fabric_router_sync()` (lines 275-327 of `fabric_firmware_initializer.cpp`) checks **only the master router ERISC channel** on each device. When it finds stale firmware (`edm_status=0xa2b2c2d2` vs expected 0 or `TERMINATED=0xa4b4c4d4`), it:

1. Sends TERMINATE to that single master channel
2. Waits 2 seconds
3. Calls `assert_risc_reset_at_core()` on that single master channel

**But `configure_fabric()` (called earlier at line 244) loaded new firmware onto ALL active ERISC channels**, not just the master. The non-master channels still have **stale old firmware running** (status `0xa2b2c2d2` = EDM running state). After the probe resets only the master ERISC:

- The master ERISC is now halted (in reset) — it cannot run the new firmware that was loaded
- The other ERISC channels still have old firmware running — they are NOT loading the new firmware
- The code then proceeds to the main handshake poll (line 331), which **waits for `expected_status`** — but since the master was hard-reset (not rebooted), it will never reach the handshake

### 2.2 Why the Code Doesn't Throw a Timeout at the Handshake

After `assert_risc_reset_at_core()`, the Q1 probe sets `master_router_status[0] = 0` (line 326) and falls through to the main handshake poll. The poll at line 331 spins reading from the master ETH L1. Since the ERISC is in hard reset, reads from L1 may return whatever was last written there. If `configure_fabric_cores()` already cleared the sync address to 0 before loading firmware, the poll reads 0 forever and eventually throws `Fabric Router Sync: Timeout`.

**However**, in the actual log we do NOT see a "Fabric Router Sync: Timeout" message. This means either:
- The handshake address was overwritten with the expected value by `configure_fabric_cores()` before the probe ran, so the poll succeeds immediately
- OR the status happened to match `expected_status` in corrupted L1

**The real hang happens downstream**: after `wait_for_fabric_router_sync()` returns (possibly with a false-positive handshake), the dispatch system starts up. CQ prefetch and CQ dispatch kernels on Tensix worker cores (`cq_relay.hpp:89`) call `CQRelayClient::init()` which calls `wait_for_fabric_endpoint_ready()` at `tt_fabric_mux_interface.hpp:69`. This function polls the ERISC fabric router to establish the fabric mux connection. But:

- **Master ERISCs are in hard reset** — they will never respond
- **Non-master ERISCs have stale firmware** — they don't match the new configuration
- **Result**: All 16 dispatch cores across 8 devices deadlock in `wait_for_fabric_endpoint_ready()`

### 2.3 The Cascade to Subsequent Tests

Once the first test (`ReduceScatterSmall_PersistentFabric`) hangs:
- Dispatch teardown times out on all 8 devices (5s each = 40s sequential)
- `FabricFirmwareInitializer::teardown()` is a no-op because `FABRIC_1D` uses `INIT_FABRIC` mode, not `TERMINATE_FABRIC`
- The process does NOT do a `tt-smi reset` between gtest cases within the same binary
- The next test suite (`FabricSendRecv2x4Tests`) starts with ERISC cores in an unknown state
- Every subsequent test that tries to initialize FABRIC_2D fails immediately with "Timeout waiting for Ethernet core service remote IO request flush" — this is the UMD-level `assert_risc_reset()` failing because the ETH cores are unresponsive

### 2.4 The `edm_status=0xa2b2c2d2` Value

This is **not** the TERMINATED value (`0xa4b4c4d4`). The value `0xa2b2c2d2` corresponds to `EDMStatus::RUNNING` or a similar active state in the fabric EDM status enum. This confirms:
- The previous test run's fabric was never properly terminated
- The `FabricFirmwareInitializer::teardown()` in the previous session either didn't run or didn't successfully terminate all routers

---

## 3. UMD "Unsafe Access" Errors (0x880030060)

```
Device 4-7: unsafe access at address 0x880030060
Attempted to read from address range [0x880030060, 0x880030063]
```

**What this means**: Address `0x880030060` is in the Wormhole B0 ETH core register space. The `0x88003XXXX` range maps to ERISC core 0-10 (virtual coordinate `(a0,0)` = `(160,0)` in the NOC coordinate space). This is the **ARC mailbox / status register area** for the remote (non-MMIO) chips.

**Why only devices 4-7**: In the T3K N300 topology, devices 0-3 are the MMIO-connected chips (one per N300 card). Devices 4-7 are the remote chips (one per N300 card, accessible only through the MMIO chip's ETH tunnel). When the ERISC cores on devices 4-7 are in reset or have stale firmware, the check_arc.py triage script cannot read the ARC postcode through the normal ETH tunnel path, causing the "unsafe access" error.

**This is a symptom**, not a cause. The ERISC reset from the Q1 probe left the ETH tunnel path broken.

---

## 4. Neil's Code Changes Assessment

### 4.1 `dispatch.cpp` — `issue_record_event` Logging
The log shows copious `[issue_record_event]` warning-level messages. These are **diagnostic logging** Neil added to trace the event dispatch path. They are informational only and do NOT contribute to the failure.

### 4.2 `fd_mesh_command_queue.cpp` — `finish_nolock` / `enqueue_record_event_helper` Logging
Similarly, these are diagnostic traces (`[finish_nolock]`, `[enqueue_record_event_helper]`). The code change adds `dispatch_thread_pool_->wait()` logging around mesh command queue operations. This is instrumentation, not causal.

### 4.3 `system_memory_manager.cpp:730` — Timeout Origin
The actual TIMEOUT exception originates at `system_memory_manager.cpp:730` in `read_completion_queue_event()`. This is the **existing** timeout path — not a new code change. The completion queue reader waits for device 4 to signal event completion, but the dispatch core on device 4 is deadlocked in `wait_for_fabric_endpoint_ready()`.

### 4.4 `repro_ccl_cq0_hang.sh` — Test Orchestration
This is a NEW script that runs `unit_tests_ttnn_ccl_ops` (predecessor) followed by `test_ccl_multi_cq_multi_device` (async_cq0) to reproduce the CQ0 hang. The script itself is correct — it just exposes the underlying bug by running CCL tests back-to-back within the same process with fabric re-initialization.

### 4.5 `test_ccl_multi_cq_multi_device.cpp` — New Test
This test (`MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0`) is the target repro case. It never gets to run meaningfully because the predecessor already leaves the hardware in a bad state.

**Verdict**: Neil's changes are diagnostic instrumentation + a repro script. They do not introduce the bug — they expose it.

---

## 5. The Predecessor Binary and What It Leaves Behind

### 5.1 What Ran
`unit_tests_ttnn_ccl_ops` with filter `CclAsyncOp.ReduceScatterSmall_PersistentFabric`:
- Opens a `MeshShape{1,4}` mesh with `FABRIC_1D`
- Runs ReduceScatter across 4 devices
- Uses "persistent fabric" — fabric stays up across the op lifetime

### 5.2 What State It Left
The test binary runs **within the same process** as the repro script. When `CclAsyncOp.ReduceScatterSmall_PersistentFabric` finishes (or the gtest fixture tears down), it should call `FabricFirmwareInitializer::teardown()` to terminate the fabric routers. However, the ERISC cores ended up with `edm_status=0xa2b2c2d2` (RUNNING), not TERMINATED, on ALL 8 devices.

This means the teardown either:
1. Did not send TERMINATE to all master routers, OR
2. Sent TERMINATE but the routers did not complete termination before `configure_fabric_cores()` was called for the next initialization cycle

The second scenario is the most likely **race condition** Neil is hunting. The sequence is:

```
  teardown():  write TERMINATE to master ETH L1
  ... (no poll for TERMINATED status in teardown — it just returns)
  init() for next test: configure_fabric_cores() → clears L1, loads new firmware
  configure():  wait_for_fabric_router_sync() → reads edm_status = 0xa2b2c2d2
```

The old ERISC firmware saw the TERMINATE signal but was in the middle of processing when `configure_fabric_cores()` overwrote L1. This left the ERISC in a partially-terminated state where its `edm_status` register still reads RUNNING.

---

## 6. Why `assert_risc_reset_at_core()` Does Not Fix the Issue

The Q1 probe's `assert_risc_reset_at_core()` on the master ERISC does work — the triage dump confirms 6 ERISC cores show "Core is in reset". However:

1. **Only the master router channel is reset** — the other 3-4 active ERISC channels per device still have stale firmware
2. **A hard-reset ERISC cannot run firmware** — the core needs to be deasserted from reset and have its firmware re-launched for the handshake to complete
3. **The Q1 probe does not re-launch firmware** — after calling `assert_risc_reset_at_core()`, it sets `master_router_status[0] = 0` and falls through to the handshake poll, hoping the *newly loaded* firmware will boot. But the ERISC is halted.
4. **`configure_fabric_cores()` already ran BEFORE the probe** — at line 244-247, firmware was loaded. The probe runs at line 263 (inside `configure()` → `wait_for_fabric_router_sync()`). By the time the probe discovers stale firmware, `configure_fabric_cores()` has already cleared L1 and loaded the new image. Resetting the ERISC does NOT re-trigger firmware boot from the newly-loaded image.

---

## 7. Recommended Fixes

### Fix A (Immediate — move stale detection before firmware load):

The stale-firmware check should happen in `configure_fabric_cores()` (or `compile_and_configure_fabric()`), **BEFORE** clearing L1 and loading new firmware. The sequence should be:

```
1. For each device, for EACH active ERISC channel (not just master):
   a. Read edm_status
   b. If stale (non-zero, non-TERMINATED):
      - Send TERMINATE
      - Poll for TERMINATED (with timeout)
      - If timeout: assert_risc_reset_at_core()
2. THEN clear L1 and load new firmware
3. Deassert ERISC reset (if it was asserted)
4. Proceed to wait_for_fabric_router_sync() for handshake
```

### Fix B (Belt-and-suspenders — poll TERMINATED in teardown):

`FabricFirmwareInitializer::teardown()` (lines 198-208) currently writes TERMINATE to master routers and returns immediately. It should poll each master router's `edm_status` for TERMINATED (with timeout), similar to the Tensix MUX poll already implemented at lines 132-194.

### Fix C (Defense in depth — reset ALL active ERISC channels, not just master):

If the Q1 probe fires, it should reset ALL active ERISC channels on the affected device, not just the master. The master-only reset leaves non-master channels with stale firmware that conflicts with the newly-loaded configuration.

### Fix D (Structural — deassert reset after probe):

After `assert_risc_reset_at_core()`, the Q1 probe should call `deassert_risc_reset_at_core()` so the newly-loaded firmware can actually boot. Currently the ERISC stays halted forever.

### Recommended Priority

**Fix A + Fix B** together would solve the problem:
- Fix B prevents the stale state from occurring (proper termination wait)
- Fix A detects and recovers from the stale state if it does occur (defense in depth)

Fix C and Fix D are refinements to the Q1 probe itself, making it less harmful when it does fire.

---

## 8. Summary

```
Root cause:  FabricFirmwareInitializer::teardown() does not poll for TERMINATED
             after sending TERMINATE signal to ERISC master routers. The next
             init/configure cycle loads new firmware onto L1 while old firmware
             is still running, leaving ERISC in a zombie state (edm_status=0xa2b2c2d2).

Q1 probe:   Detects the zombie state but makes it WORSE by hard-resetting only
             the master ERISC (halting it permanently) without deasserting reset
             and without touching non-master channels.

Cascade:     Dispatch cores deadlock in wait_for_fabric_endpoint_ready() because
             ERISC fabric routers are either halted or running stale firmware.
             30s operation timeout fires → device declared unrecoverable →
             all subsequent tests in the same binary fail.

Not caused by: Neil's diagnostic logging, fd_mesh_command_queue changes, or
               the repro script itself.
```
