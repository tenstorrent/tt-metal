<!--
SUMMARY: Comprehensive code audit of nsexton/0-racecondition-hunt branch (iter 27) — 210+ commits fixing T3K fabric race conditions, dispatch deadlocks, and quiesce hangs
KEYWORDS: audit, race-condition, fabric, ETH, ERISC, quiesce, teardown, dispatch, T3K, dead-relay, system_memory_manager, fetch_queue
SOURCE: git diff origin/main...HEAD on worktree /workspace/group/worktrees/nsexton-0-racecondition-hunt + AI-JOURNAL.md
SCOPE: All changes vs main: fabric_firmware_initializer, device.cpp, risc_firmware_initializer, system_memory_manager, ccl_common, device_operation, llrt, UMD submodule (5 commits), test files
USE WHEN: Reviewing this branch for merge readiness, understanding fix interactions, or planning follow-up work
-->

# Branch Audit: nsexton/0-racecondition-hunt (Iteration 27)

**Date**: 2026-04-26
**Branch**: `nsexton/0-racecondition-hunt`
**Commits**: 210+ vs `origin/main`
**Lines changed**: ~13,784 additions, ~393 deletions across 87 files
**Auditor**: BrAIn (Opus 4.6)

---

## Executive Summary

This branch addresses a cascade of interconnected race conditions on T3K (8-device Wormhole B0) systems where ERISC L1 corruption from killed predecessor processes causes chain-reaction hangs across fabric init, quiesce, teardown, and dispatch. The fixes are organized into roughly 40 labeled FIXes (A through AQ-2) plus several unlabeled improvements.

**Verdict**: The core logic is sound. The fixes demonstrate deep understanding of the hardware behavior. However, the branch has accumulated significant debug instrumentation that must be cleaned before merge, and several architectural concerns need resolution.

---

## Category 1: Fabric Init Dead-Relay Cascade (FIX A through FIX H)

### FIX A: UMD ETH Relay Queue Drain

**What**: After `terminate_stale_erisc_routers` probe reads time out, the UMD relay queue has accumulated stuck commands. Drains the queue before returning so the next session starts clean.

**Correctness**: GOOD. The drain is best-effort with proper try/catch. The comment explicitly notes the next session's `relay_broken` guard provides defense-in-depth.

**Happy path cost**: ZERO. Drain only runs when `relay_dead_devices` is non-empty, which only happens when relay probe reads actually timed out.

**Action**: None required.

### FIX B: Poll ALL Active ETH Channels in Teardown

**What**: Original code only polled a single master channel; this polls all active channels.

**Correctness**: GOOD. Simple loop extension.

**Happy path cost**: Marginal — adds N channel polls instead of 1, but N is typically 4-8 and each poll is a fast L1 read.

**Action**: None required.

### FIX C: Skip write_launch_msg / WriteRuntimeArgs / ConfigureDeviceWithProgram for Dead ETH Channels

**What**: New overloads of `WriteRuntimeArgsToDevice()` and `ConfigureDeviceWithProgram()` in `tt_metal.cpp` accept a `logical_cores_to_skip` set. Device::configure_fabric() uses this to skip dead channels.

**Correctness**: GOOD. The overloads delegate to the original function when skip set is empty. The ETH-only guard (`core_type == CoreType::ETH`) is correct — dead-relay only affects ETH cores.

**Happy path cost**: ZERO. `all_dead_channels` is empty on happy path, so the overloads delegate directly to originals.

**Concern**: The `tt_metal.cpp` overloads duplicate ~100 lines of the original `WriteRuntimeArgsToDevice` and `ConfigureDeviceWithProgram`. If the originals change, these copies can drift.

**Action** (nice-to-have): Refactor to add the skip set as an optional parameter to the originals rather than duplicating the full function bodies.

### FIX E/E2/E3: Dead-Relay Device Tracking and Static Config Population

**What**: FIX E skips dispatch kernel init for dead-relay non-MMIO devices. FIX E2 marks devices as dead-relay when any probe read times out (restricted to non-MMIO devices per later fix `fde0e6c`). FIX E3 populates `static_config` for dead-relay devices to prevent `bad_optional_access`.

**Correctness**: GOOD. The restriction of FIX E2 to non-MMIO devices (commit `fde0e6c`) is critical — MMIO devices have PCIe-direct paths and should never be marked dead-relay.

**Happy path cost**: ZERO. Guard checks `dead_relay_devices_.count()` which is empty on happy path.

**Action**: None required.

### FIX G: Skip Dead-Relay Devices in wait_for_fabric_router_sync and verify_all_fabric_channels_healthy

**What**: Both functions now skip devices in `dead_relay_devices_`.

**Correctness**: GOOD. Dead-relay devices can never complete the fabric handshake, so polling them is guaranteed to timeout.

**Happy path cost**: ZERO.

**Action**: None required.

### FIX H: Short-Circuit Relay Probe for Subsequent Non-MMIO Devices

**What**: Once `relay_broken=true` on one non-MMIO device, all subsequent non-MMIO devices skip `terminate_stale_erisc_routers` and go directly to degraded mode.

**Correctness**: GOOD. On T3K, all non-MMIO devices share the same MMIO relay path. If one relay is dead, all are dead.

**Concern**: The assumption "all non-MMIO devices share one relay" holds for T3K but may not hold for Galaxy (multiple MMIO chips). The code uses `any_relay_broken` as a global flag, not per-MMIO-host.

**Action** (should-fix for Galaxy): Make `relay_broken` per-MMIO-host rather than global. For T3K-only merge, this is acceptable.

---

## Category 2: Transitive Dead-Peer Propagation (FIX I, FIX I2, mmio_dead_peer_devices_)

### FIX I: MMIO Dead-Peer Device Detection

**What**: After configure, walks `eth_connections` to find MMIO devices whose master ETH channel connects to a dead-relay non-MMIO device. Stores in `mmio_dead_peer_devices_`.

**Correctness**: GOOD. Critical distinction: `mmio_dead_peer_devices_` is NOT added to `dead_relay_devices_`, so MMIO dispatch (PCIe path) proceeds normally.

### Transitive Propagation (`propagate_dead_mmio_peers` in risc_firmware_initializer.cpp)

**What**: Fixed-point algorithm: if MMIO device M's ETH peer is already in `mmio_dead_peer_devices_`, add M too. Repeats until stable.

**Correctness**: GOOD. The algorithm terminates because the MMIO ID set is finite and we only add, never remove. The break on first dead peer per device and re-scan is correct.

**Concern**: The fixed-point loop runs in O(|MMIO|^2) worst case but MMIO count is typically 1-4, so this is negligible.

**Action**: None required.

### FIX I2: Skip Phase 5 for mmio_dead_peer_devices

**What**: In `wait_for_fabric_workers_ready`, skips the handshake poll and health check for MMIO devices whose peers are dead.

**Correctness**: GOOD. Firmware was loaded but the peer will never handshake.

**Happy path cost**: ZERO.

---

## Category 3: Quiesce Phase Ordering and ETH Launch (FIX AE, AF, N, O, P, AD)

### FIX AE: Three-Pass Deferred ETH Launch

**What**: Phase 3 runs configure_fabric_cores + runtime args but skips ETH `write_launch_msg`. The mesh-level caller then runs `launch_eth_cores_for_quiesce()` in the correct order (MMIO before non-MMIO, non-MMIO sequentially).

**Correctness**: GOOD. This is the most architecturally significant fix. Simultaneous ETH launches caused deadlocks because sender and receiver ERISCs both tried to handshake before the peer was running. The three-pass approach ensures the correct launch ordering.

**Happy path cost**: MINIMAL. The deferred launch adds one extra pass over devices but no additional L1 reads or writes — just reorders existing write_launch_msg calls.

**Concern**: The `pending_eth_launch_` / `pending_quiesce_newly_dead_eth_chans_` / `pending_phase25_force_reset_chans_` state fields in device_impl.hpp are NOT thread-safe (noted in comments). Currently safe because mesh-level serialization guarantees ordering. Future parallelization would break this.

**Action** (nice-to-have): Document the serialization invariant in a single location, not scattered across field comments.

### FIX AF: Poll for STARTED Between Non-MMIO Launches

**What**: After launching ETH on one non-MMIO device, polls until all channels show non-zero `edm_status` (STARTED or beyond) before launching the next device.

**Correctness**: GOOD. Prevents the "simultaneous handshake deadlock" where both sides enter the handshake loop before their peer is listening.

**Happy path cost**: MINIMAL. One extra L1 read per channel per device (fast PCIe read). Timeout is 500ms but should complete in <50ms.

### FIX N: Skip Soft Reset for Non-MMIO ETH Channels in Quiesce Phase 3

**What**: For non-MMIO devices, passes all active ETH channels as `skip_soft_reset_channels` to `configure_fabric_cores`.

**Correctness**: GOOD. Non-MMIO soft reset kills the ETH relay, cascading into a full hang. The fix correctly identifies that TERMINATED firmware already polls for launch messages — no soft reset needed.

**Happy path cost**: ZERO. Only fires for non-MMIO devices during quiesce.

### FIX AD: Skip MMIO ETH Soft-Reset in Quiesce Phase 3

**What**: Skips soft-resetting MMIO ETH channels during quiesce to preserve the relay.

**Correctness**: GOOD. Same logic as FIX N but for MMIO devices — soft-resetting the relay ERISC kills the relay path.

**Action**: None required.

---

## Category 4: Teardown Safety (FIX AC, AI, AJ, AK, AL, AB)

### FIX AC: Two-Phase ETH Reset in risc_firmware_initializer.cpp Teardown

**What**: Complete rewrite of teardown flow:
1. Detect relay_broken and teardown_timed_out
2. Hard-reset MMIO ETH via PCIe + 500ms sleep (if relay broken)
3. assert_cores/l1_barrier with skip for non-MMIO when relay broken
4. Skip set_internal_routing_info when relay broken
5. Hard-reset MMIO ETH when only teardown timed out

**Correctness**: GOOD. The ordering is critical and correct: MMIO ERISCs must be reset BEFORE assert_cores to avoid 5s timeouts per non-MMIO device. The 500ms sleep gives ERISCs time to reboot from ROM.

**Happy path cost**: ZERO on happy path. Steps 2-5 only fire when `relay_broken_non_mmio` or `any_teardown_timed_out` is true. The happy path runs the original assert_cores + set_internal_routing sequence unmodified.

**Concern**: The `500ms` and `200ms` sleeps are empirically determined. If hardware takes longer to reboot, these could be insufficient.

**Action** (nice-to-have): Replace fixed sleeps with polled readiness checks (read ERISC status after reset, wait for base-UMD sentinel).

### FIX AI: Assert + Deassert ALL RISCs in Teardown Force-Reset

**What**: After `assert_risc_reset_at_core(ALL)`, immediately `deassert_risc_reset_at_core(ALL)` so NCRISC (ETH PHY) and BRISC come back up.

**Correctness**: CRITICAL FIX. Without the deassert, NCRISC stays in hardware reset, ETH PHY goes down, and the next session can't discover the device at all. This was the root cause of "Physical chip id not found for eth coord" crashes.

**Happy path cost**: ZERO. Force-reset only fires when ETH channels timed out during teardown.

### FIX AI-2: Phase 2.5 Force-Reset Tracking and Deassert in Phase 3

**What**: Tracks which channels Phase 2.5 force-halted in `pending_phase25_force_reset_chans_`. Phase 3 and `launch_eth_cores_for_quiesce()` deassert these after `write_launch_msg`.

**Correctness**: GOOD. Without deassert, the ERISC can't pick up the launch message (it's in hardware reset).

### FIX AJ: Skip l1_barrier for Relay-Dead Non-MMIO Devices in Teardown

**What**: During the teardown force-reset drain loop, skips `l1_barrier` for devices whose relay path was confirmed dead during the force-reset.

**Correctness**: GOOD. `l1_barrier` on a non-MMIO device with dead relay BLOCKS INDEFINITELY.

**Concern**: `relay_dead_devices` (local) is populated inside the force-reset loop's anonymous block. The original scoping bug (variable declared inside the block, used outside) was fixed in a subsequent commit `1ab3b3239b8`.

### FIX AK: Transitive Dead-Relay Guard on l1_barrier in Teardown

**What**: Wraps the l1_barrier drain loop with a check: if ANY device in `relay_dead_devices` has an MMIO host shared with the current device, skip the barrier.

**Correctness**: GOOD. Correctly handles the T3K case where all non-MMIO devices share one MMIO relay.

**Happy path cost**: ZERO. `relay_dead_devices` is empty on happy path.

### FIX AL: Read hw_fence at reset() Instead of Hardcoding Sentinel

**What**: `SystemMemoryManager::reset()` reads `PREFETCH_Q_RD_PTR_ADDR` from hardware instead of assuming the sentinel value.

**Correctness**: CRITICAL FIX. When dispatch kernel was NOT reloaded (quiesce didn't fully reinit), hardcoding the sentinel caused `count_consumed(sentinel, stale_value)` to credit fictitious consumed slots, overwriting in-flight entries.

**Happy path cost**: ONE extra PCIe read per CQ per reset. Negligible.

---

## Category 5: Dispatch Fetch Queue Deadlock (count_consumed, in-flight counter)

### Fetch Queue Reserve Back Rewrite

**What**: Complete rewrite of `fetch_queue_reserve_back()` replacing address-comparison-based full detection with an explicit `prefetch_q_in_flight` counter.

**Correctness**: GOOD. The core insight is sound: comparing `ptrs` to `fences` cannot distinguish depth=0 from depth=N (aliasing at mod-N boundary). The in-flight counter eliminates the aliasing.

**Full-wrap aliasing disambiguation**: When `fence == old_fences` and `in_flight >= N`, reads the next slot to check if firmware cleared it. Includes a TOCTOU guard (re-reads fence after slot check).

**Happy path cost**: FAST PATH is faster than before — when `in_flight < N`, returns immediately with zero PCIe reads. The original code always did a PCIe read to check the fence.

**Concerns**:
1. `count_consumed()` sentinel transition path: `(new_fences - prefetch_q_base) / entry_size + 1` — the `+1` assumes the sentinel is exactly one entry past the last valid slot. This is correct given `prefetch_q_limit = base + N * entry_size` and firmware writes `base + (consumed_slot_index) * entry_size`.
2. Underflow guard: `(consumed <= in_flight) ? in_flight - consumed : 0` — clamping to 0 is correct for robustness but masks potential bugs. The comment should note when underflow indicates a logic error.

**Action** (should-fix): Add a `TT_ASSERT(consumed <= in_flight, ...)` in debug builds alongside the clamp.

---

## Category 6: Quiesce Phase 5b Health Check (FIX W, V, U, AK, AM)

### FIX W: Extend All-Dead Clean-Return to MMIO Devices

**What**: When all Phase 5b channels show 0x0 (dead), return cleanly for BOTH MMIO and non-MMIO devices instead of throwing.

**Correctness**: GOOD. An MMIO device whose peers are all dead will also show all-0x0. Throwing here triggers a teardown cascade with rescue_stuck_dispatch_cores timeouts.

### FIX AK: Partial-Mesh Quiesce Non-Fatal

**What**: If all truly-unhealthy channels are at STARTED/REMOTE_HANDSHAKE_COMPLETE/LOCAL_HANDSHAKE_COMPLETE (handshake-incomplete states), return cleanly instead of throwing.

**Correctness**: GOOD. In partial-mesh quiesce (e.g., 1x4 on T3K), out-of-mesh peers run base-UMD firmware and don't respond to EDM handshake. These channels will never reach READY_FOR_TRAFFIC.

**Key nuance**: Does NOT set `fabric_relay_path_broken_` — the relay IS working; only the fabric handshake is incomplete.

### FIX AM: Skip verify_all_fabric_channels_healthy When Dead-Relay Devices Exist

**What**: After `wait_for_fabric_router_sync`, skips the full health check if `dead_relay_devices_` is non-empty.

**Correctness**: GOOD. The health check would throw on devices whose router sync was intentionally skipped (FIX G/I).

**Happy path cost**: ZERO.

---

## Category 7: Buffer Reuse Race (device_operation.hpp, MeshBuffer pending events)

### Completion Event Tracking

**What**: After each `EnqueueMeshWorkload`, records a host-visible completion event. Buffers referenced by the op register this event. Before buffer deallocation, `wait_for_pending_events` waits on all registered events.

**Correctness**: GOOD design. The trace-capture guard (skip during trace because buffer lifetime is managed by trace infrastructure) is correct.

**Happy path cost**: ONE extra `enqueue_record_event_to_host` per op that touches device tensors. Gated by `any_tensor_needs_completion_tracking` to skip ops that don't need it.

**Concern**: The `log_info` calls in the hot path (`[device_operation] EnqueueMeshWorkload start/done`, `enqueue_record_event_to_host start/done`) should be `log_trace` for production.

**Action** (should-fix before merge): Demote `log_info` to `log_trace` in `device_operation.hpp` hot-path logging.

---

## Category 8: UMD Submodule Changes (5 commits)

### FIX AQ: Catch Relay Timeout in Topology Discovery

**What**: `topology_discovery.cpp` catches relay timeouts during remote device discovery and marks devices as discovered-but-skipped.

**Correctness**: GOOD. Prevents cascade crash when a remote device is unreachable.

### FIX AQ-2: Skip ETH Connections to Skipped Remote Devices

**What**: Guards `ethernet_connections.push_back()` against adding connections to devices that were skipped in AQ.

**Correctness**: GOOD. Prevents `asic_id_to_chip_id.at()` throwing `out_of_range` later.

### FIX W (UMD): Skip Dead ETH Channels in Topology Discovery

**What**: Dead ETH channels (no link partner) are excluded from the discovery loop.

### FIX X: Skip ETH Training Wait for Force-Reset Cores + Heartbeat Race Fix

**What**: Two UMD commits: (1) force-reset cores skip the ETH training wait entirely, (2) heartbeat check moved inside the training loop after 2000ms of IN_PROGRESS to avoid false skip on fresh boot.

**Correctness**: The v2 fix (heartbeat check inside loop after 2000ms) is correct. The v1 (pre-loop check) was buggy — checking heartbeat before the training loop could race with legitimate boot.

---

## Category 9: CCL Changes (ccl_common.cpp/hpp)

### Templatized fabric_mux_connection_rt_args

**What**: Refactored `fabric_mux_connection_rt_args` into a template `fabric_mux_connection_rt_args_impl` that works with both `Program` and `ProgramDescriptor`.

**Correctness**: GOOD. Clean generic refactor. The template dispatches to `CreateSemaphore` for `Program` and to `find_available_semaphore_id` for `ProgramDescriptor`.

**Happy path cost**: ZERO. Compile-time dispatch via `if constexpr`.

---

## Cross-Cutting Concerns

### 1. Debug Instrumentation Left Behind (BLOCKER)

**159 new `log_warning` calls** across the branch. Many are legitimate degradation warnings, but several are clearly debug instrumentation that should be `log_trace` or removed:

- `tt_metal/impl/event/dispatch.cpp`: 4 `log_warning` calls for `[issue_record_event]` diagnostics — these fire on EVERY event recording. **Must be demoted to log_trace or removed.**
- `device_operation.hpp`: `log_info` calls on every `EnqueueMeshWorkload` — **must be log_trace.**
- Various `device.cpp` ENTRY snapshot and Phase logging: Many are appropriate for degradation scenarios but some fire unconditionally.

**Action (MUST-FIX)**: Audit all 159 `log_warning` additions. Production hot-path calls must be `log_trace`. Degradation-scenario warnings are appropriate at `log_warning`.

### 2. Sleep Calls in Production Paths

Multiple `sleep_for` calls added:
- `500ms` in FIX AC teardown (MMIO ERISC reboot wait) — acceptable, only fires in error path
- `200ms` in FIX AC timeout-only path — acceptable, only fires in error path
- `100ms` in various poll loops — acceptable as backoff
- `15s` sleeps appear in test files only — acceptable

No sleeps were found in happy-path production code.

### 3. Thread Safety

Several new `device_impl.hpp` fields lack synchronization:
- `fabric_pre_dead_channels_`: documented as thread-unsafe, relies on external serialization
- `pending_eth_launch_`, `pending_quiesce_newly_dead_eth_chans_`, `pending_phase25_force_reset_chans_`: same

The `fabric_relay_path_broken_` and `fabric_teardown_timed_out_` are correctly `std::atomic<bool>`.

**Action** (nice-to-have): Add `TT_ASSERT` that the caller holds the mesh-level lock when accessing non-atomic fields.

### 4. Code Duplication

- `edm_status_name()` is defined in both `fabric_firmware_initializer.cpp` (anonymous namespace) and `device.cpp` (anonymous namespace). Two separate implementations of the same lookup table.
- `WriteRuntimeArgsToDevice` and `ConfigureDeviceWithProgram` overloads in `tt_metal.cpp` duplicate ~100 lines each from originals.

**Action** (should-fix): Move `edm_status_name()` to a shared header. Refactor the `tt_metal.cpp` overloads to use optional skip-set parameter.

### 5. Test Coverage

The branch adds extensive test infrastructure in `test_async_teardown_race.cpp` (~4200 lines of diff, ~30 scenarios). The tests use test seams (`CompileFabricFn`, `StatusOverrideFn`, `ConfigureFabricCoresInjectFn`) to exercise error paths without hardware. This is well-designed.

**Concern**: The test seams use `static thread_local` function pointers. These are zero-cost in production (empty `std::function` check is a single branch) but do add a branch to every probe read and compile call. Acceptable.

---

## Risk Assessment

### Merge-Ready?

**NO** — two blockers must be resolved:

1. **Debug log_warning calls in hot paths** (dispatch.cpp event recording, device_operation.hpp workload logging). These will spam production logs and may affect performance.

2. **Verify the UMD submodule bump is compatible** with current main. The UMD is pinned to `4234371aa6e` (5 commits ahead of main's `453a1a1a91e`). These commits are branch-specific fixes — they need to be merged to UMD main first, or the tt-metal merge will break other branches that use the stock UMD.

### After Blockers Resolved

The fix logic is sound and well-tested. The branch should be safe to merge with the understanding that:
- Happy-path performance is unaffected (all guards check empty sets or false flags)
- Degraded-mode behavior is dramatically improved (clean errors instead of hangs)
- The fetch_queue rewrite is a net performance improvement (fast-path avoids PCIe reads)

---

## Prioritized Action Items

### MUST-FIX Before Merge

| # | Item | File(s) | Effort |
|---|------|---------|--------|
| 1 | Demote 4 `log_warning` in event dispatch to `log_trace` | `tt_metal/impl/event/dispatch.cpp` | 5 min |
| 2 | Demote `log_info` in device_operation hot path to `log_trace` | `ttnn/api/ttnn/device_operation.hpp` | 5 min |
| 3 | Audit remaining 155 `log_warning` additions — demote debug-only ones | Multiple files | 30 min |
| 4 | Verify UMD submodule commits are merged to UMD main or will be before tt-metal merge | UMD repo | External dependency |

### SHOULD-FIX (High Priority)

| # | Item | File(s) | Effort |
|---|------|---------|--------|
| 5 | Add `TT_ASSERT(consumed <= in_flight)` in debug builds for underflow detection | `system_memory_manager.cpp` | 5 min |
| 6 | De-duplicate `edm_status_name()` — move to shared header | `fabric_firmware_initializer.cpp`, `device.cpp` | 20 min |
| 7 | Refactor `WriteRuntimeArgsToDevice`/`ConfigureDeviceWithProgram` overloads to use optional parameter | `tt_metal.cpp` | 30 min |

### NICE-TO-HAVE (Low Priority)

| # | Item | File(s) | Effort |
|---|------|---------|--------|
| 8 | Replace fixed sleeps (500ms/200ms) in FIX AC with polled readiness checks | `risc_firmware_initializer.cpp` | 1 hr |
| 9 | Make relay_broken per-MMIO-host for Galaxy support | `fabric_firmware_initializer.cpp` | 2 hr |
| 10 | Add serialization invariant assertions for non-atomic device fields | `device_impl.hpp` | 30 min |
| 11 | Document the full quiesce phase ordering in a single location | New doc or existing header | 1 hr |

---

## Per-FIX Quick Reference

```
FIX A   relay queue drain                  fabric_firmware_init    OK — happy path zero-cost
FIX B   poll all ETH channels teardown     fabric_firmware_init    OK
FIX C   skip dead ETH in launch/config     device.cpp, tt_metal    OK — code duplication concern
FIX E   skip dispatch for dead-relay       fabric_firmware_init    OK
FIX E2  mark dead-relay on probe timeout   fabric_firmware_init    OK — restricted to non-MMIO
FIX E3  populate static_config dead-relay  device_manager          OK
FIX F   guard teardown l1_barrier          fabric_firmware_init    OK
FIX G   skip dead-relay in sync/health     fabric_firmware_init    OK
FIX H   short-circuit relay probe          fabric_firmware_init    OK — Galaxy concern
FIX I   MMIO dead-peer detection           fabric_firmware_init    OK
FIX I2  skip Phase 5 for dead peers        device.cpp              OK
FIX J   guard teardown_fabric_config       metal_env.cpp           OK
FIX K   suppress TERMINATE for corrupt     fabric_firmware_init    OK
FIX L   skip zero-write for base-UMD       fabric_firmware_init    OK
FIX M   skip soft reset for base-UMD       fabric_firmware_init    OK — critical
FIX N   skip non-MMIO soft reset quiesce   device.cpp              OK
FIX O   corrupt channels -> probe_dead     fabric_firmware_init    OK
FIX P   throw if MMIO excluded from mesh   topology_mapper         OK
FIX P2  skip pre-dead channels Phase 5     device.cpp              OK
FIX Q   UMD null-guard SIGSEGV            UMD submodule           OK
FIX R   skip Phase 2.5 relay-broken       device.cpp              OK
FIX S   relay_broken on snapshot timeout   device.cpp              OK
FIX T   canary, relay reset, log quality   device.cpp              OK
FIX U   clean return Phase 5 relay fail    device.cpp              OK
FIX V   relay_broken on Phase 5 timeout    device.cpp              OK
FIX W   Phase 5b all-dead MMIO extend      device.cpp              OK
FIX X   ETH training heartbeat race        UMD submodule           OK (v2 fix)
FIX Z   relay_broken CQ fast throw         fd_mesh_command_queue   OK
FIX AA  skip AllGather relay broken        device_operation        OK
FIX AB  hard-reset MMIO ETH teardown       risc_firmware_init      OK
FIX AC  two-phase ETH reset teardown       risc_firmware_init      OK — sleep concern
FIX AD  skip MMIO ETH soft-reset quiesce   device.cpp              OK
FIX AE  three-pass deferred ETH launch     device.cpp              OK — key architectural fix
FIX AF  poll STARTED between launches      device.cpp              OK
FIX AH  guard ETH TX flush                 fabric_firmware_init    OK
FIX AI  assert+deassert ALL in teardown    fabric_firmware_init    OK — critical
FIX AI2 Phase 2.5 force-reset tracking     device.cpp              OK
FIX AJ  skip l1_barrier relay-dead         fabric_firmware_init    OK — scoping bug fixed
FIX AK  transitive dead-relay l1_barrier   fabric_firmware_init    OK
FIX AL  read hw_fence at reset()           system_memory_manager   OK — critical
FIX AM  skip health check dead-relay       fabric_firmware_init    OK
FIX AN  set relay_broken Phase 2.5 catch   device.cpp              OK
FIX AO  skip termination writes relay-dead dispatch                OK
FIX AP  skip non-MMIO relay ops teardown   risc_firmware_init      OK
FIX AQ  UMD topology relay timeout         UMD submodule           OK
FIX AQ2 UMD skip ETH connections to skipped UMD submodule          OK
```

---

## Conclusion

This is a thorough and systematic fix of deeply interconnected hardware race conditions. The fixes show strong understanding of the T3K ETH relay architecture, ERISC firmware lifecycle, and UMD relay protocol. The main risks are operational (debug logging, UMD submodule coordination) rather than correctness. The fetch_queue rewrite is independently valuable and should be considered for a separate PR if the full branch is too large to merge.
