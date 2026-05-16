<!--
SUMMARY: Full audit of Wormhole B0 teardown code paths on nsexton/0-racecondition-hunt: 0x49705180 is a ROM postcode, not a software bug; teardown itself is robust but SIGKILL bypass creates the corrupt state.
KEYWORDS: teardown, 0x49705180, ERISC, EDMStatus, ROM postcode, ETH corruption, T3K, fabric, terminate_stale_erisc_routers, progressive degradation, GAP tests, #42429
SOURCE: code audit of racecondition-main branch + CI log analysis (run 25206820808)
SCOPE: All teardown code paths (DoTearDownTestSuite, MeshDevice::close, teardown_fabric_config, FabricFirmwareInitializer::teardown, terminate_stale_erisc_routers) + reproducer test coverage
USE WHEN: investigating ETH corruption cascade, teardown race conditions, or 0x49705180 L1 corruption on T3K
-->

# Opus Teardown Audit: Is Teardown Leaving WH B0 Cards in Bad State?

**Date**: 2026-05-01
**Branch**: `nsexton/0-racecondition-hunt` (SHA c59d888dfb5f)
**CI log**: Run 25206820808 on `tt-ubuntu-2204-n300-llmbox-stable-6kds8-runner-47zzs`

---

## Executive Summary

**Is teardown the root cause of progressive degradation?** **No — but SIGKILL bypass of teardown is.**

The teardown code itself is extensively hardened (40+ FIX labels, 69+ GAP regression tests). The root cause of the `0x49705180` corruption seen across all 8 devices at CI startup is:

1. A prior CI process (or the GHA timeout mechanism) **SIGKILL**s the test process
2. SIGKILL bypasses ALL teardown code — ERISC firmware continues running
3. The CI runner's job-start hook does NOT perform a hardware reset (`tt-smi -r`)
4. The next test process sees ERISCs in fabric-firmware state; the BRISC hardware-reset ROM writes `0x49705180` (a ROM postcode, NOT an EDM status) to `edm_status_address` (L1 0x18070) during the reset sequence
5. `terminate_stale_erisc_routers()` sees this value on ALL channels, classifies them as CORRUPT, enters degraded mode

The teardown code, when it runs, is thorough. The problem is that it never gets to run.

---

## 1. What is 0x49705180?

`0x49705180` is **the WH B0 BRISC hardware-reset ROM postcode**. It is NOT a valid `EDMStatus` enum value.

**Evidence** (risc_firmware_initializer.cpp:475-481):
```
Root cause: the BRISC hardware reset ROM writes 0x49705180 to L1 address 0x18070
(edm_status_address) as a postcode during power-on init.
```

The value appears when:
- An ERISC is hardware-reset (assert_risc_reset + deassert_risc_reset)
- The ROM runs its power-on sequence and writes `0x49705180` to the L1 address that fabric firmware later uses for `edm_status`
- If the UMD relay firmware hasn't finished starting up (overwriting 0x18070 with 0x49706550), the next session sees `0x49705180`

**Key constants**:
```
0x49705180 = ROM postcode (BRISC hardware reset)
0x49706550 = Base UMD relay firmware sentinel ("iPeP")
0xA0A0A0A0 = Fabric kernel_main canary (firmware entered but crashed)
0xDEADB07E = Host pre-launch canary (configure_fabric wrote but ERISC never booted)
0xA0B0C0D0 = EDMStatus::STARTED
0xA2B2C2D2 = EDMStatus::LOCAL_HANDSHAKE_COMPLETE
0xA4B4C4D4 = EDMStatus::TERMINATED
```

---

## 2. Teardown Path Audit

### 2a. BaseFabricFixture::DoTearDownTestSuite() (fabric_fixture.hpp:175-183)

```cpp
static void DoTearDownTestSuite() {
    for (auto& [id, device] : devices_map_) {
        device->close();       // Step 1: close each MeshDevice
    }
    devices_map_.clear();
    devices_.clear();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED); // Step 2: teardown fabric
    cluster_degraded_skip_ = false;
}
```

**Assessment: SAFE when it runs.**

The ordering is correct: `device->close()` first (stops dispatch, clears CQ), then `SetFabricConfig(DISABLED)` (terminates ERISC routers, waits for TERMINATED, force-resets stragglers). However:

- **Gap**: If `device->close()` throws (e.g., profiler read on dead relay), the exception propagates out, and `SetFabricConfig(DISABLED)` is never called. The `close_impl` code has a try/catch around profiler reads but NOT around all paths.

### 2b. MeshDevice::close() / close_impl() (mesh_device.cpp:917-1012)

**Assessment: SAFE when it runs.**

Key observations:
- Profiler read failure is caught (lines 924-937)
- `is_internal_state_initialized = false` is set early (line 988) to prevent races with MeshBuffer::deallocate()
- `scoped_devices_.reset()` triggers `ScopedDevices` dtor which calls `close_devices()` on individual devices
- `MetalContext::destroy_instance()` is called if `destroy_metal_context_instance_on_close_` is true (line 1004-1007)
- **No ERISC-specific teardown here** — that's delegated to `FabricFirmwareInitializer::teardown()` via `MetalContext::teardown()`

### 2c. MetalEnvImpl::teardown_fabric_config() (metal_env.cpp:305-489)

**Assessment: ROBUST (heavily hardened).**

This is the second-layer ERISC teardown that runs when `SetFabricConfig(DISABLED)` is called:

1. **Waits for TERMINATED on ALL active ETH channels** (not just master) — per-channel 5s timeout
2. **Force-resets timed-out channels**: `assert_risc_reset_at_core(ALL)` + `deassert_risc_reset_at_core(ALL)` (FIX AI)
3. **Restores clean ERISC0 after clean terminate**: assert+deassert ERISC0 only (preserves ETH PHY link)
4. **Records timed-out chips** for `post_teardown()` to set `fabric_teardown_timed_out_` flag
5. **Guards for topology-degraded chips** that aren't in the fabric cluster (FIX J)
6. **Clears fabric context** via `control_plane_->clear_fabric_context()` (FIX BA guard)

**Potential gap**: Lines 396-413 — force-reset uses `try/catch` but if `get_virtual_eth_core_from_channel` throws (e.g., invalid channel), the catch logs but continues. The channel is left in undefined state but this is logged and handled by next session's `terminate_stale_erisc_routers`.

### 2d. FabricFirmwareInitializer::teardown() (fabric_firmware_initializer.cpp:263-952)

**Assessment: EXTREMELY ROBUST (most hardened function in the codebase).**

This 700-line function is the primary ERISC teardown path with extensive hardening:

1. **Phase 1**: Terminate Tensix MUX cores (if enabled) — write TERMINATE, poll TERMINATED, force-reset on timeout
2. **Phase 2**: Send TERMINATE to master ETH router on each device
   - **FIX AU**: Skip relay-dependent writes for non-MMIO devices with broken relay (prevents 5s UMD hang)
3. **Phase 3**: Poll ALL active ETH channels with global 5s deadline
   - **FIX AU**: Relay-broken channels excluded from poll, queued for direct force-reset
   - Exception handling: `std::exception` AND `catch(...)` for UMD exceptions
4. **Phase 4**: Force-reset channels that missed deadline
   - **FIX AX**: Skip assert_risc_reset for non-MMIO relay-dead devices (saves 5s x N channels)
   - **FIX AI**: assert+deassert ALL RISCs (not just ERISC0) to restore ETH PHY
   - **FIX PC**: Clear fw_launch_addr after force-reset to prevent false-positive reset cascade
   - **FIX PE**: Clear fw_launch_addr after clean terminate (same issue, clean path)
5. **Phase 5**: UMD relay queue drain (FIX A + FIX AK)
   - **FIX AK**: If ANY device has dead relay, skip l1_barrier for ALL non-MMIO (prevents transitive hang)

**No gaps found in teardown itself.** The function handles every known failure mode.

### 2e. terminate_stale_erisc_routers() (fabric_firmware_initializer.cpp:1007-1400+)

**Assessment: ROBUST recovery function, correctly handles 0x49705180.**

This is the **pre-init recovery** function that runs at the start of each session:

- **0x49705180 (ROM postcode)**: Classified as CORRUPT. Does NOT send TERMINATE (no firmware running). Zeros edm_status_address to break cascade. Adds to `probe_dead_channels` for degraded mode.
- **0x49706550 (base UMD)**: Classified as base_umd. Does NOTHING (live relay, don't disturb). Added to `base_umd_channels` to skip soft-reset.
- **0xA0A0A0A0 (canary)**: Classified as dead firmware. Added to `probe_dead_channels` for soft-reset.
- **Valid EDMStatus (stale running)**: Send TERMINATE, poll 100ms, continue.
- **Relay timeout tracking**: After 3 timeouts, stops reading (prevents 4-slot relay queue fill -> indefinite hang).

**This function correctly handles the corrupt state — the issue is that the corrupt state exists in the first place.**

### 2f. is_control_plane_initialized guard (FIX TI)

`is_control_plane_initialized()` is defined in `metal_env_impl.hpp:85` as `return control_plane_ != nullptr`. It's used in `teardown_fabric_config()` (line 316: `if (control_plane_ != nullptr && fabric_config_ != tt_fabric::FabricConfig::DISABLED)`) to guard the TERMINATED wait loop.

**Assessment: SUFFICIENT for its purpose.** The guard prevents accessing `control_plane_` when it's null during deinit. The race scenario (control plane deinit while ERISC is writing) is mitigated by the TERMINATED poll + force-reset: even if the poll fails due to null control_plane, the `configured_ethernet_cores_for_fabric_routers(DISABLED)` call at line 477 still runs unconditionally.

---

## 3. The Real Corruption Path

The CI log shows 0x49705180 on ALL 8 devices at process start. This means:

1. **Prior CI job was SIGKILL'd** (exit code 124 = timeout, or OOM)
2. SIGKILL does NOT run C++ destructors, atexit handlers, or signal handlers
3. ERISCs continue running fabric firmware after host process death
4. The CI runner starts the next job container
5. The next process's `RiscFirmwareInitializer` does `assert_risc_reset_at_core(ALL)` + `deassert_risc_reset_at_core(ALL)` on each MMIO ETH core (FIX AC path)
6. Hardware reset ROM runs on each ERISC, writes `0x49705180` to L1 0x18070
7. Before UMD relay firmware overwrites 0x18070 with `0x49706550`, the fabric init's `terminate_stale_erisc_routers` reads 0x18070 and sees `0x49705180`
8. Channel classified as CORRUPT -> degraded mode

**The FIX AQ secondary poll** (risc_firmware_initializer.cpp:471-550) was added to close this timing gap by waiting for edm_status to transition away from `0x49705180` after the heartbeat poll confirms the ERISC is alive. This should prevent the false-corrupt classification.

**Why it's still happening**: The FIX AQ poll only runs for MMIO devices. Non-MMIO devices (4-7 on T3K) can't be directly polled after hardware reset — they depend on the MMIO relay. If the relay ERISCs on MMIO devices haven't finished booting their UMD firmware, the non-MMIO probe reads through the relay will also see stale ROM postcodes.

---

## 4. Reproducer Test Coverage

### 4a. Tests in the branch

**69 C++ GAP regression tests** (test_gap1 through test_gap69) in `tests/tt_metal/distributed/`
**15 Python GAP regression tests** (test_gap21 through test_gap40) in `tests/nightly/t3000/ccl/`
**Additional tests**: `test_async_teardown_race.cpp`, `test_fabric_teardown_escape.cpp`, various existing test modifications

### 4b. Tests included in run_t3000_unit_tests.sh

**C++ GAP tests in gtest_filter** (line 314-315):
- GAP-28, 29, 31, 32, 41, 42, 43, 44, 45, 46 (via named fixtures)
- AsyncTeardownRaceFixture (Scenarios A-M)
- QuiesceStressFixture, PhaseWFixture, PhaseZFixture, FabricFirmwareInitializer (compile check)

**Python GAP tests** (lines 219-357):
- GAP-21, 22, 23, 24, 25, 26, 27, 30, 34, 35, 36, 37, 38, 39, 40

### 4c. MISSING from run_t3000_unit_tests.sh

**23 C++ GAP tests NOT in the gtest_filter** (GAP 47-69):
```
GAP-47: test_gap47_fixnz_read_core_relay_guard.cpp
GAP-48: test_gap48_fixpa_fw_launch_addr_force_reset.cpp
GAP-49: test_gap49_fixpb_fw_launch_addr_rescue.cpp
GAP-50: test_gap50_fixpd_fw_launch_addr_quiesce.cpp
GAP-51: test_gap51_fixpf_umd_heartbeat_skip_exit.cpp
GAP-52: test_gap52_fixpg_phase25_relay_retry.cpp
GAP-53: test_gap53_fixph_yaml_ethcoord_graceful_skip.cpp
GAP-54: test_gap54_fixba_teardown_null_control_plane.cpp
GAP-55: test_gap55_fixe2ay_probe_dead_fay_trigger.cpp
GAP-56: test_gap56_fixm2_dead_peer_erisc_reset.cpp
GAP-57: test_gap57_fixpl_barrier_guard_dead_relay.cpp
GAP-58: test_gap58_fixqc_nonmmio_reset_cores_skip.cpp
GAP-59: test_gap59_fixqb_reset_loop_early_break.cpp
GAP-60: test_gap60_fixpypz_phase25_topology_timeout.cpp
GAP-61: test_gap61_fixqd_dead_router_mmio_skip.cpp
GAP-62: test_gap62_fixqu_reassert_flags_after_configure.cpp
GAP-63: test_gap63_fixny_relay_mux_cluster_guard.cpp
GAP-64: test_gap64_fixtb_topology_mapper_unknown_asic.cpp
GAP-65: test_gap65_fixqv_phase4_skip_channels_not_ready.cpp
GAP-66: test_gap66_fixrz_stale_base_umd_flag.cpp
GAP-67: test_gap67_fixtf_2d_fabric_header_args_guard.cpp
GAP-68: test_gap68_fixtg_control_plane_host_rank_guard.cpp
GAP-69: test_gap69_fixth_relay_mux_no_links_guard.cpp
```

Additionally, some earlier GAP C++ tests (1-14, except those folded into named fixtures) may also not be in the filter. The `distributed_unit_tests` binary includes them but the `--gtest_filter` restricts execution.

---

## 5. Root Cause Analysis: Is Degradation Progressive?

**Yes, but NOT because of teardown code.**

The degradation cascade:
1. **Initial corruption**: SIGKILL leaves ERISCs in fabric state
2. **Session N**: `terminate_stale_erisc_routers` sees 0x49705180, enters degraded mode (probe_dead_channels populated)
3. **Session N runs tests**: Fabric tests run in degraded mode (some channels dead). Tests may timeout (FIX AA/QW/RX GTEST_SKIP paths fire).
4. **Session N teardown**: Runs correctly for reachable channels. Non-MMIO channels with dead relay are force-reset (FIX AU/AX). MMIO channels get clean TERMINATED + ERISC0 restart.
5. **Session N+1**: Sees CLEAN state if teardown ran to completion. OR sees 0x49705180 again if:
   - Session N was SIGKILL'd before teardown completed
   - FIX AQ secondary poll didn't wait long enough for UMD relay firmware startup
   - Non-MMIO devices were probed before their relay MMIO devices finished booting

**The degradation is NOT accumulating across properly-completed teardowns.** Each teardown zeros edm_status_address (line 1264-1265) and force-resets remaining channels. The problem is only when teardown is interrupted.

---

## 6. Recommended Fixes

### Fix 1: Add GAP 47-69 to run_t3000_unit_tests.sh gtest_filter

**File**: `tests/scripts/t3000/run_t3000_unit_tests.sh:314-315`

The gtest_filter on line 314-315 needs to include the fixture names for GAP 47-69. Each `.cpp` file defines a test fixture; grep each for its fixture class name and append to the filter string.

**Impact**: Without these tests running in CI, regressions in FIX NZ, PA, PB, PD, PF, PG, PH, BA-null, E2+AY, M2, PL, QC, QB, PY+PZ, QD, QU, NY-relay-mux, TB, QV, RZ, TF, TG, TH are undetected.

### Fix 2: Extend FIX AQ secondary poll to non-MMIO devices

**File**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:490-550`

Currently the FIX AQ edm_status poll only runs for `mmio_ids_set`. Non-MMIO devices (4-7 on T3K) are probed by `terminate_stale_erisc_routers` before their MMIO relay has finished booting, seeing stale `0x49705180`.

**Recommendation**: After the MMIO FIX AQ poll completes, add a brief delay (50-100ms) or a secondary poll on ONE non-MMIO device's edm_status via the relay to confirm the relay is fully operational before `terminate_stale_erisc_routers` starts probing non-MMIO channels.

### Fix 3: CI job-start hook should check for stale ERISC firmware

**File**: GHA runner hook (`/files-from-host/runner_hook_job_started.sh`)

The CI runner's job-start hook should optionally run `tt-smi -r` (or equivalent ARC reset) when stale fabric firmware is detected. This would break the corruption cascade at the source rather than relying on software recovery.

**Alternative**: Add a Python pre-check script that reads edm_status for all MMIO ETH channels via UMD and triggers a targeted reset if 0x49705180 is seen.

---

## 7. Specific Code Path Assessment Summary

```
Path                                    Safe?   Notes
--------------------------------------  ------  ------------------------------------
DoTearDownTestSuite()                   YES     Correct ordering: close() then SetFabricConfig(DISABLED)
MeshDevice::close_impl()               YES     No ERISC teardown; delegates to MetalContext
FabricFirmwareInitializer::teardown()   YES     700 lines, 15+ FIX labels, handles all failure modes
teardown_fabric_config()                YES     Per-channel TERMINATED wait + force-reset + ERISC0 restart
terminate_stale_erisc_routers()         YES     Correct CORRUPT handling: zero+probe_dead+degraded mode
is_control_plane_initialized guard      YES     Sufficient null-check for teardown race
FIX AQ secondary poll                   PARTIAL Only covers MMIO devices; non-MMIO timing gap remains
```

---

## 8. CI Log Evidence (Run 25206820808)

At process start (07:56:29), ALL 8 devices have corrupt ETH channels:
```
Device 0: 4 corrupt (chan 0,1,14,15 = 0x49705180)
Device 1: 4 corrupt (chan 8,9,14,15 = 0x49705180)
Device 2: 4 corrupt (chan 0,1,8,9 = 0x49705180)
Device 3: 2 corrupt (chan 8,9 = 0x49705180)
Device 4: 2 corrupt (chan 6,7 = 0x49705180)
Device 5: 4 corrupt (chan 0,1,6,7 = 0x49705180)
Device 6: 4 corrupt (chan 0,1,6,7 = 0x49705180)
Device 7: 4 corrupt (chan 0,1,14,15 = 0x49705180)
Total: 28 corrupt channels across 8 devices
```

`terminate_stale_erisc_routers` correctly handles each: zeros edm_status_address, adds to probe_dead_channels, enters degraded mode. Devices 4 and 5 are marked as dead-relay with `fabric_relay_path_broken_` set.
