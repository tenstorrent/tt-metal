<!-- SUMMARY: Root-cause analysis of non-MMIO device silent teardown failures in nsexton/0-racecondition-hunt vs main, covering the full code path from FabricFirmwareInitializer::teardown through ~Cluster and the UMD destructor fix
KEYWORDS: non-mmio, teardown, silent-failure, fabric, ETH, relay-broken, stale_base_umd, FIX AE, FIX AC, FIX AY, FIX M, FIX BA, UMD destructor
SOURCE: code analysis of branch nsexton/0-racecondition-hunt + main branch comparison + /tmp/ttnn_tests_log.txt
SCOPE: non-MMIO device teardown path on nsexton/0-racecondition-hunt vs main
USE WHEN: investigating non-MMIO silent failure root causes -->

# Non-MMIO Silent Failure Root Cause Analysis (Researcher A)

## 1. Architecture Context

On T3K (2x4 mesh), devices 0-3 are MMIO (direct PCIe access), devices 4-7 are non-MMIO (accessible only via UMD ethernet relay through MMIO ERISCs). All L1 reads/writes to non-MMIO devices go through `write_to_non_mmio` / `read_non_mmio` in UMD, which use a 4-slot ETH relay command queue.

The silent failure pattern:
1. A test uses fabric on all 8 devices
2. Teardown fails to fully clean up non-MMIO devices 4-7
3. ETH channels on devices 4-7 remain in "base UMD firmware" or "FABRIC firmware" state
4. The NEXT process sees stale firmware and either hangs or operates in degraded mode
5. No error was raised during the original teardown — the failure is silent

## 2. Main Branch Teardown (VULNERABLE)

**File**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` (main)
**Lines 189-208**:
```
void RiscFirmwareInitializer::teardown(...) {
    for (ChipId device_id : all_devices) {
        assert_cores(device_id);      // <-- HANGS 5s per dead non-MMIO device
        cluster_.l1_barrier(device_id); // <-- ALSO HANGS 5s per dead non-MMIO device
    }
    cluster_.set_internal_routing_info_for_ethernet_cores(control_plane, false);
}
```

**Main branch is critically vulnerable**:
- No relay-broken detection
- No try/catch around `assert_cores` or `l1_barrier` for non-MMIO devices
- Each dead relay channel blocks for 5 seconds (UMD timeout)
- With 4 non-MMIO devices x multiple cores = 40-320 seconds of wasted blocking
- `set_internal_routing_info_for_ethernet_cores` writes to ALL devices including non-MMIO — if relay is dead, these writes also hang

**Main branch `~Cluster`** (lines 477-482):
```
Cluster::~Cluster() {
    this->driver_->close_device();  // <-- NO TRY/CATCH, NO relay_broken marking
}
```
- No `mark_relay_broken` before close_device
- No exception handling — `close_device()` → `RemoteChip::close_device()` → `set_power_state()` → `wait_for_non_mmio_flush()` HANGS for 5s per non-MMIO chip
- If it throws, `std::terminate()` is called (destructor is noexcept)

**Main branch `FabricFirmwareInitializer::teardown`** (lines 66-138):
- Writes TERMINATE signal to all devices via `WriteToDeviceL1` — no relay-broken check
- Calls `cluster_.l1_barrier(dev->id())` for all devices — hangs on dead non-MMIO relay
- No timeout, no try/catch, no skip logic

## 3. Branch Teardown (Heavily Patched — 30+ FIX annotations)

The branch has an extensive multi-step teardown in `RiscFirmwareInitializer::teardown()` spanning ~600 lines (lines 241-850+):

### Step 1: Scan device flags (lines 275-333)
- Builds `relay_broken_non_mmio` set from two sources:
  - `dev->is_fabric_relay_path_broken()` — set during quiesce/fabric operations
  - **FIX BA**: `dev->is_fabric_channels_not_ready_for_traffic()` — catches case where FIX AM fired (STARTED early-exit) but relay wasn't explicitly marked broken

### Step 2: Early MMIO ETH PCIe reset (lines 335-694) — FIX AC
- Hard-resets MMIO ETH channels via PCIe (no relay needed)
- Polls heartbeat address (0x1F80 on WH) for ERISC reboot confirmation
- **FIX AQ**: Secondary poll of edm_status_address (0x18070) to close race between "heartbeat incrementing" and "UMD relay sentinel 0x49706550 written"
- **FIX AY**: Deferred non-MMIO ETH ERISC reset via restored MMIO relay
- **FIX AV**: Short-circuit per-device on first relay failure (avoids N x 5s waste)
- **FIX PG**: If ALL MMIO heartbeats timed out, skip FIX AY entirely

### Step 3: assert_cores/l1_barrier with skip logic (lines 696-780)
- Skips ALL non-MMIO devices when any relay is broken
- **FIX AZ**: Dynamic detection — if assert_cores throws for ANY non-MMIO device, sets `relay_dead_detected_step3` and skips all subsequent non-MMIO devices
- **FIX AZ+**: Also skips l1_barrier for MMIO devices when assert_cores throws (handles misclassified MMIO devices on some runners)

### Step 4: set_internal_routing_info skip when relay broken

### Step 5: Fallback MMIO ETH reset when teardown timed out but relay wasn't broken

### FabricFirmwareInitializer::teardown (lines 263-661)
- **Phase 1**: l1_barrier with try/catch per device (lines 325-345)
- **Phase 2**: Tensix MUX TERMINATE with timeout (lines 350-440)
- **Phase 3**: ERISC TERMINATE write — FIX AU skips relay-broken non-MMIO devices (lines 458-476, 537-661)
- **Phase 4**: l1_barrier poll for TERMINATED status — relay-broken non-MMIO channels queued into `relay_broken_force_reset` instead (lines 537-661)

### ~Cluster (lines 509-553) — FIX AE + FIX J
```cpp
// FIX AE: Mark ALL remote chips relay-broken before close_device()
for (chip_id : all_chip_ids()) {
    if (is_chip_remote(chip_id))
        driver_->mark_relay_broken(chip_id);
}
// FIX J: try/catch around close_device()
try { driver_->close_device(); } catch (...) { log_warning(...); }
```

### tt_cluster.cpp write_core / read_core guards
- **FIX NY** (line 887): `relay_broken_chips_` set — skip write_core immediately if relay known broken
- **FIX NX** (line 903): Catch flush timeout, mark relay broken
- **FIX NZ** (line 934): Skip read_core for relay-broken non-MMIO chips

## 4. The Topology Check Problem

**File**: `tests/scripts/t3000/run_t3000_unit_tests.sh` line 120:
```bash
raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>/dev/null)
```

This Python one-liner:
1. `import ttnn` → creates MetalContext → creates Cluster → UMD opens all 8 devices
2. UMD installs base-UMD firmware (0x49706550) on ALL ETH channels including non-MMIO
3. `print(...)` → program exits → Python garbage collector runs → MetalContext destructor runs
4. On **main branch**: `~Cluster` calls `close_device()` with NO relay-broken marking → non-MMIO `wait_for_non_mmio_flush` may hang or leave incomplete cleanup
5. Non-MMIO ETH L1 channels are left with base-UMD firmware sentinel (0x49706550) in `edm_status_address` (0x18070)
6. The NEXT process's `terminate_stale_erisc_routers()` sees 0x49706550, classifies these as "stale base UMD" channels → FIX M kicks in → `fabric_stale_base_umd_channels_ = true` → degraded mode

## 5. UMD Destructor Fix Analysis

**Commit**: `5a2e723c3ceade3a06cd23923d18822c0dcb9e1f`
**File**: `device/chip_helpers/silicon_sysmem_manager.cpp`

**What it does**: Wraps `unpin_or_unmap_sysmem()` in `SiliconSysmemManager::~SiliconSysmemManager()` with try/catch. Before this fix, if `TENSTORRENT_IOCTL_UNPIN_PAGES` returned ENODEV (device already partially cleaned up), the destructor threw → `std::terminate()` → SIGABRT.

**Is it sufficient?** No — this fix addresses a narrow scenario:
- It prevents SIGABRT when the PCIe device is already gone during sysmem cleanup
- It does NOT address the core non-MMIO silent failure — that's about ETH relay firmware state, not DMA page unmapping
- It's a defense-in-depth fix that prevents process crashes during cleanup, but the stale ETH L1 state is the real root cause

## 6. Where Silent Failures Occur — Specific Code Locations

### Location 1: FabricFirmwareInitializer::teardown Phase 3 (branch line ~472)
**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:472`
```
if (is_non_mmio && dev->is_fabric_relay_path_broken()) {
    // SKIPS TERMINATE write + l1_barrier for relay-broken non-MMIO
}
```
When relay is broken, TERMINATE is never sent to non-MMIO ERISCs. They remain running FABRIC firmware. On main branch, this skip doesn't exist — but the unprotected WriteToDeviceL1 hangs instead.

### Location 2: RiscFirmwareInitializer::teardown Step 2/FIX AY (branch line ~603-667)
FIX AY attempts deferred reset of non-MMIO ERISCs via restored MMIO relay. If relay re-sync fails (FIX AV device_relay_dead), those ERISCs are NEVER reset. Log says:
```
"FIX AY/AV — {succeeded}/{total} non-MMIO ETH ERISCs reset ... {failed} failed/skipped"
```
These failed channels carry stale FABRIC firmware into the next session.

### Location 3: ~Cluster on main (NO relay-broken marking)
**File**: `tt_metal/llrt/tt_cluster.cpp:477-482` (main)
`close_device()` without `mark_relay_broken` means UMD attempts `wait_for_non_mmio_flush` for every non-MMIO chip → 5s x N hangs → process exit leaves non-MMIO firmware unclean.

### Location 4: Topology check script
**File**: `tests/scripts/t3000/run_t3000_unit_tests.sh:120`
The `import ttnn` + exit installs base-UMD firmware on all ETH channels then exits without proper fabric teardown (no fabric was ever configured — just base UMD). The base-UMD firmware sentinel 0x49706550 remains in `edm_status_address` on non-MMIO channels.

### Location 5: terminate_stale_erisc_routers relay queue saturation
**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:1037-1064`
On non-MMIO devices, each probe read that times out fills one slot in the 4-slot ETH relay queue. After 3 timeouts (`kMaxRelayTimeouts = 3`), the function declares `relay_broken = true` and adds ALL remaining channels to `probe_dead_channels`. These channels are never probed or cleaned.

## 7. Main Branch vs Branch Comparison

| Aspect | main | nsexton/0-racecondition-hunt |
|---|---|---|
| Teardown assert_cores | No protection — hangs 5s/device | Try/catch + relay-broken skip |
| ~Cluster relay marking | None | FIX AE: marks all remote chips broken |
| ~Cluster close_device | No try/catch (std::terminate on throw) | FIX J: try/catch |
| Fabric teardown non-MMIO | No skip — WriteToDeviceL1 hangs | FIX AU/AY: skip relay-broken, queue for force-reset |
| MMIO ETH PCIe reset | None | FIX AC: hardware reset + heartbeat poll |
| Non-MMIO deferred reset | None | FIX AY: via restored relay (best-effort) |
| write_core/read_core guards | None | FIX NY/NX/NZ: relay_broken_chips_ skip + catch |
| Topology check stale FW | No mitigation | Same topology check exists — FIX M handles stale base-UMD on next init |

## 8. Remaining Gaps on the Branch

1. **FIX AY relay re-sync failure**: When UMD relay protocol state isn't re-synced after MMIO ERISC PCIe reset (common — UMD maintains per-chip state that isn't reset by hardware), ALL deferred non-MMIO resets fail. The log shows `FIX AY/AV — 0/N succeeded`. These non-MMIO ERISCs remain in whatever firmware state they were in.

2. **Topology check side effect**: The `python3 -c "import ttnn; print(ttnn.GetNumAvailableDevices())"` call installs base-UMD firmware on non-MMIO ETH channels. Even with FIX AE in ~Cluster, marking relay broken only makes `wait_for_non_mmio_flush` return instantly — it does NOT reset the non-MMIO ERISCs. The base-UMD firmware (0x49706550) persists.

3. **No cross-process cleanup protocol**: There is no mechanism for a new process to force-reset non-MMIO ERISCs before opening UMD. `terminate_stale_erisc_routers` is the "next session recovery" mechanism, but it runs AFTER the cluster is opened — and probe reads to non-MMIO devices may themselves timeout/hang during discovery.

4. **FIX BA edge case**: `fabric_channels_not_ready_for_traffic_` without `fabric_relay_path_broken_` means the relay is working but firmware handshake didn't complete. FIX BA treats this as relay-broken for teardown, which triggers FIX AC (MMIO PCIe reset) — this may be overly aggressive and reset healthy MMIO ERISCs unnecessarily.

5. **`relay_timeout_count` in terminate_stale_erisc_routers**: After 3 timeouts (`kMaxRelayTimeouts = 3`), all remaining channels are marked dead without ever being probed. On a healthy cluster where only one channel is truly dead, this cascades to mark healthy channels as dead too — because they share the same 4-slot relay queue.

## 9. Conclusions

- **Main branch is severely vulnerable**: zero protection against non-MMIO teardown failures. Every unclean exit leaves stale firmware that corrupts the next session.
- **The branch has ~30 FIX annotations** creating an extensive defensive teardown, but the fundamental problem remains: **non-MMIO devices cannot be directly reset via PCIe — they MUST go through the UMD ETH relay, which is the very thing that's broken during teardown.**
- **FIX AY** (deferred reset via restored relay) is the only mechanism that can actually clean up non-MMIO ERISCs, and it fails when UMD relay protocol state isn't re-synced.
- **The UMD destructor fix (5a2e723c3)** prevents SIGABRT during DMA page cleanup but does NOT address the core non-MMIO stale firmware problem.
- **The real fix** would need to happen at a lower level: either (a) a PCIe-level mechanism to reset non-MMIO devices without going through the ETH relay, or (b) UMD maintaining relay protocol state across PCIe ERISC resets so FIX AY consistently succeeds, or (c) a hardware watchdog on non-MMIO ERISCs that auto-resets to base firmware when the host process disappears.
