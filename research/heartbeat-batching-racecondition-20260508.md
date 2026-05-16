<!--
SUMMARY: Investigation of capture_fabric_heartbeats and batched heartbeat monitoring on nsexton/0-racecondition-hunt
KEYWORDS: fabric-heartbeat, batched-reads, health-monitoring, router-state, AERIS, telemetry, racecondition-hunt
SOURCE: Analysis of racecondition-main worktree + batch-t3k reference + prior Addendum A evals
SCOPE: Current soft health checks, batched heartbeat design, per-test vs per-suite applicability, hardware register mapping
USE WHEN: Deciding whether/how to implement capture_fabric_heartbeats on nsexton/0-racecondition-hunt
-->

# Heartbeat Batching & Fabric Health Monitoring — racecondition-hunt

*2026-05-08*

---

## 1. What Soft Health Checking Already Exists in racecondition-hunt

The branch has **four distinct categories** of fabric health monitoring, operating at different granularities and lifecycle phases.

### 1.1 FIX TV — MMIO ETH Heartbeat Poll (Firmware Init)

**File**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:237-333`

**Purpose**: After `reset_cores()` during `run_launch_phase()`, poll MMIO ETH channels to confirm base-UMD firmware has rebooted before `terminate_stale_erisc_routers()` probes them.

**What it reads**:
- `hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT)` — firmware mailbox heartbeat address
  - WH: `0x1F80` (test_results[48], written by base UMD relay firmware)
  - BH: `MEM_SYSENG_ETH_HEARTBEAT = 0x7CC70` (eth_status_t.heartbeat[0])
  - QA: Same as BH (`0x7CC70`)
- Each read: `cluster_.read_reg(&hb_val, tt_cxy_pair(chip_id, virt_core), hb_addr)` — single 4-byte PCIe MMIO read

**Fields read per poll iteration**:
- `hb_val` (uint32_t): raw heartbeat counter value
- Detection logic: `0x0` = not yet running; `(hb_val >> 16) == 0xABCDu` = UMD static marker (FIX TW); `hb_val != prev_hb` = incrementing = alive

**Granularity**: Per-core, per-MMIO-chip. Each `EthRebootPollState` tracks one `tt_cxy_pair`.

**How data is used**: Gate for subsequent `terminate_stale_erisc_routers()`. If channels time out (3000ms), log warning — stale probes may cause `probe_dead` cascade.

**Lifecycle**: Once per `MeshDevice::create()` (per-test in racecondition-hunt's open/close model).

### 1.2 FIX AR — Bulk Parallel Heartbeat Poll (Firmware Teardown)

**File**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:534-619`

**Purpose**: After FIX AC PCIe force-reset of MMIO ETH cores in teardown, poll heartbeats in parallel to confirm base-UMD firmware has rebooted before FIX AY relay restoration.

**What it reads**: Identical register/address as FIX TV — `hal_.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT)` via `cluster_.read_reg()`.

**Granularity**: Per-core, per-MMIO-chip. Same `CorePollState` struct as FIX TV.

**How data is used**:
- `ac_heartbeat_any_ready` flag (line 556): if NO individual core confirmed heartbeat, FIX AY relay restoration is skipped entirely
- Used as a gate for downstream relay-dependent operations

**Lifecycle**: Once per `MeshDevice::close()` teardown (when FIX AC fires).

### 1.3 Phase 5b — ERISC Health Check (Quiesce Restart)

**File**: `tt_metal/impl/device/device.cpp:2405-2755`

**Purpose**: After quiesce-and-restart (Phases 1-5), verify all active ERISC channels reached `READY_FOR_TRAFFIC` (0xA3B3C3D3).

**What it reads**:
- `router_sync_addr` from `builder_ctx.get_fabric_router_sync_address_and_status()` — this is `edm_status_address` in the ERISC's L1
- Each read: `detail::ReadFromDeviceL1(this, eth_logical_core, router_sync_addr, 4, status_buf, CoreType::ETH)` — L1 read (MMIO for local, UMD relay for remote)

**Fields read**: `EDMStatus` uint32_t — compared against `EDMStatus::READY_FOR_TRAFFIC` (0xA3B3C3D3)

**Granularity**: Per-channel, per-device. Iterates `active_channels` set.

**How data is used**:
- Channels not at expected status after 2000ms are classified as "truly unhealthy"
- Distinguishes `LOCAL_HANDSHAKE_COMPLETE` (partial-mesh, FIX AK, non-fatal) from corrupt values (TT_THROW)
- Sets `fabric_channels_not_ready_for_traffic_` flag on device
- Non-MMIO devices with unexpected states return non-fatal (FIX AK-2)

**Lifecycle**: Once per quiesce-and-restart cycle (triggered by `quiesce_devices()` in TearDown).

### 1.4 verify_all_fabric_channels_healthy — Post-Init Channel Verification

**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:2372-2550+`

**Purpose**: After initial fabric init (not quiesce), verify ALL active ERISC channels (not just master) are at `READY_FOR_TRAFFIC`.

**What it reads**: Same as Phase 5b — `edm_status_address` via L1 reads.

**Granularity**: Per-channel, per-device. 3-retry with 50ms backoff (150ms total window).

**How data is used**:
- Fail-fast: throws `TT_THROW` if any channel not healthy after retries
- Skips dead-relay devices (FIX G), MMIO dead-peer devices (FIX I), pre-dead master channels (FIX AN), ring-sync timeout devices (FIX TI)
- External channels (FIX EXT) treated as non-participants
- Sets `fabric_channels_not_ready_for_traffic_` on timeout devices (FIX QD, FIX TK)

**Lifecycle**: Once per `FabricFirmwareInitializer::init()` (per `MeshDevice::create()`).

### 1.5 FabricCommandInterface::read_router_state (Test Infrastructure)

**File**: `tests/tt_metal/tt_fabric/common/fabric_command_interface.cpp:108-129`

**Purpose**: Read `RouterState` (INITIALIZING/RUNNING/PAUSED/DRAINING/RETRAINING) from ERISC L1 for test-level assertions.

**What it reads**:
- `hal.get_dev_addr(ACTIVE_ETH, ROUTER_STATE)` — the `RouterState` field in the ERISC's telemetry region
- Read via `cluster.read_core()` — single 4-byte read per channel

**Granularity**: Per-channel, per-device.

**How data is used**: Test assertions (e.g., verify all routers reach RUNNING after PAUSE/DRAIN/RUN cycle). Used by exactly 1 test (`test_fabric_traffic_generator_kernel.cpp`).

**Lifecycle**: Ad-hoc per test invocation.

### 1.6 Fixture-Level Fabric State Logging

**File**: `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:212-224` (SetUp) and `:266-278` (TearDown)

**Purpose**: Log per-device fabric degradation flags at SetUp/TearDown boundaries.

**What it reads**: Device flag accessors (no hardware reads):
- `idev->is_fabric_relay_path_broken()`
- `idev->is_fabric_channels_not_ready_for_traffic()`
- `idev->is_fabric_stale_base_umd_channels()`

**Granularity**: Per-device (flag-level, not per-channel).

**How data is used**:
- TearDown FIX RX (line 280-312): if any device has broken fabric, skip `quiesce_devices()` and call `close()` directly (saves ~72s)
- Logging only in SetUp

---

## 2. Reference Implementation from batch-t3k (`capture_fabric_heartbeats`)

### 2.1 `fabric_health_detail` Namespace

**File**: `/workspace/group/worktrees/nsexton/0-batch-t3k-ttnn-unit/tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:385-739`

Contains four mechanisms:

**`FabricHeartbeatSnapshot`** (line 404-407): A `std::map<ChipId, std::vector<std::pair<uint64_t, uint64_t>>>` — maps each chip to per-channel `(tx_heartbeat_total, rx_heartbeat_total)`.

**`check_fabric_routers_healthy()`** (lines 422-507):
- Iterates `mesh->get_device_ids()`
- For each chip: calls `tt::tt_fabric::read_fabric_telemetry(fabric_node_id)` — returns `vector<FabricTelemetrySample>`
- For each sample's erisc entries: checks `erisc.router_state != FabricTelemetryRouterState::Active`
- Heartbeat-delta short-circuit: if `before_snapshot` provided and aggregate heartbeats unchanged, skip the per-erisc state walk for that device
- Returns false (unhealthy) if any erisc not Active

**`capture_fabric_heartbeats()`** (lines 510-539):
- Iterates `mesh->get_device_ids()`
- For each chip: calls `read_fabric_telemetry(fabric_node_id)`
- For each sample: sums `erisc.tx_heartbeat` and `erisc.rx_heartbeat` across the 2 erisc entries per channel
- Stores per-channel `(tx_total, rx_total)` in the snapshot map

**`check_fabric_heartbeats_advanced()`** (lines 545-584):
- Captures a new snapshot via `capture_fabric_heartbeats()`
- Compares before vs after per-channel
- Flags stall if heartbeats were non-zero before but did not advance after
- Used for per-test heartbeat stall detection

**`drain_fabric_routers()`** (lines 601-737):
- PAUSE -> DRAIN -> RUN cycle via L1 writes/polls
- Not heartbeat-related, but part of the health-detail namespace

### 2.2 How Heartbeats Are Used in batch-t3k Fixture

```
SetUpShared():
  fabric_heartbeat_before_ = capture_fabric_heartbeats(mesh)   // line 936

SetUp() per-test:
  check_fabric_routers_healthy(mesh, &fabric_heartbeat_before_) // line 956
    -> if unhealthy: drain_fabric_routers()                     // line 962
    -> re-check after drain                                     // line 971

TearDown() per-test:
  check_fabric_heartbeats_advanced(mesh, fabric_heartbeat_before_) // line 983
```

The key pattern: **snapshot before test, compare after test**. This catches routers that stalled during the test (heartbeats stopped advancing despite prior activity).

### 2.3 What `read_fabric_telemetry` Actually Does

**File**: `tt_metal/fabric/fabric_telemetry_reader.cpp:86-113`

Per device:
1. `cluster.l1_barrier(physical_chip_id)` — one L1 barrier per device
2. For each active ETH channel:
   - `cluster.read_from_device(buffer, chip_id, eth_core, telemetry_addr, telemetry_size)` — one DMA read per channel
   - `telemetry_size` = 128 bytes (QA) or 160 bytes (WH/BH) — reads the full `FabricTelemetry` struct from `MEM_AERISC_FABRIC_TELEMETRY_BASE`
3. Unpacks: `router_state`, `tx_heartbeat`, `rx_heartbeat`, bandwidth counters

**Cost on T3K (8 devices, ~4 active ETH channels each)**: 8 L1 barriers + 32 DMA reads per `read_fabric_telemetry` call. Two calls per test (SetUp + TearDown) = ~64 DMA reads per test.

---

## 3. Hardware/Firmware Register Mapping

### 3.1 Registers Used by Current Health Checks

```
Register/Field                    Address Source                              Used By              Read Method
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
FW heartbeat (WH)                 0x1F80 (test_results[48])                  FIX TV, FIX AR       cluster_.read_reg() [PCIe MMIO]
FW heartbeat (BH/QA)              0x7CC70 (MEM_SYSENG_ETH_HEARTBEAT)         FIX TV, FIX AR       cluster_.read_reg() [PCIe MMIO]
EDMStatus (edm_status_address)    FabricBuilderContext::get_router_sync_addr  Phase 5b, verify_*   ReadFromDeviceL1() [MMIO or relay]
RouterState                       HalL1MemAddrType::ROUTER_STATE             FabricCommandIntf    cluster.read_core()
FabricTelemetry (full struct)     MEM_AERISC_FABRIC_TELEMETRY_BASE           read_fabric_telem.   cluster.read_from_device() [DMA]
  - .dynamic_info.erisc[i].router_state     (offset within struct)           batch-t3k health     (unpacked from DMA buffer)
  - .dynamic_info.erisc[i].tx_heartbeat     (offset within struct)           batch-t3k heartbeat  (unpacked from DMA buffer)
  - .dynamic_info.erisc[i].rx_heartbeat     (offset within struct)           batch-t3k heartbeat  (unpacked from DMA buffer)
  - .dynamic_info.tx_bandwidth.*            (offset within struct)           batch-t3k (optional)  (unpacked from DMA buffer)
```

### 3.2 Key Distinction: Two Different "Heartbeat" Concepts

1. **Firmware mailbox heartbeat** (0x1F80 / 0x7CC70): Written by base-UMD relay firmware. Static `0xABCDxxxx` marker (UMD) or incrementing counter (fabric firmware). Read via PCIe MMIO `read_reg()`. Used by FIX TV/AR to detect boot completion.

2. **Telemetry heartbeat** (inside `MEM_AERISC_FABRIC_TELEMETRY_BASE`): Written by fabric ERISC router firmware's `update_telemetry()` function (`fabric_erisc_router.cpp:1514-1547`). `tx_heartbeat.full++` on every TX packet, `rx_heartbeat.full++` on every RX packet. Read via DMA `read_from_device()`. Used by batch-t3k's `capture_fabric_heartbeats()` for stall detection.

These are **different registers at different addresses** serving different purposes.

---

## 4. Batched API Design

### 4.1 What "Batched" Would Mean

The batch-t3k `capture_fabric_heartbeats()` already does a per-device batch: it calls `read_fabric_telemetry(fabric_node_id)` which internally iterates all channels and issues `l1_barrier` once per device. There is no single MMIO transaction that reads all channels' telemetry at once — the hardware does not support it. Each channel's telemetry lives in that channel's ERISC L1, which is a separate core with its own L1 memory.

A true "batched read" would require one of:
- **Firmware aggregation**: ERISC firmware copies heartbeats to a shared L1 region readable in one DMA transaction. Does not exist.
- **Host-side scatter-gather**: UMD `read_from_device_batch()` that issues multiple reads in one PCIe transaction. Does not exist.
- **NOC multicast read**: A NOC read that reads from multiple cores in one operation. Not supported by the TT NOC architecture for L1 reads (multicast is write-only).

### 4.2 Proposed API (If We Were to Add It)

Given hardware constraints, the best available optimization is a **software-level batch** that amortizes L1 barriers and reduces function-call overhead:

```cpp
namespace fabric_health_detail {

struct FabricHealthSnapshot {
    struct ChannelState {
        uint32_t eth_chan_id;
        FabricTelemetryRouterState router_state;
        uint64_t tx_heartbeat;
        uint64_t rx_heartbeat;
    };
    // Key: physical chip ID
    std::map<tt::ChipId, std::vector<ChannelState>> per_chip;
    std::chrono::steady_clock::time_point timestamp;
};

// Capture telemetry heartbeats and router states in one pass.
// Cost: 1 l1_barrier + N DMA reads per device (N = active channels).
// T3K: 8 barriers + ~32 DMA reads total.
FabricHealthSnapshot capture_fabric_health(
    const std::shared_ptr<MeshDevice>& mesh);

// Compare two snapshots. Returns:
//   - list of channels where router_state != Active
//   - list of channels where heartbeats were non-zero but stalled
struct FabricHealthDelta {
    struct Issue {
        tt::ChipId chip_id;
        uint32_t eth_chan_id;
        std::string description;
    };
    std::vector<Issue> unhealthy_routers;
    std::vector<Issue> stalled_heartbeats;
    bool is_healthy() const {
        return unhealthy_routers.empty() && stalled_heartbeats.empty();
    }
};

FabricHealthDelta compare_fabric_health(
    const FabricHealthSnapshot& before,
    const FabricHealthSnapshot& after);

}  // namespace fabric_health_detail
```

### 4.3 Performance Characteristics

```
Operation                  Current (racecondition-hunt)    Batched (proposed)
────────────────────────────────────────────────────────────────────────────
FIX TV init poll           N_mmio_cores * M_iterations     Same (cannot batch PCIe reads)
                           * 1 read_reg (4B each)
FIX AR teardown poll       N_mmio_cores * M_iterations     Same (cannot batch PCIe reads)
                           * 1 read_reg (4B each)
Phase 5b health check      N_channels * 1 ReadFromDevL1   1 l1_barrier + N DMA reads
                           (4B each, serial)               (128-160B each, serial)
verify_all_channels        N_devices * N_channels          Same as Phase 5b
                           * 3 retries * 1 ReadFromDevL1
Per-test telemetry         Does not exist                  2 * (8 barriers + 32 DMA reads)
(if added)                                                 = ~64 DMA reads per test
```

For FIX TV and FIX AR, batching is **not applicable** — they use `cluster_.read_reg()` (PCIe MMIO, 4 bytes) specifically because the ERISC may be in ROM boot and unable to service L1 reads. The telemetry DMA path requires the ERISC to be running fabric firmware.

For Phase 5b and verify_all_channels, replacing the per-channel `ReadFromDeviceL1(4 bytes)` calls with `read_from_device(128-160 bytes)` telemetry reads would actually be **slower** — more bytes per read, and the telemetry struct requires the ERISC to have progressed past init (which is exactly what these checks verify).

---

## 5. Which Health Checks Could Use Batched Reads vs Per-Core Granularity

### 5.1 Checks That REQUIRE Per-Core Granularity

**FIX TV** (risc_firmware_initializer.cpp:237-333):
- Needs to detect whether each individual MMIO ETH core has rebooted after PCIe force-reset
- Must distinguish `0x0` (not yet booted) from `0xABCDxxxx` (UMD static marker) from incrementing counter
- Cannot use telemetry API — ERISC may be in ROM boot (pre-firmware)
- **Verdict**: Per-core required. Cannot batch.

**FIX AR** (risc_firmware_initializer.cpp:534-619):
- Same pattern as FIX TV but in teardown
- `ac_heartbeat_any_ready` gate controls FIX AY execution
- **Verdict**: Per-core required. Cannot batch.

**Phase 5b** (device.cpp:2405-2755):
- Reads `EDMStatus` at `edm_status_address` — this is a specific L1 address, not the telemetry struct
- Needs to classify each channel's exact `EDMStatus` value for diagnostics (STARTED, LOCAL_HANDSHAKE_COMPLETE, READY_FOR_TRAFFIC, corrupt, etc.)
- Different code paths for different status values (FIX AK, FIX AK-2, FIX AM)
- **Verdict**: Per-channel required. Could theoretically read telemetry instead of EDMStatus, but router_state in telemetry (Active/Standby/Paused/Draining) has **lower resolution** than EDMStatus (STARTED/REMOTE_HANDSHAKE_COMPLETE/LOCAL_HANDSHAKE_COMPLETE/READY_FOR_TRAFFIC/TERMINATED). The Phase 5b diagnostics depend on the EDMStatus distinctions.

**verify_all_fabric_channels_healthy** (fabric_firmware_initializer.cpp:2372+):
- Same EDMStatus reads as Phase 5b
- **Verdict**: Same — per-channel EDMStatus required.

### 5.2 Checks That COULD Work with Aggregated/Batched Data

**Fixture-level fabric state logging** (multi_device_fixture.hpp:212-224, 266-278):
- Already uses device-level flags (no hardware reads)
- A batched telemetry snapshot would give richer data (per-channel router state, heartbeats) at negligible additional cost since it's done once per SetUp/TearDown
- **Verdict**: Could benefit, but the current approach (flag checks) is already efficient.

**A hypothetical per-test health check** (does not exist today):
- If racecondition-hunt were to add before/after test health comparison
- Would benefit from the batch-t3k pattern: `capture_fabric_heartbeats()` in SetUp, `check_fabric_heartbeats_advanced()` in TearDown
- **Verdict**: This is where batched reads would add value — but only if the branch adopts a shared-fixture model.

### 5.3 Summary Matrix

```
Health Check                  Per-Core    Telemetry    Could Use    Would Help?
                              Required?   API Usable?  Batch?
─────────────────────────────────────────────────────────────────────────────
FIX TV (init heartbeat)       YES         NO (ROM)     NO           N/A
FIX AR (teardown heartbeat)   YES         NO (ROM)     NO           N/A
Phase 5b (quiesce health)     YES         Partial*     NO           NO
verify_all_channels           YES         Partial*     NO           NO
FabricCommandIntf read_state  YES         YES          YES          Minimal
Fixture flag logging          NO          YES          YES          Minimal
Per-test heartbeat snapshot   NO          YES          YES          YES — but
                                                                    doesn't exist
```

*Partial: telemetry provides `router_state` (4 values) but not `EDMStatus` (10+ values). The coarser resolution loses diagnostic power.

---

## 6. Recommendation

### 6.1 Should `capture_fabric_heartbeats` Be Added to racecondition-hunt?

**No, not in the current branch.** The reasons:

1. **Architecture mismatch**: Racecondition-hunt uses a per-test open/close model. Every test pays full `SetFabricConfig -> MeshDevice::create -> MeshDevice::close -> SetFabricConfig(DISABLED)`. There is no shared mesh across tests, so there is no persistent fabric state to snapshot between tests.

2. **All existing health checks require per-core granularity**: FIX TV and FIX AR operate at the PCIe register level on potentially-rebooting ERISCs. Phase 5b and verify_all_channels read EDMStatus, not telemetry heartbeats. None of these can be replaced by the telemetry-based batched approach.

3. **No detection gap**: The current health checks catch the failure modes that matter for race-condition testing:
   - FIX TV/AR: detect whether ERISCs are running after PCIe reset
   - Phase 5b: detect whether channels completed handshake after quiesce
   - verify_all_channels: detect corrupt L1 at init time
   - Fixture flags: detect relay-broken, channels-not-ready, stale-base-UMD

   A heartbeat stall detector (batch-t3k's Mitigation 4) catches routers that are alive but stopped processing traffic — a failure mode that requires a **shared fixture** where the router was active before the test and should still be active after. In the per-test open/close model, the router's lifetime exactly matches the test's, so "stall during test" is not a meaningful concept.

4. **Performance non-issue**: The existing per-core reads in FIX TV (3s budget), FIX AR (5s budget), and Phase 5b (2s budget) are already optimized with parallel polling and early-exit. These are init/teardown costs, not per-operation costs. Batching would not reduce them because the hardware does not support multi-core reads in one transaction.

### 6.2 When Would It Be Useful?

`capture_fabric_heartbeats` becomes useful if/when:

1. **racecondition-hunt adopts a shared fixture model** (e.g., if A.2.2 `EnsureInitialized/ReleaseIfUnused` lands and the mesh stays open across tests). Then per-test heartbeat snapshots would detect fabric degradation between tests without full reinit.

2. **A new failure mode emerges** where a router stalls mid-test (alive but not processing traffic) and the test completes without detecting the stall. This would manifest as "test N passes but test N+1 hangs because router is stalled from N." In the per-test open/close model, this is impossible because the router is torn down between tests.

3. **Integration with the `analyze_fabric_hang_log.sh` script** — adding a post-test telemetry dump (router states, heartbeat counts, bandwidth counters) to the log output would give the analysis script richer data for post-mortem diagnosis. This is independent of batching and could be done with a simple telemetry dump in TearDown.

### 6.3 What WOULD Help racecondition-hunt's Health Monitoring

Instead of batched heartbeats, the branch would benefit from:

1. **Telemetry dump on failure**: When a test fails (or when `fabric_broken == true` in TearDown), dump the full `FabricTelemetrySnapshot` for all devices. This gives per-channel `router_state`, `tx_heartbeat`, `rx_heartbeat`, and bandwidth counters — useful for post-mortem even though it's not a pre/post comparison. Cost: one `read_fabric_telemetry` per device (8 devices, ~4 channels each = ~32 DMA reads). Only on failure paths, so zero cost on happy path.

2. **EDMStatus snapshot in SetUp**: Before the test body runs, capture each channel's `EDMStatus` (already read by verify_all_channels). If a test fails, compare against TearDown's Phase 5b status. This is effectively what Phase 5b already does but with explicit "expected healthy baseline" comparison. Could catch cases where a channel degrades from READY_FOR_TRAFFIC to a corrupt state during the test.

3. **Structured health state in analyze_fabric_hang_log.sh**: The script already parses FIX counters and timeline events. Adding telemetry data (router states, heartbeat deltas) to the structured log format would allow automated detection of "router was Active but stalled" patterns.

### 6.4 Revisiting the Prior Eval's Conclusion

The prior Addendum A eval (addendum-a-A2_4_5-eval.md) concluded A.2.4 was N/A because:
1. "`capture_fabric_heartbeats` doesn't exist in this branch" — **Correct**
2. "fixes need per-core granularity" — **Partially correct**: FIX TV, FIX AR, Phase 5b, and verify_all_channels need per-core/per-channel granularity. However, the eval conflated "all health checks" with "heartbeat batching." The correct framing is: **no health check in racecondition-hunt could be replaced by batched heartbeat reads, AND the branch has no use case for the batch-t3k-style per-test heartbeat comparison.**

The nuance the prior eval missed: even if a batched read were available, it would not replace any existing check. The question is not "can we batch what we already read?" but "is there a new check (stall detection) that a batched approach would enable?" The answer is no — not in a per-test open/close model.

---

## Appendix: Register Quick Reference

```
Register                          Address               Size    Access         What It Contains
────────────────────────────────────────────────────────────────────────────────────────────────
FW Heartbeat (WH)                 0x1F80                4B      PCIe MMIO      UMD: 0xABCDxxxx static; Fabric: incrementing
FW Heartbeat (BH/QA)              0x7CC70               4B      PCIe MMIO      Same semantics as WH
EDMStatus                         edm_status_address*   4B      L1 (relay ok)  STARTED/HANDSHAKE/READY/TERMINATED/etc.
RouterState (hostdev)             ROUTER_STATE offset   4B      L1 read_core   INITIALIZING/RUNNING/PAUSED/DRAINING/RETRAINING
FabricTelemetry (full struct)     FABRIC_TELEMETRY_BASE 128-160B DMA read      Static + Dynamic (router_state, heartbeats, BW)

* edm_status_address is computed by FabricBuilderContext — varies per config
```

```
T3K Device Count:  8 chips (2 MMIO, 6 non-MMIO)
Active ETH channels per chip:  ~4 (varies by topology)
Total channels:  ~32
FabricTelemetry struct:  2 erisc entries per channel
```
