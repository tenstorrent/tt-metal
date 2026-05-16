<!--
SUMMARY: RCA of t3000-apc-fast-tests failure in CI run 25137442210 — topology_mapper TT_FATAL after FIX AX teardown
KEYWORDS: RCA, topology_mapper, chip_topology_mapping_, FIX AX, FIX PL, FIX M2, ASIC missing, dispatch timeout, fabric hang, dead_relay_devices_, relay_broken, GAP-31
SOURCE: CI log /workspace/group/ci_data/t3k_fast_failure_run25137442210.txt, branch nsexton/0-racecondition-hunt
SCOPE: Single CI run failure, t3000-apc-fast-tests, Fabric2DFixture and T3kCustomMeshGraphFabric2DFixture
USE WHEN: Investigating chip_topology_mapping_ TT_FATAL or ASIC missing errors post-FIX AX teardown
-->

# RCA: t3000-apc-fast-tests failure — ASIC 14521749668 not found in chip_topology_mapping_

**CI run**: 25137442210, job 73680628895
**Runner**: `tt-ubuntu-2204-n300-llmbox-stable-6kds8-runner-9jspk` (aus2 group)
**Branch**: `nsexton/0-racecondition-hunt` at `49def231906`
**Date**: 2026-04-29 22:55–22:57 UTC

---

## Executive Summary

1. **Pre-existing hardware/firmware corruption**: The runner had ALL 8 devices with corrupt ERISC L1 (`0x49705180`) BEFORE the first test ran. 28 channels across 8 devices were marked as `probe_dead`. This is NOT caused by our branch — it is stale state from a prior process.

2. **Primary failure (dispatch timeout)**: Fabric2DFixture tests dispatched work to devices 4/5 (non-MMIO) which had dead dispatch kernels (skipped by FIX E2). The `completion_queue_wait_front` timed out after 5 seconds on each device because dispatch was never initialized.

3. **Secondary failure (ASIC missing from topology_mapper)**: After FIX AX skipped `assert_risc_reset_at_core` for 12 non-MMIO channels (devices 5/6/7), those ERISCs remained with dead firmware. When the next fixture (T3kCustomMeshGraphFabric2DFixture) created a new `TopologyDiscovery`, UMD's `FIX W` skipped all gateway ETH cores with `heartbeat=0x0`. Three N300 boards reported only 1/2 chips. The `PhysicalSystemDescriptor` was built from 5 ASICs instead of 8. `TopologyMapper` then tried to map 8 fabric nodes to 5 ASICs and hit `TT_FATAL` at line 292.

4. **Root cause of the gap**: FIX AY (deferred non-MMIO ERISC reset) was designed to handle this exact scenario but **never fired**. FIX AY is triggered by `relay_broken_non_mmio`, which is populated from `dev->is_fabric_relay_path_broken()` (line 290) and `dev->is_fabric_channels_not_ready_for_traffic()` (FIX BA, line 319). In this run, NEITHER flag was set — the non-MMIO devices were classified as `dead_relay_devices_` via `probe_dead_channels`, which is a third, uncovered codepath.

5. **Verdict**: **(c) Both** — pre-existing hardware flakiness exposed a gap in our branch's teardown logic. The initial corruption is pre-existing. The failure to recover between fixtures is a regression introduced by FIX AX (which correctly prevents hangs but lacks a compensating reset for the `dead_relay_devices_` codepath).

---

## Detailed Analysis

### Phase 1: System state at init (22:55:02–22:55:12)

TopologyMapper successfully finds 8 chips and maps a 2x4 mesh at 22:55:02.

`terminate_stale_erisc_routers` then examines all ETH channels and finds widespread corruption:

```
Device 0: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
Device 1: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
Device 2: corrupt=2 canary=0 stale_running=0 probe_dead=2 base_umd=0
Device 3: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
Device 4: corrupt=2 canary=0 stale_running=0 probe_dead=2 base_umd=0
Device 5: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
Device 6: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
Device 7: corrupt=4 canary=0 stale_running=0 probe_dead=4 base_umd=0
```

All corrupt values are `0x49705180` — NOT a valid EDMStatus, NOT the base-UMD sentinel `0x49706550`. This indicates ERISC L1 was corrupted by a prior process crash.

FIX E2 marks devices 4/5/6/7 as `dead_relay_devices_` (all non-MMIO). Note: `relay_broken=false` for all devices — this will be critical later.

Dispatch kernel init is skipped for devices 4/5/6/7 (dead relay). FIX AM skips `verify_all_fabric_channels_healthy()`.

### Phase 2: Fabric2DFixture.TestUnicastRaw dispatch timeout (22:55:12–22:56:17)

The test attempts to run a unicast between FabricNodeId 4 and 5 (both non-MMIO, both dead-relay). At 22:55:14, `completion_queue_wait_front device=3` starts. Each device times out after 5 seconds:

```
22:55:14 → 22:55:19  device=3  TIMEOUT (5s)
22:55:50             device=0  TIMEOUT
22:55:55             device=0  TIMEOUT (another read)
22:56:00             device=1  TIMEOUT
22:56:07             device=7  TIMEOUT
22:56:12             device=5  TIMEOUT
22:56:17             device=6  TIMEOUT
```

The test fails. The fixture teardown begins.

### Phase 3: First fixture teardown — FIX AX fires (22:56:17–22:56:38)

`close_devices` calls `FabricFirmwareInitializer::teardown()` at 22:56:18.709.

FIX AX fires for devices 5, 6, 7 (non-MMIO, relay confirmed dead): skips `assert_risc_reset_at_core` for channels 0, 1, 6, 7 on each. 12 channels total NOT reset.

```
22:56:28.722  Device 5 chan={0,1,6,7} — FIX AX skip
22:56:33.725  Device 6 chan={0,1,6,7} — FIX AX skip
22:56:38.728  Device 7 chan={0,1,6,7} — FIX AX skip
```

Note the 5-second gaps — these are from the diagnostic reads on the FIRST channel of each device timing out. FIX AX then skips subsequent channels on the same device.

FIX AK also fires: skips l1_barrier relay drain for all non-MMIO devices.

`FabricFirmwareInitializer::teardown()` returns at 22:56:38.728. Remaining teardown steps complete in ~6ms. Devices 0-7 are closed by 22:56:38.734.

**Critical gap**: `RiscFirmwareInitializer::teardown()` runs per-device during device close (not as a separate DeviceManager step). It checks `dev->is_fabric_relay_path_broken()` and `dev->is_fabric_channels_not_ready_for_traffic()` to populate `relay_broken_non_mmio`. In this run:
- `fabric_relay_path_broken_ = false` for all devices (relay was never established, not "broken")
- `fabric_channels_not_ready_for_traffic_ = false` (FIX AM set this only for the STARTED early-exit case, not for probe_dead)
- Therefore `relay_broken_non_mmio` is empty
- **FIX AC (MMIO ETH PCIe reset) does NOT fire**
- **FIX AY (deferred non-MMIO ERISC reset) does NOT fire**

### Phase 4: Second fixture hits ASIC missing (22:56:38.872)

T3kCustomMeshGraphFabric2DFixture starts at 22:56:38.872 — 138ms after device close.

`TopologyDiscovery::discover_remote_devices()` runs. UMD polls ETH gateway heartbeats:

```
ASIC 10226782225: 4 gateways dead (heartbeat=0x0)  → MMIO chip, 2 gateway pairs dead
ASIC 10226782297: 2 gateways dead
ASIC 10226782337: 4 gateways dead
ASIC 10226782372: 4 gateways dead
ASIC 14521749521: 2 gateways dead
```

With these many dead gateways, UMD cannot discover 3 of the 4 non-MMIO chips. Three boards report 1/2 chips:

```
Board 0x100014611905059: 1 chip (expected 2)
Board 0x100014611905081: 1 chip (expected 2)
Board 0x1000146119050a4: 1 chip (expected 2)
```

`PhysicalSystemDescriptor` is built from only 5 discovered ASICs. `TopologyMapper::initialize_chip_topology_mapping_map()` creates 5 entries in `chip_topology_mapping_` and `asic_id_to_mapping_`.

The provided `logical_mesh_chip_id_to_physical_chip_id_mapping` (from the custom MGD) has 8 entries. When iterating over them, physical_chip_id for the missing non-MMIO chips maps to unique_id 14521749668 (one of the missing ASICs). The lookup `asic_id_to_mapping_.find(asic_id)` fails → `TT_FATAL` at topology_mapper.cpp:292.

The topology solver then also fails: "Graph specified in MGD could not fit in the discovered physical topology" (line 4531).

---

## Branch Contribution Analysis

### FIX AX — **Direct contributor (but correct design, incomplete coverage)**

FIX AX correctly avoids the 5s-per-channel `assert_risc_reset_at_core` hang on non-MMIO devices with dead relays. Without FIX AX, teardown would have spent 12 × 5s = 60s of serial timeouts — enough to blow the CI step timeout (the old failure mode).

However, FIX AX leaves non-MMIO ERISCs un-reset. The compensating mechanism (FIX AY) was designed to handle this, but FIX AY is gated on `relay_broken_non_mmio`, which is NOT populated when devices are in `dead_relay_devices_` via the `probe_dead_channels` path.

**Contribution**: Direct. FIX AX skipping the reset + FIX AY not firing = non-MMIO ERISCs left with dead firmware → UMD can't rediscover them → ASIC missing.

### FIX PL — **Not a contributor**

FIX PL guards `l1_barrier` and `dram_barrier` in `clear_l1_state`/`clear_dram_state` against dead ERISC relays on non-MMIO chips. It catches timeout exceptions and continues with a warning.

Analysis of what `clear_l1_state` does (lines 989-1055):
- Writes zero vectors to all Tensix L1, ETH L1, and DRAM L1
- Then calls `l1_barrier` to flush the writes

If the `l1_barrier` times out and is caught by FIX PL, the zero writes may not have flushed. However, this only affects the CURRENT session's init — the writes were issued, just not confirmed. Since the init continued and fabric was configured in degraded mode, any unflushed zeros would at worst leave stale data in L1 that the test would overwrite anyway.

**Contribution**: None to this failure. FIX PL is purely protective — it prevents a throw during init, not a state corruption.

### FIX M2 — **Not a contributor**

FIX M2 removes channels from `base_umd_channels_map` when the peer non-MMIO device is in `dead_relay_devices_`. This allows `configure_fabric_cores()` to hard soft-reset those MMIO-side channels instead of skipping them (they have nothing to serve).

False positive risk: FIX M2 checks `dead_relay_devices_.count(peer_chip_id) > 0`. This set is only populated by `FIX E2` when `probe_dead_channels >= threshold`. In this run, all non-MMIO devices genuinely have corrupt firmware (all channels corrupt). There is no false positive.

**Contribution**: None. FIX M2 operates on MMIO-side channels (allowing reset of the relay ERISC on the MMIO device whose peer is dead). It does not affect non-MMIO chip discovery.

### FIX AY — **Absent (should have compensated for FIX AX but didn't fire)**

FIX AY is in `RiscFirmwareInitializer::teardown()` (line 556). It is gated on `relay_broken_non_mmio` being non-empty. This set is populated from:
1. `dev->is_fabric_relay_path_broken()` — checks `fabric_relay_path_broken_` (line 290)
2. `dev->is_fabric_channels_not_ready_for_traffic()` — FIX BA addition (line 319)

In this run, `relay_broken=false` because the relay was never established (firmware was corrupt from the start — `probe_dead_channels` path, not `relay_broken` path). `fabric_channels_not_ready_for_traffic_` was also not set because FIX AM only sets it for the STARTED early-exit case.

**This is the gap**: `dead_relay_devices_` (populated via probe_dead_channels) is a third entry path to "non-MMIO devices need cleanup" that is NOT covered by FIX AY's precondition check.

---

## Verdict

**(c) Both**: pre-existing hardware flakiness (widespread ERISC L1 corruption from a prior process crash) exposed a gap in the branch's teardown logic.

- The initial corruption (`0x49705180` in all ETH channels) is pre-existing — our branch did not cause it.
- The dispatch timeouts in the first fixture are a consequence of degraded-mode operation (expected when all non-MMIO relays are dead).
- The ASIC-missing TT_FATAL in the second fixture is a **regression introduced by our branch** via the interaction between FIX AX (skips non-MMIO reset) and FIX AY (supposed to compensate, but gated on `relay_broken_non_mmio` which is empty for the `dead_relay_devices_` codepath).

On `main`, without FIX AX, teardown would have attempted `assert_risc_reset_at_core` on all non-MMIO channels. It would either succeed (resetting the ERISCs) or hang for 60+ seconds and get killed by the CI timeout. Either way, the second fixture's topology_mapper would not have encountered this specific ASIC-missing error.

---

## Recommended Fix

**Extend FIX AY's precondition** to also trigger when `dead_relay_devices_` from `FabricFirmwareInitializer` contains non-MMIO devices with probe_dead channels.

Specifically, in `RiscFirmwareInitializer::teardown()` (around line 290), add a third condition:

```
// Existing conditions:
// 1. dev->is_fabric_relay_path_broken()
// 2. dev->is_fabric_channels_not_ready_for_traffic() [FIX BA]
//
// NEW condition needed:
// 3. dev was in dead_relay_devices_ (probe_dead path — firmware
//    never loaded, relay never established, but ERISCs still have
//    stale/corrupt firmware that blocks UMD discovery)
```

This requires propagating `dead_relay_devices_` information from `FabricFirmwareInitializer` to `RiscFirmwareInitializer`. Options:
- Set `fabric_relay_path_broken_` on the Device when it enters `dead_relay_devices_` (simplest — reuses existing flag, but changes its semantics)
- Add a new per-device flag `fabric_dead_relay_` and check it in FIX AY's precondition
- Pass `dead_relay_devices_` from FabricFirmwareInitializer to RiscFirmwareInitializer via the DeviceManager

The simplest fix is to set `dev->set_fabric_relay_path_broken(true)` when marking a device as dead-relay in `compile_and_configure_fabric` (around line 1595 of `fabric_firmware_initializer.cpp`). This makes FIX AY (and FIX AC) fire for these devices during RiscFirmwareInitializer::teardown.

---

## Timeline

```
22:55:02  TopologyMapper: 8 chips, 2x4 mesh — successful init
22:55:09  terminate_stale_erisc_routers: ALL channels corrupt (0x49705180)
22:55:09  FIX E2: devices 4,5,6,7 → dead_relay_devices_ (relay_broken=false)
22:55:09  configure_fabric: all 8 devices in degraded mode (2-4 dead channels each)
22:55:09  Dispatch init skipped for devices 4,5,6,7
22:55:12  Fabric2DFixture.TestUnicastRaw starts
22:55:14  completion_queue_wait_front device=3 — starts 5s timeout cascade
22:56:17  All dispatch timeouts complete; fixture teardown begins
22:56:18  FabricFirmwareInitializer::teardown() starts
22:56:28  FIX AX: Device 5 channels 0,1,6,7 — skip assert_risc_reset
22:56:33  FIX AX: Device 6 channels 0,1,6,7 — skip assert_risc_reset
22:56:38  FIX AX: Device 7 channels 0,1,6,7 — skip assert_risc_reset
22:56:38  FIX AK: skip l1_barrier for all non-MMIO
22:56:38  FabricFirmwareInitializer::teardown() returns
22:56:38  Device close (0-7) — FIX AY does NOT fire (relay_broken_non_mmio empty)
22:56:38  T3kCustomMeshGraphFabric2DFixture starts
22:56:38  TopologyDiscovery: 16 dead gateways across 5 MMIO ASICs
22:56:38  UMD: 3 boards show 1/2 chips (3 non-MMIO chips invisible)
22:56:38  TopologyMapper: ASIC 14521749668 not found → TT_FATAL
22:56:38  "Graph specified in MGD could not fit" → process exits rc=1
```
