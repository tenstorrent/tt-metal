<!--
SUMMARY: Deep dive root-cause analysis of quiesce_devices() hang caused by contaminated ETH channel states after prior test ~Cluster timeout
KEYWORDS: quiesce_devices, AllGather, hang, ETH, fabric, base-UMD, skip_soft_reset, FIX M, contamination, teardown, ERISC, EDMStatus
SOURCE: Code analysis of nsexton/0-racecondition-hunt-fix-ae branch in tt-metal
SCOPE: Full contamination path from ~Cluster timeout through quiesce_devices hang, with remediation proposals
USE WHEN: Investigating fabric hangs, AllGather timeouts, quiesce_devices deadlocks, or ETH channel corruption across test boundaries
-->

# quiesce_devices() Hang Analysis: Contaminated ETH Channel States

## 1. Architecture Overview

### 1.1 quiesce_devices() Flow

`quiesce_devices()` is the mechanism to restart fabric workers between operations (e.g., before/after AllGather). It lives in `MeshDeviceImpl::quiesce_devices()` → `quiesce_internal()`:

```
File: tt_metal/distributed/mesh_device.cpp:1690-1700
```

`quiesce_internal()` (mesh_device.cpp:1571-1688) executes in two major phases:

**Phase 1 (three sub-passes):**
- Pass 1a: All devices — Phase 2.5 (terminate ERISCs) + Phase 3 setup (configure_fabric_cores, runtime args, l1_barrier, WORKER write_launch_msg). ETH write_launch_msg deferred.
- Pass 1b: MMIO devices only — ETH write_launch_msg (fast, direct PCIe).
- Pass 1c: Non-MMIO devices only — ETH write_launch_msg via UMD relay, one device at a time. After each device, polls until all channels show EDMStatus::STARTED before launching next device (FIX AF).

**Phase 2 (handshake wait):**
- For each device, calls `wait_for_fabric_workers_ready()` which:
  - Phase 4: Polls each MUX core for READY_FOR_TRAFFIC (5s timeout)
  - Phase 5: Polls master ERISC channel for LOCAL_HANDSHAKE_COMPLETE (0xA2B2C2D2) with 10s timeout, then writes READY_FOR_TRAFFIC
  - Phase 5b: Per-channel health check confirms all channels are healthy

### 1.2 EDMStatus Values

```
File: tt_metal/fabric/fabric_edm_packet_header.hpp:48-84

STARTED                     = 0xA0B0C0D0  (ERISC entered kernel_main)
REMOTE_HANDSHAKE_COMPLETE   = 0xA1B1C1D1
LOCAL_HANDSHAKE_COMPLETE    = 0xA2B2C2D2  (the "RUNNING canary")
READY_FOR_TRAFFIC           = 0xA3B3C3D3
TERMINATED                  = 0xA4B4C4D4
```

### 1.3 Non-EDMStatus Sentinels

```
File: tt_metal/impl/device/edm_status_utils.hpp:28-55

BASE_UMD_FIRMWARE_SENTINEL  = 0x49706550  (base relay firmware, ASCII "iPeP")
HOST_PRE_LAUNCH_CANARY      = 0xDEADB07E  (host wrote canary before launch msg)
```

### 1.4 Observed Corrupt Values at Hang

```
Device 4 chan 0,1,6,7:  status=0x00000000   (dead/zeroed — firmware never started)
Device 6 chan 0,1,6,7:  status=0x40404040   (NOT a valid EDMStatus or sentinel)
Device 1,3 inter-chip:  status=0x3f803f80 / 0x40004000  (NOT valid EDMStatus or sentinel)
```

These are NOT any of the defined EDMStatus values, NOT the base-UMD sentinel (0x49706550), NOT host canary (0xDEADB07E), and NOT the fabric kernel_main canary (0xA0A0A0A0). They are truly corrupt/transitional L1 values.


## 2. Root Cause Analysis

### 2.1 The Contamination Path (Step by Step)

**Step 1: Prior test's AllGather or CCL operation runs normally.**

AllGather programs ERISC fabric routers on all ETH channels. After completion, ERISCs are in READY_FOR_TRAFFIC state.

**Step 2: Prior test's teardown fails with timeout.**

The `~Cluster` destructor calls `teardown_fabric_config()` (metal_env.cpp:321). This function:
1. Sends TERMINATE to each ETH router
2. Polls for EDMStatus::TERMINATED (5s timeout per channel)
3. On timeout: force-resets the ERISC (assert + deassert with RiscType::ALL)

The `Timeout waiting for Ethernet core service remote IO request` error occurs when step 2 times out — the ERISC is in a state where it cannot respond to the UMD relay protocol.

**Step 3: Force-reset leaves L1 in transitional state.**

When `teardown_fabric_config()` times out and calls:
```cpp
cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ALL);
cluster.deassert_risc_reset_at_core(core_loc, tt::umd::RiscType::ALL);
```
(metal_env.cpp:414-415)

The ERISC0 (BRISC) is halted then immediately restarted. The BRISC re-enters the C-runtime startup, which begins zeroing .bss. However, the `edm_status_address` in L1 is NOT in .bss — it's in a firmware-specific L1 region. The force-reset restarts the ERISC into base UMD firmware, which writes `0x49706550` to its own status address once .bss init completes.

**BUT**: if the prior timeout was caused by ETH link issues (not just firmware unresponsiveness), the force-reset + deassert may not fully restore the ERISC. The values 0x40404040, 0x3f803f80, 0x40004000 represent:

- **Partial L1 initialization state**: The base UMD firmware's .bss init or the fabric firmware's L1 clear left partial/interrupted values. These are likely intermediate values from the `configure_fabric_cores()` L1 clear loop writing `router_zero_buf` (a buffer of zeros) — but the write was interrupted or only partially completed.
- **ETH stream register residuals**: The values 0x40004000 and 0x3f803f80 could be NOC/ETH stream register configuration residuals that leaked into the edm_status L1 address during an incomplete firmware transition.
- **Concurrent L1 access corruption**: When the ERISC is being force-reset while L1 writes are still in-flight from the prior firmware, the resulting L1 content is a mix of old and new values.

**Step 4: Next test's MeshDevice::create() → FabricFirmwareInitializer::init() runs.**

`terminate_stale_erisc_routers()` (fabric_firmware_initializer.cpp:1007) probes each channel:

For channels with 0x49706550 (base-UMD sentinel):
- Added to `base_umd_channels` set
- **NOT** added to `probe_dead_channels`
- No TERMINATE sent, no L1 zeroed — left untouched

For channels with corrupt values (0x40404040, 0x3f803f80, 0x40004000):
- Hit the `!known_status` branch (fabric_firmware_initializer.cpp:1152)
- NOT base-UMD sentinel, NOT canary values
- Falls into the "truly corrupt" case (line 1248)
- edm_status_address is zeroed to break cascade
- Added to `probe_dead_channels` for soft-reset by configure_fabric_cores

**Step 5: configure_fabric() passes base_umd_channels as skip_soft_reset_channels.**

In `Device::configure_fabric()` (device.cpp:404):
```cpp
const auto health = tt::tt_fabric::configure_fabric_cores(this, pre_dead_channels, skip_soft_reset_channels);
```

**FIX M** (fabric_init.cpp:172-187): For channels in `skip_soft_reset_channels` (base_umd_channels), the soft reset (assert_risc_reset_at_core) is skipped. Instead, `write_launch_msg_to_core` is used to transition the ERISC from base-UMD firmware to fabric firmware without halting.

**Step 6: FIX RZ sets fabric_stale_base_umd_channels_.**

After configure_fabric_cores, if `skip_soft_reset_channels` is non-empty on a non-MMIO device:
```cpp
// device.cpp:663-671
fabric_stale_base_umd_channels_ = true;
log_warning("configure_fabric: Device {} (non-MMIO) has {} base-UMD channel(s) transitioned "
            "via launch_msg (FIX M).  AllGather on this cluster may hang.");
```

**THIS IS THE WARNING THAT FIRES BEFORE THE HANG.**

**Step 7: quiesce_devices() called before AllGather.**

The test calls `mesh_device_->quiesce_devices()` (test line 357). This invokes `quiesce_internal()`:

Pass 1a calls `quiesce_and_restart_fabric_workers(defer_eth_launch=true)` for each device. Phase 2.5 tries to terminate the ERISC and Phase 3 reloads firmware. However:

- For channels that were base-UMD and got launch_msg'd in Step 5: the ERISC may have **failed to transition** from base-UMD to fabric firmware. The launch_msg was written but the ERISC was in a state where it couldn't process it (e.g., mid-.bss-init from the force-reset in Step 3). The channel's edm_status remains 0x0 or transitions to a non-standard value.

- For channels that had corrupt values (0x40404040 etc.) that were zeroed and soft-reset: these may have been successfully reset but their peer channels (on the other end of the ETH link) were NOT reset — the peer is still in the corrupt state from Step 3.

Pass 1b/1c launches ETH cores, but the corrupted peer channels cannot complete the ETH handshake.

Phase 2: `wait_for_fabric_workers_ready()` polls the master ERISC channel for `LOCAL_HANDSHAKE_COMPLETE (0xA2B2C2D2)` with a 10s timeout. The handshake requires BOTH endpoints to be running valid fabric firmware. If the peer is in corrupt state (0x40404040), it never responds → **HANG**.

### 2.2 Why the Hang is Infinite (Not Just a Timeout)

In Phase 5 (device.cpp:3020-3040), the poll loop reads from the master channel. For NON-MMIO devices, `ReadFromDeviceL1` routes through the UMD ETH relay on the MMIO device. If the relay ERISC on the MMIO device is alive but running fabric firmware (not UMD relay protocol), the read **HANGS** rather than throws — there is no per-read timeout for this case. The process-level `TT_METAL_OPERATION_TIMEOUT_SECONDS` is the only safety net.

For MMIO devices, the direct PCIe read does return, but the poll loops indefinitely since the status never reaches the expected value. The 10s timeout + FIX AL's 3s STARTED early-exit provide bounded behavior for MMIO — but non-MMIO can truly hang.

### 2.3 Summary: The Three Failure Modes

1. **Dead channels (status=0x0)**: ERISC never started. Phase 5 times out after 10s and sets `fabric_relay_path_broken_`. This is handled.

2. **Stuck at STARTED (0xA0B0C0D0)**: ERISC started but peer is not responding. FIX AL provides a 3s early-exit. This is handled.

3. **Peer in corrupt/transitional state (0x40404040, 0x3f803f80, 0x40004000)**: The local ERISC may reach STARTED but the peer never initiates the handshake. If the local device is non-MMIO and the relay read hangs, there is **no timeout** — only the process-level kill. **This is the unhandled case.**


## 3. The FIX M Warning — Detailed

### 3.1 Location and Condition

```
File: tt_metal/fabric/fabric_init.cpp:172-187
```

FIX M fires when `skip_soft_reset_channels.count(router_chan) > 0`. These are channels where `terminate_stale_erisc_routers()` found `edm_status == 0x49706550` (base-UMD firmware sentinel) and the host pre-launch canary was NOT found (confirming this is genuine live base-UMD, not a crashed-before-kernel-main state).

### 3.2 What "Transitioned via launch_msg" Means

Instead of the normal reset cycle:
```
assert_risc_reset → (ERISC halted) → write L1 → deassert_risc_reset → (ERISC boots fresh)
```

FIX M uses:
```
skip assert_risc_reset → write L1 (firmware data) → write_launch_msg_to_core → (ERISC transitions from base-UMD to fabric firmware via launch mailbox)
```

### 3.3 Why This Causes AllGather Hang

The launch_msg transition works when the base-UMD firmware is healthy and polling the launch mailbox. When the base-UMD firmware is in a degraded state (e.g., from a force-reset during teardown timeout), it may:
- Not be polling the mailbox (stuck in .bss init)
- Be polling but unable to properly initialize fabric firmware (L1 partially corrupt from prior session)
- Initialize fabric firmware but crash before reaching EDMStatus::STARTED

In all these cases, the channel never reaches RUNNING state, and AllGather/quiesce Phase 5 hangs.


## 4. Containment Failure Point

The containment fails at **two specific points**:

### 4.1 Failure Point 1: terminate_stale_erisc_routers (fabric_firmware_initializer.cpp:1194)

When a channel has `edm_status == 0x49706550` (base-UMD sentinel), it is added to `base_umd_channels` and passed through to `configure_fabric()` as `skip_soft_reset_channels`. The code assumes this means the relay is healthy and the ERISC can transition via launch_msg.

**The assumption is wrong when**: the prior test's teardown force-reset left the ERISC in a degraded base-UMD state where it writes `0x49706550` (the sentinel is in ROM/flash, not L1) but cannot actually process launch messages.

**There is no validation** that the base-UMD firmware is actually functional beyond the sentinel check. The code cannot distinguish "healthy base-UMD, ready for launch_msg" from "base-UMD booted after force-reset, .bss partially initialized, launch mailbox not yet polling".

### 4.2 Failure Point 2: quiesce_internal Phase 5 (device.cpp:3020)

Phase 5 polls for `LOCAL_HANDSHAKE_COMPLETE` without a mechanism to detect that the channel's PEER is in a corrupt/non-ERISC state. The handshake is a two-party protocol:
1. Sender writes MAGIC to receiver's L1 via eth_send_packet
2. Receiver responds

If the peer is in state 0x40404040, it never participates. The local ERISC can get stuck at STARTED forever. While FIX AL provides a 3s timeout for the STARTED case on MMIO devices, non-MMIO devices can still hang if the relay read itself hangs.


## 5. Remediation Proposals

### 5.1 Proposal A: Pre-launch Health Check for Base-UMD Channels (Recommended)

**Location**: `terminate_stale_erisc_routers()` (fabric_firmware_initializer.cpp), between the sentinel check (line 1190) and adding to `base_umd_channels` (line 1203).

**Change**: After detecting `edm_status == 0x49706550`, perform a **liveness probe** — write a test pattern to the launch mailbox and read it back. If the read-back matches, the relay path is functional and the ERISC can handle launch_msg. If the read-back fails or returns wrong data, treat the channel as corrupt (zero edm_status, add to probe_dead_channels for soft-reset).

```cpp
if (is_base_umd) {
    // NEW: Liveness probe — verify the relay path actually works
    // before trusting the base-UMD sentinel.
    bool relay_functional = false;
    try {
        // Write a probe value to a scratch L1 address and read it back
        const uint32_t probe_addr = router_sync_address + 4;  // adjacent scratch word
        std::vector<uint32_t> probe_write{0xCAFEBABE};
        detail::WriteToDeviceL1(dev, eth_logical_core, probe_addr, probe_write, CoreType::ETH);
        std::vector<uint32_t> probe_read(1, 0);
        detail::ReadFromDeviceL1(dev, eth_logical_core, probe_addr, 4, probe_read, CoreType::ETH);
        relay_functional = (probe_read[0] == 0xCAFEBABE);
    } catch (...) {
        relay_functional = false;
    }

    if (relay_functional) {
        base_umd_channels.insert(eth_chan_id);
        // ... existing log ...
    } else {
        // Base-UMD sentinel present but relay is not functional —
        // treat as corrupt and schedule for soft-reset.
        probe_dead_channels.insert(eth_chan_id);
        try {
            std::vector<uint32_t> zero_buf(1, 0);
            detail::WriteToDeviceL1(dev, eth_logical_core, router_sync_address, zero_buf, CoreType::ETH);
        } catch (...) {}
        log_warning(tt::LogMetal,
            "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x49706550 "
            "(base-UMD sentinel) but relay liveness probe FAILED — treating as corrupt, "
            "scheduling for soft-reset in configure_fabric_cores.",
            dev->id(), eth_chan_id);
    }
}
```

**Risk**: Low. The probe is a single write+read (~microseconds for MMIO, ~milliseconds for non-MMIO relay). If the probe itself hangs, it means the relay is truly dead, which is the same failure mode we'd see later in configure_fabric anyway — but at this point we have the relay timeout protections in terminate_stale_erisc_routers to bail out.

### 5.2 Proposal B: Pre-AllGather Channel Health Check in quiesce_internal

**Location**: `MeshDeviceImpl::quiesce_internal()` (mesh_device.cpp), after Pass 1c completes and before Pass 2 starts (between lines 1673 and 1679).

**Change**: After all ETH cores are launched (Pass 1c), but before waiting for handshake (Pass 2), probe each channel's edm_status. If any channel shows a non-valid-EDMStatus value (i.e., not one of the 0xA*B*C*D* or 0xB*C*D*E* postcodes), skip Phase 5 for that device and set `fabric_relay_path_broken_` to prevent AllGather from running.

```cpp
// NEW: Pass 1d — channel sanity check before handshake wait
log_info(tt::LogMetal, "quiesce_internal: Pass 1d — pre-handshake channel health check");
for (auto* idev : get_fabric_quiesce_ready_order(get_devices())) {
    auto* dev = dynamic_cast<Device*>(idev);
    if (dev && !dev->is_mmio_capable()) {
        // Check if any channel has a non-valid EDMStatus
        if (!dev->pre_handshake_eth_health_check()) {
            log_warning(tt::LogMetal,
                "quiesce_internal: Pass 1d — Device {} has contaminated ETH channels, "
                "setting relay_path_broken to prevent Phase 5 hang",
                dev->id());
            dev->set_fabric_relay_path_broken(true);
        }
    }
}
```

**Risk**: Low-medium. Adds a non-MMIO relay read per channel, which could itself hang if the relay is broken. Needs the same relay-timeout protections as the ENTRY snapshot. Could be bounded with a per-channel deadline.

### 5.3 Proposal C: Strengthen teardown_fabric_config to Zero edm_status After Force-Reset

**Location**: `MetalEnvImpl::teardown_fabric_config()` (metal_env.cpp:411-429), after the force-reset.

**Change**: After force-resetting a timed-out ERISC, wait briefly for base-UMD firmware to write the 0x49706550 sentinel, then write 0x0 to edm_status_address. This ensures the next session sees a clean 0x0 instead of stale/corrupt values.

```cpp
// After force-reset (existing lines 414-415):
cluster.assert_risc_reset_at_core(core_loc, tt::umd::RiscType::ALL);
cluster.deassert_risc_reset_at_core(core_loc, tt::umd::RiscType::ALL);

// NEW: Wait for base-UMD firmware to complete .bss init, then zero edm_status
// so the next session sees clean 0x0 instead of stale/corrupt values.
std::this_thread::sleep_for(std::chrono::milliseconds(50));  // .bss init takes ~2ms
try {
    std::vector<uint32_t> zero_status(1, 0);
    cluster.write_core(chip_id, eth_logical_core, edm_status_address,
                       zero_status.data(), sizeof(uint32_t));
} catch (...) {
    // Best effort — if the write fails, terminate_stale_erisc_routers
    // in the next session will handle it via the corrupt-value path.
}
```

**Risk**: Very low. This is a best-effort cleanup that only runs on the teardown timeout path (already degraded). The 50ms sleep is bounded and only fires when we already spent 5s on timeout.

### 5.4 Proposal D: Add Process-Level Timeout to Non-MMIO Phase 5 Relay Reads

**Location**: `Device::wait_for_fabric_workers_ready()` (device.cpp:3020), the Phase 5 poll loop.

**Change**: Instead of relying on the process-level `TT_METAL_OPERATION_TIMEOUT_SECONDS`, add a per-poll-iteration deadline that detects relay read hangs (read takes >6s when healthy reads take <1ms) and breaks out of the loop.

```cpp
// In the Phase 5 poll loop, wrap each ReadFromDeviceL1 with a time check:
auto read_start = std::chrono::steady_clock::now();
try {
    detail::ReadFromDeviceL1(this, master_logical_core, router_sync_addr, 4, sync_buf, CoreType::ETH);
} catch (const std::exception& e) {
    // ... existing catch handling ...
}
auto read_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - read_start).count();
if (read_elapsed > 6000) {
    // A single relay read took >6s — the relay is hanging, not throwing.
    fabric_relay_path_broken_ = true;
    log_warning(tt::LogMetal,
        "wait_for_fabric_workers_ready: Device {} Phase 5: relay read took {}ms "
        "(expected <1ms) — relay path is hanging (not throwing). Setting "
        "fabric_relay_path_broken_ to prevent further relay reads.",
        this->id(), read_elapsed);
    break;
}
```

**Risk**: Low. This is purely defensive and only fires when a read hangs for >6s (which is already a broken state). The 6s threshold is conservative — healthy reads take <1ms.


## 6. Recommended Fix Strategy

**Priority order:**

1. **Proposal C** (zero edm_status after force-reset in teardown) — simplest, lowest risk, prevents contamination at the source.

2. **Proposal A** (liveness probe for base-UMD channels) — catches the case where teardown force-reset left a degraded ERISC that writes the sentinel but can't process launch messages.

3. **Proposal D** (relay read hang detection in Phase 5) — defensive safety net that converts the infinite hang into a bounded timeout with degraded-mode fallback.

4. **Proposal B** (pre-handshake health check) — most complex, highest risk, but provides the most comprehensive protection.

Implementing **C + A + D** together would cover all identified failure modes with minimal risk.


## 7. Key File References

```
tt_metal/distributed/mesh_device.cpp:1571-1700     quiesce_internal(), quiesce_devices()
tt_metal/impl/device/device.cpp:404-690             configure_fabric()
tt_metal/impl/device/device.cpp:692-1400            quiesce_and_restart_fabric_workers() Phases 1-3
tt_metal/impl/device/device.cpp:1808-2200           launch_eth_cores_for_quiesce()
tt_metal/impl/device/device.cpp:2202-2350           wait_for_eth_cores_launched()
tt_metal/impl/device/device.cpp:2712-3120           wait_for_fabric_workers_ready() Phases 4-5
tt_metal/fabric/fabric_init.cpp:91-268              configure_fabric_cores() (FIX M at line 172)
tt_metal/fabric/fabric_edm_packet_header.hpp:48-84  EDMStatus enum values
tt_metal/impl/device/edm_status_utils.hpp:28-55     EthDiagSentinel enum
tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:1007-1340  terminate_stale_erisc_routers()
tt_metal/impl/context/metal_env.cpp:321-470         teardown_fabric_config()
tests/ttnn/.../test_ccl_multi_cq_multi_device.cpp:268-389  AsyncExecutionWorksCQ0 test
```
