<!-- SUMMARY: Root cause analysis of FIX SC / relay_broken setup failures in nsexton/0-racecondition-hunt
KEYWORDS: fixsc, gap76, relay_broken, compile_and_configure_fabric, go_msg, setup, allgather, FIX SB2, FIX M, base_umd
SOURCE: code analysis of nsexton/0-racecondition-hunt + CI log 25592656281
SCOPE: fabric_firmware_initializer.cpp FIX SC, compile_and_configure_fabric, relay_broken inference, risc_firmware_initializer.cpp
USE WHEN: investigating t3k_ttnn_tests topology-degraded failures -->

# FIX SC / relay_broken Setup Failure Analysis

**Branch**: `nsexton/0-racecondition-hunt`
**CI Run**: 25592656281
**Date**: 2026-05-09

---

## 1. Exact Behavior of FIX SC (Code-Based, Not Comment-Based)

FIX SC is implemented in two locations:

### Location A: `risc_firmware_initializer.cpp` lines 2599-2656

**When it fires**: After the firmware multicast write + `l1_barrier` + `deassert_risc_reset` for Tensix cores.

**What it does, step by step**:
1. Reads the `go_msg_t` signal byte from each Tensix core's L1 SRAM at the `GO_MSG` address.
2. Compares the signal byte against six known valid RUN_MSG constants: `RUN_MSG_DONE` (0x00), `RUN_MSG_INIT` (0x40), `RUN_MSG_GO` (0x80), `RUN_MSG_RESET_READ_PTR` (0xc0), `RUN_MSG_RESET_READ_PTR_FROM_HOST` (0xe0), `RUN_MSG_REPLAY_TRACE` (0xf0).
3. If the signal is NOT one of these six values (e.g., 0x02, 0x55), it:
   - Logs a WARNING: "FIX SC (GAP-76): Device N core (x,y) has stale go_msg=0xNN"
   - Calls `assert_risc_reset_at_core(ALL)` to halt ALL RISCs on that core
   - Writes `RUN_MSG_DONE` to the go_msg address (so `wait_until_cores_done` will see it as completed)
4. **Does NOT deassert reset** — the core is left halted.

### Location B: `llrt.cpp` lines 289-320 (inside `wait_until_cores_done` polling)

**When it fires**: During the polling loop that waits for cores to reach `RUN_MSG_INIT`.

**What it does**: Same as Location A — assert_risc_reset + write RUN_MSG_DONE inline. This catches cores that transitioned to a stale value between FIX SC's pre-scan and this polling check.

### Assessment

FIX SC itself is **correctly implemented**. It is a defensive measure that detects truly stale SRAM contents and prevents a 10-second hang per affected core. The assert_risc_reset + RUN_MSG_DONE write is the right recovery action.

---

## 2. go_msg=0x02 Protocol Meaning

### go_msg_t Structure (dev_msgs.h lines 194-204)

```cpp
struct go_msg_t {
    union {
        uint32_t all;
        struct {
            uint8_t dispatch_message_offset;  // byte 0
            uint8_t master_x;                 // byte 1
            uint8_t master_y;                 // byte 2
            uint8_t signal;                   // byte 3 (MSB in little-endian)
        };
    };
};
```

### Valid signal values

| Constant | Value | Meaning |
|----------|-------|---------|
| RUN_MSG_DONE | 0x00 | Core finished |
| RUN_MSG_INIT | 0x40 | Initialize firmware |
| RUN_MSG_GO | 0x80 | Start kernel execution |
| RUN_MSG_RESET_READ_PTR | 0xc0 | Reset read pointer |
| RUN_MSG_RESET_READ_PTR_FROM_HOST | 0xe0 | Host-initiated read pointer reset |
| RUN_MSG_REPLAY_TRACE | 0xf0 | Replay trace |

### What is go_msg=0x02?

`0x02` is **NOT any valid RUN_MSG_* value**. It is garbage left in Tensix SRAM from a prior session.

After `tt-smi -r`, ASIC registers are reset but **Tensix L1 SRAM is NOT cleared**. The L1 contents from the previous process (or from power-on randomness) persist. The value 0x02 is likely:

- A stale `RUN_SYNC_MSG_WAITING_FOR_RESET` (0x02) from `dev_msgs.h` line 100, if the sync_msg and go_msg share nearby L1 addresses and a prior session wrote sync values.
- Or simply residual SRAM content (random/stale data in byte 3 of the go_msg word).

**Verdict**: go_msg=0x02 is a **true positive** — it IS stale SRAM that does not correspond to any valid firmware state. FIX SC correctly identifies and handles it. This is NOT the root cause of the failure.

---

## 3. relay_broken Inference: The Real Problem

### The Cascade Chain

The actual failure cascade is:

```
tt-smi -r resets ASICs
    → All ETH ERISCs reboot to UMD relay firmware
    → edm_status at router_sync_address = 0x49706550 on ALL channels
    → This is the NORMAL, EXPECTED state after reset

PHASE 1 (terminate_stale_erisc_routers):
    → Reads 0x49706550 on all channels
    → Correctly identifies as base-UMD firmware (is_base_umd=true)
    → Adds to base_umd_channels (FIX M: skip soft-reset to preserve relay)
    → No probe timeouts → relay_broken=false, probe_dead_channels empty
    → Result: base_umd_channels_map has entries for ALL devices

PHASE 2 (configure_fabric):
    Pass 1: Non-MMIO devices configured first (FIX J2)
        → configure_fabric_cores skips soft-reset for base_umd channels
        → Loads EDM firmware via write_launch_msg_to_core
        → Device::configure_fabric sets fabric_stale_base_umd_channels_=true (line 699)
    Pass 2: MMIO devices configured second
        → Same: skip soft-reset, load via launch_msg

POST-PHASE 2: FIX SB2 (lines 2143-2188) — THE CASCADE TRIGGER
    → Iterates base_umd_channels_map
    → For every MMIO host with non-empty base_umd_channels:
        → Marks ALL non-MMIO devices behind that host as relay_broken
        → dev->set_fabric_relay_path_broken()
    → On a T3K, ALL 4 MMIO hosts have base_umd channels (because 0x49706550 is normal!)
    → Therefore ALL 4 non-MMIO devices (4,5,6,7) get fabric_relay_path_broken_=true

PHASE 4 (wait_for_fabric_router_sync → configure):
    → dead_relay_devices_ may be empty (no probe timeouts on clean boot)
    → But fabric_relay_path_broken_ is set on non-MMIO devices
    → FIX QU re-asserts these flags after configure()
    → Test guards (FIX QW/QS) see degraded fabric → GTEST_SKIP all tests

TEARDOWN:
    → FIX AU-2: non-MMIO relay_broken devices attempt force-reset of ETH channels
    → assert_risc_reset_at_core on non-MMIO channels goes through ETH relay
    → Relay ERISCs now run EDM firmware, not UMD relay → timeout
    → 5s timeout × N channels → "Timeout waiting for Ethernet core service..."
    → Channels left unreset → "4 channel(s) could NOT be force-reset"
    → Topology degraded: 4/8 chips visible
```

### Is FIX SB2 Correct?

FIX SB2 is **correct about the danger**: after PHASE 2, MMIO relay ERISCs run EDM firmware and cannot serve UMD relay protocol reads. Any subsequent `ReadFromDeviceL1` or `l1_barrier` on non-MMIO devices through that MMIO host would hang.

BUT FIX SB2 is **incorrect in its trigger condition**: it fires whenever `base_umd_channels` is non-empty for an MMIO host. On a clean boot after `tt-smi -r`, this is **ALWAYS true** — every channel starts at 0x49706550. This means FIX SB2 marks non-MMIO devices as relay_broken on **EVERY fresh boot**, turning a healthy cluster into a degraded one.

### The Circular Dependency

FIX M says: "Don't soft-reset base-UMD channels because that kills the relay BRISC."
FIX SB2 says: "If base-UMD channels were transitioned via launch_msg (not soft-reset), the relay is now dead."

Together, they create a paradox:
- If you soft-reset: relay dies immediately during PHASE 1 (the original race condition FIX M was designed to prevent).
- If you skip soft-reset and use launch_msg: relay dies after PHASE 2 (which FIX SB2 correctly detects).
- Either way, the non-MMIO device gets marked broken.

---

## 4. Main Branch Contrast

The main branch `fabric_firmware_initializer.cpp` is **262 lines** vs **2828 lines** on the branch. Main branch:

- Has NO `terminate_stale_erisc_routers` (no PHASE 1 probe)
- Has NO FIX M, FIX SB2, FIX SC, FIX E2, or any of the 40+ FIX codes
- Has NO relay_broken tracking
- Has NO dead_relay_devices_ set
- Simply calls `dev->compile_fabric()` async, then `dev->configure_fabric()` synchronously
- `configure_fabric()` on main takes NO probe_dead_channels or base_umd_channels parameters
- `wait_for_fabric_router_sync` on main has no skip logic — it just polls until success or timeout

Main branch's `risc_firmware_initializer.cpp` (`initialize_and_launch_firmware`) is also much simpler:
- No FIX SC (no stale go_msg scanning)
- No FIX SB (no IDLE_ETH guarding)
- No FIX XX (no relay-broken deassert guard)
- Just: write firmware, deassert reset, wait for cores done, throw on timeout

On main, after `tt-smi -r`, the fresh boot just works because:
1. `configure_fabric_cores()` in `Device::configure_fabric()` does `assert_risc_reset_at_core` + `deassert_risc_reset_at_core` on ALL channels (including those at 0x49706550). This cleanly resets the ERISC and loads new firmware.
2. Since soft-reset is done for ALL channels, the relay temporarily dies but it doesn't matter because the non-MMIO devices are configured through this same reset cycle.
3. The PHASE 1/PHASE 2 ordering problem that FIX J was designed to solve doesn't exist on main because main doesn't have a separate probe phase.

---

## 5. Root Cause Hypothesis

### Primary Root Cause: FIX SB2 (fabric_firmware_initializer.cpp lines 2143-2188)

FIX SB2 fires on every fresh boot because 0x49706550 (base-UMD firmware) is the NORMAL post-reset state of all ETH channels. Its trigger condition `base_umd_chans.empty()` is never true after `tt-smi -r`.

The logic error is that FIX SB2 treats the post-PHASE-2 state as permanently broken, but in reality:
- The EDM firmware loaded in PHASE 2 **IS the intended firmware** — the transition from base-UMD to EDM is the desired outcome.
- After EDM firmware is running and ring-sync completes, the fabric is fully operational.
- UMD relay reads are NOT needed after fabric init succeeds.
- The `fabric_relay_path_broken_` flag set by FIX SB2 should only apply during the init-to-sync window, not permanently.

### Secondary Root Cause: FIX M's Paradox

FIX M correctly prevents soft-reset of relay channels during PHASE 1 (before non-MMIO probing completes). But the FIX J2 ordering (non-MMIO first, MMIO second in PHASE 2) means that by the time MMIO channels are configured, non-MMIO configure_fabric_cores has ALREADY completed via the relay. The relay is no longer needed for non-MMIO configuration.

The only remaining use of the relay after PHASE 2 is `wait_for_fabric_router_sync` and `verify_all_fabric_channels_healthy` in `configure()`. But EDM firmware on the ERISC can handle `ReadFromDeviceL1` through the fabric protocol — it just can't serve the legacy UMD relay protocol. The branch code assumes ALL reads go through UMD relay, but with EDM firmware running, reads should go through the fabric path.

### Tertiary Issue: FIX SC Fires But Is Not Causal

FIX SC fires because Tensix SRAM is not cleared by `tt-smi -r`. The stale go_msg=0x02 values are correctly detected and handled. FIX SC itself does not cause the cascade failure. It is a symptom of SRAM persistence, not a bug in the recovery logic.

---

## 6. Concrete Fix Recommendations

### Fix 1 (Highest Impact): Revise FIX SB2 trigger condition

**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` lines 2143-2188

FIX SB2 should NOT fire when the base-UMD → EDM transition is the expected, intended behavior (i.e., on a clean boot). The condition should be:

- **Only fire** when base_umd_channels exist on an MMIO host AND the non-MMIO device's probe detected actual problems (probe timeouts, corrupt channels, relay_broken).
- **Do NOT fire** when all probes succeeded and the non-MMIO device is healthy.

Specifically: change the FIX SB2 loop to skip non-MMIO devices that are NOT in `dead_relay_devices_` and whose own probe returned no probe_dead_channels. If the non-MMIO device's probe succeeded (relay was working at probe time), the configure-via-launch-msg path will work correctly, and post-PHASE-2 the fabric protocol replaces UMD relay for reads.

### Fix 2 (Alternative): Clear relay_broken after successful ring-sync

If FIX SB2 must remain conservative during init, add a recovery path in `configure()` after `wait_for_fabric_router_sync` succeeds:

- If ring-sync succeeded for all devices, clear `fabric_relay_path_broken_` for non-MMIO devices that were marked broken ONLY by FIX SB2 (not by probe failures).
- The existing FIX RZ2 logic (lines 259-292) partially does this for `fabric_stale_base_umd_channels_` but does NOT clear `fabric_relay_path_broken_`.

### Fix 3 (Complementary): Skip MMIO base_umd_channels from FIX SB2 when all non-MMIO probes passed

In the FIX SB2 loop, before marking a non-MMIO device relay_broken, check if that device's probe returned clean (no dead channels, no timeouts). If the probe was clean, the relay WAS functional at probe time. The EDM firmware transition happened after configure_fabric ran for that non-MMIO device, so the device doesn't need UMD relay anymore.

### Fix 4 (Surgical): Don't include base_umd_channels for MMIO devices whose non-MMIO peers all probed successfully

In FIX M2 (lines 2046-2103), the code removes base_umd channels whose peer is dead. Add the inverse: if ALL non-MMIO peers behind an MMIO host probed successfully, clear that MMIO host's base_umd_channels entirely. FIX SB2 then sees empty base_umd_channels and skips the relay_broken marking.

---

## Summary Table

```
Component       Status    Assessment
-----------     ------    ----------
FIX SC          OK        Correctly detects stale SRAM, correctly halts affected cores
go_msg=0x02     OK        Genuine stale data after tt-smi -r (SRAM not cleared)
FIX M           OK*       Correct during PHASE 1, but creates the setup for FIX SB2 cascade
FIX SB2         BUG       Fires on every fresh boot because 0x49706550 is the normal state
FIX E2          OK        Only marks non-MMIO as dead when probe detected actual problems
FIX J/J2        OK        Correct ordering: probe before configure, non-MMIO before MMIO
relay_broken    BUG*      Set by FIX SB2 even when fabric init succeeds → permanent degradation
Main branch     OK        Simple, no probe phase, configure_fabric soft-resets everything
```

*FIX SB2 is the root cause of the cascade. All other FIX codes operate correctly in isolation.
