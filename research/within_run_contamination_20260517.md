<!--
SUMMARY: Analysis of within-run state contamination introduced by racecondition-hunt branch teardown vs main — maps every FIX label to the dirty state it leaves.
KEYWORDS: teardown, contamination, force-reset, FIX AI, FIX BU, FIX CL, FIX AC, FIX PE, edm_status, 0x49705180, ROM hang, ETH link training, within-run
SOURCE: Code analysis of fabric_firmware_initializer.cpp and risc_firmware_initializer.cpp on racecondition-hunt branch vs main (2026-05-17)
SCOPE: fabric teardown path, ERISC reset lifecycle, L1 state between tests within a single CI run
USE WHEN: Investigating within-run contamination symptoms (ROM hang, stale edm_status, probe-dead channels) on the racecondition-hunt branch
-->

# Within-Run Contamination Analysis: racecondition-hunt vs main

**Date**: 2026-05-17
**Branch**: `nsexton/0-racecondition-hunt`
**Files analyzed**:
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` (branch: ~1700 lines, main: ~262 lines)
- `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` (branch: ~1180 lines, main: ~1120 lines)

## 1. Key Premise

Between CI dispatch runs, machines are hardware-reset. Any contaminated ETH/ERISC state observed within a run was introduced by the branch's own teardown logic between tests.

## 2. Main's Teardown: Fire-and-Forget

Main's `FabricFirmwareInitializer::teardown()` (lines 66-138) is ~70 lines of straightforward code:

1. Write `termination_signal` to each Tensix MUX core, then `l1_barrier`.
2. Write `termination_signal` to the master ETH router on each device.
3. Clear `devices_`, `initialized_`, erase `init_done` key.
4. **No polling, no force-reset, no assert/deassert ERISC reset.**

Main's `RiscFirmwareInitializer::teardown()` (lines 189-207) is ~18 lines:

1. `teardown_simulator_ethernet_cores()`
2. For each device: `assert_cores(device_id)` + `cluster_.l1_barrier(device_id)`
3. `set_internal_routing_info_for_ethernet_cores(false)` — signals ERISCs to return to base firmware.

**`assert_cores()` in main** (lines 363-370):
- Asserts Tensix workers (ALL RISCs)
- If ETH firmware is NOT cooperative: asserts active ETH cores (ERISC0 only, preserving NCRISC)
- Asserts inactive ETH cores (ALL RISCs)

**Critical difference**: main's `assert_active_ethernet_cores_to_reset()` asserts `ALL_TENSIX & ~ERISC0` — it resets all subordinate RISCs but **leaves ERISC0 running** on active ETH cores. It does NOT call `deassert`. The `set_internal_routing_info_for_ethernet_cores(false)` then tells the still-running ERISC0 to return to base UMD firmware cooperatively.

**Result for main**: after teardown, active ETH cores are still running ERISC0 in base-UMD firmware. NCRISC is reset (subordinate). ETH PHY link is maintained because ERISC0 (which manages link training) was never taken down. The ERISC naturally transitions to base firmware. L1 is NOT zeroed — but since ERISC0 is alive and cooperative, it can respond to the next session's probe reads. **Between CI runs, the hardware reset cleans everything.**

**Main does NOT handle teardown failures gracefully.** If a relay dies mid-test, main's teardown throws (TT_FATAL on dispatch ordering, unguarded l1_barrier, etc.) and the process crashes. The hardware reset before the next CI run is the recovery mechanism.

## 3. Branch Teardown: What Was Added

The branch transforms teardown from a 70-line fire-and-forget into an 800+ line defensive cleanup system with 30+ FIX labels. The goal: survive relay failures, timeout gracefully, force-reset stuck ERISCs, and leave clean state for the **next test within the same run** (not just the next CI run).

### 3.1 FabricFirmwareInitializer::teardown() — Branch Additions

#### Phase 1: Tensix MUX Termination (same as main, plus guards)
- **Added**: try/catch around l1_barrier (lines 353-368)
- **Added**: Per-MUX-core TERMINATE poll with 5s timeout + force assert_risc_reset on timeout (lines 370-461)
- **Contamination**: assert_risc_reset on stuck Tensix MUX core is assert-only (no deassert). Core stays in reset until next init re-initializes it. **Low risk** — Tensix cores are fully re-initialized each test.

#### Phase 2: ETH Router Termination
- **FIX AU / AU-2** (lines 467-537): Relay-broken non-MMIO devices attempt TERMINATE write but catch failures. Added try/catch around master router TERMINATE + l1_barrier.
- **Contamination from FIX AU-2**: If TERMINATE write fails, the channel enters the force-reset second pass. The force-reset path (FIX AI) is where contamination is created. **FIX AU itself is just routing — not a contamination source.**

#### Phase 3: ETH Router Poll (lines 540-704)
- **FIX B**: Added global-deadline poll for ALL active ETH channels (not just master).
- **FIX PE** (lines 610-690): Clear `fw_launch_addr` for cleanly-terminated channels. Without this, `erisc_app_still_running()` fires false-positive 500ms wait on next test.
- **FIX AU**: Relay-broken non-MMIO channels skip poll, queued directly for force-reset.
- **Contamination from FIX PE**: If FIX PE write fails (catch-all), the stale `fw_launch_addr` persists → next test's `reset_cores()` sees non-zero → 500ms wait → force-reset cascade. **Low severity — cosmetic timeout, not functional breakage.**

#### Phase 4: Force-Reset Second Pass (lines 706-1013) — **PRIMARY CONTAMINATION SOURCE**

This is where the branch diverges most from main. Channels that didn't TERMINATE cleanly get force-reset via assert+deassert.

**FIX BU** (lines 807-833): Non-MMIO relay-dead channels **skip assert_risc_reset entirely**.
- Contamination: ERISC stays running fabric firmware. L1 retains stale routing tables, edm_status, launch config.
- Mitigation: FIX CL-2 records these in `pending_pre_dead_non_mmio_` so the next init injects them into `probe_dead_channels_map`.
- **Gap**: L1 is NOT zeroed (relay is dead, no PCIe path to non-MMIO). The ERISC continues running old firmware, which means its edm_status could be anything (RUNNING, STARTED, REMOTE_HANDSHAKE_COMPLETE). The next init's `terminate_stale_erisc_routers()` must handle this — and it does, but at the cost of timeout cascades.

**FIX DK-2** (lines 852-908): For channels stuck at REMOTE_HANDSHAKE_COMPLETE, attempt graceful IMMEDIATELY_TERMINATE before force-reset. 500ms poll window.
- Contamination: If graceful exit succeeds, channel is added to `cleanly_terminated` (FIX PE clears fw_launch_addr). **Clean path, no contamination.**
- If graceful exit fails, falls through to FIX AI force-reset below.

**FIX CL** (lines 909-953): **The pre-reset L1 zeroing.** Zeroes three regions before assert_risc_reset:
1. `router_sync_address` (edm_status) — 4 bytes
2. `ROUTING_TABLE` region — full size
3. `LAUNCH` region — full size

Only for MMIO channels (PCIe-direct write).

**What FIX CL covers**:
- edm_status_address → zeroed → ROM won't see stale RUNNING/STARTED status
- Routing table → zeroed → ROM init won't interpret stale routes
- Launch config → zeroed → ROM init won't see stale launch messages

**What FIX CL does NOT cover (potential gaps)**:
- `fw_launch_addr` — handled separately by FIX PC (line 961-980)
- `GO_MSG` / `RUN_MSG` addresses — NOT zeroed by FIX CL. If stale go_msg is non-zero, the rebooted ERISC could interpret it as a launch command.
- ETH firmware data buffers (packet queues, semaphores, counters in unreserved L1) — NOT zeroed
- ETH heartbeat address (0x1F80 on WH) — NOT zeroed (but ROM naturally overwrites during boot)
- Non-MMIO channels — NOT zeroed (relay dead, no PCIe path)

**FIX AI** (lines 835-1013): assert + deassert ERISC reset for MMIO/live channels.
- `cluster_.assert_risc_reset_at_core(ALL)` — halts ALL RISCs (ERISC0 + NCRISC + subordinates)
- `cluster_.deassert_risc_reset_at_core(ALL)` — restarts ALL RISCs from ROM
- **FIX PC** (lines 961-980): Clear `fw_launch_addr` after deassert to prevent false-positive stall.
- **FIX XY-2** (lines 981-995): Clear `relay_broken` flag if assert+deassert succeeded.

**Contamination from FIX AI**:

The assert+deassert sequence triggers a full ERISC ROM reboot:
1. Assert halts ERISC mid-execution (could be mid-packet, mid-handshake, mid-link-training)
2. Deassert starts ROM boot sequence
3. ROM writes postcode **0x49705180** to `edm_status_address` (0x18070) during init
4. ROM zeroes parts of L1 during boot (including heartbeat address)
5. ROM begins ETH PHY link training with peer on the other side of the ETH cable
6. If link training succeeds: ROM jumps to base-UMD firmware, which writes sentinel 0x49706550 to edm_status
7. If link training **fails** (because peer is dead/in-reset/running incompatible firmware): ROM hangs at 0x49705180

**This is the contamination lifecycle**:

```
Test N teardown                     Test N+1 init
┌─────────────────────┐             ┌──────────────────────────┐
│ FIX CL: zero L1     │             │ terminate_stale_erisc_   │
│   status=0           │             │ routers() reads:         │
│   routing_table=0    │             │                          │
│   launch_config=0    │             │  edm_status == 0x49705180│
│                      │             │  → ROM postcode! Hung!   │
│ FIX AI: assert ALL   │             │  → probe_dead_channels   │
│         deassert ALL │             │  → degraded fabric init  │
│                      │             │  → 45s timeout cascade   │
│ ROM boots:           │             │                          │
│   writes 0x49705180  │             │ OR                       │
│   starts ETH link    │             │                          │
│   training...        │             │  edm_status == 0x49706550│
│                      │             │  → UMD ready, clean!     │
│ FIX XZ: poll for     │             │                          │
│   heartbeat 0xABCD   │             │                          │
│   (up to 5s)         │             │                          │
└─────────────────────┘             └──────────────────────────┘
```

**The race**: FIX XZ (lines 1038-1087+) polls MMIO channels for heartbeat (0xABCDxxxx) confirming base-UMD firmware is up. But even after heartbeat appears, edm_status may still be 0x49705180 for a few more ms until UMD firmware writes 0x49706550. If the next test's init reads edm_status during that window, it sees ROM postcode → probe_dead.

### 3.2 RiscFirmwareInitializer::teardown() — Branch Additions

**FIX AC** (lines 385-610): Two-phase ETH reset when relay is broken.
- Step 1: Scan for relay_broken_non_mmio, any_teardown_timed_out.
- **FIX BA** (lines 432-490): Treat non-MMIO devices with `channels_not_ready_for_traffic` as relay-broken even if `relay_path_broken_` is not set.
- **FIX TK** (lines 459-490): Guard against triggering FIX BA when channels_not_ready came from ring sync timeout (FIX TI path).
- Step 2: PCIe-reset ALL MMIO ETH channels (assert+deassert). **FIX DU** (lines 516-570): pre-scan and skip channels already at UMD base firmware.
- **FIX AR2** (line 619): 100ms post-deassert delay to let ROM zero heartbeat address.
- **FIX AR** (lines 625-729): Bulk poll all MMIO ETH channels for heartbeat (0xABCDxxxx), 5s window.
- **FIX AQ** (lines 740-939): Secondary poll of edm_status_address until no longer 0x49705180, 10s window.
  - **FIX DN** (lines 825-921): Fallback when fabric_context is torn down — use hardcoded 0x18070 as edm_status_address.

**Contamination from FIX AC**:
- The assert+deassert on MMIO ETH channels triggers the **exact same ROM reboot cycle** as FIX AI.
- FIX AC does NOT zero L1 before assert (unlike FIX CL in fabric teardown). It relies on ROM to overwrite.
- FIX AQ polls for edm_status to clear 0x49705180, but with a 10s timeout. If a channel doesn't clear in 10s, it persists to the next test.
- **FIX DU** mitigates the worst case: channels already at UMD firmware are skipped, preventing a double-reset that would cause ETH link deadlock.

**FIX AY** (lines 942-979+): Deferred non-MMIO ETH ERISC reset via restored relay.
- After MMIO relay is restored (FIX AC), use write-only assert+deassert for non-MMIO ETH ERISCs.
- Goes through: PCIe → MMIO relay ERISC (now running base UMD) → NOC → non-MMIO SOFT_RESET register.
- **FIX AV**: Break on first failure per device (all cores share relay).
- **FIX PG** guard: Skip FIX AY entirely if no MMIO heartbeat was confirmed ready.

**Contamination from FIX AY**:
- Non-MMIO ERISCs are assert+deassert'd. They reboot into ROM.
- Non-MMIO ROM will attempt ETH link training with their MMIO peer.
- If the MMIO peer is already running base-UMD (thanks to FIX AC), link training succeeds → clean boot.
- If the MMIO peer is still in ROM (FIX AC didn't fully complete), both sides are in ROM simultaneously → link training should succeed (ROM-to-ROM is a standard boot path).
- **Risk**: timing. If test N+1 starts before non-MMIO ERISCs finish link training, init reads edm_status=0x49705180 on non-MMIO devices → probe_dead.

## 4. Contamination Lifecycle Per Channel Type

### 4.1 MMIO Channel — Clean Terminate (happy path)
```
Teardown:  TERMINATE write → poll → EDMStatus::TERMINATED ✓
           FIX PE: zero fw_launch_addr
State left: ERISC running base-UMD firmware, edm_status=TERMINATED (or stale but ignored),
            fw_launch_addr=0, L1 has stale routing table + launch config (zeroed by next init).
Next init:  terminate_stale_erisc_routers() reads TERMINATED → clean.
            configure_fabric_cores() writes new routing table / launch config.
VERDICT:    ✅ CLEAN
```

### 4.2 MMIO Channel — Force-Reset (FIX AI path)
```
Teardown:  TERMINATE write → poll timeout (5s) → FIX CL: zero status, routing table, launch config
           FIX AI: assert ALL + deassert ALL → ROM boot → 0x49705180
           FIX PC: zero fw_launch_addr
           FIX XZ: poll heartbeat 0xABCDxxxx (5s window)
State left: ERISC rebooting. If link training succeeds: base-UMD, edm_status=0x49706550.
            If link training hangs: ROM stuck at 0x49705180.
            fw_launch_addr=0, routing table=0, launch config=0 (zeroed by FIX CL before assert).
            BUT after deassert, ROM overwrites L1 including edm_status=0x49705180. FIX CL's zeroing
            only prevents the ROM from misinterpreting stale data during its own boot — it does NOT
            prevent the ROM from writing 0x49705180 as its natural postcode.
Next init:  If edm_status=0x49706550 (link training succeeded): clean.
            If edm_status=0x49705180 (still in ROM or link hung): probe_dead → degraded init.
VERDICT:    ⚠️ RACE — depends on whether ROM finishes link training + UMD firmware startup before
            the next test's terminate_stale_erisc_routers() reads edm_status.
            FIX XZ (5s heartbeat poll) + FIX AQ (10s edm_status poll) in risc teardown try to
            close this window, but FabricFirmwareInitializer::teardown's FIX XZ is a separate
            shorter poll that may not be sufficient.
```

### 4.3 Non-MMIO Channel — Relay Alive (FIX AI path via relay)
```
Teardown:  Same as 4.2 but assert+deassert goes through relay (relay must be alive).
           FIX CL: NOT applied (FIX CL only runs for MMIO channels).
           FIX AI: assert+deassert via relay.
State left: ERISC rebooting. L1 still has stale routing table + launch config (not zeroed).
            ROM may misinterpret stale L1 during boot — but in practice ROM overwrites
            edm_status and the boot sequence doesn't parse L1 fabric structures.
Next init:  Same race as 4.2.
VERDICT:    ⚠️ Same race as MMIO, but L1 is dirtier (no FIX CL zero).
```

### 4.4 Non-MMIO Channel — Relay Dead (FIX BU path)
```
Teardown:  FIX BU: skip assert_risc_reset (relay dead, 5s timeout per channel).
           FIX CL-2: record in pending_pre_dead_non_mmio_.
           ERISC is STILL RUNNING fabric firmware (STARTED, RUNNING, or REMOTE_HANDSHAKE_COMPLETE).
           L1 retains ALL stale fabric data.
State left: ERISC running fabric firmware. edm_status=STARTED/RUNNING/REMOTE_HANDSHAKE_COMPLETE.
            L1 fully dirty. fw_launch_addr non-zero.
Next init:  FIX AC in risc_firmware_initializer::teardown PCIe-resets MMIO ETH to restore relay.
            FIX AY then uses restored relay to assert+deassert non-MMIO ERISCs.
            If FIX AY succeeds: non-MMIO ERISCs reboot into base-UMD. Still a race for link training.
            If FIX AY fails (FIX PG: no heartbeat confirmed): non-MMIO ERISCs stay running fabric FW.
            Next test's terminate_stale_erisc_routers() will see stale edm_status → probe_dead or
            fabric firmware → timeout → degraded init.
VERDICT:    ❌ CONTAMINATED — recovery depends on FIX AC + FIX AY chain succeeding AND
            non-MMIO ERISCs finishing ROM boot + link training before next test reads them.
```

## 5. Does FIX CL Close the Loop?

**FIX CL addresses one specific problem**: preventing stale L1 data from causing the ROM to hang at 0x49705180 instead of progressing through boot.

**FIX CL does NOT close the contamination loop because:**

1. **FIX CL runs BEFORE assert, but ROM writes 0x49705180 AFTER deassert.** FIX CL's zeroing prevents ROM from misinterpreting stale L1 structures during boot, but the ROM's own postcode (0x49705180) is written naturally as part of every boot sequence. The contamination symptom (seeing 0x49705180 on next init) is not caused by stale L1 — it's caused by reading edm_status while the ROM is still in progress.

2. **FIX CL only covers MMIO channels.** Non-MMIO channels with dead relay (FIX BU path) cannot have L1 zeroed because there's no PCIe path. These channels retain fully stale L1.

3. **FIX CL does not cover all L1 regions.** GO_MSG, semaphores, packet buffers, and other ETH firmware state are not zeroed. These may not cause ROM hang but could confuse the next init if read before the new firmware writes them.

4. **The root contamination is the assert+deassert itself.** Main avoids this entirely by never calling assert+deassert on active ETH cores — it uses cooperative firmware transition (set_internal_routing false). The branch MUST do assert+deassert because cooperative transition requires the ERISC to be alive and responsive to the command, which fails when the channel is stuck/unresponsive.

## 6. The Fundamental Tradeoff

**Main's approach**: fire-and-forget. If teardown works, great. If it fails, crash. Hardware reset between CI runs recovers everything. Within-run contamination is not handled because main doesn't expect within-run recovery.

**Branch's approach**: defensive cleanup. Try to leave clean state for the next test within the same run. This requires force-resetting stuck ERISCs (assert+deassert), which triggers ROM reboot → 0x49705180 window → potential contamination race with the next test's init.

The branch creates a **new class of contamination** (assert+deassert reboot race) that main never had. Main's contamination was simpler: if teardown failed, the process crashed and the hardware reset cleaned up. The branch survives teardown failures but introduces timing-dependent state that the next test may read as corrupt.

## 7. Recommended Next Steps

If FIX CL is not sufficient (and per the analysis above, it is not — it addresses a symptom, not the root cause):

1. **Option A: Wait for ROM boot completion in FabricFirmwareInitializer::teardown.**
   FIX XZ already polls for heartbeat, but it should also poll edm_status (like FIX AQ does in risc teardown). Ensure that after FIX AI's assert+deassert, edm_status has cleared 0x49705180 before returning from fabric teardown. This would close the race for MMIO channels.

2. **Option B: Skip assert+deassert entirely and let RiscFirmwareInitializer::teardown handle it.**
   FabricFirmwareInitializer::teardown would only do the soft TERMINATE + poll. Channels that don't respond are left for risc teardown (FIX AC) to handle via the existing PCIe-reset + heartbeat + edm_status poll chain. This avoids the double-reset problem (fabric teardown resets, then risc teardown resets again).

3. **Option C: Insert a test-barrier between teardown and init.**
   After FabricFirmwareInitializer::teardown but before the next test's init, wait for ALL channels to confirm base-UMD firmware via edm_status != 0x49705180. This is essentially what FIX TV does in `run_launch_phase` on the init side, but doing it on the teardown side guarantees the next init always sees clean state.

4. **For non-MMIO relay-dead channels**: These cannot be fixed within teardown (no path to reach them). The current approach (FIX BU skip + FIX CL-2 pending_pre_dead + FIX AY deferred reset) is the best available. The risk is that FIX AY may not complete before the next test's init reads these channels. Consider extending FIX AY to include an edm_status poll for non-MMIO channels (via restored relay) before returning.
