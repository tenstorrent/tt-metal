# Feature Map — racecondition-hunt (#42429)

This branch accumulated ~200 incremental `FIX XX` markers during investigation.
Each maps to one of the eight features below. Use this as a legend when reading
diffs or CI logs.

> **Note**: `FIX XX` names appear verbatim in runtime log strings and in the
> `scripts/analyze_fabric_hang_log.sh` pattern list — they are NOT renamed in
> code, preserving log parseability.

---

## Feature 1 — ETH_HANDSHAKE_SYMMETRIC

**What it fixes**: STARTED→STARTED deadlock class. Both sides of the ETH link now
participate symmetrically (TCP-style send+poll) instead of one side spinning
unconditionally. `local_value` is zeroed at Object Setup, before `edm_status=STARTED`.

**FIX tags**: `FIX A`, `FIX B`, `FIX C`, `FIX D`, `FIX AD`

**Key commit**: `749979f96f4` (FIX AD — symmetric handshake landed 2026-05-16)

**Files**: `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc`

---

## Feature 2 — FW_LAUNCH_ADDR_ORDERING

**What it fixes**: Stale `fw_launch_addr` (0x9004 = `LAUNCH_ERISC_APP_FLAG`) left at 1
after a fabric session ends. On the next session's RISC reset, ERISC sees `fw_launch_addr=1`
after deassert and launches old (zeroed/stale) L1 firmware before the host writes the new
binary — landing at `0xDEADB07E` and hanging forever.

**Fix**: Zero `fw_launch_addr` while ERISC is held in reset (assert window), then restore
it only _after_ `write_launch_msg_to_core` writes the real binary.

**FIX tags**:
- `FIX EG` — zero during RISC assert in `fabric_init.cpp` (normal S9 path)
- `FIX EG RR` — same fix applied in the `FIX RR` dead-channel-recovery path
- `FIX IJ` — move fw_launch_addr restore to AFTER `write_launch_msg_to_core` in `device.cpp`
- `FIX GH` — BUGGY predecessor of FIX IJ (restored BEFORE load → ERISC ran zeroed L1)
- `FIX MM` — fw_launch_addr handling for MMIO channels after quiesce Phase 2.5
- `FIX MN` — pre-zero snapshot (diagnostic capture before zeroing)
- `FIX GI`, `FIX GI RR` — readback-verify the zero write
- `FIX PF` — dispatch ETH channels: clear fw_launch_addr in `RiscFirmwareInitializer`
- `FIX PD` — Phase 2.5 force-reset channels: clear fw_launch_addr after deassert
- `FIX PC` — fw_launch_addr clear after fabric teardown force-reset
- `FIX PE` — fw_launch_addr clear for channels that terminated cleanly (GAP-56)
- `FIX QQ`, `FIX QQ-V` — fw_ready/fw_launch_addr timeout snapshot diagnostics

**Key commits**: `449d6bd0aba` (FIX GH — BUGGY), `6805e11439d` (FIX IJ — correct)

**Files**: `tt_metal/fabric/fabric_init.cpp`, `tt_metal/impl/device/device.cpp`,
`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`,
`tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`

---

## Feature 3 — MMIO_ETH_RESET_SEQUENCE

**What it fixes**: Incomplete or racy RISC reset sequence for MMIO ETH channels.
Covers the full assert→zero-fw_launch_addr→deassert→sleep→ROM-postcode-poll→
boot-fence→session-id boot flow.

**FIX tags**:
- `FIX S9` — assert/deassert on MMIO base-UMD channels in `configure_fabric_cores()`
- `FIX S8` — boot fence: host writes token before ERISC advances past Object Setup
- `FIX S7`, `FIX S7-MOVE` — session_id token gate at ERISC firmware startup
- `FIX EF` — 500ms poll for `fw_ready` after deassert (timeout → dead)
- `FIX BH` — 5000ms ROM-postcode poll after deassert (pre-SA fallback)
- `FIX DW` — 50ms sleep after deassert before DU poll (prevents stale-ROM false positive)
- `FIX DU` — ROM-postcode poll: confirm ERISC has reached ROM before proceeding
- `FIX RP` — ROM-postcode timeout detection
- `FIX OP` — timing log for S9 assert→deassert window
- `FIX SENDGO` — go_msg write sequencing
- `FIX QR-S9`, `FIX V11-QS7`, `FIX V11-QS89` — quiesce-path S7/S8/S9 variants
- `FIX V11-WH-TERM` — Wormhole termination signal check
- `FIX SA-A` — firmware-side FW_READY gate (ERISC signals host it booted cleanly)

**Files**: `tt_metal/fabric/fabric_init.cpp`, `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc`,
`tt_metal/hw/inc/hostdev/dev_msgs.h`

---

## Feature 4 — DEAD_CHANNEL_RECOVERY

**What it fixes**: MMIO ETH channels that are already dead (stuck at `0x49705180` ROM
postcode or `0xDEADDEAD`) at the start of `configure_fabric_cores()`. Previously left
as permanently dead; now recovered via PCIe-direct soft reset + deferred firmware load.

**Strategy A** (FIX SA): deferred deassert — channel stays halted (assert), host writes
firmware binary via `ConfigureDeviceWithProgram`, then deasserts and waits for `FW_READY`.
Replaces fragile `FIX BH` 5000ms blind wait.

**Strategy B / SA-B**: Pre-known-dead channels from `FIX RR` merged into Strategy A path.

**FIX tags**:
- `FIX RR` — PCIe-direct soft reset for pre-known-dead MMIO channels
- `FIX SA` — Strategy A: deferred deassert + FW_READY gate for recovered channels
- `FIX SA-A` — firmware-side FW_READY signal (shared with Feature 3)
- `FIX SA-B` — merge FIX RR pre-dead into SA deferred path (Unified Channel Recovery)
- `FIX SA-GV` — Strategy A global-value readback verification
- `FIX SA-S`, `FIX SA-ESC` — escape/skip variants for SA path
- `FIX SC`, `FIX SC-ADDR`, `FIX SC2` — configure_fabric readback verification
- `FIX GS`, `FIX GS-2`, `FIX GS-3` — channel health status tracking
- `FIX GJ` — newly-dead channel classification
- `FIX NO` — configure_fabric SA path verification (all 7 steps)
- `FIX GI`, `FIX GI RR` — readback-verify fw_launch_addr zero (shared with Feature 2)

**Files**: `tt_metal/fabric/fabric_init.cpp`, `tt_metal/impl/device/device.cpp`

---

## Feature 5 — NON_MMIO_RELAY_GUARD

**What it fixes**: Operations that go through the non-MMIO relay (L1 writes, l1_barrier,
assert_risc_reset) crashing or hanging when the relay ERISC is dead. All relay-dependent
paths now check relay health and degrade gracefully.

**FIX tags**:
- `FIX M`, `FIX M2` — skip soft reset for non-MMIO base-UMD relay channels
- `FIX AU`, `FIX AU-2` — skip/attempt TERMINATE + l1_barrier when relay is broken
- `FIX AX`, `FIX AX-2` — skip assert_risc_reset for relay-confirmed-dead non-MMIO channels
- `FIX BU` — supersedes FIX AX-2: confirmed-dead relay channels skip assert entirely
- `FIX CL`, `FIX CL-2` — zero L1 before reset (MMIO direct); skip for dead-relay channels
- `FIX AJ` — mark device relay-dead when assert_risc_reset throws
- `FIX AI` — assert+deassert for non-MMIO channels to restart into base-UMD firmware
- `FIX XY`, `FIX XY-2` — clear relay_broken after successful ERISC force-reset
- `FIX XZ` — MMIO ETH heartbeat poll for XZ diagnostics

**Files**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`,
`tt_metal/fabric/fabric_init.cpp`

---

## Feature 6 — DEGRADED_CLUSTER_GUARDS

**What it fixes**: Tests and initializers crashing or producing false failures when the
cluster is partially degraded (relay path broken, channels dead, ring sync failed). All
test fixtures now detect degraded state and either skip, bail, or fast-fail cleanly.

**Sub-areas**:
- *Relay-broken detection*: flags set when relay path confirmed dead
- *Test skip guards*: check flags before running; skip with clear reason
- *Fast-fail paths*: bail before expensive operations (ring sync, AllGather, etc.)
- *Stale-base-UMD reset*: clear stale flag after health confirmed (`FIX RZ2`)

**FIX tags**:
- `FIX AM`, `FIX AL` — skip health check / ring sync for already-known relay-broken devices
- `FIX AK`, `FIX AK-2` — skip l1_barrier drain in teardown when relay dead
- `FIX AA` — skip AllGather when relay path broken
- `FIX QU` — re-assert degraded flags after `Device::configure_fabric()` resets them
- `FIX QW`, `FIX QW-B`, `FIX QW2` — test skip guard: skip if `stale_base_umd` or relay broken
- `FIX QS` — is_fabric_relay_path_broken() guard
- `FIX QR`, `FIX QR-S9` — quiesce readback + sentinel
- `FIX QV` — validation guard
- `FIX RX` — TearDown: skip quiesce_devices() when fabric broken
- `FIX RY`, `FIX RZ`, `FIX RZ2`, `FIX RZ3`, `FIX RZ4` — relay/stale flag management
- `FIX LM`, `FIX LM-2`, `FIX LM-3` — fast-fail on cold-start relay-dead
- `FIX NY`, `FIX NZ`, `FIX NX`, `FIX NW`, `FIX NV`, `FIX NU`, `FIX NT`, `FIX NS`, `FIX NP`, `FIX NO` — topology/relay guard variants
- `FIX V`, `FIX W`, `FIX X`, `FIX V11-UNUSED` — early-exit / clean-return variants
- `FIX TB`, `FIX TC`, `FIX TD`, `FIX TE`, `FIX TF` — cluster degradation bail guards

**Files**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`,
`tests/tt_metal/distributed/test_*.cpp`, `tt_metal/impl/device/device.cpp`

---

## Feature 7 — TEARDOWN_RESET

**What it fixes**: After a fabric session ends (quiesce timeout, relay failure, or clean
shutdown), the next binary's `init_tt_device()` could hang because MMIO ETH channels were
still running fabric firmware. Two-phase teardown: (1) detect relay-broken state, (2) PCIe-
direct hard-reset all MMIO ETH channels after all non-MMIO teardown completes.

**FIX tags**:
- `FIX AB` — original teardown ordering bug (description of root cause)
- `FIX AC` — two-phase ETH reset in `RiscFirmwareInitializer::teardown()`
- `FIX AY`, `FIX AZ` — teardown cleanup for specific channel states
- `FIX AE`, `FIX AF`, `FIX AH`, `FIX AV` — teardown state tracking
- `FIX AQ`, `FIX AQ2` — UMD timeout recovery in teardown (5s per relay-broken non-MMIO)
- `FIX AR`, `FIX AR2` — Pass-0 deassert of Phase-2.5 force-reset channels before launch msg
- `FIX AS` — track successfully deasserted cores for write_launch_msg ordering
- `FIX AT` — timing/sequencing in teardown Pass-0
- `FIX AN`, `FIX AO`, `FIX AP` — additional teardown guard variants
- `FIX BA`, `FIX BC` — base-UMD teardown ordering
- `FIX DK`, `FIX DK-2` — fw_launch_addr stuck-at check before force-reset
- `FIX DY` — teardown relay-dead deassert path

**Files**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`,
`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`

---

## Feature 8 — RING_SYNC_SKIP

**What it fixes**: `wait_for_fabric_router_sync()` (ring sync) hangs for up to 120s when
base-UMD channels are present but the ring cannot complete (relay broken, non-MMIO peer
missing). Now detects the impossible-ring case early and bails before burning the timeout.

**FIX tags**:
- `FIX TH`, `FIX TH2`, `FIX TH3` — detect base-UMD channels; extend or skip sync timeout
- `FIX TI` — skip ring sync when base-UMD channels caused a timeout
- `FIX TJ` — propagate ring-sync-failed flag so subsequent devices skip immediately
- `FIX TK` — clear ring-sync-timed-out at start of fresh `configure_fabric()`
- `FIX TL`, `FIX TL-2` — ring sync skip in additional call sites
- `FIX TG`, `FIX TG2` — partial L1 clear (preserves 0x49706550 sentinel for S9 path)
- `FIX TF`, `FIX TE`, `FIX TD`, `FIX TC` — ring sync guard/bail variants
- `FIX TM`, `FIX TN`, `FIX TO` — timing and metrics
- `FIX TV`, `FIX TW` — ring sync launch-phase variants
- `FIX NT` — EthCoord topology guard during ring sync

**Files**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`

---

## Named Strategies (separate from FIX tags)

| Name | Commit | Description |
|------|--------|-------------|
| `STRATEGY_HOST_SEQUENCED_BARRIER` | `cbd7dcdbb9a` | Eliminate ring sync: host-side ordered barrier replaces ERISC ring message |
| `STRATEGY_G` | `db6a9408d26` | Soft-reset-free quiesce prototype (single-channel N300 MMIO only) |

---

## Tag Index (alphabetical)

For a quick lookup of which feature a tag belongs to:

```
FIX A/B/C/D/AD      → F1 ETH_HANDSHAKE_SYMMETRIC
FIX AA              → F6 DEGRADED_CLUSTER_GUARDS
FIX AB              → F7 TEARDOWN_RESET
FIX AC              → F7 TEARDOWN_RESET
FIX AE/AF/AH/AV     → F7 TEARDOWN_RESET
FIX AJ              → F5 NON_MMIO_RELAY_GUARD
FIX AI              → F5 NON_MMIO_RELAY_GUARD
FIX AK              → F6 DEGRADED_CLUSTER_GUARDS
FIX AL/AM           → F6 DEGRADED_CLUSTER_GUARDS
FIX AN/AO/AP        → F7 TEARDOWN_RESET
FIX AQ/AQ2          → F7 TEARDOWN_RESET
FIX AR/AR2          → F7 TEARDOWN_RESET
FIX AS              → F7 TEARDOWN_RESET
FIX AT              → F7 TEARDOWN_RESET
FIX AU/AU-2         → F5 NON_MMIO_RELAY_GUARD
FIX AX/AX-2         → F5 NON_MMIO_RELAY_GUARD
FIX AY/AZ           → F7 TEARDOWN_RESET
FIX BA/BC           → F7 TEARDOWN_RESET
FIX BH              → F3 MMIO_ETH_RESET_SEQUENCE
FIX BU              → F5 NON_MMIO_RELAY_GUARD
FIX CL/CL-2         → F5 NON_MMIO_RELAY_GUARD
FIX DK/DK-2         → F7 TEARDOWN_RESET
FIX DU/DW           → F3 MMIO_ETH_RESET_SEQUENCE
FIX DY              → F7 TEARDOWN_RESET
FIX EF              → F3 MMIO_ETH_RESET_SEQUENCE
FIX EG/EG RR        → F2 FW_LAUNCH_ADDR_ORDERING
FIX GH              → F2 FW_LAUNCH_ADDR_ORDERING (BUGGY — do not use)
FIX GI/GI RR        → F2+F4 (shared readback)
FIX GJ              → F4 DEAD_CHANNEL_RECOVERY
FIX GS              → F4 DEAD_CHANNEL_RECOVERY
FIX IJ              → F2 FW_LAUNCH_ADDR_ORDERING
FIX LM              → F6 DEGRADED_CLUSTER_GUARDS
FIX M/M2            → F5 NON_MMIO_RELAY_GUARD
FIX MM              → F2 FW_LAUNCH_ADDR_ORDERING
FIX MN              → F2 FW_LAUNCH_ADDR_ORDERING
FIX NO              → F4 DEAD_CHANNEL_RECOVERY
FIX NP/NS/NT..NZ    → F6 DEGRADED_CLUSTER_GUARDS
FIX OP              → F3 MMIO_ETH_RESET_SEQUENCE
FIX PA/PB           → F2+F3 (dispatch cascade)
FIX PC/PD/PE/PF     → F2 FW_LAUNCH_ADDR_ORDERING
FIX QQ/QQ-V         → F2 FW_LAUNCH_ADDR_ORDERING
FIX QR/QR-S9        → F3+F6
FIX QS/QU/QV/QW     → F6 DEGRADED_CLUSTER_GUARDS
FIX RP              → F3 MMIO_ETH_RESET_SEQUENCE
FIX RR              → F4 DEAD_CHANNEL_RECOVERY
FIX RX/RY/RZ        → F6 DEGRADED_CLUSTER_GUARDS
FIX S7/S8/S9        → F3 MMIO_ETH_RESET_SEQUENCE
FIX SA/SA-A..SA-GV  → F4 DEAD_CHANNEL_RECOVERY  (SA-A shared with F3)
FIX SC/SC-ADDR      → F4 DEAD_CHANNEL_RECOVERY
FIX SENDGO          → F3 MMIO_ETH_RESET_SEQUENCE
FIX TB..TF          → F6+F8
FIX TG/TG2          → F8 RING_SYNC_SKIP
FIX TH..TN          → F8 RING_SYNC_SKIP
FIX TV/TW           → F8 RING_SYNC_SKIP
FIX V/W/X           → F6 DEGRADED_CLUSTER_GUARDS
FIX V11-*           → F3 MMIO_ETH_RESET_SEQUENCE
FIX XY/XY-2/XZ      → F5 NON_MMIO_RELAY_GUARD
```
