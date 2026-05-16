<!--
SUMMARY: Evaluation of Addendum A recommendation A.2.7 (document the fabric init contract) against nsexton/0-racecondition-hunt
KEYWORDS: addendum-a, A.2.7, fabric-init, teardown, quiesce, racecondition-hunt, documentation-gap
SOURCE: Branch analysis of nsexton/0-racecondition-hunt vs origin/main
SCOPE: TT-Fabric init/teardown lifecycle documentation completeness
USE WHEN: Deciding whether to add fabric init contract documentation to TT-Fabric-Architecture.md
-->

# Addendum A.2.7 Evaluation: Document the Fabric Init Contract

**Branch**: `nsexton/0-racecondition-hunt`
**Worktree**: `/workspace/group/worktrees/racecondition-main/`
**Date**: 2026-05-08

---

## Verdict: A.2.7 STRONGLY APPLIES

The branch introduces a dramatically more complex fabric init/teardown lifecycle than what existed on `main`, with at least 6 quiesce phases, 3 new channel classification categories, degraded-mode fallback logic, and cross-session state persistence contracts. Almost none of this is documented in tech reports — it exists only in inline code comments scattered across ~5 source files.

---

## Q1: What does TT-Fabric-Architecture.md currently say about init/teardown lifetime?

**Very little.** The document is primarily an architecture spec for the data plane, routing layers, and APIs. Init/teardown coverage:

- **Section 1.1.2 (Control Plane)** (line 135–157): Says the control plane "sets up and launches the Data Plane" and "continuously monitors the system." No mention of teardown, lifecycle states, or re-init.

- **Section 7.1 (Fabric Node Status Mailbox)** (line 921–931): The ONLY section that touches teardown, and it was **added by this branch**. It documents:
  - The quiesce/restart cycle concept (line 929)
  - The requirement to poll `EDMStatus::TERMINATED` on every active ERISC channel before L1 overwrite (line 929)
  - MUX core termination polling + BRISC halt requirement on Wormhole (line 931)

- **No section covers**: `SetFabricConfig` state machine, `MeshDevice::create` → `configure_fabric` call chain, idempotency, soft-reset model, or the "between tests" lifecycle.

The BasicEthernetGuide.md (line 537+) was also modified by this branch to add a "Host-Side Teardown: L1 Overwrite Ordering" subsection — another gap-fill from the race-condition investigation.

## Q2: Does racecondition-hunt add or modify any tech report documentation?

**Yes, two files:**

1. **`tech_reports/TT-Fabric/TT-Fabric-Architecture.md`** — Added section "Status Mailbox and Teardown Ordering" (6 lines) under Section 7.1. Covers ERISC TERMINATED polling and MUX BRISC halt requirement.
   - `git diff origin/main...HEAD -- tech_reports/TT-Fabric/TT-Fabric-Architecture.md` shows the diff at line 927.

2. **`tech_reports/EthernetMultichip/BasicEthernetGuide.md`** — Added "Host-Side Teardown: L1 Overwrite Ordering" subsection (8 lines). Covers L1 overwrite hazard, per-channel TERMINATED confirmation, and `assert_risc_reset_at_core` PHY teardown warning.

Both additions are narrow and tactical — they document the specific invariants discovered during the race investigation but do not describe the overall init lifecycle, phases, state machine, or cost model.

## Q3: Does the branch add inline code comments that partially address the doc gap?

**Yes, extensively.** The branch adds hundreds of lines of inline comments that constitute an implicit specification. Key locations:

### fabric_init.hpp (lines 18–52)
- `FabricCoresHealth` struct documentation: `all_channels_healthy`, `newly_dead_channels` semantics
- `configure_fabric_cores()` contract: pre_known_dead_channels, skip_soft_reset_channels parameter semantics
- FIX M rationale: why soft-resetting base-UMD relay BRISC kills the non-MMIO relay path

### fabric_init.cpp (lines 100–300)
- BRISC-only soft reset rationale (line 113): "perform a BRISC-only soft reset (assert + deassert) before writing L1"
- Dead channel cascade prevention (line 127–139): relay queue saturation model
- FIX TG / FIX TG2 (line 260–290): partial L1 clear contract — zero sync addresses but preserve `edm_status_address` for base-UMD detection

### device.cpp — configure_fabric() (lines 404–720)
- Flag reset contract (lines 412–445): `fabric_relay_path_broken_`, `fabric_channels_not_ready_for_traffic_`, `fabric_ring_sync_timed_out_`, `fabric_stale_base_umd_channels_`, `fabric_pre_dead_channels_`, `fabric_external_umd_channels_` — all cleared at configure_fabric() entry
- Degraded mode contract (lines 471–486): newly_dead vs pre_known distinction
- Pre-launch canary (line 638): host writes 0xA0A0A0A0 so next session's `terminate_stale_erisc_routers()` can distinguish "crash" from "never launched"

### device.cpp — quiesce_and_restart_fabric_workers() (lines 727–3400+)
- **Phase 1** (line 882): Send IMMEDIATELY_TERMINATE to each MUX worker core
- **Phase 2** (line 911): Poll each MUX core until TERMINATED; halt BRISC unconditionally after
- **Phase 2.5** (line 1007): Terminate ERISC fabric routers before L1 overwrite; skip for relay-broken non-MMIO
- **Phase 3** (line 1271): Re-configure and re-launch fabric workers via configure_fabric_cores()
- **Phase 4** (line 2841): Wait for each MUX core to reach READY_FOR_TRAFFIC
- **Phase 5** (line 2946): Wait for ERISC handshake completion (ring-sync)
- **Phase 5b** (line 3271): Post-ready ERISC health check

### fabric_firmware_initializer.cpp (lines 1041–1200+)
- `terminate_stale_erisc_routers()`: probe_dead_channels, base_umd_channels, external_umd_channels classification
- Relay queue saturation model (kMaxRelayTimeouts = 3, 4-slot queue)
- ROM postcode parallel poll (FIX RP PARALLEL)
- Cross-session state: fw_launch_addr canary model

### metal_env.cpp (lines 100–175)
- GAP-78 (line 145): TT_THROW on use_count > 0 at teardown — dispatch must be drained before fabric teardown
- Exception-safe destructor wrapping for teardown_fabric_objects() and cluster_.reset()

## Q4: What new implicit contracts does racecondition-hunt introduce that need documentation?

### 4.1 Channel Classification Taxonomy (NEW)
Three mutually exclusive channel states at init time, discovered by `terminate_stale_erisc_routers()`:
- **probe_dead_channels**: physically dead link, probe L1 read threw exception
- **base_umd_channels**: BRISC alive with base-UMD relay firmware (`edm_status == 0x49706550`)
- **external_umd_channels**: base-UMD but no in-cluster peer (external ETH, out-of-mesh)

Each classification triggers different handling in `configure_fabric_cores()` (skip soft-reset, partial L1 clear, full skip).

### 4.2 Six-Phase Quiesce Protocol (NEW)
The quiesce_and_restart_fabric_workers() function implements a 6-phase protocol (Phases 1, 2, 2.5, 3, 4, 5 + 5b). This did not exist as a formalized protocol before this branch. Each phase has ordering dependencies, skip conditions (relay-broken, DISABLED), and failure handling.

### 4.3 Degraded-Mode Fabric Operation (NEW)
The branch introduces a formal degraded-mode concept where fabric continues operating with dead/corrupt channels excluded. configure_fabric() distinguishes "pre-known dead" (warning, continue) from "newly dead" (TT_THROW).

### 4.4 Cross-Session State Persistence (NEW)
Several L1 values persist across sessions and carry semantic meaning:
- `edm_status_address = 0x49706550` → base-UMD relay firmware (skip soft-reset)
- `edm_status_address = 0xA0A0A0A0` → host pre-launch canary (ERISC crashed or never launched)
- `edm_status_address = 0xDEADB07E` → crashed firmware
- `fw_launch_addr` contents → distinguish "launch message consumed" from "never launched"
- Stale `edm_local_sync_address = 0xa1b1c1d1` → REMOTE_HANDSHAKE_COMPLETE from prior session (causes ring-sync timeout if not cleared)

### 4.5 Relay Queue Saturation Model (NEW)
Non-MMIO L1 reads route through a 4-slot ETH relay queue. Each timed-out probe read leaves one stuck command. After 3 timeouts (kMaxRelayTimeouts), all further reads hang indefinitely. This is a hard hardware constraint that the init code must respect.

### 4.6 Teardown Ordering Invariant (NEW — GAP-78)
Dispatch threads must be fully drained before fabric teardown begins. Violation causes ETH L1 corruption that persists across `tt-smi -r` resets. Enforced via TT_THROW in `MetalEnvImpl::check_use_count_zero()` (metal_env.cpp:145).

### 4.7 SetFabricConfig State Machine (EXISTING but undocumented)
`DISABLED → FABRIC_1D/FABRIC_2D → DISABLED` is the only valid transition sequence. Non-DISABLED to non-DISABLED transitions that differ TT_FATAL. This is enforced in `metal_env.cpp:260–275` but not documented anywhere.

### 4.8 ERISC Soft-Reset Semantics (NEW)
`assert_risc_reset_at_core(ERISC0)` halts only BRISC; subordinate NCRISC continues and maintains ETH PHY link. This is safe for brief reset windows. On Wormhole, a full ERISC reset tears down the PHY link entirely. Documented only in inline comments at `fabric_init.cpp:113` and `edm_handshake.hpp:61`.

## Q5: Concrete Documentation Outline

The following should be added to `tech_reports/TT-Fabric/TT-Fabric-Architecture.md`, either as a new top-level section or as subsections under an expanded Section 7.

### Proposed: Section 8 — Fabric Lifecycle and Host Control Plane Operations

```
8  Fabric Lifecycle and Host Control Plane Operations

8.1  Lifecycle Overview
     - DISABLED → ACTIVE → DISABLED state machine
     - SetFabricConfig() as the state transition API
     - Relationship to MeshDevice::create() / MeshDevice::close()
     - "Between tests" model: each test fixture calls SetFabricConfig(FABRIC_1D),
       runs workloads, calls SetFabricConfig(DISABLED); the next test starts fresh
     - No idempotency: calling SetFabricConfig(FABRIC_1D) twice without an
       intervening DISABLED is a fatal error

8.2  Initialization Sequence
     8.2.1  MetalEnvImpl::set_fabric_config()
            - Transition validation (DISABLED↔non-DISABLED only)
            - Channel trimming export on teardown path
     8.2.2  FabricFirmwareInitializer::init()
            - terminate_stale_erisc_routers(): probe, classify, recover stale firmware
            - Channel classification: probe_dead, base_umd, external_umd
            - Relay queue saturation model (4-slot queue, kMaxRelayTimeouts=3)
            - ROM postcode parallel poll (kRomPostcodePollTotalMs=5000ms)
     8.2.3  compile_and_configure_fabric()
            - Device::compile_fabric() → create_and_compile_tt_fabric_program()
            - Device::configure_fabric()
              - Flag reset contract (relay_broken, channels_not_ready, ring_sync_timed_out, etc.)
              - configure_fabric_cores(): soft-reset bounce, partial L1 clear, dead channel skip
              - Pre-launch canary write (0xA0A0A0A0)
              - Launch message dispatch (write_launch_msg_to_core)
     8.2.4  verify_all_fabric_channels_healthy()
            - Ring-sync + per-channel READY_FOR_TRAFFIC confirmation
            - FIX RZ2: stale_base_umd flag clear after healthy verification

8.3  Quiesce and Restart Protocol (quiesce_and_restart_fabric_workers)
     - Purpose: reload fabric firmware between workloads without full teardown
     - Entry guards: FabricConfig check, TensixConfig check, relay-broken early-return
     - ENTRY snapshot: per-device health flags captured before phase execution
     8.3.1  Phase 1 — MUX Terminate Signal
            - Send IMMEDIATELY_TERMINATE to each Tensix MUX worker
     8.3.2  Phase 2 — MUX Termination Poll + BRISC Halt
            - Poll EDMStatus::TERMINATED on each MUX core
            - Unconditional assert_risc_reset_at_core after TERMINATED (close_finish race)
     8.3.3  Phase 2.5 — ERISC Terminate
            - Send terminate signal to each active ERISC channel
            - Poll for TERMINATED; force-reset on timeout
            - Skip entirely for relay-broken non-MMIO devices (FIX R)
     8.3.4  Phase 3 — Reconfigure and Relaunch
            - configure_fabric_cores(): L1 clear + fresh firmware load
            - write_launch_msg_to_core for all healthy channels
            - Deassert Phase 2.5 force-reset channels
     8.3.5  Phase 4 — MUX Ready Wait
            - Poll each MUX core for READY_FOR_TRAFFIC
     8.3.6  Phase 5 — ERISC Handshake (Ring-Sync)
            - Poll master channel for LOCAL_HANDSHAKE_COMPLETE then READY_FOR_TRAFFIC
            - kSyncTimeoutMs = 10s per stage
     8.3.7  Phase 5b — Post-Ready Health Check
            - Per-channel ERISC health verification
            - Degraded-mode diagnosis for partial-mesh scenarios

8.4  Teardown Sequence
     8.4.1  SetFabricConfig(DISABLED) path
            - Channel trimming export
            - teardown_fabric_objects()
            - FabricFirmwareInitializer::teardown(): post_teardown() per device
     8.4.2  Teardown ordering invariant (GAP-78)
            - Dispatch threads MUST be drained before fabric teardown
            - Violation causes ETH L1 corruption persistent across tt-smi -r
            - Enforced via TT_THROW in check_use_count_zero()

8.5  Cross-Session State Model
     - L1 values that persist across process boundaries:
       - edm_status_address sentinels: 0x49706550 (base-UMD), 0xA0A0A0A0 (canary),
         0xDEADB07E (crashed), TERMINATED, READY_FOR_TRAFFIC
       - edm_local_sync_address: stale REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1) hazard
       - fw_launch_addr: consumed vs never-launched detection
     - terminate_stale_erisc_routers() as the cross-session recovery bridge
     - Partial L1 clear (FIX TG2): zero sync addrs, preserve edm_status sentinel

8.6  Degraded-Mode Operation
     - Channel health tiers: healthy, pre-known-dead, newly-dead, base-UMD, external
     - pre-known-dead: continue with warning (degraded mode)
     - newly-dead: TT_THROW (unrecoverable without hardware reset)
     - Per-device flags: fabric_relay_path_broken_, fabric_pre_dead_channels_,
       fabric_external_umd_channels_, fabric_stale_base_umd_channels_
     - Flag lifecycle: cleared at configure_fabric() entry, set during init/quiesce,
       consumed by subsequent phases

8.7  ERISC Soft-Reset Semantics
     - assert_risc_reset_at_core(ERISC0) halts BRISC only; NCRISC maintains PHY
     - Full ERISC reset tears down ETH PHY link (Wormhole)
     - Base-UMD relay channels: soft-reset prohibited (kills non-MMIO relay path)
     - Relay queue model: 4-slot CMD_BUF, stuck commands from timed-out reads
     - Recovery: launch_msg transition (no reset needed for base-UMD → fabric)

8.8  Cost Model
     - SetFabricConfig(FABRIC_1D): full init — topology discovery, routing table build,
       firmware compile, per-device configure_fabric_cores, ring-sync. ~seconds.
     - quiesce_and_restart_fabric_workers(): Phase 1–5b cycle. Typical: <1s per device.
       Worst case with timeouts: up to 120s (ring-sync) + 5s×N (dead channel probes).
     - SetFabricConfig(DISABLED): teardown + channel trimming export. ~hundreds of ms.
     - terminate_stale_erisc_routers(): per-channel probe + classify. Parallel ROM
       postcode poll up to 5s shared deadline. Dead channels add 5s×min(N,3) from
       relay timeouts.
```

### Also needed in BasicEthernetGuide.md
The branch already added the "Host-Side Teardown: L1 Overwrite Ordering" subsection, which is a good start. It should cross-reference the new TT-Fabric-Architecture.md Section 8 for the full lifecycle.

---

## Summary of Files Examined

| File | Lines of Interest | Content |
|------|-------------------|---------|
| `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` | 921–932 | Only init/teardown doc (added by this branch) |
| `tech_reports/EthernetMultichip/BasicEthernetGuide.md` | 537–550 | L1 overwrite ordering (added by this branch) |
| `tt_metal/fabric/fabric_init.hpp` | 18–52, 54–83 | FabricCoresHealth struct, configure_fabric_cores contract, test seam |
| `tt_metal/fabric/fabric_init.cpp` | 94–300 | Soft-reset bounce, dead channel handling, partial L1 clear |
| `tt_metal/impl/device/device.cpp` | 394–720, 727–3400+ | configure_fabric(), 6-phase quiesce protocol |
| `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` | 1041–1200+ | terminate_stale_erisc_routers, channel classification |
| `tt_metal/impl/context/metal_env.cpp` | 100–175, 240–270 | GAP-78 teardown ordering, SetFabricConfig state machine |
| `tt_metal/fabric/fabric_builder_context.hpp` | 28 | FabricConfig::DISABLED enum |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp` | 61 | ERISC soft-reset PHY semantics |
