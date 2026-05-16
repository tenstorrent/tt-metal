<!--
SUMMARY: Analysis of ERISC/BRISC reset mechanisms on Wormhole, feasibility of single-ERISC reset without PHY teardown, and gaps for implementing #42429
KEYWORDS: ERISC, BRISC, reset, PHY, ETH, fabric, Wormhole, soft_reset, corrupt, recovery, #42429
SOURCE: tt-metal codebase analysis (commit d6cff88638 on nsexton/0-racecondition-hunt)
SCOPE: Wormhole architecture ETH core reset registers, host-side and firmware-side reset paths, halt detection, recovery mechanisms
USE WHEN: Working on #42429 (surgical per-channel ERISC reset), debugging corrupt ERISC state after process crash, understanding fabric initialization/teardown
-->

# ERISC/BRISC Reset Mechanisms on Wormhole

## 1. Architecture: ETH Core RISC Processors

Each Wormhole ETH (Ethernet) core contains multiple RISC processors sharing the same debug register space:

- **BRISC (ERISC0)**: Main Ethernet RISC — runs fabric router firmware or base UMD firmware
- **NCRISC (subordinate ERISC)**: Subordinate processor — can reset BRISC via register writes
- **TRISCs**: Compute RISC processors (rarely used on ETH cores)

The key register is:

```
RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
  (RISCV_DEBUG_REGS_START_ADDR | 0x1B0, where START = 0xFFB12000)

Bit fields (from tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h):
  RISCV_SOFT_RESET_0_NONE   = 0x00000  (all running)
  RISCV_SOFT_RESET_0_BRISC  = 0x00800  (halt BRISC)
  RISCV_SOFT_RESET_0_NCRISC = 0x40000  (halt NCRISC)
  RISCV_SOFT_RESET_0_TRISCS = 0x07000  (halt all TRISCs)
```

This is the same register referenced as `ETH_RISC_RESET` at offset `0x21B0` in `tt_eth_ss_regs.h`.

## 2. Existing Reset Mechanisms

### 2.1 UMD Host-Side Reset (`assert_risc_reset_at_core` / `deassert_risc_reset_at_core`)

Path: `Cluster::assert_risc_reset_at_core()` -> `driver_->assert_risc_reset(chip, core_coord, soft_resets)`

This writes to `RISCV_DEBUG_REG_SOFT_RESET_0` on the target core, setting the bits corresponding to the `RiscType` argument. Key `RiscType` values:

- `RiscType::ALL` — halts all RISCs on the core (BRISC + NCRISC + TRISCs + ERISC0)
- `RiscType::ALL_TENSIX` — halts BRISC/NCRISC/TRISCs but NOT ERISC0
- `RiscType::BRISC` — halts only BRISC
- `RiscType::ERISC0` — halts only the main ERISC

**Critical behavior on Wormhole**: When `RiscType::ALL` is asserted on an ETH core, the ERISC stops executing. The ETH PHY link management runs partially in firmware — if the ERISC is halted for too long, the PHY link drops. The comment in the codebase (multiple locations) warns:

> "resetting a WH ERISC tears down the ETH PHY link and breaks non-MMIO L1 access for the rest of the mesh"

**The assert+deassert bounce pattern**: `risc_firmware_initializer.cpp` lines 542-550 show a workaround:
```cpp
cluster_.assert_risc_reset_at_core(core, RiscType::ALL);
// Immediately deassert to return ERISC to base firmware
cluster_.deassert_risc_reset_at_core(core, RiscType::ALL);
```
This quickly resets the ERISC and restarts it into base firmware. The comment says without the deassert, "ETH PHY links go down, and concurrent topology discovery / dispatch fabric operations hang or crash."

### 2.2 Firmware-Side Soft Reset (Subordinate ERISC Resets BRISC)

In `tt_metal/hw/firmware/src/tt-1xx/subordinate_erisc.cc` line 119-126:
```c
// Trigger a soft reset of ERISC0. Wait 100 cycles, and then deassert
WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_BRISC);
riscv_wait(100);
WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NONE);
```

The subordinate ERISC (NCRISC-equivalent) can reset just the BRISC without touching the PHY — because the subordinate ERISC continues running and can manage link retraining if needed. This is the **only existing mechanism for resetting just the BRISC on an ETH core without full core reset**.

### 2.3 Normal Teardown: `assert_active_ethernet_cores_to_reset()`

In `risc_firmware_initializer.cpp` line 344-353:
```cpp
// For cooperative ETH mode (WH):
tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
```

Normal teardown halts everything EXCEPT ERISC0. The main ERISC keeps running (to maintain the PHY link and run base firmware). If in 2-ERISC mode, it first calls `return_to_base_firmware_and_wait_for_heartbeat()` to switch ERISC0 back to base firmware before halting the subordinate processors.

### 2.4 `tt-smi -r` (Full Chip Reset)

`tt-smi -r` is a full chip reset invoked externally. It is NOT callable from within tt-metal as a library function. The codebase mentions it only in documentation and test scripts:
- `tt_metal/tt-llk/tests/python_tests/helpers/hardware_controller.py` calls `run_shell_command("tt-smi -r")`
- `fabric_firmware_initializer.cpp` line 831: "Corrupt channels require a tt-smi chip reset to recover"

There is no programmatic API equivalent to `tt-smi -r` exposed within tt-metal. It operates at the PCIe/ARC level, resetting the entire device.

## 3. Can We Detect a Halted BRISC from the Host?

### 3.1 Reading `RISCV_DEBUG_REG_SOFT_RESET_0` (0xFFB121B0)

Yes. The `fabric_erisc_dumper.py` tool already does this:
```python
reset_status = self.read_address(device, loc, 0xFFB121B0)
if reset_status != 0:
    status_info.append("not_reset")  # some RISC is halted
else:
    status_info.append("in_reset")   # all RISCs running (or all halted?)
```

**Interpretation**: Reading this register from the host shows which RISC soft-reset bits are currently asserted. If bit `0x800` (BRISC) is set, the BRISC is halted.

**Caveat**: The register reads `0` when no RISC is in soft reset. A value of `0` means the BRISC is running (or has been de-asserted). But a running BRISC might be running *base firmware* (not fabric router firmware) — the register doesn't tell you *what* firmware is executing.

### 3.2 Wall Clock Registers

```
ETH_RISC_WALL_CLOCK_0 = 0xFFB121F0  (low 32 bits, free-running)
ETH_RISC_WALL_CLOCK_1 = 0xFFB121F4  (high 32 bits)
```

These are hardware counters that increment regardless of RISC execution state. They are NOT useful for detecting whether the BRISC is halted.

### 3.3 Performance Counter / Instruction Count Registers

```
RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0 = 0xFFB12000
```

Could potentially be read twice with a delay to detect if the instruction counter is advancing (BRISC running) or stalled (BRISC halted). This is NOT currently used anywhere for halt detection but could be a reliable mechanism.

### 3.4 L1 Status Word (Current Approach)

The current approach reads `router_sync_address` from L1 and checks if the value is a valid `EDMStatus`. This is an indirect check — a corrupt/non-EDMStatus value strongly suggests the BRISC is halted mid-handshake, but it's not definitive. A halted BRISC could also have a valid EDMStatus value at the moment it was halted.

## 4. What `Device::close()` Does

`Device::close()` itself (line 917 in `device.cpp`) does NOT directly halt any cores. It clears queues and resets internal state.

The BRISC halt happens during the **firmware initializer teardown path**:
1. `RiscFirmwareInitializer::teardown()` -> `assert_active_ethernet_cores_to_reset()` halts all RISCs except ERISC0
2. `FabricFirmwareInitializer::teardown()` sends TERMINATE to fabric routers, polls for TERMINATED, and on timeout calls `assert_risc_reset_at_core(RiscType::ALL)` which halts ERISC0 too

When a process is **killed** (SIGKILL), none of these teardown paths execute. The ERISC cores are left in whatever state they were in — potentially mid-handshake with corrupt L1 values. The BRISC continues running but executing code that will never make progress (spinning on a handshake that will never complete, or executing in a tight loop).

## 5. Tech Report Findings

### BasicEthernetGuide.md

Key passage (line 546):
> "On Wormhole hardware, `assert_risc_reset_at_core()` on an ERISC tears down its ETH PHY link. This breaks non-MMIO L1 access for the full mesh, so force-reset should be used only as a last resort for truly unresponsive firmware, and must be followed by `deassert_risc_reset_at_core()` to restore base firmware before new firmware can be loaded."

Also (line 126):
> "Ethernet links may occasionally go down from time to time... retraining requires the assistance of the ERISC to execute retraining routines in software. Therefore it is important to allow any user written kernels to be written with an eventual code path to the lower level ethernet firmware."

### TT-Fabric-Architecture.md

Key passage (line 929):
> "The host must poll the status mailbox of **every active ERISC channel** and confirm `EDMStatus::TERMINATED` before overwriting that channel's L1 with new firmware."

## 6. The PHY Teardown Problem: Why It Happens

The ETH PHY link on Wormhole requires periodic link maintenance run in firmware (via `run_routing()` calls). When all RISCs on an ETH core are halted:

1. No firmware runs to handle link keepalive
2. The PHY link times out and goes down
3. Non-MMIO chips lose connectivity (they route through ETH links)
4. L1 reads/writes to remote chips fail

The assert+deassert bounce (lines 542-550 in `risc_firmware_initializer.cpp`) works because:
- Assert halts all RISCs, clearing any stale firmware state
- Deassert restarts all RISCs from reset vectors
- ERISC0 boots into base UMD firmware which handles link maintenance
- The window where all RISCs are halted is brief (microseconds, limited by PCIe round-trip)

## 7. Feasibility Assessment: Single-ERISC Reset Without PHY Teardown

### Option A: Host-Side assert+deassert Bounce (Current Workaround)

**How it works**: Write `RiscType::ALL` to the soft-reset register, immediately deassert.
**PHY impact**: Brief — the ERISC is halted for only the PCIe round-trip time (microseconds). If link keepalive timers are in the range of milliseconds, this should be safe.
**Already used**: Yes, in `reset_cores()` for cooperative ETH mode (lines 542-550).
**Gap**: No one has verified this doesn't occasionally drop the PHY on WH under load. The F5a experiment in `metal_env.cpp` line 327 deliberately SKIPS force-reset to test whether it causes hangs.

### Option B: Subordinate-ERISC-Assisted Reset

**How it works**: The subordinate ERISC (NCRISC-equivalent) writes `RISCV_SOFT_RESET_0_BRISC` to soft-reset only the BRISC, waits 100 cycles, then de-asserts.
**PHY impact**: None — the subordinate ERISC continues running and can handle link maintenance.
**Problem**: The subordinate ERISC must be alive and running known-good firmware. In the corrupt-state scenario, the subordinate ERISC may also be in an unknown state.

### Option C: Host Writes to `RISCV_DEBUG_REG_SOFT_RESET_0` Directly

**How it works**: Instead of `assert_risc_reset_at_core(RiscType::ALL)`, the host writes `RISCV_SOFT_RESET_0_BRISC` (0x800) to the ETH core's `RISCV_DEBUG_REG_SOFT_RESET_0`, waits briefly, then writes `RISCV_SOFT_RESET_0_NONE` (0x0).
**PHY impact**: The NCRISC/subordinate ERISC stays running — it can maintain the link while BRISC is briefly halted.
**Feasibility**: `RISCV_DEBUG_REG_SOFT_RESET_0` is in the WH HAL's valid register whitelist (line 314 in `wh_hal.cpp`). The host can write to it via the cluster driver's `write_core()` path.
**Gap**: Nobody has tried this from the host side on WH ETH cores. The subordinate ERISC firmware may need to be in a specific state to handle the BRISC reset correctly (e.g., it needs to save/restore BRISC state or re-trigger IRAM load).

### Option D: `IERISC_RESET_PC` Register (BH/Quasar Only)

Blackhole and Quasar have dedicated `IERISC_RESET_PC` and `SUBORDINATE_IERISC_RESET_PC` registers for programming the ERISC reset vector. **These do not exist on Wormhole.** Not applicable for the current WH-focused #42429.

## 8. Current Gaps and What Needs to Change for #42429

### Gap 1: No Host-Side BRISC-Only Reset Path

The host always uses `assert_risc_reset_at_core` which maps to `RiscType::ALL` in the force-reset paths. There is no existing code path for the host to reset only the BRISC while keeping the subordinate ERISC alive on WH.

**To implement**: Need a `reset_erisc_brisc_only()` helper that:
1. Reads current soft-reset register state
2. Writes `RISCV_SOFT_RESET_0_BRISC` to halt only BRISC
3. Waits a few microseconds
4. Writes `RISCV_SOFT_RESET_0_NONE` to de-assert
5. Optionally triggers IRAM reload for the new firmware

### Gap 2: No Verification That BRISC Restarted Successfully

After the assert+deassert bounce, there's no check that the BRISC actually started executing the new firmware. `configure_fabric_cores()` writes L1 (zeroing buffers) and `write_launch_msg_to_core()` sends a GO message, but if the BRISC is still halted (deassert failed, or subordinate ERISC interfered), the firmware never starts.

**To implement**: After firmware load and GO message, poll `router_sync_address` with a short timeout (100ms) to verify the ERISC progressed past `STARTED`. If it doesn't, the channel is dead and should be reported.

### Gap 3: No Halt Detection Before Recovery Attempt

`terminate_stale_erisc_routers()` classifies channels as CORRUPT vs STALE_RUNNING based on L1 values, but doesn't check if the BRISC is actually halted. A CORRUPT L1 value doesn't necessarily mean the BRISC is halted — it could be running in a tight loop.

**To implement**: Before attempting recovery, read `RISCV_DEBUG_REG_SOFT_RESET_0` on the ETH core. If bit `0x800` is set, BRISC is halted and must be de-asserted before any L1 write/GO message will have effect. Also consider reading instruction count registers twice to detect spinning cores.

### Gap 4: PHY Link Verification After Reset

After any ERISC reset (even a brief bounce), there's no verification that the PHY link is still up. If the link dropped during the reset window, non-MMIO L1 access will fail silently or with confusing errors.

**To implement**: After reset, verify ETH link status via the appropriate MAC registers or by attempting a small L1 read on the remote chip.

### Gap 5: No Tracking of Force-Reset Channels Across Sessions

Partially addressed — `force_reset_channels_` set in `FabricFirmwareInitializer` tracks channels force-reset during teardown, and `verify_all_fabric_channels_healthy()` uses it to classify DEGRADED channels. But this only works within the same process lifetime. Cross-process tracking would require persistent state (shared memory or file).

## 9. Recommended Implementation Plan for #42429

1. **Implement host-side BRISC-only soft-reset** (Option C): Write `RISCV_SOFT_RESET_0_BRISC` to `RISCV_DEBUG_REG_SOFT_RESET_0`, wait, deassert. This is the safest approach because the subordinate ERISC continues running and can maintain the PHY link.

2. **Add halt detection**: Before recovery, read the soft-reset register to classify the core state (halted BRISC vs running BRISC with stale L1).

3. **Verify on hardware**: Test the BRISC-only reset on a real T3K under load to confirm the PHY link stays up. Key test: reset BRISC on one channel while traffic flows on adjacent channels on the same chip.

4. **Add post-reset verification**: After `configure_fabric_cores()` + GO message, poll for `EDMStatus::STARTED` or later within 100ms. Report dead channels if the ERISC doesn't progress.

5. **Update `terminate_stale_erisc_routers()`**: Replace the current "send TERMINATE best-effort, skip poll" path for CORRUPT channels with:
   a. Read soft-reset register to confirm BRISC is halted
   b. If halted: assert BRISC-only soft reset, wait, deassert
   c. After deassert: BRISC starts base firmware
   d. `configure_fabric_cores()` overwrites L1 with new firmware
   e. GO message starts the router
   f. Poll for STARTED/READY_FOR_TRAFFIC

## Key Files Referenced

- `/workspace/group/worktrees/racecondition-main/tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` — main recovery logic
- `/workspace/group/worktrees/racecondition-main/tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` — assert/deassert patterns
- `/workspace/group/worktrees/racecondition-main/tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h` — register definitions
- `/workspace/group/worktrees/racecondition-main/tt_metal/hw/inc/internal/ethernet/tt_eth_ss_regs.h` — ETH register definitions
- `/workspace/group/worktrees/racecondition-main/tt_metal/hw/firmware/src/tt-1xx/subordinate_erisc.cc` — firmware-side BRISC reset
- `/workspace/group/worktrees/racecondition-main/tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp` — register whitelist
- `/workspace/group/worktrees/racecondition-main/tt_metal/llrt/tt_cluster.cpp` — host-side reset API
- `/workspace/group/worktrees/racecondition-main/tt_metal/impl/context/metal_env.cpp` — F5a experiment
- `/workspace/group/worktrees/racecondition-main/tt_metal/fabric/fabric_init.cpp` — configure_fabric_cores() (L1 clear)
- `/workspace/group/worktrees/racecondition-main/tt_metal/fabric/debug/fabric_erisc_dumper.py` — halt detection via register read
- `/workspace/group/worktrees/racecondition-main/tech_reports/EthernetMultichip/BasicEthernetGuide.md` — ETH reset documentation
- `/workspace/group/worktrees/racecondition-main/tech_reports/TT-Fabric/TT-Fabric-Architecture.md` — fabric teardown requirements
