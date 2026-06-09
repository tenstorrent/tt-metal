# Multichip Sim Rebase — Errors & Fixes

This document records each fix made while rebasing the multichip MP simulator work
onto current bases and getting the N300/P300 fabric smoke tests to run:

- **craq-sim** PR: rebase `nkapre/multichip-mp-daemon` (orig PR #51) onto `main`
  → branch `rsong/multichip-mp-daemon-rebased-main` (PR #83)
- **tt-metal** PR: rebase `nkapre/multichip-mp-ttsim-mock-cluster-rank-binding-20260527`
  (orig PR #45379) onto `blaze-metal-main`
  → branch `rsong/multichip-mp-ttsim-rebased-blaze-metal-main` (PR #46536)

Each section lists the commit, the error it addressed, the root cause, and the fix.

---

## 1. craq-sim `aa187267` — guard WH-only NCRISC IRAM paths for BH build

**File:** `src/tile.cpp`

### Error
Building the Blackhole library (`TT_VERSION=1`, `_out/release_bh/tile.o`) failed:

```
tile.cpp:893: error: 'RV32_IRAM_BASE' was not declared in this scope
tile.cpp:894: error: 'RV32_IRAM_SIZE' was not declared in this scope
tile.cpp:897: error: 'struct TensixTile' has no member named 'ncrisc_iram'
tile.cpp:2935: error: 'RV32_IRAM_BASE' was not declared in this scope
tile.cpp:2955: error: 'struct TensixTile' has no member named 'ncrisc_iram'
```

### Root cause
The MP-daemon branch added NCRISC IRAM handling (reset-PC binary-present guard and
`t_tile_mmio_rd32` IRAM reads) using `RV32_IRAM_BASE`, `RV32_IRAM_SIZE`, and
`TensixTile::ncrisc_iram[]`. Those symbols are **Wormhole-only** — in `sim.h` the
`ncrisc_iram` field is declared under `#if TT_VERSION == 0`, and the IRAM constants
likewise. The new code was compiled unconditionally, so the Blackhole build
(`TT_VERSION == 1`) could not see them.

### Fix
Wrap the NCRISC-IRAM branches in `#if TT_VERSION == 0`:
- In the soft-reset binary-present check, only consult `ncrisc_iram[]` for WH; on
  other archs fall through to the L1/SRAM read and set `pc_in_iram = false`.
- Gate the `t_tile_mmio_rd32` IRAM window read with `#if TT_VERSION == 0`.

Result: both `release_wh` and `release_bh` `libttsim.so` build cleanly.

---

## 2. tt-metal `1308a15c23a` — blaze-metal-main build fixes after MP rebase

**Files:** `tt_metal/llrt/tt_cluster.hpp`, `tt_metal/llrt/tt_cluster.cpp`,
`tt_metal/fabric/physical_system_discovery.cpp`

### Errors
After rebasing the MP commits onto `blaze-metal-main`, the unity build failed with:

```
tt_cluster.cpp:1637: error: no member named 'sim_arm_launch_watcher' in 'tt::umd::Cluster'
fabric_firmware_initializer.cpp:461: error: no member named 'advance_device_execution' in 'tt::Cluster'
physical_system_discovery.cpp:273: error: unused function 'erase_one_sided_connections' [-Werror,-Wunused-function]
```

### Root cause
The MP branch was developed against a newer UMD and a different
`physical_system_discovery.cpp`:
1. `Cluster::sim_arm_launch_watcher(...)` called `get_driver()->sim_arm_launch_watcher(...)`,
   but the UMD pinned on `blaze-metal-main` (`cb0be91a`) does not expose that API.
2. `fabric_firmware_initializer.cpp` calls `cluster_.advance_device_execution(...)`, but
   tt-metal's `Cluster` wrapper had no such method (UMD has it; the tt-metal forwarder
   was missing on this base).
3. The conflict resolution kept `blaze-metal-main`'s `validate_graphs` (which uses
   `TT_FATAL` and does **not** call `erase_one_sided_connections`), leaving that helper
   defined-but-unused under `-Werror`.

### Fix
1. Stub `Cluster::sim_arm_launch_watcher(...)` as a no-op with a comment (UMD/craq-sim on
   this base don't expose the underlying call yet) — keeps the debug hook compiling.
2. Add `Cluster::advance_device_execution(ChipId)` to `tt_cluster.hpp`/`.cpp` forwarding
   to `get_driver()->advance_device_execution(chip_id)`.
3. Restore the `blaze-metal-main` `validate_graphs` variant that *does* call
   `erase_one_sided_connections` (drops one-sided links with a warning instead of
   `TT_FATAL`), so the helper is used and link retraining behavior is preserved.

> Note: during the rebase the upstream commit `aeec9eb9f6e`
> ("restore ClusterDescriptor API in physical_system_discovery") was **skipped** — it
> reverts to a `tt::umd::ClusterDescriptor&` signature that is incompatible with
> `blaze-metal-main`'s `tt::umd::Cluster&` API.

---

## 3. tt-metal `92f56a65b98` — root FD tunnel discovery only at MMIO-capable chips

**File:** `tt_metal/llrt/tunnels_from_mmio_device.cpp`

### Error
The N300 (Wormhole) sim datapath test (`Fabric1DFixture.TestUnicastConnAPI` with the WH
`libttsim.so`) aborted during `MetalContext` init:

```
TT_ASSERT @ tt_metal/llrt/tunnels_from_mmio_device.cpp:31:
  cluster.get_cluster_description()->is_chip_mmio_capable(mmio_chip_id)
 --- tt::llrt::discover_tunnels_from_mmio_device(tt::umd::Cluster&)
 --- tt::Cluster::Cluster(...)
```

### Root cause
In simulation, `Cluster::construct_chip_from_cluster` creates a directly-addressable
`SimulationChip` for **every** chip (there is no `RemoteChip`/gateway in sim). As a
result `Cluster::add_chip` classifies all sim chips as "local" (UMD `cluster.cpp`:
`chip_type == ChipType::SIMULATION || ... || is_chip_mmio_capable(chip_id)`), so
`get_target_mmio_device_ids()` returns **both** chips `{0, 1}`.

`discover_tunnels_from_mmio_device` iterated that set as tunnel roots and asserted each
was MMIO-capable. On the N300 mock descriptor only chip 0 is in `chips_with_mmio`
(chip 1 is the remote/tunnel chip), so chip 1 tripped the assert. The P300 test passed
only because its descriptor (`blackhole_P300_both_mmio.yaml`) marks **both** chips MMIO.

### Fix
Skip non-MMIO-capable chips when choosing tunnel roots:

```cpp
for (const auto& mmio_chip_id : cluster.get_target_mmio_device_ids()) {
    if (!cluster.get_cluster_description()->is_chip_mmio_capable(mmio_chip_id)) {
        continue;  // remote chips are discovered as tunnel stops below
    }
    ...
}
```

This preserves the real N300 topology (chip 0 MMIO, chip 1 remote) — chip 1 is still
discovered as a tunnel stop — and works unchanged on silicon (where every target-MMIO
id is already MMIO-capable). Deliberately **not** worked around by marking both chips
MMIO in the descriptor.

---

## Known remaining issue (in progress) — WH worker FW-init timeout

With fix #3 in place the N300 WH sim test gets past tunnel discovery and device open,
then fails later in firmware init:

```
Device 0: Timeout (10000 ms) waiting for physical cores to finish: 18-18 ... 25-25
Device 0 init: failed to initialize FW! Try resetting the board.
 --- tt::tt_metal::RiscFirmwareInitializer::initialize_and_launch_firmware(int)
```

The full WH worker tensix grid never reports init-FW complete within the 10 s budget
(`firmware_wait_timeout_ms()` returns 10000 for `.so` sim backends). This is a separate,
deeper craq-sim WH simulation/convergence issue (likely related to NCRISC IRAM execution
and per-tile clocking) and is being investigated with the parallel-clock / teleport sim
env vars (`TT_METAL_SIMULATOR_PARALLEL_TENSIX_TILE_CLOCK`, `..._PARALLEL_CLOCK_THREADS`,
`..._L1_TELEPORT`, `..._DRAM_TELEPORT`). Not yet fixed.
