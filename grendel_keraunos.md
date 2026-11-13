# Keraunos on MMK

Keraunos on MMK should reach Metal the way an ERISC core does on Wormhole and Blackhole: the soc descriptor lists the ethernet cores, the firmware initializer loads them, and a test launches a kernel with `EthernetConfig`. The chiplet, the windows, and bring-up follow the Mimir path.

This is a new package. `tt_metal/third_party/umd/tests/soc_descs/mimir_1x1.yaml` and `mimir_2x_package.yaml` stay as they are. They are the one-Mimir and two-Mimir packages. The MMK package, two Mimirs plus Keraunos, is a new file, `tt_metal/third_party/umd/tests/soc_descs/mmk.yaml`.

`bringup_mmk.py` is the chippy bring-up for that package. The Mimir suites keep using `bringup_mimir_1x1.py` and `bringup_two_mimir.py` with their own descriptors.

## What stays the same

- `TargetDevice` selects the transport. `address_space: mimir_local` selects the chippy windows. Dispatch mode stays on `TT_METAL_SLOW_DISPATCH_MODE`.
- Chippy owns clocks, cold reset, the link, D2D, and the uncore. Metal resets the programmable processors and loads firmware.
- A contract test opens a unit `MeshDevice` and uses the public API. It skips from the descriptor before the mesh opens.
- The host writes any base or route the kernel reads. The kernel does not hardcode an MMK address or a D2D port.
- A test does not send `INIT` or `SELECT_CHIPLET`.

## What is different from the Mimir-only packages

Keraunos is chiplet `k0`. The emulator's default chiplet is `k0`, which is why a Mimir socket selects `m0` before it touches a Mimir. `mmk.yaml` records `k0` the same way `mimir_channels` records `m0` and `m1`: chiplet id, an explicit JTAG TAP, and the cores on that chiplet. The Mimir entries in the new file are the same topology as `mimir_2x_package.yaml`.

The programmable cores are ethernet cores. Wormhole and Blackhole list those under `eth:` and launch them with `EthernetConfig`. `mmk.yaml` does the same. The firmware and the L1 map are the Keraunos ones. The Quasar HAL slot that compiles `tt-1xx/active_erisc.cc` is the on-package Quasar ethernet image. Quasar descriptors today have `eth: []`, so that image is not loaded. Keraunos does not reuse it.

`keraunos_address_map.hpp` is the package SPA table (HSIO 0–4 and the other BAR0 rows). It is unvalidated. The chippy path uses the chiplet-local window, the way Mimir uses `0x40000000` for CCE SRAM while the package ATT uses `0x1280000000`. The SPA table is the `full_package` description, for the PCIe step in `grendel_mimir_addressing.md`.

## Proposed sequence

Each step leaves the current Mimir targets green. Those targets keep their descriptors. The new checks run with `mmk.yaml`.

The next work is local Keraunos kernels: a core writes its own SRAM, then a core writes SRAM on the other HSIO tiles. The Mimir link stays as `bringup_two_mimir.py` left it. Releasing Keraunos D2D2 waits for the later D2D step. `bringup_mmk.py` defaults to `MMK_KLINK_RELEASE=none`, which leaves K0 D2D2 in reset. `m1` also leaves the M0 → M1 GDDR copy working. `k0` and `both` release K0 D2D2 and stall that copy.

The core count, the local SRAM window, and the reset registers come from the Keraunos spec and from chippy, the way the CCE hart count and `mimir_cce_control_address` offsets did. This sequence does not invent them from the five HSIO rows in the SPA table.

### Step 0. Baseline

Run the existing Mimir contract suite on `mimir_1x1.yaml` and on `mimir_2x_package.yaml`, and UMD `EmuTTDevice.*`, with the bring-up those descriptors already use. `mmk.yaml` does not exist yet, so this run is the yardstick that the new file and the new code leave those packages alone.

### Step 1. Add `mmk.yaml`

Add `tt_metal/third_party/umd/tests/soc_descs/mmk.yaml`. Start from the two-Mimir contents of `mimir_2x_package.yaml`: `address_space: mimir_local`, both `mimir_channels` entries, the programmable DRAM views, `eth` empty, `functional_workers` empty.

Then add Keraunos without editing the other two files:

- A chiplet entry for `k0`, including the JTAG TAP. The TAP is a field on the entry. It is not the digits of `k0`.
- The ethernet cores that chiplet actually has, under `eth:`, as package-local coordinates of the same kind the Mimir file already uses for its cores. `eth_l1_size` is the Keraunos local SRAM size.
- The local window base for those cores, and which HSIO each core belongs to, in that chiplet entry.

A descriptor with `eth: []` has no Keraunos cases. `mimir_1x1.yaml` and `mimir_2x_package.yaml` stay in that state.

`mmk.yaml` records the chippy map. `k0` is JTAG chain 0 in `mk_bringup` (`m0` is 1, `m1` is 2). HSIO tile 0 TL1 is chiplet-local `0x22800000`, tiles are `0x4000000` apart, and each tile has two 4 MiB CCE regions. The emulator implements HSIO tiles 0–3; hardware tile 4 is not in that model. `bringup_mmk.py` releases CCE0 on those four tiles. `eth` is those four cores, with `x = 4` and `y = hsio`. CCE1 stays off the list until the address step below. The D2D link to M1 stays D2D 2. Opening this descriptor installs the idle-ethernet image. The one-core contract test is step 5.

Check: the new file parses, the two Mimir files are unchanged, and a Mimir suite pointed at `mimir_2x_package.yaml` still skips every ethernet case.

### Step 2. Open `k0` and resolve its window

`EmuTTDevice` opens one chippy transport per chiplet the descriptor names. The Mimir entries still select `m0` and `m1`. The Keraunos entry selects `k0`. A socket opened for a Mimir stays on that Mimir.

`GrendelNocAddressResolver` grows a Keraunos local window on the `mimir_local` path. The ethernet core coordinate selects the core inside that window. SRAM is `local_window + hsio * tile_stride + cce * eth_l1_size` from the descriptor. Cores outside the Quasar mesh, including the Keraunos column, keep falling out of the Quasar config aperture rather than aliasing onto a Tensix tile. The `k0` socket is the last chiplet transport, after the Mimir sockets. JTAG opens that socket from the `tap` field so the same map can be built. The protocol path still returns the first Mimir transport.

Check: a UMD read of one Keraunos core's local SRAM round-trips on the mock and on the MMK model at the flat address the resolver names. The same test on `mimir_2x_package.yaml` skips, because that file has no `k0` entry. Mimir CCE addresses are unchanged.

### Step 3. Reset only the ethernet processors

Keraunos reset is a per-core offset. The ethernet coordinate selects the core, and the resolver adds that core's place in the HSIO tile. Control starts at chiplet-local `0x20000000`, CCE 1 is one 16 MiB slot above CCE 0, and PF_CTRL reset is `0x200000` above the CCE origin. Hart bits are the Mimir PF_CTRL bits. There is no DMRISC remap. `RiscType::ALL` moves the harts and leaves the uncore bit unchanged.

Firmware load asserts and deasserts the ethernet processors. It leaves the uncore, the link, and the remap as `bringup_mmk.py` set them. A mask that also holds the uncore is the Mimir failure mode: a later SRAM store stalls the AXI master, and the stall survives a new client.

Check: two UMD attaches in a row on the MMK model, with no `INIT` from the test. The second attach still reads the core. `TT_UMD_EMU_INIT=1` stays the mock-only switch.

### Step 4. Load firmware and launch an ethernet kernel

When the opened descriptor lists Keraunos ethernet cores, the Quasar HAL installs the ethernet memory map and firmware for `HalProgrammableCoreType::IDLE_ETH`. The processor count is eight, from the PF_CTRL hart field. `l1_noc_offset` is 0: the ethernet coordinate selects the core, and the resolver adds the SRAM window. The device mailbox is the uncached alias at `+0x800000`. The host writes the 0-based offsets. `fw_launch_addr` is 0, and a register-width write there programs `tt_cluster_ctrl` for every hart. A larger write at the same address stays in SRAM.

The firmware is `tt-2xx/idle_erisc.cc`. It is built with `COMPILE_FOR_IDLE_ERISC`, and the host map is compiled with `COMPILE_FOR_ERISC`. It uses that map, the mailbox, the launch message, and the go message. It does not program a NOC. `tt-1xx/active_erisc.cc` stays the image when the descriptor has no Keraunos chiplet. Today's Quasar descriptors have `eth: []`, so that image is not loaded.

Fabric discovery does not mark these cores active, and the idle-ethernet walk would write the tt-1xx map into their SRAM. The initializer walks `logical_eth_core_to_chan_map` instead, holds the harts with `RiscType::ALL_NEO_DMS`, loads the one firmware binary, writes the mailbox, programs the reset vector, and releases the same harts. The uncore stays as bring-up left it. Those cores are idle ethernet, so `CreateKernel(..., EthernetConfig)` uses this map. They are not the dispatch core. `TT_METAL_SLOW_DISPATCH_MODE` stays the user's choice.

Check: the Mimir contract suite on `mimir_1x1.yaml` and `mimir_2x_package.yaml` is unchanged. A Quasar descriptor with `eth: []` still loads no ethernet firmware.

### Step 5. One-core contract test

Add a contract case that runs only when the selected descriptor has ethernet cores. It opens a unit `MeshDevice`, launches one `EthernetConfig` kernel on ethernet core 0, and has that kernel write one word to its local SRAM. The host reads it back with the HAL NOC address for that core.

The kernel reads a host-written base when it needs another chiplet. The first kernel writes its own SRAM and does not take an MMK address as a compile argument.

The case skips on `mimir_1x1.yaml` and `mimir_2x_package.yaml`. It skips on `mmk.yaml` while `TT_METAL_SLOW_DISPATCH_MODE` is unset. Bring the model up with `bringup_mmk.py` first. Run the case twice without bringing the model up again. The second run catches a reset that left the uncore held.

Watcher and DPRINT follow once the mailbox layout matches the ERISC one. They are not part of this step.

Check: the one-core case on the MMK model, and a skip on both Mimir-only descriptors.

### Step 6. SRAM on the other HSIO tiles

Repeat the firmware load and the SRAM round trip for every ethernet core `mmk.yaml` lists. The count comes from that list. A core the emulator does not back stays out of the file, the same way a DRAM view stays inside the GDDR the model actually has.

A kernel that touches another HSIO tile reads a host-written table, filled from `keraunos.local_window`, `tile_stride`, and the core list after firmware load. The kernel does not hardcode an MMK address. The table names chiplet-local SRAM. It does not name a Mimir or a D2D port. `bringup_mmk.py` still owns the remap that makes those windows reachable. A path onto a Mimir waits for the D2D step.

Check: every listed core on the MMK model, including a write from one HSIO tile into another's SRAM. The Mimir GDDR and CCE suites on `mimir_2x_package.yaml` stay on their own bring-up.

### Later. The Keraunos D2D path

This waits until the local kernels are green.

`bringup_mmk.py` can stage M1 D2D1 and K0 D2D2 and release them with `MMK_KLINK_RELEASE`. Releasing M1 D2D1 alone leaves `CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition2` passing. Releasing K0 D2D2, alone or with M1 D2D1, stalls that copy on Mimir 0 CCE0. The stalled access is M0 D2D1 → M1 D2D0 → M1 GDDR. Keraunos fabric init and the CCE SRAM remap are in the passing `none` and `m1` runs, so they are not the stall.

The host helper programs Keraunos BARs, the endpoint table, and the ATT, then the HSIO2 NOC2AXI remap after `initialize_fabrics`. That remap is the nine-window table from `set_ker_noc2axi_remappers()`. The emulator's `qsr1_boot` table for HSIO2 is five entries and includes `0x8_0000_0000_0000` → HSIO2 SRAM at `0x2A80_0000`. The nine-window table does not. The next D2D experiment is to program the emulator's per-tile table, then release K0 D2D2 again.

Check: the cross-Mimir GDDR copy still passes with K0 D2D2 released, and a Keraunos ↔ Mimir access uses that link.

### Later. CCE1 addresses

This waits until the D2D path is green.

Each HSIO tile has a second CCE. Its SRAM is the next 4 MiB of the tile, and the current CPU map puts that core's instructions, mailbox, and reset vector at local +4 MiB. A caller should not have to add that offset. CCE1 should take the same local addresses as CCE0. The host names the core, and the resolver plus `bringup_mmk.py` place that core's window. `mmk.yaml` then lists `5-y` beside `4-y`.

Until then the ethernet list is the four CCE0 cores, the idle-ethernet image addresses local 0, and the reset vector is the host-supplied value with no CCE term added.

Check: a kernel on CCE1 writes its own SRAM at the same local offset a CCE0 kernel uses, and the host reads that word from CCE1's window.

### Step 7. JTAG

This waits until the emulator path is green. Boards are scarce.

The TAP for `k0` is the field from step 1. Data goes through the per-chiplet flat transport. The protocol path that sends every core to the first transport is the Mimir limitation already recorded in `grendel_mimir_addressing.md`, and a Keraunos access uses the flat transport instead.

Check: the step 5 case on JTAG with `mmk.yaml`.

## Decisions

- **New descriptor.** `mmk.yaml` is the two-Mimir plus Keraunos package. The one-Mimir and two-Mimir files stay the regression for packages that have no Keraunos.
- **Same launch path as ERISC.** Logical ethernet cores, `IDLE_ETH`, `EthernetConfig`. The memory map and the firmware are the eight-hart image. The tt-1xx active ERISC image stays with on-package Quasar ethernet.
- **Who owns bring-up.** `bringup_mmk.py` releases the uncore and brings up the link and D2D. Metal resets the ethernet processors.
- **Address space.** Chippy uses the chiplet-local window. `keraunos_address_map.hpp` is the package SPA table for a later `full_package` path.
- **Dispatch.** `TT_METAL_SLOW_DISPATCH_MODE` still chooses. A Keraunos case on a package with no Tensix cores skips while that variable is unset.
