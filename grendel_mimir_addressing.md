# Two Grendel access flows

The runtime has to support two ways of reaching Mimir / DRISC (CCE) cores, through the same Metal and UMD APIs. Tests may differ by which chiplets are present and how they are wired, but they should call the same APIs.

1. **Mimir-local.** Only Mimirs and Keraunos, reached through chippy. FPGA emulation uses the emu AXI server. Silicon uses chippy plus JTAG. Addresses are chiplet-local.
2. **Full Grendel.** Quasar, Keraunos and Mimir together, reached over PCIe, or whatever the FPGA equivalent of PCIe is. Addresses are the package NOC address translation table (ATT), or another logical address space.

These two flows do not share the runtime today. Metal mostly asks for addresses the same way on both (core coordinates, HAL addresses, DRAM views). What sits underneath does not, and three things carry most of the gap:

- Transport and addressing are fused. Talking to chippy implies Mimir-local addressing, and talking to anything else implies package ATT addressing.
- The CCE control plane is Mimir-specific. Reset, the reset vector and the address remap all assume the chippy model's layout.
- Kernels and tests hardcode the MMK package. They assume MMK's Mimir numbering and D2D wiring, and the tests call backend-specific internal APIs.

The two current chippy paths, emulation and JTAG, already diverge from each other.

## What differs today

### 1. Transport and address space are separate

Done in step 4. `GrendelTTDevice` is the device. It is built from a protocol (`TTDeviceModel`), a `GrendelNocAddressResolver` from `address_windows_for`, and a `FlatAddressTransport` chosen by the descriptor's address space. `MimirChippyMemoryMap` is the `mimir_local` implementation: it folds the flat address back to a chiplet-local AXI address on that Mimir's transport. The `full_package` implementation is one transport, and the flat address is the SPA. `EmuTTDevice` and `GrendelJtagTTDevice` only open the protocol. Their sockets are still chiplet-local, so they still require `address_space: mimir_local`. PCIe, and the FPGA equivalent, add a protocol rather than a device class.

`RtlSimulationTTDevice` still installs its own resolver and is not this device. `TTDevice::create` still has no Grendel PCIe case (step 15).

### 2. CCE control is one implementation on the Mimir-local path

Done in step 2. Reset vector, PF_CTRL and DMRISC remap offsets are relative to one CCE (`0x02000000`, `0x02200000`, `0x02202000`). The DRAM-core coordinate selects the CCE. `GrendelNocAddressResolver::cce_control_address` produces the flat config address, and the flat-address transport turns that into the access. On the Mimir-local path that is `MimirChippyMemoryMap`. Emulation and JTAG share that path.

### 3. The descriptor selects the address space

Done in step 3. `address_space: mimir_local` selects the chiplet-local windows, including the `0x2000000000` CCE SRAM tag. `address_space: full_package`, and a descriptor that omits the key, selects the qsr.s1 ATT. That table now has CCE SRAM at `0x1280000000` (4 MiB per CCE), Mimir config at `0x1300000000` (128 MiB per Mimir, lane 0), and CCE control as a per-CCE offset in that window. In package mode the HAL `l1_noc_offset` for a DRAM core is `0x1280000000`. The `0x2000000000` tag stays on the Mimir-local path. Emulation and JTAG still require `mimir_local`.

### 4. The descriptor records the Mimir topology

Done in step 5. Each DRAM channel lists its SPA instance, its chippy chiplet id, and the D2D port that reaches each peer. `mimir_1x1.yaml` is channel 0, instance 0, chiplet `m0`. `mimir_2x_package.yaml` is the MMK wire: M0 D2D1 to M1, M1 D2D0 to M0. Emulation selects that chiplet id. `program_cce_dmrisc_remap` takes the SPA bounds from the package windows and the instance from the channel entry. A descriptor that omits `mimir_channels` (the Quasar simulation descriptors) still numbers the instance by DRAM channel. JTAG opens one transport per entry. One attached chiplet still uses `TT_METAL_GRENDEL_JTAG_CHIPLET` as its TAP. A second entry opens the TAP in its chiplet id.

### 5. Bring-up owns the uncore and the remap

Done in step 6. Chippy brings up clocks, cold reset, GDDR, D2D, firewalls and the remaps. SMC firmware owns uncore release and the DMRISC and MLA remaps on full Grendel. Chippy fills that role on the Mimir-local path.

Metal's DRAM firmware load asserts and deasserts `RiscType::ALL_NEO_DMS`. That is the eight CCE harts. It does not hold the uncore, and it does not call `release_cce_uncore_for_firmware_load`, so it does not rewrite the remap. Teardown still resets harts with `RiscType::ALL_DATA_MOVEMENT` and leaves the uncore running.

### 6. Device-side addresses come from the host

Done in step 7. `cce_gddr_*` and `cce_sram_*` read `CceSpaRoutes` immediately after the CCE bank-to-NOC staging range. The host fills that table per CCE after firmware has copied the staging tables, from the package windows and from `mimir_channels` routes. Local GDDR uses the D2D0 window. A recorded peer route selects the other port. With no topology, Mimir 0 still reaches other instances through D2D1. Tensix kernels still use the NOC/ATT and do not read this table.

### 7. Metal picks behaviour from the descriptor

Done in step 8. `TargetDevice::EmuAxi` and `TargetDevice::GrendelJtag` only select the transport. `tt_cluster.cpp` builds the matching `ClusterOptions` from `TT_METAL_EMU_*` and `TT_METAL_GRENDEL_JTAG_*`, including `emu_skip_init` so chippy's bring-up is kept. `rtoptions` turns those targets on from the same variables and leaves dispatch on `TT_METAL_SLOW_DISPATCH_MODE`.

A global Tensix reset runs for every target except Mock and Emule. UMD skips it when the descriptor has no Tensix cores. Dispatch mode stays with `TT_METAL_SLOW_DISPATCH_MODE`: set selects slow dispatch, and unset selects fast dispatch.

The Quasar choices in `risc_firmware_initializer.cpp` (reset CCE harts, leave the uncore and the remap to bring-up, tear down with the uncore left released) stay keyed on the architecture. Worker firmware init already follows the Tensix grid size, so a full Grendel package initializes workers and a Mimir-only package does not.

`Cluster::is_mock_or_emulated()` is Mock and Emule only. EmuAxi and GrendelJtag take the silicon firmware path.

### 8. The tests are tied to the backend

`tests/tt_metal/tt_metal/context/test_mimir_emu.cpp` once mixed public APIs with a direct `Device` open and raw DRAM-channel access. The contract now opens a unit mesh and reaches GDDR through its helpers. It runs when a chippy server or `TT_METAL_SOC_DESC` names a package with programmable DRAM. A Mimir-only package also needs `TT_METAL_SLOW_DISPATCH_MODE`.

`tt_metal/third_party/umd/tests/api/test_emu_tt_device.cpp` checks results through `EmuTTDevice`, including a flat read at the address `GrendelNocAddressResolver` names. It does not open a second chippy client.

## Address spaces in use

| What | Address space |
| --- | --- |
| Per-CCE offsets plus `cce_control_address` | CCE register plane relative to one CCE. Reset vector `0x02000000`, PF_CTRL `0x02200000`, DMRISC remap `0x02202000`. The DRAM core selects the CCE. |
| `program_cce_dmrisc_remap` | Package windows for the channel's SPA instance. Config, both CCE SRAMs, then that Mimir's GDDR. |
| Default `GrendelAddressWindows` | Package ATT flat addresses for qsr.s1. NEO L1, Quasar config, GDDR, CCE SRAM `0x1280000000`, Mimir config `0x1300000000`. |
| `mimir_local` `FlatAddressTransport` (`MimirChippyMemoryMap`) | Chiplet-local AXI. Config at 0, CCE SRAM at `0x40000000`, GDDR at `0x800000000`, CCE SRAM tag `0x2000000000`. |
| `full_package` `FlatAddressTransport` | One transport. The flat address is the SPA. |

| Who is looking | CCE SRAM | GDDR |
| --- | --- | --- |
| Host, through Metal | DRAM core coordinate plus `MEM_CCE_L1_NOC_OFFSET` (`0x2000000000`), or `get_dev_noc_addr` | NOC worker endpoint plus the view's `address_offset` |
| CCE kernel | Its own SRAM at local 0; another CCE's SRAM from `CceSpaRoutes` | The route base in `CceSpaRoutes`, through a volatile pointer |
| Firmware link | `MEM_CCE_SRAM_LOCAL_BASE == 0` | — |

## Proposed sequence

Each step should leave the current targets green before the next one starts. The current targets are:

- the chippy mock
- a single-Mimir SiVal model
- the MMK SiVal model
- the UMD resolver golden tests (`tests/baremetal/test_grendel_noc_address_resolver.cpp`)
- the Quasar ZeBu regression (`tests/scripts/quasar/quasar_regression_tests.yaml`)

JTAG silicon is step 16. Boards are scarce, so it runs after PCIe, and only when a board is already up.

### Step 0. Baseline

Record one scripted run, per target, of: bring-up, UMD `api_tests` with `EmuTTDevice.*`, the Metal Mimir suite run twice in a row without bringing the model up again, and the resolver and Quasar regressions. The second Metal run catches a test that left the CCE uncore held.

Run the emulator baseline on both a single Mimir (`mimir_1x1.yaml`, `bringup_mimir_1x1.py`) and MMK with both Mimirs up (`mimir_2x_package.yaml`, `bringup_two_mimir.py`). The single-Mimir server ignores `SELECT_CHIPLET`, so it is the run that catches a change which starts requiring a select. MMK with `mimir_1x1.yaml` only touches `m0` and does not count as the two-Mimir run. The mock, the resolver golden tests, and the Quasar ZeBu regression need neither model.

### Step 1. Split the tests into a shared contract suite and backend suites

Done. The contract suite is `tests/tt_metal/tt_metal/context/test_mimir_emu.cpp`. Shared helpers are `mimir_contract.hpp`. Raw `Cluster` access is `test_mimir_backend.cpp`. Both are listed in `sources.cmake`.

The contract suite uses a unit `MeshDevice`, its slow-dispatch helpers, buffers, programs, and `write_cce_sram` / `read_cce_sram`. It skips unless the selected package has programmable DRAM cores, and it sizes itself from the descriptor (how many CCEs, how many Mimirs, which pairs can reach each other).

Gtest names are unchanged: `MimirEmu.*`, `MimirEmuSameMimirDirect.*`, `MimirEmuCrossMimirNoc.*`, plus the two Cluster cases now compiled from the backend file.

Check: same pass and fail as the baseline. This suite is the yardstick for every later step.

### Step 2. One CCE control plane

Done. `GrendelNocAddressResolver::cce_control_address` takes a DRAM core and a per-CCE offset. `EmuTTDevice` and `GrendelJtagTTDevice` both write that flat address through `MimirChippyMemoryMap`. The chippy register-accessor overrides and `cce_control_addr` are gone.

Check: UMD reset tests on the mock and on SiVal, the Metal suite, and JTAG. Host-side, `GrendelNocAddressResolver.CceControlOffsetIsRelativeToOneCce` covers the address math.

### Step 3. Choose the address space from the descriptor, and finish the package windows

Done. The descriptor key is `address_space`. `mimir_1x1.yaml` and `mimir_2x_package.yaml` set `mimir_local`. A descriptor that omits the key, including the Quasar simulation descriptors, stays `full_package`. `address_windows_for` is what `EmuTTDevice`, `GrendelJtagTTDevice`, and `RtlSimulationTTDevice` call. Emulation and JTAG reject `full_package` because their transport is still chiplet-local.

Package windows, from `models/qsr.s1/att/address_map.csv`: CCE SRAM `0x1280000000` stride 4 MiB, Mimir config `0x1300000000` stride 128 MiB. CCE control is the per-CCE offset in that window. Package-mode HAL `l1_noc_offset` for a DRAM core is `0x1280000000`. Mimir-local keeps `MEM_CCE_L1_NOC_OFFSET` (`0x2000000000`). `RtlSimulationTTDevice` keeps its existing coordinate-plus-core-local-address protocol for every `full_package` descriptor; it does not infer an ATT geometry from the descriptor size.

Check: `GrendelNocAddressResolver.PackageCceSramMatchesGoldenAddressMap` and `PackageMimirConfigCarriesCceControl`. Behaviour on current Mimir hardware does not change.

### Step 4. Separate the transport from the address space in UMD

Done. `FlatAddressTransport` is the interface. `MimirChippyMemoryMap` is the `mimir_local` implementation. The `full_package` implementation is one transport and the flat address is the SPA. `GrendelTTDevice` holds the protocol, the resolver, and that transport. `EmuTTDevice` and `GrendelJtagTTDevice` open the protocol and still reject `full_package`, because those sockets are chiplet-local. A PCIe or FPGA protocol is another factory for the same device.

Check: `GrendelFlatAddress.MimirLocalTranslatesToChipletAxi` and `GrendelFlatAddress.FullPackagePassesTheSpaThrough`. Behaviour on current Mimir hardware does not change, so the step 0 targets stay the regression.

### Step 5. Put the topology in the descriptor

Done. `mimir_channels` records, per DRAM channel, the SPA instance, the chippy chiplet id, and the D2D route to each peer. Emulation selects that id. The remap uses the package windows and that instance. JTAG no longer refuses a second Mimir: the second entry is a second TAP. A swapped `m0` / `m1` in this list is what MMK shows.

Check: `SocArchDescriptor.Mimir1x1RecordsOneChiplet`, `SocArchDescriptor.Mimir2xRecordsMmKRoutes`, and `GrendelNocAddressResolver.CceDmriscRemapUsesPackageWindows`. On MMK, the two-Mimir descriptor with the chiplets reversed would program the wrong die.

### Step 6. Metal only resets harts

Done. DRAM firmware load asserts and deasserts `RiscType::ALL_NEO_DMS` and no longer calls `release_cce_uncore_for_firmware_load`. The uncore and the DMRISC remap stay as bring-up set them. `RiscType::ALL` on a CCE still holds the uncore, and deasserting it still reprograms the remap; Metal's load does not use that mask.

Check: two Metal runs in a row on each target, including teardown. The second run catches a teardown that left the uncore held.

### Step 7. Device-side addresses come from the host

Done. DRAM firmware load writes `CceSpaRoutes` immediately after the CCE bank-to-NOC staging range, after firmware has copied the bank tables. `core_info_msg_t` is unchanged: Blackhole sizes its DRAM mailbox to that struct. `cce_gddr_*` and `cce_sram_*` read the table. The kernels are unchanged and are not on the dataflow API. The GDDR port comes from `mimir_channels` when the descriptor has it.

Check: the Metal kernel tests on both models, and the Quasar ZeBu regression so the Tensix path is unchanged.

### Step 8. Stop keying Metal behaviour on the target type

Done. `TargetDevice` stays a choice of transport. The `EmuAxi` and `GrendelJtag` branches in `tt_cluster.cpp` that skipped the global reset and the Blackhole firmware check are gone. UMD skips the reset when the descriptor has no Tensix cores. Selecting either transport no longer forces slow dispatch or disables multi-erisc.

Check: the contract suite on every target.

### Step 9. Give the emulator its own device model

Done. `EmuTTDevice` and `GrendelJtagTTDevice` each supply a protocol, and `GrendelTTDevice` builds one model around that protocol. The protocol is usable when the device is constructed. Data and registers still move on the shared flat-address path.

Check: UMD `EmuTTDevice.*` on the mock and on SiVal, and the Metal contract suite on `mimir_1x1.yaml` and on MMK.

### Step 10. The contract suite only uses the shared API

Done. Every contract case opens a unit `MeshDevice`; none constructs a `Device` or initializes
`MetalContext` directly. `write_gddr_view` and `read_gddr_view` use the mesh slow-dispatch API,
and the tests keep calling those helpers. CCE and DRAM-view parameters come from the selected
descriptor. `cce_gddr_address` traps on an instance outside `CCE_SPA_ROUTES_MAX_INSTANCES`.

With no `mimir_channels`, CCE route setup uses the MMK wiring: Mimir 0 reaches other instances on
D2D1 and every other Mimir uses D2D0. `TT_METAL_EMU_BRINGUP` is removed. Dispatch mode stays with
`TT_METAL_SLOW_DISPATCH_MODE`.

Both transports still open `target_devices = {0}`, and the backend tests still assert one chip. That stays until the PCIe step. The contract suite does not treat "one chip" as part of the API.

Check: the contract suite on the single-Mimir model and on MMK.

### Step 11. Skip fast dispatch on a Mimir-only package

Done. `TT_METAL_SLOW_DISPATCH_MODE` still chooses the dispatch mode. On a package with programmable DRAM cores and no Tensix cores, `skip_unless_programmable_dram_package()` skips while that variable is unset. The skip happens before the mesh opens, so the suite does not reach `dispatch_core_manager`. A package with Tensix cores still opens in fast dispatch.

Check: the contract suite on `mimir_1x1.yaml` with the variable unset skips, and with it set it runs. A Tensix package is unchanged.

### Step 12. The contract suite names the package, not the transport

Done. With a chippy server set, the suite reads that server's descriptor (`TT_METAL_EMU_SOC_DESC` or `TT_METAL_GRENDEL_JTAG_SOC_DESC`). With no server, `TT_METAL_SOC_DESC` names the package and is enough to open. A PCIe run sets that variable and does not set a chippy server. Unset everything still skips.

Check: the contract suite on the emulator and on JTAG with their existing variables, and a run that sets only `TT_METAL_SOC_DESC` reaches device open.

### Step 13. Emulator tests check through EmuTTDevice

Done. `test_emu_tt_device.cpp` no longer opens a chippy transport or a `chippy::grendel::Mimir`. Writes are read back with `read_from_device` and with `GrendelTTDeviceTestAccess::read_flat` at the resolver's flat address. `read_flat_address` stays protected. CCE reset, PF_CTRL, and the DMRISC remap are read at `cce_control_address`.

Check: UMD `EmuTTDevice.*` on the mock and on SiVal.

### Step 14. Keraunos on the Mimir-local path

Flow 1 is Mimirs and Keraunos through chippy. `mimir_channels` only describes Mimir DRAM channels. Keraunos shows up as a chiplet to avoid: the emulator selects `m0` so a naked socket does not program `k0`, and the resolver skips the Keraunos column. Add Keraunos to the descriptor and reach it through the same Metal and UMD APIs. This is the last Mimir-local step before PCIe.

Check: MMK, which has a Keraunos, through the emulator. A Mimir-only descriptor skips the Keraunos cases.

### Step 15. Full Grendel over PCIe or the FPGA equivalent

Add the Quasar / Grendel case to `TTDevice::create`, with a package descriptor and package addressing. Run the contract suite unchanged. Only the cases that depend on which Mimirs exist, and which routes exist, should skip.

### Step 16. Run the same suite on JTAG

This waits until step 15. Boards are scarce.

The JTAG register path now shares the flat-address transport with emulation, and it has not been run on silicon since that change. Launch messages and CCE control need a board run.

JTAG multi-chiplet is still incomplete:

- `GrendelJtagProtocol` sends every core to the first chiplet's transport (`transports.front()`). Data goes through the per-Mimir flat transport, so that path is fine. Anything still on the protocol is chiplet 0 only.
- The TAP number is the digits of the chiplet id (`stoul` of `chiplet.substr(1)`). That assumes `mN` is TAP N. It fails for `k0`, and it has not been checked on an MMK board.

Fix those, then run the contract suite on JTAG. A one-Mimir board and an MMK board are both in scope. Keraunos on JTAG uses the descriptor work from step 14.

Check: the contract suite on JTAG silicon.

## Decisions

- **CCE SRAM address on full Grendel.** The host and dispatch use the real NOC/ATT window. That window is the HAL `l1_noc_offset` for a DRAM core. The `0x2000000000` tag stays inside the Mimir-local UMD path.
- **Who owns bring-up.** SMC firmware releases the CCE uncore and programs the DMRISC and MLA remaps. Metal only resets the cores. On the Mimir-local path, chippy fills the same role until SMC firmware is present.
- **Kernel API.** CCE kernels keep `cce_gddr_*` and `cce_sram_*`. The host supplies the bases.

Steps 1 and 2 are useful on their own. Step 1 produces the shared suite, and step 2 removes a difference between emulation and JTAG that already exists.

## Porting Mimir tests

Use this when moving a Mimir test onto the current tree. Copy the contract-test call patterns. A contract test opens a unit `MeshDevice` and uses buffers, programs, and the helpers in namespace `tt::tt_metal::mimir_contract`. Call those helpers. Do not reimplement CCE-to-Mimir indexing, and do not open a `Cluster` or initialize `MetalContext` directly.

Run the suite from `$TT_METAL_HOME`:

`TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_context --gtest_filter=*MimirEmu*`

### Package descriptor

- A Mimir soc descriptor sets `address_space: mimir_local`. Omitting it selects `full_package`. Emulation and JTAG then refuse to attach, and a FullPackage HAL uses `0x1280000000` as the DRAM-core NOC offset instead of `0x2000000000`. Keep FullPackage addresses out of a Mimir-local test.
- A Mimir soc descriptor lists `mimir_channels`, one entry per DRAM channel: `channel`, `instance`, `chiplet` (`m0`), and `routes` (`peer`, `d2d`) for each other Mimir. The chiplet id is what emulation selects. The instance is the SPA number used for the DMRISC remap. Channel 0 is not assumed to be `m0`.

### When the test runs

Start the test body with `skip_unless_programmable_dram_package()`. It is a macro, and it has to be invoked directly in the test: `GTEST_SKIP` returns from the function that calls it, so wrapping it in another function does not skip the test. It skips when the selected package has no programmable DRAM cores, when no transport is configured, and on a Mimir-only package while `TT_METAL_SLOW_DISPATCH_MODE` is unset. Any value of that variable, including `0`, selects slow dispatch. Unset selects fast dispatch, which has no Tensix dispatch core on a Mimir-only package. A package with Tensix cores still opens in fast dispatch. `TT_METAL_EMU_SERVER` is not the identity of the suite: with no chippy server, `TT_METAL_SOC_DESC` is enough to open.

The descriptor is the one the runtime will open. A chippy server uses `TT_METAL_EMU_SOC_DESC` or `TT_METAL_GRENDEL_JTAG_SOC_DESC`. With no server, `TT_METAL_SOC_DESC` names the package. The contract suite does not send INIT. Bring the model up with the chippy script first, then run the tests. `bringup_mimir_1x1.py` matches `mimir_1x1.yaml`. `bringup_two_mimir.py` matches `mimir_2x_package.yaml`. Leave `TT_UMD_EMU_INIT` unset on a live SiVal server. Run a Mimir-only package with `TT_METAL_SLOW_DISPATCH_MODE=1`.

Size the case from the descriptor:

- The Mimir count is `metal_SocDescriptor::get_num_dram_channels()`, exposed as `num_mimirs_on(device)`. `IDevice::num_dram_channels()` is the Metal DRAM view count, not the Mimir count. Passing the view count to a kernel as the SPA instance index walks off the last die.
- The CCE count is `num_cces_on(device)`: channel count times the DRAM grid height. The descriptor supplies that height. `mimir_index_for_cce` and `location_for_cce` divide by it.
- The logical DRAM coordinate is `{view, endpoint index}` from `logical_cce_core(soc_desc, cce_index)`. `write_gddr_view` takes that view index. Launching CCE `n` at `{n / 2, n % 2}` hits the wrong core once a channel has a height other than 2.
- A cross-Mimir case needs two DRAM channels. The gtest suite name contains `CrossMimir`, because `is_cross_mimir_case()` keys off that. `mimir_1x1.yaml` on MMK only touches `m0` and is the one-Mimir configuration.

### Host DRAM and CCE SRAM

- Write and read GDDR through `write_gddr_view` / `read_gddr_view`. The integer is the Metal view index. A contract test reaches DRAM through those helpers, or through `slow_dispatch::WriteToDRAMChannel` / `ReadFromDRAMChannel` on the unit mesh.
- `WriteToDRAMChannel` rejects an address below the allocator's DRAM base (`get_base_allocator_addr(HalMemType::DRAM)`, the same value as `HalDramMemAddrType::UNRESERVED`). Kernel GDDR offsets start at that base. `kDramOffset` (`0x800`) is a displacement above it. The channel-alias test uses the base plus `kDramOffset`. Passing `0x800` by itself is inside the firmware-reserved region.
- A DRAM view may be 512 MiB (`dram_view_size` 536870912). `dram_bank_size` stays 8 GiB. The second half's `address_offset` is 4 GiB (4294967296). `gddr_sweep_from_soc` treats a positive view smaller than 1 GiB as one slot whose stride is the view size. The view does not have to be a multiple of 1 GiB. Kernel slot math stays `(slot_base + slot) * slot_stride` in `uint64`. `slot_base` is the view's `address_offset` divided by the slot stride, so the 4 GiB half is slot 8 of a 512 MiB stride.
- Host CCE SRAM goes through `write_cce_sram` and `read_cce_sram`. Pass a logical DRAM core and a HAL NOC address: `cce_sram_noc_addr(offset)` or `hal.get_dev_noc_addr(...)`. The helper applies the HAL offset. The chiplet-local AXI address `0x40000000 + cce_index * 0x400000` and an extra `MEM_CCE_L1_NOC_OFFSET` are both the wrong address in contract test code. Kernel-local SRAM stays 0.
- Peer SRAM and peer GDDR in a CCE kernel use `cce_gddr.h`. `cce_gddr_address`, `cce_gddr_read`, and `cce_gddr_write` take the destination Mimir index and the offset. `cce_sram_*` takes the destination CCE index. The core's `CceSpaRoutes` selects the D2D port, so the kernel does not also take the sender's Mimir index. `CreateDevice` writes that table after the CCE bank-to-NOC staging range, from the package windows and `mimir_channels`. SPA bases stay out of the kernel. CCE kernels stay on these helpers, and porting a host test leaves the existing CCE kernels alone.

### Live SiVal server

- A test does not send INIT. INIT wipes chippy bring-up.
- A test does not release CCE harts onto a reset vector that is not real code, and it does not assert `RiscType::ALL` and then keep using CCE SRAM. `RiscType::ALL` holds the uncore. A store to CCE SRAM while the uncore is held stalls the emulator AXI master (`WR_STALL`, for CCE0 this is `0x40000800`). The stall survives a new client. Restart the SiVal server and run the chippy bring-up script again.
- `CreateDevice` resets the eight CCE harts and leaves the uncore and the DMRISC remap as chippy bring-up set them. A test that needs a different remap gets it from bring-up. A contract test does not call `release_cce_uncore_for_firmware_load`.
- UMD sends `SELECT_CHIPLET` with that channel's `mimir_channels` chiplet id only when `GET_CHIPLET` returns a chiplet id (`k0`, `m0`, `m1`, `q0`). The single-chiplet SiVal server replies `GET_CHIPLET called in single-chiplet model.` and UMD does not send `SELECT_CHIPLET` in that case. An empty payload is the same. A non-empty reply is a chiplet id only when it matches that form, and test code does not send `SELECT_CHIPLET`. A one-Mimir descriptor on MMK still selects `m0`, because the server's default chiplet is Keraunos and `GET_CHIPLET` there returns an id. Bring-up of that one Mimir uses chippy package `mk_bringup`, not `mimir`.

### UMD tests

- `EmuTTDevice` tests attach to a live server without INIT. Set `TT_UMD_EMU_INIT=1` only against the chippy mock.
- Set `TT_UMD_EMU_DRAM=1` only after GDDR bring-up.
- A per-hart reset mask releases every hart only when the request is the full `RiscType::ALL_NEO_DMS` mask: `(risc_type & ALL_NEO_DMS) == ALL_NEO_DMS`. Any non-zero overlap is not enough. `RiscType::DM0` releases bit 1 only (`1 << 1`). `DM7` releases bit 8.
- CCE control offsets are relative to one CCE: reset vector `0x02000000`, PF_CTRL `0x02200000`, DMRISC remap `0x02202000`. The DRAM-core coordinate selects the CCE. Test code does not add `0x04000000` per Mimir or `0x01000000` per CCE, and it does not call `cce_control_addr`. Writing `CCE_RESET_VECTOR_BASE` through `write_to_device_reg` on a DRAM core programs that core's CCE. Read it back with `GrendelTTDeviceTestAccess::read_flat` at `cce_control_address`. The test does not open a chippy transport.
