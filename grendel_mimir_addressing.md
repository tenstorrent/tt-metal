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

### 1. Each UMD device class owns both a transport and an address map

`EmuTTDevice` and `GrendelJtagTTDevice` both build `GrendelNocAddressResolver` from `mimir_local_address_windows()`, then send the resulting flat address through `MimirChippyMemoryMap`. That map turns it back into a chiplet-local address on a per-Mimir chippy transport.

`RtlSimulationTTDevice` installs `GrendelNocAddressResolver` with the default package windows (the qsr.s1 ATT) instead.

There is no PCIe Grendel device. `TTDevice::create` only handles Wormhole and Blackhole.

Addressing is chosen by which class is constructed, not by the descriptor. A future FPGA build that looks like PCIe but uses ATT addressing has nowhere to go.

### 2. CCE control is one implementation on the Mimir-local path

Done in step 2. Reset vector, PF_CTRL and DMRISC remap offsets are relative to one CCE (`0x02000000`, `0x02200000`, `0x02202000`). The DRAM-core coordinate selects the CCE. `GrendelNocAddressResolver::cce_control_address` produces the flat config address, and `MimirChippyMemoryMap` turns that into the chiplet-local AXI address. Emulation and JTAG share that path. The package ATT does not use this mapping. `program_cce_dmrisc_remap` still hardcodes the MMK SPA constants (step 5).

### 3. The descriptor selects the address space

Done in step 3. `address_space: mimir_local` selects the chiplet-local windows, including the `0x2000000000` CCE SRAM tag. `address_space: full_package`, and a descriptor that omits the key, selects the qsr.s1 ATT. That table now has CCE SRAM at `0x1280000000` (4 MiB per CCE), Mimir config at `0x1300000000` (128 MiB per Mimir, lane 0), and CCE control as a per-CCE offset in that window. In package mode the HAL `l1_noc_offset` for a DRAM core is `0x1280000000`. The `0x2000000000` tag stays on the Mimir-local path. Emulation and JTAG still require `mimir_local`.

### 4. The Mimir topology is assumed, not described

UMD assumes DRAM channel `n` is Mimir instance `n`, chippy chiplet `"m{n}"`, and SPA instance `n`. That shows up in `EmuTTDevice::Impl` (the select name) and in `program_cce_dmrisc_remap` (`mimir = cce_index / locations_per_channel`).

`program_cce_dmrisc_remap` hardcodes the MMK SPA layout: config `0x1300000000`, SRAM `0x1280000000`, GDDR `0x1000000000000`.

`GrendelJtagTTDevice` refuses to attach unless the descriptor has exactly one Mimir.

### 5. Bring-up ownership is split across chippy, UMD and Metal

Chippy brings up clocks, cold reset, GDDR, D2D, firewalls and the remaps.

Metal still asserts `RiscType::ALL` on a CCE, which holds the uncore, and then calls `release_cce_uncore_for_firmware_load`. That reprograms DMRISC from the constants above. See the DRAM branch of `initialize_firmware` in `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`.

SMC firmware owns uncore release and the DMRISC and MLA remaps on full Grendel. Chippy owns them on the Mimir-local path. Metal only resets harts, so the `release_cce_uncore_for_firmware_load` step has to leave the Metal firmware-load path.

### 6. Device-side addressing is wired for MMK

`tt_metal/hw/inc/experimental/cce_gddr.h` hardcodes the SPA bases, and it hardcodes the MMK route: a sender on Mimir 0 reaches another die through D2D1 (`0x1002000000000`), and every other sender uses the D2D0 window. M0 D2D0 is not connected to Mimir 1.

Kernels keep calling `cce_gddr_*` and `cce_sram_*`. The bases move from these constants to values the host supplies. Tensix kernels on full Grendel still reach GDDR through the NOC/ATT, so a CCE kernel and a Tensix kernel stay different sources.

### 7. Metal picks behaviour by target type

`tt_metal/llrt/tt_cluster.cpp` special-cases `TargetDevice::EmuAxi` and `TargetDevice::GrendelJtag` (skip the global reset, skip cluster-descriptor discovery, and build the matching `ClusterOptions`). `rtoptions` turns those targets on from `TT_METAL_EMU_*` and `TT_METAL_GRENDEL_JTAG_*`.

The Quasar choices in `risc_firmware_initializer.cpp` (reset all harts, release the uncore before the firmware image, tear down with the uncore left released) are keyed on the architecture, which is right. They also assume a Mimir-only package: no Tensix, no allocator, slow dispatch only. Full Grendel has workers, an allocator and fast dispatch.

`Cluster::is_mock_or_emulated()` is Mock and Emule only. EmuAxi and GrendelJtag take the silicon firmware path, with `emu_skip_init` so chippy's bring-up is kept.

### 8. The tests are tied to the backend

`tests/tt_metal/tt_metal/context/test_mimir_emu.cpp` mixes public APIs (`CreateDevice`, buffers, `ReadFromDeviceDRAMChannel`, programs) with internals: a raw `Cluster::write_core` at `MEM_CCE_L1_NOC_OFFSET`, `CreateDeviceMinimal`, and the view and slot helpers. It is gated on `TT_METAL_EMU_*` / `TT_METAL_GRENDEL_JTAG_*`, and it assumes one chip, two CCEs per Mimir, and the MMK cross-Mimir route.

`tt_metal/third_party/umd/tests/api/test_emu_tt_device.cpp` checks results by reading back through a raw chippy transport. That is a backend test, not the shared contract.

## Address spaces in use

| What | Address space |
| --- | --- |
| Per-CCE offsets plus `cce_control_address` | CCE register plane relative to one CCE. Reset vector `0x02000000`, PF_CTRL `0x02200000`, DMRISC remap `0x02202000`. The DRAM core selects the CCE. |
| `program_cce_dmrisc_remap` constants | Package SPA, per Mimir. Config `0x1300000000`, SRAM `0x1280000000`, GDDR `0x1000000000000`. |
| Default `GrendelAddressWindows` | Package ATT flat addresses for qsr.s1. NEO L1, Quasar config, GDDR, CCE SRAM `0x1280000000`, Mimir config `0x1300000000`. |
| `mimir_local_address_windows` plus `MimirChippyMemoryMap` | Chiplet-local AXI. Config at 0, CCE SRAM at `0x40000000`, GDDR at `0x800000000`, CCE SRAM tag `0x2000000000`. |

| Who is looking | CCE SRAM | GDDR |
| --- | --- | --- |
| Host, through Metal | DRAM core coordinate plus `MEM_CCE_L1_NOC_OFFSET` (`0x2000000000`), or `get_dev_noc_addr` | NOC worker endpoint plus the view's `address_offset` |
| CCE kernel, today | Its own SRAM at local 0; another CCE's SRAM at `0x1280000000` | `0x1000000000000`, or the D2D1 window, through a volatile pointer |
| Firmware link | `MEM_CCE_SRAM_LOCAL_BASE == 0` | — |

## Proposed sequence

Each step should leave the current targets green before the next one starts. The current targets are:

- the chippy mock
- a single-Mimir SiVal model
- the MMK SiVal model
- JTAG silicon, where a board is available
- the UMD resolver golden tests (`tests/baremetal/test_grendel_noc_address_resolver.cpp`)
- the Quasar ZeBu regression (`tests/scripts/quasar/quasar_regression_tests.yaml`)

### Step 0. Baseline

Record one scripted run, per target, of: bring-up, UMD `api_tests` with `EmuTTDevice.*`, the Metal Mimir suite run twice in a row without bringing the model up again, and the resolver and Quasar regressions. The second Metal run catches a test that left the CCE uncore held.

Run the emulator baseline on both a single Mimir (`mimir_1x1.yaml`, `bringup_mimir_1x1.py`) and MMK with both Mimirs up (`mimir_2x_package.yaml`, `bringup_two_mimir.py`). The single-Mimir server ignores `SELECT_CHIPLET`, so it is the run that catches a change which starts requiring a select. MMK with `mimir_1x1.yaml` only touches `m0` and does not count as the two-Mimir run. The mock, the resolver golden tests, and the Quasar ZeBu regression need neither model.

### Step 1. Split the tests into a shared contract suite and backend suites

Done. The contract suite is `tests/tt_metal/tt_metal/context/test_mimir_emu.cpp`. Shared helpers are `mimir_contract.hpp`. Raw `Cluster` access is `test_mimir_backend.cpp`. Both are listed in `sources.cmake`.

The contract suite uses `CreateDevice` / `CreateDeviceMinimal`, buffers, `ReadFromDeviceDRAMChannel` / `WriteToDeviceDRAMChannel`, programs, and `write_cce_sram` / `read_cce_sram`. It skips unless the selected package has programmable DRAM cores, and it sizes itself from the descriptor (how many CCEs, how many Mimirs, which pairs can reach each other).

Gtest names are unchanged: `MimirEmu.*`, `MimirEmuSameMimirDirect.*`, `MimirEmuCrossMimirNoc.*`, plus the two Cluster cases now compiled from the backend file.

Check: same pass and fail as the baseline. This suite is the yardstick for every later step.

### Step 2. One CCE control plane

Done. `GrendelNocAddressResolver::cce_control_address` takes a DRAM core and a per-CCE offset. `EmuTTDevice` and `GrendelJtagTTDevice` both write that flat address through `MimirChippyMemoryMap`. The chippy register-accessor overrides and `cce_control_addr` are gone. `program_cce_dmrisc_remap` still uses the hardcoded MMK SPA constants; step 5 removes those.

Check: UMD reset tests on the mock and on SiVal, the Metal suite, and JTAG. Host-side, `GrendelNocAddressResolver.CceControlOffsetIsRelativeToOneCce` covers the address math.

### Step 3. Choose the address space from the descriptor, and finish the package windows

Done. The descriptor key is `address_space`. `mimir_1x1.yaml` and `mimir_2x_package.yaml` set `mimir_local`. A descriptor that omits the key, including the Quasar simulation descriptors, stays `full_package`. `address_windows_for` is what `EmuTTDevice`, `GrendelJtagTTDevice`, and `RtlSimulationTTDevice` call. Emulation and JTAG reject `full_package` because their transport is still chiplet-local.

Package windows, from `models/qsr.s1/att/address_map.csv`: CCE SRAM `0x1280000000` stride 4 MiB, Mimir config `0x1300000000` stride 128 MiB. CCE control is the per-CCE offset in that window. Package-mode HAL `l1_noc_offset` for a DRAM core is `0x1280000000`. Mimir-local keeps `MEM_CCE_L1_NOC_OFFSET` (`0x2000000000`).

Check: `GrendelNocAddressResolver.PackageCceSramMatchesGoldenAddressMap` and `PackageMimirConfigCarriesCceControl`. Behaviour on current Mimir hardware does not change.

### Step 4. Separate the transport from the address space in UMD

Make `MimirChippyMemoryMap` one implementation of a "flat address to transport" interface. The package implementation is a single transport where the flat address is the SPA. Build one Grendel device out of a protocol, a resolver and an address space, instead of a device class per backend. PCIe, and the FPGA equivalent, then only add a protocol.

Check: everything from step 0.

### Step 5. Put the topology in the descriptor

Per DRAM channel, record the Mimir instance and the chippy chiplet id, and record the D2D route from each Mimir to each peer. Remove the assumption that the channel number is the instance number, and remove the MMK constants from `program_cce_dmrisc_remap`. Drop the one-Mimir limit in `GrendelJtagTTDevice` when a second chiplet is actually attached.

Check: the single-Mimir descriptors, the two-Mimir descriptors, and MMK. MMK is where a swapped `m0` / `m1` shows up.

### Step 6. Metal only resets harts

SMC firmware owns uncore release and the DMRISC and MLA remaps on full Grendel. Chippy owns them on the Mimir-local path. Remove `release_cce_uncore_for_firmware_load` and the remap programming from Metal's firmware load. Metal asserts and deasserts hart reset only, and it leaves the uncore as bring-up set it.

Check: two Metal runs in a row on each target, including teardown.

### Step 7. Device-side addresses come from the host

The host writes each core's GDDR and peer-SRAM route bases into core info or the bank-to-NOC tables, computed from the descriptor. `cce_gddr.h` keeps `cce_gddr_*` and `cce_sram_*` and reads those bases instead of the constants. CCE kernels do not move to the dataflow API.

Check: the Metal kernel tests on both models, and the Quasar ZeBu regression so the Tensix path is unchanged.

### Step 8. Stop keying Metal behaviour on the target type

`TargetDevice` stays a choice of transport. Everything else comes from the descriptor or from HAL capabilities: whether Tensix cores exist, whether there is an allocator, and whether fast dispatch is allowed. Remove the `EmuAxi` and `GrendelJtag` branches in `tt_cluster.cpp` that are not about the transport.

Check: the contract suite on every target.

### Step 9. Full Grendel over PCIe or the FPGA equivalent

Add the Quasar / Grendel case to `TTDevice::create`, with a package descriptor and package addressing. Run the contract suite unchanged. Only the cases that depend on which Mimirs exist, and which routes exist, should skip.

## Decisions

- **CCE SRAM address on full Grendel.** The host and dispatch use the real NOC/ATT window. That window is the HAL `l1_noc_offset` for a DRAM core. The `0x2000000000` tag stays inside the Mimir-local UMD path.
- **Who owns bring-up.** SMC firmware releases the CCE uncore and programs the DMRISC and MLA remaps. Metal only resets the cores. On the Mimir-local path, chippy fills the same role until SMC firmware is present.
- **Kernel API.** CCE kernels keep `cce_gddr_*` and `cce_sram_*`. The host supplies the bases.

Steps 1 and 2 are useful on their own. Step 1 produces the shared suite, and step 2 removes a difference between emulation and JTAG that already exists.

## Porting Mimir tests

Use this when moving a Mimir test onto the current tree. The contract tests and the helpers are the API. Copy those call patterns. Do not copy a chiplet-local AXI address or a raw `Cluster` open into a contract test.

### Which file

- A test that uses `CreateDevice` or `CreateDeviceMinimal`, buffers, programs, and `ReadFromDeviceDRAMChannel` / `WriteToDeviceDRAMChannel` goes in `tests/tt_metal/tt_metal/context/test_mimir_emu.cpp`.
- A test that constructs `Cluster` (`llrt::RunTimeOptions` plus `Cluster`), or that calls `write_core` / `read_core` itself, goes in `tests/tt_metal/tt_metal/context/test_mimir_backend.cpp`.
- Register a new `.cpp` in `tests/tt_metal/tt_metal/context/sources.cmake`, inside the `TT_UMD_BUILD_GRENDEL_JTAG` list. The binary is `./build/test/tt_metal/unit_tests_context`.
- Coordinate math, the GDDR sweep, and the SRAM helper live in `tests/tt_metal/tt_metal/context/mimir_contract.hpp`, namespace `tt::tt_metal::mimir_contract`. Call them. Do not reimplement CCE-to-Mimir indexing in the test.
- A Mimir soc descriptor must set `address_space: mimir_local`. Omitting it selects `full_package`. Emulation and JTAG then refuse to attach, and a FullPackage HAL uses `0x1280000000` as the DRAM-core NOC offset instead of `0x2000000000`. Do not put a FullPackage address in a Mimir-local test.

### When the test runs

Start the test body with `skip_unless_programmable_dram_package()`. It is a macro, and it has to be invoked directly in the test: `GTEST_SKIP` returns from the function that calls it, so wrapping it in another function does not skip the test. It skips when the selected package descriptor has no programmable DRAM cores. Do not skip only because `TT_METAL_EMU_SERVER` is unset, and do not use that variable as the identity of the suite.

Opening the device still needs a descriptor and a transport: `TT_METAL_EMU_SOC_DESC` or `TT_METAL_GRENDEL_JTAG_SOC_DESC`, and the matching `*_SERVER`. The contract suite does not send INIT. Bring the model up with the chippy script first, then run the tests. `bringup_mimir_1x1.py` matches `mimir_1x1.yaml`. `bringup_two_mimir.py` matches `mimir_2x_package.yaml`. Leave `TT_UMD_EMU_INIT` unset on a live SiVal server.

Size the case from the descriptor:

- The Mimir count is `metal_SocDescriptor::get_num_dram_channels()`, exposed as `num_mimirs_on(device)`. `IDevice::num_dram_channels()` is the Metal DRAM view count, not the Mimir count. Passing the view count to a kernel as the SPA instance index walks off the last die.
- The CCE count is `num_cces_on(device)`: channel count times the DRAM grid height. Do not hardcode two CCEs per Mimir. `mimir_index_for_cce` and `location_for_cce` divide by that height.
- The logical DRAM coordinate is `{view, endpoint index}` from `logical_cce_core(soc_desc, cce_index)`. It is not `{channel, location}`. Launching CCE `n` at `{n / 2, n % 2}` hits the wrong core on the split descriptor.
- A cross-Mimir case needs two DRAM channels. The gtest suite name must contain `CrossMimir`, because `is_cross_mimir_case()` keys off that. `mimir_1x1.yaml` on MMK only touches `m0` and is not the two-Mimir configuration.

### Host DRAM and CCE SRAM

- Write and read GDDR through `write_gddr_view` / `read_gddr_view`, or through `detail::WriteToDeviceDRAMChannel` / `detail::ReadFromDeviceDRAMChannel`. The integer is the Metal view index. Do not call `cluster.write_dram_vec` or `cluster.read_dram_vec` from a contract test.
- Those two DRAM-channel functions are deprecated internal APIs and are scheduled for removal after 2026-10-30. The contract suite calls them because a Mimir-only device has no allocator, so the buffer API is not the host path for these cases. When the functions are removed, change `write_gddr_view` and `read_gddr_view` in `mimir_contract.hpp`. Leave the tests calling the helpers.
- `CreateDeviceMinimal` on a package with no Tensix cores does not build an allocator. `device->allocator()` is an empty pointer. Test the pointer. Do not dereference it.
- A DRAM view may be 512 MiB (`dram_view_size` 536870912). `dram_bank_size` stays 8 GiB. The second half's `address_offset` is 4 GiB (4294967296). `gddr_sweep_from_soc` treats a positive view smaller than 1 GiB as one slot whose stride is the view size. Do not require the view to be a non-zero multiple of 1 GiB. Kernel slot math stays `(slot_base + slot) * slot_stride` in `uint64`. `slot_base` is the view's `address_offset` divided by the slot stride, so the 4 GiB half is slot 8 of a 512 MiB stride. Leave `kDramOffset` at `0x800`.
- Host CCE SRAM goes through `write_cce_sram` and `read_cce_sram`. Pass a logical DRAM core and a HAL NOC address: `cce_sram_noc_addr(offset)` or `hal.get_dev_noc_addr(...)`. Do not form the chiplet-local AXI address `0x40000000 + cce_index * 0x400000`, and do not add `MEM_CCE_L1_NOC_OFFSET` in contract test code. The helper applies the HAL offset. Kernel-local SRAM stays 0.
- Peer SRAM and peer GDDR in a CCE kernel still use `cce_gddr.h` (`cce_gddr_*`, `cce_sram_*`). Step 7 replaces the constants with bases the host supplies. Do not move CCE kernels onto the dataflow API. Do not edit the kernels under `tests/tt_metal/tt_metal/test_kernels/misc/` as part of porting a host test.

### Live SiVal server

- Do not send INIT from a test. INIT wipes chippy bring-up.
- Do not release CCE harts onto a reset vector that is not real code. Do not assert `RiscType::ALL` and then keep using CCE SRAM. `RiscType::ALL` holds the uncore. A store to CCE SRAM while the uncore is held stalls the emulator AXI master (`WR_STALL`, for CCE0 this is `0x40000800`). The stall survives a new client. Restart the SiVal server and run the chippy bring-up script again.
- UMD sends `SELECT_CHIPLET m{n}` only when `GET_CHIPLET` returns a chiplet id (`k0`, `m0`, `m1`, `q0`). The single-chiplet SiVal server replies `GET_CHIPLET called in single-chiplet model.` and UMD does not send `SELECT_CHIPLET` in that case. An empty payload is the same. Do not treat any non-empty reply as multi-chiplet, and do not send `SELECT_CHIPLET` from test code. A one-Mimir descriptor on MMK still selects `m0`, because the server's default chiplet is Keraunos and `GET_CHIPLET` there returns an id. Bring-up of that one Mimir uses chippy package `mk_bringup`, not `mimir`.

### UMD tests

- `EmuTTDevice` tests attach to a live server without INIT. Set `TT_UMD_EMU_INIT=1` only against the chippy mock.
- Set `TT_UMD_EMU_DRAM=1` only after GDDR bring-up.
- A per-hart reset mask must release every hart only when the request is the full `RiscType::ALL_NEO_DMS` mask: `(risc_type & ALL_NEO_DMS) == ALL_NEO_DMS`. Any non-zero overlap is not enough. `RiscType::DM0` releases bit 1 only (`1 << 1`). `DM7` releases bit 8.
- CCE control offsets are relative to one CCE: reset vector `0x02000000`, PF_CTRL `0x02200000`, DMRISC remap `0x02202000`. The DRAM-core coordinate selects the CCE. Do not add `0x04000000` per Mimir or `0x01000000` per CCE in test code, and do not call `cce_control_addr`. Writing `CCE_RESET_VECTOR_BASE` through `write_to_device_reg` on a DRAM core still programs that core's CCE. Reading the register back through a chippy accessor still works.
