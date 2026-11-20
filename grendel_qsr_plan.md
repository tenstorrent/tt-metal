# Mimir and Keraunos in the full Grendel package (qsr.s1 / qsr1_a1): plan

Status: plan for review, 2026-11-20. Nothing here has been implemented.

Goal: run the same tt-metal code, and the same tests and programs built on it, on mmk (Mimir + Mimir + Keraunos) and on
the full package with a Quasar chiplet between the Mimirs and Keraunos, with every difference absorbed by tt-umd.

Sources: grendelemulation (`models/qsr.s1`, `models/qsr1_a1`, `docs/GLOBAL_ADDRESSING*.md`, `docs/att-smc-fw.md`,
`docs/qmk_att_dump.md`), tt-metal main's Quasar ATT support, tt-umd's Grendel resolver. File and line references are in
the research notes; the key ones are repeated here.

## 1. What the package is

| | mmk | qsr.s1 / qsr1_a1 |
|---|---|---|
| Chiplets | 2 Mimir, 1 Keraunos | 1 Quasar, 4 Mimir, 1 Keraunos (`models/qsr.s1/att/topology.yaml`) |
| Links | M0-M1 (Mimir peer), M1-K | Q north to M0 and M1, Q south to M2 and M3, Q east to K (4 lanes). No Mimir-Mimir or Mimir-K links (`emu_hdl/tb_emu_d2d_link.sv`) |
| Keraunos HSIO tiles modelled | 4 | 5 |
| Host attach in emulation | one SMC AXI transactor per chiplet | the same, one per chiplet: Q0, M0..M3, K0 (`py/common/tb.py:49-66`); PCIe pins only on K0 |
| Bring-up | chippy `bringup_mmk_fw.py` | per-chiplet SMC firmware (qsr1_boot / emu_boot) from `tb.init_and_reset()`; no tt-metal bring-up script yet |

qsr.s1 is deprecated in its own testbench in favour of qsr1_a1 (`models/qsr.s1/py/common/tb.py:22`), which keeps the
topology and firmware. **Target qsr1_a1.**

## 2. What stays the same

The package address map is the same POR mask table on all three chiplet types (`docs/qmk_att_dump.md:67-92`), and it
matches what the mmk work already uses:

| Window | Package address | Used today by |
|---|---|---|
| Keraunos HSIO SRAM | `0x12_0680_0000 + tile * 0x400_0000` (+ CCE1 at 4 MiB) | `KeraunosSpaRoutes`, `MimirCceSpaRoutes.keraunos_*` |
| Mimir CCE SRAM | `0x12_8000_0000 + cce * 4 MiB` | `MimirCceSpaRoutes.sram_spa_base`, `KeraunosSpaRoutes` |
| GDDR lane A / lane B | `0x1_0000_0000_0000 + i * 8 GiB` / `0x1_0020_0000_0000 + i * 8 GiB` | `gddr_d2d_base[0..1]` |

So device code (`experimental/keraunos_cce.h`, `experimental/gddr_cce.h`) and the host-written route tables already use
the right addresses. A Keraunos-to-Mimir transfer goes K, then Q, then M in qsr instead of K, then M1, then M0, but the
SPA is the same and the routing is done by the chiplets' ATTs, which their SMC firmware programs. The CCE reset-vector and
hart-release flow is also the same: Keraunos qsr1_boot leaves CCE cores in reset for the host to preload
(`docs/att-smc-fw.md:627-638`).

## 3. What differs, and where each difference should live

| # | Difference | Where it belongs |
|---|---|---|
| D1 | Package composition: Quasar Tensix and dispatch cores, 4 Mimir channels, 5 HSIO tiles | tt-umd SoC descriptor (new `qsr1_a1.yaml`) |
| D2 | Host transport: per-chiplet SMC AXI (as mmk) or package access through the Quasar ATT (as the existing qsr RTL-sim path and real hardware) | tt-umd device and resolver (`address_space: mimir_local` vs `full_package`) |
| D3 | Mimir lane meaning: in qsr both lanes reach every Mimir through Quasar; mmk's `mimir_channels.routes` (peer, d2d) has no equivalent | tt-umd descriptor and resolver; route tables are built from UMD data |
| D4 | Firmware already programs the Mimir and Keraunos CCE DMRISC remappers on qsr; tt-umd's `program_mimir_dmrisc_remap` would overwrite Mimir emu_boot's entries 0-3 with a layout emu_boot warns against | tt-umd: a descriptor flag for firmware-owned remaps |
| D5 | Package constants (HSIO SRAM SPA base, Mimir lane-instance math) are hard-coded in tt-metal (`risc_firmware_initializer.cpp`, `mimir_cce_spa_routes.h`) | move into tt-umd once; tt-metal reads them |
| D6 | ATT map selection: tt-metal picks `grendel_qsr1` when the simulator name starts with `emu-qsr-s1` (`rtoptions.hpp`, `metal_env.cpp`) | key it from the SoC descriptor (tt-umd), not the simulator name |
| D7 | `quasar_drisc_l1_noc_offset_for_hal` rejects mixing `mimir_local` and `full_package` (`metal_env.cpp:181-201`) | follows from D2: pick one address space per package in the descriptor |
| D8 | Coordinate frame: the existing qsr descriptor uses origin (1,1), POR is (3,3); `GrendelAddressWindows` defaults are hard-coded qsr.s1 values | tt-umd descriptor and resolver |
| D9 | Device bring-up: firmware and chiplet selection for the model | chippy: a `bringup_qmk_fw.py` (or reuse of qsr1_boot through the testbench) and an emu tool model entry |

## 4. Plan

### Phase A: bring up on the per-chiplet transport (closest to mmk)

1. **tt-umd: `qsr1_a1.yaml`** with the Quasar Tensix grid and dispatch cores, Mimir CCEs as DRAM cores (4 Mimirs), Keraunos
   CCE0 and CCE1 as ethernet cores (5 HSIO tiles), the `grendel`/`mimir`/`keraunos` sections, POR frame, and
   `address_space: mimir_local` for the Mimir and Keraunos cores.
2. **tt-umd: resolver** generalized for D3 (lane routes come from the descriptor; both lanes valid for every Mimir) and D8.
3. **tt-umd: firmware-owned remaps** (D4): a descriptor flag; when set, UMD does not program the Mimir DMRISC remap and
   reads back what firmware programmed, for assertions only.
4. **chippy and emu**: a qsr1_a1 bring-up script and an `emu` model entry (D9).
5. **tt-metal**: no change expected. Run the mmk suites unchanged: `KeraunosEmu`, `KeraunosCce1`, `KeraunosApi`,
   `KeraunosFabric`, `KeraunosPaths`, `KeraunosFromMimir`, `MimirEmu`, the GDDR copy suites, and the watcher suites.
   Tests that name chiplets (HSIO 2 on D2D2, "M1 owns D2D", M0-through-M1) must take those from the descriptor; where a
   test hard-codes an mmk topology fact, fix the test to ask UMD (this is a test fix, not a runtime difference).

### Phase B: package transport (what real hardware needs)

6. **tt-umd: full-package device path** (D2): Mimir and Keraunos CCE control, SRAM and GDDR through the Quasar ATT, as
   the existing RTL-sim qsr path does for Tensix. The resolver already has package-mode CCE SRAM and control addresses;
   add CCE reset and hart release through the package (the GE remote server today only resets Tensix and DM cores).
7. **tt-metal: one-time moves** (applies to mmk too, so it is not an mmk-vs-qsr difference):
   - Route tables (`KeraunosSpaRoutes`, `MimirCceSpaRoutes`) built from UMD-provided package constants (D5).
   - ATT map chosen from the descriptor (D6); `quasar_drisc_l1_noc_offset_for_hal` reads the address space from the
     descriptor (D7).
8. **Mixed-core programs**: on qsr a program can place kernels on Tensix, Mimir CCE and Keraunos CCE cores together.
   Today CCE kernels run only under slow dispatch. Fast dispatch for DRAM and IDLE_ETH programmable cores (per-CCE
   kernel text offsets) is a prerequisite for running qsr workloads with fast dispatch, and is the same work mmk needs.

### Phase C: Quasar-specific additions (new capability, not a difference)

9. Tensix-to-Mimir GDDR and Tensix-to-Keraunos paths (Tensix NOC through the tile NIU ATT) and the corresponding tests.
   These are new tests; the mmk tests stay as they are.

## 5. Can every difference live in tt-umd?

Yes for runtime behaviour, with three one-time tt-metal changes that are not mmk-vs-qsr special cases:

- **Package constants and route tables (D5).** They are in tt-metal today. Moving them behind a UMD API is a tt-metal
  change, but afterwards tt-metal has no package-specific code.
- **ATT map and address-space selection (D6, D7).** Today keyed on the simulator name and asserted in tt-metal. Reading
  both from the SoC descriptor removes the special case.
- **Fast dispatch for CCE cores.** Not a qsr difference, but mixed Tensix/CCE programs on qsr make it visible sooner.

Two things cannot be hidden, and should not be:

- **Topology-dependent tests.** A test that checks "HSIO 2 owns the Mimir link" or "M0 is reached through M1" is testing
  mmk's topology. Those tests should query the descriptor for the link owner and route; they then pass on both.
- **Availability of Tensix workers.** Programs that use worker cores run on qsr and not on mmk. That is a capability
  difference of the hardware, not of the runtime.

## 6. Risks and unknowns

- Whether tt-metal's `EmuTTDevice` path works against the qsr1_a1 SiVal server with chiplet selection (plausible from the
  code, untested; no qsr bring-up script exists in chippy).
- Whether qsr1_a1 changes any address against qsr.s1 (the notes say the topology is shared; not diffed).
- Firmware-written CCE1 reset vectors on Keraunos vs tt-metal's reset-vector writes.
- Emulation caveats from the GE docs: stubbed NEO L1s hang on read; the SMU ATT gasket/table toggle is needed for host
  global access; non-posted returns need landing buffers in the 0x100000-0x1FFFFF loopback window; Mimir self-SPA access
  is skipped in GE tests (A4-003); Keraunos HSIO SRAM as a CCE target is untested in GE and needs the TL1 hash to agree.
- Remote atomics: on mmk only same-tile IDMA accumulate works (see `keraunos_worklog.md`). qsr routes all cross-chiplet
  traffic through Quasar; whether that path preserves L1 accumulate is unknown and should be one of the first qsr probes.

## 7. Questions for review

1. Target qsr1_a1 rather than qsr.s1 (deprecated)?
2. Phase A first (per-chiplet transport, quickest), or go straight to Phase B (package transport, matches hardware)?
3. Who owns the qsr1_a1 bring-up script in chippy (SysEng flow vs ours)?
4. Should the one-time tt-metal moves (D5 to D7) land in the upstreaming PR series (`grendel_pr_plan.md`), before qsr work
   starts?
