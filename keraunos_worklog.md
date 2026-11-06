# Keraunos CCE paths: work log

Plan: `keraunos_plan.md`. Each entry records what was tried, what happened, and where the emulator log is.

## 2026-11-06: start
- Baseline before the plan: `KeraunosEmu.HartZeroRunsEthernetKernelWritesLocalSram` passed on `mmk` (93 s). Log: `~/.local/state/grendel-emu/yyzeon24-special-kstevens-for-reservation-263094/logs/2026-11-06_18-03-06_mmk/test_18-07-08.log`.
- Uncommitted at start:
  - tt-metal: `test_mimir_emu.cpp` (WritesMimirGddr and ReadsMimirGddr tests), two untracked eth GDDR kernels, UMD submodule pointer
  - chippy: `bringup_mmk.py` always-release diff

## Phase 0: emu tool support for chippy (2026-11-06)
- Extended `tt-umd-simulators/emu/grendel-session/emu`:
  - New `emu chippy [--model] [--fw-script S|none] [--raw] -- <script> [args]`. Runs `uv run --project validation python <script> <host> <port> [args]` from `$CHIPPY_DIR`; `--raw` runs any command there. It sets `TT_METAL_EMU_SERVER`, `EMU_HOST`, `EMU_PORT` and `EMU_MODEL`, and reports output, events and stalls the same way as `emu test`.
  - `--fw-script` now also works on `emu test` and `emu chippy`. It takes a bare name (`validation/metal_bringup/<name>`), a path under the chippy root, or `none`, which starts the server with no bring-up so SysEng flows that do their own init can run. If the script differs from the loaded session's, the session restarts.
  - The shared run/report logic is factored into `run_attached()`.
- New read-only probe: `chippy/validation/metal_bringup/probe/keraunos_cce_status.py`. For each CCE it dumps the PF_CTRL reset state, `local_tl1_base`, the valid DMRISC entries and any latched iDMA/AXI errors.
- Check passed:
  - Command: `emu chippy --model mmk -- validation/metal_bringup/probe/keraunos_cce_status.py --hsio 0 2`
  - HSIO0 and HSIO2 CCE0: `local_tl1_base` 0x22800000 / 0x2a800000; DMRISC[0] `[0,4M)` → own SRAM; [1] `[8M,12M)` → own SRAM; [6–9] identity HSIO windows; no errors latched.
  - Log: `.../logs/2026-11-06_19-32-43_mmk/test_19-36-15.log`
- Build dir fix: CMake had cached `CHIPPY_SOURCE_DIR`/`CHIPPY_BUILD_DIR` under `/proj_sw/user_dev/kstevens/work/`, which no longer exists. Reconfigured with `/proj_sw/user_dev/kstevens/chippy` and `.../chippy-build`, as in `mimir_setup.md`.

## Phase 1a: overlay inventory (2026-11-06)
- **Register map.** `chippy/validation/common/include/registers/ker/cce/cce_reg.h` (the Keraunos CCE local map) matches tt-metal's `tt-2.0.0` (GRENDEL) headers for `tt_cluster_ctrl`, `tt_rocc_accel`, `tt_cache_controller` and `memory_port_*`: 1241 shared register names, 0 differing values. The remaining differences are naming only (`_0_` vs `_0__`). Metal reports "Quasar IP variant: GRENDEL", so the idle-eth build already uses these headers.
- **Blocks present on the K CCE** (per hart CPU0–7 unless noted):
  - RoCC CMD_BUF_R, CMD_BUF_W, SIMPLE_CMD_BUF, ADDRESS_GEN_R, ADDRESS_GEN_W (`TT_ROCC_ACCEL` @ 0x03004000)
  - `TT_IDMA_SIDEBAND` @ 0x03002000
  - `TT_LLK_INTERFACE` tile counters 0–15 @ 0x03003000. These exist on the CCE, but `llk_intf_api.hpp` is aimed at Tensix; not tested.
  - `TT_CACHE_CONTROLLER` @ 0x04010000
  - PF_CTRL, DMRISC and snoop remap, interrupts and MSI-X, mailboxes, desc rings (CCE-specific, outside the overlay library)
- **Absent:** no FDS (`fds_*`) and no counter remapper (`remapper_api.hpp`) block in `cce_reg.h`. Both are out of scope.
- **Unknown:** whether NOC (non-IDMA) cmdbuf transactions have a target on Keraunos. SysEng's Keraunos firmware uses only IDMA, and the HSIO tile's overlay NIU coordinates are not documented. Deferred until IDMA works.
- **No `mstatus.XS` setup** anywhere in tt-metal (`dm.cc` or `idle_erisc.cc`) or in SysEng's CCE apps, yet SysEng's IDMA works. So RoCC on this core does not need XS set; `KeraunosOverlay.IdmaCopy*` confirms it.
- **Likely cause of the original IDMA hang (hypothesis, to be tested):** `eth_read_mimir_gddr.cpp` used `setup_vcs_cmdbuf(NocVcs::READ)`, which gives response VC 12, the read-data VC. tt-metal's own `idma_copy_cmdbuf` and SysEng's `ker_cce_dma_m2k_remap` both use write VCs (request 1, response 13) plus an explicit `setup_trids_cmdbuf`. An IDMA copy is a write transaction (`misc.write_trans=1`), so its completion ack may never be seen on VC 12. The Quasar DM examples (`quasar_idma/kernels/*.cpp`) also use `NocVcs::READ`, which may only work in the functional simulator.
- **Example kernels can't be reused directly.** The Quasar DM example kernels take 32-bit Tensix-L1 compile-time addresses (`experimental/kernel_args.h`), so they can't carry 64-bit SPA/HSIO addresses. New kernels in `tests/tt_metal/tt_metal/test_kernels/misc/keraunos/` follow the same API sequence with runtime 64-bit addresses and a status block the host reads even after a hang.

## Baseline (2026-11-06, session 2026-11-06_19-32-43_mmk)
- Passed: `KeraunosEmu.{HartZero,AllHarts,EveryEthernetCore,EthernetKernelWritesOtherTileSram}` (94–109 s each).
- **Failed:** `KeraunosEmu.EthernetKernelWritesMimirGddr`. The host read 0 instead of the magic word. No hang, emulator state FW_LOADED. Previously recorded as passing.
- Mimir representatives were not run in that batch. The gtest filter had the Mimir patterns after `-`, which excludes them. They still need a run.

## Phase 1a result: overlay libraries on the K CCE (session 2026-11-06_19-47-25_mmk)
`KeraunosOverlay.*`: **8/8 pass** on HSIO0 CCE0, all in one MeshDevice open.

| Test | Time |
| --- | --- |
| `IdmaCopyGlobalAddressesWriteVcs` (SysEng sequence) | 4.9 s |
| `IdmaCopyHelper` (`idma_copy_cmdbuf`) | 4.1 s |
| `IdmaCopyCmdBuf1` | 4.8 s |
| `IdmaCopyAutoIncrementAllAcked` (8 issues, autoinc, `free_space` gating, `all_idma_acked`) | 22.8 s |
| `IdmaCopyChipletLocalAddresses` (0x2280_0000-based) | 4.1 s |
| `AddrgenStridedIdma` (addrgen 0 → cmdbuf 0, 16 elements, stride 2) | 11.6 s |
| `AllHartsIdmaCopyInParallel` (8 harts, each with its own cmdbuf) | 27.4 s |
| `IdmaCopyReadVcs` (`NocVcs::READ`) | 4.2 s |

- IDMA works with both package-SPA and chiplet-local addresses.
- RoCC needs no XS enable.
- **The VC hypothesis is disproved:** READ VCs complete fine locally. The original GDDR-read hang is something else.
- Not yet covered: scatter-list, L1 atomic accumulate, NOC (non-IDMA) cmdbuf, per-trid interrupts, `noc_async_*`. Those come after the D2D question is settled.

## Phase 1 probe ladder, first run (same session)
- **Within Keraunos, all pass:** `KeraunosPaths.{CpuWritesOtherTileSram, IdmaWritesOtherTileSram, IdmaReadsOtherTileSram, CpuReadsOtherTileSram}` (HSIO0 ↔ HSIO1). CPU and IDMA reads of another HSIO tile work.
- **Across D2D, all fail, writes included:**
  - `IdmaWritesM1CceSram` and `IdmaWritesM1Gddr`: the IDMA completes and is acked, but the data never lands (host reads 0).
  - `IdmaReadsM1CceSram` and `IdmaReadsM1Gddr`: time out (120 s). `CpuReadsM1CceSram` also times out.
  - `EthernetKernelWritesMimirGddr` fails again (reads 0).
  - No iDMA/AXI error is latched on HSIO2 CCE0, and the emulator never stalled.
- **D2D links are DOWN right after bring-up.** New probe `chippy/validation/metal_bringup/probe/d2d_link_status.py` reads `BLYNX_BOW_LL_TOP_RXINTERFACECONTROLSTATUS` (tile+0x26004, bits[1:0]==3 means RX_TRAINING_RUN; same gate as chip_init `wait_d2d_link_up`). Session 2026-11-06_20-1x:
  - Right after `bringup_mmk.py`: all four endpoints (M0 D2D1, M1 D2D0, M1 D2D1, K0 D2D2) read 0x3fe6c (DOWN), progress 0.
  - A few minutes later: all read 0x88ab (UP), progress 0xbb1d.
  - `bringup_mmk.py` releases the D2D CPUs and returns without waiting for training.
- **Cross-check:** the `grendelemulation` model has not changed since 0df5a13 (2026-11-04). The bring-up working copy dates from 2026-11-05 02:08, after the GDDR write kernel (2026-11-04 22:22).
- **SysEng k2m_remap flow can't run yet:** no SysEng firmware is built under `validation/firmware/build`, and the RISC-V toolchain path in `cmake/toolchain/riscv-baremetal.cmake` (`/proj_syseng/shared/toolchains/...`) does not exist on this machine.
