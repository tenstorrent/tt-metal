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
  - `IdmaWritesM1MimirCceSram` and `IdmaWritesM1Gddr`: the IDMA completes and is acked, but the data never lands (host reads 0).
  - `IdmaReadsM1MimirCceSram` and `IdmaReadsM1Gddr`: time out (120 s). `CpuReadsM1MimirCceSram` also times out.
  - `EthernetKernelWritesMimirGddr` fails again (reads 0).
  - No iDMA/AXI error is latched on HSIO2 CCE0, and the emulator never stalled.
- **D2D links are DOWN right after bring-up.** New probe `chippy/validation/metal_bringup/probe/d2d_link_status.py` reads `BLYNX_BOW_LL_TOP_RXINTERFACECONTROLSTATUS` (tile+0x26004, bits[1:0]==3 means RX_TRAINING_RUN; same gate as chip_init `wait_d2d_link_up`). Session 2026-11-06_20-1x:
  - Right after `bringup_mmk.py`: all four endpoints (M0 D2D1, M1 D2D0, M1 D2D1, K0 D2D2) read 0x3fe6c (DOWN), progress 0.
  - A few minutes later: all read 0x88ab (UP), progress 0xbb1d.
  - `bringup_mmk.py` releases the D2D CPUs and returns without waiting for training.
- **Cross-check:** the `grendelemulation` model has not changed since 0df5a13 (2026-11-04). The bring-up working copy dates from 2026-11-05 02:08, after the GDDR write kernel (2026-11-04 22:22).
- **SysEng k2m_remap flow can't run yet:** no SysEng firmware is built under `validation/firmware/build`, and the RISC-V toolchain path in `cmake/toolchain/riscv-baremetal.cmake` (`/proj_syseng/shared/toolchains/...`) does not exist on this machine.

## Phase 1: wait for links, then retest (session 2026-11-06_20-xx_mmk)
- Waited with `d2d_link_status.py --wait 900`. All four endpoints came UP after 503 s (0x88ab).
- Then ran the D2D probes. Same failures as before:
  - `EthernetKernelWritesMimirGddr`: FAIL (reads 0)
  - `IdmaWritesM1MimirCceSram`: FAIL (acked, data never lands)
  - `IdmaReadsM1MimirCceSram`: FAIL (120 s timeout)
  - `IdmaWritesM1Gddr`: FAIL; this time it timed out (123 s) instead of acking
- Links were still UP after the batch.
- **Conclusion:** a down link at bring-up is real (the bring-up should wait for it), but it is not the cause of the failures.
- **Time-box reached:** 3 emulator iterations with no new hypothesis inside tt-metal. Moved to a chippy-only reproduction.

## K→M1 D2D path parked: chippy reproduction (2026-11-06)
- Location: `chippy/validation/metal_bringup/repro/ker_d2d_access/`.
- Contents:
  - `cce_access.S`: a 92-byte bare-metal program, built by `build.sh` with the sfpi `riscv-tt-elf-gcc -mcpu=tt-qsr64-rocc`. Hart 0 performs one 64-bit store (then a fence) or one load to a target SPA, and records its stage in its own SRAM.
  - `ker_d2d_access.py`: the host script. It runs everything through chippy's `cce_cores_hold` / `cce_reset_vector_set` / `cce_cores_release` / `cce_read_error_status`.
- Run with one command after the default mmk bring-up: `emu chippy --model mmk -- validation/metal_bringup/repro/ker_d2d_access/ker_d2d_access.py`.
- First run (log `.../logs/2026-11-06_20-56-42_mmk/test_21-54-18.log`). Links were not yet UP after its 900 s wait:

  | Case (from K0 HSIO2 CCE0) | Target SPA | Result |
  | --- | --- | --- |
  | `store_hsio1_sram` | 0x12_0ABF_0000 | PASS |
  | `load_hsio1_sram` | 0x12_0ABF_0000 | PASS |
  | `store_m1_cce_sram` | 0x12_80BF_0000 | FAIL: store retires, no error; M1-local 0x4000_0000+0x3F0000 stays 0 |
  | `store_m1_gddr` | 0x1_0022_0010_0000 | FAIL: store retires; M1 0x8_0010_0000 stays 0 |
  | `load_m1_cce_sram` | 0x12_80BF_0000 | FAIL: the load never returns (30 s) |
  | `load_m1_gddr` | 0x1_0022_0010_0000 | FAIL: the load never returns (30 s) |

  - No CCE error was latched and the emulator did not stall.
- This reproduces the tt-metal result without tt-metal.
- Second run (session `2026-11-06_22-10-57_mmk`):
  - All four links reported RX_TRAINING_RUN (0x88ab) after 584 s.
  - Identical result: both HSIO1 controls pass. Both M1 stores retire with no error but never land, and the M1 loads never return.
  - **This is the run to send to SysEng.**
- **Parked:** every K→Mimir path (GDDR, CCE SRAM, and the Phase 3 K↔Mimir and atomic paths) is parked pending SysEng feedback. Work continues on intra-Keraunos paths, CCE1, and the APIs.

## Phase 4: CCE1, native layout (design, 2026-11-06)

### chippy `bringup_mmk.py`
- `initialize_fabrics` now runs for `hsio.N.cce.0` and `hsio.N.cce.1`, followed by the tile-level steps. Each `initialize_fabrics` call closes the HSIO firewalls again, so `disable_ker_hsio_filters` must come after both.
- `program_keraunos_cce_local_sram` programs both CCEs:
  - `local_tl1_base` is set to the CCE's own SRAM only (avoids the Mimir CCE1 `local_tl1_base` bug, `grendelemulation/docs/dx/mimir-cce1-local-tl1-base-claims-peer-sram.md`).
  - Entry 0 maps the own cached window. Entry 1 maps the own uncached window: 8 MiB + own base, so 12 MiB for CCE1, as the user confirmed.
  - Entry 2 is the peer's cached window and entry 3 the peer's uncached window.
  - Entries 6–9 remain the identity HSIO windows.
  - Both CCEs therefore see CCE0 SRAM at [0, 4 MiB) and CCE1 SRAM at [4, 8 MiB).
- New chippy probe `probe/keraunos_cce1/` (`cce_ops.S` is position-independent; `keraunos_cce1_native.py`) checks this without tt-metal. It covers:
  - CCE1 own cached and uncached accesses;
  - CCE1↔CCE0 in both directions;
  - CCE1 to HSIO1 CCE0 and CCE1;
  - reporting CCE1's `mhartid`.

### UMD (left unstaged, PR draft in `tt_metal/third_party/umd/PR_DRAFT_keraunos_cce1.md`)
- New descriptor field `keraunos.eth_cces_per_core` (1 or 2, default 1). `mmk.yaml` sets it to 2 with `eth_l1_size` 8 MiB, so each ethernet core is a whole tile.
- `keraunos_control_address(core, offset, eth_cce_in_core)` selects CCE1's control slot, 16 MiB above CCE0's.
- Reset vector:
  - A register-width write at the start of either CCE's SRAM (core offset 0 or 4 MiB, hart slots 0–7) programs that CCE's reset vectors.
  - Hart 0 broadcasts to all eight harts of that CCE.
- Hart reset: an all-hart `RiscType` (ALL / ALL_NEO_DMS) moves every CCE the core spans. A single DMn moves only CCE0's hart.
- The chippy memory map covers both control slots.
- **Assumption (unattended, per the UMD CLAUDE.md):**
  - An explicit descriptor field was chosen over deriving the CCE count from `eth_l1_size`.
  - No new `RiscType` bits were added: per-hart control of CCE1 is not needed by tt-metal today.

### tt-metal
- **Processor layout: interleaved.** Processor `2*hart + cce`: DM0 = CCE0 hart 0, DM1 = CCE1 hart 0, and so on.
  - **This deviates from the "harts 0–7 = CCE0, 8–15 = CCE1" wording in the plan.** Two reasons:
    - `EthernetConfig.processor` is the public `DataMovementProcessor` enum, which stops at `RISCV_7`. A block layout would make CCE1 unreachable from a kernel config without changing that public enum. With interleaving, RISCV_0/RISCV_1 are CCE0/CCE1 hart 0, exactly like Blackhole erisc0/erisc1.
    - The JIT picks firmware image `processor % num_fw_binaries`. That fits interleaving with no build-system change.
  - **To review:** extend `DataMovementProcessor` to RISCV_15 so all 16 harts are reachable. Today, processors 8–15 (harts 4–7 of each CCE) only run firmware.
- `EthProcessorTypes` now has 16 entries.
- `MaxNumKernels` (startup slots) is 16.
- Eth device-print is sized for 16 processors.
- `MEM_IERISC_MAILBOX_SIZE` is 48 KiB.
- `MEM_IERISC_ETH_CCES` = 2 and `MEM_IERISC_ETH_CCE_BASE(cce)` added.
- HAL: two firmware images, one per CCE.
  - Each image, plus its globals, TLS and kernels, is linked `MEM_IERISC_ETH_CCE_BASE(cce)` up.
  - Firmware targets are `erisc` and `eth_cce1`.
  - The `ETH_PHYSICAL_CCE_ID` define selects the CCE.
  - CCE1's reset vector is written at core offset 4 MiB.
  - The BASE size is 8 MiB; UNRESERVED stays in CCE0's SRAM.
- `idle_erisc.cc`:
  - Hart 0 of each CCE runs crt and `risc_init` for its own image.
  - Processor 0 is the only master.
  - Subordinate slots cover processors 1–15 (two words; the host seeds 15 INIT bytes).
- `hw_thread.h`: the idle-eth processor index is `(mhartid % 8) * 2 + ETH_PHYSICAL_CCE_ID`.
- Tests:
  - `KeraunosCce1.{Cce1HartWritesOwnSram, Cce1HartWritesOtherTileCce1Sram, Cce1IdmaCopyCce0ToCce1, Cce0Cce1PingPong}`
  - the existing all-hart tests now use processors 0–7 (four harts on each CCE).

## Mapping working flows to existing APIs (Phase 6 draft)

| Flow | chippy / grendelemulation | tt-metal |
| --- | --- | --- |
| CCE reset, hold, release, reset vector | `sival_arch.utility.cce`: `cce_cores_hold/release`, `cce_reset_vector_set` | UMD `assert/deassert_risc_reset`, reset-vector write at core offset 0 / 4 MiB (`apply_keraunos_*`) |
| CCE fabric/uncore bring-up | `gdb_stub.fabric_init.initialize_fabrics("hsio.T.cce.C")` | none (bring-up owns it) |
| DMRISC windows (own, peer, uncached, HSIO identity) | `utility.cce._program_remap_entry` from `bringup_mmk.py`. SysEng's equivalent is `set_ker_cce_remappers` (firmware basic_init) | `MEM_IERISC_ETH_CCE_BASE`, `MEM_IERISC_UNCACHED_BASE` |
| IDMA copy / addrgen | SysEng `libs/idma/include/cmdbuf_accel.h` (C) | `overlay/cmdbuff_api.hpp`, `addrgen_api.hpp`; same register map (tt-2.0.0) |
| Status / error readback | `cce_read_error_status`, `probe/keraunos_cce_status.py` | none yet |
| D2D link state | `utility.d2d.d2d_tile_addrs`, `probe/d2d_link_status.py` | none (UMD could gate device open on it) |

## Phase 4: chippy CCE1 probe result (session started 22:4x, log `.../scratchpad/cce1_probe.log`)
- CCE1 boots its image from local 4 MiB (reset vector 0x400000). Its harts report **`mhartid` 0–7**, the same as CCE0, so crt0's TLS indexing needs no change.
- **PASS:**
  - uncached stores and loads: CCE1 own (12 MiB alias), CCE1→CCE0 (8 MiB alias), CCE0→CCE1 (12 MiB alias);
  - package-SPA stores and loads: CCE1 → HSIO1 CCE1 and HSIO1 CCE0.
- **Cached stores** (CCE1 own at 4 MiB, CCE1→CCE0 at 0, CCE0→CCE1 at 4 MiB) are not visible to the host. The store sits in the writer's write-back L2, so this is expected and not a mapping fault. Anything another agent reads must be written through the uncached alias or flushed; `experimental/keraunos_cce.h` uses the uncached alias for peer flags.

## Phase 4 tt-metal result (2026-11-06, log `.../scratchpad/phase4.log`)
**20/20 pass on mmk:**
- `KeraunosCce1.{Cce1HartWritesOwnSram, Cce1HartWritesOtherTileCce1Sram, Cce1IdmaCopyCce0ToCce1, Cce0Cce1PingPong}`
- `KeraunosOverlay.*` (8/8)
- `KeraunosPaths.*OtherTileSram` (4/4)
- `KeraunosEmu.{HartZero, AllHarts, EveryEthernetCore, EthernetKernelWritesOtherTileSram}`

So the 16-hart firmware (two images, interleaved processors, shared mailbox) boots and runs kernels on both CCEs.

## Mimir regression and Phase 3 atomics probe (log `.../scratchpad/atomics.log`)
- **Mimir representatives, 11/11 pass:**
  - `MimirEmu.{RuntimeFirmwareInitializesThroughPublicDeviceApi, MimirCceSramChannelsDoNotAliasThroughMetalCluster, AllCcesCopyLocalMimirBlock}`
  - `CceSramRoundTrip*` (8)

  The shared Quasar header changes did not break Mimir CCEs.
- **`idma_l1_atomic_accum_cmdbuf` with the API defaults (fmt 0, op 9):** acked, but it behaves as a plain copy. The destination read 1 instead of 100+i+3, for both own SRAM and HSIO1 SRAM.
  - The encodings are undocumented in tt-metal, chippy and the model. The tests are kept as `DISABLED_` with a note.
  - **Question for SysEng:** which fmt/op is an int32 add?
  - Until then, Keraunos credits use monotonic counters written with `write32` (Blackhole's counter-mode credits).
- Dropped the old uncommitted GDDR attempt: the `test_mimir_emu.cpp` WritesMimirGddr/ReadsMimirGddr tests and the `eth_{read,write}_mimir_gddr.cpp` kernels. `KeraunosPaths` and the chippy repro supersede them. Patch saved at `scratchpad/old_gddr_attempt.patch`.
