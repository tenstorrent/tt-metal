# Keraunos CCE paths on MMK: plan

## Context

Keraunos (K0) on the MMK emulation (M0 + M1 + K0) runs tt-metal idle-ethernet kernels on HSIO CCE0, HSIO tiles 0–3. Writes already work:
- own SRAM
- other HSIO tiles' SRAM
- M1 GDDR through K0 D2D2 → M1 D2D1

Every read of M1 GDDR hangs, whether by CPU load or IDMA (`keraunos_mimir_gddr_read.md`). CCE1 is not enabled. The goal (`keraunos_project.md`):
- every CCE path works, in both directions and as reads and writes
- CCE1 runs at its native 4 MiB base
- device-side APIs hide CCE0/CCE1 and route addresses
- tests show how a Quasar/Keraunos fabric would use those APIs

Setup should go through existing grendelemulation, chippy and tt-metal APIs, the way Blackhole fabric does. A path that can't be fixed with reasonable effort gets a small chippy reproduction and is parked until SysEng gives feedback.

### Decisions already made
- **CCE1 layout:** native. CCE1 code and data run from local 4 MiB. Both CCEs see CCE0 SRAM at 0 and CCE1 SRAM at 4 MiB.
- **CCE1 in tt-metal:** one ethernet core per HSIO tile, like Blackhole erisc0/erisc1. The 8 MiB TL1 is that core's L1. Processors 0–7 are CCE0 harts at base 0 and processors 8–15 are CCE1 harts at base 4 MiB. The user selects CCE1 with `EthernetConfig.processor`.
- **Extra paths in scope:**
  - Mimir CCE → HSIO SRAM
  - remote atomic increment
  - K CCE ↔ Mimir CCE SRAM, read and write
  - CCE0 ↔ CCE1 sync
- **chippy `bringup_mmk.py` diff:** the uncommitted change that always releases M1 D2D1 and K0 D2D2 stays uncommitted. It is committed later only if it turns out to be needed.

### Key facts from exploration
- **Bring-up DMRISC.** `bringup_mmk.py` programs DMRISC entry 0 as `[0,4MiB)` → own SRAM and entry 1 as uncached `[8MiB,+4MiB)` → own SRAM. Entries 6–9 are identity windows over HSIO SPA `0x12_0400_0000 + n*0x400_0000`. There is no GDDR entry, so GDDR is reached on a DMRISC miss. Each tile also has a qsr1_boot `noc2axi` remap.
- **SysEng setup differs.** SysEng's `set_ker_cce_remappers` (`firmware/libs/basic_init/src/basic_init.c:404`) uses `tl1_base=0`, maps HSIO windows to local addresses (not identity), and uses the nine-window `noc2axi` table.
- **SysEng verification.** SysEng verified K HSIO2 CCE0 IDMA **reads of Mimir CCE SRAM** over BAR1 (`0x12_80C0_1000`) in `k2m_remap/ker_cce_dma_m2k_remap/src/main.c`. No app reads Mimir GDDR from Keraunos.
- **K D2D ATT decode** (`noc2axi_m2k.py:219`):
  - `0x1_0022_…` → M1 D2D1, the direct link.
  - `0x1_0020_…` → M0 D2D1, which goes through M1.
  - M0 GDDR from K is therefore a multi-hop path.
- **Device code today.** tt-metal has no Keraunos device API. `gddr_cce.h` and `MimirCceSpaRoutes` are Mimir DRISC only. Keraunos tests hardcode `0x1206800000` and `0x1002200000000`.
- **Blackhole fabric paths** that Keraunos needs equivalents for:
  - local SRAM → remote SRAM bulk copy
  - remote 32-bit write or atomic increment (credits)
  - worker → router-channel SRAM write (Mimir → K)
  - router → destination L1/DRAM write and atomic
  - remote reads (UDM)
  - erisc0/erisc1 split with a local sync
- **NIU write-ack bug** (`grendelemulation/docs/dx/NIU-WRITE-ACK-BUG.md`): a store being "accepted" is not proof it landed. Always check with a host readback.

## Working rules
- **Emulator access.** Use only the `emu` tool (grendel-emu skill), with the `emu chippy` subcommand added in Phase 0. Never use `--idle` or raw ssh. Batch tests and release before long analysis or rebuilds.
- **Hang-prone tests and timeouts.** `TT_METAL_OPERATION_TIMEOUT_SECONDS` only bounds the wait for the kernel to finish (`llrt.cpp:556`). It does not cover the ~90 s of Keraunos setup: firmware load has a fixed 60 s EmuAxi bound (`risc_firmware_initializer.cpp:146`), and binary/mailbox writes are unbounded.
  - So any test that might hang uses `TT_METAL_OPERATION_TIMEOUT_SECONDS=120`. That is generous for a kernel that finishes in seconds on the emulator.
  - `emu test --timeout` stays at the 420 s default. Worst case is about 90 s setup + 120 s kernel wait + teardown, well under that.
  - If one batch holds several hang-prone tests, raise `--timeout` to about 200 s per hang-prone test.
  - Put risky probes last in a batch, because a hung hart can force a restart that reloads firmware (~4 min).
- **Records:**
  - Copy this plan to `tt-metal/keraunos_plan.md`.
  - Log choices, discoveries and changes per step in `tt-metal/keraunos_worklog.md`: what was tried, the result, and the emulator log path.
- **Commits.** Commit at each phase boundary in each repo touched: tt-metal `kstevens/enable_keraunos`, the UMD submodule, chippy `kstevens/metal_bringup`, and grendelemulation only if needed.
- **Time-box per failing path.** Stop after about 3 emulator iterations that produce no new hypothesis. Then write a chippy reproduction under `chippy/validation/metal_bringup/repro/`, record it, and move on.
- **Existing GDDR-read changes.** The current uncommitted test diff and the two untracked kernels are kept only where they're useful. `EthernetKernelWritesMimirGddr` passes, so keep it. Rewrite the read kernel as part of Phase 1.

## Phase 0: emu tool support for chippy, then baseline
1. **`emu` tool enhancement** in `tt-umd-simulators/emu/grendel-session/emu` (approved; commit in tt-umd-simulators). Today `emu test -- bash -c 'cd chippy && uv run …'` works, but it is clumsy and has three gaps:
   - `--fw-script` only exists on `acquire`.
   - Scripts must live in `validation/metal_bringup/`.
   - There is no way to start the server without Metal bring-up. The SysEng `run_validation_test.py` flow does its own SMC `chip_init`, so it needs one.

   Add:
   - `emu chippy [--model M] [--fw-script S|none] [--timeout T] -- <script.py> [args…]`
     - Runs `uv run --project validation python <script> <host> <port> [args]` from `CHIPPY_DIR`.
     - Paths are relative to the chippy root.
     - Counts as busy like `emu test`, so no idle release.
     - Prints the same emulator output, events, stall detection and exit summary.
   - `--fw-script` on `emu test` and `emu chippy`:
     - accepts any path under the chippy root
     - `none` means server only, no bring-up, so the SysEng flow can be run
     - if the requested script differs from the loaded session's, the session restarts with a note
   - `emu chippy --raw -- <cmd…>` for non-Python commands such as `run_validation_test.py` with its yaml. `TT_METAL_EMU_SERVER`, `EMU_HOST` and `EMU_PORT` are set.
   - Update the `grendel-emu` skill doc and the README.

   Check: one `emu chippy` run of a read-only probe script (reads `cce_read_error_status` on HSIO2 CCE0) against an `mmk` session.
2. **Baseline (one batch).** Run all `KeraunosEmu.*` except `EthernetKernelReadsMimirGddr`, plus the representative Mimir contract suite, on `mmk`. Record the results in the worklog. Rebuild first if anything changed: `cmake --build build --target unit_tests_context`.

## Phase 1a: Quasar overlay libraries on the Keraunos CCE
The fabric team needs the `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/` libraries, the cmdbuf programming and custom RoCC instructions, to work on the Keraunos CCE DMRISC. Nothing has run them on an idle-eth hart yet, and Phase 1 relies on IDMA.

1. **Inventory.** Find which overlay blocks the K CCE actually has. Sources: chippy `common/include/registers/ker/` (CCE/overlay register views), SysEng firmware that issues RoCC/IDMA (`ker_cce_dma_m2k_remap`), and the grendelemulation CCE model. Expected blocks:
   - RoCC command buffers 0/1 plus the simple buffer, for NOC and IDMA (`cmdbuff_api.hpp`, `rocc_instructions.hpp`)
   - address generators (`addrgen_api.hpp`, `addrgen_state.h`)
   - counter remapper (`remapper_api.hpp`)
   - overlay interrupts and per-trid interrupts

   Likely Tensix-only and out of scope: `llk_intf_api.hpp`, `fds_*`. Mark each block present, absent or unknown in the worklog. Absent or unknown blocks are listed for the user, not tested.
2. **Build plumbing.** Make sure the overlay headers compile under `COMPILE_FOR_IDLE_ERISC` for Quasar: include guards, `l1_cached_view`, and the `noc_nonblocking_api_v2` selection. The mstatus.XS/RoCC enable (`xcustom_test.hpp` `RVTEST_XS_ENABLE`) must be set in `idle_erisc.cc` the same way `tt-2xx/dm.cc` sets it for DM cores.
3. **Tests.** New suite `KeraunosOverlay.*` in `tests/tt_metal/tt_metal/context/test_keraunos_overlay.cpp`. It reuses the existing Quasar DM example kernels where possible (`data_movement/quasar_examples/quasar_idma/kernels/idma_basic_example.cpp`, `idma_1d_strided_example.cpp`, `quasar_addrgen/kernels/addrgen_1d_example.cpp`, `addrgen_2d_example.cpp`). Those kernels get a small address-parameterisation shim so they take SRAM bases from runtime args instead of Tensix L1. Each kernel runs on HSIO0 CCE0 and is checked with a host readback:
   - IDMA copy, local → local: basic, 1D-strided, wrapping, and the `free_space`/`idma_acked(trid)`/`all_idma_acked` status helpers
   - IDMA scatter-list, and IDMA L1 atomic accumulate (`idma_l1_atomic_accum_cmdbuf`)
   - NOC cmdbuf read, write, inline write and atomic increment (`noc_read_cmdbuf`, `noc_write_cmdbuf`, `issue_write_inline_cmdbuf`, `noc_atomic_increment`) to own SRAM and to another HSIO tile's SRAM, if the K HSIO NoC coordinates support it
   - addrgen 1D/2D driving the IDMA cmdbuf (`paired_cmdbuf`, `push_both_addrgen`)
   - counter remapper configure, readback and error status, if present
   - per-trid count-zero interrupt pending/clear, polled and not taken
   - the same tests on all 8 harts in parallel (cmdbuf ownership per hart), and later on CCE1 (Phase 4)
   - `noc_async_read`/`noc_async_write` from `api/dataflow/dataflow_api.h` on the CCE, the user-facing layer built on these cmdbufs
4. Feed the results into Phase 1: which mechanism (CPU, IDMA, NOC cmdbuf) is known-good locally decides how the remote probes are written.

## Phase 1: isolate the read problem (probe ladder)
Add small probe tests, `KeraunosEmuProbe.*`, in `test_mimir_emu.cpp`. Each probe uses one kernel taking `{op, src, dst, len}` runtime args: CPU load, CPU store, or IDMA copy. Run them in one batch, ordered from safest to riskiest:
1. IDMA copy from own SRAM to own SRAM, global and local addresses. Does IDMA work on an idle-eth hart at all?
2. CPU load and IDMA read of **another HSIO tile's** SRAM. Is this the first remote read?
3. CPU load and IDMA read of **M1 CCE SRAM** over BAR1, the SysEng-verified path.
4. CPU load of M1 GDDR through D2D1, `0x1_0022_…`. If it hangs, `emu chippy` runs a script that dumps `cce_read_error_status` (`sival_arch/utility/cce.py:445`) and the D2D2/ATT status.

Interpretation:
- **If 3 passes and 4 hangs,** the problem is specific to GDDR: BAR2, ATT translate, or the M1 GDDR NIU read response. Next steps:
  - Compare the M0 CCE → M1 GDDR read, which works, against the K → M1 GDDR read, hop by hop.
  - Try ATT `translate_addr` and BAR variants in `noc2axi_m2k.py`/`bringup_mmk.py`.
- **If 3 also hangs,** the problem is in the read-response path or our bring-up differs from SysEng's. Next steps:
  - Run SysEng's `k2m_remap_run_emu` flow unchanged on `mmk`: `emu chippy --fw-script none --raw -- validation/tools/run_validation_test.py …` with `transport_type: emu_axi`.
  - If it passes, apply SysEng's remapper/`noc2axi` settings one at a time to `bringup_mmk.py` (`set_ker_cce_remappers`, the nine-window `noc2axi` table) to find the difference.
- **If 1 or 2 fails,** fix IDMA usage on idle-eth first. Likely causes: VC selection or `l1_cached_view`. Reference: `cmdbuff_api.hpp:1540` `idma_copy_cmdbuf`.
- **If time-boxed out,** produce a chippy reproduction:
  - Start from `ker_cce_dma_m2k_remap`, which is already SysEng's format.
  - Add one IDMA read of M1 GDDR `0x1_0022_0000_1000`.
  - Run it with the k2m_remap emulator config.
  - That tells us whether the failure is in the model/firmware or in our stack.

Any fix to `bringup_mmk.py` is expressed through the existing chippy helpers (`initialize_fabrics`, `set_ker_cce_remappers`-equivalent Python, `m2k_host_prepare_keraunos_d2d`) where possible.

## Phase 2: core goal paths with host-supplied routes
1. **Routes table (tt-metal).** Replace the hardcoded MMK constants with a host-written `KeraunosSpaRoutes` table, following `hostdev/mimir_cce_spa_routes.h` and `risc_firmware_initializer.cpp:70–106`. It holds:
   - HSIO SRAM SPA base and tile stride
   - Mimir CCE SRAM BAR1 base and stride
   - Mimir GDDR route bases per Mimir as seen from K
   - the D2D index
   It is filled from the `keraunos:` and `mimir_channels` blocks of `mmk.yaml`. Add descriptor fields in UMD only where a value isn't derivable.
2. **Tests (read and write each):**
   - own SRAM
   - every other HSIO tile's SRAM
   - M1 GDDR, both GDDR tiles, several offsets
   - M0 GDDR through M1 (multi-hop `0x1_0020_…` route)
   Every write is checked with a host readback. Every read is checked by comparing against a host-written pattern.

## Phase 3: extra paths
1. **K CCE ↔ Mimir CCE SRAM.** Read and write M0/M1 CCE0/CCE1 SRAM over BAR1. Host checks through `read_mimir_cce_sram`.
2. **Mimir CCE → HSIO SRAM.** A Mimir CCE kernel (DRISC) stores into K HSIO SRAM using the Mimir-side SPA for the K window. Get that SPA from `mimir_cce_dma_m2k*` firmware and the Mimir ATT/endpoint table. Also test the read direction. This is the "worker → router channel" path.
3. **Remote atomic increment.** Use `noc_atomic_increment` / `idma_l1_atomic_accum_cmdbuf` (`cmdbuff_api.hpp:1566/1597`). Cover:
   - K → other HSIO SRAM
   - K → Mimir CCE SRAM
   - Mimir → K SRAM
   If atomics don't cross D2D, record that. The fallback is the Blackhole counter-mode credits: a write-only monotonic counter (`fabric_router_flow_control.hpp:52`).

## Phase 4: CCE1, native layout
1. **chippy `bringup_mmk.py`.** Release CCE1 per tile and program CCE1's DMRISC with the native view:
   - local `[0,4MiB)` → CCE0 SRAM
   - local `[4,8MiB)` → CCE1 SRAM
   - CCE1's uncached alias
   - CCE1's identity HSIO windows
   Take the alias address and the CCE1 register block (+16 MiB) from `common/include/registers/ker/smc_view/hsio_tile.h` and the existing CCE0 code. Check the CCE0 entries keep the same native view. Reference: `grendelemulation/docs/dx/mimir-cce1-local-tl1-base-claims-peer-sram.md`.
2. **UMD:**
   - `mmk.yaml`: `eth_l1_size` becomes 8 MiB per tile, and each core lists both CCEs.
   - Resolver: SRAM = `local_window + hsio*tile_stride + off`.
   - `apply_keraunos_reset_vector_write` / `apply_keraunos_hart_reset` (`tt_device.cpp:115/214`): harts 8–15 go to the CCE1 control block (+16 MiB), and the reset vector is per CCE.
3. **tt-metal:**
   - Quasar `dev_mem_map.h`: add `MEM_IERISC1_*` bases at 4 MiB (firmware, kernel, local, mailbox view).
   - `qa_hal_idle_eth.cpp`: 16 processors in two processor classes, CCE0 and CCE1, each with its own firmware base and linker script.
   - Mailbox: keep it in CCE0 SRAM. CCE1 reaches it natively at 0, which is the Blackhole shared-mailbox model.
   - `idle_erisc.cc`: a `ETH_PHYSICAL_CCE_ID` build define, and each CCE answers its own run/done slots.
   - Firmware initializer: loads both images.
4. **Tests:**
   - a CCE1 hart writes its own SRAM; the host reads it at tile + 4 MiB
   - both CCEs run together
   - CCE0 ↔ CCE1 flag ping-pong (sync)

## Phase 5: device APIs and fabric-shaped tests
New header `tt_metal/hw/inc/experimental/keraunos_cce.h` (idle-eth, Quasar), named after the dataflow and fabric APIs:
- **Identity and local addressing:**
  - `my_eth_cce()`, `my_hsio()`
  - `own_sram(off)`, `peer_cce_sram(off)`
  - uncached variants of both
  These hide the 0/4 MiB difference.
- **Global addressing** (from `KeraunosSpaRoutes`):
  - `hsio_sram_addr(tile, cce, off)`
  - `mimir_cce_sram_addr(mimir, cce, off)`
  - `mimir_gddr_addr(mimir, off)`
- **Transfers:** thin wrappers over the overlay `cmdbuff_api.hpp` that Phase 1a verified, so the fabric team uses the same cmdbuf/RoCC layer. They fall back to CPU load/store only where Phase 1 shows IDMA/NOC doesn't work:
  - `async_write`, `async_read`, `inline_write32`, `atomic_inc`, `write_barrier`, `read_barrier`
- **Peer CCE sync:** `peer_signal` / `peer_wait`. Shared-SRAM flags, like `wait_for_other_local_erisc`.
- **Mimir side:** a matching `keraunos_hsio_sram_addr` in the Mimir CCE API for the worker → router direction.

Fabric-shaped tests:
1. **Worker → router → destination.** A Mimir CCE "worker" writes a packet (header + payload) into a K CCE0 channel buffer, then bumps a credit counter. K CCE0 forwards the payload to another HSIO tile's SRAM or Mimir GDDR, then returns a credit.
2. **Sender/receiver split.** CCE0 is the sender and CCE1 the receiver on the same tile, using peer sync, mirroring erisc0/erisc1.

## Phase 6: map back and wrap up
- **chippy.** Move any ad-hoc bring-up steps onto existing helpers. Note in the worklog where that wasn't possible and why.
- **tt-metal.** Check the APIs line up with `dataflow_api`/fabric naming. Update `grendel_keraunos.md`.
- **Final regression:** all `KeraunosEmu*` and the Mimir contract suite on `mmk`, plus the `mm` and `mimir` models to confirm nothing regressed. Final commits, then a summary in the worklog of what works, what's parked, and the reproductions.

## Verification (per phase)
- Phase 1a ends with `KeraunosOverlay.*` green on `mmk`, plus the existing Quasar DM example tests (`QuasarIdmaOps.*`, addrgen) unchanged wherever they run.
- Each phase ends with one batched `emu test --model mmk -- ./build/test/tt_metal/unit_tests_context --gtest_filter='KeraunosEmu*:<mimir reps>'` run. Read all three result sections and record the log paths.
- Regression on the Mimir-only packages (`mimir`, `mm` models) at the end of Phases 2, 4 and 6.
- Parked paths have a chippy reproduction that runs with one command against an acquired `mmk` session.

## Checkpoints and stop conditions (agreed 2026-11-06)
- No approval needed between phases. Commit locally (never push) and log in `keraunos_worklog.md`.
- **CCE1 uncached alias:** local +8 MiB from CCE1's base, so 4 MiB + 8 MiB = 12 MiB. The user confirmed this in earlier tests.
- **grendelemulation:** may be changed and committed locally. Never push.
- **chippy `bringup_mmk.py` always-release diff:** commit it only once it's shown to be needed.
- **Pause for user review** after the Phase 5 device API header is drafted, before writing the fabric-shaped tests.
- **Also stop for:** CCE1 facts that can't be determined, and Mimir regressions with no obvious fix.
- **Parked paths:** report them with a chippy reproduction, then carry on with the paths that don't depend on them.
- **UMD repo rules** (`tt_metal/third_party/umd/CLAUDE.md`, which take precedence there): leave UMD changes unstaged with a PR-description draft in an untracked file, never commit them without the user's review, record design assumptions when running unattended, and run pre-commit plus a full build before handing back.
