# Grendel (Mimir + Keraunos) upstreaming plan: proposed PR series

Status: proposal for review. Nothing here has been implemented yet.

Branches this plan splits up (both rebased onto main on 2026-11-19):

| Repo | Branch | Size vs main |
|---|---|---|
| tt-umd | `kstevens/grendel-emu-additions` | 66 files, about 4,800 lines (plus 355 lines of descriptor YAML) |
| tt-metal | `kstevens/enable_keraunos` | 112 files, about 8,700 lines, of which 1,175 are planning docs and about 4,100 are tests and test kernels |

Line counts below are added plus removed lines from `git diff --numstat` against main, grouped by area. They are estimates; the real split will move a few files between PRs.

## Principles

1. **tt-umd lands first.** Every tt-metal PR that needs new UMD behaviour bumps the submodule to a tt-umd main commit, never to a branch.
2. **Each PR stands alone.** It builds, its own tests pass, and it does not change behaviour for Wormhole, Blackhole or the existing Quasar targets. Grendel paths are reached only through a Grendel SoC descriptor or a Grendel target device.
3. **Keep each PR at or under about 800 changed lines, tests included.** A large PR is split by layer (types, then addressing, then device, then wiring), not by commit.
4. **Tests ride with the code they cover.** Hardware-free tests (descriptor parsing, address resolution, flat-address maps) go in the same PR and run in CI. Emulator tests (mmk) cannot run in CI; their PRs say so and quote the mmk results.
5. **Mimir before Keraunos in tt-metal.** Keraunos reuses the CCE pieces Mimir introduces (CCE HAL patterns, uncached alias, watcher ring-buffer lock, SPA route tables).

## Not for merge

- Planning and worklog documents in the tt-metal root: `keraunos_plan.md`, `keraunos_project.md`, `keraunos_worklog.md`, `grendel_keraunos.md`, `grendel_mimir_addressing.md`, `keraunos_mimir_gddr_read.md`, `mimir_setup.md` (1,175 lines). If a user-facing setup guide is wanted, condense `mimir_setup.md` into a tech report in the last tt-metal PR.
- `tt-umd/.claude/settings.json` (a local tool permission that came in with a JTAG commit, 015fa06f; keep it out of U7 unless the JTAG author wants it).
- `PR_DRAFT_keraunos_cce1.md` in tt-umd (untracked).
- The atomics probes (`test_keraunos_atomics.cpp`, `keraunos_cce_idma_accum_cfg.cpp`, `keraunos_cce_noc_atomic_inc.cpp`, the `DISABLED_` accumulate tests in `test_keraunos_overlay.cpp`, about 320 lines) until remote atomics work. They are investigation code, not regression tests.

## tt-umd series

Dependency order: U1 → U2 → U3 → (U4, U5) → U6. U7 (JTAG) only needs U3 and can go in parallel.

### U1. Grendel architecture and SoC descriptor schema (about 750 lines + YAML)
- `ARCH::GRENDEL`, `CoreType::SMC`, the `grendel` / `mimir` / `keraunos` sections of the SoC descriptor: parse, validation, serialization, schema doc (`docs/yaml_schemas/soc_descriptor.yaml`), nanobind enums.
- Descriptors: `mimir_1x1.yaml`, `mimir_2x_package.yaml`, `mmk.yaml` (355 lines; could be split off as U1b if reviewers prefer YAML separately).
- Tests: `tests/baremetal/test_soc_arch_descriptor.cpp`, `tests/api/test_soc_descriptor.cpp` additions.
- No device code. Lowest-risk entry point and the vocabulary every later PR uses.

### U2. Grendel NOC address resolver (about 1,080 lines; split in two)
- U2a: the `NocAddressResolver` interface and the coordinate-manager hook (the 3 to 6 line changes in the Wormhole, Blackhole and Quasar coordinate managers), with no Grendel implementation yet (about 150 lines).
- U2b: `GrendelNocAddressResolver`: Mimir local and package windows, CCE SRAM and GDDR windows, Keraunos HSIO tiles with CCE-relative and native CCE1 addresses, control-slot addresses (about 930 lines, roughly 300 of them baremetal tests).
- Pure address arithmetic, fully covered by `tests/baremetal/test_grendel_noc_address_resolver.cpp`.

### U3. Grendel device model and flat-address transport (about 820 lines)
- `GrendelImplementation`, `GrendelTTDevice`, `flat_address_transport`, `grendel_flat_address`, `chippy_memory` (the chippy-backed memory map), test access hooks.
- Tests: `tests/api/test_grendel_flat_address.cpp`.

### U4. Emulator backend: `EmuTTDevice` (about 820 lines)
- `EmuTTDevice` over the simulation connector, `simulation_chip` changes, cluster option plumbing for the emulator target.
- Tests: `tests/api/test_emu_tt_device.cpp` (535 lines; skipped in CI without an emulator server; the PR states the mmk results).

### U5. CCE control plane (about 330 lines)
- `TTDevice` reset-vector programming and hart reset and release for Mimir CCEs and Keraunos CCE0 and CCE1 (`apply_cce_reset_vector_write`, Keraunos hart release).
- Small but behaviour-carrying; reviewers who own `tt_device.cpp` should see it on its own.

### U6. Cluster wiring (about 230 lines)
- `tt-umd-workload` cluster and chip selection of the Grendel and emulator devices, top-level CMake changes.
- Makes U1 to U5 reachable from a client such as tt-metal.

### U7. Grendel JTAG (about 760 lines, opt-in, access-gated)
- `cmake/grendel_jtag.cmake` (off by default; needs chippy and GCC 13 or newer), `GrendelJtagProtocol`, `GrendelJtagTTDevice`, mock protocol tests.
- Independent of the emulator path. It can be the last UMD PR or wait until there is a JTAG user.

## tt-metal series

Each PR bumps tt-umd to the main commit that contains what it needs. Dependency order: M1 → M2 → (M3, M4) and M1 → M5 → M6 → M7. M8 is last.

### M1. Grendel target devices and SoC-descriptor plumbing (about 450 lines; needs U1 to U6)
- `TT_METAL_EMU_SERVER`, `TT_METAL_EMU_SOC_DESC`, the GrendelJtag target in `rtoptions` and `tt_target_device`; `tt_cluster` device-mode selection; `get_platform_architecture` mapping `ARCH::GRENDEL` to the Quasar HAL; `metal_soc_descriptor` and `core_coord` additions for Mimir and Keraunos core types.
- Generic fixes the bring-up needed in `kernel.cpp`, `program.cpp`, `sub_device_manager*`, `mesh_device.cpp`, `l1_banking_allocator.cpp` (about 180 of the 450 lines): each should be reviewed for whether it is Grendel-specific or a general bug fix. General fixes should be split into their own small PRs and land first.

### M2. Mimir CCE as a programmable DRAM core (about 640 lines + HAL share)
- HAL: `qa_hal_dram.cpp`, the Mimir section of `dev_mem_map.h`, `dram_address_map.h`, `device_print_mem.h`, the DRISC branches in `dev_msgs.h`, `hal_2xx_common.cpp`, `core_config.h`.
- Firmware: `drisc.cc`, `drisck.cc`; DRISC firmware loading and reset in `risc_firmware_initializer.cpp`.
- Smoke tests from `test_mimir_emu.cpp` (CCE SRAM round trip, hart-zero and all-harts kernels).

### M3. Mimir device APIs and GDDR (about 900 lines with tests)
- `experimental/gddr_cce.h`, `hostdev/mimir_cce_spa_routes.h` and the host route-table writer; the GDDR copy kernels and their tests (`mimir_contract.hpp`, the rest of `test_mimir_emu.cpp`, `test_mimir_backend.cpp`).
- `test_mimir_emu.cpp` (900 lines) and `mimir_contract.hpp` (495 lines) are large: split the tests between M2 and M3 by what they exercise.

### M4. Watcher and device print on CCE cores, Mimir first (about 400 lines)
- Ring-buffer lock for CCE cores, watcher reader for DRISC, `sanitize.h` DRISC checks, `assert.h` and `noc.h` cached-alias conversion (`MEM_L1_UNCACHED_OFFSET`), debug_tools test parameterization for DRAM cores (`test_ringbuf`, `test_waypoint`, `test_sanitize`, `test_assert`, `test_dram_print_output`).

### M5. Keraunos HSIO CCEs as programmable idle-ethernet cores (about 620 lines + HAL share)
- HAL: `qa_hal_idle_eth.cpp`, the Keraunos section of `dev_mem_map.h` (per-CCE windows, uncached alias, interplay with `overlay_addresses.h`), `eth_l1_address_map.h`, per-CCE firmware and kernel build variants (`build_env_manager.cpp`, `hal.cpp`/`hal.hpp`).
- Firmware: `idle_erisc.cc`, `idle_erisck.cc`; `firmware_common.h` and `semaphore_dm_impl.h` CCE-base fixes; the CCE1 kernel-text offset in `slow_dispatch.cpp`.
- Tests: `test_keraunos_emu.cpp`, `test_keraunos_cce1.cpp` (about 440 lines). This PR is the "CCE0 and CCE1 run kernels" milestone.

### M6. Keraunos device API, SPA routes and package paths (about 1,100 lines with tests; split in two)
- M6a: `experimental/keraunos_cce.h`, `hostdev/keraunos_spa_routes.h`, the route-table writer in `risc_firmware_initializer.cpp`, `test_keraunos_api.cpp` and its kernels.
- M6b: package-path coverage: `keraunos_cce_fixture.hpp`, `test_keraunos_paths.cpp`, `test_keraunos_overlay.cpp` (without the disabled atomics), `test_keraunos_fabric.cpp`, Mimir-to-Keraunos kernels and the Mimir `keraunos_hsio_sram_*` helpers in `gddr_cce.h`.

### M7. Watcher on Keraunos CCEs (about 250 lines)
- Host NOC sanitizer per-CCE windows, device eth-address and L1 checks for both CCEs, eight-hart names in the watcher reader, `test_keraunos_watcher.cpp`.
- Small; depends on M4 (shared ring-buffer and alias changes) and M5.

### M8. CI and documentation (about 240 lines)
- `tests/scripts/quasar/quasar_short_tests.yaml` entries for the emulator suites; a condensed setup guide if wanted.
- Last, because it lists tests from every earlier PR.

## Preparation before opening any PR

1. Run `pre-commit` and `clang-format` on both branches (not installed on the dev machine so far).
2. For tt-metal M1, sort the generic bring-up fixes into "general bug fix" and "Grendel-specific", with a one-line justification each.
3. Build each PR's branch on its own from main (cherry-pick into topic branches) and run the non-emulator tests; run the mmk suites for the PRs that touch device behaviour.
4. Decide the reviewers per PR: UMD descriptor and resolver owners for U1 and U2; `tt_device.cpp` owners for U5; HAL and firmware owners for M2 and M5; watcher owners for M4 and M7.

## Questions for review

1. Is the order Mimir before Keraunos right for tt-metal, or should Keraunos come first?
2. Should the descriptor YAMLs (`mmk.yaml` and the Mimir packages) land with U1, or after U4 when they can be exercised?
3. Should Grendel JTAG (U7) go upstream now, given it is opt-in and access-gated, or wait for a user?
4. Is about 800 lines per PR the right ceiling for these teams?
5. Do the emulator-only tests belong upstream if CI can never run them, or should they live in a separate test binary that is skipped by default (as the UMD `EmuTTDevice` tests are now)?
