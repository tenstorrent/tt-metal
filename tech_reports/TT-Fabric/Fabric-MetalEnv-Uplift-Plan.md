# Fabric → MetalEnv uplift: working plan

Tracking issue: [#39772](https://github.com/tenstorrent/tt-metal/issues/39772).
Background: [`Fabric-Architecture-Orientation.md`](Fabric-Architecture-Orientation.md) §9.

## Status (as of 2026-10-09)

`MetalContext::instance()` in `tt_metal/fabric/`: 161 calls when planned, 27 on `main` now (28 grep hits; one is a comment in `physical_system_discovery.cpp`). All 27 are step 4 long poles: 18 in `fabric.cpp`, 4 in `fabric_mux_config.cpp`, 3 in `fabric_switch_manager.cpp`, 2 in `pipeline_builder.cpp`.

| Item | PR | Status |
| --- | --- | --- |
| Steps 2 and 3 (PR1-PR7): internal fabric code takes its dependencies explicitly | [#57572](https://github.com/tenstorrent/tt-metal/pull/57572), [#57575](https://github.com/tenstorrent/tt-metal/pull/57575), [#57598](https://github.com/tenstorrent/tt-metal/pull/57598), [#57604](https://github.com/tenstorrent/tt-metal/pull/57604), [#57607](https://github.com/tenstorrent/tt-metal/pull/57607), [#59532](https://github.com/tenstorrent/tt-metal/pull/59532), [#57610](https://github.com/tenstorrent/tt-metal/pull/57610) | Done. Merged 2026-09-29 to 2026-10-09. |
| Regressions from express routing (#57785) and the auto-mapper (#55004), 6 lookups | [#59760](https://github.com/tenstorrent/tt-metal/pull/59760) | Merged 2026-10-08. |
| `MetalEnv::configure_fabric()` with a freeze at materialization; bug 1; narrower dispatch-fabric switch | [#59387](https://github.com/tenstorrent/tt-metal/pull/59387) | Merged 2026-10-09. |
| Step 4: public queries and `FabricMuxV2Config` take `const MeshDevice&` (-10) | [#59551](https://github.com/tenstorrent/tt-metal/pull/59551) | Merged 2026-10-09. |
| Step 1: bugs 2, 3 and 6, and the builder-state reset | none | Not started. This is the main remaining correctness work. |
| Guard against new bare lookups in `tt_metal/fabric/` | none | Not started; optional. Unblocked now that #59551 is in. |
| Step 4 long poles (27 lookups) | none | Investigated and rated in [the step 4 file](Fabric-MetalEnv-Uplift-Step4.md); not scheduled. Most need public API changes that break ttnn Python and tt-blaze. |

Related MetalEnv work outside this plan:

- Merged: [#58230](https://github.com/tenstorrent/tt-metal/pull/58230) (`MetalEnvDescriptor` is an aggregate; `create_*` takes `CreateMeshDeviceOptions`), [#57812](https://github.com/tenstorrent/tt-metal/pull/57812) (`SubDevice` no longer creates the silicon context), [#59588](https://github.com/tenstorrent/tt-metal/pull/59588) (semaphore scopes resolve against the mesh device's env), [#59584](https://github.com/tenstorrent/tt-metal/pull/59584) (`tt_metal/impl/buffers`), [#55524](https://github.com/tenstorrent/tt-metal/pull/55524) (leftover impl sites), [#59720](https://github.com/tenstorrent/tt-metal/pull/59720) (service core manager).
- Open: [#59742](https://github.com/tenstorrent/tt-metal/pull/59742) (profiler, 1 of 2; approved), [#59756](https://github.com/tenstorrent/tt-metal/pull/59756) (inspector), [#60135](https://github.com/tenstorrent/tt-metal/pull/60135) (profiler, 2 of 2; draft), [#60120](https://github.com/tenstorrent/tt-metal/pull/60120) (memory report dir), [#59941](https://github.com/tenstorrent/tt-metal/pull/59941) (read arch from the device in scope).

## Decisions

| Topic | Decision |
| --- | --- |
| Runtime fabric reconfiguration | Removed for explicit envs (#59387). Fabric is set with `MetalEnv::configure_fabric()` after construction (last call wins) and frozen once the topology is materialized, at the first `get_system_mesh()` or `create_*`. Switching config after that means re-creating the env. |
| Transition | The implicit default env (`MetalContext::instance()`) keeps legacy mutability through the internal `set_fabric_config` until the test and model migration is done. Then legacy reconfiguration is deleted. |
| Python | Bind `MetalEnv`, `MetalEnvDescriptor`, `configure_fabric` and `FabricConfigDescriptor`; conftests and demos construct an env. Details deferred. |
| Dispatch-only fabric | Automatic but narrow (#59387): `MetalEnvImpl::enable_fabric_for_dispatch()`. Making it explicit (option B below) remains the possible end state. |
| Source of truth for fabric config | The env. `MetalEnv::get_fabric_config_descriptor()` returns the *requested* config; the effective config can differ when dispatch enables fabric. The `ControlPlane` and `FabricContext` copies are read-only views until the ownership step collapses them. What's left of [#37087](https://github.com/tenstorrent/tt-metal/issues/37087) is deleting the `MetalContext::get_fabric_*` forwarders. |
| Public fabric API | Take `const MeshDevice&`. The two pre-device queries can't take `MetalEnv`, because reaching the env's control plane freezes the fabric config; `get_all_mgd_fabric_types` takes the MGD instead. Change signatures in place and port every caller in the same PR; no shims. Leave Blaze-owned APIs alone. Details in [the step 4 file](Fabric-MetalEnv-Uplift-Step4.md). |

---

## Dispatch-only fabric

### Current behavior

Under fast dispatch, opening any remote (non-MMIO) chip widens the open to every chip in the cluster, because fabric requires all chips to be active (TODO #24413). Remote chips need fabric because the dispatch kernels relay through fabric mux kernels (`relay_mux.cpp`). The dispatch link is the last forwarding link and is reserved through `ControlPlane::reserve_routing_planes`, so user traffic never gets that plane. Slow dispatch never needs fabric.

Since #59387:

- On a DISABLED env, `DeviceManager` calls `MetalEnvImpl::enable_fabric_for_dispatch()` (`device_manager.cpp:279`). It sets `FABRIC_1D`, STRICT and 1 plane, and rebuilds the control plane if one exists, checking that the system mesh keeps the same shape and device mapping. Tensix, UDM and router settings are left as they were.
- The requested config stays DISABLED; only the effective config changes.
- With fabric already enabled, it only asserts that at least one plane is active.
- `DeviceManager` still forces `fabric_reliability_mode_` to STRICT through `MetalEnvAccessor` (`device_manager.cpp:294-295`), overriding a user's RELAXED setting.

The switch exists for backward compatibility: when fast dispatch moved onto fabric and the tunneler was deleted (#24341, #24961, 2025-07), it let existing N300/T3K code keep running without stating fabric. There is no fallback path anymore, so fast dispatch to a remote chip with fabric DISABLED can't work.

Affected products: only those with remote chips, i.e. N300, N300 2x2, T3K, and multi-host setups built from them (multi-host always opens every local chip). N150, WH Galaxy, P100/P150/P300 and BH Galaxy/LoudBox/QuietBox have no remote chips.

### Possible end state: option B (explicit)

DISABLED means disabled. If fast dispatch opens a remote chip with fabric DISABLED, device open fails with a clear message, e.g. "set `FabricConfigDescriptor{FABRIC_1D, num_routing_planes=1}`". This deletes `enable_fabric_for_dispatch()` and the forced-STRICT write, so config is fully explicit and the user's reliability mode is honored. Opening only a PCIe chip behaves as today.

Cost, if the legacy default env keeps the switch until legacy deletion (assumed):

- Python: about 214 N300/T3K-relevant `test_*.py` files take `mesh_device` without stating fabric. Most go through the root `conftest.py` `mesh_device` fixture and `t3k_single_board_mesh_device`; `models/common/tests/conftest.py` already picks a fabric config from the mesh shape.
- C++: about 156 `TEST_F`s, mostly through `MeshDispatchFixture`, `N300MeshDeviceFixture`, `MeshWatcherFixture` and `DevicePrintFixture`. `MeshDeviceFixtureBase` defaults fabric to DISABLED.
- Explicit `MetalEnv`: `MetalContextIntegrationTest.HelloWorld` opens the full system mesh on a DISABLED env and would need a descriptor on N300/T3K.

To keep migrated fixtures behaving the same: request `num_routing_planes = 1`, not all planes (fewer planes leave the last PCIe-to-PCIe link and more ethernet cores free, which dispatch and watcher fixtures may rely on). Also offer a "does opening these device ids under fast dispatch need fabric?" query and point the error message at it.

---

## Step 1: remaining bug fixes

Done: bug 1 (descriptor with no plane count crashed; #59387) and bug 5 (dead `set_fabric_tensix_config`; #57572). Bug 4 (the env checks the default context for open devices) is dropped: once explicit envs are frozen, that check is reachable only from the legacy env, whose context is the default one.

Open, all inferred from code and not yet run on hardware:

- **Bug 2: closing a mesh device resets the env's fabric config.** `FabricFirmwareInitializer::post_teardown` calls `set_fabric_config(DISABLED)` (`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:433-436`). The internal setter ignores #59387's freeze. It runs before devices are closed, so `teardown_fabric_config` (`metal_env.cpp:445-455`) sees open devices and skips `clear_fabric_context()`. The env then says DISABLED while `ControlPlane` and `FabricContext` hold the old config.
- **Bug 3: a second `create_mesh_device` on the same explicit env loses fabric.** It follows from bug 2: `MetalEnv::create_mesh_device` (`metal_env.cpp:785`) doesn't re-apply the config. On N300/T3K the second open gets dispatch-only `FABRIC_1D`; on all-PCIe systems (WH Galaxy, Blackhole) it gets no fabric at all.
- **Per-session build state is write-once.** `FabricBuilderContext` records per-chip router counts and the master router channel, and setting either twice for the same chip is a TT_FATAL (`fabric_builder_context.cpp:261-286`). Today every legacy open rebuilds the control plane, which creates a fresh builder context. With frozen config nothing triggers a rebuild, so fixing bug 2 alone makes the second open hit "tried to set num initialized routers again". Bugs 2 and 3 can't be fixed without this.
- **Bug 6: no test for re-creating a silicon env with a different fabric config** (e.g. `FABRIC_1D`, then `FABRIC_2D`). That is the only way to switch config under the new model. Existing tests cover re-creation with the same config only.
- **Opening a subset of chips on N300/T3K with fabric enabled.** Fabric needs every chip active, and the open is widened only under fast dispatch when a requested chip is remote. A unit mesh on a PCIe chip, or any subset under slow dispatch, is fatal. Legacy hides this with the close-time reset. Options: widen whenever fabric is enabled on such a cluster, or require a separate env with fabric DISABLED.

**Planned fix, one PR:** stop the close-time reset for explicit envs only, and reset `FabricBuilderContext` at close; if that's awkward, move `FabricBuilderContext` into the session first. Keep the reset for the legacy default env, where it is load-bearing: without it, "open full mesh (implicit `FABRIC_1D`), close, open a single PCIe chip" hits the all-chips-active fatal. Add regression tests for bugs 3 and 6.

Deferred to the ownership step: the forced-STRICT write through `MetalEnvAccessor`; the three copies of config (env, `ControlPlane`, `FabricContext`); the lazy control-plane rebuild.

---

## Steps 2 and 3 (done)

127 lookups removed (126 in `tt_metal/fabric/`, 1 in the fabric firmware initializer). The main change: `FabricContext`, built only in `ControlPlane::initialize_fabric_context()`, stores a `ControlPlane&`, `const Hal&`, `const Cluster&`, `const RunTimeOptions&` and the full `FabricTensixConfig`, so every builder that holds a `FabricContext&` has its dependencies in scope.

| PR | Scope | Calls | GitHub |
| --- | --- | --- | --- |
| PR1 | `FabricContext` stores its dependencies; `fabric_builder_context.cpp`; `Cluster` queries instead of `GetNumAvailableDevices`/`GetNumPCIeDevices`; delete `MetalContext::set_fabric_tensix_config` | 11 | #57572 |
| PR2 | Leaves: emulation skip, PGD helpers and `generate_rank_bindings`, routing-path helpers, telemetry reader | 5 | #57575 (PGD part redone in #59760) |
| PR3 | Build entry points: `fabric_init` takes `MetalEnvImpl&`; `FabricContext&` through `FabricBuilder` and the router builders | 29 | #57598 |
| PR4 | ERISC router builder configs take `const FabricContext&`; dispatch core type threaded from `Device::compile_fabric` | 30 | #57604 |
| PR5 | Tensix config session inputs passed in through `MetalEnvImpl`; removed public `ControlPlane::initialize_fabric_tensix_datamover_config()` | 22 | #57607 |
| PR6 | Tensix builders and per-core configs hold `const FabricContext&`; internal code stops using the public size getters | 23 | #59532 |
| PR7 | `append_fabric_vc2_connection_rt_args` takes `const FabricContext&`; deleted `is_TG_gateway_connection` | 6 | #57610 |

### Traps for further work in `tt_metal/fabric/`

- **Lookups during control-plane construction.** A bare `get_control_plane()` reached while `MetalEnvImpl` builds the control plane re-enters a non-recursive mutex and deadlocks, and a bare `instance()` there can implicitly create the default silicon context (#50041/#50043). The `FabricContext` constructor must stay free of lookups.
- **Reference members break move-assignment.** Builders assigned into a `std::optional` need `std::reference_wrapper` or pointers, as `ControlPlane` does.
- **The shared host/device header.** `fabric_common.h` is shared with kernels. Host types may appear only as forward declarations, and host-only signatures stay inside the `#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)` block.

### Keeping the count down

Within a week of PR1-PR2 merging, two upstream merges (#57785, #55004) added 6 bare lookups back; #59760 removed them. An optional pre-commit hook could fail when the count of `MetalContext::instance()` in a changed file under `tt_metal/fabric/` goes up against the merge-base. Only increases are flagged, so no allowlist is needed, and it can reuse the merge-base logic in `scripts/detect_legacy_device_op.py`. With #59551 merged, the directory holds only the 27 long poles, so this can be proposed now.

---

## Step 4: the public API

Details in [`Fabric-MetalEnv-Uplift-Step4.md`](Fabric-MetalEnv-Uplift-Step4.md). 37 lookups remained after PR1-PR7; #59551 removed 10. The 27 left are long poles, all investigated and rated there on blast radius, Python impact, external impact, and whether `MetalEnv` bootstrapping makes them go away:

- The three with the widest blast radius (config getters, program-scoped connection helpers, `FabricMuxConfig`) are reached by ordinary ttnn ops and dispatch, and app migration doesn't fix them. They need API changes that break ttnn Python bindings.
- The proposals would remove 17 more, leaving about 10: the Blaze-owned set (6, needs the Blaze owners), `SetFabricConfig` (1, goes with legacy deletion) and `FabricSwitchManager` (3, or 0 if moved into its only test).
- The cheapest next step is repo-local: move the dispatch relay mux's remaining bare lookups onto its own control plane. That makes dispatch safe with a second context before any API change.

An external-facing summary of the long poles exists as a Cursor canvas for the ttnn and Blaze teams.
