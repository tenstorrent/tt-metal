<!--
SUMMARY: Addendum A recommendations evaluated for applicability to nsexton/0-racecondition-hunt vs nsexton/0-batch-t3k-ttnn-unit
KEYWORDS: tt-fabric, SetFabricConfig, fabric-init, racecondition-hunt, fixture-complexity, quiesce, idempotent, documentation
SOURCE: Serial Opus subagent evaluation of worktree /workspace/group/worktrees/racecondition-main/
SCOPE: All 7 Addendum A items (A.1 cost surface + A.2.1–A.2.7 recommendations) evaluated with file:line evidence
USE WHEN: Deciding which Addendum A changes to implement in nsexton/0-racecondition-hunt, and in what priority order
-->

# Addendum A Applicability Report — nsexton/0-racecondition-hunt

*2026-05-08*

---

## Executive Summary

Of the 7 Addendum A items, *one strongly applies* (A.2.7 documentation), *one applies but should be deferred to a follow-up PR* (A.2.2 decoupling), and *five do not apply or are blocked by architecture*.

The core reason: racecondition-hunt uses a per-test open/close model with explicit `FABRIC_2D` configs, not the shared-fixture model that batch-t3k uses. Most Addendum A optimizations target the shared-fixture pattern.

```
Priority  Item    Verdict   Effort   Notes
────────  ──────  ────────  ───────  ──────────────────────────────────────────────────────
1         A.2.7   APPLY     Medium   Documentation — 8 new subsections in TT-Fabric-Architecture.md
2         A.2.2   DEFER     High     API restructuring (EnsureInitialized/ReleaseIfUnused) — follow-up PR
3         A.1     PARTIAL   N/A      Cost surface diagnosis only — 3/5 bullets inherited, 2/5 batch-t3k-only
4         A.2.1   DEFER     Low      Idempotency guard — correct but zero impact (never fires today)
5         A.2.3   N/A       —        PAUSE-DRAIN-RUN Quiesce() — branch uses TERMINATE+RELAUNCH instead
6         A.2.6   N/A       —        Reconfigure() — architecturally blocked by TT_FATAL guard
7         A.2.4   N/A       —        Batched heartbeat reads — no per-test telemetry on this branch
8         A.2.5   N/A       —        ETH-dispatch auto-enable opt-out — all fixtures explicitly set FABRIC_2D
```

---

## Per-Item Findings

### A.1 — SetFabricConfig Cost Surface

*Verdict*: *PARTIAL* — Racecondition-hunt inherits 3 of 5 cost-surface bullets from main and amplifies them with 49 new GAP tests, but does NOT have the batch-t3k-only patterns (PAUSE/DRAIN drain cycle, auto_enable_fabric trait).

*Evidence*:
- `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:168` — `SetFabricConfig(...)` in every `MeshDeviceFixtureBase::SetUp()`
- `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:319` — `SetFabricConfig(DISABLED)` in every `MeshDeviceFixtureBase::TearDown()`
- `tt_metal/distributed/mesh_device.cpp:491` — `MeshDevice::create()` triggers full `initialize_fabric_and_dispatch_fw()`
- 49 new `test_gap*.cpp` files contain 136 total `SetFabricConfig` calls — each paying full init/teardown cost

*What it looks like in racecondition-hunt*: The cost surface is inherited and extended, not introduced. Every GTest pays full `SetFabricConfig(FABRIC_2D)` -> `MeshDevice::create()` -> `MeshDevice::close()` -> `SetFabricConfig(DISABLED)` per test. The branch adds +2 exception-path `SetFabricConfig(DISABLED)` calls (`multi_device_fixture.hpp:200,391`) and FIX RX's skip-quiesce shortcut that saves ~72s on broken-fabric teardown.

*Interaction with race-condition fixes*: FIX BC (line 175-209), FIX RX (line 280-321), FIX QW2, FIX QU, and FIX TK/TL all *depend on* the current init/teardown pattern. They are workarounds for, not solutions to, the cost surface. Changing A.1 would require re-evaluating all of them.

---

### A.2.1 — Make SetFabricConfig Idempotent/Lazy

*Verdict*: *DEFER* — Correct optimization, straightforward to implement (8-line guard in `MetalEnvImpl::set_fabric_config`), but has *zero measurable impact* today because the call pattern never triggers it.

*Evidence*:
- `tt_metal/impl/context/metal_env.cpp:248` — `force_reinit_ = true` is unconditional (no idempotency)
- `tt_metal/impl/context/metal_env_impl.hpp:120-128` — All 7 config fields already stored as member variables (cache already exists)
- Call pattern is always `DISABLED -> FABRIC_X -> DISABLED -> FABRIC_X` — the DISABLED call always intervenes, so same-config back-to-back calls *never occur*
- `fabric_firmware_initializer.cpp:994` — `post_teardown()` unconditionally calls `set_fabric_config(DISABLED)` on every `MeshDevice::close()`

*What it looks like in racecondition-hunt*: An 8-line comparison guard at the top of `MetalEnvImpl::set_fabric_config()` (metal_env.cpp:240) comparing all 7 params against stored state. Would be dead code today.

*Interaction with race-condition fixes*: None. FIX AY (test_gap31) and FIX M/FIX SB2 implicitly depend on full reinit, but since same-config calls never happen, the guard would never fire and cannot break anything. The guard becomes valuable only if a future "warm restart" path (TODO at `fabric_switch_manager.cpp:40`) skips the DISABLED intermediate.

---

### A.2.2 — Decouple SetFabricConfig from MeshDevice::create (EnsureInitialized/ReleaseIfUnused)

*Verdict*: *DEFER* (strongly applicable, but follow-up PR) — This is the highest-impact recommendation for the codebase long-term, but implementing it on the race-condition branch is scope creep.

*Evidence*:
- `test_gap14_teardown_reopen_eth_ordering.cpp:279-301` — 5-cycle close/reopen loop, each requiring `SetFabricConfig` before `MeshDevice::create`. Comment at line 263: "close()/post_teardown() resets it to DISABLED"
- `test_async_teardown_race.cpp` — 22 `SetFabricConfig` calls across 5 scenarios
- `test_ccl_multi_cq_multi_device.cpp:85-98` (FIX QW2) — manually tracks `fabric_1d_initialized_` boolean because the API doesn't manage fabric lifecycle
- `fabric_firmware_initializer.cpp:994` — `post_teardown()` unconditionally calls `set_fabric_config(DISABLED)`, the single root cause of the coupling

*What it looks like in racecondition-hunt*: A new `tt_fabric::EnsureInitialized()` / `tt_fabric::ReleaseIfUnused()` API pair in `fabric.hpp:159+`, backed by a `fabric_ref_count_` atomic in `MetalEnvImpl`. `post_teardown()` would call `ReleaseIfUnused()` instead of `set_fabric_config(DISABLED)`, keeping fabric alive when refcount > 0. The multi_device_fixture would call `EnsureInitialized()` once in `SetUpTestSuite` instead of per-test.

*Interaction with race-condition fixes*: FIX QW2 and FIX BC would become unnecessary (they exist to work around the tight coupling). GAP-78 teardown ordering (metal_env.cpp:145) is orthogonal — it governs ordering *within* a teardown, not *whether* teardown happens. No fix *requires* fabric teardown on mesh close as a correctness property.

---

### A.2.3 — tt_fabric::Quiesce() (PAUSE-DRAIN-RUN API)

*Verdict*: *N/A* — The branch uses TERMINATE+RELAUNCH quiesce, not PAUSE-DRAIN-RUN. These are fundamentally different mechanisms.

*Evidence*:
- `tests/tt_metal/tt_fabric/common/fabric_command_interface.cpp:40-46` — `pause_routers()` and `resume_routers()` exist but have only 1 test consumer (`test_fabric_traffic_generator_kernel.cpp:164-180`)
- No `drain_routers()` method exists — the full PAUSE->DRAIN->RUN cycle is never composed anywhere on this branch
- `tt_metal/impl/device/device.cpp:727-1474` — `quiesce_and_restart_fabric_workers()` is ~750 lines implementing TERMINATE -> poll TERMINATED -> reconfigure -> relaunch
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:2087-2118` — Firmware supports DRAIN but no host code drives it

*What it looks like in racecondition-hunt*: Would replace `FabricCommandInterface` in 1 test file. Would NOT replace `quiesce_and_restart_fabric_workers()` which is a heavier mechanism for a different purpose.

*Interaction with race-condition fixes*: None. The two mechanisms (PAUSE/DRAIN/RUN vs TERMINATE/RELAUNCH) are independent. All race-condition fixes operate on the TERMINATE+RELAUNCH path.

---

### A.2.4 — Batched Fabric Heartbeat Reads

*Verdict*: *N/A* — Racecondition-hunt has no per-test heartbeat capture. This is a batch-t3k-only optimization.

*Evidence*:
- `capture_fabric_heartbeats` returns zero results on racecondition-hunt (it's batch-t3k at `multi_device_fixture.hpp:510-539`)
- FIX TV (`risc_firmware_initializer.cpp:237-333`) and FIX AR (`risc_firmware_initializer.cpp:534-589`) do per-core heartbeat polls during init/teardown, but *require per-core granularity* — a batched read would destroy the ability to distinguish "core 3 on chip 0 is still in ROM boot" from "all cores healthy"

*What it looks like in racecondition-hunt*: Nothing to change. The heartbeat reads that exist are part of firmware init, not per-test telemetry.

*Interaction with race-condition fixes*: FIX TV and FIX AR depend on per-core poll granularity. A batched API would break them.

---

### A.2.5 — Explicit ETH-Dispatch Fabric Auto-Enable Opt-Out

*Verdict*: *N/A* — All new fixtures explicitly set `FABRIC_2D`. No MMIO-only fixture pays unnecessary fabric init cost.

*Evidence*:
- `device_manager.cpp:272-275` — auto-enable FABRIC_1D path exists (unchanged from main) but is never the relevant path
- All 50+ `test_gap*.cpp` files use `.fabric_config = tt_fabric::FabricConfig::FABRIC_2D`
- FIX QW2 handles the auto-enable failure case but is a crash fix, not a performance optimization

*What it looks like in racecondition-hunt*: Nothing to change.

*Interaction with race-condition fixes*: FIX QW2 is the only fix touching auto-enable behavior, and the opt-out flag wouldn't help — the test still needs fabric.

---

### A.2.6 — tt_fabric::Reconfigure(new_config) for Minimal-Diff Reroute

*Verdict*: *N/A* — Architecturally blocked by `TT_FATAL` guard. No test benefits.

*Evidence*:
- `tt_metal/impl/context/metal_env.cpp:265-269` — Hard `TT_FATAL` if config changes from non-DISABLED to a *different* non-DISABLED value. Only legal transitions are `DISABLED <-> X`.
- All 70+ GAP tests use `FABRIC_2D` + `STRICT_SYSTEM_HEALTH_SETUP_MODE` exclusively — no parameter-only transitions exist
- `tt_metal/fabric/fabric_types.hpp:87` — `DYNAMIC_RECONFIGURATION_SETUP_MODE` is an unimplemented placeholder enum
- FIX RZ2, FIX QU, FIX M all critically depend on the full teardown/reinit cycle

*What it looks like in racecondition-hunt*: Cannot be implemented without removing the TT_FATAL guard. Even then, no test would benefit.

*Interaction with race-condition fixes*: Would *break* FIX RZ2 (`fabric_firmware_initializer.cpp:282-297`), FIX QU (`fabric_firmware_initializer.cpp:230-261`), and FIX M (`device.cpp:664`) — all depend on full `configure_fabric()` running during reinit.

---

### A.2.7 — Document the Fabric Init Contract

*Verdict*: *APPLY* — The branch introduces a dramatically more complex lifecycle (6-phase quiesce, 3 channel classifications, degraded-mode operation, cross-session state persistence) that exists only in inline code comments. The tech reports have ~14 lines total on init/teardown.

*Evidence*:
- `tech_reports/TT-Fabric/TT-Fabric-Architecture.md:921-932` — The ONLY init/teardown documentation (6 lines, added by this branch)
- `tech_reports/EthernetMultichip/BasicEthernetGuide.md:537-550` — L1 overwrite ordering subsection (8 lines, added by this branch)
- `device.cpp:727-3400+` — ~2700 lines of `quiesce_and_restart_fabric_workers()` with 6 phases, none documented outside inline comments
- `fabric_firmware_initializer.cpp:1041-1200+` — `terminate_stale_erisc_routers()` with channel classification taxonomy undocumented
- `metal_env.cpp:100-175` — GAP-78 teardown ordering invariant enforced by `TT_THROW` but not in any spec

*What it looks like in racecondition-hunt*: A new "Section 8 — Fabric Lifecycle and Host Control Plane Operations" in `TT-Fabric-Architecture.md` with 8 subsections:
- 8.1 Lifecycle Overview (SetFabricConfig state machine)
- 8.2 Initialization Sequence (MetalEnvImpl, FabricFirmwareInitializer, channel classification)
- 8.3 Quiesce and Restart Protocol (Phases 1, 2, 2.5, 3, 4, 5, 5b)
- 8.4 Teardown Sequence (GAP-78 ordering invariant)
- 8.5 Cross-Session State Model (L1 sentinels: 0x49706550, 0xA0A0A0A0, 0xDEADB07E)
- 8.6 Degraded-Mode Operation (pre-known-dead vs newly-dead)
- 8.7 ERISC Soft-Reset Semantics (BRISC-only, PHY preservation)
- 8.8 Cost Model (per-operation timing)

*Interaction with race-condition fixes*: Pure documentation. Zero risk of breaking any fix. Protects fixes by making their invariants explicit for future developers.

---

## Simplification Impact Assessment

The original Addendum A claimed:

> "These changes would let the test fixtures simplify dramatically: no static `Traits::always_recover()` overrides, no `drain_fabric_routers` re-implementation in tests, no PAUSE/DRAIN/RUN bookkeeping outside of tt-fabric."

Evaluated for racecondition-hunt:

- *"no static Traits::always_recover() overrides"* — *N/A*. Racecondition-hunt does not have `Traits::always_recover()`. That's a batch-t3k pattern.

- *"no drain_fabric_routers re-implementation in tests"* — *N/A*. `drain_fabric_routers()` does not exist on this branch. Zero matches.

- *"no PAUSE/DRAIN/RUN bookkeeping outside of tt-fabric"* — *Mostly N/A*. `FabricCommandInterface` has PAUSE/RUN (no DRAIN) and is used by exactly 1 test. Negligible complexity.

*What complexity WOULD actually go away with the applicable recommendations:*

- *A.2.2 (EnsureInitialized/ReleaseIfUnused)*: Would eliminate per-test `SetFabricConfig`/`SetFabricConfig(DISABLED)` dance in `MeshDeviceFixtureBase` (~50+ tests). Would make FIX QW2's manual `fabric_1d_initialized_` tracking unnecessary. Would make FIX BC's try/catch `SetFabricConfig(DISABLED)` cleanup automatic. This is the *real* simplification win.

- *A.2.7 (documentation)*: Would not simplify code, but would prevent future developers from accidentally violating the invariants that the race-condition fixes depend on — reducing the rate at which new GAP tests need to be written.

*Bottom line*: The Addendum A simplification claim is *true for batch-t3k* but *only partially true for racecondition-hunt*. The batch-t3k-specific patterns (`always_recover`, `drain_fabric_routers`, PAUSE/DRAIN/RUN bookkeeping) don't exist on this branch. The complexity that *does* exist — per-test SetFabricConfig cycling, manual fabric state tracking in FIX QW2/BC, 2700 lines of undocumented quiesce protocol — would be addressed by A.2.2 and A.2.7.

---

## Recommended Implementation Order

*1. A.2.7 — Document the fabric init contract (APPLY NOW)*
- Zero risk. Pure documentation.
- The branch already adds 14 lines of init/teardown docs — extend to the full 8-subsection spec.
- Protects the race-condition fixes from future regressions by making invariants explicit.
- Can be done in the same PR as the race-condition fixes since it doesn't change behavior.
- *Justification for going first*: Every other recommendation requires understanding the lifecycle, and there's no authoritative reference for it today.

*2. A.2.1 — Idempotency guard (DEFER to follow-up PR)*
- Low effort (8-line guard), but zero impact today.
- Becomes the natural precondition for A.2.2's "warm restart" path.
- Ship it as a tiny preparatory PR before A.2.2.
- *Justification for second*: Cheapest prerequisite for A.2.2, validates config-comparison logic before the refcount mechanism depends on it.

*3. A.2.2 — EnsureInitialized/ReleaseIfUnused (DEFER to follow-up PR)*
- Highest-impact change. Would eliminate per-test fabric cycling for ~50+ tests.
- Requires careful audit of all 22+ `SetFabricConfig` call sites and the `post_teardown()` invariant.
- Should use racecondition-hunt's test patterns as the first consumers.
- *Justification for third*: Depends on A.2.1 for config-comparison logic. Too high-risk to bundle with race-condition fixes. Standalone PR after racecondition-hunt merges.

*4-8. A.1, A.2.3, A.2.4, A.2.5, A.2.6 — NOT APPLICABLE / NO ACTION*
- A.1: Diagnosis only — the cost surface is inherited, not fixable on this branch.
- A.2.3 (Quiesce API): Branch uses TERMINATE+RELAUNCH. Different mechanism entirely.
- A.2.4 (Batched heartbeat): No per-test telemetry. Fixes require per-core granularity.
- A.2.5 (Auto-enable opt-out): All fixtures explicitly enable FABRIC_2D.
- A.2.6 (Reconfigure): Blocked by TT_FATAL guard and would break FIX RZ2/QU/M.
