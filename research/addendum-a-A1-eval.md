<!--
SUMMARY: Evaluation of Addendum A.1 (SetFabricConfig cost surface) applicability to nsexton/0-racecondition-hunt branch
KEYWORDS: A.1, SetFabricConfig, cost surface, fabric init, racecondition-hunt, batch-t3k, PAUSE, DRAIN, auto_enable_fabric
SOURCE: Code inspection of worktrees racecondition-main and nsexton/0-batch-t3k-ttnn-unit
SCOPE: A.1 bullet-by-bullet evaluation with file:line citations
USE WHEN: Assessing whether racecondition-hunt inherits, worsens, or is orthogonal to the SetFabricConfig cost surface
-->

# Addendum A.1 Evaluation: SetFabricConfig Cost Surface on `nsexton/0-racecondition-hunt`

## Summary

**A.1 partially applies.** The racecondition-hunt branch inherits the same core cost surface as `main` (SetFabricConfig on every SetUp/TearDown, MeshDevice::create triggering full firmware init). However, it does **not** contain the batch-t3k-specific patterns (PAUSE/DRAIN drain cycle, `mesh_fixture_open`/`mesh_fixture_close`, `auto_enable_fabric` trait). It **significantly amplifies** the cost surface by adding ~49 new GAP regression tests that each independently call SetFabricConfig + MeshDevice::create (136 total SetFabricConfig calls across GAP tests alone).

---

## A.1 Bullet-by-Bullet Analysis

### Bullet 1: "SetFabricConfig called on every DoSetUpTestSuite and every mesh_fixture_open (when fabric_config != DISABLED)"

**Partially applies.**

- **DoSetUpTestSuite**: YES, applies identically.
  - `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:121`: `tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode, num_routing_planes, fabric_tensix_config, fabric_udm_mode);`
  - Called from every derived fixture's `SetUpTestSuite()` — e.g., `Fabric1DFixture::SetUpTestSuite()` at line 247, `Fabric2DFixture::SetUpTestSuite()` at line 302.

- **MeshDeviceFixtureBase::SetUp()**: YES, per-test SetFabricConfig.
  - `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:168`: `tt_fabric::SetFabricConfig(...)` in `MeshDeviceFixtureBase::SetUp()` — called on every test when `fabric_config != DISABLED`.

- **mesh_fixture_open**: NO — does NOT exist on this branch. That helper is batch-t3k-only.
  - Confirmed: `grep -rn "mesh_fixture_open"` returns zero results on racecondition-main.
  - On batch-t3k: `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:96` defines `mesh_fixture_open()`.

**Verdict**: Same cost, but through different code paths. The per-test `MeshDeviceFixtureBase::SetUp()` pattern is the racecondition-hunt equivalent of `mesh_fixture_open`.

### Bullet 2: "SetFabricConfig(DISABLED) called on every mesh_fixture_close and exception paths"

**Partially applies.**

- **TearDown (equivalent of mesh_fixture_close)**: YES.
  - `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:319`: `tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);` in `MeshDeviceFixtureBase::TearDown()`.
  - `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:181`: `tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);` in `BaseFabricFixture::DoTearDownTestSuite()`.

- **Exception paths**: YES, racecondition-hunt ADDS exception-path cleanup that main does not have.
  - `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:200`: `tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);` in catch block for FIX BC degraded-cluster exceptions.
  - `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:391`: Same pattern in `MeshDeviceFixture4x8DispatchAgnostic::SetUp()` catch block.

- **mesh_fixture_close**: NO — does not exist. Batch-t3k-only.

**Verdict**: Same cost surface, plus **additional** SetFabricConfig(DISABLED) calls on exception paths (new to this branch, not on main).

### Bullet 3: "MeshDevice::create triggers initialize_fabric_routing_firmware_on_cluster"

**YES, applies identically.**

- `tt_metal/distributed/mesh_device.cpp:491`: `ctx.device_manager()->initialize_fabric_and_dispatch_fw();` — called at the end of `MeshDevice::create()`.
- `tt_metal/distributed/mesh_device.cpp:599`: Same call in `MeshDeviceImpl::create_unit_meshes()`.
- `tt_metal/impl/device/device_manager.cpp:468-616`: `DeviceManager::initialize_fabric_and_dispatch_fw()` does the full firmware initialization.

Every `MeshDevice::create()` call triggers this — both in fixture SetUp and in GAP regression tests.

**Verdict**: Identical cost. No change from main.

### Bullet 4: "PAUSE->DRAIN->RUN drain cycle is fixture-driven (not fabric-driven) as workaround for full reinit cost"

**DOES NOT APPLY.**

- `drain_fabric_routers()` is defined **only** on the batch-t3k branch at `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:601`.
- On racecondition-hunt, `grep -rn "drain_fabric_routers\|PAUSE.*DRAIN\|drain_fabric"` returns **zero results**.
- The `FabricCommandInterface` class exists at `tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp:18-58` with PAUSE/DRAIN primitives, but it is NOT used by any fixture on this branch.

**Verdict**: Not applicable. Racecondition-hunt uses full SetFabricConfig(DISABLED) + SetFabricConfig(FABRIC_*) reinit cycles instead of the PAUSE/DRAIN shortcut. This means racecondition-hunt pays the **full reinit cost** on every test cycle, whereas batch-t3k amortizes via the drain workaround.

### Bullet 5: "T3K auto-enables FABRIC_1D even when trait says DISABLED"

**DOES NOT APPLY.**

- `auto_enable_fabric` trait: does not exist on racecondition-hunt. Zero matches for `auto_enable_fabric` or `has_auto_enable_fabric`.
- On batch-t3k: `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:393-401` defines the `has_auto_enable_fabric` type trait, and lines 776-796 implement the auto-enable logic.

**Verdict**: Not applicable. On racecondition-hunt, fixtures either explicitly set `fabric_config = FABRIC_1D/FABRIC_2D` in their `Config` struct, or leave it `DISABLED`. There is no implicit auto-enable.

---

## Question 2: Has racecondition-hunt changed any of these call sites?

YES. Changes to the SetFabricConfig call sites in fixture code:

### multi_device_fixture.hpp (diff: +130 lines vs main)

1. **SetUp exception path (NEW)**: `multi_device_fixture.hpp:199-200` — Added try/catch around `MeshDevice::create()` with `SetFabricConfig(DISABLED)` cleanup on degraded-cluster exceptions (FIX BC).
2. **TearDown fabric-broken guard (NEW)**: `multi_device_fixture.hpp:287-305` — Added FIX RX: skip `quiesce_devices()` when fabric is broken, go straight to `close()`. This changes the **teardown ordering** but still calls `SetFabricConfig(DISABLED)` at line 319.
3. **Diagnostic logging (NEW)**: Lines 125-132, 212-224, 261-278, 307-320 — Extensive `log_info`/`log_warning` around every SetFabricConfig and MeshDevice lifecycle call.
4. **Watchdog thread (NEW)**: Lines 229-250 — `test_budget_ms` option kills hung tests via SIGKILL.
5. **MeshDeviceFixture4x8DispatchAgnostic (NEW)**: Lines 334-441 — Entire class duplicates the SetUp() pattern with its own SetFabricConfig + try/catch + watchdog.

### fabric_fixture.hpp (diff: +87 lines vs main)

1. **Degraded cluster guard (NEW)**: Lines 120-163 — After `SetFabricConfig()`, filters chip IDs through control plane; sets `cluster_degraded_skip_` if topology is degraded (FIX TK/TL).
2. **Fabric2DFixture skip guard (NEW)**: Lines 303-320 — Per-test `SetUp()` override checks `is_fabric_relay_path_broken()` before running (FIX SA).

### Key observation
None of the changes **remove** SetFabricConfig calls. They **add** new call sites (exception paths, guard paths) and defensive checks. The cost surface is **preserved and extended**.

---

## Question 3: New call sites — better or worse?

**Significantly worse in aggregate, but individually justified.**

### New SetFabricConfig call sites added by racecondition-hunt

1. **49 new GAP regression test files** (`tests/tt_metal/distributed/test_gap*.cpp`), containing **136 total SetFabricConfig calls**. Each test independently:
   - Calls `SetFabricConfig(FABRIC_1D or FABRIC_2D, ...)` — full fabric init
   - Calls `MeshDevice::create()` — triggers `initialize_fabric_and_dispatch_fw()`
   - Calls `SetFabricConfig(DISABLED)` — full fabric teardown
   - Some tests cycle this 2-3 times (e.g., `test_gap14` has 4 calls, `test_gap56` has 3 calls)

2. **Exception-path SetFabricConfig(DISABLED)** in `multi_device_fixture.hpp:200,391` — 2 new call sites.

3. **FIX QW2 commit** (`257c42a96d5`): Guards `SetFabricConfig(DISABLED)` in TearDown when FABRIC_1D init threw — prevents double-disable but does not reduce the number of calls.

### Cost impact

If all 49 GAP tests run in a single binary (no suite-level sharing), each test pays:
- 1x `SetFabricConfig(FABRIC_*)` — fabric control plane + topology discovery
- 1x `MeshDevice::create()` → `initialize_fabric_and_dispatch_fw()` — full firmware flash
- 1x `SetFabricConfig(DISABLED)` — fabric teardown

This is the **full reinit cost** per test, not amortized. By contrast, batch-t3k's PAUSE/DRAIN cycle avoids the firmware reflash on each iteration.

---

## Question 4: Race-condition fixes that interact with or depend on the current costly init/teardown pattern

YES — multiple fixes are **coupled to** the init/teardown pattern:

### FIX BC (multi_device_fixture.hpp:175-209)
- **Depends on**: The fact that `SetFabricConfig()` is called before `MeshDevice::create()`. If create fails, FIX BC must clean up the global fabric state by calling `SetFabricConfig(DISABLED)`.
- **Race interaction**: Without this, a throw from `MeshDevice::create()` leaves fabric config armed but no device open — the next test's `SetFabricConfig()` call would layer on top of stale state.

### FIX RX (multi_device_fixture.hpp:280-321)
- **Depends on**: `quiesce_devices()` being the standard teardown step. FIX RX skips it when fabric is broken (saves ~72s), but the `SetFabricConfig(DISABLED)` at line 319 still runs.
- **Race interaction**: If `quiesce_devices()` hangs on a broken cluster, the teardown blocks until timeout, and the subsequent `SetFabricConfig(DISABLED)` may operate on stale firmware state.

### FIX QU (test_gap62, fabric_firmware_initializer.cpp)
- **Depends on**: `Device::configure_fabric()` resetting degraded flags during the `initialize_fabric_and_dispatch_fw()` flow. FIX QU re-asserts flags after configure, which only matters because the full reinit path is taken on every test.
- **Race interaction**: If a cheaper drain-only cycle were used (a la batch-t3k), `configure_fabric()` wouldn't be called and this bug wouldn't manifest — but the fix also wouldn't be exercised.

### FIX TK/TL (fabric_fixture.hpp:120-163)
- **Depends on**: `SetFabricConfig()` triggering topology discovery. If the cluster is degraded after a prior test's teardown, the next `SetFabricConfig()` call discovers fewer chips. FIX TK/TL handles this gracefully instead of crashing.
- **Race interaction**: This is a **direct consequence** of the full-reinit-per-suite pattern — topology rediscovery on every `DoSetUpTestSuite()` exposes stale cluster state from prior tests.

### FIX QW2 (commit 257c42a96d5)
- **Directly addresses** a race in the init/teardown pattern: if `SetFabricConfig(FABRIC_1D)` throws during SetUp, TearDown must not call `SetFabricConfig(DISABLED)` on a config that was never successfully armed.

---

## Summary Table

```
A.1 Bullet                              Applies?  Notes
────────────────────────────────────── ─────────  ──────────────────────────────────
SetFabricConfig on SetUp/DoSetUp        YES       Same pattern, different helpers
SetFabricConfig(DISABLED) on TearDown   YES       + new exception-path calls
MeshDevice::create → full FW init       YES       Identical path
PAUSE→DRAIN→RUN drain cycle             NO        batch-t3k only; not on this branch
T3K auto-enables FABRIC_1D              NO        batch-t3k only; not on this branch
```

```
Cost surface change                     Direction  Magnitude
────────────────────────────────────── ─────────  ──────────
49 new GAP tests × full reinit cycle    WORSE      136 new SetFabricConfig calls
Exception-path cleanup calls            WORSE      +2 new call sites (justified)
FIX RX skip-quiesce shortcut            BETTER     Saves ~72s per broken-fabric TearDown
No PAUSE/DRAIN amortization             NEUTRAL    Was never there; same as main
```

The racecondition-hunt branch **inherits** 3 of 5 A.1 bullets from main and **amplifies** the cost surface with 49 new test files that each pay full init/teardown cost. The PAUSE/DRAIN optimization and auto-enable trait are batch-t3k-specific and not present here.
