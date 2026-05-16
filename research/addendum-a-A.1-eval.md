# Addendum A — Recommendation A.1 Evaluation: racecondition-hunt branch

Evaluated against worktree `/workspace/group/worktrees/racecondition-main/`
(branch `nsexton/0-racecondition-hunt`).

---

## Q1: Does the A.1 problem exist in racecondition-hunt?

**Yes, fully present.** Every component of the A.1 cost surface is present:

### SetFabricConfig called on every SetUp

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:168-173`:
```cpp
if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
    tt_fabric::SetFabricConfig(
        config_.fabric_config,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        config_.fabric_tensix_config,
        config_.fabric_udm_mode);
}
```
This is in `MeshDeviceFixtureBase::SetUp()` — called for **every test case**, not per-suite.

### SetFabricConfig(DISABLED) called on every TearDown

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:317-319`:
```cpp
if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
    log_info(tt::LogMetal, "[fixture_teardown] calling SetFabricConfig(DISABLED) from base TearDown");
    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
}
```

### SetFabricConfig(DISABLED) on exception paths

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:199-200`:
```cpp
if (config_.fabric_config != tt_fabric::FabricConfig::DISABLED) {
    tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
}
```
Inside the catch block for MeshDevice::create() failures.

### MeshDevice::create triggers initialize_fabric_and_dispatch_fw

`tt_metal/distributed/mesh_device.cpp:491`:
```cpp
ctx.device_manager()->initialize_fabric_and_dispatch_fw();
```
Called unconditionally at the end of `MeshDevice::create()`. Every fixture SetUp → MeshDevice::create → full fabric init.

### PAUSE→DRAIN→RUN cycle is fixture-driven

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:307-308` (TearDown):
```cpp
mesh_device_->quiesce_devices();
```
This calls `quiesce_internal()` at `tt_metal/distributed/mesh_device.cpp:1571`, which implements the full three-pass ETH relaunch cycle (Pass 1a/1b/1c + Pass 2 handshake wait). The entire cycle runs per-test during TearDown, then the next test's SetUp rebuilds everything from scratch.

### T3K auto-enables FABRIC_1D even when trait says DISABLED

`tt_metal/impl/device/device_manager.cpp:272-288`:
```cpp
if (any_remote_devices && !is_mock) {
    auto fabric_config = ctx_.get_fabric_config();
    if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
        fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D;
        ctx_.set_fabric_config(
            fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        ...
        log_info(tt::LogMetal,
            "Enabling {} only for dispatch. ...", fabric_config);
    }
}
```
When `any_remote_devices` is true (always true on T3K), FABRIC_1D is force-enabled even if the fixture config says DISABLED. This means every MeshDevice::create on T3K pays the full fabric init cost regardless of what the test requested.

---

## Q2: Better, worse, or same as in batch-t3k-ttnn-unit?

**Worse.** The cost surface is significantly heavier in racecondition-hunt.

### What batch-t3k-ttnn-unit mitigates

The batch branch introduces `SharedMeshDeviceFixture` (`multi_device_fixture.hpp:680+`) which:
- Calls `mesh_fixture_open` / `mesh_fixture_close` in `SetUpTestSuite` / `TearDownTestSuite` — **once per suite**, not per test.
- The per-test `SetUp()` only reopens the mesh if `needs_recovery_` is true (a test crashed the fabric).
- Has `fabric_enabled()` runtime detection (`multi_device_fixture.hpp:709-723`) that catches T3K auto-enable and applies health checks even for DISABLED-config fixtures.
- Helper functions `mesh_fixture_open` and `mesh_fixture_close` (`multi_device_fixture.hpp:96-134`) centralize the SetFabricConfig / MeshDevice::create / close / SetFabricConfig(DISABLED) pattern.

### What racecondition-hunt does instead

- **No shared fixture** — every test in the suite calls `MeshDeviceFixtureBase::SetUp()` which does the full: SetFabricConfig → MeshDevice::create → initialize_fabric_and_dispatch_fw.
- **No shared mesh** — TearDown does quiesce_devices() + close() + reset() + SetFabricConfig(DISABLED) on every test.
- **Far heavier quiesce cycle** — racecondition-hunt's `quiesce_internal()` (`mesh_device.cpp:1571-1687`) implements a three-pass ETH launch (Pass 1a/1b/1c) with per-device STARTED polling, which is considerably more expensive than the baseline quiesce. This was added specifically to fix simultaneous-handshake deadlocks (FIX AE/AF).
- **More SetFabricConfig calls overall** — the 60+ `test_gap*.cpp` files each have their own SetFabricConfig calls in their custom fixtures (grep shows 40+ hits across these files alone), adding to the per-test cost surface.

**Net effect**: On T3K, each test in racecondition-hunt pays:
1. SetFabricConfig(FABRIC_*) — global state setup
2. MeshDevice::create — opens all 8 devices, triggers initialize_fabric_and_dispatch_fw
3. [test body]
4. quiesce_devices() — three-pass ETH relaunch + handshake wait
5. close() — device teardown
6. SetFabricConfig(DISABLED) — global state cleanup

In batch-t3k-ttnn-unit, steps 1-2 and 5-6 happen once per **suite** (typically 10-50+ tests), and step 4 runs only between tests as a lightweight quiesce (not the full teardown/rebuild).

---

## Q3: What would implementing A.1 concretely look like?

The recommendation would involve three changes to the racecondition-hunt branch fixture:

1. **Promote `MeshDeviceFixtureBase` to a suite-level shared pattern.** Move the SetFabricConfig + MeshDevice::create + initialize_fabric_and_dispatch_fw sequence from the per-test `SetUp()` into a static `SetUpTestSuite()`. The mesh device becomes a `static` member shared across tests in the same GTest suite. TearDown calls quiesce_devices() but does NOT close+reset the device or call SetFabricConfig(DISABLED) — that happens in `TearDownTestSuite()`.

2. **Add a recovery gate in per-test SetUp.** Track `needs_recovery_` (set when a test detects `fabric_relay_path_broken` or similar). In SetUp, if `needs_recovery_` is true, close and reopen the mesh (calling the full SetFabricConfig cycle). Otherwise, reuse the existing shared mesh.

3. **Handle the T3K auto-enable explicitly.** Add a `fabric_enabled()` helper (like the batch branch has at line 709-723) that queries `GetFabricConfig()` at runtime after device open, so health checks and recovery logic fire even when the fixture config says DISABLED but T3K auto-enabled FABRIC_1D.

The gap test files (`test_gap*.cpp`) are a special case — they intentionally exercise specific failure modes and would likely keep their per-test SetFabricConfig patterns since they're not performance-sensitive production tests.

---

## Q4: Interactions with race-condition fixes

The racecondition-hunt branch has several fixes that interact with or complicate A.1 implementation:

### Three-pass ETH launch (FIX AE/AF) — makes shared mesh MORE valuable

`tt_metal/distributed/mesh_device.cpp:1600-1672` — The three-pass quiesce (1a/1b/1c) was added to fix simultaneous ETH handshake deadlocks. This is the **most expensive quiesce cycle in the codebase**. Making the mesh shared across tests means this expensive cycle runs between tests (as quiesce_devices) but the full teardown + rebuild (SetFabricConfig + MeshDevice::create) is avoided. This makes the A.1 recommendation even more valuable in racecondition-hunt than in the batch branch.

### fabric_relay_path_broken recovery guards — need to be wired into recovery gate

`tt_metal/impl/device/device.cpp:200,414-424,711,811-874` — The branch has extensive tracking of `fabric_relay_path_broken_` state, used to skip relay reads/writes when the fabric is dead. The fixture TearDown already checks this (`multi_device_fixture.hpp:287-296`). A shared fixture would need to check these flags after each test and set `needs_recovery_ = true` when any device reports broken relay. The existing FIX RX guard at line 301-305 (skip quiesce when fabric broken) already handles the "don't waste 72s on a dead fabric" case — this translates directly to a `needs_recovery_` flag.

### FIX BC exception-safe setup — already implements half the pattern

`multi_device_fixture.hpp:175-209` — The try/catch around MeshDevice::create with GTEST_SKIP on degraded cluster exceptions is exactly what a shared fixture's `SetUpTestSuite` would need. This code can be lifted almost directly.

### Watchdog thread — needs lifecycle adjustment

`multi_device_fixture.hpp:229-249` — The per-test watchdog (kill process after budget_ms) is started in SetUp and stopped in TearDown. With a shared fixture, the watchdog must be per-test (not per-suite), so it stays in SetUp/TearDown — no conflict.

### No conflicts with gap test fixtures

The gap tests (`test_gap*.cpp`) use their own custom fixtures with dedicated SetFabricConfig calls and are not derived from `MeshDeviceFixtureBase` (they typically build their own fixture class from scratch). Implementing A.1 in the base fixture would not affect them.

### channels_not_ready / stale_base_umd flags — additional recovery triggers

`tt_metal/impl/device/device.cpp` introduces `is_fabric_channels_not_ready_for_traffic()` and `is_fabric_stale_base_umd_channels()` as additional degraded-state indicators beyond `fabric_relay_path_broken_`. The fixture already checks all three at TearDown entry (line 267-277). A shared fixture recovery gate should check all three, not just relay_broken.

---

## Summary

The A.1 cost surface is **fully present and worse** in racecondition-hunt than in batch-t3k-ttnn-unit. The batch branch already mitigates it with `SharedMeshDeviceFixture`. Implementing A.1 in racecondition-hunt is straightforward — the building blocks (exception-safe setup, recovery flags, fabric health detection) are already in the code, they just need to be wired into a suite-level shared pattern. The expensive three-pass quiesce cycle makes this optimization even more impactful than in the batch branch.
