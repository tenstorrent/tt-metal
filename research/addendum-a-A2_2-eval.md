<!--
SUMMARY: Evaluation of Addendum A recommendation A.2.2 (decouple SetFabricConfig from MeshDevice::create) against branch nsexton/0-racecondition-hunt
KEYWORDS: A.2.2, SetFabricConfig, MeshDevice, fabric, decoupling, EnsureInitialized, ReleaseIfUnused, racecondition-hunt
SOURCE: Code analysis of /workspace/group/worktrees/racecondition-main/
SCOPE: Call chain analysis, test pattern inventory, applicability assessment
USE WHEN: Evaluating whether to implement A.2.2 fabric decoupling on the racecondition-hunt branch
-->

# Addendum A.2.2 Evaluation: Decouple SetFabricConfig from MeshDevice::create

Branch: `nsexton/0-racecondition-hunt`
Worktree: `/workspace/group/worktrees/racecondition-main/`

---

## 1. Call Chain: MeshDevice::create -> Fabric Init

The chain has two distinct paths depending on whether fabric was pre-configured by the caller.

### Path A: Caller pre-sets fabric (typical test fixture path)

```
tt_fabric::SetFabricConfig(FABRIC_1D/2D, ...)                    # fabric.cpp:449
  -> MetalContext::set_fabric_config(...)                         # metal_context.cpp:592
    -> MetalEnvImpl::set_fabric_config(...)                       # metal_env.cpp:240
      -> sets fabric_config_, fabric_reliability_mode_
      -> initializes control plane + routing tables
```

Then separately:

```
MeshDevice::create(config, ...)                                   # mesh_device.cpp:355
  -> MeshDeviceImpl::create(DEFAULT_CONTEXT_ID, config, ...)      # mesh_device.cpp:392
    -> MetalContext::create_unit_meshes(...)                       # metal_env.cpp:827
      -> DeviceManager::initialize(device_ids, ..., init_fabric=true)  # device_manager.cpp:191
        -> env_impl_.initialize_fabric_config()                   # device_manager.cpp:268 (no-op if already done)
        -> ctx_.set_fabric_config(fabric_config, STRICT_...)      # device_manager.cpp:291 (re-arm for reliability mode)
        -> ctx_.initialize_fabric_tensix_datamover_config()       # device_manager.cpp:303
        -> init_firmware_on_active_devices()                      # device_manager.cpp:605
          -> if (initialize_fabric_and_dispatch_fw_)              # device_manager.cpp:616
            -> initialize_fabric_and_dispatch_fw()                # device_manager.cpp:468
              -> FabricFirmwareInitializer::init(active_devices)  # device_manager.cpp:499
              -> DispatchKernelInitializer::init(dispatch_devices) # device_manager.cpp:535
```

### Path B: Caller does NOT set fabric (legacy auto-enable path)

When `fabric_config == DISABLED` but remote devices exist:

```
DeviceManager::initialize(...)                                    # device_manager.cpp:274
  -> if (any_remote_devices && !is_mock)
    -> ctx_.set_fabric_config(FABRIC_1D, STRICT_..., 1)          # device_manager.cpp:278
    -> ctx_.initialize_fabric_config()                            # device_manager.cpp:283
```

This auto-enables FABRIC_1D for dispatch — the comment at `device_manager.cpp:277` explicitly notes:
> "Externally, we should decide how/where to have SetFabricConfig on the correct MetalEnv"

### Teardown path (close -> DISABLED)

```
MeshDevice::close()                                               # mesh_device.cpp:1867
  -> MeshDeviceImpl::close_impl()                                 # mesh_device.cpp:917
    -> scoped_devices_.reset()                                    # mesh_device.cpp:991
      -> ScopedDevices dtor -> DeviceManager::close_devices()     # device_manager.cpp:~720
        -> FabricFirmwareInitializer::teardown(init_done)         # device_manager.cpp:754
        -> FabricFirmwareInitializer::post_teardown()             # device_manager.cpp:779
          -> set_fabric_config(DISABLED)                          # fabric_firmware_initializer.cpp:994
            -> MetalEnvImpl::teardown_fabric_config()             # metal_env.cpp:321
              -> sends TERMINATE, polls EDMStatus, force-resets on timeout
```

Key observation: `post_teardown()` at `fabric_firmware_initializer.cpp:994` **unconditionally** calls `set_fabric_config(DISABLED)`. This means every `MeshDevice::close()` resets the global fabric state to DISABLED, requiring any subsequent `MeshDevice::create()` to be preceded by a fresh `SetFabricConfig()` call.

---

## 2. New Code Paths with MeshDevice Open/Close Cycling

The branch introduces multiple explicit close/reopen patterns:

### test_gap14_teardown_reopen_eth_ordering.cpp (NEW)

At lines 279-301, a 5-cycle close/reopen loop:
```
mesh_device_->close();
mesh_device_.reset();
tt_fabric::SetFabricConfig(FABRIC_2D, STRICT_...);   // line 285
mesh_device_ = MeshDevice::create(...);                // line 290
```
Comment at line 263: "close()/post_teardown() resets it to DISABLED" — the test **must** re-call SetFabricConfig every cycle. This is the exact pain point A.2.2 targets.

### test_async_teardown_race.cpp (NEW)

**Scenario D** (lines 419-430): close + SetFabricConfig + MeshDevice::create
**Scenario E** (lines 477+): N back-to-back FABRIC_2D cycles, each requiring SetFabricConfig before create.

22 total `SetFabricConfig` calls in this single file.

### test_gap24_rapid_close_reopen_cycling.py (NEW, Python)

Python subprocess runs 10 cycles of `ttnn.open_mesh_device()` / `ttnn.close_mesh_device()`. Each `open_mesh_device` internally triggers the DeviceManager path that auto-enables fabric (Path B above).

### multi_device_fixture.hpp (MODIFIED)

The fixture's `SetUp()` calls `SetFabricConfig(config)` at line 168, and `TearDown()` calls `SetFabricConfig(DISABLED)` at line 319. Every GTest that uses this fixture pays the full fabric init/teardown cost per test case.

### test_ccl_multi_cq_multi_device.cpp (MODIFIED — FIX QW2)

FIX QW2 at lines 85-98: introduces a `fabric_1d_initialized_` boolean to track whether `SetFabricConfig(FABRIC_1D)` succeeded, because calling `SetFabricConfig(DISABLED)` in TearDown when init threw causes a cascading timeout failure. This is a symptom of tight coupling — the test must manually track fabric state because the API doesn't.

---

## 3. Race-Condition Fixes: Fabric Stay-Alive vs Tear-Down Dependency

### Fixes that REQUIRE fabric teardown on mesh close

- **GAP-78 (teardown ordering)** at `metal_env.cpp:139-158`: `check_use_count_zero()` throws if dispatch threads are still active during `teardown_fabric_config()`. The fix enforces strict ordering: all dispatch must drain BEFORE fabric teardown begins. This fix is **orthogonal** to A.2.2 — it governs the ordering within a single teardown, not whether teardown should happen at all.

- **FIX AB / post_teardown** at `fabric_firmware_initializer.cpp:988-1012`: `post_teardown()` propagates `teardown_timed_out_chips_` to Device objects for hard-reset at process exit. This cleanup logic is essential per-close.

### Fixes that would BENEFIT from fabric staying initialized

- **FIX QW2** (`test_ccl_multi_cq_multi_device.cpp:85-98`): The entire fix exists because SetFabricConfig(DISABLED) in TearDown can fail when init threw. With EnsureInitialized/ReleaseIfUnused, the test wouldn't need to track `fabric_1d_initialized_` manually.

- **FIX BC** (`multi_device_fixture.hpp:175-209`): The try/catch around `MeshDevice::create()` must call `SetFabricConfig(DISABLED)` on failure to undo the `SetFabricConfig(FABRIC_*)` that preceded it. With refcounted fabric, this cleanup would be automatic.

### Fixes that are NEUTRAL

- All the ERISC firmware fixes (FIX AE/AF/AC/AD/AT/RP etc.) operate within a single fabric session. They don't depend on whether fabric stays initialized across mesh boundaries.

**Verdict**: No fix in this branch requires fabric to be torn down on mesh close as a correctness property. The teardown happens because of the current API design, not because any fix depends on it.

---

## 4. What EnsureInitialized/ReleaseIfUnused Would Look Like

### Where they would live

```
tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp   (public API, next to SetFabricConfig at line 159)
tt_metal/fabric/fabric.cpp                                 (implementation, next to SetFabricConfig at line 449)
```

Internally they would operate on:
```
tt_metal/impl/context/metal_env_impl.hpp                   (fabric_config_, control_plane_, use_count_)
```

### What they would do

```cpp
// fabric.hpp — new public API
namespace tt::tt_fabric {

// Increment fabric refcount. If fabric is not initialized with the given config,
// initialize it. If already initialized with the same config, no-op (just bump refcount).
// If initialized with a DIFFERENT config, TT_FATAL.
void EnsureInitialized(
    FabricConfig config,
    FabricReliabilityMode mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
    std::optional<uint8_t> num_routing_planes = std::nullopt);

// Decrement fabric refcount. When refcount hits 0, tear down fabric
// (equivalent to current SetFabricConfig(DISABLED)).
void ReleaseIfUnused();

}  // namespace tt::tt_fabric
```

### Internal state changes (MetalEnvImpl)

The existing `use_count_` at `metal_env_impl.hpp:118` is already an `atomic<int>` used for GAP-78 ordering checks. A fabric-specific refcount would be a new field:

```cpp
// metal_env_impl.hpp — new member
std::atomic<int> fabric_ref_count_{0};  // tracks EnsureInitialized/ReleaseIfUnused
```

### How current patterns would change

**Before** (test_gap14, line 279):
```cpp
mesh_device_->close();           // triggers post_teardown -> SetFabricConfig(DISABLED)
mesh_device_.reset();
SetFabricConfig(FABRIC_2D, ...); // re-init from scratch
mesh_device_ = MeshDevice::create(...);
```

**After**:
```cpp
// SetUp (once per test):
tt_fabric::EnsureInitialized(FABRIC_2D, STRICT_...);

// Each cycle:
mesh_device_->close();           // post_teardown calls ReleaseIfUnused() instead of SetFabricConfig(DISABLED)
                                  // refcount > 0, so fabric stays alive
mesh_device_.reset();
mesh_device_ = MeshDevice::create(...);  // fabric already initialized, skips full init

// TearDown (once):
tt_fabric::ReleaseIfUnused();    // refcount -> 0, now tears down
```

### Key implementation detail

`FabricFirmwareInitializer::post_teardown()` at `fabric_firmware_initializer.cpp:994` currently calls `set_fabric_config(DISABLED)` unconditionally. With A.2.2, it would call `ReleaseIfUnused()` instead, which only tears down when `fabric_ref_count_` reaches 0.

The firmware-level ERISC teardown (TERMINATE signals, EDMStatus polling) would still happen per close — that's the `FabricFirmwareInitializer::teardown()` at `device_manager.cpp:754`, which is separate from `post_teardown()`. The distinction is:
- `teardown()` = stop ERISC firmware on these specific devices (per-session)
- `post_teardown()` = reset global fabric state (control plane, routing tables) — this is what EnsureInitialized would keep alive

---

## 5. Distributed Tests That Would Benefit

All new test files in `tests/tt_metal/distributed/` that call `SetFabricConfig` before `MeshDevice::create`:

```
test_gap14_teardown_reopen_eth_ordering.cpp  — 5-cycle close/reopen, SetFabricConfig per cycle
test_async_teardown_race.cpp                 — 22 SetFabricConfig calls across Scenarios A-E
test_gap52_fixpg_phase25_relay_retry.cpp     — SetFabricConfig + create in test body
test_gap58_fixqc_nonmmio_reset_cores_skip.cpp — SetFabricConfig + create in test body
test_gap8_init_router_sync_dead_relay.cpp    — SetFabricConfig + create in test body
test_gap67_fixtf_2d_fabric_header_args_guard.cpp — SetFabricConfig + create
test_gap60_fixpypz_phase25_topology_timeout.cpp  — SetFabricConfig + create
test_gap32_fixaz_l1barrier_skip_no_prior_fabric.cpp — SetFabricConfig + create
test_gap41_fixnt_ethcoord_preserved_aq_skip.cpp     — SetFabricConfig + create
test_gap5_relay_broken_teardown.cpp                 — SetFabricConfig + create
```

The `multi_device_fixture.hpp` fixture (used by ~50+ tests) does SetFabricConfig in SetUp and SetFabricConfig(DISABLED) in TearDown for every single test. With A.2.2:
- A test suite could call `EnsureInitialized` once in `SetUpTestSuite` and `ReleaseIfUnused` once in `TearDownTestSuite`
- Individual tests would skip the 5-15s fabric init/teardown per test case
- This is the biggest single performance win available from A.2.2

---

## Summary Assessment

| Question | Answer |
|---|---|
| Does A.2.2 apply? | **Yes, strongly.** |
| Does any fix depend on fabric being torn down per mesh close? | **No.** All teardown-ordering fixes (GAP-78) are about ordering within a teardown, not about requiring teardown. |
| Does any fix depend on fabric staying alive across close? | **Not explicitly**, but FIX QW2 and FIX BC are workarounds for the tight coupling that A.2.2 would eliminate. |
| Would it reduce test time? | **Significantly.** Each SetFabricConfig(DISABLED) + SetFabricConfig(FABRIC_2D) cycle costs ~5-15s. The 5-cycle test in gap14 spends ~50-75s just on fabric cycling. |
| Implementation risk? | **Moderate.** The `post_teardown()` -> `set_fabric_config(DISABLED)` path is a load-bearing invariant that many fixes depend on for cleanup. Changing it to refcounted release requires careful audit of all 22+ SetFabricConfig call sites. |
| Should it be done on this branch? | **No — scope creep.** This branch is fixing race conditions, not restructuring the fabric API. A.2.2 should be a follow-up PR after racecondition-hunt merges, using the test patterns established here as the first consumers. |
