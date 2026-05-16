# A.2.2 Evaluation: Decouple SetFabricConfig from MeshDevice::create

Branch: `nsexton/0-racecondition-hunt`
Worktree: `/workspace/group/worktrees/racecondition-main/`

---

## 1. Does the problem described in A.2.2 exist in racecondition-hunt?

**Yes, strongly.** The tight coupling between `SetFabricConfig` and `MeshDevice::create`/`close` is fully present and is in fact the *central pain point* that most of the FIX-series patches are working around.

### The coupling, in code

**Per-test fixtures** (`MeshDeviceFixtureBase` and its ~15 subclasses):
Each test's `SetUp()` calls `SetFabricConfig(FABRIC_*)` then `MeshDevice::create()`, and each `TearDown()` calls `mesh_device_->close()` then `SetFabricConfig(DISABLED)`.

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:168-170` (SetUp):
```cpp
tt_fabric::SetFabricConfig(
    config_.fabric_config,
    tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, ...);
```

`tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:313-320` (TearDown):
```cpp
mesh_device_->close();
...
tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
```

**Per-suite fixtures** (`BaseFabricFixture` and its subclasses):
Same pattern but at suite level — `DoSetUpTestSuite` calls `SetFabricConfig` + `create_unit_meshes`, `DoTearDownTestSuite` calls `close` + `SetFabricConfig(DISABLED)`.

`tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:121-123` (DoSetUpTestSuite):
```cpp
tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode, num_routing_planes, ...);
```

`tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:165-170` (DoTearDownTestSuite):
```cpp
device->close();
...
tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
```

**The underlying implementation** makes this mandatory: `set_fabric_config()` in `MetalEnvImpl` has a `TT_FATAL` that blocks setting a new non-DISABLED config over an existing non-DISABLED config.

`tt_metal/impl/context/metal_env.cpp:260-268`:
```cpp
if (this->fabric_config_ == tt_fabric::FabricConfig::DISABLED ||
    fabric_config == tt_fabric::FabricConfig::DISABLED) {
    this->fabric_config_ = fabric_config;
} else {
    TT_FATAL(this->fabric_config_ == fabric_config,
        "Tried to override previous value of fabric config: {}, with: {}", ...);
}
```

This means you **must** go through `DISABLED` before you can re-enable — there is no way to leave fabric running across an open/close cycle.

**Inside `MeshDevice::create`** (`tt_metal/distributed/mesh_device.cpp:491`), the function calls `ctx.device_manager()->initialize_fabric_and_dispatch_fw()` which in turn calls `initialize_fabric_config()` (`tt_metal/impl/device/device_manager.cpp:268`). So fabric initialization is tightly bound to device creation — not a standalone operation.

**Inside `MeshDevice::close`** → device teardown → `FabricFirmwareInitializer::teardown()` (`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:989`) calls `set_fabric_config(DISABLED)` which triggers `teardown_fabric_config()`. Fabric teardown is bound to device close.

---

## 2. Better, worse, or the same as nsexton/0-batch-t3k-ttnn-unit?

**Worse in complexity, same in structure.**

The coupling structure is identical — `SetFabricConfig` → `create` → test → `close` → `SetFabricConfig(DISABLED)` is the same forced sequence on both branches. But racecondition-hunt has accumulated **substantially more workaround code** that exists specifically *because* this coupling cannot be broken:

- **FIX QW2** (`tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp:85-292`): A boolean `fabric_config_set_` flag to guard `SetFabricConfig(DISABLED)` in TearDown when `SetFabricConfig(FABRIC_1D)` threw in SetUp. This exists solely because the coupling makes the fabric state machine impossible to reason about when exceptions occur mid-cycle.

- **FIX BC** (`multi_device_fixture.hpp:180-200`): A try/catch around `MeshDevice::create()` that calls `SetFabricConfig(DISABLED)` on exception to clean up stale global fabric state. If fabric were decoupled, the global state wouldn't need cleanup.

- **FIX RX** (`multi_device_fixture.hpp:286-306`): Skip `quiesce_devices()` when fabric is broken because the close/teardown cycle (~72 seconds of force-resets) is catastrophically expensive when it fails. This entire pattern is an artifact of fabric teardown being coupled to device close.

- **FIX TG2, FIX TW, FIX RP, FIX TV, etc.** — An entire cascade of fixes in `fabric_firmware_initializer.cpp` and `metal_env.cpp` dealing with the consequences of fabric being torn down and re-initialized on every test boundary, including ROM-postcode polling (FIX RP), heartbeat detection (FIX TV/TW), L1 partial clears (FIX TG2), and ring-sync timeouts (FIX TH3).

The batch-t3k-ttnn-unit branch dealt mostly with test batching at the Python level. The racecondition-hunt branch has gone deeper into the C++ infrastructure and has exposed the full depth of the coupling problem — making it a worse situation (more brittle, more workarounds) despite the underlying code being structurally the same.

---

## 3. What would implementing A.2.2 look like concretely?

The recommendation is to expose `EnsureInitialized(config)` and `ReleaseIfUnused()` so tests can leave fabric running across open/close cycles.

**Neither API exists today** — confirmed by grep returning zero results for `EnsureInitialized` and `ReleaseIfUnused`.

Concretely, the implementation would be:

### A. Add reference-counted fabric lifecycle to MetalEnvImpl

In `tt_metal/impl/context/metal_env_impl.hpp` / `metal_env.cpp`:

- Add `EnsureInitialized(FabricConfig config, ...)` — if fabric is already running with the requested config, increment a reference count and return immediately. If DISABLED, set the config and initialize (what `set_fabric_config` does today). If running with a *different* config, either error or teardown-and-reinit.

- Add `ReleaseIfUnused()` — decrement reference count. Only call `teardown_fabric_config()` when count reaches zero.

- Remove the `TT_FATAL` at `metal_env.cpp:265` that prevents re-setting a non-DISABLED config to the same value.

### B. Remove SetFabricConfig from test fixture SetUp/TearDown

In `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp`:

- `SetUp()`: Replace `SetFabricConfig(...)` with `EnsureInitialized(...)`.
- `TearDown()`: Replace `SetFabricConfig(DISABLED)` with `ReleaseIfUnused()`.

Same change in `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp` `DoSetUpTestSuite` / `DoTearDownTestSuite`.

### C. Decouple MeshDevice::create from fabric init

In `tt_metal/distributed/mesh_device.cpp:491`:

- Currently `MeshDeviceImpl::create()` unconditionally calls `ctx.device_manager()->initialize_fabric_and_dispatch_fw()`.
- With the decoupled API, this call would check if fabric is already initialized (via the ref-count) and skip re-initialization.

### D. Decouple device close from fabric teardown

In `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:989`:

- Currently `teardown()` calls `set_fabric_config(DISABLED)`.
- With the decoupled API, device close would call `ReleaseIfUnused()` instead, only tearing down fabric when the last user releases.

### E. Retire FIX QW2, FIX BC, and parts of FIX RX

These workarounds exist solely to handle the state-machine hazards of mandatory fabric teardown. With ref-counted fabric lifecycle, the exception-safety and broken-fabric guards simplify dramatically.

---

## 4. Interactions with race-condition fixes already in this branch

The race-condition fixes on this branch interact with A.2.2 in several ways:

### Fixes that A.2.2 would **partially obviate**

- **FIX QW2** (guard `SetFabricConfig(DISABLED)` after SetUp exception): If fabric isn't torn down per-test, the exception-safety problem disappears — `ReleaseIfUnused()` is always safe to call.

- **FIX BC** (try/catch around `MeshDevice::create` to cleanup fabric state): With decoupled lifecycle, `MeshDevice::create` failure doesn't leave fabric in a half-initialized state because fabric was already initialized before create was called.

- **FIX RX** (skip quiesce when fabric broken): The 72-second teardown penalty only exists because fabric is torn down on every close. If fabric persists across cycles, a broken test doesn't trigger a full teardown — it just releases its mesh device, and fabric cleanup happens when the suite ends.

### Fixes that A.2.2 would **interact with but not replace**

- **FIX RP, FIX TV, FIX TW** (ROM-postcode polling, heartbeat detection): These fix genuine race conditions in the fabric initialization *sequence itself*. Even with ref-counted fabric, the *first* initialization still needs these fixes. But they'd only run once per suite instead of once per test, dramatically reducing their failure surface.

- **FIX TG2** (partial L1 clear for base-UMD channels): This is a teardown-correctness fix. It remains necessary when fabric is eventually torn down, but it would fire once at suite end instead of N times (once per test).

- **FIX TH3** (120s ring-sync timeout + break infinite warm-up loop): Ring-sync happens during fabric init. With persistent fabric, this only runs once, but the fix is still needed for that one time.

### Fixes that A.2.2 **must not break**

- **`teardown_fabric_config()` force-resets** (`metal_env.cpp:321-500`): This code has extensive race-condition logic (timeout detection, force-reset of stuck ETH channels, ERISC0 restore). The `ReleaseIfUnused()` path must still call this exact code when the ref-count reaches zero. A.2.2 must *not* introduce a different teardown path.

- **Channel trimming capture export** (`metal_env.cpp:253-258`): Currently exported before `teardown_fabric_config()`. Must still happen at the final `ReleaseIfUnused()` call.

- **`force_reinit_` flag** (`metal_env.cpp:249`): `set_fabric_config` sets `force_reinit_ = true`. The ref-counted `EnsureInitialized` must respect this — if a prior teardown set force_reinit, the next EnsureInitialized with the same config must still reinitialize.

### Key risk: stale fabric state leaking across tests

The fundamental risk of A.2.2 is that a test corrupts fabric state (e.g., leaves ERISCs in a bad state, corrupts L1 routing tables) and this corruption leaks into the next test because fabric wasn't torn down. The branch's existing degraded-cluster detection (`is_fabric_relay_path_broken()`, `is_fabric_channels_not_ready_for_traffic()`, `is_fabric_stale_base_umd_channels()`) would need to be checked between tests, with a forced teardown if any flag is set. This is essentially the `ReleaseIfUnused` → "force release" escape hatch.

---

## Summary

| Question | Answer |
|----------|--------|
| A.2.2 problem exists? | **Yes** — `SetFabricConfig` ↔ `MeshDevice::create`/`close` are tightly coupled at every level |
| Compared to batch-t3k-ttnn-unit? | **Worse** — same structural coupling but more workarounds (FIX QW2, BC, RX, etc.) making the code more fragile |
| Concrete implementation? | Ref-counted `EnsureInitialized`/`ReleaseIfUnused` on `MetalEnvImpl`, decouple from `MeshDevice::create`/`close`, retire FIX QW2/BC/RX |
| Interactions with existing fixes? | Would partially obviate 3 FIXes, reduce failure surface of ~5 others, but must preserve teardown correctness and not leak stale fabric state |
