# Shared-fixture contract for TTNN tests

This document captures the shared-fixture pattern used by both the C++
gtest infrastructure and the Python pytest infrastructure for TTNN tests.
Read this before adding a new test directory or a new test class — it
explains what the fixtures guarantee, what tests must NOT do, and how to
opt out when a test genuinely needs a fresh device.

## Why shared fixtures

Every device open/close cycle pays a non-trivial fixed cost:
`MeshDevice::create_*` initialises the cluster driver, the lock-step
allocator, the dispatch firmware, and (on T3K) auto-enables FABRIC_1D for
ETH dispatch. Doing this once per test multiplied across thousands of
tests is the dominant wall-clock cost of CI.

The shared-fixture pattern opens a device (or mesh) **once per test
suite** (C++ gtest) or **once per session keyed by `device_params`**
(Python pytest), and keeps it alive across tests. On test failure the
device is automatically torn down and re-opened before the next test, so
a stale-state failure cannot cascade.

## C++ gtest

### Available fixtures (see `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp` and `tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp`)

Single-card, suite-shared:

- `ttnn::TTNNUnitMeshCQSharedFixture` — single MMIO device, 1 CQ.
  `device_` is `MeshDevice*`. The default for any new TTNN gtest.
- `ttnn::TTNNUnitMesh2CQSingleCardSharedFixture` — single MMIO device,
  2 CQs (ETH dispatch on T3K).

Multi-card, suite-shared (all keyed by trait + recovery on `HasFailure()`):

- `MeshDevice1x2SharedFixture` — `MeshShape{1,2}`, no fabric.
- `MeshDevice2x4SharedFixture` — `MeshShape{2,4}`, no fabric.
- `MeshDevice4x8SharedFixture` — `MeshShape{4,8}`, no fabric (Galaxy).
- `MeshDevice1x4Fabric1DSharedFixture` — fabric 1D.
- `MeshDevice2x4Fabric2DSharedFixture` — fabric 2D.
- `MeshDevice1x4Fabric2DUDMSharedFixture`, `MeshDevice2x4Fabric2DUDMSharedFixture` — fabric 2D + UDM.
- `MultiCQMeshDevice1x4Fabric1DSharedFixture`, `MultiCQMeshDevice2x4Fabric1DSharedFixture` — 2 CQs.
- `GenericMeshDeviceSharedFixture` — auto-detected mesh shape, 1 CQ.
- `MultiCommandQueueT3KFixture` (TTNN-specific) — 8 unit meshes × 2 CQs, ETH dispatch.

Per-test (open/close every test) — use only when correctness requires it:

- `MeshDispatchFixture` — opens all user-exposed chips per test.
- `MeshDeviceFixtureBase` and aliases (`MeshDevice2x4Fixture` etc.).
- `MultiCommandQueueSingleDeviceFixture` (TTNN) — 2-CQ, per-test.

### What tests using a shared fixture MUST NOT do

A shared device persists across test boundaries on the happy path. Tests
that mutate persistent device state without resetting it will leak that
state to the next test. Specifically:

- Do NOT register a sub-device manager and forget to clear it. Either use
  scoped registration (`load_sub_device_manager` + RAII unload at end of
  test) or manually call `clear_loaded_sub_device_manager()` at the end.
- Do NOT call `disable_and_clear_program_cache()` unless your test's
  correctness depends on a known-empty cache. If you must clear it,
  re-`enable_program_cache()` afterwards and accept the cold-cache cost
  the next test pays.
- Do NOT open additional `MeshDevice` instances on chips already owned by
  the shared fixture. The shared fixture's `MeshDevice` shared-ptr
  remains live; opening another mesh on the same chip will fail.
- Do NOT mutate global tt-fabric configuration at test scope. Trait
  `fabric_config` is set once in `SetUpTestSuite`; calling
  `tt_fabric::SetFabricConfig` inside a test breaks fabric health
  detection in the fixture's `TearDown`.
- Do NOT assume the program cache is empty at test start.
- Do NOT assume the L1 / DRAM allocator is empty at test start. The
  shared fixture allocator IS reset between tests via the standard
  cleanup path, but tests that reach into `MetalContext` directly bypass
  that path.

If a test must violate any of the above, it should NOT use a shared
fixture — see the next section.

### Opting out: when to use a per-test fixture

Use a per-test fixture (or add an `always_recover()` trait override) when:

- The test must observe an empty program cache from line 1 (e.g. cache
  miss-rate assertions).
- The test deliberately corrupts persistent device state and recovery is
  cheaper at the device level than at the test level.
- The test opens its own device internally (e.g. multi-thread + async
  paths in `test_multiprod_queue.cpp`).

To force per-test recovery on an otherwise-shared `MeshDeviceConfigSharedFixture`,
add `static constexpr bool always_recover() { return true; }` to the trait struct.

### `Traits::auto_enable_fabric()` opt-in

`MeshDeviceConfigSharedFixture<Traits>` uses a cached runtime fabric
config to make per-test `fabric_enabled()` cheap. By default, traits
declaring `fabric_config = DISABLED` are assumed to genuinely have no
fabric and the per-test fabric health check is skipped at compile time.

Some configurations — notably T3K, where ETH dispatch auto-enables
FABRIC_1D under the hood — need the fixture to consult the runtime
config even when the trait says DISABLED. For those, opt in by adding:

```cpp
static constexpr bool auto_enable_fabric() { return true; }
```

to the trait struct. The shipped traits that need this
(`GenericMeshDeviceSharedTraits`, `MeshDevice2x4SharedTraits`,
`MeshDevice4x8SharedTraits`) already do.

## Python pytest

### Helper module

`tests/ttnn/conftest_helpers.py` exposes the reusable pieces:

- `DeviceManager` — module-scoped single-config device.
- `ParamKeyedDeviceManager` — session-scoped device, auto-detects
  compatible runs by `device_params` key.
- `ParamKeyedMeshDeviceManager` — session-scoped mesh, keyed by
  `(mesh_shape, device_params)`.
- `register_device_markers(config)` — registers `requires_fresh_device`
  and `manages_own_device` markers.
- `register_mesh_device_markers(config)` — registers
  `requires_fresh_mesh_device`.
- `resolve_device_id(config)` — TG-aware CLI device id.
- `reset_device_state(device)` / `reset_mesh_device_state(mesh_device)` —
  per-test cleanup helpers (call from autouse fixture).
- `sort_items_by_device_params(items)` — call from
  `pytest_collection_modifyitems` to batch tests with the same
  `device_params` so the manager's cached device is reused.

### Per-directory conftest pattern

Each directory that wants shared-device behaviour adds a `conftest.py`
that imports from the helper module. See
`tests/ttnn/unit_tests/operations/conv/conftest.py` for the canonical
single-card param-keyed template, and
`tests/nightly/t3000/ccl/conftest.py` for the canonical mesh-device
session manager.

### Markers

- `@pytest.mark.requires_fresh_device` — the shared device is suspended
  for this test, a brand-new device is opened, used, closed, and the
  shared device is lazily reopened on the next test that needs it. Use
  for tests that depend on an empty program cache or other persistent
  state.
- `@pytest.mark.manages_own_device` — the shared device is suspended;
  the test is responsible for opening and closing its own handle. The
  fixture yields `None`.

### What tests using shared fixtures MUST NOT do

- Do NOT call `ttnn.close_device` / `ttnn.close_mesh_device` on the
  fixture-provided handle. The manager owns the lifecycle.
- Do NOT call `ttnn.SetDefaultDevice(None)` mid-test (the helper sets
  default device on open; clearing it leaves a stale state for the next
  test).
- Do NOT call `disable_and_clear_program_cache()` unless your test's
  correctness depends on it. The manager's per-test reset deliberately
  does NOT clear the cache (clearing it breaks ETH CQ init).
- Do NOT mutate fabric config at test scope. The manager handles
  `set_fabric` / `reset_fabric` based on `device_params["fabric_config"]`.

### When to add a NEW shared fixture / NEW conftest

Add a new directory conftest if:

- The directory has 20+ tests that share a small number of distinct
  `device_params` configurations.
- Tests do not currently use `mesh_device` (use a single-device
  `ParamKeyedDeviceManager`) OR they do (use a
  `ParamKeyedMeshDeviceManager`).
- The directory does not already have a stub conftest, OR the stub says
  "not promoted because of `device_params` parametrization" (the
  param-keyed manager handles that case automatically — replace the stub).

Add a new C++ shared trait (in `multi_device_fixture.hpp`) if:

- A test uses `MeshDeviceFixtureBase` or one of its non-shared aliases
  AND the test is one of N similar tests that all open the same
  configuration.
- The trait should declare `auto_enable_fabric() = true` if it is a
  multi-card configuration that may auto-enable fabric on T3K.

## Adding tests safely

Default to:

- C++: derive from `TTNNUnitMeshCQSharedFixture` or one of the
  `*SharedFixture` typedefs.
- Python: place tests under a directory that already has a shared-fixture
  conftest, or add one using the helper module.

Only deviate from the defaults when one of the "MUST NOT" rules above
applies; in that case, explain why in a code comment so the next reader
knows the cost is intentional.
