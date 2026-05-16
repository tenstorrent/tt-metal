<!--
SUMMARY: Correctness and lifecycle review of shared device test fixture rollout to ttnn (branch nsexton/0-batch-t3k-ttnn-unit)
KEYWORDS: fixture, shared-device, gtest, lifecycle, correctness, ttnn, review
SOURCE: Code review of commit c4713fcb98 "rollout shared device test fixture to ttnn"
SCOPE: UnitMeshCQSingleCardSharedFixture, TTNNUnitMeshCQSharedFixture, MeshDeviceConfigSharedFixture, MultiCommandQueueT3KFixture, plus all migrated test files
USE WHEN: Evaluating correctness, lifecycle safety, and test isolation of the shared fixture pattern
-->

# Fixture Review A: Correctness and Lifecycle Safety

**Reviewer**: Reviewer A (Correctness Focus)
**Branch**: `nsexton/0-batch-t3k-ttnn-unit`
**Commit**: `c4713fcb98` — "rollout shared device test fixture to ttnn"
**Date**: 2026-04-01

---

## Summary

This commit does two things:
1. **Introduces new shared fixture infrastructure**: `TTNNUnitMeshCQSharedFixture`, `MultiCommandQueueT3KFixture` (shared variant), `MeshDeviceConfigSharedFixture<Traits>` template, and `mesh_device_shared_detail` helper namespace.
2. **Migrates ~20 test files** from per-test device fixtures (`TTNNFixtureWithDevice`, `TTNNFixtureWithSuiteDevice<T>`, per-test `MeshDeviceFixtureBase`) to the shared-device pattern.

The shared pattern creates devices once per test suite in `SetUpTestSuite()`, shares them across all tests, and triggers device recreation (recovery) only when `HasFailure()` is detected.

---

## Findings

### F1 — `destroy_shared_devices` does not call `close()` before clearing shared_ptrs

- **Severity**: HIGH
- **Location**: `command_queue_fixture.hpp:313-317` (`UnitMeshCQSingleCardSharedFixture::destroy_shared_devices`) and `ttnn_test_fixtures.hpp:214-217` (`MultiCommandQueueT3KFixture::destroy_shared_devs`)
- **Description**: Both `destroy_shared_devices()` / `destroy_shared_devs()` simply call `.clear()` on the shared_ptr containers without first calling `->close()` on each device. The assumption is that the MeshDevice destructor (triggered by shared_ptr refcount going to zero) will handle cleanup. However, the per-test `TearDown` in the old fixtures (`UnitMeshCQSingleCardFixture`, `MultiCommandQueueSingleDeviceFixture`) explicitly called `device->close()` before resetting. This behavioral difference may cause issues if `MeshDevice::~MeshDevice()` does not perform the same cleanup as `close()`, or if there are other shared_ptr holders keeping the device alive.
- **Contrast**: `MeshDeviceConfigSharedFixture` (in `multi_device_fixture.hpp:391-395`) correctly calls `mesh_fixture_close(shared_mesh_, cfg)` which does `mesh->close(); mesh.reset();`. This is inconsistent with the two fixtures above.
- **Why it matters**: If `~MeshDevice` does not perform full device teardown (e.g., flushing command queues, releasing hardware resources), the hardware could be left in a dirty state, causing flaky test failures on the next suite's `SetUpTestSuite()`. The inconsistency between fixture families makes it hard to reason about which path is correct.

### F2 — Exception in `create_shared_devices` leaves `devices_valid_` as false with no recovery path in `SetUpTestSuite`

- **Severity**: HIGH
- **Location**: `command_queue_fixture.hpp:216-222` (`UnitMeshCQSingleCardSharedFixture::SetUpTestSuite`)
- **Description**: If `create_shared_devices()` throws (e.g., `create_unit_meshes` fails), the exception propagates out of `SetUpTestSuite()`. Per GoogleTest behavior, when `SetUpTestSuite()` throws, the entire suite is marked as failed, but `TearDownTestSuite()` is still called. Since `devices_valid_` was never set to `true`, `TearDownTestSuite` calling `destroy_shared_devices()` will `.clear()` potentially partially-constructed containers. More critically, in `SetUp()`, the recovery path `if (needs_recovery_ || !devices_valid_)` will attempt `destroy_shared_devices(); create_shared_devices();` — but this is only reached if `SetUpTestSuite` did not throw (if it threw, GTest skips all tests without calling `SetUp`). So the exception path is actually safe, but the error message will be confusing (a raw exception rather than a GTEST_SKIP).
- **Why it matters**: A device allocation failure at suite setup becomes a hard crash rather than a graceful skip. The `MeshDeviceConfigSharedFixture::SetUpTestSuite` (line 378-389) handles this better by checking the skip reason first, though it also does not wrap `mesh_fixture_open` in a try-catch.

### F3 — Program cache state leaks between tests sharing a device

- **Severity**: HIGH
- **Location**: `test_generic_op.cpp:878-915` (`TestGenericOpProgramCache` test)
- **Description**: The `TestGenericOpProgramCache` test asserts specific program cache entry counts (e.g., `this->device_->num_program_cache_entries() == 2`). With the shared fixture, if any other test using `TTNNUnitMeshCQSharedFixture` in the same suite runs before this test and populates the program cache, the count will not be 2. The test was migrated from `TTNNFixtureWithDevice` (which created a fresh device per test) to `TTNNUnitMeshCQSharedFixture` (shared device). The program cache is device-level state that is never cleared between tests.
- **Why it matters**: This test will fail non-deterministically depending on test ordering. GTest does not guarantee test execution order within a suite. Any test that creates programs on the device will increase the cache count, causing `TestGenericOpProgramCache` to fail with a count mismatch. This is a correctness regression.

### F4 — No per-test cleanup contract for device-allocated resources

- **Severity**: MEDIUM
- **Location**: `command_queue_fixture.hpp:205-207` (shared fixture comment), all migrated tests
- **Description**: The fixture header comment says "Tests using this fixture must NOT modify persistent device state (sub-device managers, etc.)." but there is no mechanism to enforce this. There is no per-test `TearDown` that resets device state (e.g., clears buffers, programs, events, or the program cache). Tests are expected to self-clean, but this is a soft contract with no validation. Several migrated tests allocate buffers and tensors on the device (e.g., `test_create_tensor.cpp:59`, `test_generic_op.cpp` creates programs) without explicit cleanup.
- **Why it matters**: Device memory fragmentation and resource accumulation across tests within a suite. If a test leaks a buffer allocation, subsequent tests may see OOM or unexpected allocation failures that appear as flaky test failures. The only safety net is the `HasFailure()` -> recovery path, which recreates the device — but only after a test has already failed, not before.

### F5 — `HasFailure()` in `TearDown()` does not catch C++ exceptions / TT_FATAL

- **Severity**: MEDIUM
- **Location**: `command_queue_fixture.hpp:243-246`, `ttnn_test_fixtures.hpp:183-186`, `multi_device_fixture.hpp:415-418`
- **Description**: `TearDown()` uses `HasFailure()` to detect test failures and set `needs_recovery_`. However, `HasFailure()` only detects GTest assertion failures (`ASSERT_*`, `EXPECT_*`, `ADD_FAILURE()`). It does not detect C++ exceptions or `TT_FATAL` calls. Many tests in this codebase use `TT_FATAL` for correctness checks (e.g., `test_generic_op.cpp` has 12 `TT_FATAL` calls). If `TT_FATAL` throws, GTest catches it and marks the test as failed — and `HasFailure()` should return true in that case. However, if the exception corrupts device state before being caught, the recovery may be insufficient (just `needs_recovery_ = true` followed by device recreation on next test setup).
- **Why it matters**: The recovery mechanism is sound for GTest assertion failures, but if a `TT_FATAL` or other exception leaves the device in a partially-programmed state (e.g., mid-command-queue-operation), the device may need a more thorough reset than just `destroy_shared_devices(); create_shared_devices();`. The shared_ptr `clear()` (without explicit `close()`) compounds this concern (see F1).

### F6 — `TTNNUnitMeshCQSharedFixture` exposes raw pointer `device_` that can dangle

- **Severity**: MEDIUM
- **Location**: `ttnn_test_fixtures.hpp:70-78`
- **Description**: `TTNNUnitMeshCQSharedFixture::SetUp()` sets `device_ = devices_[0].get()` — a raw pointer extracted from a shared_ptr. The `devices_` member is a copy of the static `shared_devices_` vector (set in `UnitMeshCQSingleCardSharedFixture::SetUp` line 239). If device recovery occurs between tests (another test fails, `needs_recovery_` is set, and the next `SetUp()` calls `destroy_shared_devices()` then `create_shared_devices()`), the raw pointer `device_` from the previous test's `SetUp()` would be invalidated. However, since `SetUp()` re-extracts the pointer each time, this is only dangerous if a test stores `device_` in a member that outlives a single test — which does not appear to happen in the current codebase.
- **Why it matters**: Low practical risk today, but the pattern is fragile. Any future test that caches `device_` (e.g., in a helper object constructed in `SetUp()` that survives into `TearDown()`) could hit a use-after-free. A shared_ptr or weak_ptr would be safer.

### F7 — `GTEST_SKIP()` in `SetUp()` still runs `TearDown()` — no guard in shared `TearDown()`

- **Severity**: LOW
- **Location**: `command_queue_fixture.hpp:226-247`, `ttnn_test_fixtures.hpp:160-186`
- **Description**: When `SetUp()` calls `GTEST_SKIP()`, GTest still invokes `TearDown()`. In `UnitMeshCQSingleCardSharedFixture::TearDown()`, the only action is checking `HasFailure()` and setting `needs_recovery_`. For a skipped test, `HasFailure()` returns false, so `TearDown()` is a no-op. This is correct behavior. Similarly, `MultiCommandQueueT3KFixture::TearDown()` only checks `HasFailure()`. So skip handling is safe.
- **Why it matters**: No bug, but worth noting for completeness: the skip path is clean.

### F8 — Parameterized test suites sharing the same base fixture share static state

- **Severity**: MEDIUM
- **Location**: `test_graph_query_op_constraints.cpp` (5 different parameterized test classes all using `TTNNUnitMeshCQSharedFixture`), `test_create_tensor.cpp` (2 classes)
- **Description**: When multiple parameterized test classes in the same translation unit inherit from `TTNNUnitMeshCQSharedFixture`, they all share the same static `shared_devices_`, `devices_valid_`, and `needs_recovery_` variables (inherited from `UnitMeshCQSingleCardSharedFixture`). GTest treats each `INSTANTIATE_TEST_SUITE_P` as a separate suite, calling `SetUpTestSuite()`/`TearDownTestSuite()` for each. The first suite's `TearDownTestSuite()` calls `destroy_shared_devices()`, which clears `shared_devices_` and sets `devices_valid_ = false`. The next suite's `SetUpTestSuite()` then calls `create_shared_devices()` again, recreating devices. This means devices are NOT truly shared across different parameterized suites in the same TU — they are destroyed and recreated at each suite boundary.
- **Why it matters**: The performance benefit of the shared fixture is partially lost for files with multiple parameterized test suites. Each suite still pays device creation/teardown cost. This is not a correctness bug, but it means the optimization is less effective than expected for files like `test_graph_query_op_constraints.cpp` (5 suites). The old `TTNNFixtureWithSuiteDevice<Derived>` CRTP pattern had the same behavior (per-derived-class statics), so this is not a regression.

### F9 — Fabric config state leak between `MeshDeviceConfigSharedFixture` suites

- **Severity**: MEDIUM
- **Location**: `multi_device_fixture.hpp:378-395` (`MeshDeviceConfigSharedFixture::SetUpTestSuite` / `TearDownTestSuite`)
- **Description**: `SetUpTestSuite()` calls `mesh_fixture_close(shared_mesh_, cfg)` before opening a new mesh. The close function resets fabric config to DISABLED if the config had fabric enabled. However, if `SetUpTestSuite` is called but the skip reason is detected (line 380-384), it closes the mesh (resetting fabric to DISABLED) and returns without opening a new one. If a previous suite had fabric DISABLED and this suite's config has fabric ENABLED, the close is a no-op (mesh is null), and no open happens — correct. But if a previous suite left fabric in an unexpected state (e.g., due to a test crash before `TearDownTestSuite`), the fabric state is not explicitly reset at the start of `SetUpTestSuite`.
- **Why it matters**: In normal operation this is fine because `TearDownTestSuite` always runs (even after `SetUpTestSuite` failure). But if the process crashes mid-test (e.g., segfault, which hardware tests can trigger), fabric state could leak to subsequent test runs in the same process.

### F10 — `MeshDevice1x4Fabric1DSharedFixture` alias in CCL tests changes lifecycle semantics

- **Severity**: MEDIUM
- **Location**: `test_multi_tensor_ccl.cpp:32` — `using MeshDevice1x4Fixture = MeshDevice1x4Fabric1DSharedFixture;`
- **Description**: The old `MeshDevice1x4Fixture` was per-test (created/destroyed each test via `MeshDeviceFixtureBase`). The new alias points to `MeshDevice1x4Fabric1DSharedFixture` which is `MeshDeviceConfigSharedFixture<MeshDevice1x4Fabric1DSharedTraits>` — a suite-level shared fixture. CCL tests (AllGather, AllReduce, etc.) are inherently stateful operations that program fabric routers and modify device state. The shared fixture's recovery mechanism (`HasFailure()` -> recreate) may not be sufficient if a CCL operation leaves fabric hardware in a bad state without triggering a GTest assertion failure.
- **Why it matters**: CCL operations are among the most likely to corrupt device state. Moving them from per-test to shared-device fixtures increases the blast radius of any single test's device-state corruption. If a CCL test partially programs fabric and then silently succeeds (no assertion failure), the next test inherits corrupted fabric state.

### F11 — Python conftest changes are independent and correct

- **Severity**: LOW (no issue)
- **Location**: `tests/ttnn/distributed/conftest.py`, `tests/ttnn/unit_tests/operations/transformers/conftest.py`
- **Description**: Both conftest files introduce session-scoped (distributed) and module-scoped (transformers) Python fixtures for `mesh_device`. These are Python-side fixtures that do not interact with the C++ shared fixture infrastructure. They correctly use `yield` with proper teardown (close submeshes, close mesh, reset fabric). The `prefetcher_multi_device_mesh` fixture is module-scoped, which is appropriate for its usage in `test_prefetcher.py`.
- **Why it matters**: No correctness issue. These are well-structured Python fixtures that follow the same "reduce device open/close overhead" philosophy as the C++ changes.

### F12 — Parallel test execution safety: fixtures are not thread-safe

- **Severity**: LOW (acceptable given GTest execution model)
- **Location**: All shared fixtures with `inline static` members
- **Description**: The `inline static` shared state (`shared_devices_`, `devices_valid_`, `needs_recovery_`) is not protected by any synchronization primitives. If tests within a suite were executed concurrently (e.g., via `--gtest_parallel`), there would be data races. However, standard GTest execution runs tests sequentially within a suite, so this is not a practical concern under normal usage.
- **Why it matters**: Not a bug under normal GTest usage. Would become a bug if someone introduces intra-suite parallelism. The `--gtest_filter` flag does not introduce parallelism.

---

## Top-Priority Items

1. **F3 (HIGH)**: `TestGenericOpProgramCache` will break — it asserts exact program cache counts that are no longer valid with shared devices. This is a guaranteed test failure depending on ordering.
2. **F1 (HIGH)**: Missing `close()` calls in `destroy_shared_devices` / `destroy_shared_devs` — inconsistent with `MeshDeviceConfigSharedFixture` which correctly calls `close()` before `reset()`.
3. **F2 (HIGH)**: No graceful handling of device creation failure in `SetUpTestSuite` — raw exception rather than skip.
4. **F10 (MEDIUM)**: CCL tests moving to shared fixtures increases blast radius of fabric state corruption.
5. **F4 (MEDIUM)**: No per-test cleanup contract enforcement for device-allocated resources.

---

## Fabric State Corruption — In-Depth Investigation

*Follow-up to finding A-F10. Investigates whether fabric state corruption can be detected programmatically and proposes concrete mitigations.*

### 1. Fabric Lifecycle in Shared Fixtures

The `MeshDeviceConfigSharedFixture<Traits>` manages fabric through a well-defined lifecycle:

```
SetUpTestSuite:
  SetFabricConfig(FABRIC_1D, STRICT_SYSTEM_HEALTH_SETUP_MODE, ...)
  MeshDevice::create(...)         // triggers DeviceManager::initialize_fabric_and_dispatch_fw()
    -> FabricFirmwareInitializer::init()
       -> control_plane_.write_routing_tables_to_all_chips()
       -> compile_and_configure_fabric()     // compile + configure on each device
    -> FabricFirmwareInitializer::configure()
       -> wait_for_fabric_router_sync()      // poll EDMStatus::LOCAL_HANDSHAKE_COMPLETE
       -> write READY_FOR_TRAFFIC signal

Per-test SetUp:
  if (needs_recovery_ || !devices_valid_):
    mesh_fixture_close(shared_mesh_) -> mesh->close() + SetFabricConfig(DISABLED)
    mesh_fixture_open(cfg)           -> full reinit

Per-test TearDown:
  if (HasFailure()): needs_recovery_ = true

TearDownTestSuite:
  mesh_fixture_close(shared_mesh_)
    -> mesh->close() -> DeviceManager::close_devices()
       -> FabricFirmwareInitializer::teardown()
          -> write IMMEDIATELY_TERMINATE to each master router
       -> FabricFirmwareInitializer::post_teardown()
          -> set_fabric_config(DISABLED)
    -> SetFabricConfig(DISABLED)
       -> teardown_fabric_config()
          -> clear_fabric_context()
```

**Key files:**
- `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp` (lines 357-425)
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`
- `tt_metal/impl/context/metal_env.cpp` (lines 263-268)

### 2. What "Fabric State" Means Concretely

The fabric is a persistent mesh of router firmware running on ethernet (eRISC) cores. Each router manages:

**Per-router L1 state** (on each ethernet core):
- `EDMStatus` at `edm_status_address` — lifecycle: `STARTED -> REMOTE_HANDSHAKE_COMPLETE -> LOCAL_HANDSHAKE_COMPLETE -> READY_FOR_TRAFFIC -> TERMINATED` (defined in `tt_metal/fabric/fabric_edm_packet_header.hpp:48-94`)
- `RouterStateManager` at `routing_l1_info_t.state_manager` — contains:
  - `RouterState state` (device-written, host-readable): `INITIALIZING=0, RUNNING=1, PAUSED=2, DRAINING=3, RETRAINING=4` (defined in `tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h:15`)
  - `RouterCommand command` (host-written, device-readable): `RUN=0, PAUSE=1, DRAIN=3, RETRAIN=4` (defined in `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h:558-571`)
- `TerminationSignal` at `termination_signal_address`: `KEEP_RUNNING=0, GRACEFULLY_TERMINATE=1, IMMEDIATELY_TERMINATE=2`

**Per-channel state:**
- Sender channel: `buffer_index_address`, `local_flow_control_semaphore_address`, `connection_semaphore_address`, `producer_terminate_connection_address`
- Receiver channel: `downstream_teardown_semaphore_address`
- Credit counters: `to_sender_channel_remote_ack_counters`, `to_sender_channel_remote_completion_counters`
- Connection info: `worker_conn_info_base_address` per sender channel

**Telemetry layer** (read from host via `read_fabric_telemetry`):
- `FabricTelemetryRouterState`: `Standby=0, Active=1, Paused=2, Draining=3`
- Per-eRISC heartbeats: `tx_heartbeat`, `rx_heartbeat` (monotonically increasing 64-bit counters)
- Bandwidth counters: `elapsed_active_cycles`, `words_sent`, `packets_sent`

### 3. What "Silent Corruption" Looks Like

A CCL operation (all_gather, reduce_scatter, send_recv) interacts with fabric by:
1. Establishing connections via `append_fabric_connection_rt_args` (writes semaphore addresses, buffer indices)
2. Running worker kernels that send/receive packets through fabric mux channels
3. The fabric routers forward packets between chips using credit-based flow control

**Silent corruption scenarios** (no GTest assertion failure, but fabric state is wrong):
1. **Stuck credit counters**: A sender channel's `local_flow_control_semaphore` is decremented but never incremented back (receiver didn't send ack). Subsequent tests on the same channel see 0 credits and deadlock or timeout.
2. **Connection info stale**: A worker connection's `buffer_index_rdptr` points to a freed buffer. Next test's EDM reads garbage pointers.
3. **Undrained packets in transit**: Packets were sent but not fully received. Router ring buffers contain partial data. Next operation reads stale packet headers.
4. **Mux channel state leak**: A `FabricMuxConfig` channel's `buffer_index_region_` or `flow_control_region_` retains values from a previous test's connection.
5. **Router paused or draining**: If a CCL op issued a PAUSE/DRAIN command and crashed before resuming to RUN, all subsequent traffic on that router stalls.

### 4. Existing Detection Capabilities

**The codebase already has the building blocks for detection — they just aren't wired into the test fixture.**

#### 4a. Telemetry Reader API (PUBLIC, HOST-SIDE)

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp
std::vector<FabricTelemetrySample> read_fabric_telemetry(const FabricNodeId& fabric_node_id);
```

Returns per-channel snapshots including `router_state`, heartbeats, and bandwidth counters. This is a **non-invasive read** from host via `cluster.read_from_device()`. No firmware interaction needed.

**What it can detect:**
- Router stuck in non-RUNNING state (Paused, Draining)
- Heartbeat stall (tx/rx heartbeat stopped incrementing between tests — router is hung)

**File:** `tt_metal/fabric/fabric_telemetry_reader.cpp` (lines 69-113)

#### 4b. EDMStatus Readback

The `edm_status_address` is a known L1 offset (uniform across all router cores). It can be read via:
```cpp
detail::ReadFromDeviceL1(dev, eth_core, edm_status_address, 4, status_buffer, CoreType::ETH);
```

**What it can detect:**
- Router not in `READY_FOR_TRAFFIC` state (indicating initialization failure or premature termination)

The `FabricBuilderContext` exposes these addresses:
```cpp
builder_ctx.get_fabric_router_sync_address_and_status()  // -> {edm_status_address, LOCAL_HANDSHAKE_COMPLETE}
builder_ctx.get_fabric_router_ready_address_and_signal()  // -> {edm_status_address, READY_FOR_TRAFFIC}
```

**File:** `tt_metal/fabric/fabric_builder_context.cpp` (lines 211-216)

#### 4c. RouterStateManager Host Read

The `RouterStateManager.state` field is explicitly documented as "written by device, read by host" (`fabric_common.h:575`). The host can read this at a known offset within `routing_l1_info_t` on any ethernet core. A healthy idle router should be in `RouterState::RUNNING(1)`.

#### 4d. MeshDevice::quiesce_devices()

CCL tests already call `quiesce_devices()` before operations (`test_multi_tensor_ccl.cpp`). This waits for all command queues to complete and resets event counters. However, **it does not check fabric router state** — it only synchronizes the host-side command queues.

### 5. Recovery Mechanism Analysis

The shared fixture's recovery path (`mesh_fixture_close` + `mesh_fixture_open`) triggers:

```
mesh_fixture_close:
  mesh->close()              // closes MeshDevice, triggers DeviceManager::close_devices()
    -> DispatchKernelInitializer::teardown()
    -> FabricFirmwareInitializer::teardown()
       -> writes IMMEDIATELY_TERMINATE to all master routers
       -> writes termination signal to all tensix mux cores
    -> Device::close() per device  // resets device hardware
  SetFabricConfig(DISABLED)
    -> teardown_fabric_config()
       -> clear_fabric_context()

mesh_fixture_open:
  SetFabricConfig(FABRIC_1D, STRICT_SYSTEM_HEALTH_SETUP_MODE)
  MeshDevice::create(...)     // full device init + fabric reinit
```

**This is a full nuclear reset** — it terminates all routers, closes all devices, clears all fabric context, and reinitializes from scratch. It **does** fully reset fabric state. The problem is it only triggers on `HasFailure()`, which requires a GTest assertion to fire.

**There is no lighter-weight fabric reset.** The `RouterCommand::PAUSE` + `RouterCommand::DRAIN` + `RouterCommand::RUN` sequence could theoretically quiesce and restart a router, but there's no host-side API to issue this sequence — it would require writing directly to the `RouterStateManager.command` field in L1.

### 6. Proposed Mitigations

#### Mitigation 1: Post-Test Fabric Health Assert (LOW EFFORT, HIGH VALUE)

**What:** Add a fabric router state check in `MeshDeviceConfigSharedFixture::TearDown()` that reads router telemetry/state after every test and triggers recovery if any router is not in a healthy state.

**Protects against:** Routers stuck in PAUSED/DRAINING state, router crashes (EDMStatus != READY_FOR_TRAFFIC), heartbeat stalls.

**Implementation:**

In `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp`, modify TearDown:

```cpp
void TearDown() override {
    if (HasFailure()) {
        needs_recovery_ = true;
    }
    // Fabric health check: detect silent corruption even without assertion failures
    if (!needs_recovery_ && shared_mesh_ && Traits::config().fabric_config != tt_fabric::FabricConfig::DISABLED) {
        needs_recovery_ = !check_fabric_routers_healthy(shared_mesh_);
    }
}
```

The `check_fabric_routers_healthy` function would use `read_fabric_telemetry()` for each fabric node:
```cpp
static bool check_fabric_routers_healthy(const std::shared_ptr<MeshDevice>& mesh) {
    for (auto* device : mesh->get_devices()) {
        auto fabric_node_id = tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
        auto samples = tt_fabric::read_fabric_telemetry(fabric_node_id);
        for (const auto& sample : samples) {
            if (sample.snapshot.dynamic_info) {
                for (const auto& erisc : sample.snapshot.dynamic_info->erisc) {
                    if (erisc.router_state != tt_fabric::FabricTelemetryRouterState::Active) {
                        return false;  // Router not in active/running state
                    }
                }
            }
        }
    }
    return true;
}
```

**Effort:** Low — ~30 lines of code in the fixture header, uses existing public APIs.
**Risks:** Adds ~1-5ms per test for L1 readback (negligible for T3K CCL tests that take 100ms+). May produce false positives if routers legitimately enter non-Active states transiently — need to validate timing.
**Where:** `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp`, TearDown of `MeshDeviceConfigSharedFixture`.

#### Mitigation 2: EDMStatus Readback in TearDown (LOW EFFORT, TARGETED)

**What:** Read the `edm_status_address` on each router's master core after every test and verify it's still `READY_FOR_TRAFFIC`.

**Protects against:** Router firmware crashes, premature termination, initialization corruption.

**Implementation:** Simpler than Mitigation 1 — directly reads a single 4-byte L1 word per device:

```cpp
static bool check_edm_status_healthy(const std::shared_ptr<MeshDevice>& mesh) {
    const auto& fabric_context = MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& builder_ctx = fabric_context.get_builder_context();
    auto [edm_status_addr, expected] = builder_ctx.get_fabric_router_sync_address_and_status();
    auto ready_signal = builder_ctx.get_fabric_router_ready_address_and_signal();
    uint32_t expected_status = ready_signal ? static_cast<uint32_t>(ready_signal->second) : expected;

    for (auto* device : mesh->get_devices()) {
        auto num_routers = builder_ctx.get_num_fabric_initialized_routers(device->id());
        if (num_routers == 0) continue;
        auto master_chan = builder_ctx.get_fabric_master_router_chan(device->id());
        auto core = MetalContext::instance().get_cluster().get_soc_desc(device->id())
            .get_eth_core_for_channel(master_chan, CoordSystem::LOGICAL);
        std::vector<uint32_t> status(1);
        detail::ReadFromDeviceL1(device, core, edm_status_addr, 4, status, CoreType::ETH);
        if (status[0] != expected_status) return false;
    }
    return true;
}
```

**Effort:** Low — ~20 lines, uses existing internal APIs already used by `wait_for_fabric_router_sync`.
**Risks:** Only catches router-level crashes, not channel-level corruption (stale credits, undrained buffers). But it catches the most severe class of failure.
**Where:** Same location as Mitigation 1.

#### Mitigation 3: Heartbeat Delta Check (MEDIUM EFFORT, BROAD COVERAGE)

**What:** Snapshot fabric heartbeat counters in `SetUp`, then compare in `TearDown`. If heartbeats stopped incrementing during a test that performed fabric operations, something is wrong.

**Protects against:** Fabric router hangs (stuck in a loop, deadlocked on credits), partially-functioning routers that respond to status queries but don't actually route packets.

**Implementation:**
- Store `tx_heartbeat` / `rx_heartbeat` per channel in SetUp
- In TearDown, re-read and verify they advanced (for tests that did fabric ops)
- Requires `read_fabric_telemetry` in both SetUp and TearDown

**Effort:** Medium — need to store per-test baseline state, manage the comparison logic, and handle edge cases (what if a test legitimately doesn't use fabric?).
**Risks:** False positives for tests that don't do CCL operations (heartbeats may not advance). Needs a mechanism to tag tests as "fabric-using" or to only check when heartbeats were nonzero at start.
**Where:** `MeshDeviceConfigSharedFixture` SetUp/TearDown.

#### Mitigation 4: Always-Recover for CCL Suites (LOW EFFORT, BRUTE FORCE)

**What:** For fabric-enabled shared fixtures, set `needs_recovery_ = true` unconditionally in TearDown. Every test gets a fresh device+fabric.

**Protects against:** Everything — there's no shared state between tests.

**Implementation:** One line change in TearDown:
```cpp
void TearDown() override {
    needs_recovery_ = true;  // Always recover for fabric suites
}
```

**Effort:** Trivially low.
**Risks:** Defeats the purpose of shared fixtures. Device re-creation on T3K takes ~2-5 seconds per test. For a CCL suite with 30 tests, this adds ~60-150 seconds. Essentially reverts to per-test fixtures without the code organization benefits.
**Where:** Override TearDown in a CCL-specific subclass of `MeshDeviceConfigSharedFixture`, or make it configurable via a Traits flag.

#### Mitigation 5: RouterCommand PAUSE/DRAIN/RUN Cycle (HIGH EFFORT, THOROUGH)

**What:** After each test, issue a PAUSE -> DRAIN -> wait-for-drain-complete -> RUN command sequence to each router via the `RouterStateManager.command` field. This flushes all in-flight packets and resets the router to a clean operational state without full device teardown.

**Protects against:** Undrained packets, stuck credits, partial transfers. The DRAIN state causes routers to "accept messages but drop them" (per `fabric_common.h:568`), effectively flushing the pipeline.

**Implementation:**
1. Write `RouterCommand::PAUSE` to each router's `state_manager.command` field via `detail::WriteToDeviceL1`
2. Poll `state_manager.state` until it transitions to `PAUSED`
3. Write `RouterCommand::DRAIN`
4. Poll until state transitions to `DRAINING` then back to `PAUSED` (drain complete)
5. Write `RouterCommand::RUN`
6. Poll until state is `RUNNING`

**Effort:** High — no existing host-side API for this command sequence. Would need to: (a) expose router L1 addresses from `FabricBuilderContext`, (b) implement the polling loop with timeout, (c) handle error conditions (router doesn't respond to PAUSE). The `routing_l1_info_t` structure and `state_manager` offset are well-defined though.
**Risks:** Uncertain timing — DRAIN may take variable time depending on in-flight packets. If a router is truly hung, PAUSE won't take effect. Adds ~10-50ms per test depending on fabric size.
**Where:** New utility function in `multi_device_fixture.hpp` or a new fabric test utility header.

### 7. Recommendation

**Start with Mitigation 1 (telemetry-based health check) + Mitigation 2 (EDM status check).** Together they cover:
- Router crashes / premature termination (via EDMStatus)
- Router stuck in wrong state (via telemetry RouterState)
- Router hangs (via heartbeat comparison if Mitigation 3 is added later)

These are low-effort, non-invasive, and use existing public APIs. If they prove insufficient in practice (i.e., corruption at the channel level slips through), escalate to Mitigation 5 (PAUSE/DRAIN/RUN cycle) or Mitigation 4 (always-recover) as a fallback.

A practical intermediate step: add a `Traits::always_recover()` flag that CCL-specific fixture traits can set to `true`, giving per-suite control over the recovery policy without modifying the generic fixture template.

### 8. Key Code References

```
Fixture lifecycle:
  tests/tt_metal/tt_metal/common/multi_device_fixture.hpp

Fabric firmware init/teardown:
  tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp

Fabric config management:
  tt_metal/impl/context/metal_env.cpp (lines 263-294)

Telemetry reader (host-side, non-invasive):
  tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp
  tt_metal/fabric/fabric_telemetry_reader.cpp

Telemetry types:
  tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry.hpp

Router state machine:
  tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h (lines 558-605)
  tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h (line 15)

EDM status enum + termination signals:
  tt_metal/fabric/fabric_edm_packet_header.hpp (lines 37-94)

Router L1 addresses:
  tt_metal/fabric/fabric_builder_context.hpp (lines 153-165)

EDM builder config (all router L1 addresses):
  tt_metal/fabric/erisc_datamover_builder.hpp (lines 260-328)

Erisc router firmware (drain/pause/run logic):
  tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp (lines 2041-2111)

CCL test using shared fabric fixture:
  tests/ttnn/unit_tests/gtests/ccl/test_multi_tensor_ccl.cpp
  tests/ttnn/unit_tests/gtests/ccl/test_send_recv_ops.cpp
```
