<!--
SUMMARY: Reviewer B analysis of shared device test fixture rollout to ttnn — completeness, best practices, migration gaps
KEYWORDS: fixture, gtest, shared-device, migration, ttnn, code-review, test-infrastructure
SOURCE: Branch nsexton/0-batch-t3k-ttnn-unit, commit "rollout shared device test fixture to ttnn"
SCOPE: All 26 changed files plus unmigrated files in tests/ttnn/unit_tests/gtests/
USE WHEN: Reviewing or extending the shared fixture migration
-->

# Fixture Review B: Completeness of Migration & Best Practices

**Branch:** `nsexton/0-batch-t3k-ttnn-unit`
**Reviewer:** B (completeness & best practices focus)
**Date:** 2026-04-01

---

## Executive Summary

The migration is well-executed for the files it touches. All migrated C++ test files correctly
use `TTNNUnitMeshCQSharedFixture` with proper `TEST_F`/`TEST_P` patterns. However, there are
two files that should have been migrated but were not, two dead-code fixture classes that should
be cleaned up, and a potential resource accumulation concern in the heavyweight gelu_bw_ulp test
suite. The Python conftest changes are clean and pose no double-init risk with the C++ side.

---

## Findings

### F1: Unmigrated file — `test_graph_query_op_runtime.cpp` still opens its own device

- **Severity:** HIGH
- **Location:** `tests/ttnn/unit_tests/gtests/test_graph_query_op_runtime.cpp:41-56`
- **Description:** This file defines its own `TTNNFixtureWithTraceEnabledDevice` class that
  manually calls `ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, 200000)` in `SetUp()` and
  `ttnn::close_device(*device_)` in `TearDown()`. This is the exact per-test open/close pattern
  the migration is eliminating.
- **Why it matters:** Every test in `BinaryOpTraceRuntime` (parameterized) reopens the device.
  This is slow and inconsistent with the rest of the migrated suite. The custom trace_region_size
  (200000) is the likely reason it was skipped — `TTNNUnitMeshCQSharedFixture` uses
  `DEFAULT_TRACE_REGION_SIZE`. The author would need to either:
  (a) accept the default trace region size, or
  (b) parameterize the shared fixture to support custom trace region sizes.

### F2: Unmigrated file — `test_conv2d.cpp` creates its own device inline

- **Severity:** MEDIUM
- **Location:** `tests/ttnn/unit_tests/gtests/test_conv2d.cpp:129-130`
- **Description:** `Conv2DFixture` inherits from bare `::testing::Test` and creates a
  `MeshDevice::create_unit_mesh(0, l1_small_size)` inline in the test body with a custom
  `l1_small_size = 16384`. Similar to F1, the custom config is the reason it wasn't migrated.
- **Why it matters:** Per-test device creation. Less critical than F1 because the custom
  `l1_small_size` is a genuine configuration difference, but it still defeats shared-device
  benefits.

### F3: `test_generic_op.cpp` — `MeshDevice1x4FabricFixture` still uses per-test `MeshDeviceFixtureBase`

- **Severity:** LOW
- **Location:** `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:1031-1039`
- **Description:** While the 10 `TTNNUnitMeshCQSharedFixture` tests in this file were migrated,
  the `MeshDevice1x4FabricFixture` (line 1031) still inherits from `MeshDeviceFixtureBase`
  (per-test open/close). The comment in `multi_device_fixture.hpp:361-362` explicitly calls this
  out as a "future migration candidate."
- **Why it matters:** Acknowledged tech debt. The fabric config requirement makes this non-trivial
  to migrate. Low severity since it's documented.

### F4: Dead code — `TTNNFixtureWithDevice` and `TTNNFixtureWithSuiteDevice` are unused

- **Severity:** MEDIUM
- **Location:** `tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp:52-68` and `:89-117`
- **Description:** After this migration, `TTNNFixtureWithDevice` (the old per-test fixture) and
  `TTNNFixtureWithSuiteDevice` (a CRTP-based suite-level fixture) have zero consumers outside
  the header itself. Grep across the entire `tests/ttnn/` tree shows no test file inherits from
  either class.
- **Why it matters:** Dead code adds confusion. Future contributors may use the old fixture
  thinking it's the canonical pattern. These should be removed (or at minimum marked
  `[[deprecated]]`) in this PR or a follow-up.

### F5: Resource accumulation in `test_gelu_bw_ulp.cpp` — 19 TEST_F on shared device

- **Severity:** MEDIUM
- **Location:** `tests/ttnn/unit_tests/gtests/test_gelu_bw_ulp.cpp` (entire file)
- **Description:** Two fixture classes (`GeluBwUlpTest`, `GeluBwPolyTest`) each derive from
  `TTNNUnitMeshCQSharedFixture`. Together they have 19 `TEST_F` methods. Several tests
  (e.g., `ComprehensiveULPByRegion`, `ComprehensiveULPAnalysis`, `ExpBasedRegionFullDump`)
  sweep all ~65,000 BF16 values using batched tensor operations. Each test creates tensors
  on-device via `ttnn::full()` and `ttnn::experimental::gelu_bw()`.

  With the old per-test fixture, these tensors were freed between tests (device close/reopen).
  With the shared fixture, tensors created on-device must be explicitly deallocated or go out
  of scope before their backing buffers are freed. The test code uses local variables inside
  TEST_F bodies, so they do go out of scope at test end — **this is likely fine**. However,
  the sheer volume (65K-element sweeps across 19 tests) means any leaked tensor reference
  would accumulate silently.
- **Why it matters:** No immediate bug, but this is the highest-risk file for subtle resource
  leaks under the new shared model. Worth a targeted L1 memory watermark check in CI.

### F6: `MultiCommandQueueSingleDeviceFixture` — per-test open/close, not migrated

- **Severity:** LOW
- **Location:** `tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp:119-140`
- **Description:** This fixture (used by `test_async_runtime.cpp` and `test_multiprod_queue.cpp`)
  still opens/closes the device per test via `SetUp()`/`TearDown()`. It creates a device with
  **2 command queues** and conditional ethernet dispatch, which differs from the shared fixture's
  single-CQ default.
- **Why it matters:** Legitimate configuration difference (2 CQs). Not a migration gap per se,
  but worth noting for completeness. The `MultiCommandQueueT3KFixture` (used by
  `test_multi_cq_multi_dev.cpp`) already implements suite-level sharing with recovery — the
  single-device variant could follow the same pattern.

### F7: Python conftest — session-scoped `mesh_device` is isolated from C++ gtest

- **Severity:** NONE (informational)
- **Location:** `tests/ttnn/distributed/conftest.py` and
  `tests/ttnn/unit_tests/operations/transformers/conftest.py`
- **Description:** The distributed conftest defines a `session`-scoped `mesh_device` fixture.
  The transformers conftest defines a `module`-scoped `prefetcher_multi_device_mesh`. Both
  use `ttnn.open_mesh_device()` / `ttnn.close_mesh_device()` through the Python bindings.
  The C++ gtests run in separate processes and never share device state with Python tests.
- **Why it matters:** No double-initialization risk. Python and C++ tests are in separate
  executables. The `test_prefetcher.py` correctly uses `prefetcher_multi_device_mesh` (not
  `mesh_device`) and has its own `device_params` indirect parametrization for trace region
  size. Clean separation.

### F8: Header hygiene — `ttnn_test_fixtures.hpp` uses `#pragma once`, includes are correct

- **Severity:** NONE (informational)
- **Location:** `tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp:5`
- **Description:** Uses `#pragma once` (correct for this codebase). All new classes
  (`TTNNUnitMeshCQSharedFixture`, `MultiCommandQueueT3KFixture`) are in `namespace ttnn`.
  No ODR violations — the `TTNNUnitMeshCQSharedFixture` has no inline static members (it
  delegates to `UnitMeshCQSingleCardSharedFixture` which handles the statics). The CRTP
  `TTNNFixtureWithSuiteDevice` correctly uses out-of-line static member definitions.
  `using namespace tt::tt_metal;` at file scope (line 27) is messy but pre-existing.

### F9: Skip logic is correct but has different granularity across fixtures

- **Severity:** LOW
- **Location:** Various
- **Description:**
  - `TTNNUnitMeshCQSharedFixture::SetUp()` delegates to `UnitMeshCQSingleCardSharedFixture::SetUp()`,
    which skips per-test if slow dispatch is active or no devices are available. The suite-level
    `SetUpTestSuite()` also checks slow dispatch and bails early (no device creation). This means
    the entire suite is skipped on slow-dispatch machines — correct behavior.
  - `MultiCommandQueueT3KFixture::SetUp()` checks `num_devices_ < 8` and `arch_ != WORMHOLE_B0`
    per-test, with device recovery. Suite-level `SetUpTestSuite()` checks the same conditions
    before creating devices. This is correct — suite skip prevents unnecessary device creation.
  - No test in the migrated set has per-test hardware skip logic that would conflict with
    suite-level decisions.
- **Why it matters:** The skip granularity is appropriate. Suite-level skip avoids expensive
  device creation when the hardware doesn't match. Per-test skip (e.g., `MultiCommandQueueT3KFixture`)
  handles recovery after failure.

---

## Summary Table

```
ID   Sev       File / Area                              Issue
F1   HIGH      test_graph_query_op_runtime.cpp          Manual device open, should migrate
F2   MEDIUM    test_conv2d.cpp                          Inline device creation (custom l1_small_size)
F3   LOW       test_generic_op.cpp:1031                 MeshDevice1x4FabricFixture still per-test (documented)
F4   MEDIUM    ttnn_test_fixtures.hpp:52-117            TTNNFixtureWithDevice + TTNNFixtureWithSuiteDevice are dead code
F5   MEDIUM    test_gelu_bw_ulp.cpp                     19 heavy tests on shared device — resource accumulation risk
F6   LOW       ttnn_test_fixtures.hpp:119               MultiCommandQueueSingleDeviceFixture not migrated (2-CQ config)
F7   NONE      conftest.py files                        No double-init risk (separate processes)
F8   NONE      ttnn_test_fixtures.hpp                   Header hygiene OK
F9   LOW       Various                                  Skip logic correct, granularity appropriate
```

## Recommendations

1. **Migrate `test_graph_query_op_runtime.cpp`** (F1) — either accept default trace region or
   add a config param to the shared fixture.
2. **Remove dead fixture classes** (F4) — `TTNNFixtureWithDevice` and `TTNNFixtureWithSuiteDevice`
   have no consumers. Delete them or add `[[deprecated]]` annotations.
3. **Add L1 memory watermark CI check** for `test_gelu_bw_ulp.cpp` (F5) to catch accumulation
   regressions early.
4. **File a follow-up issue** for `test_conv2d.cpp` (F2) and `MultiCommandQueueSingleDeviceFixture`
   (F6) migration when the shared fixture supports custom configs.
