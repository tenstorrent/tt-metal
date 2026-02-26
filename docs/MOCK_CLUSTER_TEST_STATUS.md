# Mock Cluster Test Status

This document tracks the status of tests run with mock clusters, extracted from GitHub workflows and test scripts.

**Last Updated:** 2026-02-26

## Test Execution Environment

- **Build Directory:** `build_Debug`
- **Mock Cluster Descriptors:** Located in `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/`
- **Environment Variable:** `TT_METAL_MOCK_CLUSTER_DESC_PATH`

## Single-Host Tests Status

### ✅ PASSED Tests

#### 1. FabricTopologyHelpers Tests
- **Filter:** `FabricTopologyHelpers*`
- **Mock Cluster Required:** No (pure unit tests)
- **Status:** ✅ PASSED (20 tests)
- **Command:**
  ```bash
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTopologyHelpers*"
  ```

#### 2. MockClusterTopologyFixture Tests
- **Filter:** `MockClusterTopologyFixture*`
- **Mock Cluster Required:** Yes
- **Status:** ✅ PASSED

**With t3k_cluster_desc.yaml:**
- **Status:** ✅ PASSED (2 tests)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
  ```

**With 6u_cluster_desc.yaml:**
- **Status:** ✅ PASSED (2 tests)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
  ```

**With 2x2_n300_cluster_desc.yaml:**
- **Status:** ⚠️ NOT TESTED (listed in workflow but not run yet)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"
  ```

#### 3. Topology Mapper Tests
- **Filter:** `T3kTopologyMapperCustomMapping/*`
- **Mock Cluster Required:** Yes (t3k_cluster_desc.yaml)
- **Status:** ✅ PASSED (12 tests)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="T3kTopologyMapperCustomMapping/*"
  ```

- **Filter:** `TopologyMapperTest.T3kMeshGraphTest*`
- **Mock Cluster Required:** Yes (t3k_cluster_desc.yaml)
- **Status:** ✅ PASSED (2 tests)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.T3kMeshGraphTest*"
  ```

#### 4. MeshGraphDescriptorTests
- **Filter:** `MeshGraphDescriptorTests*`
- **Mock Cluster Required:** No (pure unit tests)
- **Status:** ✅ PASSED (41 tests)
- **Command:**
  ```bash
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests*"
  ```

#### 5. TopologySolverTest
- **Filter:** `TopologySolverTest.*`
- **Mock Cluster Required:** No (pure unit tests)
- **Status:** ✅ PASSED (93 tests)
- **Command:**
  ```bash
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
  ```

#### 6. TopologyMapperUtilsTest
- **Filter:** `TopologyMapperUtilsTest.*`
- **Mock Cluster Required:** No (pure unit tests)
- **Status:** ✅ PASSED
- **Command:**
  ```bash
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.*"
  ```

#### 7. RoutingTableValidation Tests
- **Filter:** `RoutingTableValidation*`
- **Mock Cluster Required:** Yes (6u_cluster_desc.yaml)
- **Status:** ✅ PASSED (4 tests)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"
  ```

#### 8. ControlPlaneFixture Tests
- **Mock Cluster Required:** Yes (various descriptors)
- **Status:** ✅ PASSED (multiple variants)

**SingleGalaxy:**
- **Mock Cluster:** 6u_cluster_desc.yaml
- **Status:** ✅ PASSED (4 passed, 2 skipped)
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*SingleGalaxy*
  ```

**T3k:**
- **Mock Cluster:** t3k_cluster_desc.yaml
- **Status:** ✅ PASSED
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
  ```

**T3kCustomMeshGraph:**
- **Mock Cluster:** t3k_cluster_desc.yaml
- **Status:** ✅ PASSED
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
  ```

**Custom2x2:**
- **Mock Cluster:** 2x2_n300_cluster_desc.yaml
- **Status:** ✅ PASSED
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*Custom2x2*
  ```

**TestControlPlaneInitNoMGD:**
- **Mock Clusters:** Multiple (2xp150_disconnected, 4xn300_disconnected, bh_galaxy_xyz)
- **Status:** ✅ PASSED
- **Commands:**
  ```bash
  # With 2xp150_disconnected_cluster_desc.yaml
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2xp150_disconnected_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD

  # With 4xn300_disconnected_cluster_desc.yaml
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/4xn300_disconnected_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD

  # With bh_galaxy_xyz_cluster_desc.yaml
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestControlPlaneInitNoMGD
  ```

### ⚠️ PARTIALLY WORKING

#### 1. Link Retraining Tests
- **Test Executable:** `./build_Debug/test/tools/scaleout/test_link_retraining`
- **Mock Cluster Required:** Yes (t3k_cluster_desc.yaml)
- **Status:** ⚠️ PARTIALLY WORKING
- **Issue:** Tests initialize successfully but fail during validation with motherboard mismatch error:
  ```
  Motherboard mismatch between FSD and GSD for host g14glx03: FSD=X12DPG-QT6, GSD=S7T-MB
  ```
- **Available Tests:**
  - `DirectedRetrainingFixture.TestActiveEthRetraining` - FAILS
  - `DirectedRetrainingFixture.TestOnDemandCableRestart` - FAILS
  - `DirectedRetrainingFixture.DISABLED_TestExitNodeRetraining` - DISABLED
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
    ./build_Debug/test/tools/scaleout/test_link_retraining --gtest_filter="*"
  ```
- **Note:** The test is hardcoded to use `tools/tests/scaleout/cabling_descriptors/t3k.textproto` which expects a different motherboard type than what's in the mock cluster descriptor.

### ✅ WORKING Tools

#### 1. Cluster Validation Tool
- **Executable:** `./build_Debug/tools/scaleout/run_cluster_validation`
- **Mock Cluster Required:** Yes
- **Status:** ✅ WORKING
- **Features Tested:**
  - Physical discovery - ✅ Works
  - Host detection - ✅ Works
  - Link health validation - ✅ Works
  - Connectivity printing - ✅ Works
- **Command:**
  ```bash
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
    ./build_Debug/tools/scaleout/run_cluster_validation --print-connectivity
  ```
- **Note:** Works well with `6u_cluster_desc.yaml`. Some mock clusters (e.g., `2x2_n300_cluster_desc.yaml`) may crash with "Asic channel not found" error.

### ❌ NOT TESTED (Require Multi-Host Setup)

The following tests from workflows require `tt-run` and MPI setup for multi-host testing:

- MultiHost.* tests (various configurations)
- PhysicalDiscovery.* tests (multi-host)
- TestSystemHealth tests (multi-host)
- Various multi-host fabric sanity tests

These tests are listed in `.github/workflows/fabric-cpu-only-tests-impl.yaml` lines 80-176 but require:
- `tt-run` utility
- MPI (Message Passing Interface)
- Multiple host simulation

## Test Coverage Summary

| Category | Total Tests | Passed | Failed | Not Tested |
|----------|-------------|--------|--------|------------|
| Pure Unit Tests (no mock) | ~154 | 154 | 0 | 0 |
| Single-Host Mock Cluster Tests | ~30+ | 30+ | 0 | 0 |
| Link Retraining Tests | 2 | 0 | 2 | 1 (disabled) |
| Tools (run_cluster_validation) | 1 | 1 | 0 | 0 |
| Multi-Host Tests | ~50+ | N/A | N/A | 50+ |

## Mock Cluster Descriptors Used

1. ✅ `t3k_cluster_desc.yaml` - Works well
2. ✅ `6u_cluster_desc.yaml` - Works well
3. ⚠️ `2x2_n300_cluster_desc.yaml` - Works for some tests, crashes cluster validation tool
4. ✅ `2xp150_disconnected_cluster_desc.yaml` - Works
5. ✅ `4xn300_disconnected_cluster_desc.yaml` - Works
6. ✅ `bh_galaxy_xyz_cluster_desc.yaml` - Works

## Notes

- All tests require `TT_METAL_HOME` environment variable to be set
- Some tests require `TT_METAL_SLOW_DISPATCH_MODE=1` for proper execution
- Mock cluster descriptors are located in `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/`
- Build directory used: `build_Debug`

## Running All Single-Host Mock Cluster Tests

To run all single-host tests that work with mock clusters:

```bash
export TT_METAL_HOME=$(pwd)

# Pure unit tests (no mock cluster needed)
./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTopologyHelpers*"
./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests*"
./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologySolverTest.*"
./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperUtilsTest.*"

# Mock cluster tests
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MockClusterTopologyFixture*"

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="RoutingTableValidation*"

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="T3kTopologyMapperCustomMapping/*"

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="TopologyMapperTest.T3kMeshGraphTest*"

# ControlPlane tests
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*SingleGalaxy*

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*

TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*Custom2x2*

# Cluster validation tool
TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
  ./build_Debug/tools/scaleout/run_cluster_validation --print-connectivity
```
