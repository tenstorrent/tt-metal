# Golden ASIC Mapping Files

This directory contains golden reference files for ASIC to Fabric node ID mappings generated during control plane initialization tests.

## Purpose

These golden files are used to verify that the ASIC mapping generation remains consistent across test runs. After each control plane initialization test, the generated mapping files are compared against these golden files to detect any regressions or unexpected changes.

## Generating Golden Files

**IMPORTANT**: Before regenerating golden files:
1. Verify this is an intentional change, not a regression
2. Get approval from topology users and scaleout team + Umair Cheema, Aditya Saigal, Allan Liu, Joseph Chu, Ridvan Song
3. Ensure all affected topologies are tested

### Single-Host Tests

For single-host tests, run the test with appropriate environment variables and copy the generated file:

1. **Run the test** with `TT_METAL_SLOW_DISPATCH_MODE=1` and the appropriate cluster descriptor:
   ```bash
   TT_METAL_SLOW_DISPATCH_MODE=1 \
   TT_METAL_MOCK_CLUSTER_DESC_PATH=<cluster_desc> \
   [TT_MESH_GRAPH_DESC_PATH=<mesh_graph_desc>] \
   ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=<TestName>
   ```

2. **Copy rank 1's generated file** (always `rank_1_of_1` for single-host):
   ```bash
   cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml \
      tests/tt_metal/tt_fabric/golden_mapping_files/<GoldenFileName>.yaml
   ```

**Examples:**

- `ControlPlaneFixture_SingleGalaxy`:
  ```bash
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*SingleGalaxy*"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/ControlPlaneFixture_SingleGalaxy.yaml
  ```

- `ControlPlaneFixture_T3k`:
  ```bash
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*T3k*"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/ControlPlaneFixture_T3k.yaml
  ```

- `ControlPlaneFixture_Custom2x2`:
  ```bash
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.*Custom2x2*"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/ControlPlaneFixture_Custom2x2.yaml
  ```

- `TestControlPlaneInitNoMGD` (various cluster descriptors):
  ```bash
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/<cluster_desc>.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestControlPlaneInitNoMGD"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestControlPlaneInitNoMGD.yaml
  ```

### Multi-Host Tests

For multi-host tests, use `tt-run` with the appropriate cluster and rank binding configurations:

1. **Run the test** with `tt-run`:
   ```bash
   tt-run --mock-cluster-rank-binding <cluster_desc_mapping> \
          --rank-binding <rank_bindings> \
          --mpi-args "--tag-output --allow-run-as-root" \
          ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
          --gtest_filter=<TestName>
   ```

2. **Copy rank 1's generated file** (world_size is the number of ranks):
   ```bash
   cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_<world_size>.yaml \
      tests/tt_metal/tt_fabric/golden_mapping_files/<GoldenFileName>.yaml
   ```

**Examples:**

- `TestDualGalaxyControlPlaneInit` (world_size=2):
  ```bash
  tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml \
         --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml \
         --mpi-args "--tag-output --allow-run-as-root" \
         ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
         --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_2.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestDualGalaxyControlPlaneInit.yaml
  ```

- `TestDual2x4ControlPlaneInit` (world_size=2):
  ```bash
  tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml \
         --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml \
         --mpi-args "--tag-output --allow-run-as-root" \
         ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
         --gtest_filter="MultiHost.TestDual2x4ControlPlaneInit"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_2.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestDual2x4ControlPlaneInit.yaml
  ```

- `TestBigMesh2x4ControlPlaneInit` (world_size=4):
  ```bash
  tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/big_mesh_2x4_cluster_desc_mapping.yaml \
         --rank-binding tests/tt_metal/distributed/config/big_mesh_2x4_rank_bindings.yaml \
         --mpi-args "--tag-output --allow-run-as-root" \
         ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
         --gtest_filter="MultiHost.TestBigMesh2x4ControlPlaneInit"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_4.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestBigMesh2x4ControlPlaneInit.yaml
  ```

- `TestBHQB4x4ControlPlaneInit` (world_size=4):
  ```bash
  tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bhqb_4x4_cluster_desc_mapping.yaml \
         --rank-binding tests/tt_metal/distributed/config/bhqb_4x4_rank_bindings.yaml \
         --mpi-args "--tag-output --allow-run-as-root" \
         ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
         --gtest_filter="MultiHost.TestBHQB4x4ControlPlaneInit"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_4.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestBHQB4x4ControlPlaneInit.yaml
  ```

- `TestClosetBox3PodTTSwitchControlPlaneInit` (world_size=3):
  ```bash
  tt-run --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/closet_box_3pod_ttswitch_cluster_desc_mapping.yaml \
         --rank-binding tests/tt_metal/distributed/config/closet_box_3pod_ttswitch_rank_bindings.yaml \
         --mpi-args "--tag-output --allow-run-as-root" \
         ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
         --gtest_filter="MultiHost.TestClosetBox3PodTTSwitchControlPlaneInit"

  cp generated/fabric/asic_to_fabric_node_mapping_rank_1_of_3.yaml \
     tests/tt_metal/tt_fabric/golden_mapping_files/TestClosetBox3PodTTSwitchControlPlaneInit.yaml
  ```

3. **Verify the changes** by reviewing the diff:
   ```bash
   git diff tests/tt_metal/tt_fabric/golden_mapping_files/<GoldenFileName>.yaml
   ```

4. **Commit the updated golden file** along with your changes

## Test Coverage

The following tests generate and compare ASIC mapping files:

### Single-Host Tests
- `ControlPlaneFixture_SingleGalaxy`
- `ControlPlaneFixture_T3k`
- `ControlPlaneFixture_Custom2x2`
- `TestControlPlaneInitNoMGD` (various cluster descriptors)
- `T3kCustomMeshGraphControlPlaneTests*`

### Multi-Host Tests
- `TestDualGalaxyControlPlaneInit` (2 hosts)
- `TestDual2x4ControlPlaneInit` (2 hosts)
- `TestBigMesh2x4ControlPlaneInit` (4 hosts)
- `TestBHQB4x4ControlPlaneInit` (4 hosts)
- `TestClosetBox3PodTTSwitchControlPlaneInit` (3 hosts)

## File Naming Convention

Golden files are named using the pattern:
```
{TestName}.yaml
```

Only rank 1's file is saved as a golden reference, since all ranks should have the same mapping. For example:
- `ControlPlaneFixture_SingleGalaxy.yaml`
- `TestDualGalaxyControlPlaneInit.yaml`
- `TestDual2x4ControlPlaneInit.yaml`

## Implementation Details

The comparison logic is implemented in:
- `tests/tt_metal/tt_fabric/common/utils.cpp`: `compare_asic_mapping_files()` and `check_asic_mapping_against_golden()`
- `tests/tt_metal/tt_fabric/fabric_router/test_multi_host.cpp`: Multi-host version of `check_asic_mapping_against_golden()`

The comparison is order-independent and validates:
- ASIC IDs
- Tray IDs and ASIC locations
- Fabric node IDs (mesh_id and chip_id)
