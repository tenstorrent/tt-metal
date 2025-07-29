# Deinterleave Hardcoded Data Movement Tests

This test suite implements tests that measure the performance (i.e. bandwidth) of deinterleave transactions between Tensix cores.
They are based on kernel runtime arguments of existing metal tests.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow

This test creates a buffer for various deinterleave patterns as gotten from `pytest test_deinterleave.py`. It plots the bandwidth of each core.

It does not check pcc as the afformentioned test does this.

The deinterleave patterns are the exact ones as gotten from the base tests, as such this is a directed test is is not general.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*DeinterleaveHardcoded*"
```

## Test Parameters
| Parameter                 | Data Type                          | Description |
| ------------------------- | ---------------------              | ----------- |
| test_id                   | uint32_t                           | Test id for signifying different test cases. |
| dest_core_set             | std::vector<CoreRangeSet>          | Set of destination cores to which the data will be moved. |
| dest_core_compile_args    | std::vector<std::vector<uint32_t>> | Compile-time arguments for the destination core. |
| dest_core_runtime_args    | std::vector<std::vector<uint32_t>> | Runtime arguments for the destination core. |
| noc                       | N/A                                | Specify which NOC to use for the test |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. Deinterleave Single Core - Runs a single core deinterleave test with a specific pattern.
2. Deinterleave Multi Core - Runs a deinterleave test with multiple cores, using a specific pattern.
