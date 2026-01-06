# PCIe Read Bandwidth Tests

This test suite implements a test that measures the functionality and performance (i.e. bandwidth) of data movement transactions from Host(PCIe memory) to L1 memory on a single Tensix core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
The master core issues NOC read transactions to retrieve data from PCIe memory. Data is read from PCIe memory into the L1 using `noc_async_read`. The test measures the time taken and calculates the bandwidth. Results are logged with detailed performance metrics.

Test attributes such as transaction sizes and number of transactions, as well as latency measures like kernel and pre-determined scope cycles, are recorded by the profiler. DeviceTimestampedData is recorded for Python wrapper compatibility.

Test expectations are that sufficient test attribute data is captured by the profiler for higher-level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*PCIeReadBandwidth*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord         | CoreCoord             | Logical coordinates for the master core. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| pages_per_transaction     | uint32_t              | Size of the issued noc transactions in pages. |
| bytes_per_page            | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| noc_id                    | NOC                   | NOC to use for data movement. |

## Test Cases
Test case uses Float32 as L1 data format.

1. PCIe Read Bandwidth: Tests PCIe read bandwidth with the maximum possible bandwidth on the selected device.

**Note**: This test is based on the `test_bw_and_latency.cpp` test from the performance microbenchmark dispatch suite, specifically the PCIe read functionality (`-m 0`).
