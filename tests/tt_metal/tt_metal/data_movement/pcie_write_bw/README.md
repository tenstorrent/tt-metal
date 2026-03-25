# PCIe Write Bandwidth Tests

This test suite measures the bandwidth of data movement transactions from L1 memory on a single Tensix core to Host (PCIe memory).

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
The master core issues NOC write transactions to push data from L1 into PCIe memory using `noc_async_write`. The test measures the time taken and calculates the bandwidth. Results are logged with detailed performance metrics.

Test attributes such as transaction sizes and number of transactions, as well as latency measures like kernel and pre-determined scope cycles, are recorded by the profiler. DeviceTimestampedData is recorded for Python wrapper compatibility. The host test queries the real device clock frequency via `device->get_clock_rate_mhz()` and passes it to the kernel, which logs it for GB/s conversion in the reporting pipeline.

Test expectations are that sufficient test attribute data is captured by the profiler for higher-level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*PCIeWriteBandwidth*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord         | CoreCoord             | Logical coordinates for the master core. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| bytes_per_transaction     | uint32_t              | Size of each issued noc transaction in bytes. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| noc_id                    | NOC                   | NOC to use for data movement. |

## Test Cases
Test case uses Float32 as L1 data format.

1. PCIe Write Bandwidth Sweep (ID 604): Sweeps transaction sizes from flit size up to the NOC max packet size (8 kB on Wormhole, 16 kB on Blackhole) with 1M transactions at each size. Transaction sizes increase by powers of 2.
