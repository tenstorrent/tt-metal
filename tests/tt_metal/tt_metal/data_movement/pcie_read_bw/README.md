# PCIe Read Bandwidth Tests

This test suite implements a test that measures the functionality and performance (i.e. bandwidth) of data movement transactions from Host(PCIe memory) to L1 memory on a single Tensix core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
The master core issues NOC read transactions to retrieve data from PCIe memory. Data is read from PCIe memory into the L1 using `noc_async_read`. The test measures the time taken and calculates the bandwidth. Results are logged with detailed performance metrics.

Test attributes such as transaction sizes and number of transactions, as well as latency measures like kernel and pre-determined scope cycles, are recorded by the profiler. DeviceTimestampedData is recorded for Python wrapper compatibility. The host test queries the real device clock frequency via `device->get_clock_rate_mhz()` and passes it to the kernel, which logs it for GB/s conversion in the reporting pipeline.

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

1. PCIe Read Bandwidth (ID 603): Tests PCIe read bandwidth with the maximum possible bandwidth on the selected device.
2. PCIe Read Bandwidth Sweep (ID 605): Sweeps transaction sizes from flit size up to the NOC max packet size (8 kB on Wormhole, 16 kB on Blackhole) with 1M transactions at each size. Transaction sizes increase by powers of 2.
3. PCIe Host Read (D2H) Bandwidth Sweep (ID 607): Host-side bandwidth test using `distributed::ReadShard`. Sweeps buffer sizes from 4 KB to 16 MB, timed with `std::chrono`. No kernel involved — measures the full dispatch path including command queue and DMA overhead.

**Note**: The original test (ID 603) is based on the `test_bw_and_latency.cpp` test from the performance microbenchmark dispatch suite, specifically the PCIe read functionality (`-m 0`).
