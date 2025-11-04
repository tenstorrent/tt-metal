# PCIe Read Bandwidth Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions from Host(PCIe memory) to L1 memory on a single Tensix core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
A circular buffer is allocated in L1 memory on a worker core. The worker core issues NOC read transactions to retrieve data from PCIe memory. Data is read from PCIe memory into the L1 circular buffer using `noc_async_read`. The kernel executes 1000 iterations, each reading 4 pages of 65536 bytes. The test measures the time taken and calculates bandwidth. Results are logged with detailed performance metrics.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. DeviceTimestampedData is recorded for Python wrapper compatibility.

Test expectations are that sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*PCIeReadBandwidth*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| worker_core_coord         | CoreCoord             | Logical coordinates for the worker core. |
| iterations                | uint32_t              | Number of test iterations to run. |
| warmup_iterations         | uint32_t              | Number of warmup iterations before timing. |
| page_size_bytes           | uint32_t              | Size of each page in bytes. Fixed at 65536 bytes. |
| batch_size_k              | uint32_t              | Batch size in KB. Fixed at 256. |
| page_count                | uint32_t              | Number of pages to transfer per iteration. Calculated as batch_size_k / page_size_bytes. |
| l1_data_format            | DataFormat            | Data format for L1 memory. |
| noc_id                    | NOC                   | NOC to use for data movement. |

## Test Cases
Test case uses Float32 as L1 data format and 65536 bytes as page size.

1. PCIe Read Bandwidth: Tests PCIe read bandwidth with fixed parameters (batch size 256K, page size 65536 bytes, 1000 iterations).

**Note**: This test is based on the `test_bw_and_latency.cpp` test from the performance microbenchmark dispatch suite, specifically the PCIe read functionality (`-m 0`).
