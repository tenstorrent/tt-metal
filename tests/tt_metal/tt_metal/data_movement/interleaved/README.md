# Interleaved Page Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between DRAM or L1 interleaved buffers and a Tensix core using `noc_async_*_page` APIs.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Runs a reader kernel and/or a writer kernel on a tensix core.

The reader kernel issues NOC instructions to read data from an interleaved buffer, initialized with data, into the L1 address of the Tensix core calling the read_page API. A read barrier is placed after these transactions in order to ensure data validity. If the reader kernel isn't run, this data is initially written directly into L1 memory.

The writer kernel issues NOC instructions to write data from the L1 address of the Tensix core into an interleaved buffer. A write barrier is placed after these transactions in order to ensure data validity.

Transactions exceeding 16 pages will consecutively overwrite the same 16 pages so as not to take up excess L1 memory. Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is read from the output interleaved buffer if the writer kernel is run (or directly from L1 memory otherwise), cross-checked with original data, and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Interleaved*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of transactions in test. |
| num_pages                 | uint32_t              | Number of page read/writes per transaction. |
| page_size_bytes           | uint32_t              | Size of a page in bytes (max 1 packet - 16kB for BH, 8kB for WH). |
| l1_data_format            | DataFormat            | Type of data in transaction. |
| cores                     | CoreRangeSet          | Logical coordinates of Tensix core running kernels. |
| is_dram                   | bool                  | True if buffer is interleaved in DRAM, false if interleaved in L1. |
| read_kernel               | bool                  | True if test runs reader kernel. |
| write_kernel              | bool                  | True if test runs writer kernel. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. DRAM Interleaved Page Numbers: Tests page reading and writing with interleaved DRAM buffer over varying number of transactions and transaction sizes.
2. DRAM Interleaved Page Core Locations: Tests page reading and writing with interleaved DRAM buffer over varying Tensix core coordinates.
3. DRAM Interleaved Page Read Numbers: Tests page reading only with interleaved DRAM buffer over varying number of transactions and transaction sizes.
4. DRAM Interleaved Page Write Numbers: Tests page writing only with interleaved DRAM buffer over varying number of transactions and transaction sizes.
5. DRAM Interleaved Page Directed Ideal: Tests the most optimal transactions for reading/writing pages by maximizing the number of pages to amortize initialization overhead and saturate the bandwidth.
6. DRAM Interleaved Page Read Noc Swap: Same as Read Numbers, but uses RISC0, NOC0 instead for the reader kernel.
7. DRAM Interleaved Page Write Noc Swap: Same as Write Numbers, but uses RISC1, NOC1 instead for the writer kernel.

Each DRAM test has a corresponding test case using L1 interleaved buffers.
