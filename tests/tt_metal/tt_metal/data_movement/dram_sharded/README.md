# DRAM Sharded Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement reads from sharded DRAM buffer into a Tensix core using the stateful one_packet APIs.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Runs the reader kernel on a Tensix core. After initializing a sharded DRAM buffer with data, the kernel issues NOC instructions to read this data into L1 memory using the stateful one_packet APIs. A read barrier is placed after these transactions to ensure data validity.

Test attributes such as number of DRAM banks to shard into and number of tiles per bank, as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

An additional flag can be set to use a stateful read with transaction IDs, for testing the functionality of those APIs.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*TensixDataMovementDRAMShardedRead*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of NOC transactions. |
| num_banks                 | uint32_t              | Number of DRAM banks to shard. |
| pages_per_bank            | uint32_t              | Number of pages of data per bank. |
| page_size_bytes           | uint32_t              | Size of a page in bytes. |
| l1_data_format            | DataFormat            | Type of data in transaction. |
| cores                     | CoreRangeSet          | Tensix core that the kernel is run on. |
| use_trid                  | bool                  | Whether to use transaction IDs. |
| num_of_trids              | uint32_t              | Number of transaction IDs to use. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case uses a 32 x 32 datum tile as a page.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. DRAM Sharded Read Directed Ideal: Tests the most optimal transactions for reading packets by maximizing the number of pages, banks, and transactions to amortize initialization overhead and saturate the bandwidth.
2. DRAM Sharded Read Tile Numbers: Tests reading over varying number of pages per DRAM bank.
3. DRAM Sharded Read Bank Numbers: Tests reading over varying numbers of DRAM banks.
4. DRAM Sharded Read Trid Directed Ideal: Tests reading from DRAM sharded with transaction IDs.
