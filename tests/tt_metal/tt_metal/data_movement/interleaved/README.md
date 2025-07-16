# Interleaved Tile Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between DRAM or L1 interleaved buffers and a Tensix core using `noc_async_*_tile` APIs.

## Test Flow
Runs a reader kernel and/or a writer kernel on a tensix core.

The reader kernel issues NOC instructions to read data from an interleaved buffer, initialized with data, into the L1 address of the Tensix core calling the read_tile API. A read barrier is placed after these transactions in order to ensure data validity. If the reader kernel isn't run, this data is initially written directly into L1 memory.

The writer kernel issues NOC instructions to write data from the L1 address of the Tensix core into an interleaved buffer. A write barrier is placed after these transactions in order to ensure data validity.

Each transaction consists of 16 tiles so that data is interleaved over all banks. Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is read from the output interleaved buffer if the writer kernel is run (or directly from L1 memory otherwise), cross-checked with original data, and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions         | uint32_t             | Number of transactions in test. |
| num_tiles    | uint32_t             | Size of input data in tiles, and the number of tile read/writes per transaction. |
| tile_size_bytes         | uint32_t              | Size of a tile in bytes (32 x 32 x size of datatype). |
| l1_data_format                      | DataFormat                  | Type of data in transaction. |
| cores                      | CoreRangeSet                  | Logical coordinates of Tensix core running kernels. |
| is_dram                      | bool                  | True if buffer is interleaved in DRAM, false if interleaved in L1. |
| read_kernel                      | bool                  | True if test runs reader kernel. |
| write_kernel                      | bool                  | True if test runs writer kernel. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. DRAM Interleaved Tile Numbers: Tests tile reading and writing with interleaved DRAM buffer over varying number of transactions.
2. DRAM Interleaved Tile Core Locations: Tests tile reading and writing with interleaved DRAM buffer over varying Tensix core coordinates.
3. DRAM Interleaved Tile Read Numbers: Tests tile reading only with interleaved DRAM buffer over varying number of transactions.
4. DRAM Interleaved Tile Write Numbers: Tests tile writing only with interleaved DRAM buffer over varying number of transactions.
5. DRAM Interleaved Tile Directed Ideal: Tests the most optimal transactions for reading/writing tiles by maximizing the number of tiles to amortize initialization overhead and saturate the bandwidth.

Each DRAM test has a corresponding test case using L1 interleaved buffers.
