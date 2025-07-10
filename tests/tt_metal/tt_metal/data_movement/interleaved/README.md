# Interleaved Tile Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between tile interleaved buffers (DRAM or L1) and a single Tensix core.

## Test Flow
Runs either the reader kernel, the writer kernel, or both.

If both kernels are run, data is loaded into an input interleaved buffer and the reader kernel reads this data into L1 memory on the tensix core. The writer kernel writes this data from the L1 memory into the output interleaved buffer. The kernels transfer data one tile at a time.

If only the reader kernel is run, we directly look at the L1 memory read by the kernel as the output, and if only the writer kernel is run, we directly initialize the input into L1 memory for the kernel to use.

Test attributes such as transaction sizes as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler.
Resulting data is cross-checked with original data and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_tiles                 | uint32_t              | Number of tiles (32 x 32 datums) in the transaction. |
| tile_size_bytes           | uint32_t              | Size of a tile in bytes (32 x 32 x size of the data type in bytes). |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| cores                     | CoreRangeSet          | Tensix cores that are used for the transactions. |
| is_dram                   | bool                  | True if the buffers are interleaved in DRAM, false if the buffers are interleaved in L1. |
| read_kernel               | bool                  | True if the reader kernel is enabled for the test. |
| write_kernel              | bool                  | True if the writer kernel is enabled for the test. |

## Test Cases
Each test case uses bfloat16 as L1 data format and a tile size of 32 x 32 datums.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. DRAM Interleaved Tile Numbers: Tests both tile reading and tile writing over a varying number of tiles with interleaved DRAM buffers.
2. DRAM Interleaved Core Locations: Tests different cores on device (core grid dependent on architecture) by varying the cores parameter.
3. DRAM Interleaved Tile Read Numbers: Tests only tile reading over a varying number of tiles with interleaved DRAM buffers.
4. DRAM Interleaved Tile Write Numbers: Tests only tile writing over a varying number of tiles with interleaved DRAM buffers.
5. DRAM Directed Ideal: Tests the most optimal transactions for reading tiles to and from interleaved DRAM buffers by maximizing the number of tiles to amortize initialization overhead.

Each DRAM test has a corresponding L1 test, where the buffers are interleaved over L1.
