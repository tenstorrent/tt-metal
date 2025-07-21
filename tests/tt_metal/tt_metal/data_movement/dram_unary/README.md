# DRAM Unary Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between DRAM and a single Tensix core.

## Test Flow
Data is loaded into an L1 circular buffer by the reader kernel and written out to DRAM from the same L1 buffer by the writer kernel. A compute kernel is omitted as these tests are purely for data movement.
Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler.
Resulting data is cross-checked with original data and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| transaction_size_pages    | uint32_t              | Size of the issued noc transactions in pages. |
| page_size_bytes           | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| cores                     | CoreRangeSet          | Tensix cores that are used for the transactions. |
| tensor_shape_in_pages     | array<uint32_t, 2>    | 2D-Shape of an arbitrary tensor in pages. Used mainly for sharding. |
| num_dram_banks            | array<uint32_t, 2>    | Number of DRAM banks, for two dimensions, that will be used for sharding. |

## Test Cases
Three different test cases are implemented using the above parameters.
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. DRAM Packet Sizes: Tests different number of transactions and transaction sizes by varying the num_of_transactions and pages_per_transaction parameters.
2. DRAM Core Locations: Tests different cores on device (core grid dependent on architecture) by varying the core_coord parameter.
3. DRAM Channels: Tests different DRAM channels by varying the dram_channel parameter.
4. DRAM Directed Ideal: Tests the most optimal transaction between two cores that maximizes the transaction size and performs enough transactions to amortize initialization overhead.
