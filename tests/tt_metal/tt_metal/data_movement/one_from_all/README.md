# One From All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between a gatherer Tensix core and a grid of sender Tensix cores.

## Test Flow
Data is written directly into the L1 memory of the sender cores. The gatherer kernel issues read NOC transactions to request a transfer of this data from each sender core to its L1 memory. A read barrier is placed after these transactions in order to ensure data validity.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter              | Data Type    | Description |
| ---------------------- | ------------ | ----------- |
| test_id                | uint32_t     | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions    | uint32_t     | Number of noc transactions/calls that will be issued. |
| transaction_size_pages | uint32_t     | Size of the issued noc transactions in pages. |
| page_size_bytes        | uint32_t     | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format         | DataFormat   | Data format data that will be moved. |
| master_core_coord      | CoreCoord    | Logical coordinates for the gatherer core. |
| subordinate_core_set   | CoreRangeSet | Set of logical coordinates for the sender cores. |
| virtual_channel        | N/A          | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc                    | N/A          | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. One From All Packet Sizes: Tests different number of transactions and transaction sizes by varying the num_of_transactions and transaction_size_pages parameters. Sweeps values that are multiples of 2 (including 1) in the range [1, 64] for both parameters (skips cases where total data size exceeds L1 capacity). Uses master logical coordinate (0,0) and a grid of 16 subordinate cores: (1,1)-(4,4).
2. One From All Directed Ideal: Tests optimal data movement setup between master logical coordinate (0,0) and a grid of 16 subordinate cores (1,1)-(4,4). Uses large transaction size and number of transactions to amortize initialization overhead and maximize bandwidth utilization.
