# One to all Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between Tensix cores.

## Test Flow
Sharded L1 buffers are created on all Tensix cores: one sender and multiple receiver cores getting the same data. Data is written into the L1 buffer on the sender core. The sender kernel issues NOC transactions to transfer this data into the L1 buffer on the receiver core. Once data is transferred, the sender kernel signals to the receiver kernel that it is done by incrementing a semaphore. Receiver kernel waits on/polls this semaphore and completes its execution when it is incremented.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| transaction_size_pages    | uint32_t              | Size of the issued noc transactions in pages. |
| page_size_bytes           | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| master_core_coord         | CoreCoord             | Logical coordinates for the sender core. |
| grid_size                 | CoreCoord             | Grid size of the receiver cores, with origin at 0-0 |
| virtual_channel           | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc_id                    | N/A                   | Specify which NOC to use for the test |
| posted                    | N/A                   | Posted flag. Determines if write is posted or non-posted (TODO) |
| loopback                  | bool                  | Whether to include the sender core in the receiver core list. |
| is_multicast              | bool                  | Whether to do a multicast rather than sending to each core individually. |
| is_linked                 | bool                  | Whether multicast is linked. |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. One to All 2x2 Packet Sizes: Tests one to all on a 2x2 grid.
2. One to All 4x4 Packet Sizes: Tests one to all on a 4x4 grid.
3. One to All 10x10 Packet Sizes: Tests one to all on a 10x10 grid.
4. One to All Multicast 2x2 Packet Sizes: Tests one to all multicast on a 2x2 grid.
5. One to All Multicast 5x5 Packet Sizes: Tests one to all multicast on a 5x5 grid.
6. One to All Multicast 11x10 Packet Sizes: Tests one to all multicast on a 11x10 grid.
7. One to All Multicast Linked 2x2 Packet Sizes: Tests one to all linked multicast on a 2x2 grid.
8. One to All Multicast Linked 5x5 Packet Sizes: Tests one to all linked multicast on a 5x5 grid.
9. One to All Multicast Linked 11x10 Packet Sizes: Tests one to all linked multicast on a 11x10 grid.
10. One to All Directed Ideal: Tests the most optimal one to all linked multicast data movement setup. A high number of transactions is performed to amortize initialization overhead.
