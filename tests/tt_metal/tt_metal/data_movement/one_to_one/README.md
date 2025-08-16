# One to One Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between two Tensix cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on two Tensix cores: one sender and one receiver core. Data is written into the L1 memory on the sender core. The sender kernel issues NOC transactions to transfer this data into the L1 memory on the receiver core. Once data is transferred, the sender kernel uses hardware barriers to ensure data validity and completion of the transaction.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through a pcc check.

Test expectations are that pcc checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*OneToOne*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord         | CoreCoord             | Logical coordinates for the sender core. |
| subordinate_core_coord    | CoreCoord             | Logical coordinates for the receiver core. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| pages_per_transaction     | uint32_t              | Size of the issued noc transactions in pages. |
| bytes_per_page            | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| virtual_channel           | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc                       | N/A                   | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|
| posted                    | N/A                   | Posted flag. Determines if write is posted or non-posted (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. One to One Packet Sizes: Tests different number of transactions and transaction sizes by varying the num_of_transactions and transaction_size_pages parameters.
2. One to One Directed Ideal: Tests the most optimal data movement setup between two cores that maximizes the transaction size and performs enough transactions to amortize initialization overhead. This test is performed between neigbouring cores on a torus to minimize latency.
