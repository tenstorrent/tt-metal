# One From All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions from all subordinate cores to one master core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on multiple Tensix cores: multiple subordinate cores (senders) and one master core (receiver). Data is written into L1 memory on all subordinate cores. The master core issues NOC read transactions to retrieve data from all subordinate cores' L1 memory. Once data is transferred from all subordinate cores, the master core uses hardware barriers to ensure data validity and completion of the transaction.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*OneFromAll*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord             | CoreCoord             | Logical coordinates for the master (receiver) core. |
| subordinate_core_set          | CoreRangeSet          | Set of logical coordinates for all subordinate (sender) cores. |
| num_of_transactions           | uint32_t              | Number of noc transactions that will be issued. |
| transaction_size_pages        | uint32_t              | Size of the issued noc transactions in pages. |
| page_size_bytes               | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format                | DataFormat            | Data format of data that will be moved. |
| virtual_channel               | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc                           | N/A                   | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **One From All Packet Sizes**: Tests different number of transactions and transaction sizes by varying the num_of_transactions and pages_per_transaction parameters. Multiple subordinate cores send data to a single master core.

2. **One From All Directed Ideal**: Tests the most optimal data movement setup from multiple subordinate cores to one master core that maximizes the transaction size and performs enough transactions to amortize initialization overhead. This test uses neighboring cores to minimize latency.
