# Loopback Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between Tensix cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on a single Tensix core with separate address offsets for input and output data. Data is written into the input L1 memory location. The sender kernel issues NOC transactions to transfer this data from the input L1 address to the output L1 address within the same core (loopback). Once data is transferred, the sender kernel uses hardware barriers to ensure data validity and completion of the transaction.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Loopback*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of noc transactions/calls that will be issued. |
| transaction_size_pages    | uint32_t              | Size of the issued noc transactions in pages. |
| page_size_bytes           | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format            | DataFormat            | Data format data that will be moved. |
| master_core_coord         | CoreCoord             | Logical coordinates for the sender core. |
| virtual_channel           | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc_id                    | N/A                   | Specify which NOC to use for the test |
| posted                    | N/A                   | Posted flag. Determines if write is posted or non-posted (TODO) |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. Loopback Packet Sizes: Tests loopback
2. Loopback Directed Ideal: Tests the most optimal data movement setup on the core itself that reduces the number of transactions while maximizing transaction size to minimize the effects of initialization overhead.
