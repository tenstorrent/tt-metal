# Core Bidirectional Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of bidirectional data movement transactions between two Tensix cores where one core simultaneously sends and receives data.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on two Tensix cores: one master core and one subordinate core. The master core can run either a single kernel that performs both sending and receiving operations, or separate sender and receiver kernels. Data is written into both cores' L1 memory. The kernels perform simultaneous NOC write and read transactions between the two cores. Hardware barriers ensure data validity and completion of all transactions.

Test attributes such as transaction sizes, number of transactions, and virtual channel configuration are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*CoreBidirectional*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord             | CoreCoord             | Logical coordinates for the master core that performs bidirectional operations. |
| subordinate_core_coord        | CoreCoord             | Logical coordinates for the subordinate core that participates in bidirectional operations. |
| num_of_transactions           | uint32_t              | Number of transaction pairs (send + receive) to perform. |
| pages_per_transaction         | uint32_t              | Number of pages per individual transaction. |
| bytes_per_page                | uint32_t              | Size of each page in bytes. |
| l1_data_format                | DataFormat            | Type of data in transaction (typically Float16_b). |
| write_vc                      | uint32_t              | Virtual channel to use for write operations (0-3). |
| same_kernel                   | bool                  | True if using a single kernel for both send/receive, false for separate kernels. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **Directed Ideal Same Kernel**: Tests optimal bidirectional transactions using a single kernel that performs both send and receive operations.

2. **Directed Ideal Different Kernels**: Tests optimal bidirectional transactions using separate sender and receiver kernels.

3. **Same VC Same Kernel**: Tests bidirectional operations where both send and receive use the same virtual channel with a single kernel.

4. **Same VC Different Kernels**: Tests bidirectional operations where both send and receive use the same virtual channel with separate kernels.

5. **Write VC Sweep Same Kernel**: Tests bidirectional operations sweeping through different virtual channels (0-3) for write operations using a single kernel.

6. **Write VC Sweep Different Kernels**: Tests bidirectional operations sweeping through different virtual channels (0-3) for write operations using separate kernels.

7. **Packet Sizes Same Kernel**: Tests bidirectional operations over varying transaction sizes using a single kernel.

8. **Packet Sizes Different Kernels**: Tests bidirectional operations over varying transaction sizes using separate kernels.

9. **Custom Test**: Configurable test case using a single kernel with write VC 0 for specific parameter testing.
