# One to One Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between two Tensix cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on two Tensix cores: one sender and one receiver core. Data is written into the L1 memory on the sender core. The sender kernel issues NOC transactions to transfer this data into the L1 memory on the receiver core. Once data is transferred, the sender kernel uses hardware barriers to ensure data validity and completion of the transaction.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

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
| num_virtual_channels      | uint32_t              | Number of virtual channels to cycle through (must be > 1 for cycling). |
| noc_id                    | NOC                   | Specify which NOC to use for the test. |
| use_2_0_api               | bool                  | Determines if the test uses the experimental device 2.0 API. |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **TensixDataMovementOneToOnePacketSizes** (Test ID: 4) - Tests different number of transactions and transaction sizes by varying the num_of_transactions and pages_per_transaction parameters. Sweeps through multiple combinations to test various packet configurations.

2. **TensixDataMovementOneToOneDirectedIdeal** (Test ID: 50) - Tests the most optimal data movement setup between two cores that maximizes the transaction size and performs enough transactions to amortize initialization overhead. Uses neighboring cores (0,0) → (0,1) to minimize latency.

3. **TensixDataMovementOneToOneVirtualChannels** (Test ID: 150) - *[Currently Skipped]* Tests virtual channel functionality by cycling through multiple NOCs (NOC_0, NOC_1), transaction sizes, and virtual channels (1-4). Validates that different virtual channels can be used effectively for data movement.

4. **TensixDataMovementOneToOneCustom** (Test ID: 151) - *[Currently Skipped]* Custom test case with configurable parameters for specialized testing scenarios. Uses 256 transactions, 1 page per transaction, and 4 virtual channels.

5. **TensixDataMovementOneToOnePacketSizes2_0** (Test ID: 158) - Device 2.0 API version of the packet sizes test. Tests the same packet size variations as test ID 4 but uses the experimental NOC API with structured endpoints and virtual channel support.

## Device 2.0 API Tests
This test suite now includes tests using the new device 2.0 experimental NOC API. These tests provide the same functionality as the original tests but use an updated API design:

### Key Features of Device 2.0 API Tests:
- **Experimental NOC API**: Uses `experimental::Noc` and `experimental::UnicastEndpoint` for structured NOC operations
- **Structured Arguments**: Source and destination arguments are defined using structured `noc_traits_t` types

### Device 2.0 Kernels:
- `sender_2_0.cpp`: Implements the sender functionality using the experimental NOC API
- `sender.cpp`: Original sender kernel for comparison

Both API versions run the same test cases but use different underlying implementations. The device 2.0 tests serve as a validation and performance comparison for the new experimental API.
