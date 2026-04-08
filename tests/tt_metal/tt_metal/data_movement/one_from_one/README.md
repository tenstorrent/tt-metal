# One From One Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions from one subordinate core to one master core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on two Tensix cores: one subordinate core (sender) and one master core (receiver). Data is written into the L1 memory on the subordinate core. The master core issues NOC read transactions to retrieve data from the subordinate core's L1 memory. Once data is transferred, the master core uses hardware barriers to ensure data validity and completion of the transaction.

Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*OneFromOne*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| master_core_coord             | CoreCoord             | Logical coordinates for the master (receiver) core. |
| subordinate_core_coord        | CoreCoord             | Logical coordinates for the subordinate (sender) core. |
| num_of_transactions           | uint32_t              | Number of noc transactions that will be issued. |
| transaction_size_pages        | uint32_t              | Size of the issued noc transactions in pages. |
| page_size_bytes               | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format                | DataFormat            | Data format of data that will be moved. |
| virtual_channel               | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| noc                           | N/A                   | Specify which NOC to use for the test, (1) Use only one specified NOC, (2) Use both NOCs (TODO)|

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **TensixDataMovementOneFromOnePacketSizes** (Test ID: 5) - Tests different number of transactions and transaction sizes by varying the num_of_transactions and transaction_size_pages parameters. The master core reads data from the subordinate core using various packet configurations.

2. **TensixDataMovementOneFromOneDirectedIdeal** (Test ID: 51) - Tests the most optimal data movement setup from one subordinate core to one master core that maximizes the transaction size and performs enough transactions to amortize initialization overhead. Uses neighboring cores (0,0) ← (0,1) to minimize latency.

3. **TensixDataMovementOneFromOneVirtualChannels** (Test ID: 152) - *[Currently Skipped]* Tests virtual channel functionality by cycling through multiple NOCs (NOC_0, NOC_1), transaction sizes, and virtual channels (1-4). Validates that different virtual channels can be used effectively for read operations.

4. **TensixDataMovementOneFromOneCustom** (Test ID: 153) - *[Currently Skipped]* Custom test case with configurable parameters for specialized testing scenarios. Uses 256 transactions, 1 page per transaction, and 4 virtual channels for read operations.

5. **TensixDataMovementOneFromOnePacketSizes2_0** (Test ID: 159) - Device 2.0 API version of the packet sizes test. Tests the same packet size variations as test ID 5 but uses the experimental NOC API with structured endpoints and virtual channel support for async read operations.

## Device 2.0 API Tests
This test suite now includes tests using the new device 2.0 experimental NOC API. These tests provide the same functionality as the original tests but use an updated API design:

### Key Features of Device 2.0 API Tests:
- **Experimental NOC API**: Uses `experimental::Noc` and `experimental::UnicastEndpoint` for structured NOC operations
- **Structured Arguments**: Source and destination arguments are defined using structured `noc_traits_t` types

### Device 2.0 Kernels:
- `requestor_2_0.cpp`: Implements the requestor (master/receiver) functionality using the experimental NOC API with async read operations
- `requestor.cpp`: Original requestor kernel for comparison

Both API versions run the same test cases but use different underlying implementations. The device 2.0 tests serve as a validation and performance comparison for the new experimental API.
