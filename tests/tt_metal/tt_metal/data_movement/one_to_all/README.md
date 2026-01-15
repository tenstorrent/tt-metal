# One To All Core Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions from one master core to all subordinate cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
L1 memory is allocated on multiple Tensix cores: one master core (sender) and multiple subordinate cores (receivers). Data is written into the L1 memory on the master core. The master core issues NOC transactions (either unicast or multicast) to transfer its data to L1 memory on all subordinate cores. Once data is transferred to all subordinate cores, hardware barriers ensure data validity and completion of the transaction.

Test attributes such as transaction sizes, number of transactions, and grid configurations as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*OneToAll*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| mst_core_coord                | CoreCoord             | Logical coordinates for the master (sender) core. |
| sub_start_core_coord          | CoreCoord             | Starting logical coordinates for the subordinate core grid. |
| sub_grid_size                 | CoreCoord             | Size of the subordinate core grid. |
| num_of_transactions           | uint32_t              | Number of noc transactions that will be issued. |
| pages_per_transaction         | uint32_t              | Size of the issued noc transactions in pages. |
| bytes_per_page                | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format                | DataFormat            | Data format of data that will be moved. |
| noc_id                        | NOC                   | Specifies which NOC to use for the test. |
| loopback                      | bool                  | Determines if loopback (self-send) is included in the test. |
| is_multicast                  | bool                  | Determines if the test uses multicast or unicast NOC transactions. |
| is_linked                     | bool                  | Determines if linked/chained NOC transactions are used. |
| multicast_scheme_type         | uint32_t              | Specifies the multicast scheme type for advanced multicast tests. |
| virtual_channel               | N/A                   | (1) Option to specify unicast VC for each transaction, (2) Option for a sub-test that uses a separate VC for each transaction (TODO)|
| use_2_0_api                   | bool                  | Determines if the test uses the experimental device 2.0 API |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **One To All Unicast Tests**: Tests unicast data movement from one master core to various grid configurations:
   - 2x2 Packet Sizes: Master core sends to 2x2 grid of subordinate cores using unicast
   - 5x5 Packet Sizes: Master core sends to 5x5 grid of subordinate cores using unicast
   - All Packet Sizes: Master core sends to full compute grid using unicast

2. **One To All Multicast Tests**: Tests multicast data movement with various configurations:
   - Multiple multicast schemes and grid configurations
   - Support for linked/chained transactions
   - Support for loopback operations

3. **Advanced Multicast Schemes**: Tests different multicast implementation strategies for optimal performance across various grid sizes and communication patterns.

## Device 2.0 API Tests
This test suite now includes tests using the new device 2.0 experimental NOC API. These tests provide the same functionality as the original tests but use an updated API design.

### Key Features of Device 2.0 API Tests:
- **Experimental NOC API**: Uses `experimental::Noc`, `experimental::UnicastEndpoint` and `experimental::MulticastEndpoint` for structured NOC operations
- **Structured Arguments**: Source and destination arguments are defined using structured `noc_traits_t` types

### Device 2.0 Kernels:
- `sender_multicast_2_0.cpp`: Implements the sender functionality using the experimental NOC API with multicast async write operations
- `sender_unicast_2_0.cpp`: Implements the sender functionality using the experimental NOC API with unicast async write operations

Both API versions run the same test cases but use different underlying implementations. The device 2.0 tests serve as a validation and performance comparison for the new experimental API.
