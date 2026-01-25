# DRAM Unary Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between DRAM and a single Tensix core.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Data is loaded into an L1 circular buffer by the reader kernel and written out to DRAM from the same L1 buffer by the writer kernel. A compute kernel is omitted as these tests are purely for data movement. Hardware barriers ensure data validity and completion of transactions.

Test attributes such as transaction sizes, number of transactions, core locations, and DRAM channels as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is cross-checked with original data and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*DRAM*"
```

## Test Parameters
| Parameter                     | Data Type             | Description |
| ----------------------------- | --------------------- | ----------- |
| test_id                       | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions           | uint32_t              | Number of DRAM transactions that will be issued. |
| pages_per_transaction         | uint32_t              | Size of the issued DRAM transactions in pages. |
| bytes_per_page                | uint32_t              | Size of a page in bytes. Arbitrary value with a minimum of flit size per architecture. |
| l1_data_format                | DataFormat            | Data format of data that will be moved. |
| core_coord                    | CoreCoord             | Logical coordinates for the Tensix core. |
| dram_channel                  | uint32_t              | Specifies which DRAM channel to use for the test. |
| virtual_channel               | uint32_t              | Option to specify unicast VC for each transaction. |
| use_2_0_api                   | bool                  | Determines if the test uses the experimental device 2.0 API. |

## Test Cases
Each test case uses bfloat16 as L1 data format and flit size (32B for WH, 64B for BH) as page size.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **TensixDataMovementDRAMPacketSizes** (Test ID: 0) - Tests different number of transactions and transaction sizes by varying the num_of_transactions and pages_per_transaction parameters. Various packet configurations are tested between DRAM and Tensix core.

2. **TensixDataMovementDRAMCoreLocations** (Test ID: 1) - Tests data movement between DRAM and different Tensix core locations to evaluate performance across various core positions on the chip.

3. **TensixDataMovementDRAMChannels** (Test ID: 2) - Tests data movement using different DRAM channels to evaluate bandwidth utilization across multiple memory interfaces.

4. **TensixDataMovementDRAMDirectedIdeal** (Test ID: 3) - Tests the most optimal data movement setup between DRAM and a Tensix core that maximizes the transaction size and performs enough transactions to amortize initialization overhead.

5. **TensixDataMovementDRAMPacketSizes2_0** (Test ID: 40) - Device 2.0 API version of the packet sizes test. Tests the same packet size variations as test ID 0 but uses the experimental NOC API with structured endpoints and virtual channel support.

## Device 2.0 API Tests
This test suite now includes tests using the new device 2.0 experimental NOC API. These tests provide the same functionality as the original tests but use an updated API design:

### Key Features of Device 2.0 API Tests:
- **Experimental NOC API**: Uses `experimental::Noc`, `experimental::UnicastEndpoint`, `experimental::Semaphore` and `experimental::AllocatorBankType` for structured NOC operations
- **Structured Arguments**: Source and destination arguments are defined using structured `noc_traits_t` types

### Device 2.0 Kernels:
- `writer_unary_2_0.cpp`: Implements the writer functionality using the experimental NOC API
- `reader_unary_2_0.cpp`: Implements the reader functionality using the experimental NOC API
