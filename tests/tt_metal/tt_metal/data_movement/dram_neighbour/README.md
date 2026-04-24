# DRAM Neighbour Data Movement Tests

This test suite implements tests that measure the functionality and performance of data movement transactions between DRAM banks and Tensix cores using optimal core-to-DRAM mappings.

## Mesh Device API Support

This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

> **Note:** The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow

Data is distributed across multiple DRAM banks using sharded buffers. Each Tensix core reads data from its assigned adjacent DRAM bank using optimal core-to-DRAM mappings. The test uses barrier synchronization to ensure all cores start data movement simultaneously and validates that each core reads the correct data portion from its assigned bank.

Test attributes such as transaction sizes, number of transactions, core locations, and DRAM bank assignments are recorded by the profiler. Resulting data is cross-checked with original data and validated through equality checks.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests

The tests use the Mesh Device API with fast dispatch mode:

```bash
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*DramNeighbour*"
```

## Test Parameters

| Parameter            | Data Type                         | Description                                                                 |
|---------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `test_id`           | `uint32_t`                        | Test id for signifying different test cases. Can be used for grouping different tests. |
| `num_of_transactions` | `uint32_t`                      | Number of DRAM transactions that will be issued.                            |
| `num_banks`         | `uint32_t`                        | Number of DRAM banks to use for the test.                                   |
| `pages_per_bank`    | `uint32_t`                        | Size of the issued DRAM transactions in pages per bank.                     |
| `page_size_bytes`   | `uint32_t`                        | Size of a page in bytes.                                                    |
| `l1_data_format`    | `DataFormat`                      | Data format of data that will be moved.                                     |
| `core_dram_map`     | `std::map<uint32_t, uint32_t>`     | Mapping between core coordinates and DRAM bank IDs.                          |
| `dram_index_map`    | `std::map<uint32_t, IndexRange>`   | Index ranges for golden data validation per DRAM bank.                       |

## Test Cases

Each test case uses bfloat16 as L1 data format and tile size as page size. Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

### Core-to-DRAM Mapping Strategies

- **Ideal Mapping (`core_dram_mapping_ideal`)** - Uses optimal DRAM bank to logical worker assignment from the device.
- **Neighbour Mapping (`add_neighbour_cores_dram_mapping`)** - Extends ideal mapping to include adjacent cores, allowing multiple cores to access the same DRAM bank for neighbour access patterns.
- **Single Row Mapping (`add_single_row_cores_dram_mapping`)** - Maps all cores in a single row to one DRAM bank for testing row-wise access patterns.

### Test Implementations

- **TensixDataMovementDramNeighbourDirectedIdeal (Test ID: 502)** - Tests the optimal data movement setup using ideal core-to-DRAM mapping with neighbour cores. Uses a single transaction to validate the mapping correctness.
- **TensixDataMovementDramNeighbourNumPagesSweep (Test ID: 503)** - Tests different number of pages per bank by sweeping `pages_per_bank` from 1 to 32 while keeping the number of banks fixed. Validates performance across various data sizes.
- **TensixDataMovementDramNeighbourNumBankSweep (Test ID: 504)** - Tests different numbers of DRAM banks by sweeping from 1 to the maximum available banks. Each configuration uses optimal neighbour mapping for the given number of banks.
- **TensixDataMovementDramNeighbourSingleRowSweep (Test ID: 505)** - Tests single-row mapping where all cores in a row access the same DRAM bank. Sweeps through different page sizes to test contention patterns.

## Helper Functions

The test suite includes several helper functions for test configuration and execution:

- `run_single_test()` - Executes a single test configuration with specified parameters
- `run_sweep_test()` - Executes parameter sweeps over transactions and page counts
- `run_bank_sweep_test()` - Executes sweeps over number of banks with optimal mapping
- `get_golden_index_ranges()` - Calculates index ranges for data validation per DRAM bank

## Notes

- The test uses sharded DRAM buffers to distribute data across multiple banks
- Barrier synchronization ensures all cores start data movement simultaneously
- Core coordinates are encoded as 32-bit keys: `(x << 16) | y` for efficient mapping lookup
- The test validates data integrity by comparing each core's output with the expected golden data for its assigned DRAM bank
