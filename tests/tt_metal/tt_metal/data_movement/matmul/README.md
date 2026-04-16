# 1D Matmul Data Movement Tests

This test suite implements tests that measure the functionality of data movement patterns for a 1D matrix multiplication. The test exercises two concurrent data movement paths: in0 multicast from L1 sender cores to all cores in the same row, and in1 reads from DRAM to each core's L1.

Two variants are tested:
- **v1** (`test_matmul_1d.cpp`): Fixed sender — the first column always multicasts in0 data.
- **v2** (`test_matmul_1d_v2.cpp`): Rotating sender — K subblocks are distributed round-robin across columns, and each column takes turns multicasting its K subblock.

Both variants share the same test configurations defined in `test_matmul_common.hpp`, avoiding duplication. Each configuration is run as both a v1 and v2 test, producing 52 total test cases (26 per variant).

## Mesh Device API Support

This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

> **Note:** The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow

A 2D grid of Tensix cores is configured with `num_subblocks_r_dim` rows and `num_subblocks_c_dim` columns. Two data movement kernels run concurrently on each core (RISCV_0 and RISCV_1), both preceded by a global barrier synchronization to ensure all cores start simultaneously.

### in0 Path (RISCV_0 — L1 Multicast)

**v1 (Fixed Sender):**
1. The host writes in0 data into L1 of the first column of cores (the "sender" column).
2. Each sender core multicasts its data to all cores in the same row using semaphore-based synchronization:
   - Receiver cores signal readiness by incrementing the sender's semaphore.
   - The sender waits for all receivers, then multicasts the data and signals completion via a receiver semaphore.
   - For single-core rows, a unicast self-write is used instead (known HW limitation with single-core multicast loopback).
3. All cores in the row receive identical in0 data at the multicast output address.

**v2 (Rotating Sender):**
1. The host distributes in0 K subblocks round-robin across all columns: column `c` holds K subblocks `{c, c+C, c+2C, ...}` for its row.
2. For each K iteration `k`, column `k % C` is the sender. It multicasts its K subblock to all cores in the row using the same semaphore protocol as v1.
3. After all K iterations, every core has the complete ordered sequence of K subblocks for its row.

### in1 Path (RISCV_1 — DRAM Read)

1. The host writes in1 data to a DRAM bank.
2. Each core reads its column's slice of in1 data from DRAM into its L1 output address. The DRAM address and size are partitioned based on the core's column index.

### Barrier Synchronization

Both RISCV_0 and RISCV_1 kernels use a global barrier (`barrier_sync.hpp`) before starting data movement. Each RISC processor has an independent barrier semaphore. A coordinator core (the first matmul core) is used as the synchronization point — all cores increment the coordinator's semaphore and poll until all cores have arrived.

### Verification

After program execution, the host reads back L1 memory from every core and verifies:
- **in0**: Each core's multicast output matches the golden in0 data for its row.
- **in1**: Each core's DRAM read output matches the golden in1 data for its column.

## Running the Tests

```bash
# Run all 1D matmul tests (both v1 and v2)
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*1DMatmul*"

# Run only v1 tests
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul1DSweep*"

# Run only v2 tests
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul1DV2Sweep*"

# Run a specific test by ID
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ID1000*"

# List all test names without running
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*1DMatmul*" --gtest_list_tests
```

## Test Parameters

| Parameter             | Data Type    | Description                                                                 |
|-----------------------|--------------|-----------------------------------------------------------------------------|
| `test_id`             | `uint32_t`   | Test ID for identifying different test cases (1000-1025).                   |
| `start_logical_core`  | `CoreCoord`  | Starting logical coordinate for the matmul core grid. Default: (0,0).       |
| `num_subblocks_r_dim` | `uint32_t`   | Number of rows in the core grid (R dimension). Default: 2.                  |
| `num_subblocks_c_dim` | `uint32_t`   | Number of columns in the core grid (C dimension). Default: 2.               |
| `num_subblocks_k_dim` | `uint32_t`   | Number of subblocks in the accumulation dimension (K). Default: 1.          |
| `subblock_r_dim`      | `uint32_t`   | Number of pages per subblock in R dimension. Default: 1.                    |
| `subblock_c_dim`      | `uint32_t`   | Number of pages per subblock in C dimension. Default: 1.                    |
| `subblock_k_dim`      | `uint32_t`   | Number of pages per subblock in K dimension. Default: 1.                    |
| `page_size_bytes`     | `uint32_t`   | Size of a page in bytes (flit size: 32B for WH, 64B for BH). Auto-computed.|
| `l1_data_format`      | `DataFormat` | Data format of data that will be moved. Default: `Float16_b`.               |
| `dram_bank_id`        | `uint32_t`   | DRAM bank that all cores read in1 from. Default: 0.                         |

## Test Cases

All 26 test configurations are defined in `test_matmul_common.hpp` and shared by both v1 and v2.
Test names follow the pattern: `ID{id}_R{r}_C{c}_K{k}_sr{sr}_sc{sc}_sk{sk}_X{x}Y{y}_bank{b}`

### Grid Shape Tests

| ID   | Grid (RxC) | K | Description                                         |
|------|-----------|---|-----------------------------------------------------|
| 1000 | 2x2       | 1 | Default 2x2 grid.                                   |
| 1001 | 1x1       | 1 | Sender multicasts to itself, reads full in1.         |
| 1002 | 1x3       | 1 | One sender multicasts in0 to 3 receivers.            |
| 1003 | 3x1       | 1 | 3 independent senders each multicast to self.        |
| 1004 | 2x3       | 1 | Non-square grid (R=3, C=2).                          |
| 1005 | 3x2       | 1 | Non-square grid (R=2, C=3).                          |
| 1006 | 4x4       | 1 | Large square grid.                                   |
| 1007 | 6x6       | 2 | 36 cores, 6-way multicast per row.                   |
| 1008 | 1x8       | 1 | Maximum multicast fan-out (8 receivers).              |
| 1009 | 8x1       | 4 | 8 independent senders, deep accumulation.            |

### Non-Origin Start Tests

| ID   | Grid (RxC) | Start  | K | Description                                |
|------|-----------|--------|---|--------------------------------------------|
| 1010 | 2x2       | (2,2)  | 1 | Grid starting at logical core (2,2).       |
| 1011 | 5x3       | (2,3)  | 2 | Offset + large non-square grid.            |

### K Dimension Tests

| ID   | Grid (RxC) | K | Description                                |
|------|-----------|---|--------------------------------------------|
| 1012 | 2x2       | 2 | K=2 on default 2x2 grid.                  |
| 1013 | 3x2       | 3 | K=3 with non-square grid.                  |

### Subblock Dimension Tests

| ID   | Grid (RxC) | Subblocks (r,c,k) | K | Description                        |
|------|-----------|--------------------|----|-----------------------------------|
| 1014 | 2x2       | 2,1,1              | 1  | subblock_r=2.                     |
| 1015 | 2x2       | 1,2,1              | 1  | subblock_c=2.                     |
| 1016 | 2x2       | 1,1,2              | 1  | subblock_k=2.                     |
| 1017 | 2x2       | 2,2,2              | 2  | All subblocks=2, K=2.             |
| 1018 | 2x2       | 4,4,4              | 2  | All subblocks=4, K=2.             |
| 1019 | 2x6       | 3,2,2              | 3  | Mixed large subblock dims, K=3.   |

### Stress Tests

| ID   | Grid (RxC) | K | Subblocks (r,c,k) | Description                          |
|------|-----------|---|--------------------|--------------------------------------|
| 1020 | 4x4       | 4 | 2,2,2              | All dimensions large simultaneously. |
| 1021 | 1x6       | 3 | 3,2,2              | Big multicast + big DRAM reads.      |

### DRAM Bank Tests

| ID   | Grid (RxC) | Description                                |
|------|-----------|---------------------------------------------|
| 1022 | 2x2       | Reads in1 from DRAM bank 1 instead of 0.   |

### K-vs-C Edge Case Tests

| ID   | Grid (RxC) | K | Description                                |
|------|-----------|---|--------------------------------------------|
| 1023 | 2x4       | 2 | K < C: only some columns ever send.       |
| 1024 | 2x3       | 3 | K == C: each column sends exactly once.    |
| 1025 | 2x3       | 5 | K % C != 0: uneven round-robin.           |

## Helper Functions

- `run_dm_1d_matmul()` / `run_dm_1d_matmul_v2()` — Core test function that sets up kernels, semaphores, barrier synchronization, launches the program, and verifies results.
- `run_single_test()` — Convenience wrapper that computes physical constraints (`page_size_bytes`) and derives `end_logical_core` from `start_logical_core` and grid dimensions, then calls the core test function.
- `get_matmul_test_configs()` — Returns the shared hardcoded test configurations.

## L1 Memory Layout Per Core

### v1
```
l1_base_address:                              in0 source data
l1_base_address + in0_size + 0x10:            in0 multicast output
aligned(l1_base + 2*(in0_size + 0x10)):       in1 DRAM read output
align16(in1_output + in1_size):               RISCV_0 barrier scratch (16 bytes)
align16(in1_output + in1_size) + 16:          RISCV_1 barrier scratch (16 bytes)
```

### v2
```
l1_base_address:                              in0 source data (per-core K subblocks)
l1_base_address + max_source_data + 0x10:     in0 multicast output (all K subblocks)
aligned(output_end + 0x10):                   in1 DRAM read output
align16(in1_output + in1_size):               RISCV_0 barrier scratch (16 bytes)
align16(in1_output + in1_size) + 16:          RISCV_1 barrier scratch (16 bytes)
```

## Notes

- All test cases use bfloat16 data format and architecture-specific page sizes (flit size: 32B for Wormhole, 64B for Blackhole).
- The in1 DRAM output address is aligned to `NOC_DRAM_READ_ALIGNMENT_BYTES` (64 on Blackhole) to satisfy DRAM read alignment requirements.
- Barrier synchronization uses a coordinator-based polling pattern from `barrier_sync.hpp`, shared with the `dram_neighbour` test suite.
- For single-column grids (`num_subblocks_c_dim = 1`), the in0 kernel uses a unicast self-write instead of multicast loopback due to a known hardware limitation.
- Tests that request a grid exceeding the device's compute grid are automatically skipped (not failed).
