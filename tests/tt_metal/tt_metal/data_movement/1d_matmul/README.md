# 1D Matmul Data Movement Tests

This test suite implements tests that measure the functionality of data movement patterns for a 1D matrix multiplication. The test exercises two concurrent data movement paths: in0 multicast from L1 sender cores to all cores in the same row, and in1 reads from DRAM to each core's L1.

## Mesh Device API Support

This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

> **Note:** The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow

A 2D grid of Tensix cores is configured with `num_subblocks_r_dim` rows and `num_subblocks_c_dim` columns. Two data movement kernels run concurrently on each core (RISCV_0 and RISCV_1), both preceded by a global barrier synchronization to ensure all cores start simultaneously.

### in0 Path (RISCV_0 — L1 Multicast)

1. The host writes in0 data into L1 of the first column of cores (the "sender" column).
2. Each sender core multicasts its data to all cores in the same row using semaphore-based synchronization:
   - Receiver cores signal readiness by incrementing the sender's semaphore.
   - The sender waits for all receivers, then multicasts the data and signals completion via a receiver semaphore.
   - For single-core rows, a unicast self-write is used instead (known HW limitation with single-core multicast loopback).
3. All cores in the row receive identical in0 data at the multicast output address.

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
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*1DMatmul*"
```

## Test Parameters

| Parameter             | Data Type    | Description                                                                 |
|-----------------------|--------------|-----------------------------------------------------------------------------|
| `test_id`             | `uint32_t`   | Test ID for identifying different test cases.                               |
| `start_logical_core`  | `CoreCoord`  | Starting logical coordinate for the matmul core grid.                       |
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

Each test case uses bfloat16 as L1 data format and flit size as page size. Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

### Grid Shape Tests

| Test Name                    | ID  | Grid     | K | Description                                             |
|------------------------------|-----|----------|---|---------------------------------------------------------|
| Test1DMatmulIdeal            | 650 | 2x2      | 1 | Default 2x2 grid.                                      |
| Test1DMatmulSingleCore       | 651 | 1x1      | 1 | Sender multicasts to itself, reads full in1 from DRAM.  |
| Test1DMatmulSingleRow        | 652 | 1x3      | 1 | One sender multicasts in0 to 3 receivers.               |
| Test1DMatmulSingleColumn     | 653 | 3x1      | 1 | 3 independent senders each multicast to self.           |
| Test1DMatmulNonSquare2x3     | 654 | 2x3      | 1 | Non-square grid (R=3, C=2).                             |
| Test1DMatmulNonSquare3x2     | 655 | 3x2      | 1 | Non-square grid (R=2, C=3).                             |
| Test1DMatmulLargeGrid4x4     | 656 | 4x4      | 1 | Large square grid.                                      |
| Test1DMatmulLargeGrid6x6     | 657 | 6x6      | 2 | 36 cores, 6-way multicast per row.                      |
| Test1DMatmulWideMulticast8Col| 658 | 1x8      | 1 | Maximum multicast fan-out (8 receivers).                 |
| Test1DMatmulTall8RowDeepK    | 659 | 8x1      | 4 | 8 independent senders, deep accumulation.               |

### Non-Origin Start Tests

| Test Name                          | ID  | Grid     | Start    | K | Description                                    |
|------------------------------------|-----|----------|----------|---|------------------------------------------------|
| Test1DMatmulNonOriginStart         | 660 | 2x2      | (2,2)    | 1 | Grid starting at logical core (2,2).           |
| Test1DMatmulNonOriginLargeNonSquare| 661 | 5x3      | (2,3)    | 2 | Offset + large non-square grid.                |

### K Dimension Tests

| Test Name                     | ID  | Grid     | K | Description                                    |
|-------------------------------|-----|----------|---|------------------------------------------------|
| Test1DMatmulLargerK           | 662 | 2x2      | 2 | K=2 on default 2x2 grid.                      |
| Test1DMatmulMultipleKSubblocks| 663 | 3x2      | 3 | K=3 with non-square grid.                      |

### Subblock Dimension Tests

| Test Name                      | ID  | Grid | Subblocks (r,c,k) | Description                            |
|--------------------------------|-----|------|--------------------|----------------------------------------|
| Test1DMatmulLargerSubblockR    | 664 | 2x2  | 2,1,1              | subblock_r=2.                          |
| Test1DMatmulLargerSubblockC    | 665 | 2x2  | 1,2,1              | subblock_c=2.                          |
| Test1DMatmulLargerSubblockK    | 666 | 2x2  | 1,1,2              | subblock_k=2.                          |
| Test1DMatmulAllSubblocksLarger | 667 | 2x2  | 2,2,2              | All subblocks=2, K=2.                  |
| Test1DMatmulMaxSubblockDims    | 668 | 2x2  | 4,4,4              | All subblocks=4, K=2.                  |
| Test1DMatmulAsymmetricSubblocks| 669 | 2x6  | 3,2,2              | Mixed large subblock dims, K=3.        |

### Stress Tests

| Test Name                             | ID  | Grid | K | Subblocks (r,c,k) | Description                              |
|---------------------------------------|-----|------|---|--------------------|------------------------------------------|
| Test1DMatmulLargeGridDeepKLargeSubblocks | 670 | 4x4  | 4 | 2,2,2              | All dimensions large simultaneously.     |
| Test1DMatmulWideMulticastLargePayload | 671 | 1x6  | 3 | 3,2,2              | Big multicast payload + big DRAM reads.  |

### DRAM Bank Tests

| Test Name            | ID  | Grid | Description                                    |
|----------------------|-----|------|------------------------------------------------|
| Test1DMatmulDramBank1| 672 | 2x2  | Reads in1 from DRAM bank 1 instead of bank 0. |

## Helper Functions

- `run_dm_1d_matmul()` — Core test function that sets up kernels, semaphores, barrier synchronization, launches the program, and verifies results.
- `run_single_test()` — Convenience wrapper that computes physical constraints (`page_size_bytes`) and derives `origin_logical_core` / `end_logical_core` from `start_logical_core` and grid dimensions, then calls `run_dm_1d_matmul()`.

## L1 Memory Layout Per Core

```
l1_base_address:                              in0 source data
l1_base_address + in0_size + 0x10:            in0 multicast output
aligned(l1_base + 2*(in0_size + 0x10)):       in1 DRAM read output
in1_output + in1_size:                        RISCV_0 barrier scratch (4 bytes)
in1_output + in1_size + 4:                    RISCV_1 barrier scratch (4 bytes)
```

## Notes

- All test cases use bfloat16 data format and architecture-specific page sizes (flit size: 32B for Wormhole, 64B for Blackhole).
- The in1 DRAM output address is aligned to `NOC_DRAM_READ_ALIGNMENT_BYTES` (64 on Blackhole) to satisfy DRAM read alignment requirements.
- Barrier synchronization uses a coordinator-based polling pattern from `barrier_sync.hpp`, shared with the `dram_neighbour` test suite.
- For single-column grids (`num_subblocks_c_dim = 1`), the in0 kernel uses a unicast self-write instead of multicast loopback due to a known hardware limitation.
