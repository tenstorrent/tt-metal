# Matmul Data Movement Tests

This test suite validates the data movement patterns required for matrix multiplication on Tenstorrent hardware. The tests do not perform actual matrix multiplication compute — they verify that the correct data arrives at the correct L1 memory location on every core.

The multiplication being modeled is:

```
out[M, N] = in0[M, K_total] x in1[K_total, N]
```

where `M`, `N`, and `K_total` are derived from the core grid dimensions and subblock sizes (see [Dimensions and Core Grid Mapping](#dimensions-and-core-grid-mapping) below).

Three variants are tested, covering different strategies for distributing input matrices across cores:

- **1D v1** (`test_matmul_1d.cpp`): Fixed sender — column 0 holds all `in0` data and multicasts it to every core in the same row. `in1` is read from DRAM independently by each core.
- **1D v2** (`test_matmul_1d_v2.cpp`): Rotating sender — `in0` K subblocks are distributed round-robin across columns, and each column takes turns multicasting its portion. `in1` is read from DRAM independently by each core.
- **2D** (`test_matmul_2d.cpp`): Rotating sender for both inputs — `in0` K subblocks are distributed round-robin across columns and multicast row-wise (same as 1D v2), while `in1` K subblocks are distributed round-robin across rows and multicast column-wise. No DRAM is used; all data is written to L1 by the host.

All three variants share test configurations defined in `test_matmul_common.hpp`. Each configuration runs as a 1D v1, 1D v2, and 2D test, producing 78 total test cases (26 per variant).

## Dimensions and Core Grid Mapping

A 2D grid of `R x C` Tensix cores is allocated, where:

- **R** (`num_subblocks_r_dim`) = number of **rows** of cores
- **C** (`num_subblocks_c_dim`) = number of **columns** of cores

Each core computes one subblock of the output matrix. The core at grid position `(r, c)` (0-indexed) is responsible for output subblock at row `r`, column `c`.

The full matrix dimensions are:

| Matrix | Rows | Columns | Description |
|--------|------|---------|-------------|
| **in0** | `R * subblock_r_dim` | `K * subblock_k_dim` | Left input matrix |
| **in1** | `K * subblock_k_dim` | `C * subblock_c_dim` | Right input matrix |
| **out** | `R * subblock_r_dim` | `C * subblock_c_dim` | Output matrix |

Here **K** (`num_subblocks_k_dim`) is the number of subblocks in the accumulation dimension. The subblock dimensions (`subblock_r_dim`, `subblock_c_dim`, `subblock_k_dim`) control how many pages each subblock contains.

### What each core needs

To compute its output subblock, core `(r, c)` needs:

- **in0**: The entire row `r` of the `in0` matrix — all `K` subblocks worth of data. This is the same for every core in the same row (columns 0 through C-1 all need the same in0 row data).
- **in1**: The column `c` slice of the `in1` matrix — the portion corresponding to its column index. This is the same for every core in the same column (rows 0 through R-1 all need the same in1 column data).

The `in0` data is distributed via NOC multicast across the row. The `in1` data is read independently by each core from DRAM. These two paths run concurrently on separate RISC-V processors (RISCV_0 for in0, RISCV_1 for in1).

## v1: Fixed Sender Methodology

In v1, column 0 is the sole sender for every row. All `in0` data is loaded into the first column of cores, and each column-0 core multicasts the complete data for its row to all other cores in that row.

### Host-side data placement

The host writes `in0` data to L1 of the **first column only**. For a grid with R rows:

```
Column 0, Row 0  <--  in0 row 0 data (all K subblocks for row 0)
Column 0, Row 1  <--  in0 row 1 data (all K subblocks for row 1)
...
Column 0, Row R-1 <-- in0 row R-1 data (all K subblocks for row R-1)
```

Columns 1 through C-1 receive no `in0` data from the host — they get it via multicast.

### On-device multicast (in0_kernel.cpp)

Each core determines if it is the sender by checking whether its physical X coordinate matches the first column:

```
is_sender = (my_x == physical_start_x)
```

**Sender (column 0):**
1. Waits for all C-1 receivers in the row to signal readiness (via `sender_sem`).
2. Multicasts the entire in0 row data from its L1 source address to the multicast output address on all cores in the row (`noc_async_write_multicast_loopback_src`).
3. Signals all receivers that data has arrived by multicasting `sender_valid_sem` into `receiver_sem` on every core.

**Receiver (columns 1 through C-1):**
1. Increments the sender's `sender_sem` via `noc_semaphore_inc` to signal readiness.
2. Waits for `receiver_sem` to be set to 1 by the sender's multicast.

**Special case — single column (C = 1):** When there is only one column, the sender uses a unicast self-write instead of multicast loopback, due to a known hardware limitation with single-core multicast.

### Example: R=3, C=4, K=2

```
Host writes to column 0 only:

     Col 0       Col 1       Col 2       Col 3
Row 0: [K0,K1] --> mcast --> [K0,K1]   [K0,K1]   [K0,K1]
Row 1: [K0,K1] --> mcast --> [K0,K1]   [K0,K1]   [K0,K1]
Row 2: [K0,K1] --> mcast --> [K0,K1]   [K0,K1]   [K0,K1]

After multicast: every core in the same row has identical in0 data.
```

## v2: Rotating Sender Methodology

In v2, the `in0` K subblocks are distributed across **all columns** using a round-robin assignment. Instead of one fixed sender per row, the sender rotates each K iteration, spreading the multicast workload evenly across cores.

### Host-side data placement (round-robin distribution)

For each row, the host assigns K subblock `k` to column `k % C`:

```
Column c holds K subblocks: { c, c + C, c + 2C, ... } for each row
```

Concretely, for row `r`, column `c` receives the following K subblocks in its L1 source buffer (in order):

```
K subblocks at indices: c, c+C, c+2C, ...  (while index < K)
```

### On-device K-loop (in0_kernel_v2.cpp)

The kernel iterates through all K subblocks. At each iteration `k`, the sender is column `k % C`:

```cpp
for (k = 0; k < K; k++) {
    sender_col = k % C;
    output_addr = mcast_output_base + k * k_subblock_size;

    if (my_col == sender_col) {
        // SENDER: multicast K subblock from local source to all cores in row
    } else {
        // RECEIVER: signal readiness to sender
    }
}
```

Each K subblock is written to a distinct offset in the output buffer (`output_addr = base + k * size`), so after all K iterations, every core in the row has the complete, correctly-ordered sequence `[K0, K1, K2, ..., K_{K-1}]`.

The sender maintains a `local_k_send_idx` counter that tracks which of its locally-held subblocks to send next. Since the subblocks were loaded in round-robin order by the host, sending them sequentially produces the correct global K index at each iteration.

### K-dimension partitioning scenarios

The relationship between K (number of K subblocks) and C (number of columns) determines how the multicast workload is distributed. Three key scenarios arise:

#### K < C: Some columns never send

When K < C, there are fewer K subblocks than columns. Only columns 0 through K-1 ever become the sender. The remaining columns participate only as receivers.

**Example: R=2, C=4, K=2**

```
Host placement (round-robin: column c gets K subblock c, c+4, c+8, ...):
  Column 0: holds K0        Column 2: holds nothing
  Column 1: holds K1        Column 3: holds nothing

K-loop (2 iterations):
  k=0: sender_col = 0%4 = 0  -->  Column 0 multicasts K0 to all 4 cores in row
  k=1: sender_col = 1%4 = 1  -->  Column 1 multicasts K1 to all 4 cores in row

Result: Every core has [K0, K1]. Columns 2 and 3 never sent anything.
```

#### K = C: Each column sends exactly once

When K equals C, each column holds exactly one K subblock and sends it during its corresponding iteration. This is the balanced case — every column contributes equally.

**Example: R=2, C=3, K=3**

```
Host placement:
  Column 0: holds K0
  Column 1: holds K1
  Column 2: holds K2

K-loop (3 iterations):
  k=0: sender_col = 0%3 = 0  -->  Column 0 multicasts K0
  k=1: sender_col = 1%3 = 1  -->  Column 1 multicasts K1
  k=2: sender_col = 2%3 = 2  -->  Column 2 multicasts K2

Result: Every core has [K0, K1, K2]. Each column sent exactly once.
```

#### K > C: Columns send multiple times

When K > C, the round-robin wraps around. Each column sends `ceil(K/C)` or `floor(K/C)` times depending on whether `K % C == 0`.

**Example: K evenly divisible — R=2, C=3, K=6**

```
Host placement:
  Column 0: holds K0, K3    (k=0 and k=3)
  Column 1: holds K1, K4    (k=1 and k=4)
  Column 2: holds K2, K5    (k=2 and k=5)

K-loop (6 iterations):
  k=0: Column 0 multicasts K0     k=3: Column 0 multicasts K3
  k=1: Column 1 multicasts K1     k=4: Column 1 multicasts K4
  k=2: Column 2 multicasts K2     k=5: Column 2 multicasts K5

Result: Every core has [K0, K1, K2, K3, K4, K5]. Each column sent exactly twice.
```

**Example: K not evenly divisible — R=2, C=3, K=5**

```
Host placement:
  Column 0: holds K0, K3    (2 subblocks)
  Column 1: holds K1, K4    (2 subblocks)
  Column 2: holds K2         (1 subblock)

K-loop (5 iterations):
  k=0: sender_col = 0  -->  Column 0 multicasts K0 (its 1st local subblock)
  k=1: sender_col = 1  -->  Column 1 multicasts K1 (its 1st local subblock)
  k=2: sender_col = 2  -->  Column 2 multicasts K2 (its 1st local subblock)
  k=3: sender_col = 0  -->  Column 0 multicasts K3 (its 2nd local subblock)
  k=4: sender_col = 1  -->  Column 1 multicasts K4 (its 2nd local subblock)

Result: Every core has [K0, K1, K2, K3, K4].
Column 0 sent 2 times. Column 1 sent 2 times. Column 2 sent 1 time.
```

### Why rotating sender matters

In v1, column 0 bears the entire multicast burden — it must send all K subblocks for its row. This creates a bottleneck at column 0, especially for large K values.

In v2, the multicast workload is spread across all columns. When K >= C, each column contributes approximately `K/C` multicasts. This distributes the NOC write traffic more evenly across the physical network, reduces congestion at a single sender, and allows each sender to use a smaller L1 source buffer (it only stores its own K subblocks, not the full row).

## in1 Path (1D v1 and v2)

The `in1` data movement is identical in both 1D variants. Each core independently reads its column's portion of `in1` from DRAM.

### Host-side data placement

The host writes the complete `in1` matrix to a single DRAM bank. The `in1` data is laid out as a flat array partitioned by column:

```
DRAM:  [ column 0 slice | column 1 slice | ... | column C-1 slice ]
```

Each column slice has size `in1_pages_bytes / C` bytes.

### On-device DRAM read (in1_kernel.cpp)

Each core reads from a DRAM address determined by its column index:

```
dram_read_addr = base_dram_addr + col_idx * per_core_read_size
```

All cores read the same number of bytes, but from different offsets. Cores in the same column read the same DRAM region (they need identical in1 column data).

## 2D: Rotating Sender for Both in0 and in1

The 2D variant (`test_matmul_2d.cpp`) extends the rotating sender approach to **both** input matrices. Unlike the 1D variants where `in1` is read from DRAM, the 2D variant distributes `in1` via multicast down columns — making all data movement L1-to-L1 with no DRAM involvement.

### in0 path (row-wise multicast, RISCV_0)

The in0 data movement is identical to 1D v2: K subblocks are distributed round-robin across columns, and each column takes turns multicasting its portion across the row. See [v2: Rotating Sender Methodology](#v2-rotating-sender-methodology) above.

### in1 path (column-wise multicast, RISCV_1)

The in1 data movement mirrors the in0 rotating sender pattern, but operates along columns instead of rows:

- **Host-side placement**: For each column, the host distributes K subblocks round-robin across rows. Row `r` receives K subblocks at indices `{r, r+R, r+2R, ...}` for each column.
- **On-device K-loop** (`in1_kernel_2d.cpp`): At each iteration `k`, the sender is row `k % R`. The sender multicasts its K subblock to all cores in its column. After all K iterations, every core in the same column has the complete, correctly-ordered in1 data.

```cpp
for (k = 0; k < K; k++) {
    sender_row = k % R;
    output_addr = mcast_output_base + k * k_subblock_size;

    if (my_row == sender_row) {
        // SENDER: multicast K subblock from local source to all cores in column
    } else {
        // RECEIVER: signal readiness to sender
    }
}
```

The same K-vs-R partitioning scenarios apply (K < R, K = R, K > R), mirroring the K-vs-C behavior of the in0 path.

### NOC assignment

The two multicast paths run concurrently on separate RISC-V processors and separate NOCs:
- **RISCV_0 / NOC0**: in0 row-wise multicast
- **RISCV_1 / NOC1**: in1 column-wise multicast

NOC1 has reversed routing direction, so the in1 kernel swaps `start_y` and `end_y` when constructing multicast addresses.

### Example: R=3, C=4, K=2

```
in0 (row-wise multicast, same as 1D v2):
  Each row's K subblocks distributed across columns round-robin.
  k=0: column 0 multicasts K0 across row
  k=1: column 1 multicasts K1 across row

in1 (column-wise multicast):
  Each column's K subblocks distributed across rows round-robin.
  k=0: row 0 multicasts K0 down column
  k=1: row 1 multicasts K1 down column

After both K-loops complete: every core has all in0 data for its row
and all in1 data for its column.
```

### Why 2D matters

In 1D, every core independently reads in1 from DRAM. Cores in the same column read the same DRAM region, creating redundant DRAM traffic that scales linearly with R. The 2D approach eliminates this redundancy: each row reads its portion of in1 once, then multicasts it to the rest of the column. This trades DRAM bandwidth for NOC bandwidth, which is more plentiful and has lower latency on Tenstorrent hardware.

## Barrier Synchronization

Both RISCV_0 (in0 multicast) and RISCV_1 (in1 DRAM read or column multicast) execute a global barrier before starting data movement. This ensures all cores are ready before any multicast or DRAM read begins.

Each RISC processor has its own independent barrier using two semaphores:
- **Arrival semaphore**: Each core increments the coordinator's semaphore to signal it has reached the barrier.
- **Done semaphore**: The coordinator waits for all cores to arrive, then multicasts the done signal to all cores.

The coordinator is always the first matmul core (top-left corner of the grid).

## Verification

After program execution, the host reads back L1 memory from every core and verifies:

- **in0**: Each core's multicast output buffer contains the complete, correctly-ordered in0 data for its row. In v1, this is the data that column 0 multicast. In v2 and 2D, this is the reassembled sequence of K subblocks from the rotating senders.
- **in1**: Each core's output buffer matches the golden in1 data for its column. In 1D (v1 and v2), this is the DRAM read output. In 2D, this is the column-wise multicast output.

## Test Parameters

| Parameter             | Type         | Description |
|-----------------------|--------------|-------------|
| `test_id`             | `uint32_t`   | Unique test case identifier (1000-1025). |
| `start_logical_core`  | `CoreCoord`  | Top-left logical coordinate of the core grid. Default: (0,0). |
| `num_subblocks_r_dim` | `uint32_t`   | R — number of rows of cores. Default: 2. |
| `num_subblocks_c_dim` | `uint32_t`   | C — number of columns of cores. Default: 2. |
| `num_subblocks_k_dim` | `uint32_t`   | K — number of subblocks in the accumulation dimension. Default: 1. |
| `subblock_r_dim`      | `uint32_t`   | Pages per subblock in the R dimension. Default: 1. |
| `subblock_c_dim`      | `uint32_t`   | Pages per subblock in the C dimension. Default: 1. |
| `subblock_k_dim`      | `uint32_t`   | Pages per subblock in the K dimension. Default: 1. |
| `page_size_bytes`     | `uint32_t`   | Size of one page in bytes (flit size: 32B for WH, 64B for BH). Auto-computed. |
| `l1_data_format`      | `DataFormat` | Data format for moved data. Default: `Float16_b`. |
| `dram_bank_id`        | `uint32_t`   | DRAM bank that all cores read in1 from. Default: 0. |

## Test Cases

All 26 configurations are defined in `test_matmul_common.hpp` and shared by both v1 and v2.
Test names follow the pattern: `ID{id}_R{r}_C{c}_K{k}_sr{sr}_sc{sc}_sk{sk}_X{x}Y{y}_bank{b}`

### Grid Shape Tests

| ID   | Grid (RxC) | K | Description |
|------|-----------|---|-------------|
| 1000 | 2x2       | 1 | Default 2x2 grid. |
| 1001 | 1x1       | 1 | Single core — sender multicasts to itself. |
| 1002 | 1x3       | 1 | Single row, 3-way multicast. |
| 1003 | 3x1       | 1 | 3 rows, single column — each row is its own sender. |
| 1004 | 2x3       | 1 | Non-square: 2 rows, 3 columns. |
| 1005 | 3x2       | 1 | Non-square: 3 rows, 2 columns. |
| 1006 | 4x4       | 1 | Large square grid. |
| 1007 | 6x6       | 2 | 36 cores, 6-way multicast per row, K=2. |
| 1008 | 1x8       | 1 | Maximum multicast fan-out (8 receivers). |
| 1009 | 8x1       | 4 | 8 independent single-column rows, deep K. |

### Non-Origin Start Tests

| ID   | Grid (RxC) | Start  | K | Description |
|------|-----------|--------|---|-------------|
| 1010 | 2x2       | (2,2)  | 1 | Grid offset to logical core (2,2). |
| 1011 | 5x3       | (2,3)  | 2 | Offset + large non-square grid. |

### K Dimension Tests

| ID   | Grid (RxC) | K | Description |
|------|-----------|---|-------------|
| 1012 | 2x2       | 2 | K=2, two multicast rounds (v2: each column sends once). |
| 1013 | 3x2       | 3 | K=3 with C=2 — v2: K > C, columns send multiple times. |

### Subblock Dimension Tests

| ID   | Grid (RxC) | Subblocks (r,c,k) | K | Description |
|------|-----------|--------------------|----|-------------|
| 1014 | 2x2       | 2,1,1              | 1  | Larger subblock in R. |
| 1015 | 2x2       | 1,2,1              | 1  | Larger subblock in C. |
| 1016 | 2x2       | 1,1,2              | 1  | Larger subblock in K. |
| 1017 | 2x2       | 2,2,2              | 2  | All subblocks=2, K=2. |
| 1018 | 2x2       | 4,4,4              | 2  | All subblocks=4, K=2. |
| 1019 | 2x6       | 3,2,2              | 3  | Mixed large subblocks, K=3, C=6. |

### Stress Tests

| ID   | Grid (RxC) | K | Subblocks (r,c,k) | Description |
|------|-----------|---|--------------------|-------------|
| 1020 | 4x4       | 4 | 2,2,2              | All dimensions exercised simultaneously. |
| 1021 | 1x6       | 3 | 3,2,2              | Single row, 6-way multicast, large data. |

### DRAM Bank Test

| ID   | Grid (RxC) | Bank | Description |
|------|-----------|------|-------------|
| 1022 | 2x2       | 1    | Reads in1 from DRAM bank 1 instead of 0. |

### K-vs-C Edge Case Tests (v2 specific)

These test the three key relationships between K and C that affect v2's rotating sender behavior:

| ID   | Grid (RxC) | K | Relationship | v2 Behavior |
|------|-----------|---|--------------|-------------|
| 1023 | 2x4       | 2 | K < C        | Only columns 0-1 send; columns 2-3 are always receivers. |
| 1024 | 2x3       | 3 | K = C        | Each column sends exactly once — balanced workload. |
| 1025 | 2x3       | 5 | K % C != 0   | Columns 0-1 send twice, column 2 sends once — uneven. |

## Running the Tests

```bash
# Run all matmul tests (1D v1, 1D v2, and 2D)
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul*"

# Run only 1D v1 tests
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul1DSweep*"

# Run only 1D v2 tests
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul1DV2Sweep*"

# Run only 2D tests
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul2DSweep*"

# Run a specific test by ID (runs across all variants)
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ID1000*"

# List all test names without running
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*Matmul*" --gtest_list_tests
```

## L1 Memory Layout Per Core

### v1
```
l1_base_address:                              in0 source data (column 0 only)
l1_base_address + in0_size + 0x10:            in0 multicast output (all cores)
aligned(l1_base + 2*(in0_size + 0x10)):       in1 DRAM read output
align16(in1_output + in1_size):               RISCV_0 barrier scratch (16 bytes)
align16(barrier_scratch_0 + 16):              RISCV_1 barrier scratch (16 bytes)
```

### v2
```
l1_base_address:                              in0 source data (this core's K subblocks)
l1_base_address + max_source_data + 0x10:     in0 multicast output (all K subblocks)
aligned(mcast_output_end + 0x10):             in1 DRAM read output
align16(in1_output + in1_size):               RISCV_0 barrier scratch (16 bytes)
align16(barrier_scratch_0 + 16):              RISCV_1 barrier scratch (16 bytes)
```

The `in1` DRAM output address is aligned to 64 bytes (`NOC_DRAM_READ_ALIGNMENT_BYTES` on Blackhole) so that the low address bits match the DRAM source address.

### 2D
```
l1_base_address:                              in0 source data (this col's K subblocks)
in0_source_end + 0x10:                        in0 multicast output (K * in0_k_sub_size)
in0_mcast_end + 0x10:                         in1 source data (this row's K subblocks)
in1_source_end + 0x10:                        in1 multicast output (K * in1_k_sub_size)
align16(in1_mcast_end):                       RISCV_0 barrier scratch (16 bytes)
align16(risc0_scratch + 16):                  RISCV_1 barrier scratch (16 bytes)
```

In the 2D variant, both in0 and in1 have separate source and multicast output regions in L1. No DRAM alignment is needed since all data movement is L1-to-L1.

## Notes

- All test cases use bfloat16 data format and architecture-specific page sizes (32B for Wormhole, 64B for Blackhole).
- For single-column grids (C = 1), the in0 kernel uses a unicast self-write instead of multicast loopback due to a known hardware limitation. Similarly, for single-row grids (R = 1) in 2D, the in1 kernel uses a unicast self-write.
- Tests that request a grid exceeding the device's compute grid are automatically skipped (not failed).
- Barrier synchronization uses a coordinator-based polling pattern from `barrier_sync.hpp`, shared with other data movement test suites.
- This test suite uses the TT-Metal Mesh Device API with `GenericMeshDeviceFixture`, running on single-device unit meshes. The Mesh Device API only supports fast dispatch mode.
