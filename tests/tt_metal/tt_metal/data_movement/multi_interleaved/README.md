# Multi Interleaved Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of data movement transactions between DRAM interleaved buffers and multiple Tensix cores.

## Mesh Device API Support
This test suite uses the TT-Metal Mesh Device API, which provides a unified interface for single and multi-device operations. The tests use `GenericMeshDeviceFixture` and run on single-device unit meshes.

**Note**: The Mesh Device API only supports fast dispatch mode internally and does not support slow dispatch mode. This provides optimal performance for data movement operations.

## Test Flow
Runs a reader kernel and/or a writer kernel on a grid of tensix cores.

The reader kernel issues NOC instructions to read data from an interleaved DRAM buffer, initialized with data, into the L1 address of the Tensix cores calling the read_page API. A read barrier is placed after these transactions in order to ensure data validity. If the reader kernel isn't run, this data is initially written directly into L1 memory.

The writer kernel issues NOC instructions to write data from the L1 address of the Tensix cores into an interleaved DRAM buffer. A write barrier is placed after these transactions in order to ensure data validity. By default the writes are non-posted (each write requests an ack, and the ack-based `noc_async_write_barrier()` gates completion). The writer can instead issue posted writes via `TT_DM_POSTED_WRITES=1`; posted writes carry no ack, so the writer waits on `noc_async_posted_writes_flushed()` (requests departed, not landed) and the ack-based phase/page counters instead track `NIU_MST_POSTED_WR_REQ_SENT`. The reader has no posted concept.

Transactions exceeding 16 pages will consecutively overwrite the same 16 pages so as not to take up excess L1 memory. Test attributes such as transaction sizes and number of transactions as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. Resulting data is read from the output interleaved buffer if the writer kernel is run (or directly from L1 memory otherwise), cross-checked with original data, and validated through an equality check.

Test expectations are that the equality checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Running the Tests
The tests use the Mesh Device API with fast dispatch mode:
```
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*MultiInterleaved*"
```

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| num_of_transactions       | uint32_t              | Number of transactions in test. |
| num_pages                 | uint32_t              | Number of page read/writes per transaction. |
| page_size_bytes           | uint32_t              | Size of a page in bytes (max 1 packet - 16kB for BH, 8kB for WH). |
| l1_data_format            | DataFormat            | Type of data in transaction. |
| cores                     | CoreRangeSet          | Logical coordinates of Tensix cores running kernels. |
| read_kernel               | bool                  | True if test runs reader kernel. |
| write_kernel              | bool                  | True if test runs writer kernel. |

## Test Cases
Each test case uses bfloat16 as L1 data format.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. Multi Interleaved Sizes: Tests reading and writing with interleaved DRAM buffer over varying number of transactions and transaction sizes over the full grid of Tensix cores.
2. Multi Interleaved Page Directed Ideal: Tests the most optimal transactions for reading/writing pages by maximizing the number of pages and page size to amortize initialization overhead and saturate the bandwidth over the full grid of Tensix cores.
3. Read tests: Run only the reader kernel.
4. Write tests: Run only the writer kernel.
5. Grid configuration tests: Run kernel(s) on a 2x2 or on a 6x6 grid of Tensix cores.
6. Multi Interleaved Read Grid Sweep (test 129): Read-only packet-size sweep over a runtime-configurable core block. The geometry is set via environment variables so a single build can cover row, column, and NxN block topologies without editing the test list:
   - `TT_DM_GRID_COLS`: cores along x (width of a row); defaults to and is clamped by the compute grid x.
   - `TT_DM_GRID_ROWS`: cores along y (height of a column); defaults to and is clamped by the compute grid y.

   A row sweep is `TT_DM_GRID_COLS=k TT_DM_GRID_ROWS=1`; a column sweep is `TT_DM_GRID_COLS=1 TT_DM_GRID_ROWS=k`; an NxN block is `TT_DM_GRID_COLS=TT_DM_GRID_ROWS=k`.

## Environment Variables
| Variable            | Default | Description |
| ------------------- | ------- | ----------- |
| TT_DM_PHASE_COUNTERS| unset (off) | When set to a non-zero value, compiles the device-side `t0..t3` phase counters into the kernel (issue/fill/stream/drain split). Both the reader and writer emit the same generic marker names (`dm_t0_issue_start`, `dm_t1_issue_end`, `dm_t2_first_return`, `dm_t3_barrier_clear`) so one tool (`phase_heatmap.py`) handles both. The progress counter behind them is kernel-specific: reads use `NIU_MST_RD_RESP_RECEIVED`, non-posted writes use `NIU_MST_WR_ACK_RECEIVED`, and posted writes use `NIU_MST_POSTED_WR_REQ_SENT`. Batched (write-only / read-only, i.e. `!sync`) path only. Compiled out by default for clean timing. |
| TT_DM_PAGE_COUNTERS | 0 (off) | When non-zero, emits one timestamp marker (`dm_page_issued`) per page transaction on the batched (`!sync`) path, whose payload is the cumulative return count (see `TT_DM_PHASE_COUNTERS` for the per-kernel counter). WARNING: one marker per page, so large Q (roughly > 110) overflows the per-RISC profiler L1 buffer and the tail is dropped. |
| TT_DM_POSTED_WRITES | 0 (off) | When non-zero, the writer kernel issues posted writes (no per-write ack) instead of the default non-posted writes. Completion is tracked via `noc_async_posted_writes_flushed()` and the phase/page counters use `NIU_MST_POSTED_WR_REQ_SENT`. Posted completion only guarantees departure from the NIU, not arrival at the destination. Writer-only; no effect on the reader kernel. |
| TT_DM_GRID_COLS     | full grid x | Cores along x for the Read Grid Sweep (test 129). Clamped to the compute grid. |
| TT_DM_GRID_ROWS     | full grid y | Cores along y for the Read Grid Sweep (test 129). Clamped to the compute grid. |
| TT_DM_RAND_OFFSET   | 0 (off) | Randomizes each reader core's DRAM-bank access order (read-only; compiled out of the reader kernel when 0; seeded deterministically for reproducibility). `1` = static rotation: random start bank, sequential order, same every transaction iteration. `2` = advancing random permutation: random per-core page/bank order whose start advances each transaction iteration (e.g. D3 D7 D1 D4 ...), decorrelating both the start bank and the batch-to-batch order. In all modes each page is still read exactly once per iteration into its own L1 slot, so the equality check holds. |
| TT_DM_WARMUP_ITERS  | 0 (off) | Runs N uninstrumented passes over the batched read loop before the single measured pass, so the timing counters capture steady-state behavior. Compiled out of the reader kernel when 0. Read-only (`!sync`) path only. |
| TT_DM_DEFAULT_NOC   | 0 (off) | When non-zero, swaps the reader to RISCV1/NOC1 (coordinate mirror that routes the opposite way around the torus) to compare directional return-path routing against the default NOC0/RISCV0. Read Grid Sweep (test 129). |
| TT_DM_INJECT_DELAY  | 0 (off) | Fixed injection-rate throttle (mode 1): spin `N` RISC wall-clock cycles after every `noc_async_read` to deflate the per-core issue rate. Read-only batched path; compiled out when off. Ignored if the MIN/MAX random vars below are set. |
| TT_DM_INJECT_DELAY_MIN / TT_DM_INJECT_DELAY_MAX | 0 (off) | Random injection-rate throttle (mode 2): spin a per-transaction random delay in `[MIN, MAX]` cycles after every read (per-core PRNG seeded deterministically on the host, forced nonzero). Setting either takes precedence over the fixed `TT_DM_INJECT_DELAY`. Read-only batched path. |
| TT_DM_STAGGER       | 0 (off) | Releases the grid one group at a time instead of all cores at once (read-only batched path). `1` = by column (x, isolates DRAM-distance), `2` = by row (y, isolates congestion). Release mechanism is selected by `TT_DM_STAGGER_DELAY`. |
| TT_DM_STAGGER_DIR   | 0 (ascending) | Group release order for `TT_DM_STAGGER`: `0` = ascending group index, `1` = descending. |
| TT_DM_STAGGER_DELAY | 0 (event handoff) | Group release timing for `TT_DM_STAGGER`. `0` = event handoff (each group waits on a progress semaphore until preceding groups finish issuing, then releases the next; adaptive but the release travels the shared read-response path so it can be drain-gated). `N>0` = fixed-delay staircase (group `g` spins `g*N` wall-clock cycles after the global barrier before issuing; open-loop, no handoff traffic). |
| TT_DM_READ_VC       | unset (init VC=1) | Forces every reader core onto a single static request VC `k` (VC-validation sweep). Valid unicast request VCs are `0..5`. Read-only (`!sync`) path. Overridden by `TT_DM_READ_VC_BANDS`. |
| TT_DM_READ_VC_BANDS | 0 (off) | Splits the cores into `N` contiguous equal-count bands along `TT_DM_READ_VC_AXIS` and puts each band on its own static request VC (band index == VC; only the partition boundaries matter, not the VC identity). Targets the row-dominated return-path funnel. Takes precedence over `TT_DM_READ_VC`. Read-only path. |
| TT_DM_READ_VC_AXIS  | 0 (rows) | Banding axis for `TT_DM_READ_VC_BANDS`: `0` = rows (y, the contention/funnel axis), `1` = columns (x, the DRAM-distance/drain axis). |
| TT_DM_L1_PROVIDERS  | 0 (off) | Read-only experiment: instead of reading the interleaved DRAM buffer, readers read the L1 of `N` provider cores (round-robin per page), removing the DRAM controllers from the path. All readers read the same small shared resident buffer. Provider placement and the reserved reader set are set by `TT_DM_PROVIDER_LAYOUT`. Mutually exclusive with `TT_DM_RAND_OFFSET`. |
| TT_DM_PROVIDER_LAYOUT | 0 (leftcol) | Provider placement for `TT_DM_L1_PROVIDERS`. `0` = leftcol: first N cores of the leftmost column; that whole column is reserved so readers stay a clean rectangle. `1` = mirror: providers live in the two logical columns adjacent to the physical DRAM links (heatmap x=1 left, x=11 middle); both whole columns are always reserved (constant reader set across an N-sweep). N is split ~3:4 (left:right, matching the 3/4 active DRAM channels), clamped to 10 rows/column and spread evenly; the special case N=7 uses the exact DRAM-mirror rows (top-right provider at y=4). |
| TT_DM_RESERVE_COL   | 0 (off) | Matched DRAM baseline for the L1-provider experiment: reserves the same reader set as the active `TT_DM_PROVIDER_LAYOUT` (without serving from L1), so an apples-to-apples DRAM run uses the identical reader grid. Read-only path. |
