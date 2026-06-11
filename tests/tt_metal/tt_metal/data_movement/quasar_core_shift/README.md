# Quasar Core Shift Bandwidth Benchmark

Quasar-only benchmark that measures L1-to-L1 NOC **write** bandwidth from a sender core to a receiver core, and compares bandwidth when the receiver is shifted one hop **left**, **right**, **up**, or **down** from a baseline position.

## Requirements

- Quasar architecture (`ARCH_NAME=quasar`)
- Simulator or emulator: set `TT_METAL_SIMULATOR` or `TT_METAL_EMULE_MODE=1`
- Slow dispatch: `TT_METAL_SLOW_DISPATCH_MODE=1` (required by `QuasarMeshDeviceSingleCardFixture`)

## Test Flow

1. Host writes a golden data pattern to sender L1.
2. Sender kernel issues `num_of_transactions` non-posted NOC `async_write` calls to receiver L1, then `async_write_barrier` (the barrier waits for all acks, i.e. data committed at the receiver).
3. Elapsed cycles are measured in-kernel with the RISC-V `rdcycle` CSR, bracketing the write loop + barrier. (The wall-clock debug register read via `get_timestamp()` crashes the simulator, so it is not used.)
4. Sender writes the cycle count to a dedicated L1 slot via the uncached L1 alias (`MEM_L1_UNCACHED_BASE`) so the CPU store bypasses the data cache and is visible to the host read-back; host reads it back.
5. Host reads receiver L1 and verifies data matches golden.
6. Bandwidth = `(num_of_transactions × bytes_per_transaction) / cycles` (bytes/cycle, column `bandwidth_Bpc`).

Only the **sender** runs a kernel; the receiver is a passive L1 target (same model as `one_to_one`).

## Coordinate Convention

User coordinates `(row, col)` map to logical `CoreCoord{x=col, y=row}` on the compute grid (`grid.x` columns, `grid.y` rows).

Example on a 4×8 grid (4 rows × 8 columns):

| User (row,col) | CoreCoord {x,y} | Shift from baseline (2,6) |
|----------------|-----------------|---------------------------|
| (2, 6) baseline | {6, 2} | — |
| (2, 5) left | {5, 2} | x − 1 |
| (2, 7) right | {7, 2} | x + 1 |
| (3, 6) up | {6, 3} | y + 1 |
| (1, 6) down | {6, 1} | y − 1 |

If a neighbor is out of grid bounds, that position is skipped with a warning.

## Test Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_id` | uint32_t | Unique test ID for grouping results (920, 921, …) |
| `sender` | CoreCoord | Logical coordinates of the sending core |
| `receiver_baseline` | CoreCoord | Baseline receiver; L/R/U/D neighbors are derived automatically |
| `num_of_transactions` | uint32_t | Number of NOC writes per measurement (default 256) |
| `bytes_per_transaction` | uint32_t | Payload size per write (swept: 64 B, 1 KB, 8 KB) |
| NOC VC | — | Fixed to `NOC_UNICAST_WRITE_VC` (2 on Quasar); do not use VC 0 |

## Test Cases

| GTest name | Test ID | Sender (user) | Baseline receiver (user) |
|------------|---------|-----------------|----------------------------|
| `QuasarCoreShiftCorner0` | 920 | (0, 0) | (2, 6) |
| `QuasarCoreShiftCorner1` | 921 | (3, 6) | (1, 1) |
| `QuasarCoreShiftNearBaseline` | 922 | (0, 0) | (1, 1) |

Each case runs **5 receiver positions** × **3 transaction sizes** = 15 measurements per test ID.
The size sweep is intentionally small: the simulator/emulator accumulates state across
sequential program launches and becomes unstable after ~50 in one process.

## Running the Tests

Build:

```bash
./build_metal.sh --build-tests
```

Run:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_SIMULATOR=<path> \
  ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*QuasarCoreShift*"
```

Optional DPRINT from the sender kernel:

```bash
TT_METAL_DPRINT_CORES=all ...
```

## Results

- A formatted table is printed via `log_info` (position, dx/dy, size, cycles, bandwidth, pass/fail).
- CSV output: `generated/quasar_core_shift/test_<test_id>.csv`

Columns: `test_id`, sender/baseline/receiver coords, `position`, `dx`, `dy`, `bytes_per_transaction`, `cycles`, `bandwidth_bpc`, `pass`.

This test does **not** use the Python profiler / `DeviceZoneScopedN` pipeline; bandwidth is computed on the host from in-kernel wall-clock cycles.

## Adding a New Corner

1. Pick the next free test ID (see [data_movement README](../README.md)).
2. Add an entry to `python/test_mappings/test_information.yaml`.
3. Copy a `TEST_F` block in `test_quasar_core_shift.cpp` and set `test_id`, sender, and baseline receiver (using `CoreCoord{col, row}`).
4. Rebuild and run with `--gtest_filter="*YourTestName*"`.

## Files

- `test_quasar_core_shift.cpp` — host test, sweep logic, CSV export
- `kernels/sender_bw.cpp` — timed NOC write kernel
