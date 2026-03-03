# NOC Estimator Tests

Comprehensive performance sweep tests designed to generate profiling data for the NoC estimator. These tests systematically exercise every major NOC communication pattern across a wide range of transaction sizes, grid configurations, and transfer mechanisms to produce the data needed for accurate latency/bandwidth prediction.

## Dispatch Mode
All tests use **fast dispatch** via `GenericMeshDeviceFixture`.

## NOC API
- **L1 kernels** use the experimental device 2.0 NOC API (`experimental::Noc`, `experimental::UnicastEndpoint`, `experimental::MulticastEndpoint`).
- **DRAM kernels** use `TensorAccessor` with the experimental 2.0 NOC API. Buffers are allocated via `CreateBuffer(InterleavedBufferConfig)` on the host.

## Test Cases

### L1 Tests

| Test Name                    | ID  | Pattern          | Description                                                       |
|------------------------------|-----|------------------|-------------------------------------------------------------------|
| NocEstimatorL1OneToOne       | 800 | One to One       | Unicast write between two Tensix cores.                           |
| NocEstimatorL1OneFromOne     | 801 | One from One     | Unicast read between two Tensix cores.                            |
| NocEstimatorL1OneToAll       | 802 | One to All       | One core writes to a grid (unicast, multicast, multicast linked). |
| NocEstimatorL1OneFromAll     | 803 | One from All     | One core reads from all cores in the grid.                        |
| NocEstimatorL1AllToAll       | 804 | All to All       | Every core in a grid writes to every other core.                  |
| NocEstimatorL1AllFromAll     | 805 | All from All     | Every core in a grid reads from every other core.                 |
| NocEstimatorL1OneToRow       | 806 | One to Row       | One core writes to an entire row (unicast/multicast).             |
| NocEstimatorL1RowToRow       | 807 | Row to Row       | Every core in a row writes to every core in the row.              |
| NocEstimatorL1OneToColumn    | 808 | One to Column    | One core writes to an entire column (unicast/multicast).          |
| NocEstimatorL1ColumnToColumn | 809 | Column to Column | Every core in a column writes to every core in the column.        |

### DRAM Read Tests

| Test Name                             | ID  | Layout      | Cores     | Description                                        |
|---------------------------------------|-----|-------------|-----------|----------------------------------------------------|
| NocEstimatorDRAMShardedOneFromOne     | 810 | Sharded     | 1 core    | Single core reads from one dedicated DRAM bank.    |
| NocEstimatorDRAMShardedAllFromAll     | 811 | Sharded     | Full grid | Every core reads from its assigned bank.           |
| NocEstimatorDRAMInterleavedOneFromAll | 812 | Interleaved | 1 core    | Single core reads, cycling through all DRAM banks. |
| NocEstimatorDRAMInterleavedAllFromAll | 813 | Interleaved | Full grid | Every core reads, each cycling through all banks.  |

### DRAM Write Tests

| Test Name                            | ID  | Layout      | Cores     | Description                                         |
|--------------------------------------|-----|-------------|-----------|-----------------------------------------------------|
| NocEstimatorDRAMShardedOneToOne      | 814 | Sharded     | 1 core    | Single core writes to one dedicated DRAM bank.      |
| NocEstimatorDRAMShardedAllToAll      | 815 | Sharded     | Full grid | Every core writes to its assigned bank.             |
| NocEstimatorDRAMInterleavedOneToAll  | 816 | Interleaved | 1 core    | Single core writes, cycling through all DRAM banks. |
| NocEstimatorDRAMInterleavedAllToAll  | 817 | Interleaved | Full grid | Every core writes, each cycling through all banks.  |

DRAM read and write tests run in isolation (one kernel per core) to avoid L1 contention. All DRAM tests sweep both NOC_0 and NOC_1.

## Sweep Parameters

Each test sweeps over a combination of the following parameters:

| Parameter              | Values                                      | Description                                                    |
| ---------------------- | ------------------------------------------- | -------------------------------------------------------------- |
| `num_of_transactions`  | 1, 4, 16, 64, 256                           | Number of NOC transactions issued per destination.             |
| `pages_per_transaction`| 1, 2, 4, ..., up to arch-dependent max      | Pages per transaction (transaction size = pages x page size).  |
| `bytes_per_page`       | Architecture-dependent (derived at runtime) | Bytes per page, based on physical L1/DRAM constraints.         |
| `same_axis`            | true, false                                 | Whether source and destination are on the same NOC axis.       |
| `stateful`             | true, false                                 | Whether to use stateful writes/reads (set_state + with_state). |
| `loopback`             | true, false                                 | Whether the master core is inside the multicast rectangle.     |
| `mechanism`            | Unicast, Multicast, Multicast Linked        | NOC transfer mechanism.                                        |
| `grid_size`            | 2x2, 3x3, 5x5, 8x8, full device grid        | Grid dimensions for multi-core patterns.                       |

Not all parameter combinations apply to every test. For example, `same_axis` only applies to single-core patterns, `loopback` only applies to multicast patterns, and stateful unicast writes are constrained to sub-packet sizes (8 KB on Wormhole, 16 KB on Blackhole).

## Profiler Metadata

Each kernel logs structured metadata via `DeviceTimestampedData` for post-processing by the profiler and Python analysis scripts. The logged fields are:
- Test ID, NOC index, number of transactions, transaction size
- Memory type (L1/DRAM interleaved/DRAM sharded), mechanism (unicast/multicast/multicast linked), pattern
- Number of subordinates, same axis, stateful, loopback

## Kernels

| Kernel       | File                       | Processor | Description                                                              |
|--------------|----------------------------|-----------|--------------------------------------------------------------------------|
| L1 Writer    | `kernels/writer.cpp`       | RISCV_0   | Unicast single/multi, multicast, and multicast linked L1 writes.         |
| L1 Reader    | `kernels/reader.cpp`       | RISCV_1   | Single-source and multi-source L1 reads.                                 |
| DRAM Reader  | `kernels/dram_reader.cpp`  | RISCV_1   | Reads from DRAM to L1; interleaved (round-robin) or sharded (fixed bank).|
| DRAM Writer  | `kernels/dram_writer.cpp`  | RISCV_0   | Writes from L1 to DRAM; interleaved or sharded.                          |
| Log Helpers  | `kernels/log_helpers.hpp`  | --        | Shared constants and `log_estimator_metadata` helper.                    |
| Barrier Sync | `kernels/barrier_sync.hpp` | --        | Global barrier synchronization for multi-core DRAM tests.                |

## Running Tests

```bash
# Build
./build_metal.sh --build-tests

# Run all NOC estimator tests, --report flag is for generating csv reports
pytest tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py --timeout=1800 --gtest-filter="NocEstimator" --report

```
