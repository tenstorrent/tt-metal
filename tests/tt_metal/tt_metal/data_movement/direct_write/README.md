# Direct Write Data Movement Tests

This test suite implements tests that measure the functionality and performance (i.e. bandwidth) of inline direct write transactions between two Tensix cores using both stateful and non-stateful NoC APIs.

## Test Flow
Two Tensix cores are used: one sender core and one receiver core. The sender kernel performs direct writes to memory locations on the receiver core's L1 memory using either the stateful or non-stateful NoC API. The stateful approach uses `noc_inline_dw_write_set_state` to cache address and/or value information, followed by `noc_inline_dw_write_with_state` calls that can reuse the cached state. The non-stateful approach uses `noc_inline_dw_write` calls that must provide full address and value information for each transaction.

Test attributes such as transaction counts, write patterns, and API approaches as well as latency measures like kernel and pre-determined scope cycles are recorded by the profiler. The receiver memory is validated after completion to ensure all writes were performed correctly. Additionally, both tests verify the correctness of the writes by reading back the written values and comparing them against expected results.

Test expectations are that validation checks pass and sufficient test attribute data is captured by the profiler for higher level bandwidth/regression checks.

## Test Parameters
| Parameter                 | Data Type             | Description |
| ------------------------- | --------------------- | ----------- |
| test_id                   | uint32_t              | Test id for signifying different test cases. Can be used for grouping different tests. |
| sender_core_coord         | CoreCoord             | Logical coordinates for the sender core. |
| receiver_core_coord       | CoreCoord             | Logical coordinates for the receiver core. |
| num_writes                | uint32_t              | Number of direct write transactions that will be issued. |
| write_value_base          | uint32_t              | Base value for write data. Values may be incremented from this base. |
| use_posted_writes         | bool                  | Whether to use posted or non-posted writes. |
| same_destination          | bool                  | Whether all writes target the same address or different addresses. |
| use_stateful_approach     | bool                  | Whether to use stateful or non-stateful NoC API. |
| same_value                | bool                  | Whether to write the same value repeatedly (stateful only optimization). |
| addr_stride               | uint32_t              | Address increment when writing to different destinations. |
| noc_id                    | NOC                   | Which NoC to use for the transactions (NOC_0 or NOC_1). |

## Test Cases
Two different test cases are implemented using the above parameters.
Each test case has multiple runs, and each run has a unique runtime host id, assigned by a global counter.

1. **Direct Write Performance Comparison (ID: 500)**: Provides a controlled comparison between stateful and non-stateful approaches. It consistently writes different values to the same destination address, testing both posted and non-posted write modes. This establishes a baseline performance comparison demonstrating the advantage of address caching in stateful direct writes.

2. **Direct Write Address Pattern (ID: 501)**: Comprehensively evaluates performance across different usage patterns by testing all combinations of address patterns (same vs different destinations), value patterns (same vs different values), and API approaches (stateful vs non-stateful). The test includes proper usage of the stateful API for `same_value` scenarios where identical values are set once in `set_state` and reused across multiple writes by passing false as the template parameter.

3. **Multicast Inline Direct Write (ID: 507)**: Tests multicast direct write functionality using the non-stateful NoC API to broadcast writes from a single sender core to multiple receiver cores simultaneously. The receiver set is the largest rectangle of the compute grid that excludes the sender, and the test runs it for both address patterns (same destination vs different destinations with address stride). Each receiver core's memory is independently validated to ensure all multicast writes were delivered correctly with expected values. This test verifies the correctness of the multicast inline DW write primitive over the available subordinate cores.

## Quasar Notes
All three tests skip at runtime on Quasar because `noc_inline_dw_write` does not work on that target. Measured on `emu-quasar-2x3_DISPATCH`:

- A posted write completes, but the destination L1 word keeps its initial value (test 507 read back `0x00000000` where `0x1234000f` was expected).
- A non-posted write never acks, so `noc_async_write_barrier` spins forever and the test hangs.
- Writes that advance the destination address hang the same way. This is consistent with `noc_fast_write_dw_inline` deriving its byte-enable as `be << (addr & (NOC_WORD_BYTES - 1))`; Quasar's `NOC_WORD_BYTES` is 256, so any address that is not 256B-aligned shifts a `uint32_t` by 32 or more.

The suite itself is no longer the blocker. It was ported from the Metal 1.0 `DataMovementConfig` path — which `CreateKernel` rejects on Quasar — to the Metal 2.0 `KernelSpec` / `DataMovementGen2Config` path used by the `_2_0` tests in the neighbouring suites, and it passes on Gen1. Remove the skips once the NoC V2 inline-write path is fixed. Tracked by issue #55386.
