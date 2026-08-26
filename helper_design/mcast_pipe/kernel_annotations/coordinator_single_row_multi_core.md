DERIVED FROM: the 2026-08-03 sort coordinator/reader/writer snapshot, `sort_program_factory.cpp`, `mcast_pipe.hpp`/`.inl` API v9, the three prior migration logs, and `archive/reconciliation/reconcile_2026-08-03.md`.

# coordinator_single_row_multi_core.cpp (sort) — API-v9 re-audit

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp`

Role: **coordinator / control-signal sender**. It broadcasts a doorbell to all worker cores once at row start and once per bitonic sub-stage. There is no data multicast.

Current API era: object `Noc` / `Semaphore<>` primitives, but not `mcast_pipe`.

## Required behavior (independent of the current spelling)

1. Do not release a row until every worker reader has announced readiness.
2. Release every worker with one rectangle multicast at row start and at every sub-stage.
3. After each sub-stage release, do not advance until all `Wt/2` compare-exchange pairs have completed their writes.
4. Reuse each synchronization cell only after its exact-match count has been consumed and reset.

The following are **historical implementation choices, not requirements**:

- The doorbell currently uses inverted polarity: the coordinator broadcasts `0`, workers wait for `0`, then restore `VALID`. API v9 need not preserve that level-flag implementation; a monotone Counter doorbell expresses the required fresh release more directly.
- The three semaphore IDs are host `constexpr` values (`0`, `1`, `2`) but are packed into runtime args.
- The ready and done waits are written adjacent to the multicast. They are operation-level phase barriers; they do not have to be the Pipe's built-in consumer-ready handshake.
- The current code uses a write barrier after the control multicast. API v9's remote-only `send_signal()` uses its baked-in SENT fence.

## Current protocol annotation

### Setup — lines 65–76

- L66 constructs the coordinator→workers doorbell semaphore.
- L67–74 construct **two distinct inbound counters**:
  - `cores_to_coordinator_ready_sem`: one increment per worker reader, once per row.
  - `cores_to_coordinator_done_sem`: one increment per processed pair, once per sub-stage.
- L76 sets the done target to `Wt / 2`, the number of pairs, not the number of worker cores.

The split is required by the current exact-match `wait(N)` discipline. With one shared counter, a fast reader's readiness for row `h+1` can arrive during the final done window for row `h`; the value can then overshoot `Wt/2`, and an exact wait would never match. Separate counters remove that cross-phase alias.

### Row-start phase — lines 182–194

- L183 waits for `number_of_dest` readiness increments.
- L184 resets only the ready counter to `0` after the exact-match wait succeeds.
- L187–193 multicasts the coordinator→workers doorbell over the worker rectangle.
- L194 drains the multicast write before the coordinator proceeds.

### Sub-stage phase — lines 202–217

- L205–211 multicasts the same doorbell for the next sub-stage.
- L212 drains the multicast write.
- L215 waits for exactly `Wt/2` writer confirmations.
- L216 resets only the done counter to `0`.

## Mapping to `mcast_pipe` API v9

The control channel maps without a helper change:

- Construct the no-handshake Counter form of `SenderPipe` over the current rectangle (equivalently, adopt semaphore 0 through an `Mcast2D` wire configured with `handshake=false` and `DataReady=Counter`) and call `send_signal()` for both row-start and sub-stage releases.
- Keep `ready_sem.wait(number_of_dest); ready_sem.set(0)` before the row-start signal.
- Keep `done_sem.wait(Wt / 2); done_sem.set(0)` after each sub-stage signal.

`PRE_HANDSHAKE=false` is deliberate: neither inbound counter is the Pipe's uniform per-send consumer-ready handshake. The two counts have different meanings and phase placement, so they remain explicit operation protocol around a control-only Pipe.

The prior runtime-fanout blocker is also stale for this program factory. `SortProgramFactorySingleRowMultiCore` is selected only after the hybrid capacity is exceeded; that implies `total_work_units / number_of_available_cores > 0`, and its L1013–1018 path selects every worker except the coordinator. The L1213–1215 rectangle therefore covers the full worker grid plus the coordinator source, and API v9's EXCLUDE-source fan-out (`McastRect::area() - 1`) equals the current `core_range.num_cores()` count.

The semaphore-ID blocker is not a helper capability gap. The factory declares the IDs as core-uniform `constexpr` values at L1187–1189; moving them from runtime packing into compile-time kernel arguments (or equivalent compile-time constants) is a host/kernel ABI refactor.

The inverted flag is likewise not required behavior. The strongest v9 formulation is the Counter control path: each `send_signal()` performs a multicast increment, each receiver waits for the next monotone round, and no doorbell reset or polarity conversion is needed. This matters because the row-start release and first sub-stage release can be back-to-back with no receiver-consumption ACK between them; repeating one level value can collapse those two events, while monotone rounds cannot. Semaphore 0 is already host-initialized to `0`, which is the required Counter starting state.

## Hazards and invariants

- **H3/H6 — stale level flag:** avoided rather than reproduced. The v9 Counter control path uses monotone `inc_multicast` + `wait_min`, so an early next signal cannot be lost and no level reset is required.
- **H8 — fan-out count:** use v9's rectangle-derived EXCLUDE-source count; do not preserve the runtime argument merely because the old factory packed it.
- **H11 — reset ownership:** the Counter doorbell removes cross-kernel doorbell reset ownership. The coordinator still owns reset of the two bounded inbound counters.
- **Exact-match counter reuse:** ready and done must remain separate and must be reset only after their respective wait succeeds. Their host initial value remains `0`; they are remotely incremented and must not be kernel-ctor initialized by the coordinator.
- **Fence style:** the current barrier is implementation history. The required behavior is that the remote doorbell is safely issued before reuse/return; v9's `send_signal()` supplies its materialized fence.

## Verdict

**migrated with API v9 in `7337302b564`; no helper redesign required.**

The paired host/kernel ABI and control-protocol rewrite is complete. The varying ready/done counts
remain explicit outside the control-only Pipe.

Validation closed: four focused multi-iteration control-only Counter device cases pass, along with
the 72-case helper suite and the mapped sort inventories.
