DERIVED FROM: the 2026-08-03 sort coordinator/reader/writer snapshot, `sort_program_factory.cpp`, `mcast_pipe.hpp`/`.inl` API v9, the three prior migration logs, and `archive/reconciliation/reconcile_2026-08-03.md`.

# reader_single_row_multi_core.cpp (sort) — API-v9 re-audit

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_multi_core.cpp`

Role: **worker / control-signal receiver**, plus the once-per-row readiness emitter. There is no data multicast in this kernel.

Current API era: object `Noc` / `Semaphore<>` primitives, but not `mcast_pipe`.

## Required behavior (independent of the current spelling)

1. Once per row, announce that this reader is ready before waiting for the row-start release.
2. Do not begin the row or any sub-stage until a fresh coordinator doorbell arrives.
3. Clear the local doorbell after each receive so the next wait cannot consume a stale signal.

It is **not required** that the doorbell value be `0`, that the resting value be `VALID`, or that the semaphore IDs arrive as runtime arguments. Those are properties of the current implementation.

## Current protocol annotation

### Setup — lines 56–59

- L57 constructs the coordinator→workers doorbell semaphore.
- L58 constructs the workers→coordinator **ready** counter. This is no longer shared with writer completion.
- L59 sets the local doorbell to `VALID`; the current protocol then waits for the coordinator to multicast `0`.

### Row-start phase — lines 61–70

- L67 increments the coordinator's ready counter once for this row.
- L68 drains the non-posted atomic increment.
- L69 waits for the coordinator's `0` doorbell.
- L70 restores `VALID`, clearing the current doorbell under the inverted convention.

### Sub-stage phase — lines 78–84

- L83 waits for a fresh coordinator doorbell.
- L84 clears it for the next sub-stage.
- No completion increment occurs here. The writer kernel emits completions only after its output writes have landed.

## Mapping to `mcast_pipe` API v9

The control receive maps without a helper change:

- Construct the no-handshake Counter form of `ReceiverPipe` for the adopted coordinator→workers semaphore (equivalently through the matching `Mcast2D` Counter wire).
- Keep the once-per-row `ready_sem.up(...)` plus atomic barrier explicit.
- Replace every `wait(0); set(VALID)` pair with `receive_signal()`, whose Counter form waits for the next monotone round and performs no reset.

`PRE_HANDSHAKE=false` is deliberate. The Pipe's built-in pre-handshake would acknowledge every receive, but sort needs a ready increment only at row start; sub-stage completion is a different signal emitted by the writer. The operation-specific ready leg therefore stays outside the Pipe.

The existing polarity is not a semantic requirement. The stronger v9 mapping adopts the host-initialized `0` cell as a monotone Counter: `receive_signal()` waits for rounds 1, 2, ... with `wait_min`. An early signal remains observable, including when the coordinator emits row-start and first-sub-stage releases back-to-back without a consumption ACK; there is no same-value event collapse or clear/set race.

The runtime semaphore-ID blocker is stale. `sort_program_factory.cpp` declares the doorbell and ready IDs as core-uniform `constexpr` values at L1187–1189 and merely serializes them into runtime args at L1322–1323. Moving those IDs to compile-time kernel inputs is a host/kernel ABI refactor; v9 does not need runtime semaphore IDs.

## Hazards and invariants

- **H3/H6 — fresh signal:** the Counter form uses monotone rounds and needs no worker clear.
- **H11 — reset ownership:** eliminated for the doorbell by Counter signaling; ready/done bounded counters retain coordinator reset ownership.
- **Ready ordering:** keep the ready `up()` before the row-start receive. Its atomic barrier is the current explicit drain and is independent of the Pipe's write fence.
- **Split counters:** reader readiness must target only the ready semaphore. Sending it to the writer-done semaphore would recreate the Ht≥2 exact-match overshoot deadlock.
- **Lifecycle:** the Counter doorbell remains host-initialized to `0` and is never reset. The coordinator-side ready counter also remains host-initialized because workers write it remotely.

## Verdict

**migrated with API v9 in `7337302b564` as the coordinator's paired receiver.**

The monotone-control and argument-ABI refactor was migrated atomically with the coordinator.

Validation closed: focused control-only Counter coverage and all mapped sort inventories pass.
