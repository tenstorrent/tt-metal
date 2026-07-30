# reader_single_row_multi_core.cpp (sort)

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_multi_core.cpp`
Role: **worker/RECEIVER** half of the sort handshake (reader side). Iterated over `Ht` and bitonic sub-stages.
API era: legacy free-function.

## Setup
- `semaphore_ptr` = coordinator→cores go-flag slot (L40-41).
- `noc_semaphore_set(semaphore_ptr, VALID)` (L42): initialize go-flag to VALID (worker waits for it to be set to 0 by the coordinator's mcast). Inverted-flag convention.
- `coordinator_core_addr` = unicast addr of the cores→coordinator counter (L43-44).

## Block instances (RECEIVER half)

### Block A — per-row ready/start — lines 52-55
- `noc_semaphore_inc(coordinator_core_addr, 1)` (L52): report "ready" to the coordinator's counter.
- `noc_async_atomic_barrier()` (L53): **flush the atomic inc** (new spelling — atomic barrier, not write barrier).
- `noc_semaphore_wait(semaphore_ptr, 0)` (L54): wait for coordinator's mcast go-flag (value 0).
- `noc_semaphore_set(semaphore_ptr, VALID)` (L55): **reset** go-flag for next round.

### Block B — per-substage start — lines 68-69
- `noc_semaphore_wait(semaphore_ptr, 0)` (L68): wait for coordinator go.
- `noc_semaphore_set(semaphore_ptr, VALID)` (L69): reset.
(Confirmation back to coordinator is emitted by the **writer** kernel, not here.)

## Mapping to Pipe
- `Pipe::receive()` = `wait(flag) ; reset(flag)` — the inverted-flag (`wait(0); set(VALID)`) variant.
- The outbound `inc + atomic_barrier` (Block A only) is the *ready signal* that primes the coordinator's counter — the receiver-priming companion to `send()`.

## Forks
- F1: **atomic barrier** (`noc_async_atomic_barrier`, L53) — distinct from the coordinator's `noc_async_write_barrier`. Because the outbound op is an atomic inc, the correct flush is the atomic barrier. **New fork member / spelling to add to F1.**
- F2: **flag/level**, inverted polarity (wait for 0, reset to VALID). Coordinator uses normal polarity (set flag to N via mcast). Polarity mismatch is intentional: mcast `set_multicast(sem, addr, N)` writes N into the slot; worker `wait(0)`... HOLE: the value written by `set_multicast` vs the `wait(0)` target needs reconciling (see hazard below).
- F3: worker is in the coordinator's mcast rectangle (it is a destination). Coordinator is EXCLUDE_SRC relative to workers.
- KNOB pre_handshake: **dest reused** — single go-flag slot reset every iteration.

## HOLEs / hazards
- **Reset-ownership split across kernels**: worker resets the go-flag (`set(VALID)`), coordinator never does. If the helper owns reset, the two kernels must agree on which side resets.
- **Polarity/value reconciliation**: coordinator `noc_semaphore_set_multicast(sem, addr, number_of_dest)` broadcasts the *value of `sem`* (whatever the coordinator's local slot holds), while worker `wait(0)`. The `VALID`/`INVALID` constants encode this; the helper must surface flag polarity as a parameter, not hardcode.
