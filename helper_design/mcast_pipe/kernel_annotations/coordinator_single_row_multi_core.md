# coordinator_single_row_multi_core.cpp (sort)

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp`
Role: **coordinator/SENDER** half of the sort bitonic-stage handshake. Drives a rectangle of worker cores stage by stage. Iterated over `Ht` rows and over bitonic sub-stages.
API era: legacy free-function.

## Setup
- `semaphore_ptr` = cores→coordinator counter (L49-50, the inbound counter the coordinator waits on).
- `semaphore_global_multicast_addr = get_noc_multicast_addr(start..end, coordinator_to_cores_semaphore_id)` (L51-56): one rectangle covering all workers, computed once (outside loops).

## Block instances (TRUE iterated mcast-handshake)

### Block A — per-row start handshake — lines 95-100
- `noc_semaphore_wait(semaphore_ptr, number_of_dest)` (L95): wait until all `number_of_dest` workers report ready.
- `noc_semaphore_set(semaphore_ptr, 0)` (L96): **reset** the counter (flag/level discipline).
- mcast: `noc_semaphore_set_multicast(coordinator_to_cores_semaphore_id, semaphore_global_multicast_addr, number_of_dest)` (L99) — broadcast "go" flag to the whole worker rectangle.
- `noc_async_write_barrier()` (L100): flush the sem mcast before continuing.

### Block B — per-substage handshake — lines 110-117 (inside `stage`×`sub` loops)
- mcast first: `noc_semaphore_set_multicast(...)` (L111-112) — release workers for this sub-stage.
- `noc_async_write_barrier()` (L113).
- `noc_semaphore_wait(semaphore_ptr, number_of_confirmations)` (L116) where `number_of_confirmations = Wt/2` (L58): wait for the compare-exchange confirmations.
- `noc_semaphore_set(semaphore_ptr, 0)` (L117): **reset**.

Note ordering difference: Block A is wait→reset→mcast→flush; Block B is mcast→flush→wait→reset. Both are the same `send+wait+reset` cluster with different phase offsets within the loop.

## Mapping to Pipe
- This is the **SENDER/coordinator** that owns a *broadcast go-flag* + a *barrier on an inbound arrival counter*.
- `Pipe::send()` = `set_multicast(flag)` + `async_write_barrier`.
- The inbound wait+reset is the *counter-handshake* companion (`wait(N); set(0)`), i.e. a level-flag (F2 flag, not monotone counter).
- One rectangle, computed once and reused every iteration.

## Forks
- F1: **barrier** (`noc_async_write_barrier` after every mcast, L100/L113). Consistent barrier discipline.
- F2: **flag/level** — inbound counter is reset to 0 each iteration (`set(0)`), not a monotone wait_min.
- F3: legacy API — loopback opaque. Coordinator core is a *separate* core from the workers (it does index-gen + I/O), so it is naturally **EXCLUDE_SRC** (not in the worker rectangle). HOLE: not provable from kernel, but the coordinator does not wait on its own go-flag, consistent with EXCLUDE_SRC.
- KNOB pre_handshake: **dest reused** — the same `semaphore_ptr` counter slot is reset and reused across all sub-stages; go-flag slot likewise reused (reset on the worker side, see reader/writer annotations).

## HOLEs
- The go-flag is reset on the *worker* side (`noc_semaphore_set(VALID)` in the reader/writer), not here. A typed Pipe split across two kernels must agree on who resets which slot — cross-kernel reset ownership is a hazard.
- `number_of_confirmations = Wt/2` is an op-specific count; the helper must take the expected-arrival count as a parameter.
