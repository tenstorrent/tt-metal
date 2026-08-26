# reader_argmax_interleaved_multicore.cpp

Path: `ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp`
Role: SPMD reader run by ALL cores; one core (`reduce_core_id`) acts as coordinator/reducer. Iterated over `outer_dim_units` (`k` loop).
API era: **modern** (`Noc`, `Semaphore<>`, `CircularBuffer`, mcast mode template).

## Semaphores
- `start_sem` (L347): coordinator→workers start flag (counter style).
- `done_sem` (L348): workers→coordinator done counter.
- `done_sem.set(0)` (L353) once before loop.

## Block instances

### Block 1 — SENDER half (reduce core) — lines 358-374 (per `k`)
- `done_sem.set(0)` (L359): **reset** the inbound done counter for this iteration (fresh-slot pre_handshake).
- `start_sem.set(k + 1)` (L360): set local start flag to monotone value `k+1`.
- mcast cluster (two rectangles):
  - `start_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(noc, start_core_x0,y0, end_core_x0,y0, num_cores0)` (L364-365) — rectangle **containing the source/reduce core** → INCLUDE_SRC fills self.
  - `start_sem.set_multicast<Noc::McastMode::EXCLUDE_SRC>(noc, ...1..., num_cores1)` (L369-370) — disjoint rectangle, guarded by `num_cores1 > 0`.
  - `noc.async_write_barrier()` (L373).
  - Guarded: only mcast `if (k > 0)` for group0, and `if constexpr (num_cores > 1)`.

### Block 1 — RECEIVER half (all cores) — lines 378-380
- `if (k > 0) start_sem.wait(k + 1)` (L378-379): wait for the monotone start value. **F2 = monotone counter** (`k+1` grows each iter; no reset of `start_sem`).

### Block 2 — done handshake (per `k`, not-reduce_all path) — lines 417-424
- `done_sem.up(noc, reduce_core_x, reduce_core_y, 1)` (L417): worker increments coordinator's done counter (unicast atomic).
- `noc.async_atomic_barrier()` (L418): flush atomic.
- `if (is_reduce_core) { if constexpr (num_cores>1) done_sem.wait(num_cores); }` (L421-423): coordinator waits for all cores. **F2 = counter** with explicit reset at top of next iter (L359).
- (Data to the reducer is unicast `noc.async_write` at L402-414, barrier L414 — not mcast.)

### Block 2' — reduce_all variant — lines 452-475
Same structure as Block 2 but outside the `k` loop (single shot): `done_sem.up` (L468) + `async_atomic_barrier` (L469) + coordinator `done_sem.wait(num_cores)` (L474).

## Mapping to Pipe
- **The reference two-rectangle, mode-templated sender.** `start_sem.set_multicast<MODE>(...)` over a *destination set* {INCLUDE_SRC rect, EXCLUDE_SRC rect} is exactly `Pipe::send(flag)` with explicit loopback control per rectangle.
- `start_sem.wait(k+1)` = `Pipe::receive()` with a monotone counter predicate (no reset needed — the value itself advances).
- `done_sem` (up + wait + reset) = the inbound counter companion.

## Forks (this kernel exhibits BOTH sides of multiple forks)
- F1: **barrier** for the mcast (`async_write_barrier`, L373) AND **atomic barrier** for the unicast inc (`async_atomic_barrier`, L418/L469). Both members present.
- F2: **monotone counter** for `start_sem` (`set(k+1)`/`wait(k+1)`, NEVER reset) — the canonical wait_min-equivalent / no-reset counter. AND **counter+reset** for `done_sem` (`set(0)` each iter, `wait(num_cores)`). Both members present in one kernel.
- F3: **INCLUDE_SRC self-fill** (group0, L364) AND **EXCLUDE_SRC** (group1, L369) — both members present, chosen by which rectangle contains the source core. This is the cleanest F3 evidence in the whole group.
- KNOB pre_handshake: **fresh slot** — `done_sem.set(0)` (L359) explicitly resets before each iteration's mcast; `start_sem` uses a fresh monotone value instead of reset.

## HOLEs
- Two rectangles with *different* mcast modes in a single logical send — Pipe must accept a dest set where each entry carries its own loopback mode, or expose two sends. The INCLUDE_SRC rectangle existing purely to fill the source's own copy is a self-fill idiom the helper should encapsulate.
- `start_sem` monotone-counter (no reset) vs `done_sem` reset-counter coexist — helper needs both F2 modes selectable per pipe.
