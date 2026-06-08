# reader_final_topk.cpp (topk)

Path: `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp`
Role: **final/coordinator (RECEIVER) core** of topk aggregation. Multicasts a "ready to receive" flag to all local sender cores, then waits for their data. Iterated over `Ht` rows.
API era: **modern** (`Noc`, `Semaphore<>`, `CircularBuffer`, mcast mode template).

## Semaphores
- `receiver_sem` (L26): final-core→senders readiness flag (mcast outbound).
- `sender_sem` (L27): senders→final-core data-arrived counter (inbound).

## Block instance — per-row reception — lines 34-57
- `final_values_cb.reserve_back(Wt_final)` / `final_indices_cb.reserve_back(Wt_final)` (L34-35): reserve dest CB space for all incoming tiles (set-up source/dest L1).
- `sender_sem.set(INVALID)` (L39): **reset** inbound data flag.
- `receiver_sem.set(VALID)` (L40): set local readiness flag (the value to be mcast).
- mcast: `receiver_sem.set_multicast<Noc::McastMode::EXCLUDE_SRC>(noc, noc_start_x,y, noc_end_x,y, num_dests)` (L45-46): broadcast readiness to all sender cores (self excluded — final core is not a sender).
- `noc.async_write_barrier()` (L47): flush the sem mcast.
- `sender_sem.wait(Wt_final)` (L52): wait until all `Wt_final` tiles have arrived (each sender increments by Kt; sum = Wt_final).
- `final_values_cb.push_back(Wt_final)` / `final_indices_cb.push_back(Wt_final)` (L56-57): commit received data.
- trailing `noc.async_write_barrier()` (L61) once after the loop.

## Mapping to Pipe
- **The cleanest receiver-side iterated block in the group.** It is a *receive-by-invitation* pattern: coordinator mcasts an invite flag, then waits on an inbound counter, with the dest CB reserved before and committed after.
- `Pipe::receive()` here = `set(reset inbound) ; set(local ready) ; mcast(ready, EXCLUDE_SRC) ; barrier ; wait(counter)`. The CB reserve/push wraps it.

## Forks
- F1: **barrier** (`async_write_barrier`, L47).
- F2: **flag for the outbound readiness** (`set(VALID)`, level) AND **counter for the inbound** (`sender_sem.wait(Wt_final)`); inbound reset via `set(INVALID)` (L39) each iteration. Both members.
- F3: **EXCLUDE_SRC** (L45) — final core is not a data sender; loopback excluded, no self-fill.
- KNOB pre_handshake: **fresh slot** — both `sender_sem` and `receiver_sem` reset at the top of every row (L39-40).

## HOLEs
- Outbound flag value (`VALID`) is mcast while inbound completion uses a *count* (`Wt_final`) — two semaphore roles in one logical receive. Helper must model the invite-flag and the arrival-counter as separate slots.
- The reciprocal sender is `writer_local_topk.cpp` — reset/polarity must be co-designed (sender resets its own local `receiver_sem` to INVALID, see that annotation).
