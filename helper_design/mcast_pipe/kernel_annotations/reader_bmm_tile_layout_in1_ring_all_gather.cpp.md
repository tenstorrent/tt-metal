# reader_bmm_tile_layout_in1_ring_all_gather.cpp — annotation

Role: Mostly DRAM-read + remote-CB; contains a **degenerate flag-only mcast** inside `do_signaling()`. Object API.

## What matches the family
`do_signaling()` (L46-70), privileged branch:
- L61: `pv_sem.wait(target_sem_value)` — gather all non-privileged cores' increments (counter).
- L62-63: `pv_sem.set(1)`, `sig_sem.set(1)` — reset/arm local flags.
- L64-65: **flag-mcast** `sig_sem.set_multicast(noc, start_x,y, end_x,y, num_signalling_semaphores)` — broadcast a GO flag to the rectangle. **No data mcast. One-shot (per batch b==0).**
Non-privileged branch (L66-69): `pv_sem.up(pv_core_x,y,1)` + `async_atomic_barrier()` — signal-back.

## Fork signature
- **F1**: BARRIER (the caller does `early_noc.async_write_barrier()` L93 / final `async_write_barrier()` L216). No flush.
- **F2**: COUNTER on the gather (`pv_sem.wait(target_sem_value)`), then LEVEL flag set+mcast for the GO. Hybrid.
- **F3**: the `set_multicast` here is plain (no INCLUDE/EXCLUDE_SRC choice — semaphore mcast). Privileged core sets its own `sig_sem` to 1 (L63) then mcasts — effectively loopback-by-local-set.
- **KNOB pre_handshake**: this is a one-time op-level start barrier, not a per-block dest-reuse handshake. KNOB N/A.

## Verdict
This is a **barrier / start-signal**, not the producer/consumer data Pipe. It is the recognition family's "lone flag mcast" — it co-occurs with NO data mcast. Per the recognition rule ("counts ONLY if the cluster co-occurs as the recognizable block"), this is a **borderline / defer**: it is a real flag-mcast handshake but lacks the data-block half. Migration: `defer/raw` (a Pipe `send()` flag-only mode could express it, but it's a one-shot collective barrier, different lifecycle).

## HOLEs
- L142/L201 `experimental::remote_cb_wait_front/pop_front` — global/remote CB sync, an entirely different mechanism (not NoC mcast). Out of family.
