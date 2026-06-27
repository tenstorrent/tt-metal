# reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp (SEND side)

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp

API spelling: experimental OO wrapper. Uses raw `RemoteCoord` struct (not df:: helpers).
Role: sender for the PRE-allgather stage (computes partials only, NO data mcast). Block in `global_reduce_sender` lambda L85-180; called once L181.

## Block map

### Phase-1 handshake ONLY (flag + counter), EXCLUDE_SRC — NO DATA MCAST
- L99 `cb_partial_obj.wait_front(...)` — local partial ready.
- L102-113 `if num_blocks>1`:
  - L103 `reduce_sender_sem.set(VALID)`
  - L104 `reduce_receiver_sem.wait(num_blocks-1)` — COUNTER wait.
  - L105 `reduce_receiver_sem.set(0)` — counter reset (reused slot).
  - L106-112 `reduce_sender_sem.set_multicast(... num_blocks-1)` — mcast level flag VALID. **This is a flag-only multicast; no data is multicast in this kernel.**
- L116-179 gather reads only (`noc.async_read` + `async_read_barrier`, HOLE). No `async_write_multicast` anywhere.
- L182 `noc.async_write_barrier()` at top level — **F1 = barrier** (drains the sem mcast writes).

## Variant signature
- **F2 = phase-1 flag+counter hybrid** (no phase-2 in this half — data mcast is deferred to post_allgather kernel).
- **F3 = EXCLUDE_SRC**.
- **pre_handshake / no-data-mcast case**: this is the **"sem-flag-only multicast" sub-pattern** — `set_multicast` of a VALID flag with NO accompanying `async_write_multicast` of data. The Pipe `send()` must support a **flag-only mode** (mcast a 4-byte sem set with no payload).
- **F1 = barrier** (L182).

## Hazards / invariants
- INV: counter reset L105 after wait L104. Reused slot.
- HOLE: gather reads (L128-179) are the bulk of the kernel; only the L102-113 sem block + L182 barrier are Pipe-relevant.
- NEW SPELLING / FORK demand: flag-only mcast (no payload) must be a first-class Pipe.send() mode.
