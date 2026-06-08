# reader_mcast_sender_unary_sharded_ln.cpp — TIER 3.2 (refactor-high)

**Status: MIGRATED (partial)** — commit `26540b54096`
**Validation: PASS** — `test_layer_norm_sharded_single_stage[...use_welford=True-h=256-w=512-num_cores_h=4-num_cores_w=4-block_ht=2-block_wt=4-subblock_wt=1]`
(JIT cache showed a fresh recompile of `layernorm_sharded_welford`; nodeid green.)

## Op / dispatch
ttnn.layer_norm sharded, NOT_DISTRIBUTED stage (LayerNormShardedProgramFactory). Object API
(`Noc`/`Semaphore<>`/`MulticastEndpoint`), `reduce_sender_sem` (cta 1) + `reduce_receiver_sem` (cta 0).

## Handshake blocks (C3 two-phase)
1. **Phase-1 (L124-135) control-flag broadcast** — `reduce_sender_sem.set(VALID); reduce_receiver_sem.wait(num_blocks-1); set(0); reduce_sender_sem.set_multicast(EXCLUDE_SRC, num_blocks-1)`.
2. **Phase-2 (L254-285) monotone-counter streaming** — per block: `reduce_sender_sem.set(block+2); async_write_multicast(data slice); reduce_sender_sem.set_multicast(...)`, interleaved with the gather `mcast_src_offset` advance.

## Assessment
- **Phase-1 = FIT (partial).** The flag broadcast is exactly `Pipe::send_signal(VALID)` =
  `raise_flag_` (set local cell VALID + `set_multicast<EXCLUDE_SRC>`) + `fence_` (flush). The
  consumed-drain `reduce_receiver_sem.wait(num_blocks-1)/set(0)` is the protocol gate and is NOT
  part of `send_signal()` (that's a `send()` PRE_HANDSHAKE concern, but this phase carries no data
  block) — so it stays raw, immediately preceding the `send_signal()`. Moving the local `set(VALID)`
  from before the wait to inside `send_signal()` (after the wait) is safe: the sender never reads
  its own `reduce_sender_sem` cell during the wait. The added `async_writes_flushed()` flush is a
  no-op-safe addition (only flag writes outstanding).
- **Phase-2 = NO FIT (deferred raw).** It reuses the SAME `reduce_sender_sem` cell as a monotone
  counter via `set(block+2)` (absolute set, not `inc_multicast`). The Pipe `Staging::Counter` send
  path uses `inc_multicast` (relative +value) and the receiver waits `wait_min(++round_)` starting
  at 1, whereas this protocol's absolute base is `block+2` (phase-1 already left the cell at VALID=1).
  Migrating one side flips set→inc semantics and desyncs the absolute counter base against the raw
  receiver. Cannot be expressed per-side; left RAW.

## Edit
- Added include `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`.
- Replaced the phase-1 flag-broadcast block with a `Pipe<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=false>`
  built from `reduce_sender_sem` (data_ready) + `reduce_receiver_sem` (consumed) and the runtime
  mcast rect `{start_x,start_y,end_x,end_y, num_blocks-1}`; kept the consumed-drain wait/set(0) raw;
  flag broadcast via `phase1_pipe.send_signal(VALID)`.

diff: +19 / -8.
