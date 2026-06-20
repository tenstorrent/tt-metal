# sdpa_decode/dataflow_common.hpp (read_k) — DEFERRED (Tier 2c)

**Kernel:** `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`
**Status:** deferred (two helper design gaps — runtime recipient count + runtime role)
**Validation:** none run (static deferral; no kernel edits, tree green)

## Verdict
`read_k`'s mcast IS a STAR vertical-column broadcast in topology (would be expressible by SenderPipe/
ReceiverPipe shape-wise), but two parameters that the helper requires at COMPILE TIME are RUNTIME here.

## STAR shape confirmation (positive)
- Sender (L629-690): `async_write_multicast` of full K^T chunk to vertical column rect
  (mcast_x, mcast_y0)->(mcast_x, mcast_y1); then `mcast_sem.set(VALID); mcast_sem.set_multicast(...)`
  — broadcasts its OWN cell, same sem id on all receivers => **src==dst STAR** (not a relay).
- Receiver (L691-697): `mcast_sem.wait(1); mcast_sem.set(0); async_atomic_barrier()` — no ack to
  sender => PRE_HANDSHAKE=false dialect. F1 = BARRIER + linked=false (the inverse-of-chain dialect).
- `mcast_sem_id = k_mcast_semaphore_id` IS compile-time (reader_decode_all.cpp:56, factory sets =2).

## Blocking gaps (negative)
1. **Runtime recipient count.** `KMcastParams.num_dests` is a pure runtime arg
   (reader_decode_all.cpp:106, factory pushes `num_dests = q_heads_parallel_factor - 1` as a
   runtime arg at sdpa_decode_program_factory.cpp:928). The helper's `NUM_ACTIVE_RECEIVER_CORES`
   is a compile-time template param only (KNOWN HELPER DESIGN GAP: runtime per-rect recipient count).
2. **Runtime sender/receiver role.** `do_mcast = (core.y % q_heads_parallel_factor == 0)` (factory
   L845) is a per-core RUNTIME flag — the SAME compiled binary runs as sender on some cores and
   receiver on others, branching at runtime (read_k `if (mcast_params.do_mcast)`). The helper's two
   faces (SenderPipe / ReceiverPipe) are DISTINCT types chosen at COMPILE time; migrated STAR
   siblings always know their role at compile time. No runtime-role dispatch in the v7 API.

## Decision
Either gap alone blocks migration. **Helper NOT modified.** No revert needed (no edits made). MOVE ON.
Note for a future helper round: closing gap (1) (runtime count) would also need gap (2) (runtime
role) to make this kernel migratable.
