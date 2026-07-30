# flash_mla.hpp — DEFERRED (design-gap)

Kernel: `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`
Tier: 2d. Status: deferred (design-gap). No code change.

## Why
The K-chunk sender block (BRISC path, ~lines 322–447) has two structural blockers for v7:

1. **Runtime recipient/ack count.** `num_mcast_dests` is a RUNTIME arg — declared in both
   `ReaderArgs` (line 141) and `WriterArgs` (line 178), used as the mcast dest count AND the
   `receiver_ready_semaphore` ack target (lines 348, 356, 363, 497, 571, 578, 588, 597, 607).
   v7 `SenderPipe::NUM_ACTIVE_RECEIVER_CORES` is a **compile-time template param**. Runtime
   recipient count is inexpressible — the same design gap that deferred gn_v2,
   welford_..._gn_v2, and conv3d/writer.

2. **Mcast is one of two interleaved per-chunk branches + preprogram-state.** The mcast (data +
   flag + flush) only runs for `loop_iter < BRISC_MCAST_LOOPS` (line 329). After that the same
   sender switches to a preprogram-state, TRID-cycled per-page sharded read
   (`noc_async_read_one_packet_set_state` / `_with_state_with_trid`, lines 374–430) with an
   NCRISC/BRISC double-buffered sync semaphore. The receiver pre-handshake itself uses
   preprogram-state atomics (`unicast_atomic_inc_set_state` / `unicast_atomic_inc_with_state`,
   lines 319/437) — the deepseek mcast.hpp set-state family that proposed_helpers.md defers
   ("no mcast set-state in object API; future").

Either blocker alone defers; both apply.

## Verdict
DESIGN-GAP (runtime count + preprogram-state). Helper would need a runtime `num_dests` and a
set-state path. Do NOT touch the helper. Deferred.
