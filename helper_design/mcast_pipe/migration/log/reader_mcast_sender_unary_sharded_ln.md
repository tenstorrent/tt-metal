# reader_mcast_sender_unary_sharded_ln.cpp — v7
- Group G5 layernorm, sender (refactor, two-phase). Commit 00fa71996414099e717a2d7baecaca46dcbd097e. migrated v7 — PASS.
- Validation: test_layer_norm_sharded_single_stage[dtype=torch.bfloat16-...use_welford=True-h=256-w=512-num_cores_h=4-num_cores_w=4-block_ht=2-block_wt=4-subblock_wt=1] — PASS.

## Delta
Only PHASE-1 (control-flag broadcast) uses the helper. Phase-2 (monotone counter streaming via
reduce_sender_sem.set(block+2) + raw set_multicast, lines 273-305) stays DEFERRED RAW per the existing
comment — the Flag/Counter pipe verbs can't express the per-side counter base without desyncing receivers.

Phase-1 was on the old intermediate API (Staging::Flag enum + a trailing INITIAL_READY=INVALID param
that was removed in R6 + wrong arg order). Control-only (send_signal) -> PRE_HANDSHAKE=false, consumer omitted:
  SenderPipe<num_blocks-1, reduce_sender_sem_id, reduce_receiver_sem_id, Staging::Flag, false, INVALID>(noc, McastRect{...}) + send_signal(VALID)
  -> SenderPipe<noc_index, reduce_sender_sem_id, num_blocks-1, /*PRE_HANDSHAKE=*/false>(noc, McastRect<>{...}) + send_signal()
The raw reduce_receiver_sem.wait/set drain gate (the protocol gate preceding the flag) stays raw. The
removed INITIAL_READY=INVALID: v7's Flag ctor sets the local cell VALID, but phase-2 immediately
overwrites it with set(block+2) before its own mcast, and phase-1's send_signal broadcasts VALID as
intended — verified correct on device. diff_lines_removed: 4 (Staging/INITIAL_READY/reorder + send_signal arg).
