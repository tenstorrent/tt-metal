# reader_mcast_sender_unary_sharded_ln_post_allgather.cpp — v7

- Group: normalization (layernorm post_allgather SENDER), tag clean (C1)
- Status: migrated v7 — PASS
- Validation: test_post_allgather_layernorm (single-chip, num_devices simulated by input chunking).
  Smoke: num_devices=4 BFLOAT8_B is_rmsnorm=False --dev PASS. Family --run-all: 64 passed, 0 failed.
- (F1,F2,F3,pre_hs) = (barrier, FLAG, INCLUDE_SRC, NO)

## Delta
Raw two-lambda block (global_reduce_sender data mcast + global_semaphore_set flag mcast) ->
one SenderPipe::send():
  SenderPipe<noc_index, reduce_sender_sem_id, num_blocks-1, /*PRE_HANDSHAKE=*/false>
      reduce_pipe(noc, McastRect<>{...});
  reduce_pipe.send(cb_stats_reduced.read_ptr, cb_ex_global.read_ptr, stats_tiles*num_tiles_per_worker_bytes);
- True LOOPBACK: src (cb_stats_reduced c_21) != dst (cb_ex_global c_15), sender ∈ rect -> helper's
  INCLUDE_SRC loopback path (+1 self-copy). NUM_ACTIVE_RECEIVER_CORES = num_blocks-1 (recipients excl.
  self); the old raw used INCLUDE_SRC + num_blocks count.
- PRE_HANDSHAKE=false: receivers mcast into a fresh reserve_back slot; no R->S ack sem exists (single
  sem reduce_sender_sem = CTA 1). CONSUMER_READY_SEM_ID omitted.
- send() couples data + VALID flag (data-before-flag, same VC, flush). The intervening cb_ex_global
  push_back / cb_stats_reduced pop_front are local CB bookkeeping, reordered after send().
- Dropped: MulticastEndpoint, the explicit Semaphore<> reduce_sender_sem, both lambdas, the pre-flag
  reduce_sender_sem.set(VALID) (ctor sets local cell VALID; send re-asserts). diff_lines_removed: ~24.
