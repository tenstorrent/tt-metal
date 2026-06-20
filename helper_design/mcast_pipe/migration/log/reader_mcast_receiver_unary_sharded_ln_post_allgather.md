# reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp — v7

- Group: normalization (layernorm post_allgather RECEIVER), tag clean (C1)
- Status: migrated v7 — PASS
- Validation: test_post_allgather_layernorm (single-chip, num_devices simulated by input chunking).
  Co-compiled with the migrated sender. Smoke num_devices=4 --dev PASS; family --run-all: 64 passed.
- (F1,F2,F3,pre_hs) = (none, FLAG, INCLUDE_SRC, NO)

## Delta
Raw set(INVALID)/reserve_back/wait(VALID)/push_back -> ReceiverPipe::receive():
  ReceiverPipe<reduce_sender_sem_id, /*PRE_HANDSHAKE=*/false> reduce_pipe(noc);
  cb_ex_global.reserve_back(...); reduce_pipe.receive(my_x[noc_index], my_y[noc_index]); cb_ex_global.push_back(...);
- PRE_HANDSHAKE=false: NO R->S ack (single sem reduce_sender_sem = CTA 1, the data-ready flag).
  CONSUMER_READY_SEM_ID omitted. receive() does not ack, so the sender coords are unused -> pass this
  core's own coords as a harmless dummy.
- ctor sets the flag cell INVALID, folding in the old pre-loop reduce_sender_sem.set(INVALID).
- receive() waits VALID + clears for next round; data lands in the reserved cb_ex_global slot via the
  sender's INCLUDE_SRC mcast (flag arrival => data arrival, INV4). diff_lines_removed: ~3.
