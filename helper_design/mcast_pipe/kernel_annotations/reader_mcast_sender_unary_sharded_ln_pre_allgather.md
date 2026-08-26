# reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp (SEND side)

Status: **fully end-to-end migrated at API v10** as part of
`layernorm-sharded-pre-allgather`.

Path: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp`

Role: global sender for the pre-allgather partial-statistics reduction. This
kernel sends only the readiness signal; the gathered partials use operation-owned
unicast reads.

## Helper-owned wire

- Host binding: `PreAllGatherMcast` in `sharded_layernorm_factory_helpers.*`.
- Whole-grid reduce: handshaked Flag `Mcast2D`, sender inside the dense rectangle,
  fan-out `grid area - 1`.
- Two-stage reduce: handshaked Flag `Mcast1D`, one fixed sender per row or column,
  fan-out `num_blocks_first_stage - 1`.
- Existing `reduce_sender_semaphore_id` and `reduce_receiver_semaphore_id` are
  adopted as data-ready and consumer-ready descriptors.
- The complete six-word CT and four-word RT ranges are prepended as an opaque
  block. Operation arguments start at `McastArgs::next_*_args_offset()`.
- `send_signal()` owns ready-wait, counter reset, and multicast of `VALID`.

## Operation-owned behavior

- Local partial CB readiness and all gather reads.
- `reduce_second_stage_semaphore_id` and the two-stage completion protocol.
- The final write barrier.

## Validation and performance

- Exact 8x4 BFLOAT8_B RMSNorm node passed under `--dev` from a fresh isolated
  cache; the sender ELF was confirmed.
- `LN-PRE-ALLGATHER`: 126 passed; `LN-POST-ALLGATHER`: 136 passed;
  `LN-SHARDED`: 208 passed.
- Host/helper suites: 28/28 and 77/77.
- Exact-node LayerNorm device-kernel durations were 2,583, 2,564, 2,563, and
  2,656 ns (median 2,563.5 ns). Per-kernel delta is N/A because no
  operation-matched pre-migration profile exists and the DM envelope includes
  other data-movement kernels.
