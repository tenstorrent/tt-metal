# reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp (RECEIVE side)

Status: **fully end-to-end migrated at API v10** as part of
`layernorm-sharded-pre-allgather`.

Path: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp`

Role: receiver for the pre-allgather partial-statistics reduction. In two-stage
mode the same binary also runs on non-global line coordinators, where it uses
the helper sender face for that row or column.

## Helper-owned wire

- Host binding: `PreAllGatherMcast` in `sharded_layernorm_factory_helpers.*`.
- Whole-grid reduce: every non-global core uses the handshaked Flag `Mcast2D`
  receiver face and calls `receive_signal()`.
- Two-stage reduce: the first core on each row/column calls `send_signal()`;
  the remaining line members call `receive_signal()` on the matching `Mcast1D`.
- Existing semaphore descriptors are adopted without changing their host
  initialization.
- The complete six-word CT and four-word RT ranges are prepended as an opaque
  block. Operation arguments start at `McastArgs::next_*_args_offset()`.
- `receive_signal()` owns flag clear, consumer-ready increment, and ready wait.

## Operation-owned behavior

- Partial-statistics gather reads and CB ownership.
- `reduce_second_stage_semaphore_id` and second-stage completion.
- The final atomic barrier that drains operation-owned semaphore atomics.

## Validation and performance

- Exact 8x4 BFLOAT8_B RMSNorm node passed under `--dev` from a fresh isolated
  cache; both receiver variants produced ELF artifacts.
- `LN-PRE-ALLGATHER`: 126 passed; `LN-POST-ALLGATHER`: 136 passed;
  `LN-SHARDED`: 208 passed.
- New offset whole-grid, row-wise, and column-wise host geometry cases passed
  inside `McastHostFixture` 28/28; helper device suite passed 77/77.
- Exact-node LayerNorm device-kernel durations were 2,583, 2,564, 2,563, and
  2,656 ns (median 2,563.5 ns). Per-kernel delta is N/A because the profiler
  cannot isolate this face from the other LayerNorm data-movement kernels.
