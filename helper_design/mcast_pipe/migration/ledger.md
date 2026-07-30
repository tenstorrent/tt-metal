# mcast_pipe migration ledger — API v9, reconciled 2026-07-30

Source of truth: `ledger.json`. Test inventories are in `test_map.json`;
failure isolation and JIT evidence are in the per-kernel migration logs.

The v8 July ledger is historical input, not the current result. Every one of
its 19 production migrations was re-audited against the current v9 helper and
the `origin/llk_helper_library` baseline at `54d8dfb7bef`.
Remediation also added the previously raw Conv 1D sender and both GroupNorm v2
senders, producing the current 22-row production candidate set.

## Current migrations under API v9 (10)

| Area | Role | Kernel | Validation |
|---|---|---|---|
| matmul | sender/hybrid | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `MM-IN1-ALL`: 302 passed, 188 expected skips |
| matmul | receiver | `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `MM-IN1-ALL` / receiver subset and exact JIT evidence |
| Conv | sender | `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `CONV-HEIGHT`: 49 passed, 16 expected skips; 14 DRAM regressions; runtime active-ack override |
| Conv | receiver | `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | `CONV-HEIGHT`: 49 passed, 16 expected skips; 14 DRAM regressions |
| Conv | sender | `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `CONV-BLOCK`: 49 passed, 16 expected skips; 14 DRAM regressions |
| Conv | receiver | `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | `CONV-BLOCK`: 49 passed, 16 expected skips; 14 DRAM regressions |
| normalization | receiver | `reader_mcast_receiver_unary_sharded_gn_v2.cpp` | legacy inventory: 124 passed, 8 expected skips |
| normalization | sender | `reader_mcast_sender_unary_sharded_gn_v2.cpp` | legacy inventory: 108 passed, 2 expected skips; fixed/default nodes: 19 passed, 6 expected skips |
| normalization | receiver | `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp` | Welford inventory: 116 passed, 8 expected skips |
| normalization | sender | `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` | Welford inventory: 108 passed, 2 expected skips; fixed/default nodes: 19 passed, 6 expected skips |

## Deferred or rolled back (12)

All twelve files match the target baseline exactly. They are not partial
migrations.

| Area | Kernel(s) | Blocker |
|---|---|---|
| matmul | `reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp` | typed/custom `VALID` / `IGNORE_BATCH` control values |
| matmul | `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | independent data/signal loopback modes and sender-side ready participation |
| Conv | `activation_reader_width_sharded.cpp` | prior port failed 25 numeric cases; helper completion semantics changed afterward, so a fresh full re-port/retest is required |
| normalization | post-allgather LayerNorm sender/receiver | explicit INCLUDE-source loopback when the sender is outside the receiver rectangle, plus host fan-out that may differ from rectangle area; pair restored together |
| normalization | pre-allgather LayerNorm sender/receiver | acknowledged signal-only channel |
| normalization | plain sharded LayerNorm sender/receiver | acknowledged signal-only plus one-gate/multi-block mixed-mode streaming |
| TopK | `reader_final_topk.cpp`, `writer_local_topk.cpp` | race-free no-handshake receiver initialization |

## Fleet totals

The copied July census still contains 92 entries:

- 10 current production migrations at API v9;
- 12 production candidates deferred with `v9-port-blocked` and a concrete
  design-gap flag;
- 70 pre-existing deferred candidates from the July census;
- 0 pending and 0 quarantined.

The dated v8 reconcile reports and per-kernel logs are retained as historical
evidence. `reconcile_2026-07-29.md`, this ledger, `ledger.json`, and the porting
plan are authoritative for the current branch.
