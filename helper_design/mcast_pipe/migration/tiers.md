# mcast_pipe rollout — tiers for THIS run (re-entry @ v8, 2026-06-20)

- **CURRENT** = MCAST_PIPE_API_VERSION **8** (Round 10: D2 count split — `SenderPipe` drops the
  3rd template param `NUM_ACTIVE_RECEIVER_CORES`; rect derives fan-out via `area()`; new ctor arg
  `consumer_ack_count` default `ACK_EQUALS_FANOUT`). `ReceiverPipe` UNCHANGED.
- **Mode:** `run-all` (revert each failure, sweep everything, one report).
- **Worklist = stale (Tier 0) + net-new (Tier 1).** 0 pending backlog remained at v7; the net-new
  item is the user-requested conv-WS dual-pipe upgrade (now expressible via the v8 split).

## Staleness classification (the v8 caller-facing move)
- **SenderPipe sites (9): code rewrite** — delete the 3rd template arg; the rect's `area()` carries
  fan-out. Dense sites = pure arg deletion. (conv-WS handled separately in Tier 1.)
- **ReceiverPipe-only sites (10): NO code change** — `ReceiverPipe` API is unchanged at v8, so they
  JIT-compile as-is. "Remigration" = re-verify (their test runs anyway alongside the sender) + bump
  `migrated_api_version` 7→8. block_sharded has both faces; its sender arm is rewritten.

## Tier 0 — remigrate stale v7→v8 (19 kernels, grouped by shared test suite)

### 0a — matmul (5)
| kernel | face | action | validation |
|---|---|---|---|
| reader_bmm_tile_layout_in0_sender_padding.cpp | sender | drop count arg | `test_matmul_1d_multiple_output_blocks_per_core[...mcast_in0=True...grid(8,2)...n=2048-k=1024-m=256]` |
| reader_bmm_tile_layout_in1_sender_writer_padding.cpp | sender | drop count arg | same 1D case |
| reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | sender+recv | drop count arg (sender arm) | `test_matmul_2d_multiple_output_blocks_per_core --run-all` |
| reader_bmm_tile_layout_in0_receiver.cpp | receiver | verify + bump | 1D case |
| reader_bmm_tile_layout_in1_receiver_writer_padding.cpp | receiver | verify + bump | 2D case |

### 0b — layernorm sharded (6)
| kernel | face | action | validation |
|---|---|---|---|
| reader_mcast_sender_unary_sharded_ln.cpp | sender | drop count arg | `test_layer_norm_sharded_single_stage` |
| reader_mcast_receiver_unary_sharded_ln.cpp | receiver | verify + bump | `test_layer_norm_sharded_single_stage` |
| reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp | sender | drop count arg | `test_pre_allgather_layernorm` |
| reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp | receiver | verify + bump | `test_pre_allgather_layernorm` |
| reader_mcast_sender_unary_sharded_ln_post_allgather.cpp | sender | drop count arg | `test_post_allgather_layernorm` |
| reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp | receiver | verify + bump | `test_post_allgather_layernorm` |

### 0c — groupnorm v2 (2, receiver-only, no code change)
| kernel | face | action | validation |
|---|---|---|---|
| reader_mcast_receiver_unary_sharded_gn_v2.cpp | receiver | verify + bump | `test_group_norm_with_block_sharded_v2_8x4_grid` |
| welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp | receiver | verify + bump | same (welford) |

### 0d — topk (2)
| kernel | face | action | validation |
|---|---|---|---|
| reader_final_topk.cpp | sender | drop count arg | `test_topk[...H=32-W=8192-dim=3-k=50-BFLOAT16_B]` |
| writer_local_topk.cpp | receiver | verify + bump | `test_topk` W=8192 |

### 0e — conv weights (3)
| kernel | face | action | validation |
|---|---|---|---|
| writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | sender | drop count arg | `test_conv_features` (2D weights mcast) |
| writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | receiver | verify + bump | `test_conv_features` |
| reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | receiver | verify + bump | `test_conv_features HEIGHT_SHARDED 256x256 act_block_h=32` |

## Tier 1 — conv-WS dual-pipe upgrade (NET-NEW, user request)
| kernel | current | target | validation |
|---|---|---|---|
| activation_reader_width_sharded.cpp | SenderPipe-only (PRE_HANDSHAKE=false) + **raw receiver** + raw fan-in counter | **SenderPipe (v8, PRE_HANDSHAKE=true, explicit `consumer_ack_count = num_mcast_cores-1`) + ReceiverPipe** | `test_conv_features -k WIDTH_SHARDED` (full WS matrix, was 48 passed / 16 skip / 0 fail) |

**Why now feasible:** v8's split count decouples data-mcast fan-out (`num_reader_cores` = rect area)
from consumer ack (`num_mcast_cores-1`) — the exact D2 divergence that kept the receiver/handshake
raw at v7. The Round-9 per-send VALID re-assert removes the rotating-role staleness that made the
`ReceiverPipe` (clear-after-wait) hang on this self-mcast shape (same fix that lifted block_sharded).

## Coverage gaps / risks
- **conv-WS validation_set was empty + coverage_confidence=unknown** in the ledger. The Tier-1
  subagent must re-derive & device-verify `test_conv_features -k WIDTH_SHARDED` hits the kernel
  (JIT build cache) before gating the dual-pipe migration. This is the single highest-risk item
  (it was quarantined once already at v7).
- All Tier-0 sender rewrites are pure arg deletions on already-green call sites → low risk.

## Out of scope (unchanged): 50 deferred kernels
Stay deferred per the prior run + user request — 8 original design-deferred, 19 design-gap
(D1/D3–D9), 23 coverage-gap (multi-device / no single-chip test). Not touched this run.
