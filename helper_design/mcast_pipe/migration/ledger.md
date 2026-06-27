# mcast_pipe migration ledger (human mirror) — @ v8, 2026-06-20 (re-entry: v7→v8 staleness sweep + conv-WS dual-pipe)

Source of truth: ledger.json. CURRENT = MCAST_PIPE_API_VERSION 8.

## Migrated @ v8 (19)
| status | ver | tag | role | kernel |
|---|---|---|---|---|
| migrated | 8 | clean | sender | reader_bmm_tile_layout_in0_sender_padding.cpp |
| migrated | 8 | clean | receiver | reader_bmm_tile_layout_in0_receiver.cpp |
| migrated | 8 | clean | hybrid | reader_bmm_tile_layout_in1_sender_writer_padding.cpp |
| migrated | 8 | clean | receiver | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp |
| migrated | 8 | refactor | hybrid | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp |
| migrated | 8 | clean | receiver | reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 8 | clean | sender | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 8 | clean | receiver | writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 8 | refactor | hybrid | activation_reader_width_sharded.cpp |
| migrated | 8 | clean | sender | reader_mcast_sender_unary_sharded_ln_post_allgather.cpp |
| migrated | 8 | clean | receiver | reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp |
| migrated | 8 | clean | receiver | reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 8 | refactor | sender | reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp |
| migrated | 8 | refactor | receiver | reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp |
| migrated | 8 | refactor | receiver | welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 8 | refactor | sender | reader_mcast_sender_unary_sharded_ln.cpp |
| migrated | 8 | refactor | receiver | reader_mcast_receiver_unary_sharded_ln.cpp |
| migrated | 8 | clean | receiver | reader_final_topk.cpp |
| migrated | 8 | refactor | sender | writer_local_topk.cpp |

## Deferred (50) — unchanged this run
| tag | reason-class | kernel |
|---|---|---|
| refactor | - | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp |
| refactor | - | reader_mcast_transformer_group_attn_matmul.cpp |
| defer | - | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| defer | - | reader_bmm_tile_layout_in0_ring_all_gather.cpp |
| clean | - | reader_bmm_tile_layout_in0_sender_in1_sender.cpp |
| clean | - | reader_bmm_tile_layout_in0_receiver_in1_sender.cpp |
| clean | - | reader_bmm_tile_layout_in0_sender_in1_receiver.cpp |
| clean | - | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| refactor | - | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp |
| refactor | - | writer.cpp |
| refactor | - | reader_mcast_sender_unary_sharded_gn_v2.cpp |
| refactor | - | welford_reader_mcast_sender_unary_gn.cpp |
| refactor | - | welford_reader_mcast_sender_unary_sharded_gn_v2.cpp |
| refactor | - | welford_reader_mcast_receiver_unary_gn.cpp |
| refactor | - | reader_mcast_sender_unary_gn.cpp |
| refactor | - | reader_mcast_receiver_unary_gn.cpp |
| ref | - | chain_link.hpp |
| refactor | - | reader_interleaved.cpp |
| refactor | - | exp_ring_joint_reader.cpp |
| refactor | - | dataflow_common.hpp |
| oos | - | worker_writer.cpp |
| refactor | coverage-gap | reader_argmax_interleaved_multicore.cpp |
| refactor | - | coordinator_single_row_multi_core.cpp |
| refactor | - | reader_single_row_multi_core.cpp |
| refactor | - | writer_single_row_multi_core.cpp |
| refactor | - | move_interleaved_with_overlap.cpp |
| refactor | - | move_stick_layout_interleaved_with_overlap.cpp |
| ref | - | coordinator_kernel.cpp |
| ref | - | mcast.hpp |
| ref | - | dataflow_utils.hpp |
| refactor | - | flash_mla.hpp |
| clean | - | kv_cache_update.hpp |
| clean | - | sampling_kernel.cpp |
| refactor | - | reader_dispatch.cpp |
| refactor | - | reader_combine.cpp |
| refactor | coverage-gap | dm1.cpp |
| clean | - | rms_sender_reader.cpp |
| refactor | - | rms_writer.cpp |
| clean | - | worker_receiver.cpp |
| refactor | - | tilize_reader.cpp |
| refactor | - | tilize_writer.cpp |
| refactor | coverage-gap | tilize_reader.cpp |
| refactor | - | tilize_writer.cpp |
| refactor | - | reader.cpp |
| refactor | - | writer.cpp |
| refactor | - | llama_all_gather_concat_writer.cpp |
| refactor | - | all_to_all_sender_writer.cpp |
| refactor | - | unified_routed_expert_ffn_reader.cpp |
| refactor | - | persistent_h2d_receiver.cpp |
| defer | - | reader_bmm_tile_layout_in1_ring_all_gather.cpp |

## Totals
migrated@v8: 19 | pending: 0 | quarantined: 0 | deferred: 50
