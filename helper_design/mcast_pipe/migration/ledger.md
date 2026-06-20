# mcast_pipe migration ledger (human mirror) — @ v7, 2026-06-20

Source of truth: ledger.json. CURRENT = MCAST_PIPE_API_VERSION 7.

| status | ver | tag | kernel |
|---|---|---|---|
| migrated | 7 | clean | reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | reader_bmm_tile_layout_in0_receiver.cpp |
| migrated | 7 | clean | reader_bmm_tile_layout_in0_sender_padding.cpp |
| migrated | 7 | clean | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp |
| migrated | 7 | clean | reader_bmm_tile_layout_in1_sender_writer_padding.cpp |
| migrated | 7 | clean | reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | reader_mcast_sender_unary_sharded_ln.cpp |
| migrated | 7 | clean | reader_final_topk.cpp |
| quarantined | - | refactor | activation_reader_width_sharded.cpp |
| quarantined | - | refactor | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp |
| pending | - | refactor | persistent_h2d_receiver.cpp |
| pending | - | clean | sampling_kernel.cpp |
| pending | - | refactor | flash_mla.hpp |
| pending | - | clean | kv_cache_update.hpp |
| pending | - | clean | reader_bmm_tile_layout_in0_receiver_in1_sender.cpp |
| pending | - | clean | reader_bmm_tile_layout_in0_sender_in1_receiver.cpp |
| pending | - | clean | reader_bmm_tile_layout_in0_sender_in1_sender.cpp |
| pending | - | refactor | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp |
| pending | - | clean | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| pending | - | refactor | move_interleaved_with_overlap.cpp |
| pending | - | refactor | move_stick_layout_interleaved_with_overlap.cpp |
| pending | - | refactor | coordinator_single_row_multi_core.cpp |
| pending | - | refactor | reader_single_row_multi_core.cpp |
| pending | - | refactor | writer_single_row_multi_core.cpp |
| pending | - | refactor | llama_all_gather_concat_writer.cpp |
| pending | - | refactor | all_to_all_sender_writer.cpp |
| pending | - | clean | worker_receiver.cpp |
| pending | - | refactor | reader.cpp |
| pending | - | refactor | writer.cpp |
| pending | - | refactor | tilize_reader.cpp |
| pending | - | refactor | tilize_writer.cpp |
| pending | - | refactor | tilize_reader.cpp |
| pending | - | refactor | tilize_writer.cpp |
| pending | - | clean | rms_sender_reader.cpp |
| pending | - | refactor | rms_writer.cpp |
| pending | - | refactor | writer.cpp |
| pending | - | refactor | dm1.cpp |
| pending | - | refactor | reader_combine.cpp |
| pending | - | refactor | reader_dispatch.cpp |
| pending | - | refactor | unified_routed_expert_ffn_reader.cpp |
| pending | - | refactor | reader_mcast_transformer_group_attn_matmul.cpp |
| pending | - | refactor | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp |
| pending | - | refactor | reader_mcast_receiver_unary_gn.cpp |
| pending | - | refactor | reader_mcast_sender_unary_gn.cpp |
| pending | - | refactor | reader_mcast_sender_unary_sharded_gn_v2.cpp |
| pending | - | refactor | welford_reader_mcast_receiver_unary_gn.cpp |
| pending | - | refactor | welford_reader_mcast_sender_unary_gn.cpp |
| pending | - | refactor | welford_reader_mcast_sender_unary_sharded_gn_v2.cpp |
| pending | - | refactor | reader_mcast_receiver_unary_sharded_ln.cpp |
| pending | - | clean | reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp |
| pending | - | refactor | reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp |
| pending | - | clean | reader_mcast_sender_unary_sharded_ln_post_allgather.cpp |
| pending | - | refactor | reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp |
| pending | - | refactor | reader_argmax_interleaved_multicore.cpp |
| pending | - | refactor | writer_local_topk.cpp |
| pending | - | refactor | exp_ring_joint_reader.cpp |
| pending | - | refactor | reader_interleaved.cpp |
| pending | - | refactor | dataflow_common.hpp |
| deferred | - | ref | dataflow_utils.hpp |
| deferred | - | ref | mcast.hpp |
| deferred | - | ref | coordinator_kernel.cpp |
| deferred | - | defer | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | oos | worker_writer.cpp |
| deferred | - | defer | reader_bmm_tile_layout_in0_ring_all_gather.cpp |
| deferred | - | defer | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | ref | chain_link.hpp |

## Totals
migrated@v7: 11 | quarantined: 2 | pending: 48 | deferred: 8
