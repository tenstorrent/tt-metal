# mcast_pipe migration ledger (human mirror) — @ v7, 2026-06-20 (post backlog run)

Source of truth: ledger.json. CURRENT = MCAST_PIPE_API_VERSION 7.

| status | ver | tag | gap | kernel |
|---|---|---|---|---|
| migrated | 7 | clean | - | reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | - | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | - | writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | refactor | - | activation_reader_width_sharded.cpp |
| migrated | 7 | clean | - | reader_final_topk.cpp |
| migrated | 7 | refactor | - | writer_local_topk.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in0_sender_padding.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in0_receiver.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in1_sender_writer_padding.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp |
| migrated | 7 | refactor | - | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp |
| migrated | 7 | clean | - | reader_mcast_sender_unary_sharded_ln_post_allgather.cpp |
| migrated | 7 | clean | - | reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp |
| migrated | 7 | clean | - | reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | - | reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp |
| migrated | 7 | refactor | - | reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp |
| migrated | 7 | refactor | - | welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | - | reader_mcast_sender_unary_sharded_ln.cpp |
| migrated | 7 | refactor | - | reader_mcast_receiver_unary_sharded_ln.cpp |
| deferred | - | clean | coverage-gap | sampling_kernel.cpp |
| deferred | - | refactor | coverage-gap | reader_dispatch.cpp |
| deferred | - | refactor | coverage-gap | reader_combine.cpp |
| deferred | - | refactor | coverage-gap | dm1.cpp |
| deferred | - | clean | coverage-gap | rms_sender_reader.cpp |
| deferred | - | refactor | coverage-gap | rms_writer.cpp |
| deferred | - | clean | coverage-gap | worker_receiver.cpp |
| deferred | - | refactor | coverage-gap | tilize_reader.cpp |
| deferred | - | refactor | coverage-gap | tilize_writer.cpp |
| deferred | - | refactor | coverage-gap | reader.cpp |
| deferred | - | refactor | coverage-gap | writer.cpp |
| deferred | - | refactor | coverage-gap | llama_all_gather_concat_writer.cpp |
| deferred | - | refactor | coverage-gap | all_to_all_sender_writer.cpp |
| deferred | - | refactor | coverage-gap | unified_routed_expert_ffn_reader.cpp |
| deferred | - | refactor | coverage-gap | persistent_h2d_receiver.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_sender_in1_sender.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_receiver_in1_sender.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_sender_in1_receiver.cpp |
| deferred | - | refactor | coverage-gap | welford_reader_mcast_sender_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | welford_reader_mcast_receiver_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | reader_mcast_sender_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | reader_mcast_receiver_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | exp_ring_joint_reader.cpp |
| deferred | - | refactor | design-gap | flash_mla.hpp |
| deferred | - | clean | design-gap | kv_cache_update.hpp |
| deferred | - | refactor | design-gap | tilize_reader.cpp |
| deferred | - | refactor | design-gap | tilize_writer.cpp |
| deferred | - | clean | design-gap | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| deferred | - | refactor | design-gap | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp |
| deferred | - | refactor | design-gap | writer.cpp |
| deferred | - | refactor | design-gap | reader_argmax_interleaved_multicore.cpp |
| deferred | - | refactor | design-gap | coordinator_single_row_multi_core.cpp |
| deferred | - | refactor | design-gap | reader_single_row_multi_core.cpp |
| deferred | - | refactor | design-gap | writer_single_row_multi_core.cpp |
| deferred | - | refactor | design-gap | move_interleaved_with_overlap.cpp |
| deferred | - | refactor | design-gap | move_stick_layout_interleaved_with_overlap.cpp |
| deferred | - | refactor | design-gap | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp |
| deferred | - | refactor | design-gap | reader_mcast_transformer_group_attn_matmul.cpp |
| deferred | - | refactor | design-gap | reader_mcast_sender_unary_sharded_gn_v2.cpp |
| deferred | - | refactor | design-gap | welford_reader_mcast_sender_unary_sharded_gn_v2.cpp |
| deferred | - | refactor | design-gap | reader_interleaved.cpp |
| deferred | - | refactor | design-gap | dataflow_common.hpp |
| deferred | - | ref | original-8 | coordinator_kernel.cpp |
| deferred | - | ref | original-8 | mcast.hpp |
| deferred | - | ref | original-8 | dataflow_utils.hpp |
| deferred | - | defer | original-8 | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | defer | original-8 | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | defer | original-8 | reader_bmm_tile_layout_in0_ring_all_gather.cpp |
| deferred | - | ref | original-8 | chain_link.hpp |
| deferred | - | oos | original-8 | worker_writer.cpp |

## Totals
migrated@v7: 19 | pending: 0 | quarantined: 0 | deferred: 50 (8 original + 19 design-gap + 23 coverage-gap)
