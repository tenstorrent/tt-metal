# mcast_pipe migration ledger (human mirror) — @ v7, 2026-06-20

Source of truth: ledger.json. CURRENT = MCAST_PIPE_API_VERSION 7.

| status | ver | tag | flags | kernel |
|---|---|---|---|---|
| migrated | 7 | clean | - | reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | - | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | clean | - | writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp |
| migrated | 7 | refactor | - | activation_reader_width_sharded.cpp |
| migrated | 7 | clean | - | reader_final_topk.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in0_sender_padding.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in0_receiver.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in1_sender_writer_padding.cpp |
| migrated | 7 | clean | - | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp |
| migrated | 7 | refactor | - | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp |
| migrated | 7 | clean | - | reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | - | welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp |
| migrated | 7 | refactor | - | reader_mcast_sender_unary_sharded_ln.cpp |
| pending | - | refactor | tier:2d | flash_mla.hpp |
| pending | - | clean | tier:2d | kv_cache_update.hpp |
| pending | - | clean | tier:1 | sampling_kernel.cpp |
| pending | - | refactor | tier:2d | dm1.cpp |
| pending | - | refactor | tier:2d | tilize_reader.cpp |
| pending | - | refactor | tier:2d | tilize_writer.cpp |
| pending | - | clean | tier:1 | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp |
| pending | - | refactor | tier:2b | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp |
| pending | - | refactor | tier:2b | writer.cpp |
| pending | - | refactor | tier:2b | writer_local_topk.cpp |
| pending | - | refactor | tier:2b | reader_argmax_interleaved_multicore.cpp |
| pending | - | refactor | tier:3 | coordinator_single_row_multi_core.cpp |
| pending | - | refactor | tier:3 | reader_single_row_multi_core.cpp |
| pending | - | refactor | tier:3 | writer_single_row_multi_core.cpp |
| pending | - | refactor | tier:3 | move_interleaved_with_overlap.cpp |
| pending | - | refactor | tier:3 | move_stick_layout_interleaved_with_overlap.cpp |
| pending | - | refactor | tier:2b | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp |
| pending | - | refactor | tier:2b | reader_mcast_transformer_group_attn_matmul.cpp |
| pending | - | clean | tier:1 | reader_mcast_sender_unary_sharded_ln_post_allgather.cpp |
| pending | - | clean | tier:1 | reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp |
| pending | - | refactor | tier:2a | reader_mcast_sender_unary_sharded_gn_v2.cpp |
| pending | - | refactor | tier:2a | reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp |
| pending | - | refactor | tier:2a | reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp |
| pending | - | refactor | tier:2a | welford_reader_mcast_sender_unary_sharded_gn_v2.cpp |
| pending | - | refactor | tier:2a | reader_mcast_receiver_unary_sharded_ln.cpp |
| pending | - | refactor | tier:2c | reader_interleaved.cpp |
| pending | - | refactor | tier:2c | dataflow_common.hpp |
| deferred | - | ref | - | coordinator_kernel.cpp |
| deferred | - | ref | - | mcast.hpp |
| deferred | - | ref | - | dataflow_utils.hpp |
| deferred | - | refactor | coverage-gap | reader_dispatch.cpp |
| deferred | - | refactor | coverage-gap | reader_combine.cpp |
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
| deferred | - | defer | - | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | defer | - | reader_bmm_tile_layout_in1_ring_all_gather.cpp |
| deferred | - | defer | - | reader_bmm_tile_layout_in0_ring_all_gather.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_sender_in1_sender.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_receiver_in1_sender.cpp |
| deferred | - | clean | coverage-gap | reader_bmm_tile_layout_in0_sender_in1_receiver.cpp |
| deferred | - | refactor | coverage-gap | welford_reader_mcast_sender_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | welford_reader_mcast_receiver_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | reader_mcast_sender_unary_gn.cpp |
| deferred | - | refactor | coverage-gap | reader_mcast_receiver_unary_gn.cpp |
| deferred | - | ref | - | chain_link.hpp |
| deferred | - | refactor | coverage-gap | exp_ring_joint_reader.cpp |
| deferred | - | oos | - | worker_writer.cpp |

## Totals
migrated@v7: 13 | pending: 27 | quarantined: 0 | deferred: 29 (8 design + 21 coverage-gap)
