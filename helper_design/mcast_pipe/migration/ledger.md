# mcast_pipe migration ledger (human mirror) — @ v8, 2026-06-27 (reconcile: +23 candidates from llk_helper_library rebase)

Source of truth: ledger.json. CURRENT = MCAST_PIPE_API_VERSION 8.

## Migrated @ v8 (19)
| status | ver | tag | role | kernel | flags |
|---|---|---|---|---|---|
| migrated | 8 | clean | sender | reader_bmm_tile_layout_in0_sender_padding.cpp | - |
| migrated | 8 | clean | receiver | reader_bmm_tile_layout_in0_receiver.cpp | - |
| migrated | 8 | clean | hybrid | reader_bmm_tile_layout_in1_sender_writer_padding.cpp | - |
| migrated | 8 | clean | receiver | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp | - |
| migrated | 8 | refactor | hybrid | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | - |
| migrated | 8 | clean | receiver | reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | - |
| migrated | 8 | clean | sender | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | - |
| migrated | 8 | clean | receiver | writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | - |
| migrated | 8 | refactor | hybrid | activation_reader_width_sharded.cpp | - |
| migrated | 8 | clean | sender | reader_mcast_sender_unary_sharded_ln_post_allgather.cpp | tier:1 |
| migrated | 8 | clean | receiver | reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp | tier:1 |
| migrated | 8 | clean | receiver | reader_mcast_receiver_unary_sharded_gn_v2.cpp | - |
| migrated | 8 | refactor | sender | reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp | tier:2a |
| migrated | 8 | refactor | receiver | reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp | tier:2a |
| migrated | 8 | refactor | receiver | welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp | - |
| migrated | 8 | refactor | sender | reader_mcast_sender_unary_sharded_ln.cpp | - |
| migrated | 8 | refactor | receiver | reader_mcast_receiver_unary_sharded_ln.cpp | tier:2a |
| migrated | 8 | clean | receiver | reader_final_topk.cpp | - |
| migrated | 8 | refactor | sender | writer_local_topk.cpp | tier:2b |

## Pending (0) — new this reconcile (apply-dm-helper migrates next run)
| tag | role | kernel | flags |
|---|---|---|---|

## Deferred (73)
| tag | role | kernel | op_family | flags |
|---|---|---|---|---|
| refactor | hybrid | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp | matmul | tier:2b,design-gap |
| refactor | hybrid | reader_mcast_transformer_group_attn_matmul.cpp | matmul | tier:2b,design-gap |
| defer | sender | reader_bmm_tile_layout_in1_ring_all_gather.cpp | matmul | - |
| defer | n/a | reader_bmm_tile_layout_in0_ring_all_gather.cpp | matmul | - |
| clean | hybrid | reader_bmm_tile_layout_in0_sender_in1_sender.cpp | matmul | coverage-gap |
| clean | hybrid | reader_bmm_tile_layout_in0_receiver_in1_sender.cpp | matmul | coverage-gap |
| clean | hybrid | reader_bmm_tile_layout_in0_sender_in1_receiver.cpp | matmul | coverage-gap |
| clean | sender | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | conv | tier:1,design-gap |
| refactor | hybrid | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp | conv | tier:2b,design-gap |
| refactor | hybrid | writer.cpp | conv | tier:2b,design-gap |
| refactor | sender | reader_mcast_sender_unary_sharded_gn_v2.cpp | normalization | tier:2a,design-gap |
| refactor | sender | welford_reader_mcast_sender_unary_gn.cpp | normalization | coverage-gap |
| refactor | sender | welford_reader_mcast_sender_unary_sharded_gn_v2.cpp | normalization | tier:2a,design-gap |
| refactor | receiver | welford_reader_mcast_receiver_unary_gn.cpp | normalization | coverage-gap |
| refactor | sender | reader_mcast_sender_unary_gn.cpp | normalization | coverage-gap |
| refactor | receiver | reader_mcast_receiver_unary_gn.cpp | normalization | coverage-gap |
| ref | both | chain_link.hpp | transformer + sdpa | - |
| refactor | hybrid | reader_interleaved.cpp | transformer + sdpa | tier:2c,design-gap |
| refactor | hybrid | exp_ring_joint_reader.cpp | transformer + sdpa | coverage-gap |
| refactor | both | dataflow_common.hpp | transformer + sdpa | tier:2c,design-gap |
| oos | sender | worker_writer.cpp | transformer + sdpa | - |
| refactor | hybrid | reader_argmax_interleaved_multicore.cpp | data_movement + reduction | tier:2b,design-gap |
| refactor | sender | coordinator_single_row_multi_core.cpp | data_movement + reduction | design-gap,tier:3 |
| refactor | receiver | reader_single_row_multi_core.cpp | data_movement + reduction | design-gap,tier:3 |
| refactor | sender | writer_single_row_multi_core.cpp | data_movement + reduction | design-gap,tier:3 |
| refactor | sender | move_interleaved_with_overlap.cpp | data_movement + reduction | design-gap,tier:3 |
| refactor | sender | move_stick_layout_interleaved_with_overlap.cpp | data_movement + reduction | design-gap,tier:3 |
| ref | sender | coordinator_kernel.cpp | ccl / deepseek / examples | - |
| ref | both | mcast.hpp | ccl / deepseek / examples | - |
| ref | n/a | dataflow_utils.hpp | ccl / deepseek / examples | - |
| refactor | sender | flash_mla.hpp | ccl / deepseek / examples | tier:2d,design-gap |
| clean | sender | kv_cache_update.hpp | ccl / deepseek / examples | tier:2d,design-gap |
| clean | sender | sampling_kernel.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | hybrid | reader_dispatch.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | hybrid | reader_combine.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | dm1.cpp | ccl / deepseek / examples | tier:2d,coverage-gap |
| clean | sender | rms_sender_reader.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | rms_writer.cpp | ccl / deepseek / examples | coverage-gap |
| clean | receiver | worker_receiver.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | tilize_reader.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | tilize_writer.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | tilize_reader.cpp | ccl / deepseek / examples | tier:2d,design-gap |
| refactor | sender | tilize_writer.cpp | ccl / deepseek / examples | tier:2d,design-gap |
| refactor | sender | reader.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | writer.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | llama_all_gather_concat_writer.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | all_to_all_sender_writer.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | hybrid | unified_routed_expert_ffn_reader.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | persistent_h2d_receiver.cpp | ccl / deepseek / examples | coverage-gap |
| defer | sender | reader_bmm_tile_layout_in1_ring_all_gather.cpp | ccl / deepseek / examples | - |
| refactor | sender | persistent_d2h_sender.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | persistent_d2d_receiver.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | sender | persistent_d2d_sender.cpp | ccl / deepseek / examples | coverage-gap |
| refactor | hybrid | reader_indexer_score.cpp | transformer + sdpa | design-gap |
| ref | sender | mcast_sender.cpp | ccl / deepseek / examples | - |
| refactor | hybrid | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | sender | reader_bmm_tile_layout_in0_sender_padding.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | sender | reader_bmm_tile_layout_in0_sender_padding_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port,hang:#47797 |
| defer | sender | reader_bmm_tile_layout_in1_ring_all_gather.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | hybrid | reader_bmm_tile_layout_in1_sender_writer_padding.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | hybrid | reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | activation_reader_width_sharded.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | activation_reader_width_sharded_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| defer | hybrid | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port,hang:#47797 |
| clean | sender | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | sender | reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port,hang:#47797 |
| clean | sender | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| clean | sender | writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port,hang:#47797 |
| refactor | hybrid | move_interleaved_with_overlap.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |
| refactor | hybrid | move_stick_layout_interleaved_with_overlap.cpp | quasar (experimental metal 2.0 port) | quasar-metal2-port |

## Totals
migrated@v8: 19 (0 needs_recheck: ) | pending: 0 | quarantined: 0 | deferred: 73 | TOTAL: 92
