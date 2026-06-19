# mcast_pipe — migration ledger (human mirror of `ledger.json`)

`ledger.json` is the **source of truth**; this file is a generated view. Do not hand-edit —
`apply-dm-helper` / `reconcile-dm-helper` rewrite both on each run.

- **Helper:** `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` · version define `MCAST_PIPE_API_VERSION`
- **Current API version at bootstrap:** v4 (Round 4: `SenderPipe`/`ReceiverPipe`)
- **Totals:** 13 migrated · 48 pending · 8 deferred (= 69 census sites)
- **needs_recheck flags:** 13 (set by reconcile 2026-06-19 — apply-dm-helper verify-only re-runs them)

**Reconcile history:**
- 2026-06-19 (base=None): 3 added (2 pending/refactor, 1 deferred/defer); 13 migrated flagged needs_recheck; 0 removed/renamed/clobbered

**Staleness is derived, not stored:** `stale = (status==migrated && migrated_api_version < CURRENT)`.

| status | kernel | family | tag | migrated@ | flags |
|---|---|---|---|---|---|
| migrated | `activation_reader_width_sharded.cpp` | conv | refactor | 4 | needs_recheck |
| migrated | `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | conv | clean | 4 | needs_recheck |
| migrated | `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | conv | clean | 4 | needs_recheck |
| migrated | `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | conv | clean | 4 | needs_recheck |
| migrated | `reader_final_topk.cpp` | data_movement + reduction | clean | 4 | needs_recheck |
| migrated | `reader_bmm_tile_layout_in0_receiver.cpp` | matmul | clean | 4 | needs_recheck |
| migrated | `reader_bmm_tile_layout_in0_sender_padding.cpp` | matmul | clean | 4 | needs_recheck |
| migrated | `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | matmul | refactor | 4 | needs_recheck |
| migrated | `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | matmul | clean | 4 | needs_recheck |
| migrated | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | matmul | clean | 4 | needs_recheck |
| migrated | `reader_mcast_receiver_unary_sharded_gn_v2.cpp` | normalization | clean | 4 | needs_recheck |
| migrated | `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp` | normalization | refactor | 4 | needs_recheck |
| migrated | `reader_mcast_sender_unary_sharded_ln.cpp` | normalization | refactor | 4 | needs_recheck |
| pending | `persistent_h2d_receiver.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `sampling_kernel.cpp` | ccl / deepseek / examples | clean | — | — |
| pending | `flash_mla.hpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `kv_cache_update.hpp` | ccl / deepseek / examples | clean | — | — |
| pending | `llama_all_gather_concat_writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `all_to_all_sender_writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `worker_receiver.cpp` | ccl / deepseek / examples | clean | — | — |
| pending | `reader.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `tilize_reader.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `tilize_writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `tilize_reader.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `tilize_writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `rms_sender_reader.cpp` | ccl / deepseek / examples | clean | — | — |
| pending | `rms_writer.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `dm1.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `reader_combine.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `reader_dispatch.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `unified_routed_expert_ffn_reader.cpp` | ccl / deepseek / examples | refactor | — | — |
| pending | `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | conv | refactor | — | — |
| pending | `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | conv | clean | — | — |
| pending | `writer.cpp` | conv | refactor | — | — |
| pending | `move_interleaved_with_overlap.cpp` | data_movement + reduction | refactor | — | — |
| pending | `move_stick_layout_interleaved_with_overlap.cpp` | data_movement + reduction | refactor | — | — |
| pending | `coordinator_single_row_multi_core.cpp` | data_movement + reduction | refactor | — | — |
| pending | `reader_single_row_multi_core.cpp` | data_movement + reduction | refactor | — | — |
| pending | `writer_single_row_multi_core.cpp` | data_movement + reduction | refactor | — | — |
| pending | `reader_argmax_interleaved_multicore.cpp` | data_movement + reduction | refactor | — | — |
| pending | `writer_local_topk.cpp` | data_movement + reduction | refactor | — | — |
| pending | `reader_bmm_tile_layout_in0_receiver_in1_sender.cpp` | matmul | clean | — | — |
| pending | `reader_bmm_tile_layout_in0_sender_in1_receiver.cpp` | matmul | clean | — | — |
| pending | `reader_bmm_tile_layout_in0_sender_in1_sender.cpp` | matmul | clean | — | — |
| pending | `reader_mcast_transformer_group_attn_matmul.cpp` | matmul | refactor | — | — |
| pending | `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | matmul | refactor | — | — |
| pending | `reader_mcast_receiver_unary_gn.cpp` | normalization | refactor | — | — |
| pending | `reader_mcast_sender_unary_gn.cpp` | normalization | refactor | — | — |
| pending | `reader_mcast_sender_unary_sharded_gn_v2.cpp` | normalization | refactor | — | — |
| pending | `welford_reader_mcast_receiver_unary_gn.cpp` | normalization | refactor | — | — |
| pending | `welford_reader_mcast_sender_unary_gn.cpp` | normalization | refactor | — | — |
| pending | `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` | normalization | refactor | — | — |
| pending | `reader_mcast_receiver_unary_sharded_ln.cpp` | normalization | refactor | — | — |
| pending | `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | normalization | clean | — | — |
| pending | `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | normalization | refactor | — | — |
| pending | `reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` | normalization | clean | — | — |
| pending | `reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | normalization | refactor | — | — |
| pending | `exp_ring_joint_reader.cpp` | transformer + sdpa | refactor | — | — |
| pending | `reader_interleaved.cpp` | transformer + sdpa | refactor | — | — |
| pending | `dataflow_common.hpp` | transformer + sdpa | refactor | — | — |
| deferred | `dataflow_utils.hpp` | ccl / deepseek / examples | ref | — | — |
| deferred | `mcast.hpp` | ccl / deepseek / examples | ref | — | — |
| deferred | `coordinator_kernel.cpp` | ccl / deepseek / examples | ref | — | — |
| deferred | `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | ccl / deepseek / examples | defer | — | — |
| deferred | `reader_bmm_tile_layout_in0_ring_all_gather.cpp` | matmul | defer | — | — |
| deferred | `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | matmul | defer | — | — |
| deferred | `worker_writer.cpp` | transformer + sdpa | oos | — | — |
| deferred | `chain_link.hpp` | transformer + sdpa | ref | — | — |
