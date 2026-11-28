# DFB imbalance audit — family "ccl_misc"
Repo /localdev/bbradel/tt-metal @ bbradel-59396_qsr_3mb_llama. Read-only.

## Drain semantics (governs what counts as a real hazard)
tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274 finish_impl:
- DM: flush implicit-sync tail credits, then spin until read_acked == read_posted for every TC.
- TRISC unpack/pack: spin until tile_counters[tc].posted == 0, i.e. the TC is empty.
- reserve_back_impl (:143) only waits for free space and posts nothing. So a reserve_back with no push_back does NOT block the
  drain. It only hangs if the reserve itself cannot get space (ring full, consumer already done). That is the pool reserve-ahead case.
- Upstream #56556 (5d0c6547420, not in this branch) moves the drain into ~DataflowBuffer(). The drain is gated on has_traffic_,
  which only push_back/pop_front/implicit read/write set. An object that is only reserved or used as scratch therefore never drains.
- So the dangerous imbalances are: (a) producer pushes more than the consumer pops (the DM producer waits for an ack that never
  comes, or TRISC unpack posted != 0), and (b) a consumer that waits but pops less, which leaves tiles on the TC.

## 1. CCL ops — NOT REACHABLE on Quasar for these tests
- llama tests/ops (and prototype_ops) test_all_gather, test_all_gather_async, test_all_gather_matmul_async and
  test_reduce_scatter_minimal_async are all parametrized `ttnn_mesh_device=(1,2)`. Each has an explicit
  `pytest.skip("multi-device op")` when the mesh is single-device. ops/conftest.py also keeps non-(1,1) meshes out of the
  `emulator` mark.
- Model code guards every CCL call on num_devices > 1:
  - model.py:211 `_all_gather_rmsnorm_tensor` returns early when get_num_devices()==1.
  - model.py:1061 logits all_gather requires num_devices>1.
  - mlp_1d.py:405/434 reduce_scatter only runs when num_devices>1.
  - rmsnorm_1d.py:411 prefill_distributed = num_devices>1 && dim>=4096.
  - tt_sampling all_gather is multi-device only.
  - The tests/modules rmsnorm/mlp/attention CCL cases are all (1,2)/(1,8) meshes.
- qwen3_vl_ops/test_mesh_partition: ttnn::mesh_partition (ccl/mesh_partition/mesh_partition.cpp:17) returns the input unchanged
  when the cluster-axis size is 1. No kernel launches on a single device. On multi-device it reuses the slice Metal2 factory, which
  the layout_dm agent owns.
- No CCL factory or kernel has a Quasar / Metal2 path, apart from mesh_partition reusing slice.
- Spot-check findings. All are latent and unreachable; record only.
  - all_gather_async/kernels/llama_shapes_sharded_writer.cpp:82-90: cb_packet_header reserve+push x3, no consumer ever pops.
    That is a producer-only self-loop. With the Quasar drain the DM producer would spin forever (posted 3, acked 0).
    The same packet-header-CB pattern appears in about 15 other CCL writers: rms_allgather, all_reduce_async, all_to_all_*,
    reduce_to_root, ring_attention_* and others. Fix: pop the header entries, or use PacketHeaderPool / scratch instead of FIFO ops.
  - ccl/all_gather/kernels/multicast_reader.cpp:226/269: reserve_back(2) ahead with push_back(1) steps (4 push sites vs 2 reserve
    sites). This is the same reserve-ahead shape as the pool bug, so the final reserve(2) may be dangling. Benign for the drain
    unless the ring is full.
  - The remaining all_gather_async, all_gather and reduce_scatter_minimal_async reader/writer/reduction kernels have matching
    reserve/push and wait/pop call-site counts.

## 2. Coverage sweep — ops in the 4 test dirs plus the imported model code, not owned by other families
Scope searched: resnet50/quasar, llama32_1b_quasar (whole fork: tests + modules + models + sampling),
qwen3_vl_ops and yolo_ops, plus models.tt_transformers helpers they import. gpt_oss_ops is NOT imported; its fast_reduce_nc,
clamp, repeat and sparse_matmul are out of scope.

Orphans found and analyzed here:
| op | factory / kernels | exercised? | result |
|---|---|---|---|
| ttnn.plus_one | experimental/plusone/device/plusone_program_factory.cpp; kernels/reader_plusone_interleaved.cpp | yes: llama ops/test_plus_one, qwen3 test_plus_one, model.py:1078 decode | BALANCED. Self-loop DFB `in0` used as raw scratch via get_write_ptr; 0 reserve/push/wait/pop, so no traffic and no drain. |
| ttnn.manual_seed | reduction/manual_seed/device/manual_seed_program_factory.cpp (4 factories) | yes: llama ops+prototype_ops test_sampling, test_sampling_1d:1226, sampling_1d.py:398 / tt_sampling.py:712 → Case 4 SetSeedsSetCores | Cross-kernel BALANCED (kernel_communication: reader push 1 / compute wait 1 + pop 1, same work unit on every core). See note A. |
| fill_pad (indirect, via ttnn::fill_implicit_tile_padding) | data_movement/fill_pad/device/fill_pad_program_factory.cpp (DRAM + L1-sharded factories); reader / writer / sharded_reader / sharded_writer / compute | likely, indirect: runs on any non-tile-aligned TILE tensor through ttnn reshape/pad/slice (mainline + quasar), topk, gather, sum/mean, and layernorm/rms_norm_pre_all_gather | BALANCED for every config the factory creates. See note B for a latent hazard. |

Note A, manual_seed self-loop scratch: reader_manual_seed_read_all_data.cpp does `user_ids_dfb.reserve_back(1)` (line ~29) and
`seeds_dfb.reserve_back(1)` (line ~35) with no push_back. reader_manual_seed_read_user_id.cpp does the same with user_ids_dfb.
These DFBs are bound PRODUCER+CONSUMER in one kernel and used as a landing area for one plain (non-TXN_ID) async_read.
Reserves on a fresh 1-entry ring always succeed and post nothing, so this is hygiene only (category 4), not a hang.
Optional cleanup: drop the reserve_back calls (get_write_ptr is enough), as plusone does. SUSPECT-low / not a hang.

Note B, fill_pad latent: in the DRAM factory, fill_pad_writer.cpp pushes the right/bot mask tiles UNCONDITIONALLY (push_*_mask_tile).
fill_pad_compute.cpp returns early when num_right+num_bottom+num_corner == 0 and never consumes the masks.
A zero-work core would leave posted=1 / acked=0 and the writer's drain would hang. This cannot happen today:
split_work_to_cores only puts cores with ≥1 work unit in all_cores, and fill_implicit_tile_padding skips the op when there is
no padding. The L1-sharded writer already guards with `if (num_work == 0) return;` before pushing masks.
The num_work==0 condition (nw=0) is equivalent to the compute total being 0 in every mode (A, B+right, B bottom-only).
Reader/compute/writer counts match per phase: right H-1 + bottom local_valid_w-1 + corner 1 = (H-1) + (local_right_col+1).
Suggested hardening: put the writer's mask push behind `if (num_right+num_bottom+num_corner)`, mirroring compute.

Ops owned by other families (handed off, not analyzed here):
- conv_pool: conv2d, quasar.conv2d, max_pool2d, quasar.max/avg_pool2d, quasar.fold, upsample (→ halo), quasar.padded_slice, quasar.slice_write.
- matmul: matmul, linear, quasar.matmul/linear, experimental.minimal_matmul.
- layout_dm:
  - tilize / untilize: from_torch(TILE)→device tilize, tilize, quasar.tilize*, tilize_with_val_padding, quasar.tilize_with_zero_padding,
    untilize, quasar.untilize*, untilize_with_unpadding.
  - layout and memory: to_layout, to_memory_config, interleaved_to_sharded, sharded_to_interleaved, reshard.
  - shape: reshape, unsqueeze / unsqueeze_to_4D (views), reshape_on_device.
  - pad / slice / concat / split: pad, slice / Tensor.__getitem__, concat, split.
  - others: transpose, permute, typecast, to_dtype (host), clone, copy, expand, reallocate, move, zeros / zeros_like / full / fill.
- eltwise_reduce_norm:
  - eltwise: add/add_, multiply/mul, subtract, div, sigmoid, exp, log, gt, eq_, remainder, where.
  - reductions: sum, mean, max (incl. quasar reduction).
  - norms and softmax: softmax, rms_norm, rms_norm_pre/post_all_gather, layer_norm.
  - selection and sampling: argmax (incl. internal argmax_nc), topk (incl. internal topk_large_indices), sampling,
    scatter, scatter_add, gather (incl. gather_codegen).
- transformer:
  - attention: sdpa variants (mainline + quasar.transformer.*).
  - rotary: rotary_embedding_llama, rotary_embedding_llama_fused_qk.
  - heads: nlp_create_qkv_heads(_decode), nlp_concat_heads(_decode).
  - cache: fill_cache, paged_fill_cache, paged_update_cache, paged_fused_update_cache.
- embedding (layout_dm).
- No-kernel host/runtime calls: from_device, to_device, quasar.to_device (host wrapper), copy_host_to_device_tensor,
  create_global_semaphore, trace begin/end/execute/release, synchronize_device, record/event_synchronize,
  read_device_profiler, dump_device_memory_state, as_tensor/load_tensor (host + tilize), mesh mappers/composers, set_fabric_config.
- fast_reduce_nc: only reached from sum/mean over N/C dims (generic_reductions.cpp:517). In-scope sum/mean calls reduce
  H/W only: resnet test_reduce_sum_mean uses dim=2/3, and tt_log_probs uses dim=-1/2 on rank-4. So it is NOT reached.

## Balanced kernels (coverage)
- plusone/reader_plusone_interleaved.cpp: no FIFO ops (self-loop scratch).
- manual_seed: reader_manual_seed_read_all_data / read_user_id balanced on kernel_communication (push 1). The user_ids/seeds
  reserves are benign scratch.
- manual_seed: compute receive_all_data / single_seed_receive_user_id wait 1 / pop 1. set_seed has no DFB.
- fill_pad: reader, sharded_reader, writer, sharded_writer and compute all balanced (data_in, data_out, right_mask, bot_mask),
  with the latent caveat in Note B.
- CCL (unreachable): all_gather_async minimal_default reader/writer, broadcast_rm reader/writer, llama_shapes reader;
  ccl/all_gather unicast reader/writer; reduce_scatter_minimal_async ring/line/dim0 readers, writers and reductions:
  call-site counts match.
