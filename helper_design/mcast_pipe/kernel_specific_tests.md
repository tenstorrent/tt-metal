# mcast_pipe census — specific + comprehensive tests per kernel

For every kernel: **one specific test (with parametrization) that hits it**, then the **comprehensive
set** for its op. Confidence tags:

- ✅ **device-verified** — this exact nodeid ran green on BH p150a AND freshly JIT-built the kernel
  (proven by a fresh dir under `built/tt-metal-cache*/kernels/<basename>/`). Source: `migration/test_map.json` + round-3 changelog.
- 🔶 **static** — param chosen to satisfy the factory's dispatch condition, not yet device-confirmed.
  Confirm by running it then `grep -rl <basename> generated/` (or the cache-dir check).
- ⚠️ **gap** — no single-chip / no automated test reaches this kernel (multi-device, sweep-only, or example-binary).

`-k` note: pytest can't filter `name=value`; filter on a value substring only. Verify a build with
`grep -rl <kernel_basename> generated/` after the run.

---

## matmul — comprehensive set
`test_matmul.py`, `test_linear.py`, `test_custom_grids.py`, `test_experimental.py`, `test_matmul_deepseek.py`,
`test_sparse_matmul.py`, `test_ring_matmul.py`, `test_matmul_auto_tune.py` (all under `tests/ttnn/unit_tests/operations/matmul/`)
plus nightly `test_matmul.py`, `test_matmul2.py`, `test_matmul_1d_2d.py`, `test_matmul_activations.py`,
`test_matmul_tile_pack_row_major.py`, `test_matmul_dram_sharded.py`, `test_matmul_block_sharded_1d_grid.py`,
`test_bert_matmuls.py`, `test_matmul_1d_gather_in0.py`, `test_rs_matmul_1d_gather_in0.py`, `test_attn_matmul.py`.

| kernel | specific test (hits it) | conf |
|---|---|---|
| `reader_bmm_tile_layout_in0_sender_padding.cpp` | `test_matmul.py::test_matmul_1d_multiple_output_blocks_per_core[uneven_width=0-mcast_in0=True-num_out_block_w=1-num_out_block_h=1-out_sharded=False-in_sharded=False-grid_size=(8, 2)-has_bias=False-n=2048-k=1024-m=256]` | ✅ |
| `reader_bmm_tile_layout_in0_receiver.cpp` | same `matmul_1d_multiple_output_blocks_per_core` nodeid (grid 8×2 → 1 sender + 15 receivers) | ✅ |
| `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | same `matmul_1d` nodeid (in1 sender/writer on every worker core) | ✅ |
| `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core[silicon_arch_name=blackhole-mesh_device=(1, 1)-transpose_mcast=True-num_out_block_w=1-num_out_block_h=1-out_sharded=True-in0_sharded=False-grid_size=(8, 4)-has_bias=True-n=1024-k=512-m=512-b=1]` | ✅ |
| `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | `test_matmul.py::test_sharded_matmul[mcast_2d]` (BLOCK_SHARDED in0, 2D mcast) — round-3 ran the `sharded_matmul`+`in0_in1_bias_sharded` suites, 270 green, 29 JIT variants | ✅ |
| `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | `tests/ttnn/nightly/.../test_matmul_dram_sharded.py::test_matmul_in1_dram_sharded_with_program_cache` smallest M=32,K=8192,N=1280,grid(8,1) | 🔶 |
| `experimental/.../reader_mcast_transformer_group_attn_matmul.cpp` | `tests/ttnn/nightly/.../test_attn_matmul.py::test_group_attn_matmul` (the group-attn variant; `test_attn_matmul` is the non-group twin) | 🔶 |
| `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | `tests/ttnn/nightly/.../test_matmul_1d_gather_in0.py` (gather_in0=True; ring on in1) — multi-device | ⚠️ |
| `reader_bmm_tile_layout_in0_ring_all_gather.cpp` | `tests/ttnn/unit_tests/operations/matmul/test_ring_matmul.py` / `test_matmul_1d_gather_in0.py` — multi-device (t3000) | ⚠️ |
| 3 programming-example readers (`reader_bmm_tile_layout_in0_sender_in1_sender.cpp`, `..._in0_receiver_in1_sender.cpp`, `..._in0_sender_in1_receiver.cpp`) | **NO pytest** — only the `metal_example_matmul_multicore_reuse_mcast` binary | ⚠️ |

---

## conv2d — comprehensive set
`tests/ttnn/unit_tests/operations/conv/test_conv2d.py`, `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py`,
`tests/ttnn/perf_tests/operations/conv/test_conv2d_device_perf.py`. Shape-table aliases: `HS/BS/WS`.

| kernel | specific test (hits it) | conf |
|---|---|---|
| `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `test_conv2d.py::test_conv_features` HEIGHT_SHARDED row `(16,16,256,256, HS, {"act_block_h":32})` | ✅ |
| `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | same HS 256×256 row (multi-core HS → receivers exist) | ✅ |
| `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `test_conv_features` BLOCK_SHARDED row `(128,128,32,32, BS, None)` | ✅ |
| `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | same BS 32×32 row | ✅ |
| `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | same BS 32×32 row, filter 3×3 (single-shot path; CHUNKED >NOC_MAX_BURST needs a large-act-block BS shape) | ✅ (med — chunked sub-case unverified) |
| `activation_reader_width_sharded.cpp` | `test_conv_features` WIDTH_SHARDED row `(353,384,8,8, WS, None)` (no grid guard; `test_conv_ws` needs 8×8 grid) | ✅ |

`ttnn.experimental.conv3d` — comprehensive: `test_conv3d.py` (unit + nightly).

| kernel | specific test (hits it) | conf |
|---|---|---|
| `experimental/conv3d/.../writer.cpp` | `test_conv3d.py::test_conv3d_no_config[(1,32,4,8,8),32,(3,3,3),(2,2,2),1,(0,1,1),"zeros"]` (Disabled/Local path); use `test_conv3d_sweep_shapes` `groups_4` for the Mcast/Chain modes | 🔶 (mode is parallelism-derived, not a param — grep by full path, basename `writer.cpp` is generic) |

---

## normalization

### `ttnn.layer_norm` sharded (plain, NOT_DISTRIBUTED) — comprehensive: `test_layer_norm_sharded.py` + nightly `test_layernorm_sharded.py`
| kernel | specific test | conf |
|---|---|---|
| `reader_mcast_sender_unary_sharded_ln.cpp` | `test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage[dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=False-use_welford=True-h=256-w=512-num_cores_h=4-num_cores_w=4-block_ht=2-block_wt=4-subblock_wt=1]` | ✅ |
| `reader_mcast_receiver_unary_sharded_ln.cpp` | same `single_stage` nodeid | ✅ |

### distributed layer_norm — comprehensive: `test_distributed_layernorm_sharded.py` + nightly `test_distributed_layernorm_{pre,post}_allgather.py` (multi-device, simulated on one chip)
| kernel | specific test | conf |
|---|---|---|
| `reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` | `test_distributed_layernorm_sharded.py::test_post_allgather_layernorm[num_devices=4, bfloat16, core_grid=(8,2), is_rmsnorm=False]` | 🔶 (med) |
| `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | same post_allgather nodeid | 🔶 |
| `reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | `test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm[num_devices=4, bfloat16, core_grid=(8,4), is_rmsnorm=False]` | 🔶 |
| `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | same pre_allgather nodeid | 🔶 |

### `ttnn.group_norm` sharded (v2) — comprehensive: `test_group_norm.py` + nightly `test_group_norm.py`
| kernel | specific test | conf |
|---|---|---|
| `reader_mcast_sender_unary_sharded_gn_v2.cpp` | `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-legacy-N=1-C=1280-H=16-W=16-num_groups=32-device_params={'l1_small_size': 0}]` | ✅ |
| `reader_mcast_receiver_unary_sharded_gn_v2.cpp` | same `8x4_grid` legacy nodeid | ✅ |
| `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` | `test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-welford-N=1-C=1280-H=16-W=16-num_groups=32-device_params={'l1_small_size': 0}]` | ✅ |
| `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp` | same `8x4_grid` welford nodeid | ✅ |

### `ttnn.group_norm` INTERLEAVED (non-v2) — ⚠️ NO unit-test coverage (every `test_group_norm*.py` shards input first → routes to v2 factory). Reachable only via the sweep harness (`tests/ttnn/python_api_testing/sweep_tests/ttnn_ops.py`) or a custom probe.
| kernel | reach | conf |
|---|---|---|
| `reader_mcast_sender_unary_gn.cpp` | interleaved input, `use_welford=False`; needs custom probe | ⚠️ gap |
| `reader_mcast_receiver_unary_gn.cpp` | interleaved + `batch < num_virtual_rows` (mcast case) | ⚠️ gap |
| `welford_reader_mcast_sender_unary_gn.cpp` | interleaved input, `use_welford=True` | ⚠️ gap |
| `welford_reader_mcast_receiver_unary_gn.cpp` | interleaved + welford + mcast case | ⚠️ gap |

---

## transformer + sdpa

### `ttnn.transformer.scaled_dot_product_attention` (prefill) — comprehensive: `test_sdpa_prefill.py`, `test_bounded_sliding_kv_cache.py`, `test_paged_sdpa_decode_flexible_geometry.py` (unit) + nightly `test_sdpa_prefill.py`, `test_sdpa_chunked.py`, `test_sdpa_ring_distributed.py`, `test_sdpa_joint.py`
| kernel | specific test | conf |
|---|---|---|
| `sdpa/.../reader_interleaved.cpp` | `test_sdpa_prefill.py::test_sdpa_noncausal[b=1-nh=8-nkv=1-s=2048-d=128-k128-q128-bf16]` (also `test_sdpa_tt[...bfp8...q128...]` causal) | ✅ |

### `ttnn.transformer.scaled_dot_product_attention_decode` — comprehensive: `test_sdpa_decode.py` (unit) + nightly `test_sdpa_decode.py`, `test_sdpa_decode_sink.py`, `test_sdpa_decode_cache.py`
| kernel | specific test | conf |
|---|---|---|
| `sdpa_decode/.../dataflow_common.hpp` (via `reader_decode_all.cpp`/`writer_decode_all.cpp`) | non-mcast path: `test_sdpa_decode.py::test_sdpa_decode[kv_bfp8 ... b=4,nh=32,nkv=8,s=8192,d=128]`. UNLINKED K-mcast (`use_k_mcast=1`) path: `test_sdpa_decode.py::test_sdpa_decode_sharded` | 🔶 (med — which sharded param flips `q_heads_parallel_factor>1` unconfirmed; check compile arg) |

### ring/joint sdpa (headers + exp) — ⚠️ multi-device only
| kernel | specific test | conf |
|---|---|---|
| `sdpa/.../chain_link.hpp` (incl. only by `ring_joint_reader.cpp`) | ring-joint multi-device nightly (mesh-gated); NOTE `test_exp_ring_joint_sdpa.py` does NOT build chain_link.hpp (exp open-codes it) | ⚠️ gap (no single-chip case) |
| `sdpa/.../exp_ring_joint_reader.cpp` | `tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py::test_exp_ring_joint_attention_sdpa_accuracy` (needs ≥4 devices) | ⚠️ gap |

---

## reduction + data_movement

### `ttnn.topk` (multicore) — comprehensive: `test_topk.py` (unit) + nightly `test_reduction_ops.py` (has `test_topk`)
| kernel | specific test | conf |
|---|---|---|
| `reduction/topk/.../reader_final_topk.cpp` | `test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=8192-dim=3-k=50-BFLOAT16_B]` (W=8192 = multicore threshold) | ✅ |
| `reduction/topk/.../writer_local_topk.cpp` | same W=8192 topk nodeid (co-dispatched with reader_final_topk) | ✅ |

### `ttnn.argmax` (multicore) — comprehensive: `test_argmax.py` (unit) + nightly `test_reduction_ops.py` (has `test_argmax`)
| kernel | specific test | conf |
|---|---|---|
| `reduction/argmax/.../reader_argmax_interleaved_multicore.cpp` | `test_argmax.py::test_argmax[shape=[64,128], ROW_MAJOR, dim=-1, use_multicore=True, float32]` (128/16=8 worker cores → real flag-mcast). `use_multicore` is a positional value, not in the id — filter by shape `-k "64-128"` + confirm via JIT grep | 🔶 (med) |

### `ttnn.sort` (single_row_multi_core, DRAM branch) — ⚠️ DEFERRED (legacy API). Comprehensive: `tests/ttnn/unit_tests/operations/data_movement/test_sort.py` (`test_reduction_ops.py` does NOT test sort)
| kernel | specific test | conf |
|---|---|---|
| `sort/.../coordinator_single_row_multi_core.cpp` | `test_sort.py::test_sort_long_tensor[shape=[1, 524288], dim=-1, descending=False]` (huge Wt → multicore-DRAM branch) | 🔶 (Wt cutoff is grid-dependent — confirm via JIT grep) |
| `sort/.../reader_single_row_multi_core.cpp` | same `test_sort_long_tensor` large-Wt row | 🔶 |
| `sort/.../writer_single_row_multi_core.cpp` | same `test_sort_long_tensor` large-Wt row | 🔶 |

### `ttnn.move` (MULTI_CORE_OVERLAP) — ⚠️ tests live in `tests/tt_eager`, not `tests/ttnn`. Comprehensive: `tests/tt_eager/python_api_testing/unit_testing/misc/test_move.py`, `test_move_sharded.py`
| kernel | specific test | conf |
|---|---|---|
| `move/.../move_interleaved_with_overlap.cpp` | `test_move.py::test_move_op[overlap-TILE-...-in0_L1-out_L1]` shape `[1,1,32,32]` (test_id="overlap" forces src/dst L1 overlap; TILE → this kernel) | 🔶 (overlap depends on runtime allocator addrs — confirm via JIT grep) |
| `move/.../move_stick_layout_interleaved_with_overlap.cpp` | `test_move.py::test_move_op[overlap-RM-...-in0_L1-out_L1]` (RM layout → stick kernel) | 🔶 |

---

## deepseek (model demos — tests under `models/demos/deepseek_v3_b1/tests/unit_tests`, not `tests/ttnn`)

| kernel | specific test | conf |
|---|---|---|
| `micro_ops/sampling/kernels/sampling_kernel.cpp` | `test_sampling.py::test_sampling_argmax_single_device_101_cores[17-0]` (single-device loop-barrier path) | ✅ |
| `unified_kernels/kv_cache_update.hpp` (header → fused attn/decoder .cpp) | `test_kv_cache_branch.py::test_kv_cache_branch[1-True-1e-06]` (NOPE-sender mcast, single device) | 🔶 (grep the *including* fused-op kernel, not the .hpp) |
| `unified_kernels/flash_mla.hpp` | `test_flash_mla.py` (single device) | 🔶 |
| `unified_kernels/mcast.hpp` / `dataflow_utils.hpp` | REF headers — covered transitively wherever included (e.g. `test_mcast.py`, `test_attention_block.py`) | — |

### deepseek experimental ttnn ops
| kernel | specific test | conf |
|---|---|---|
| `experimental/deepseek/moe/moe_gate_mm/.../dm1.cpp` | `tests/ttnn/nightly/.../experimental/test_moe_mm.py::test_moe_mm[check_accuracy_True, M,K,N,L,C=(32,7168,256,1,1), "dispatch_row"]` — single device, kernel always created | 🔶 (high; basename `dm1` generic — confirm program hash) |
| `experimental/deepseek_prefill/dispatch/.../reader_dispatch.cpp` | `models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_dispatch.py::test_ttnn_dispatch` (min mesh 2×2) | ⚠️ gap (multi-device; COUNTER inc_multicast leg unreachable single-card) |
| `experimental/deepseek_prefill/combine/.../reader_combine.cpp` | `models/demos/deepseek_v3_d_p/tests/pcc/test_prefill_combine.py::test_ttnn_combine` (min mesh 2×2) | ⚠️ gap |

---

## ccl (experimental, mostly multi-device)

Comprehensive sets are listed per kernel below; ✅/🔶 only where single-card-reachable.

| kernel | specific test | conf |
|---|---|---|
| `ccl/moe_compute/.../tilize_reader.cpp` | `tests/ttnn/nightly/.../experimental/test_moe_compute_single_card.py::test_moe_compute_single_card_gpt_oss` (1×1 mesh) | 🔶 (single-card; basename shared w/ moe_gpt — confirm program hash) |
| `ccl/moe_compute/.../tilize_writer.cpp` | same `single_card_gpt_oss` nodeid | 🔶 |
| `ccl/rms_allgather/.../rms_sender_reader.cpp` | `tests/ttnn/unit_tests/operations/ccl/fusion_subtests/rms_test.py` (or `test_minimals.py -k rms`) — mesh | ⚠️ gap |
| `ccl/rms_allgather/.../rms_writer.cpp` | same rms test — mesh | ⚠️ gap |
| `ccl/llama_all_gather_matmul_async/.../worker_receiver.cpp` | `tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py` — ring_size>1 | ⚠️ gap |
| `ccl/moe_gpt/.../tilize_reader.cpp` | `tests/ttnn/nightly/.../experimental/test_moe_gpt_e2e.py::test_dispatch_compute` (4×8 mesh) | ⚠️ gap |
| `ccl/moe_gpt/.../tilize_writer.cpp` | same `test_dispatch_compute` — mesh | ⚠️ gap |
| `ccl/moe/selective_reduce_combine/.../reader.cpp` | `tests/nightly/tg/ccl/moe/test_selective_combine_6U.py` (galaxy) — single-card moe_compute BYPASSES it via compute_only=True | ⚠️ gap |
| `ccl/moe/selective_reduce_combine/.../writer.cpp` | same `test_selective_combine_6U.py` — galaxy | ⚠️ gap |
| `ccl/all_gather_concat_heads_fused/.../llama_all_gather_concat_writer.cpp` | `tests/ttnn/unit_tests/operations/ccl/fusion_subtests/concat_fuse_test.py` (or `test_ccl_async_TG_llama.py -k concat`) — TG/mesh | ⚠️ gap |
| `ccl/all_to_all_async_generic/.../all_to_all_sender_writer.cpp` | `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_to_all.py::test_all_to_all` (bh_1d_mesh) | ⚠️ gap |

### programming example
| kernel | specific test | conf |
|---|---|---|
| `tt_metal/programming_examples/contributed/multicast/.../coordinator_kernel.cpp` | **NO pytest** — `multicast` example binary only | ⚠️ |

---

## Rollup
- **✅ device-verified (BH p150a):** all 4 interleaved-in0 matmul kernels, block-sharded matmul, both
  topk kernels, 4 sharded group_norm v2 (legacy+welford), 6 conv2d sharded kernels, both plain sharded-LN
  kernels, sdpa `reader_interleaved`, deepseek `sampling_kernel`. (~21 kernels.)
- **🔶 static, single-chip reachable (run + JIT-grep to confirm):** dram-sharded matmul, group_attn,
  conv3d writer, distributed-LN pre/post (simulated), sdpa_decode dataflow_common, argmax multicore,
  sort ×3, move ×2, moe_gate_mm dm1, moe_compute tilize ×2, deepseek kv_cache/flash_mla.
- **⚠️ gaps (no single-chip/no test):** 4 interleaved group_norm kernels (sweep-only), ring/joint sdpa
  (chain_link.hpp + exp_ring_joint), ring all-gather matmul ×2, all multi-device CCL kernels
  (rms_allgather, llama AG-matmul, dispatch/combine, moe_gpt, selective_reduce_combine, all_gather_concat,
  all_to_all), and the 4 matmul/multicast programming-example kernels (binary only).
