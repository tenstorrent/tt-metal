# Device-verified kernel→test map — `mcast_pipe` rollout

**Machine:** single-chip Blackhole p150a (`MeshDevice 1x1`). **Multi-device CCL tests cannot run
here** — their kernels are *environment gaps*, not blind-migratable.

**Verification method (the JIT-build-cache proof):** `touch` a marker, run the candidate nodeid
under `scripts/run_safe_pytest.sh`, then check `built/tt-metal-cache*/kernels/<basename>/` — a dir
newer than the marker means the program *dispatched* (JIT-compiled) that kernel. JIT only builds
kernels a program actually instantiates, so a fresh dir == the test exercises the kernel. A green
test + fresh build = **verified**.

Full static per-family detail: `test_map_{matmul,conv,normalization,sdpa,data_movement,ccl}.md`.
Machine-readable: `test_map.json`.

## VERIFIED (green test + fresh JIT build) — 20 kernels, migration-safe with cheap validation

| # | kernel | family | tag | validation (one fast nodeid) |
|---|---|---|---|---|
| 1 | reader_bmm_tile_layout_in0_sender_padding | matmul | clean | test_matmul 1d mcast_in0 in_sharded=False m=256 |
| 2 | reader_bmm_tile_layout_in0_receiver | matmul | clean | (same 1d) |
| 3 | reader_bmm_tile_layout_in1_sender_writer_padding | matmul | clean | (same 1d; also 2d) |
| 4 | reader_bmm_tile_layout_in1_receiver_writer_padding | matmul | clean | test_matmul 2d in0_sharded=False grid(8,4) |
| 5 | reader_final_topk | reduction/topk | clean | test_topk W=8192 k=50 BFLOAT16_B |
| 6 | writer_local_topk | reduction/topk | refactor | (same topk) |
| 7 | reader_mcast_receiver_unary_sharded_gn_v2 | groupnorm | clean | gn block_sharded_v2_8x4 legacy 1280-16-16 |
| 8 | reader_mcast_sender_unary_sharded_gn_v2 | groupnorm | refactor | (same gn legacy) |
| 9 | welford_reader_mcast_sender_unary_sharded_gn_v2 | groupnorm | refactor | gn block_sharded_v2_8x4 welford 1280-16-16 |
| 10 | welford_reader_mcast_receiver_unary_sharded_gn_v2 | groupnorm | refactor | (same gn welford) |
| 11 | reader_writer_tiled_out_1d_mcast_sender_conv_weights… | conv | clean | conv_features HEIGHT_SHARDED 16/16/256 |
| 12 | reader_writer_tiled_out_1d_mcast_receiver_conv_weights… | conv | clean | (same conv HS) |
| 13 | writer_tiled_out_2d_mcast_sender_conv_weights… | conv | clean | conv_features BLOCK_SHARDED 128/128/32 |
| 14 | writer_tiled_out_2d_mcast_receiver_conv_weights… | conv | clean | (same conv BS) |
| 15 | activation_reader_width_sharded | conv | refactor | conv_features WIDTH_SHARDED 353/384/8 |
| 16 | reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2 | conv | refactor | (conv BS; chunked sub-case unverified) |
| 17 | reader_mcast_sender_unary_sharded_ln | layernorm | refactor | test_layer_norm_sharded_single_stage 256/512 4x4 |
| 18 | reader_mcast_receiver_unary_sharded_ln | layernorm | refactor | (same sharded_ln) |
| 19 | reader_interleaved (sdpa) | sdpa | refactor | test_sdpa_noncausal b1 nh8 s2048 bf16 |
| 20 | sampling_kernel (deepseek_v3_b1) | ccl/deepseek | clean | test_sampling_argmax_single_device_101_cores[17-0] |

Clean: 1-4, 5, 7, 11-14, 20 (11). Refactor: 6, 8-10, 15-19 (9).

## COVERAGE GAPS (not migrated, or migrated-at-risk) — see test_map.json `gaps`

- **No green test on this machine (build-seen only):** `dm1` (moe_gate_mm), `moe_compute` tilize
  reader/writer — their tests fail fast on a single card.
- **Multi-device only (cannot run on 1 chip):** ln_pre/post_allgather (4 kernels); all CCL legs
  (rms_allgather, llama AG-matmul, deepseek_prefill dispatch/combine, moe_gpt, selective_reduce,
  all_gather_concat, all_to_all) — ~11 kernels.
- **No test coverage at all (sweep-only):** the 4 interleaved (non-v2) group_norm mcast kernels —
  every group_norm test shards input first.
- **Unverified (not run / mode-dependent):** argmax multicore reader; conv3d writer; matmul
  in0_sender_dram_sharded (nightly); 3 matmul didactic examples (no pytest).

## DEFERRED by invocation (out of scope this round)
R6 same-core role-flip (matmul block-sharded `_in0_sender_receiver`, group_attn); R4 streaming
chunked-send (conv halo sub-case); legacy-API move/sort (need Noc/Semaphore port first); fabric /
ring CCL legs; `chain_link.hpp` (prior-art) + deepseek `mcast.hpp` (preprogram-state).
