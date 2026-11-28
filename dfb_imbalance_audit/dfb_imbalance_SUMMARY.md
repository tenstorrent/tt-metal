# DFB imbalance audit (PR #57646 forced finish()) — summary
Per-family details: dfb_imbalance_{conv_pool,matmul,layout_dm,eltwise_reduce_norm,transformer,ccl_misc}.md
Q = ttnn/cpp/ttnn/operations/experimental/quasar

finish_impl semantics (dataflow_buffer.inl:274): DM waits read_acked==read_posted; TRISC waits posted==0.
=> hangs on push-without-pop / wait-without-pop. Dangling reserve_back (reader_pool_2d is_large_kernel) posts nothing: not a hang unless reserve==push is also enforced.

## Exercised by tests (CONFIRMED)
| # | Kernel | DFB | Imbalance | Hit by |
|---|---|---|---|---|
| 1 | Q/conv2d/.../conv_bmm_tilize_metal2.cpp | OUT (borrowed self-loop) | push M*N, pop 0 | every fused conv (resnet) |
| 2 | Q/conv2d/.../conv_bmm_tilize_metal2.cpp:723 | bias | wait, no pop | all resnet convs (bias) |
| 3 | Q/pool_generic/.../compute_pool_2d.cpp:141 | in_scalar (one_scalar_per_core) | wait, no pop | max_pool2d, global avgpool |
| 4 | Q/pool_generic compute (TILE out) | out_cb self-loop | push, pop 0 | resnet global avg_pool2d |
| 5 | Q/halo/.../halo_gather.cpp:296 vs compute | src | reader pushes input_npages, compute pops referenced blocks | shape-dependent tail cores |
| 6 | Q/matmul/.../reader_bmm_tile_layout_in1_{sender,receiver}_writer_padding_metal2.cpp:498/252 | out (OUT_SHARDED) | wait, no pop | resnet fc, test_linear, kspill |
| 7 | Q/pad/.../writer_pad_tiled.cpp:53/62 | pad_val | push 1, pop 0 | llama prototype_ops/test_pad |
| 8 | Q/transpose/.../transpose_wh_rm_sharded.cpp:110 | out (borrowed) | push Ht/wait, no pop | resnet test_fold_transpose |
| 9 | reduction/sampling/.../writer_interleaved.cpp:77/88 | k, p | push 1, pop 0 | all ttnn.sampling |
| 10 | normalization layernorm sharded writers :68-70 / compute | eps | push 1, never waited/popped | rms_norm sharded (llama, qwen) |
| 11 | layernorm_sharded.cpp :669/:692 | out (borrowed self-loop) | push, pop 0 | same |
| 12 | layernorm.cpp :431/:434 vs readers | gamma/beta | push round_up(Wt,blk), pop Wt | qwen rms_norm W=128 |
| 13 | transformer/sdpa writer_interleaved.cpp:105-125 / compute | identity_scale, col_identity, causal mask | push, wait, no pop | all SDPA prefill/chunked |
| 14 | sdpa_decode writer_decode_all.cpp:228/234/254 | identity_scale, causal mask | push, wait, no pop | SDPA decode (paged+non) |
| 15 | rotary_embedding_llama_sharded.cpp:93/165 | out (borrowed self-loop) | push Ht*Wt, pop 0 | llama e2e decode, qwen rotary |
| 16 | rotary_embedding_llama_sharded{,_row_major}.cpp (fused_qk) | q_out/k_out/out | push, pop 0 | fused_qk op tests |

## Harmless under current finish (reserve-ahead, like reader_pool_2d)
reader_pool_2d.cpp:68/145-146; reader_bmm_tile_layout_in1_sender_dram_sharded.cpp (mainline+Q); manual_seed readers scratch reserve; Q writer_unary_pad_dims_interleaved.cpp:31.

## Not exercised but confirmed — see family files (~50 items). Notable: RM sharded_to_interleaved with non-divisible height; untilize_with_unpadding sharded writers; CCL packet_header DFBs (~15 writers); sharded softmax; fill_pad mask push on zero-work cores.
