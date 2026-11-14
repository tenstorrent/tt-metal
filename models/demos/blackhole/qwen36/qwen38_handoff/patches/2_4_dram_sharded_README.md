# Change 2.4: QWEN36_BFP4_DSHARD (DRAM-sharded multi-reader decode matmuls for BFP4 groups)

Scratch tree: /home/ttuser/atupe/tt-metal/.claude/worktrees/wt-dshard (detached 755a0d6461d + base.diff + this change). Not committed. NOT run on device (host py_compile only).

Files here: base.diff (Tier-2 state of qwen38-optimizations, qwen36 sources only, no *.md; text_demo.py / model_targets.yaml were NOT in it at capture time; text_demo.py now shows edits from another agent, ignored), dshard_full.diff (wt-dshard vs HEAD = base + change), dshard.patch (only the 2.4 change on top of base; `git apply --check` passes on qwen38-optimizations).
Changed files (dshard.patch): tt/tp_common.py (+~170), tt/mlp.py, tt/attention/tp.py, tt/gdn/tp.py. moe/shared.py untouched.

## Design
- Flag `QWEN36_BFP4_DSHARD` (default "1"), applied per group only when that group's BFP4 flag is on: down (QWEN36_BFP4_MLP_DOWN), qkv + wo (QWEN36_BFP4_ATTN), gdn_in (QWEN36_BFP4_GDN_IN, opt-in). BFP8 groups keep the 1D path. One INFO line per process: "QWEN36_BFP4_DSHARD=1: DRAM-sharded multi-reader decode matmuls for ['down','qkv','wo'] (...)" (`dshard_group_enabled`).
- Weights: for each eligible group an EXTRA bfloat4_b DRAM WIDTH_SHARDED copy, shard [K, N/banks] over banks=mesh.dram_grid_size().x (8 on P150), new cache stems `mlp.down_proj.weight.dsh4.tp`, `wqkv_fused_qkvg.dsh4`, `wo.dsh4`, `qkvzab.dsh4` (about 6.7 GB extra DRAM total, first load creates files). Interleaved copies unchanged (prefill, flag-off). Same dtype/fidelity as the 1D path, so numerics are the same up to accumulation order.
- Matmul: `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w, per_core_M=1, per_core_N, fused_activation=None, num_workers_per_dram_bank=readers)` (parameter exists in matmul_nanobind.cpp; B passes it identically, decoder.py:340-350). Activation `to_memory_config` to L1 WIDTH_SHARDED [32, K/cores] on cores (0,0)..(cores-1,0); `ttnn.linear(..., memory_config=L1_WIDTH_SHARDED, dtype=bf16)`; then `sharded_to_interleaved` to the placement the old 1D path produced (class `tp_common.DShardMatmul`).
- Output conversion (cheapest SAFE choice): back to the exact same memory config the 1D path returned, so every consumer is unchanged: down -> L1 interleaved (feeds decode_ar.all_reduce, which reshards to RES40 itself, or tt_all_reduce); wo -> DRAM interleaved (feeds reshape (1,1,B,N) + decode_ar / tt_all_reduce; lean path at attention/tp.py ~826 and legacy ~720 both go through `_wo_proj`); qkv -> DRAM interleaved (slices to qkv3/gate and `_make_heads_decode` unchanged, lean `decode_qkv3_l1` still slices into L1); gdn_in -> L1 when out_mc given (decode), else DRAM. Cost about 1-1.7 us/op (recon). Possible follow-up (not done, needs device validation): feed the 8-core width-sharded down/wo output straight into decode_ar.all_reduce's to_memory_config(res_mem) and skip sharded_to_interleaved; reshape of a sharded 3D wo output is the risk there.

## Per-group arithmetic (TP=4, tile=32, banks=8, B's cores/block/readers; dim=5120, hidden=17408, 6 local q-heads x 256, 1 kv head)
align = banks*readers*32 = 512. shard_width = N/banks; per_core_N = ceil(shard_tiles*banks/cores).
| group | K | N | cores | K tiles/core | block | N pad | shard w (tiles) | per_core_N | readers |
|---|---|---|---|---|---|---|---|---|---|
| down | 17408/4=4352 (136t) | 5120 (160t) | 8 | 136/8=17 | 17 | 5120 (=10*512) | 640 (20) | ceil(20*8/8)=20 | 2 |
| gdn_in | 5120 (160t) | 4096+64=4160 (130t) | 10 | 160/10=16 | 16 | 4608 (=9*512) | 576 (18) | ceil(18*8/10)=15 | 2 |
| qkv | 5120 | 6*256*2+2*256=3584 (112t) | 10 | 16 | 16 | 3584 (=7*512) | 448 (14) | ceil(14*8/10)=12 | 2 |
| wo | 24*256/4=1536 (48t) | 5120 | 8 | 48/8=6 | 6 | 5120 | 640 (20) | 20 | 2 |
All four match B's table exactly (K/N equal to B's TP4 shapes). `dsh_geometry` re-checks divisibility (K % 32*cores, K/32/cores % block, N % 512, compute grid x >= cores) and returns None otherwise -> warning once, group keeps 1D path (e.g. 35B-A3B shapes are rejected this way; MoE shared expert is excluded explicitly because it passes down_dtype=bf8).
GDN padding: fused [tp*4160, K] is zero-padded PER DEVICE block to 4608 rows before `shard_w` (dim=-1 after transpose); the matmul output is [.., 4608]. No crop op is needed: `_project_qkvzab` slices stop at 4096 (qkv), 4096 (z), and a/b's tile 4096..4128 (`_ab_end = min(az+32, shape[-1])`), so the 448 zero columns are never read.
M: per_core_M=1 covers any M<=32 rows (batched decode B<=32 is tile-padded to 32 rows; x may be 3D [1,B,K] or 4D, passed through as the existing sharded paths do). Prefill (M>32) never enters these branches (guarded by shape[-2] <= TILE_SIZE).

## Call sites changed
- tt/mlp.py: MLPWeights.w2_dsh; load_mlp_weights (interleaved-return branch, dense + mlp_1d_decode + bf4 only); Qwen36MLP.__init__ builds `self._w2_dsh`; `_forward_tp` w2 decode branch (replaces the `ttnn.linear(hidden, w.w2, ...)` when `hidden.shape[-2] <= 32`).
- tt/gdn/tp.py: load_gdn_weights_tp (`tw["qkvz_dsh"]`, padded); TPGatedDeltaNet.__init__ `self._qkvz_dsh`; `_project_qkvzab` decode branch (before the 1D branch). Covers all decode callers (fused batched, forward_decode, _forward_decode_fused).
- tt/attention/tp.py: load_attention_weights_tp (`tw["wqkv_fused_dsh"]`, `tw["wo_dsh"]`); TPAttention.__init__ `_qkv_dsh/_wo_dsh`; `_qkv` decode branch; `_wo_proj` decode branch (so legacy decode, lean decode and paged all pick it up).
- tt/tp_common.py: flag/log, DSH_PARAMS, dsh_geometry/dsh_padded_n/dsh_weight_memcfg/dsh_in_memcfg, load_dsh_weight, DShardMatmul, make_dsh. Prefill untouched everywhere.

## Risks
- L1: activation shards are small ([32,544]/[32,192]/[32,512] bf16, <35 KB/core) on row-0 cores 0..7/9; output L1 width-shard transient is per_core_N*32*2*32B <= 31 KB/core (gdn 480 cols). Row 0 overlaps nothing persistent in the decode path as far as read, but the DRAM-sharded matmul CBs with 2 readers are bigger than 1 reader: watch for L1 clashes with persistent RES40 buffer (cores 0..9 x 0..3 contain row 0!) and with trace-captured L1 tensors; the patch_1_5 QKV run at the same cores/block worked, down (8 cores, block 17, 544-wide) and wo are new.
- Padding: GDN out is 4608 wide (logical padded); verified by reading the slices only; zero rows in weight so padded columns are exact zeros.
- Cache: new stems => first run builds ~6.7 GB of files (quantisation on host); the eager torch pad of the GDN fused weight runs even on cache hits (small).
- bf4 DRAM-sharded as_tensor with multi-reader alignment is unproven for the "dsh4" stems (bf8 proven in patch_1_5); B loads the same layout so should be fine. A cache hit restores the stored layout, so do not reuse stems across geometry changes.
- Extra DRAM per chip; `mesh.dram_grid_size().x` assumed 8 (geometry adapts, parameters validated).
- Batched decode: M=32 rows with per_core_M=1 relies on tile padding of B<32; test_model_tp decode_batched covers it.
- 1.1 all-reduce interplay: partial layout unchanged by design (interleaved), so no change to DecodeResidualAllReduce.

## Test plan (device, not run)
1. Flag-on vs flag-off PCC (QWEN36_BFP4_DSHARD=1 vs 0, same BFP4 flags): test_mlp_tp (decode + prefill), test_attention_tp (+_paged, _paged_peruser), test_gdn_tp with QWEN36_BFP4_GDN_IN=1 (test_gdn_tp, _fused_batched_decode, _fused_batched_decode_trace, _peruser_state), test_model_tp contract + decode_batched, test_decode_bucketing (test_decode_width_scaling_traced for trace-safety), test_prefill_trace_any_len. Expect essentially identical PCC (same dtype/fidelity; only accumulation order).
2. Check the INFO line and that .dsh4 files appear; run with QWEN36_BFP4_DSHARD=0 to confirm bit-identical old behaviour (no extra weights loaded).
3. Perf: traced decode t/s/u (traced_128) flag on vs off, expect ~ -0.9 to -1.0 ms/token (recon: down 8.4us x64, gdn_in 5.6 x48, qkv 11.7 x16, wo 2.7 x16, minus ~1-1.7us/op conversions); accuracy_512 top-1/top-5 unchanged vs DSHARD=0.
4. If an L1 clash appears: disable per group by dropping its DSH_PARAMS use (or QWEN36_BFP4_DSHARD=0) and bisect down/wo first.
