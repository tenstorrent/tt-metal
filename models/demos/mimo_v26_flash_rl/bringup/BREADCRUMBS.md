# mimo_v26_flash_rl bring-up breadcrumbs

## C.attn_dense.attn_norm (layer 0)

- `tt/rms_norm.py` `RMSNorm`: stock `ttnn.rms_norm`, weight `[1, 1, 1, 4096]` bf16 TILE replicated, eps from config
  (1e-6), plain `w` (not `1 + w`). Compute config from `tt/settings.py` `NORM_FIDELITY` (HiFi4) / `NORM_FP32_ACC`
  (True); both are in MAX_PRECISION.
- Runs on each chip's local residual rows `[1, 1, 160, 4096]` (plan: SP row r = tokens r*640.., TP column c = rows
  c*160.. of those). The all_gather over TP before qkv is not part of the norm: the attention module owns it.
- `tt/mesh_io.py`: host <-> mesh layout of the residual (`ShardTensor2dMesh` dims (0, 1) over [8, 4] of
  x.reshape(8, 4, 160, 4096)); used only by the component adapters in `bringup/hooks.py`.
- `bringup/hooks.py` runs the component adapter with `ttnn.CONFIG.enable_fast_runtime_mode = False` (`_op_hooks`):
  in fast runtime mode (the C++ default on this tt-metal) ttnn skips its pre-operation hooks, so the framework's
  device proof counted 0 ops (`device_ops_attn_norm = 0`) although the op ran. Reuse `_op_hooks` for every component.
- Gate: PCC 0.999998, rel vs CPU 0.00198 (limit 0.0061), auto_margin_min 2.83, 1 ttnn op, host FLOPs 0, determinism ok.
- Re-run: `bfw job check --task C.attn_dense.attn_norm --gate-only`.

## C.attn_dense.attention (layer 0), attempt 1: not passed yet

- `tt/attention.py` `FullAttention`: all_gather over TP (`tt/ccl.py gather_rows`) -> one fused qkv `ttnn.linear` per
  chip from `rank_weights` (the checkpoint's own rank slab: 16 q + 1 k heads x 192, v 128 zero-padded to 192 so
  `nlp_create_qkv_heads` splits it; q rows x 192^-0.5 and v rows x 0.707 folded in fp32 before bfp8; SDPA runs at
  scale 1) -> HF rotate_half RoPE on dims 0-63 (`ttnn.experimental.rotary_embedding`, cos / sin rows looked up with
  `ttnn.embedding` at each local row's absolute position from `ChunkMeta.positions`) -> v sliced to 128 ->
  `update_padded_kv_cache` + `zero_padded_kv_cache(pad_align 32)` into the served caches (tensor/metadata form) ->
  `ring_joint_scaled_dot_product_attention` causal over SP axis 0 (separate K 192 / V 128, `slot_id` +
  `kv_actual_isl_tensor`, gather buffers of the full 56320 rows) -> `nlp_concat_heads` -> o_proj on this rank's 2048
  input columns -> `reduce_scatter` over TP (`scatter_rows`).
- `tt/kv_cache.py`: served caches `[slots 2, 1, 7040, 192 | 128]` bfp8 ND-sharded `[1, 1, 32, w]` ROUND_ROBIN_1D,
  allocated + DRAMZeroFill on device; `chunk_positions(start)` = the server's block-cyclic reshuffle (checked on CPU
  against the contract's start-2944 example). `FullAttention.load_prefix` (harness boundary only) places the golden
  prefix in that layout; `rank_weights` checked on CPU against `Checkpoint.qkv` (max err 5e-7).
- Ring SDPA with `kv_actual_isl_tensor` runs only its streaming compute: `fp32_dest_acc_en=True` raises TT_FATAL
  "kv_actual_isl requires the ring-joint streaming compute path" (ring_joint_sdpa_program_factory.cpp:1397). The
  SDPA compute config is therefore HiFi4 + fp32 dest off (no SDPA fp32 switch in settings.py).
- Run 2 hit that TT_FATAL on the 8x4 mesh and the device then timed out (dispatch timeout 5 s; triage: every worker
  idle, fabric routers mid receiver step). The board stayed hung (stage reset refused), so the fix (fp32 dest off)
  is not yet measured. Next attempt: run the gate first; it runs with `--dev` after a hang.
- Re-run: `bfw job check --task C.attn_dense.attention --gate-only`.
- Attempts 2-4 (16:48-16:51 UTC, bh-glx-120-b04u08): the gate failed in the fixture before any model code, at
  `ttnn.get_pcie_device_ids()`: `Read 0xffffffff over PCIe ID 24: the board should be reset` (UMD also logs
  `Setting power state failed on device 24: Input/output error`). The automatic reset after attempt 1's hang did not
  bring chip 24 back, so a person has to reset or power-cycle it. The attempt-1 code (SDPA fp32 dest off) is still not
  measured on the device. Next attempt: run the gate first.

## C.attn_dense.attention (layer 0), attempt 12: PASS

- Attempt 1's fix (ring SDPA fp32 dest off) measured on bh-glx-120-b06u08: PCC 0.99998, but auto_margin_min 0.236
  (output column 3820 rel 0.119 vs limit 0.028; the x 2^-k "small" probe 0.038 > 0.03).
- Root cause, from device dumps of q / caches / heads: inputs are fine (fp32 SDPA on the device q/k/v is 0.0035 from the
  reference heads), but the source op's streaming SDPA added 0.0130 rel error on its own. Most of it is the
  hard-coded Schraudolph exp (`exp_packthread_tile<true>` in compute_streaming.hpp `sub_exp_block_bcast_cols`, ignores
  exp_approx_mode); o_proj's cancelling columns (sum |terms| / |sum| up to 46) turn it into single-column errors.
- Changes:
  - fork `ttnn.bringup.sdpa` (ttnn/ttnn/bringup/sdpa, CHANGELOG): `exact_exp=True` on
    `ring_joint_scaled_dot_product_attention` uses the accurate bf16 exp. Default off = the source program. Setting
    `SDPA_EXACT_EXP` (True). The SDPA's own error went from 0.0130 to 0.0064.
  - qkv and o_proj weights kept [out, in] and run with `transpose_b=True`, so the bfp8 blocks run along K like the
    precision model's. Stored [in, out], o_proj blocks run along the output, which put 0.09 on column 3820 by itself.
  - `ATTN_OUT_FP32`: o_proj partials in fp32 through the TP reduce_scatter, typecast to bf16 once.
  - `SDPA_K_CHUNK` 320 (margin 128: 0.88, 160: 0.87, 320: 0.99, 640: 0.94) and `SDPA_SEGMENTED` False
    (with segmented accumulation on at k 320: 0.989).
- Gate: PCC 0.99999, rel vs CPU 0.0040 (limit 0.0052), worst column 452 at 0.0256 vs 0.0256: margin 1.003, which is
  thin. 29 ops, host FLOPs 0, determinism ok. The remaining error is the streaming SDPA's bf16 running out / sum
  (fp32 dest is not allowed on the kv_actual_isl path). Next lever if a later layer misses: Float32 sum / out CBs in the
  fork (as the GLM sparse_sdpa high_precision fork did).
- Re-run: `bfw job check --task C.attn_dense.attention --gate-only`.

## C.attn_dense.attn_residual (layer 0): PASS

- `tt/residual.py` `ResidualAdd`: one `ttnn.add` (bf16 out, DRAM) on each chip's local `[1, 1, 160, 4096]` rows; both
  operands are already in the residual row layout (attention ends with its TP reduce_scatter), so there is no collective.
  Wired as `attn_residual` in `bringup/hooks.py` (`_residual_component`, two-input adapter via `tt/mesh_io.py`); the
  same class serves `mlp_residual` when that task wires it.
- Job attempts 2-6 never reached the code: the container image pull failed (`GHCR ... unauthorized`).
- Gate: PCC 0.999997, rel vs CPU 0.00176 (limit 0.0035), auto_margin_min 1.98, 1 ttnn op, host FLOPs 0, determinism ok.
- Re-run: `bfw job check --task C.attn_dense.attn_residual --gate-only`.

## C.attn_dense.ffn_norm (layer 0): PASS

- Reuses `tt/rms_norm.py` `RMSNorm` (the attn_norm module) with `post_attention_layernorm.weight`; wired as `ffn_norm`
  in `bringup/hooks.py` through `_norm_component`. Output stays on the local `[1, 1, 160, 4096]` residual rows; the
  TP all_gather the MLP needs belongs to the MLP module.
- Gate: PCC 0.999996, auto_margin_min 2.49, 1 ttnn op, host FLOPs 0.
- Re-run: `bfw job check --task C.attn_dense.ffn_norm --gate-only`.

## C.attn_dense.mlp (layer 0): PASS

- `tt/mlp.py` `DenseMLP`: all_gather over TP (`ccl.gather_rows`) -> gate / up `ttnn.linear` with this rank's 4096
  intermediate rows (fp32 out) -> `ttnn.multiply(g, u, input_tensor_a_activations=[SILU])` (fp32) -> down
  `ttnn.linear` on this rank's 4096 input columns (fp32 partials) -> `reduce_scatter` over TP -> typecast bf16.
  7 ttnn ops. Weights [out, in] with `transpose_b=True` (bfp8 blocks along K, as attention learned).
- Settings: `MLP_FIDELITY` HiFi4, `MLP_FP32_ACC` True, `MLP_INTER_FP32` True, `MLP_OUT_FP32` True (all in MAX_PRECISION).
- `ttnn.multiply` on this tt-metal takes no `compute_kernel_config` (TypeError); activations need
  `ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)`.
- First run (fp32 dequantized weights -> bfloat8_b): PCC 0.999995 but auto_margin_min 0.77, output column 525 rel 0.0288
  vs limit 0.0223. Host emulation with ttnn's own converter: fp32 -> bfp8 gives column 525 0.029, bf16 -> bfp8 gives
  0.009 (= the precision model). Fix: round the dequantized weights to bf16 first (`rank_weights`).
- Gate: PCC 0.999995, rel vs CPU 0.0029 (limit 0.0056), worst column 2415 at 0.0149 vs 0.0279, auto_margin_min 1.87,
  host FLOPs 0.
- Re-run: `bfw job check --task C.attn_dense.mlp --gate-only`.

## C.attn_dense.mlp_residual (layer 0): PASS

- Reuses `tt/residual.py` `ResidualAdd` (one `ttnn.add`, bf16, DRAM); `device_component` routes `mlp_residual` to the
  same `_residual_component` as `attn_residual`. Operands are the local `[1, 1, 160, 4096]` residual rows (the MLP ends
  with its TP reduce_scatter), so no collective.
- Gate: PCC 0.999997, auto_margin_min 2.00, 1 ttnn op, host FLOPs 0.
- Re-run: `bfw job check --task C.attn_dense.mlp_residual --gate-only`.

## C.sliding_moe.attention (layer 1): PASS

- `tt/attention.py` `Attention` (was `FullAttention`) serves full and sliding layers from `layer_dims(cfg, layer)`
  (the reference's `swa_` keys: 8 kv heads -> 2 per chip, theta 1e4, window 128, sink). Layer 0's path is unchanged
  (its gate re-run: bit-identical metrics). `hooks._attention_component` builds RoPE from the layer's theta and
  `ChunkMeta(window=...)`.
- Sliding forward: qkv -> RoPE -> v x 0.707 -> pad rows zeroed by `ChunkMeta.valid_rows` (multiply) -> served K / V and
  an unserved 192-wide zero-padded V (`LayerKV.attn_value`) written with `update_padded_kv_cache` ->
  `ttnn.bringup.ring_joint_scaled_dot_product_attention(sliding_window_size=128, attention_sink, exact_exp)` with
  compact gather buffers of 2 x 128 rows -> **its output is discarded**; its gathered halo (K / V rows before each
  run of the chip's rows) plus the chip's own bf16 k / v feed a composed fp32 softmax (`_window_softmax`: matmul
  fp32 -> x 192^-0.5 (fp32 tensor) -> + window mask -> max with sink -> exp -> sum + exp(sink - max) -> p @ v ->
  divide), per kv group -> concat heads (fp32) -> o_proj -> TP reduce_scatter.
- `zero_padded_kv_cache` takes single-head caches only (TT_FATAL "cache num-heads dim must be 1"): with 2 kv heads
  the pad rows are zeroed before the write instead (k, v x row-validity mask).
- Why not the op's softmax: the sink holds nearly all the mass (scores ~ -24, sink ~ 0.8, sum exp(scores)/exp(sink)
  ~ 1e-10), so the output scales as exp(score - sink). The streaming kernel (bf16 dest, required with the sink /
  kv_actual_isl) truncates scores to bf16 (ulp 0.125 at |24|): device SDPA alone 0.015 rel vs fp32 SDPA on the same
  device q / K / V, row norms +6 %, gate rel 0.037 and bias -0.038. Composed fp32: rel 0.014, margin 1.006.
- Other layer-1 fixes, each measured on the gate: scales applied after the projection, not folded into the bfp8
  qkv (`rank_weights(fold_scales=False)`; folded moved column 2331 to 0.19 vs 0.16); sliding weights rounded fp32 ->
  bf16 -> bfp8 (`SWA_WEIGHTS_BF16_FIRST`; column 529 0.15 -> 0.11); halo rows inside the current chunk taken from the
  SP all-gathered bf16 k / v via a one-hot matmul (`ChunkMeta.halo_select`), not from the cache: the device bf16 ->
  bfp8 typecast on the cache write is biased +0.3 %, which gave each SP slab's first 32 rows a -5 % norm error.
- Settings: `SWA_SDPA_Q_CHUNK` 128, `SWA_WEIGHTS_BF16_FIRST` True, `SWA_OUT_FP32` True (k chunk fixed 128 by the op).
- Gate: PCC 0.999976, rel vs CPU 0.0139 (limit 0.0264), worst column 529 0.1115 vs 0.1122: margin 1.006 (thin),
  88 ops, host FLOPs 0, determinism ok. Remaining: median row-norm bias -1.3 % (limit 2.5 %), not yet explained (CPU
  emulation of the composed path on the dumped device tensors matches the device to 0.0013).
- Cost / next: the ring SDPA computes a discarded output (only its halo gather is used) and the composed softmax
  materialises [8, 640, 896] fp32 scores per kv group; P.* can replace both by a fork of the streaming kernel with
  fp32 scores / an fp32 sink term. The scratch dumps used for the diagnosis are not kept.
- Re-run: `bfw job check --task C.sliding_moe.attention --gate-only`.

## C.sliding_moe.router (layer 1): PASS

- `tt/router.py` `Router`: input is the column's TP-gathered ffn_norm rows [1, 1, 640, 4096] bf16 (every TP column
  routes the same 640 rows, the input its dispatch group needs). gate [256, 4096] bf16 replicated -> `ttnn.linear`
  (HiFi4, fp32 dest, fp32 out) -> `sigmoid` fp32 -> `add` fp32 bias -> `topk(k=8, sorted)` -> slice the 8th value
  -> `ge` mask -> scores x mask -> `sum` (fp32 dest) -> `div`. Output dense [1, 1, 640, 256] fp32, the reference's
  layout.
- Bias: rounded to bf16 (config moe_router_dtype, as the reference), then recentred by a bf16 constant near its mean
  so `bias - c` is exact in fp32 and the choice score sits near 0 instead of 2 (known issues on noaux_tc precision).
  `moe_grouped_topk` (TF32 keys) not used.
- Dense output via a mask on the 8th choice value instead of `ttnn.scatter` (no fp32 TILE scatter): no layout
  round-trip, indices unused. An exact fp32 tie at the 8th value would select 9 (none on the golden / probes).
- Hook `_router_component`: residual_to_mesh -> `ccl.gather_rows` -> router -> read column 0 of each SP row.
- Settings: `ROUTER_FIDELITY` HiFi4, `ROUTER_FP32_ACC` True (both in MAX_PRECISION).
- Gate: PCC 0.999338, overlap vs CPU / golden 0.99868, matched-row rel vs CPU 8.9e-5, auto_margin_min 3.06, 10 ops,
  host FLOPs 0, deterministic.
- Re-run: `bfw job check --task C.sliding_moe.router --gate-only`.

## C.sliding_moe.experts (layer 1): PASS

- `tt/experts.py` `Experts`: EP over the whole mesh, 4 dispatch groups (one per TP column), each 8 chips along SP;
  chip (r, c) holds experts (8c + r) * 8 + [0, 8) (DeepSeek column-major `get_global_expert_idx`). Forward per chip:
  routing [1, 1, 640, 256] fp32 -> bf16 -> `topk(8)` (ids uint16, weights bf16) -> `masked_bincount` (dispatch table
  masks the other groups) -> `offset_cumsum(cluster_axis=0)` -> x: `ccl.gather_rows` -> ROW_MAJOR -> `dispatch`
  (axis 0, Linear, 1 link) -> flat buffer -> TILE -> per local expert `extract` (cap 5120) -> gate / up `linear`
  (fp32 out) -> `multiply` SILU (fp32) -> down `linear` (bf16) -> `insert` -> `combine` (axis 0, init_zeros) ->
  `post_combine_reduce(expert_dim=3)` with the dispatch table (skips other groups' slots) -> fp32 -> TP
  `reduce_scatter` (sums the 4 groups) -> bf16 [1, 1, 160, 4096]. 65 ttnn ops.
- Plan risk "multi-group dispatch on FABRIC_1D_RING" resolved: dispatch / combine / offset_cumsum run as is on the 8x4
  1D ring with 4 groups (dispatch log: sparse_mcast enabled, max_hops 10).
- Dispatch buffer per chip: 5120 x min(top_k, 8 local experts) + 7 x 32 rows (worst case; the kernel drops overflow
  silently). `EXPERTS_BUFFER_FACTOR` lowers it (0 = worst case). 336 MB bf16 per chip, as plan.md budgeted.
- Weights: mxfp4 -> bf16 (exact) -> bfloat8_b in the checkpoint's [out, in] with `transpose_b`, one 2-D [N, K]
  tensor per chip and local expert, built as host shards in a 16-thread pool and placed with `ttnn.from_host_shards`.
  The first version (`from_torch(bfloat8_b, mesh_mapper=ShardTensor2dMesh)` on [8, 4, N, K]) converts single-threaded
  (~4 s per 32 matrices, 24 of them) and pushed the test past pytest's 300 s timeout.
- Settings: `EXPERTS_FIDELITY` HiFi4, `EXPERTS_FP32_ACC`, `EXPERTS_INTER_FP32`, `EXPERTS_OUT_FP32` (all in
  MAX_PRECISION), `EXPERTS_BUFFER_FACTOR` 0, `EXPERTS_NUM_LINKS` 1, `EXPERTS_TOPOLOGY` Linear.
- Gate: PCC 0.999992, vs CPU rel 0.0034 (limit 0.0103), worst row 0.0053, ratio [0.9990, 1.0032], vs golden rel
  0.0042, auto_margin_min 2.37, host FLOPs 0, deterministic. Test wall time 228 s of the 300 s pytest timeout (most of
  it the framework's CPU reference and precision model).
- Re-run: `bfw job check --task C.sliding_moe.experts --gate-only`.

## C.attn_moe.attention (layer 5): PASS

- Same `tt/attention.py` `Attention` (full path) as layer 0; no hook change. First gate: PCC 0.99999 but auto_margin_min
  0.835, output column 774 rel 0.067 vs limit 0.056 (o_proj column with sum |terms| / |sum| ~ 89).
- Diagnosis (device dumps vs fp32 CPU): device SDPA out 0.0118 rel from fp32; of that, the kernel itself 0.0092 and the
  inputs 0.0068. Column 774 from the inputs alone: 0.044. The chunk's own k / v in the cache had row norms +0.42 % /
  +0.27 %: the device `typecast` bf16 -> bfloat8_b is biased (+0.28 % norm on random-like k / v; `clone` / `add` with
  dtype bfloat8_b +0.5 %), while fp32 -> bfloat8_b (`typecast`) rounds like the host converter (ratio 1.0000).
  A bf16 -> fp32 -> bfloat8_b detour does not help (same result as the direct cast: the bias is in rounding bf16
  values). fp32 -> bf16 `typecast` is exact RNE.
- Changes (full layers only; sliding layers unchanged, their gate re-run: margin 1.006 as before):
  - `QKV_FP32`: qkv linear out fp32, `nlp_create_qkv_heads` and `rotary_embedding` on fp32, k / v cast fp32 ->
    bfloat8_b for the cache writes, q typecast to bf16 for the SDPA.
  - `FULL_SCALES_AFTER_PROJ`: 192^-0.5 and attention_value_scale multiply the fp32 q / v (fp32 [1,1,1,1] tensors)
    instead of being folded into the bfloat8_b weight rows (as the precision model scales after bfp8 weights). This
    was the bigger lever: with QKV_FP32 alone layer 5 reached 1.02 and layer 0 dropped to 0.97-1.0001.
  - `FULL_WEIGHTS_BF16_FIRST` "all" (none / qkv / o / all): dequantized qkv / o_proj rounded to bf16 before bfloat8_b.
- Gates with all three (k chunk 320): layer 5 margin 1.222 (rel vs CPU 0.0044 / 0.0079, worst column 1005 0.033 vs
  0.040); layer 0 margin 1.114 (was 1.003). BF16_FIRST none: 1.13 / 1.046. Before the scale change, SDPA_K_CHUNK 640
  gave layer 5 1.049 but layer 0 0.868; 160 gave layer 5 0.935: kept 320.
- Checked on the device: the ring SDPA's gather buffer [1, 1, 56320, 192] holds SP row d's slot-0 cache at rows
  d*7040.. (own row block left 0), i.e. the prefix layout is right; the remaining error is the streaming kernel's bf16
  scores / running sums (fp32 dest not allowed on the kv_actual_isl path).
- Re-run: `bfw job check --task C.attn_moe.attention --gate-only`.

## C.attn_moe.router (layer 5): PASS

- Same `tt/router.py` `Router` and hook `_router_component` as layer 1. The first gate failed only on the `mixed` probe:
  1 row selected 9 experts. The `choice >= kth` mask selects every entry equal to the 8th value, so an exact fp32 tie
  there adds an expert. (The breadcrumb above warned about this; layer 5 hits it.)
- Fix: new setting `ROUTER_MASK` "scatter" (default) or "kth". "scatter" builds the mask as `ttnn.scatter(zeros bf16
  [R, 256], -1, topk idx, ones bf16 [R, 8])` -> typecast fp32, so every row has exactly 8. bf16 ROW/TILE scatter works;
  fp32 TILE scatter is TT_FATAL. The zeros and ones are load-time constants (max_rows 640), sliced only when R < 640.
  Do not deallocate a full-size slice: `ttnn.slice` with the full extent returns the constant itself (the first try
  freed it, and the next call failed with `input_tensor.is_allocated()`).
- Gate: PCC 0.999157, overlap vs CPU / golden 0.99856, matched-row rel vs CPU 1.5e-4, mixed overlap 0.99985 (no count
  mismatch), auto_margin_min 3.16, 10 ops, host FLOPs 0. C.sliding_moe.router re-gated with the scatter mask: PASS,
  margin 3.06 (unchanged).
- Re-run: `bfw job check --task C.attn_moe.router --gate-only`.

## S.sliding_moe.02 (layer 1, attn_norm + attention on device): PASS

- First gate: block PCC 0.999995, but the attention step vs the CPU step on the same inputs had rel 0.0141 (limit
  0.0138), row-norm ratio min 0.970 and median bias -1.33 % (limit 1 %): margin 0.665. This is the -1.3 % bias the
  component breadcrumb left unexplained; the swap test gates bias / ratio tighter than the component test.
- Cause: the sink holds nearly all the softmax mass on sliding layers, so the output scales as exp(q.k - sink) and
  any relative error in q / k turns into a norm bias. Three sources, fixed one at a time (gate after each):
  - bf16 q / k / v out of the qkv linear and the biased device bf16 -> bfloat8_b cache cast. `SWA_QKV_FP32`: the
    linear out, head split, RoPE and value scale in fp32 (as the full layers' `QKV_FP32`), k / v cast fp32 ->
    bfloat8_b for the caches, and the composed softmax reads fp32 q and fp32 k / v (halo rows from the cache are
    typecast to fp32; in-chunk halo rows come from the fp32 all-gather). Bias flipped to +1.1 %, ratio max 1.028.
  - `ttnn.matmul` with fp32 operands is not fp32-exact (reduced input precision). `SWA_SPLIT_MM` (`_mm32`): every
    matmul with an fp32 operand (scores, p @ v, the one-hot halo select) runs as hi*hi + hi*lo + lo*hi with bf16
    hi / lo splits (RNE typecasts), fp32 dest. Bias +0.43 %, ratio max 1.0202: margin 0.99.
  - The fused `rotary_embedding` on fp32 input with bf16 cos / sin (and fp32 cos / sin tables: `ttnn.embedding`
    takes bf16 tables only, TT_FATAL). `SWA_ROPE_FP32`: cos / sin looked up from a bf16 table plus a bf16 residual
    table and added in fp32; RoPE composed as t * cos + rotate_half(t) * sin with fp32 eltwise ops (`_rope32`).
    Fp32 tables through the fused op alone made it worse (bias +1.9 %).
- Gate: block PCC 0.999995; attention vs CPU rel 0.00265 (limit 0.0138), worst row 0.0046, ratio [0.9960, 1.0023],
  bias -0.09 %; swap_margin_min 2.73; 173 device ops (was 88), host FLOPs 0.
- C.sliding_moe.attention re-gated with the change: PASS, auto_margin_min 1.053 (was 1.006), vs CPU rel 0.00265,
  worst column 1160 0.100 vs 0.106. Full layers unchanged (all new switches are sliding-only).
- Cost: 3 matmuls per fp32 matmul and the composed RoPE; P.* can fuse (an fp32-score sink SDPA fork would remove
  the composed softmax and the discarded ring SDPA output).
- Re-run: `bfw job check --task S.sliding_moe.02 --gate-only`.

## M.1 (assemble, layers 0-5): device gates PASS, job check blocked by a stale freeze

- `tt/model.py` `MimoModel` (from `hooks.device_model`, cached per (mesh, layers, lm_head) in the process;
  `new_state` zeroes the caches with DRAMZeroFill): embed -> 6 blocks -> final norm on the device. Each block runs
  `run_block` over the reference's own graph (`reference.model.Layer.graph`) with the validated component modules
  (RMSNorm, Attention, ResidualAdd, DenseMLP | Router + Experts); the intermediate boundaries are freed after the block.
  It runs in the slow runtime mode so the ladder's device proof counts the ops.
- Embed: tokens in natural order, padded with PAD_ID, are permuted on the host into the server's reshuffled `[8, 1, 640]`
  uint32 (same as `server_rules.server_payload`, checked for starts 0 / 1312 / 2944 / 51200). That is the runner's H2D
  input, built at the harness boundary. On the device: `minimum(ids, V-1)` -> `embedding` (bf16 table replicated) ->
  `mesh_partition(dim 2, TP)`. `embed` also stages the chunk metadata (`ChunkMeta.stage`, one for the full layers and
  one for the sliding ones, shared by every layer of that kind) with end = start + the number of real tokens, so
  `layer` does no host work (host_transfers_per_layer 0). `from_host` / `to_host` / `logits` apply the same row
  permutation at the boundary.
- State = the served caches (`LayerKV` of each attention layer); `to_torch` reads each chip's shard
  (`get_device_tensors`) and inverts the block-cyclic layout; `load_prefix` is `Attention.load_prefix` for slot 0.
- lm_head: [V/4, H] bfloat8_b per TP column, HiFi4 fp32. It is not used at layers 0-5 (lm_head=False there).
- Device runs (direct safe runner): ladder s10240 PASS, worst layer PCC 0.999957, worst state PCC 0.999843, 41+ ops
  per layer, 0 host transfers, warm chunk 3.14 s. test_kv_layout PASS (6 layers, 0 problems). test_cache_writes PASS,
  lowest layer 5 value 0.9758 (pulled_back) / 0.9784 (follow_up_tile) against 0.97: thin, and it falls with depth
  (layer 0 0.99992).
- `bfw job check --task M.1 --gate-only`: FAIL in the precheck, before any test ran. M.1's frozen hashes of
  bringup/serving_contract.md and contract_tests.yaml are the first SC.1 version (taken at 13:45), not the
  re-pinned one approved at 14:36. A person must refreeze M.1 (and K.1).
- Run 20261025-055712 attempt 1: the freeze precheck now passes (the gate command ran). The first gate was started
  as a background task and was killed when the session ended, so the board was marked dirty. The foreground re-run then
  stopped at `SAFE_PYTEST: Device marked dirty from previous hang` (reset refused in a stage). No code change; the
  next attempt runs the same gate on a reset board.
- Run 20261025-055712, stage script (before that agent ran): `bfw job check` PASS (ladder worst layer PCC 0.999957,
  state 0.999843, 41 ops per layer, 0 host transfers), then SCOPE "protected tasks changed during the job" (render-tests
  had re-frozen M.1 during the job).
- Run 20261025-064412 attempt 1: no code change. `bfw job check --task M.1 --gate-only` FAILs in the precheck again:
  a binding mismatch, plus stale hashes for the two SC.1 files (tasks.yaml M.1 `frozen.at` 2026-10-24T13:45:19).
  The gate's three commands, run directly through the safe runner on a fresh board, all PASS: ladder s10240 0.999957 /
  state 0.999843, test_kv_layout, and test_cache_writes (lowest: layer 5 value 0.9758 pulled_back, 0.9784
  follow_up_tile). M.1 (and K.1) must be refrozen by render-tests.
- Re-run: `bfw job check --task M.1 --gate-only`.

## L.s56320 (ladder, all 11 chunks of 56320): SDPA_SEGMENTED on

- First gate: every PCC / rel / state check passed, but `bias_layer_L*` 0.0055-0.0073 > 0.005 on all six layers,
  already at layer 0 (s10240: 0.0008). The CPU reference on the golden's layer-0 input and bfp8-rounded prefix gives
  bias 1e-4, so the drift is the device's.
- Per-step diagnosis (scratch test: layer 0, chunk 51200 -> 56320, golden prefix loaded, device boundaries vs the CPU
  step outputs): attn_out median row norm +2.0 % (rel 0.032), every other step clean. The full-attention ring SDPA with
  `segmented_accumulation=False` carries its bf16 running output / sum across the whole 51k prefix (bf16 dest is
  forced by the kv_actual_isl streaming path), and the rescale roundings bias the norm up with prefix length.
- Fix: `SDPA_SEGMENTED` default True (per-ring-iteration accumulators). Layer 0 attn_out -0.17 %, block out +0.05 %.
  Ladder s56320 (direct): worst layer PCC 0.999858, state 0.999765, every chunk's KV >= 0.99974. Re-gated with it:
  C.attn_dense.attention PASS margin 1.14, C.attn_moe.attention PASS margin 1.22 (the older 0.989 margin with it on
  predates exact_exp). Sliding layers do not use the ring SDPA output, so they are unchanged.
- Re-run: `bfw job check --task L.s56320 --gate-only`.
- Gate status (attempt 1): not yet confirmed through `bfw job check`. Run 1 failed at mesh open (`Query mappings failed
  on device 0: No such device`, boards re-enumerated 08:28); run 2 was killed by session teardown mid-run; run 3 refused
  (device marked dirty, reset is the worker's). The next attempt only needs to run the gate on a reset board.
- Attempt 4 (debugger, 08:40 UTC, bh-glx-120-b05u02): the gate failed the same way again. Root cause is the host:
  only 31 of 32 Blackhole chips enumerate (`/dev/tenstorrent/30` and PCI `0000:87:00.0` absent from sysfs), so UMD
  topology discovery throws `Query mappings failed on device 0: No such device` before the model runs. The code
  (SDPA_SEGMENTED on) is unchanged and still needs its first gate run on a host with all 32 chips.

## K.1 (contract): prefill adapter, served KV cache, address table, acks, producer read-back

- `tt/runners/adapter.py` (`MimoPrefillAdapter`, ADAPTER_PATHS `mimo_v26_flash_rl`): light import, checkpoint dir from
  the `SERVE_HF_DIR` setting, eager single rank (SP 8 x TP 4, chunk 5120), `supports_mtp = False`.
- `tt/runners/kv_caches.py`: `allocate_kv_cache` returns `MimoKvCaches` = {layer: `LayerKV`} (tt/kv_cache.py, unchanged
  layout). `MimoModel(kv_caches=...)` / `Attention(kv=...)` now take the engine's caches instead of allocating their own,
  so the served cache is the model's state. Capacity is `max(max_seq, 2 * chunk)`: the sliding ring SDPA fork
  TT_FATALs "requires chunked prefill" when the cache holds only one chunk (the s5120 rung), as gemma4's
  `ring_cache_capacity`.
- `tt/runners/kv_chunk_table.py`: configs `key_h0-7` (6528 B) / `value_h0-7` (4352 B); heads 4-7 only on sliding layers
  (other layers unset). Address `(shard % banks) << 32 | base + shard // banks * record_bytes` over the ROUND_ROBIN_1D
  walk of [slots, local_heads, local_seq, width]; device group = (SP row of the position, TP column of the head).
  Same file: `read_slot_kv_and_check_pcc` (the producer's read-back through the table vs the golden kv_cache).
- `tt/runners/runtime.py`: per chunk stages the metadata with the slot (`MimoModel._stage(start, end, slot)`), embeds the
  engine's H2D tokens on the device, runs each layer and acks it right after (D2H `outbound_socket_service_sync` on the
  same CQ, or host sink after a sync). The table / stages may be built before `compile`.
- Generic hook in tt-metal: `PrefillModelAdapter.read_slot_kv_and_check_pcc` (default None); `prefill_producer` calls
  it before its name branches.
- Results (job-attempt-3): framework contract with a corrected `device_map` (scratch copy): PASS, layout_failures 0,
  acks_early 0, producer KV PCC key 0.99991 / value 0.99971 (slot 1, [0, 5088), layers 0-5).
  Blocked: (1) the framework's `contract.device_map` calls `ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_id,
  chip_id)`; the workspace binding takes `(mesh_device, mesh_id, chip_id)` -> TypeError (bringup_fw, not editable here).
  (2) test_runner_contract (tt-d-gen engine feeder @ ff5d9d92): the runner fails building `ttnn.H2DStreamService`:
  "Failed to pin pages for DMA buffer ... Invalid argument"; chip 15's IOMMU group is `identity` (31 others DMA-FQ),
  UMD logs "IOMMU state mismatch for chip 15: expected enabled, got disabled" on bh-glx-120-b07u08.
- Re-run: `bfw job check --task K.1 --gate-only`.
