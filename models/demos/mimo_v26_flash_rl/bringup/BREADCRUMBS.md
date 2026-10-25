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
