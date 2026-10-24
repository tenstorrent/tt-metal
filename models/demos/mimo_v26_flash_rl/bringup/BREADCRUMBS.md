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
