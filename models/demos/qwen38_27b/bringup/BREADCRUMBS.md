# qwen38_27b bring-up breadcrumbs

## Reference (R.1-R.3), 2026-10-30

- `reference/qwen38_ref.py`: standalone CPU reference of Qwen3.8-27B (text model), written from transformers 5.12
  `models/qwen3_5/modeling_qwen3_5.py`, the code that made the isl56320 trace. `bringup/reference_hooks.py`:
  `reference` (checkpoint dir = `spec.hf_dir`) and `hf_layers` (`model.model.language_model.layers`).
- One block graph for both block types: `attn_norm -> attention (stateful) -> attn_residual -> ffn_norm -> mlp ->
  mlp_residual`. `attention` is the Gated DeltaNet on linear layers and gated GQA attention on attn layers.
- State: attn `key` / `value` `[4, seq, 256]` (key after k_norm + partial RoPE, as HF caches it); linear `conv`
  `[10240, 4]` (the last 4 in_proj_qkv rows before the conv, oldest first: HF `conv_states` without the batch dim) and
  `recurrent` `[48, 128, 128]` fp32 (HF `recurrent_states` without the batch dim). conv / recurrent are fixed-size.
- Decisions: weights are held in bf16 and cast per op. The text-only interleaved mRoPE reduces to plain partial RoPE
  (64 of 256 dims, rotate-half, theta 1e7). The delta rule is upstream's chunk-64 algorithm with the state carried
  between calls. Every chunk boundary is a multiple of 64, so chunked prefill is bit-exact with one-shot prefill.
  Full attention runs in 1024-row query blocks aligned to absolute positions; a 56k chunk peaks at about 13 GB
  instead of 63 GB.
- Results: R.2 vs trace PCC >= 0.999988, rel <= 4.8e-3 at L0-L3. R.3 chunked == one-shot (PCC 1.0, maxabs 0) and
  graph replay maxabs 0. R.3 takes about 72 s on 32 threads.
- Re-run: `bfw tasks run R.1 R.2 R.3`.

## C.linear.attn_norm (device input_layernorm, layer 0), 2026-10-30

- `tt/norm.py` `TtRMSNorm`: stock `ttnn.rms_norm` on each chip's `[1, 1, 640, 5120]` rows, with the weight (1 + w)
  folded once at load into a replicated fp32 TILE gamma `[1, 1, 1, 5120]`; eps from config (1e-6); no collective.
  Switches in `tt/settings.py`: `NORM_FIDELITY` (HiFi4), `NORM_FP32_ACC` (True), `NORM_WEIGHT_DTYPE` (float32).
- `bringup/hooks.py` `device_component` serves `attn_norm` and `ffn_norm` (same module, other weight). Host I/O at the
  component boundary only: rows split over mesh axis 0 (640 per row, chunk order), replicated over axis 1; read
  back rows from axis 0 and TP column 0.
- Decisions: the gamma is TILE, not ROW_MAJOR (as `models/common/rmsnorm.py` does). A ROW_MAJOR gamma disables the op's
  streaming large-tensor kernel, and the whole-row kernel needs 1.85 MB of L1 at width 5120 (TT_THROW). The
  component call turns off `ttnn.CONFIG.enable_fast_runtime_mode` for its duration (`_hooked_dispatch`), because
  FastOperation skips the pre-operation hooks the device proof counts ops with (device_ops was 0).
- Result: PCC 0.999998, rel vs CPU 0.0020 (limit 0.0064), auto_margin_min 3.2, deterministic.
- Re-run: `bfw job check --task C.linear.attn_norm --gate-only`.

## C.linear.attention (device Gated DeltaNet, layer 0), 2026-10-30

- `tt/gdn.py` `TtGatedDeltaNet`, stock TTNN only, on the plan's layout. TP column c holds key heads 4c..4c+3 and value
  heads 12c..12c+11. Each chip projects its own 640 rows: [qkv 2560 | z 1536] in one matmul, a / b in another (both
  fp32 out); g and beta are computed in fp32 32 wide (12 heads plus zero padding). `all_gather(axis 0)` of qkv / g /
  beta, then a reorder from chip order to natural order by `chip_order_runs(start)` (slice + concat; the identity at
  chunk-aligned starts). `qkv_causal_conv1d_silu` runs on the whole chunk, with the conv state as history
  (actual_start 0, predecessor_carry = history). Then `chunk_gated_delta_rule` -> back to chip order ->
  `mesh_partition(axis 0)` -> per-head `rms_norm` (plain w) -> `nlp_concat_heads` -> * silu(z) fp32 -> out_proj
  (fp32 partials) -> `all_reduce(axis 1)` -> bf16. The 8 SP rows of a column run the conv and the recurrence
  redundantly (V-split over rows is the perf follow-up).
- State layout (`state_to_device`): conv history `[1, 3, 2560]` bf16 ROW_MAJOR (the reference's conv[:, 1:]
  transposed, the column's channels) and recurrent `[1, 12, 128, 128]` fp32 TILE per TP column, replicated over SP
  rows. `__call__` returns the new state (the last 3 conv inputs, the final recurrent state). The component hook
  uploads the golden prefix state (zeros at start 0) at the boundary.
- Decision 1: weights stay `[out, in]` and run with `ttnn.linear(..., transpose_b=True)`. The bfp8 exponent blocks
  then run along the reduction dim. With pre-transposed `[in, out]` weights, out_proj output column 3993 had rel
  0.50 (limit 0.096): a small output channel shared its exponent with 15 larger ones.
- Decision 2: `GDN_CHUNK` = 64 (rank-4 path; q / k l2-normalised in fp32 by the model) instead of the rank-3 flat
  path at chunk 32. Both are equally exact on a zero state. From a non-zero state, the op's error grows with the
  number of inter-chunk state updates. Measured with the golden state at 5120 and layer-3 inputs: o rel 0.060 / state
  rel 0.139 at chunk 32 (fused, phased, HORNER and FORWARD_SUBSTITUTION all the same) vs 0.028 / 0.066 at chunk 64.
  At chunk 32 the gate's `layer3` probe failed (rel 0.0409 > 0.0388).
- Result: PCC 0.999994, rel vs CPU 0.0032 (limit 0.015), auto_margin_min 1.32 (closest check: the layer3 probe, rel
  0.0293 vs 0.0388), 46 device ops, host FLOPs 0, deterministic.
- Not done yet (later tasks): pad masking past actual_end (g = beta = 0), the restart snapshot split, and the
  served state buffers.
- Re-run: `bfw job check --task C.linear.attention --gate-only`.

## C.linear.attn_residual (device residual add, layer 0), 2026-10-30

- `tt/residual.py` `TtResidualAdd`: one `ttnn.add` (bf16, DRAM) of two tensors in the residual layout
  `[1, 1, 640, 5120]` per chip (SP rows over axis 0, replicated over TP columns); no collective.
- `device_component` serves both `attn_residual` and `mlp_residual` (`RESIDUAL_STEPS`) with the same module, for
  both block types.
- Result: PCC 0.999997, auto_margin_min 1.97, 1 device op, host FLOPs 0.
- Re-run: `bfw job check --task C.linear.attn_residual --gate-only`.

## C.linear.mlp (device SwiGLU MLP, layer 0), 2026-10-30

- `tt/mlp.py` `TtMLP`, stock TTNN only, on the plan's Megatron split. TP column c holds the gate and up rows
  [4352 c, 4352 (c + 1)) as one `[8704, 5120]` bfp8 weight (gate then up), plus the down_proj inputs
  [4352 c, 4352 (c + 1)) as `[5120, 4352]` bfp8. Each chip runs on its 640 rows: one `ttnn.linear` (fp32 out), two
  slices, `silu(gate) * up` to bf16, down_proj (fp32 partials), `all_reduce(cluster_axis=1)`, then typecast to bf16.
  That is 8 device ops.
- Weights stay `[out, in]` and run with `transpose_b=True`, as in gdn.py, so the bfp8 exponent blocks run along the
  reduction dim. Switches: `MLP_FIDELITY` (HiFi4) and `MLP_FP32_ACC` (True) in `tt/settings.py`.
- `device_component` serves `mlp` for any layer. The module has nothing block-specific, so the attn block's mlp can
  use the same branch.
- Result: PCC 0.999984, rel vs CPU 0.0054 (limit 0.0113), auto_margin_min 2.11, host FLOPs 0, deterministic.
- The first gate run timed out at pytest's 300 s while it was still recording passing metrics. The JSON writes on
  NFS were about 10 s apart. The re-run took 137 s. The test body's run time varies with host and NFS load.
- Re-run: `bfw job check --task C.linear.mlp --gate-only`.

## C.attn.attention (device gated GQA attention, layer 3), 2026-10-31

- `tt/attention.py` `TtGatedAttention`, stock TTNN only. TP column c holds q heads 6c..6c+5 and KV head c. Each chip
  works on its own 640 rows. One fp32 matmul gives `[q 1536 | k 256 | v 256 | gate 1536]`, then
  `nlp_create_qkv_heads` (transpose_k_heads False). q_norm / k_norm are `rms_norm` with an fp32 (1 + w) gamma. Partial
  RoPE (dims 0-63) is rotate-half by slice / neg / concat in fp32. Its cos / sin rows are sliced from load-time
  tables in cache-local block-cyclic row order (`rope_offsets(start)` = the writer's update_idxt per chip; a mask
  sum over the SP row id when the chips differ, at a start inside a chunk). K / V are cast to bfp8 and written by
  `update_padded_kv_cache(kv_actual_global=start, cluster_axis=0)`. Then comes
  `ring_joint_scaled_dot_product_attention` over the cache (axis 0, Linear, 2 links, `kv_actual_isl=start`,
  `logical_n=start+T`, scale 1/16, causal, unbalanced), then `nlp_concat_heads`, `* sigmoid(gate)` in fp32, o_proj
  (fp32 partials), `all_reduce(axis 1)` and bf16. That is 43 device ops.
- `tt/kv_cache.py` `TtKVCache`: the served layout. K / V are `[slots * layers, 1, max_seq / 8, 256]` per chip, bfp8
  TILE, ND-sharded `[1, 1, 32, 256]` ROUND_ROBIN_1D over the DRAM banks, KV head h on TP column h. `load_prefix` is
  the harness-boundary upload of the golden prefix (whole cache, zeros past `start`); `cache_tensors()` returns it.
  The component hook builds one cache with 1 slot and 1 layer.
- SDPA: `kv_actual_isl` needs the streaming kernel, so SDPA fp32 dest acc is off (rule-7 exception, `SDPA_FIDELITY`
  HiFi4). q128 / k128 with `segmented_accumulation=True` (`SDPA_SEGMENTED`). k256 + segmented overflows the kernel
  config buffer by 112 B (TT_FATAL program.cpp:3760).
- Decision: `ATTN_QKV_HILO`. The q/k/v/gate weight runs as two bfp8 matmuls, bfp8(W) plus the bfp8 residual
  W - bfp8(W), with the residual taken from ttnn's own rounding. With one bfp8 weight the gate failed only the row
  norm ratio vs the golden (min 0.9805, limit 1 +- 0.0176; auto_margin_min 0.81-0.90 over k128 / k256 / segmented).
  Cause, from a CPU model on the golden: layer 3's attn_norm has an outlier channel (|x| up to 92, median channel max
  1.9). bfp8 q_proj / v_proj alone move the median row norm by -0.28 % / -0.23 % (all weights -0.54 %). Blocks along
  the output dim, weight-column power-of-2 smoothing and per-row least-squares scales each recover less than half
  of it; hi + lo recovers it (model: median -0.04 %).
- Result: PCC 0.999978, rel vs CPU 0.0050 (limit 0.0142), vs golden 0.0066, row ratio [0.9863, 1.0143],
  auto_margin_min 1.23, 43 device ops, host FLOPs 0, deterministic.
- Not done yet (later tasks): pad rows past actual_end (`valid_global` + `zero_padded_kv_cache`), the
  trace-safe tensor forms of slot / start, and a multi-slot / multi-layer cache from the model.
- Re-run: `bfw job check --task C.attn.attention --gate-only`.
