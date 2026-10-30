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
