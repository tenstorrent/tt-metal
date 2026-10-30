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
