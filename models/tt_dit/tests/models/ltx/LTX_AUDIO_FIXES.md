# LTX AV audio fixes (distilled fast pipeline)

## Production fixes (kept in tree)

| Area | Change |
|------|--------|
| RoPE | `pipeline_ltx.py`: keep **fp32** audio/video positions before `precompute_freqs_cis` (no `.bfloat16()` on positions). |
| GELU | `layers/linear.py`: `gelu_tanh` uses `x*x*x`, not `ttnn.pow(x, 3)` (NaN on negative inputs). |
| T5 RMSNorm | `encoders/t5/model_t5.py`: variance via `x*x`, not `pow`. |
| Matmul | `utils/matmul.py`: clamp fallback block sizes to tile counts. |
| SP readback | `device_to_host(..., sp_already_gathered=True)` after `inner_step` (no double SP gather). |
| A↔V norms | `ltx_transformer.py`: cross-attn uses `norm3` / `audio_norm3` (HF), not `norm1`. |
| Residuals | Gated text-CA and A↔V use `ttnn.addcmul` instead of `x + h*gate`. |
| Gates | `attention_ltx.py`: per-head gate computed on **full SP** sequence on host. |
| SDPA | `attention_ltx.py`: cross-attn SDPA at HiFi4. |
| Masks | `pipeline_ltx.py`: audio attn mask column-only (avoid padded-row NaN). |
| WAV | `pipeline_ltx_fast.py`: stereo dump as `(T, 2)`, not mono average. |

## Tests (`models/tt_dit/tests/models/ltx/`)

- `test_rope_position_precision.py` — bf16 positions vs fp32 on audio RoPE.
- `test_sp_gather_token_order.py` — SP gather order / double-gather guard.
- `test_sp_gather_audio_velocity_shape.py` — SP shard layout at audio velocity shape.
- `test_pipeline_ltx_fast_av_stage1.py` — stage-1 listen path (~4 min).

## Removed (diagnostic-only)

- `models/tt_dit/reference/*` bisect/capture scripts (except existing `reference/motif/`).
- `scripts/diag_*.py`, HF inject hooks, per-block PCC regression harness, 563-line investigation log.
