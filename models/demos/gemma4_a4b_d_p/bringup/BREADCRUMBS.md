# google/gemma-4-26B-A4B-it bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1)
- Wrote `reference/gemma4_ref.py` (standalone, safetensors direct, fp32, all layers resident ~104 GB) and the
  `reference` + `hf_layers` hooks. Attempt 0 failed only because `hf_layers` was missing (HF oracle is
  Gemma4ForConditionalGeneration; layers at `model.model.language_model.layers`).
- Block graph (same for sliding and global; attention type chosen in `chunk_context` / `_attention`), 14 steps:
  attn_norm, attention [stateful], post_attn_norm, attn_residual -> h_mid, ffn_norm, mlp, post_mlp_norm,
  router (on h_mid), moe_norm, experts(moe_norm, router), post_moe_norm, ffn_combine, post_ffn_norm,
  ffn_residual (`(h_mid + ffn_out) * layer_scalar`) -> out.
- Decision: the `router` boundary is a dense [S, 128] routing matrix (top-8 weights after renorm and
  per_expert_scale, zeros elsewhere), so one tensor carries both selection and weights; `experts` groups tokens
  per expert in ascending expert id (HF accumulation order).
- State: key/value [Hkv, max_seq, D] per layer; K post k_norm + RoPE, V = unscaled RMS(raw proj). On global layers
  V comes from k_proj (no v_proj) but is NOT equal to K.
- Attention: scale 1.0; sliding window keys (q-1024, q]; per 1024-query block the key range is bounded by the
  window, attend against the state (no prefix recompute). Global RoPE: proportional, 64 of 256 freqs non-zero.
- Logits: tied embedding + softcap 30*tanh(x/30). Embed scale sqrt(2816) in fp32.
- Gate: all 30 layers pcc=1.0000000 (maxabs <= 1.3e-3), logits pcc=1.0000000, top1_match=1.0000, ~1 min.
- Sanity (not a gate): 3000 tokens in 1000-token chunks vs one-shot on layers 0,4,5: state maxabs <= 8e-6; graph
  replay of layers 0 and 5 from recorded boundaries exact (0.0).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 512`

## R.3 reference (attempt 1)
- Attempt 0 failed on chunked vs one-shot (hidden pcc 0.99983, state 0.99986); graph replay was already exact.
- Diagnosis: comparing every boundary of chunk 2 (tokens 2048-4095) between one-shot and 2x2048, everything was
  bit-exact until `L0.experts_out`. After that, near-tie top-8 flips (1 token at L14, 31 at L29) amplified the difference.
  Cause: MKL sgemm output for a row depends on M when M is small or odd, and the per-expert group size differs
  between chunked and one-shot.
- Fix: `experts_forward` zero-pads each expert's token group to a multiple of `EXPERT_ROW_BLOCK = 32` rows and
  slices the pad off. The math is unchanged. Chunked == one-shot now bit-exact (final_norm and all K/V maxabs 0.0).
- Gate: hidden pcc=1.000000000, state min pcc=1.000000000, sliding/global graph replay max abs 0.0, 0 missing.
- check_hf was not re-run (padding rows are discarded, so the math is the same as R.2).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_reference --seq 4096 --chunk 2048`
