# ACE-Step v1.5 bring-up — ideas backlog

- Check if TTTv2 Attention1D already supports per-head q/k RMSNorm (Qwen3 style). If not,
  smallest extension is to wrap q_proj/k_proj outputs with RMSNorm1D before RoPE.
- AdaLN modulation: express `norm(x)*(1+scale)+shift` with ttnn.rms_norm + ttnn.mul/add;
  gate residual with ttnn.mul then ttnn.add. Consider a small AdaLNModulation helper in tt/.
- TimestepEmbedding sinusoidal table can be precomputed on host (LazyWeight) — only the two
  Linears + time_proj run on device.
- Cross-attention: reuse Attention1D but feed encoder_hidden_states as kv; skip RoPE. Check if
  Attention1DConfig exposes a kv-source / no-rope switch before writing a custom class.
- Sliding window (128): confirm ttnn SDPA / attention supports windowed masks on Blackhole;
  else build additive 4D mask like create_4d_mask (host) and pass as attn mask.
- FSQ (ResidualFSQ) + 1D VAE audio decoder are the biggest from-scratch pieces — check
  models/tt_dit/models/audio_vae and models/tt_dit/layers/audio_ops.py for reusable ops FIRST.
- Consider models/tt_dit/layers/normalization.py for AdaLN — DiT models there likely already
  have adaptive-norm helpers (SD3.5 / Flux use AdaLN). Reuse before writing.

## CONFIRMED reusable (found in repo)
- tt_dit AdaLN/modulation lives in: models/tt_dit/layers/embeddings.py,
  models/tt_dit/blocks/transformer_block.py, models/tt_dit/blocks/attention.py.
  READ THESE before writing the AceStepDiTLayer AdaLN — SD3.5/Flux DiT blocks already do
  scale_shift_table + temb chunk(6) gated-residual modulation on TTNN. Likely near drop-in.

## CORRECTNESS FINDING (Module: attention_pooler, iter 17)
- ttnn SDPA with an ALL-ZERO additive attn_mask is NOT equivalent to mask=None: SDPA
  tile-pads the mask with zeros, so queries attend to padding key positions -> PCC drops
  (~0.94 at seq=6). When sliding_window >= seq_len the reference mask is all-visible, so the
  correct TT input is None. Callers (pooler done; CHECK lyric_encoder + dit_stack) must pass
  sliding_mask=None when window>=seq. Fixed in test by `need_mask = seq > SLIDING_WINDOW`.
- Lyric/DiT-stack tests currently only use seq>=256 (>window 128) so they build a real mask
  and are unaffected, but a production wrapper feeding short seqs must apply the same guard.
  Consider baking the guard into a shared pipeline helper (build_sliding_mask returns None when
  window>=seq) so no caller forgets it.
