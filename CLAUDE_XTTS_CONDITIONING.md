# XTTS-v2 Conditioning Encoder + Perceiver Resampler — TTNN bringup (Block 1)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: **DONE on TT** — full Block 1 (mel → gpt_cond_latent) at **PCC 0.99999** vs coqui.
  Conditioning encoder 0.99998, Perceiver 0.99999, end-to-end 0.99999 (fp32).
- Owner: acicovic
- Started: 2026-07-20

## Files
- `reference/xtts_cond_ref.py` — CPU torch mirror (`CondReference`), PCC 1.0 vs coqui.
- `tt/ttnn_xtts_cond.py` — `TTNNConditioningEncoder` (Conv1d + 6× masked-GroupNorm attention)
  + `TTNNPerceiver` (+ `preprocess_encoder_parameters` / `preprocess_perceiver_parameters`).
- `tests/test_cond_encoder_pcc.py` (encoder), `test_cond_perceiver_pcc.py` (perceiver),
  `test_cond_pcc.py` (full block). Goldens in `golden/cond/` (gitignored).

## Key implementation notes
- **T=505 tile handling:** pad the frame axis to 512 (tile mult); mask padded positions
  with −∞ key masks in attention, so concat/attention stay tile-aligned.
- **Masked GroupNorm(32,1024) over time (solved BUG-class from FORGE):** after the 1×1 conv
  *with bias*, padded time rows become the bias (not zero), so they'd skew group stats.
  Fix: multiply by a time mask before summing, compute per-group μ/σ² manually with
  `ttnn.sum` over the logical T (`sum_t x`, `sum_t x²` → reshape `[.,1024]→[.,32,32]` →
  sum in-group channels → broadcast back → per-channel affine). Avoids native
  `ttnn.group_norm`'s tile-aligned-`H*W` limitation entirely.
- 16-head encoder attention: qkv channels are head-interleaved `[q64,k64,v64]` per head
  (coqui `QKVAttentionLegacy`) — reshape `[1,S,16,192]` then slice, not contiguous q|k|v.
- Perceiver RMSNorm == `ttnn.rms_norm(x, weight=gamma)` (`F.normalize·√1024·gamma`).

## Progress (2026-07-20)
- **Architecture fully mapped** from coqui source (see reference file for the op-by-op mirror):
  - Conditioning encoder = `Conv1d(80→1024,k1)` + 6× AttentionBlock. Each block:
    `x_norm = GroupNorm(32,1024, fp32)`; `qkv = Conv1d(1024→3072,k1)(x_norm)`;
    16-head QKV attention (scale `1/√√64`, non-causal over T); `proj = Conv1d(1024→1024,k1)`;
    **residual on the normed input**: `out = x_norm + proj`.
  - Perceiver (dim 1024, depth 2, 32 latents, 8 heads×64): per layer cross-attn
    (`context = concat([latents, frames])`, `cross_attn_include_queries=True`, scale 1/8) +
    residual, GEGLU FFN (1024→5460→GEGLU→2730→1024) + residual; final RMSNorm
    (`F.normalize(x,-1)*√1024*gamma`).
- **CPU reference** `reference/xtts_cond_ref.py` — op-for-op mirror; matches coqui golden at
  **PCC 1.0** (both enc_out and perc_out).
- **Goldens** captured from coqui: `golden/cond/{mel_in[1,80,505], enc_out[1,1024,505],
  perc_out[1,32,1024], gpt_cond_latent[1,32,1024]}` (*.pt gitignored; regenerate via the
  coqui capture — see below).
- **TTNN Perceiver** `tt/ttnn_xtts_cond.py::TTNNPerceiver` — **PCC 0.99999** vs golden
  (`tests/test_cond_perceiver_pcc.py`, fp32). Padding trick: frames T=505 padded to 512
  (tile mult) for concat/attention; padded key positions masked to −∞ in the softmax.

## Remaining: conditioning encoder (the GroupNorm-over-T problem)
- The 1×1 convs are just linears; the 16-head attention over T is standard (pad T→512 +
  mask, like the Perceiver).
- **GroupNorm(32,1024) pools over (32 channels/group × T)** — it normalizes across the time
  axis, so zero-padding T corrupts the mean/var (this is the FORGE `group_norm got 505`
  issue). Plan: compute stats manually with `ttnn.sum` over the *logical* T (padding masked)
  — per-group `μ,σ²` via sum + sum-of-squares, reshape `[.,1024]→[.,32,32]` and reduce the
  in-group channel dim, then broadcast back and apply per-channel affine. Avoids native
  `ttnn.group_norm`'s tile-aligned-`H*W` requirement. (Alt: native group_norm with a
  spatial mask if one can be supplied.)
- Golden regen (coqui venv): `/home/acicovic/.claude/jobs/.../tmp/gen_cond_golden.py`.

## Role in pipeline
**Branch A** of conditioning. Consumes the reference clip (22050 Hz) and produces the
32×1024 latents that **prefix the GPT** (Block 3). In coqui code this is `gpt.get_style_emb`
(conditioning encoder → Perceiver resampler tail). Do NOT conflate with Branch B (speaker
encoder / d-vector, Block 2).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | `reference_audio_22k` (→ mel features) | waveform @ 22050 Hz | f32 |
| out | `gpt_cond_latent` | (1, 32, 1024) | f32 → Block 3 (GPT prefix) |

`GPT_COND_LEN` (reference seconds) affects the mel time length — the FORGE log used 3 s for
the component test and 6 s in the pipeline. Pin this and record it.

## Foundation / template
Net-new TTNN. Perceiver resampler = learned latent queries doing cross-attention over the
conditioning-encoder features (32 output latents). No `speecht5_tts` analogue.

## Reference source
- coqui `TTS/tts/models/xtts.py` → `gpt.get_style_emb`, plus the conditioning encoder +
  Perceiver modules in the coqui GPT definition.

## Build steps
1. Mirror the conditioning encoder + Perceiver in `reference/conditioning_encoder.py`;
   PCC=1.0 vs coqui on a fixed `reference_audio_22k` golden input.
2. Port to TTNN in `tt/ttnn_conditioning_encoder.py` (+ `TTNNConfig` +
   `preprocess_*_parameters`).
3. Validate `gpt_cond_latent` (1,32,1024) at PCC ≈ 1.0 vs the reference golden.

## PCC validation plan
Golden input = fixed 22 kHz clip (reuse the FORGE reference clip). Target PCC ≈ 1.0 on the
32×1024 output. Test under `tests/`.

## Findings log (dated)
- (none yet)

## Open questions / TODO
- [ ] **group_norm tile-alignment risk:** FORGE FAIL #1 hit `group_norm ... got 505` here
      (mel/sequence time not a multiple of 32). Plan: pad time dim to next mult of 32
      (505→512) around group_norm, mask/slice back; verify meaned cond latents unaffected.
      Consider keeping conv/group_norm parts on CPU initially.
- [ ] Confirm mel front-end params (n_mels, hop, win) and `GPT_COND_LEN` to fix the shape.
- [ ] Confirm the Perceiver latent count is exactly 32 and channel dim 1024 in the ckpt.
