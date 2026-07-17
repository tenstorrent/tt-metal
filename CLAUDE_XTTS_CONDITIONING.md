# XTTS-v2 Conditioning Encoder + Perceiver Resampler — TTNN bringup (Block 1)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: not started
- Owner: —
- Started: —

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
