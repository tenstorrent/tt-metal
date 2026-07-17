# XTTS-v2 HiFi-GAN Vocoder — TTNN bringup (Block 4)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: not started
- Owner: —
- Started: —

## Role in pipeline
Final stage. Consumes GPT latents (Block 3) **directly** — **no mel intermediate** — and
the d-vector (Block 2), producing the 24 kHz waveform. The d-vector is injected via linear
projections at each upsampling layer (`g=speaker_embedding`).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | `gpt_latents` | (1, T_code, 1024) | f32 (from Block 3) |
| in | `speaker_embedding` (d-vector) | (1, 512, 1) | f32 (from Block 2) |
| out | `waveform` | 24 kHz mono | f32 → output .wav |

## Foundation / template
`speecht5_tts` pattern, but note SpeechT5's HiFi-GAN takes a **mel**; XTTS's takes **GPT
latents** — different input contract. **Per master decision: run on CPU first** (as
`speecht5_tts` runs its vocoder on CPU), port to TTNN **last**.

## Reference source
- coqui `TTS/tts/models/xtts.py` → `self.hifigan_decoder(gpt_latents, g=speaker_embedding)`
  and the coqui HiFi-GAN decoder module (with per-upsample-layer speaker conditioning).

## Build steps
1. Mirror the vocoder in `reference/hifigan.py`; this doubles as the CPU runtime impl.
   PCC=1.0 vs coqui on golden `gpt_latents` + `speaker_embedding`.
2. Wire it CPU-side into `tt/ttnn_xtts_model.py` so the e2e pipeline produces audio early.
3. (Later) port convs / transpose-convs / group_norm to TTNN.

## PCC validation plan
Golden inputs = reference `gpt_latents` + `speaker_embedding`. Compare output waveform
(PCC ≈ 1.0 vs coqui). Test under `tests/`.

## Findings log (dated)
- (none yet)

## Open questions / TODO
- [ ] **Hardest TTNN port** — conv1d / transpose-conv / group_norm tile-alignment (see
      master's FORGE FAIL #1). Do NOT start the TTNN port until the CPU e2e pipeline is
      correct.
- [ ] Confirm upsample factors / kernel sizes and where the d-vector projections attach.
- [ ] Confirm output sample rate 24 kHz and mono.
