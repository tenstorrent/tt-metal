# XTTS-v2 ResNet Speaker Encoder (d-vector) — TTNN bringup (Block 2)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: not started
- Owner: —
- Started: —

## Role in pipeline
**Branch B** of conditioning. Consumes the reference clip (16000 Hz) and produces the
single **d-vector** (speaker embedding) injected into the HiFi-GAN vocoder (Block 4) as
`g=speaker_embedding`. Do NOT conflate with Branch A (Perceiver prefix, Block 1).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | `reference_audio_16k` | waveform @ 16000 Hz | f32 |
| out | `speaker_embedding` (d-vector) | (1, 512, 1) | f32 → Block 4 (vocoder `g=`) |

## Foundation / template
Net-new. Conv-heavy (ResNet). **Per master decision: run on CPU first**, port to TTNN later.

## Reference source
- coqui `TTS/tts/models/xtts.py` speaker-encoder path (HiFi-GAN speaker encoder /
  d-vector). Confirm exactly which encoder class the coqui checkpoint instantiates —
  FORGE open question #3 flagged this as unconfirmed.

## Build steps
1. Mirror the ResNet speaker encoder in `reference/speaker_encoder.py`; PCC=1.0 vs coqui on
   a fixed `reference_audio_16k` golden input.
2. Keep as CPU runtime initially; expose `speaker_embedding` (1,512,1) golden for Block 4.
3. (Later) port convs to TTNN in `tt/ttnn_speaker_encoder.py` once core pipeline works.

## PCC validation plan
Golden input = fixed 16 kHz clip. Target PCC ≈ 1.0 on the (1,512,1) d-vector. Test under
`tests/`.

## Findings log (dated)
- (none yet)

## Open questions / TODO
- [ ] Confirm which speaker-encoder class the coqui XTTS-v2 checkpoint uses (FORGE Q#3).
- [ ] Confirm d-vector dim is 512 and the (1,512,1) layout the vocoder expects.
- [ ] group_norm/conv tile-alignment risk when porting to TTNN (see master).
