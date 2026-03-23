# Fix LTX-2 AudioVideo Output Quality to Match Video-Only

This ExecPlan is a living document maintained in accordance with PLANS.md at the repository root.
The sections Progress, Surprises & Discoveries, Decision Log, and Key Measurements must be
kept up to date as work proceeds.

## Purpose

The video-only LTX-2 TTNN pipeline produces visually identical output to the CPU reference (PSNR 23-24 dB). The AudioVideo pipeline produces garbage video (PSNR 14-15 dB). After this work, running `generate_audio_video.py` will produce video quality matching `generate_video.py` (PSNR > 20 dB vs CPU AV reference), with audio output.

**How to verify:** Run both pipelines with same prompt/seed and compare:
```bash
source python_env/bin/activate
# Video-only (known good)
python models/tt_dit/demos/ltx/generate_video.py --num_frames 33 --height 256 --width 256 --steps 5 --seed 10 --output /tmp/v_only.mp4
# AudioVideo (should match)
python models/tt_dit/demos/ltx/generate_audio_video.py --num_frames 33 --height 256 --width 256 --steps 5 --seed 10 --output /tmp/av.mp4
```
Compare frame 16 visually — both should show a golden retriever in sunflowers.

## Acceptance Criteria

- Per-layer AV model video PCC > 0.999 vs CPU reference (currently 0.998)
- End-to-end AV video PSNR > 20 dB vs CPU AV reference (currently 14-15 dB)
- Audio output present and decodable
- All existing audio tests pass (`pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py`)

## Reference Implementation

- CPU reference model: `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py` — `LTXModel(model_type=AudioVideo)`
- CPU reference block: `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py` — `BasicAVTransformerBlock`
- TTNN AV model: `models/tt_dit/models/transformers/ltx/audio_ltx.py` — `LTXAudioVideoTransformerModel`
- TTNN AV block: `models/tt_dit/models/transformers/ltx/audio_ltx.py` — `LTXAudioVideoTransformerBlock`
- TTNN attention: `models/tt_dit/models/transformers/ltx/attention_ltx.py` — `LTXAttention`
- Video-only model (known good): `models/tt_dit/models/transformers/ltx/transformer_ltx.py` — `LTXTransformerModel`
- AV generation script: `models/tt_dit/demos/ltx/generate_audio_video.py`
- Video-only script: `models/tt_dit/demos/ltx/generate_video.py`

## Context and Orientation

**Current state:** The video-only pipeline matches the CPU reference at PSNR 23-24 dB. The AudioVideo pipeline runs end-to-end but produces garbage video (PSNR 14-15 dB). Per-layer PCC for the AV model is 0.998 (video) and 0.999 (audio) on a 1x1 mesh. The video-only model has per-layer PCC 0.9999.

**Key difference:** The AV model has bidirectional audio↔video cross-attention (`audio_to_video_attn` and `video_to_audio_attn`) that modifies both modality paths each layer. The video-only model does not have this. The 0.998 vs 0.9999 PCC gap may come from this cross-attention, OR from a bug in how the AV block processes video/audio differently from the video-only block.

**Architecture:** Each AV transformer block processes:
1. Video self-attention (with per-head gate + split RoPE)
2. Video text cross-attention (with AdaLN)
3. Audio self-attention (with per-head gate + split RoPE)
4. Audio text cross-attention (with AdaLN)
5. Bidirectional A↔V cross-attention (audio→video, video→audio)
6. Video feedforward
7. Audio feedforward

Steps 1-2 should match the video-only block exactly (same weights, same code path). Step 5 is unique to AV and modifies both paths.

## Plan of Work (Milestones)

### Milestone 1: Isolate whether the PCC gap is from cross-attention or from the video path itself

Run the AV block with the cross-attention output zeroed out (identity) and compare video PCC. If the video path alone gives PCC 0.999, the issue is in cross-attention. If it still gives 0.998, there's a bug in the video path within the AV block.

### Milestone 2: Layer-by-layer CPU vs TTNN trace

For each sub-operation in the AV block, compare the intermediate tensor between CPU and TTNN:
1. After video self-attention
2. After video text cross-attention
3. After audio self-attention
4. After audio text cross-attention
5. After A→V cross-attention
6. After V→A cross-attention
7. After video feedforward
8. After audio feedforward

This will pinpoint exactly which operation causes the PCC drop.

### Milestone 3: Fix the identified issue

Based on milestone 2 findings, fix the root cause. Common suspects:
- Cross-attention weight loading (dimension mismatch between 4096 and 2048)
- AdaLN modulation for cross-attention path using wrong timestep
- Scale/shift table indexing off by one
- Cross-attention norm using wrong norm (video norm1 reused instead of separate norm)

### Milestone 4: Verify end-to-end quality

Run the full AV pipeline and compare against CPU reference at PSNR > 20 dB.

## Progress

- [x] (2026-03-23) Milestone 1: Measured per-layer PCC — both video-only and AV have PCC 0.998. AV cross-attention doesn't degrade per-layer PCC. The issue is compounding across 48 layers × more ops per layer.
- [x] (2026-03-23) Tested HiFi4 for attention matmuls — no improvement (0.998449 vs 0.998451). Error not from matmul fidelity.
- [x] (2026-03-23) Fixed AV pipeline: added CFG, bf16 noise dtype, negative prompt encoding, CUDA patch.
- [x] (2026-03-23) Generated CPU AV reference: step ranges match TT within 5%.
- [ ] Milestone 2: Identify which specific TT op contributes most to 0.998 PCC gap
- [ ] Milestone 3: Fix the dominant precision bottleneck
- [ ] Milestone 4: End-to-end AV PSNR > 20 dB vs CPU AV ref

## Surprises & Discoveries

(to be filled as work proceeds)

## Decision Log

(to be filled as work proceeds)

## Constraints & Workarounds

- Hardware: WH LB, 2x4 mesh (SP=2, TP=4) for full pipeline, 1x1 for per-layer PCC tests
- 48-layer model doesn't fit on 1x1 mesh (OOM at layer 22)
- Audio latent needs tile-alignment padding (34→64 for 33 video frames)
- Split RoPE requires elementwise ops (6 ttnn ops per Q/K) since rotary_embedding_llama only supports interleaved

## Key Measurements

| Test | Metric | Value | Command |
|------|--------|-------|---------|
| Video-only per-layer | PCC | 0.9999 | 1-layer, 1x1 mesh, gated ref |
| AV per-layer video | PCC | 0.998 | 1-layer, 1x1 mesh |
| AV per-layer audio | PCC | 0.999 | 1-layer, 1x1 mesh |
| Video-only e2e | PSNR | 23-24 dB | 5 steps, 256x256, 2x4 mesh |
| AV e2e video | PSNR | 14-15 dB | 5 steps, 256x256, 2x4 mesh |
| AV 0-layer | PCC | 0.999989 | patchify+adaln+output only, 1x1 mesh |
| AV 1-layer HiFi4 | PCC | 0.998449 | 1x1 mesh, HiFi4 matmuls |
| V-only 1-layer HiFi4 | PCC | 0.998484 | 1x1 mesh, same test conditions |
