# LTX-2 AudioVideo Pipeline Bringup

## Goal
Get the full LTX-2.3 22B AudioVideo pipeline working on WH LB (2x4 mesh), matching the official pipeline output for both video and audio modalities.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Use reference SPLIT RoPE with double-precision freq grid | Model trained with SPLIT; interleaved gives PCC 0.09 | Interleaved via rotary_embedding_llama + trans_mat |
| Host-side per-head gate computation | Exact fp32 gate logits; tiny matmul (N×32) negligible latency | On-device gate Linear in fp32 |
| Manual elementwise split rope (6 ttnn ops) | rotary_embedding_llama only supports interleaved via trans_mat | Fused split rope kernel (not available) |
| HiFi4 for attention matmuls | Removes TF32 truncation; no perf regression measured | HiFi2 (same PCC, slightly faster) |
| Use official `encode_prompts` for text encoding | Produces both video (4096-dim) and audio (2048-dim) context correctly | Manual encode_prompt function |
| Combined MP4 export via `ltx_pipelines.utils.media_io.encode_video` | Official muxing of video frames + audio waveform | Separate video.mp4 + audio.wav files |
| Cross PE: temporal-only dim=2048, separate Q/K rope tensors | Reference uses `positions[:, 0:1, :]` with `audio_cross_attention_dim=2048`; Q rope SP-sharded, K rope full-seq | Single shared rope for cross-attn (wrong: Q and K have different sequence lengths from different modalities) |

## Constraints & Workarounds
- Hardware: WH LB, 2x4 mesh (SP=2, TP=4)
- Video: 4096 dim, 32 heads, 128 head_dim
- Audio: 2048 dim, 32 heads, 64 head_dim
- Audio RoPE: 1D temporal (max_pos=[20]) vs video 3D (max_pos=[20,2048,2048])
- Audio token count computed from `AudioLatentShape.from_video_pixel_shape()`, padded to tile-aligned
- `caption_projection` and `*_embeddings_connector` not loaded to device — these are handled by CPU `encode_prompts` pre-processing
- VAE cache: Must clear when Conv3D C_in_block changes (stale cache = garbage output)
- Conv3D blocking sweep branch: `kevinmi/wan-conv3d-blocking-sweep` for faster VAE decode

## Surprises & Discoveries
- RoPE split vs interleaved was PCC 0.09 — single biggest video correctness bug
- Reference PCC test used ungated model by default (apply_gated_attention=False) — gave false 0.989 PCC
- **A↔V cross-attention context modulation was missing** — reference modulates BOTH Q and KV sides with separate AdaLN params; our code only modulated Q. This caused PSNR 14 dB → 21-24 dB after fix.
- Per-layer PCC 0.998 is the same for both video-only and AV models — the quality gap was entirely from the missing context modulation bug
- HiFi4 vs HiFi2 made zero difference to per-layer PCC (0.998449 vs 0.998451)
- **Per-op PCC investigation**: Largest delta contributors are self-attention (0.28 video, 0.33 audio) and feedforward (0.23 video, 6.59 audio!). A↔V cross-attention has negligible delta (0.001 video, 0.03 audio). Audio FF delta is 30× larger than video FF because audio std grows 4× through the block (0.94→20.9). Error is from bf16 SDPA/matmul precision, not input quantization (bf16 input PCC = 1.0). bf16 quantization of each intermediate only costs PCC ~0.000001–0.000003 (negligible). The ~0.001 PCC gap is from accumulated bf16 matmul errors through SDPA and FF, compounding across the chain of ops.

## Open Questions
- [x] Cross-modal positional embeddings (cross_positional_embeddings) — implemented: temporal-only RoPE on A↔V cross-attention Q/K matching reference. Separate SP-sharded Q rope and full-seq K rope.
- [ ] Conv3D blocking sweep integration for faster VAE decode
- [ ] Performance optimization: host-side gate computation adds latency per layer

## State
- [x] Video-only pipeline matching official (PSNR 23-24 dB)
- [x] Per-head gating implemented and verified
- [x] Split RoPE with double-precision freq grid
- [x] All 9 audio unit tests pass
- [x] AudioVideo pipeline end-to-end with combined video+audio MP4 export
- [x] CFG with variance rescaling for AV pipeline
- [x] A↔V cross-attention context modulation fixed (root cause of quality gap)
- [x] **AV video quality matching video-only (PSNR 21-24 dB vs CPU AV ref)**
- [x] Full 512x768, 121-frame, 30-step video-only generated (224.8s denoise)
- [x] Full 512x768, 121-frame, 30-step AV generated (166.2s denoise + audio)
- [x] Cross-modal positional embeddings for A↔V attention
- [x] Audio PCC tests added (vs PyTorch reference with 22B weights)
- [ ] Conv3D blocking sweep for faster VAE decode
- [x] Full-resolution AV re-run (512x768, 121 frames, 30 steps, 353.5s denoise)
- [x] Cross PE tile alignment fix — host-side fallback for D_half < 64, fixed SP sharding for A↔V context (both Q and K are SP-sharded since context comes from the other modality's hidden state)

## Key Measurements
| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| Video-only per-layer | PCC | 0.998 | 1x1 mesh, split RoPE. Note: an earlier test showed 0.9999 but that was before the SPLIT RoPE fix (wrong INTERLEAVED RoPE produced smaller outputs that happened to match better). Both video-only and AV have 0.998 with correct SPLIT RoPE — no discrepancy between them. |
| AV per-layer video | PCC | 0.998 | 1x1 mesh, split RoPE, with xattn context modulation fix |
| AV per-layer audio | PCC | 0.999 | 1x1 mesh, split RoPE |
| Video-only e2e (5 steps, 256x256) | PSNR | 23-24 dB | vs CPU video-only ref |
| AV e2e video (5 steps, 256x256) | PSNR | 21-24 dB | vs CPU AV ref, after xattn fix |
| AV e2e video BEFORE fix | PSNR | 12-14 dB | missing context modulation |
| AV 1-layer PCC (with cross PE) video | PCC | 0.999031 | 1x1, split RoPE, cross PE enabled |
| AV 1-layer PCC (with cross PE) audio | PCC | 0.998776 | 1x1, split RoPE, cross PE enabled |
| Video-only (30 steps, 512x768) | Latent | [-4.2, 4.1] | 121 frames, 224.8s denoise |
| AV (30 steps, 512x768) | Latent | [-2.45, 2.62] video | 121 frames, 166.2s denoise |
| AV (30 steps, 512x768, cross PE) | Denoise | 858.8s | 121 frames, host-side rope fallback adds 2.4× overhead |
| AV (30 steps, 512x768, no cross PE) | Denoise | 353.5s | 121 frames, post-context-modulation fix |

## Major Bugs Found & Fixed
1. **RoPE format** — Used INTERLEAVED (PCC 0.09 vs ref), needed SPLIT. Fixed with manual elementwise rotation.
2. **Reference PCC test** — `apply_gated_attention=False` default made gated TT vs ungated ref comparison. PCC 0.989→0.999.
3. **Position encoding** — Simple latent indices vs official pixel-space with causal fix.
4. **Denoised dtype** — float32 vs bf16 mismatch with reference X0Model.
5. **AV cross-attention context modulation** — Only modulated Q side, not KV side. PSNR 14→24 dB.
6. **AV pipeline missing CFG** — No classifier-free guidance; default guidance_scale was 1.0.
7. **AV noise dtype** — float32 randn vs bf16 randn (different random values).
