# LTX-2 AudioVideo Pipeline Bringup

## Goal
Get the full LTX-2.3 22B AudioVideo pipeline working on WH LB (2x4 mesh), matching the official pipeline output for both video and audio modalities.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Use reference SPLIT RoPE with double-precision freq grid | Model trained with SPLIT; interleaved gives PCC 0.09 | Interleaved via rotary_embedding_llama + trans_mat |
| Host-side per-head gate computation | Exact fp32 gate logits; tiny matmul (N×32) negligible latency | On-device gate Linear in fp32 |
| Manual elementwise split rope (6 ttnn ops) | rotary_embedding_llama only supports interleaved via trans_mat | Fused split rope kernel (not available) |

## Constraints & Workarounds
- Hardware: WH LB, 2x4 mesh (SP=2, TP=4)
- Video: 4096 dim, 32 heads, 128 head_dim
- Audio: 2048 dim, 32 heads, 64 head_dim
- Audio RoPE: 1D temporal (max_pos=[20]) vs video 3D (max_pos=[20,2048,2048])
- Audio cross-attention dim: 2048 (from audio_embeddings_connector projection)
- Bidirectional A↔V cross-attention: video Q (4096) × audio KV (2048), audio Q (2048) × video KV (4096)
- VAE cache: Must clear when Conv3D C_in_block changes (stale cache = garbage output)
- Conv3D blocking sweep branch: `kevinmi/wan-conv3d-blocking-sweep` for faster VAE decode

## Surprises & Discoveries
- (from video bringup) RoPE split vs interleaved was PCC 0.09 — single biggest correctness bug
- (from video bringup) Reference PCC test used ungated model by default (apply_gated_attention=False)

## Open Questions
- [ ] Do audio attention tests pass with current code?
- [ ] Audio RoPE: 1D positions — does the current precompute_freqs_cis handle 1D correctly with SPLIT format?
- [ ] Cross-modal attention dimension mismatch (4096 vs 2048) — how does TTNN handle this?
- [ ] Audio VAE decode path — CPU torch or TTNN?
- [ ] Conv3D blocking sweep integration for faster VAE

## State
- [x] Video pipeline matching official (PSNR 23-24 dB, visually identical)
- [x] Per-head gating implemented and verified (PCC 0.9999 per layer)
- [x] Split RoPE with double-precision freq grid
- [ ] Run existing audio unit tests, fix failures
- [ ] Audio self-attention with correct RoPE (1D SPLIT)
- [ ] Audio cross-attention (text→audio)
- [ ] Bidirectional A↔V cross-attention
- [ ] Audio feedforward
- [ ] Full AudioVideo transformer block
- [ ] Full AudioVideo model forward pass
- [ ] End-to-end AudioVideo generation pipeline
- [ ] Audio VAE decode
- [ ] Conv3D blocking sweep for faster VAE

## Key Measurements
| Test | PCC | Notes |
|------|-----|-------|
| Video per-layer (1 block, gated ref) | 0.999954 | 1x1 mesh, N=192 |
| Video end-to-end (5 steps, 256x256) | PSNR 23-24 dB | 2x4 mesh, visually identical |
| Video end-to-end (30 steps, 512x768) | Stable latent [-4.2, 4.1] | 121 frames, 224.8s denoise |
