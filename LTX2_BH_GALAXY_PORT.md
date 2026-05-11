# LTX-2 BH Galaxy 4x8 Port + Optimization

## Goal
Rebase LTX-2 DiT port onto current main, fix Gemma encoder correctness, and make the pipeline run on BH Galaxy 4x8 mesh (32 chips).

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Fix Gemma-3 sandwich norms in-place (tt_dit encoder) | models/demos/multimodal/gemma3 is WH-only, decode-mode, wrong parallel infra | Re-use gemma3 demo model |
| 4x8 BH Galaxy: sp_axis=1 (8 chips), tp_axis=0 (4 chips) | Matches Wan BH 4x8 config; more SP than TP benefits long-sequence transformer | sp_axis=0 tp_axis=1 (reversed) |
| Ring topology for 4x8 | Matches Wan BH 4x8 config; better than Linear for ring-connected chips | Linear topology |
| create_pipeline() auto-dispatch via mesh_shape key | Clean, zero caller changes for 4x8 | Per-script if/else |

## Constraints & Workarounds
- Hardware: BH Galaxy 4x8 mesh (32 chips), blackhole arch
- LTX 22B: dim=4096, 48 layers, 32 heads video / 32 heads audio
- Host-side ops remaining: Euler step, per-step latent transfer, CFG guidance (see HOST_OPS.md)
- Audio split RoPE fallback: TTNN subtile broadcast bug for D_half=32 — host fallback still active
- VAE encoder: torch-only (LTXVideoEncoderTorch) — not needed for inference

## Surprises & Discoveries
- Gemma-3 has 4 norms per layer (not 2): input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm. The port was missing the last 3 (post_attention, pre_ffn, post_ffn) — sandwich norms around FFN are a Gemma-3 vs Gemma-2 difference.
- `models/demos/multimodal/gemma3` is WH-only and decode-mode — cannot be used as LTX encoder.
- `FABRIC_1D` fails on 4x8 Galaxy for all-gather ring ops — must use `FABRIC_1D_RING` (matches Wan BH 4x8 pattern).
- `generate_freqs` in `rope_ltx.py` was not transposing 3D `(B, N, n_dims)` indices_grid before calling `get_fractional_positions` which expects `(B, n_dims, N)`. Fixed by adding transpose in the 3D branch of `generate_freqs`.
- `calculate_softplus_body` in BH LLK gained a second template param `is_fp32_dest_acc_en`; `compute_common.hpp` was calling it with only one. Fixed to `<APPROX, false>`.

## Open Questions
- [x] FABRIC_1D_RING required for 4x8 BH Galaxy (not FABRIC_1D) — all-gather ring routing needs it
- [ ] SDPA chunk sizes for 4x8 BH: sdpa_chunk_size_map has (True, 8, 4) entry — verify correct key lookup with sp=8, tp=4
- [ ] Does create_pipeline need `num_links=4` for BH 4x8? (Wan BH 4x8 uses 2 — used 2 here too)
- [x] Gemma encoder: test_gemma_cpu_qat.py requires ModelLedger (internal Lightricks API, not in public LTX-2 repo) — cannot run
- [x] Bad video quality root-cause investigation (2026-05-11):
  - **call_av sigma schedule bug FIXED**: compute_sigmas called without num_tokens, used MAX_SHIFT_ANCHOR=4096 default instead of actual video_N+audio_N=6170 → wrong sigma shift. Fixed in pipeline_ltx.py line 1226
  - **ge_gamma=0.0 in run_ltx_gen.py FIXED**: velocity correction (gradient estimation) was disabled; changed to ge_gamma=2.0 matching default
  - Transformer block 4x8: PCC=99.88% vs reference (SPLIT rope, good quality)
  - VAE decoder: PCC=99.9964% (not the issue)
  - test_attention_ltx.py: Fixed FABRIC_1D→FABRIC_1D_RING; added 4x8sp1tp0 variant; fixed INTERLEAVED vs SPLIT rope mismatch in test reference
  - test_transformer_ltx.py: Fixed INTERLEAVED→SPLIT rope in block/model tests; fixed LTXRopeType import clash

## State
- [x] Rebase onto main (clean, 80 commits, 2026-05-11)
- [x] Gemma-3 sandwich norms fixed: post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm
- [x] 4x8 config added to create_pipeline() in pipeline_ltx.py
- [x] run_ltx_fast.py / run_ltx_pro.py: --mesh arg, use create_pipeline
- [x] test_transformer_ltx.py: 4x8sp1tp0 added to block/model/inner_step tests
- [x] test_pipeline_ltx.py: bh_glx_4x8 added to AV pipeline test
- [x] Build succeeded (2026-05-11)
- [ ] Validate Gemma encoder PCC vs reference after norm fix
- [x] LTXTransformerBlock smoke test PASSED on BH Galaxy 4x8 (2026-05-11): shape=(1,256,4096), mean≈0, std≈1.0
- [x] 22B AV forward pass on BH Galaxy 4x8 with real checkpoint (2026-05-11): PASSED
- [x] Full end-to-end generation on BH Galaxy 4x8 (2026-05-11): 121f@512x768, 30 steps → MP4 output in 17min
- [x] call_av sigma schedule bug fixed (2026-05-11): num_tokens=video_N+audio_N instead of default 4096
- [x] run_ltx_gen.py ge_gamma fixed (2026-05-11): 0.0 → 2.0 (enables velocity correction)
- [x] Unit tests: AdaLN PASS, RoPE PASS, attention 4x8 PASS, transformer block 4x8 PCC=99.88%, VAE PCC=99.9964%
- [x] test_vae_ltx.py: 4x8 added to all 4 test functions (FABRIC_1D, matching Wan convention); all 9 variants PASS (2026-05-11)
- [x] Re-run full generation after sigma/ge_gamma fixes (2026-05-11): 30-step → ltx_output_30step.mp4 with audio (5.01s)
- [x] torchaudio CPU fix (2026-05-11): patched _extension/__init__.py to catch OSError; fixed vocoder import path and audio dtype

## Key Measurements
| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| test_ltx_transformer_block_smoke 4x8sp1tp0 | forward pass | PASS (2026-05-11) | shape=(1,256,4096), mean≈0, std≈1.0 |
| run_ltx_22b_dummy.py 4x8 25f@512x512 2-step | step time | 33.2s/step (2026-05-11) | real 22B weights, dummy prompt; video (1,1024,128), audio (1,26,128), all finite |
| run_ltx_gen.py 4x8 121f@512x768 30-step (sigma+ge_gamma fixed) | e2e time | 16.3min / 28.1s/step (2026-05-11) | Gemma 25s, denoising 843s, VAE 18.6s, audio 5.01s → ltx_output_30step.mp4 |
| test_vae_ltx 4x8: full decoder | PCC | 99.9964% (2026-05-11) | FABRIC_1D, get_device_tensors[0] |
| test_vae_ltx 4x8: causal conv3d (3 variants) | PCC | 99.9994–99.9995% (2026-05-11) | |
| test_vae_ltx 4x8: depth-to-space upsample (3 variants) | PCC | 99.9994–99.9995% (2026-05-11) | |
| test_vae_ltx 4x8: resnet block same-ch | PCC | pass (2026-05-11) | |
| test_vae_ltx 4x8: resnet block expand-ch | PCC | 99.5695% (2026-05-11) | lower due to GroupNorm(1) vs LayerNorm precision |

## Next Optimization Steps (ordered by impact)
1. **Device-side Euler step + keep latents on device** — eliminates 1 D2H + 1 H2D per step
2. **Device-side CFG** (`ttnn.lerp`) — eliminates velocity gather + host lerp per step
3. **Tune SDPA chunk sizes for 4x8 BH** — sweep q/k chunks for Galaxy SRAM
4. **Caption projection on device** (19B distilled) — small 2-layer MLP, ~2ms per step
5. **Conv3d blocking sweep for LTX VAE** — 128ch latent different from Wan
