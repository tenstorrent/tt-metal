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
- [ ] Gemma encoder correctness after sandwich norm fix — need to run test_gemma_cpu_qat.py + compare with reference

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
- [ ] Run full pipeline on BH Galaxy 4x8 with run_ltx_fast.py --mesh 4,8

## Key Measurements
| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| test_ltx_transformer_block_smoke 4x8sp1tp0 | forward pass | PASS (2026-05-11) | shape=(1,256,4096), mean≈0, std≈1.0 |

## Next Optimization Steps (ordered by impact)
1. **Device-side Euler step + keep latents on device** — eliminates 1 D2H + 1 H2D per step
2. **Device-side CFG** (`ttnn.lerp`) — eliminates velocity gather + host lerp per step
3. **Tune SDPA chunk sizes for 4x8 BH** — sweep q/k chunks for Galaxy SRAM
4. **Caption projection on device** (19B distilled) — small 2-layer MLP, ~2ms per step
5. **Conv3d blocking sweep for LTX VAE** — 128ch latent different from Wan
