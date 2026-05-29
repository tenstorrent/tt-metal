# LTX-2 Bringup Notes

Engineering history for the LTX-2.3 TT-DiT port. For user-facing run instructions see [`../../LTX2.md`](../../LTX2.md). For host-side CPU fallbacks see [`HOST_OPS.md`](HOST_OPS.md). For the original port plan see [`LTX2_DIT_PORT.md`](LTX2_DIT_PORT.md). Historical ExecPlans live in [`plans/`](plans/).

## Architecture Decisions

| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| SPLIT RoPE with double-precision freq grid | Model trained with SPLIT; interleaved gives PCC ~0.09 | `rotary_embedding_llama` + trans_mat (interleaved) |
| Host-side per-head gate (fp32 logits) | Exact gate logits; tiny matmul negligible vs SDPA | On-device gate Linear in bf16 (compounds over 48L×N steps) |
| Manual elementwise split RoPE | No fused split-rope kernel in TTNN | Fused split rope kernel |
| HiFi4 for attention matmuls | Removes TF32 truncation; no PCC regression vs HiFi2 | HiFi2 |
| SDPA + attention MM both back to HiFi4 (original LTX state) after a brief Wan-parity experiment | SDPA HiFi4 protects A↔V cross-attn quality (~5% per-block error at HiFi2); MM HiFi4 removes TF32 truncation | Wan parity (SDPA HiFi2 / MM HiFi2+approx) — reverted; per-instance `high_fidelity_sdpa` flag — removed |
| `proj_out`/`audio_proj_out` HiFi4 `packer_l1_acc=True` | Match WanTransformer hifi4 config (was `False`, inherited from norm) | `packer_l1_acc=False` |
| RoPE stays a separate `rotary_embedding_llama` op (not folded into RMSNorm) | LTX RoPE is **per-head** `[1, H, seq, head_dim]` (freqs span full inner_dim, sliced head-wise); fused norm kernel asserts head-uniform `[1, 1, seq, head_dim]`. Interleaved-vs-split is irrelevant — per-head is the blocker. Head-split fusion into norm IS used. | Fold RoPE into `wan_fused_rmsnorm_post_allgather` like Wan (fails `TT_FATAL` on per-head rope) |
| Reference `encode_prompts` (CPU Gemma) | Produces video (4096-dim) and audio (2048-dim) context correctly; TTNN Gemma explodes after layer 1 | On-device Gemma encoding |
| Combined MP4 via `ltx_pipelines.utils.media_io.encode_video` | Official mux of video + audio | Separate video/audio files |
| Cross PE: temporal-only dim=2048, separate Q/K rope | Reference uses `positions[:, 0:1, :]`; Q SP-sharded, K full-seq | Single shared rope for cross-attn |
| BH Galaxy 4×8: `sp_axis=1`, `tp_axis=0`, Ring topology | Matches Wan BH 4×8; long-seq benefits from SP | Reversed SP/TP; Linear topology |
| `FABRIC_1D_RING` on BH 4×8 | `FABRIC_1D` fails all-gather ring routing | `FABRIC_1D` |
| Fix Gemma-3 sandwich norms in tt_dit encoder | demos/gemma3 is WH-only decode-mode | Re-use gemma3 demo model |

## Hardware & Shape Constraints

- **Wormhole Loud Box**: 2×4 mesh, SP=2, TP=4, Linear topology
- **Blackhole Galaxy**: 4×8 mesh, SP=8, TP=4, Ring topology, `FABRIC_1D_RING`
- Video: 4096 dim, 32 heads, 128 head_dim, 128-channel latents
- Audio: 2048 dim, 32 heads, 64 head_dim; 1D temporal RoPE (`max_pos=[20]`) vs video 3D (`max_pos=[20,2048,2048]`)
- Audio token count from `AudioLatentShape.from_video_pixel_shape()`, padded to tile alignment
- `caption_projection` and `*_embeddings_connector` stay on CPU (handled by `encode_prompts_reference`)
- VAE cache must be cleared when Conv3D `C_in_block` changes

## Major Bugs Found & Fixed

1. **RoPE format** — INTERLEAVED via trans_mat gave PCC ~0.09; model needs SPLIT with manual elementwise rotation and double-precision freq grid.
2. **Reference PCC test** — `apply_gated_attention=False` default compared gated TT vs ungated ref (0.989 → 0.999+ after fix).
3. **Position encoding** — Simple latent indices vs official pixel-space coords with causal fix.
4. **Denoised dtype** — float32 vs bf16 mismatch with reference `X0Model`.
5. **AV cross-attention context modulation** — Only Q side modulated, not KV; PSNR 14 → 21–24 dB after fix.
6. **AV pipeline missing CFG** — No classifier-free guidance; default `guidance_scale` was 1.0.
7. **AV noise dtype** — float32 `randn` vs bf16 `randn`.
8. **Gate bf16 on device** — Invisible in single-layer tests; compounded to latent PCC 0.92 over full pipeline; reverted to fp32 host gate.
9. **Post-connector token zeroing** — Destroyed register tokens; reference keeps them alive.
10. **Padding side** — Right-padding broke register packing; reference uses left-padding.
11. **`call_av` sigma schedule** — `compute_sigmas` without `num_tokens` used anchor 4096 instead of `video_N+audio_N`.
12. **`generate_freqs` 3D branch** — Missing transpose before `get_fractional_positions`.
13. **Gemma-3 sandwich norms** — Missing post_attention, pre_ffn, post_ffn norms (Gemma-3 vs Gemma-2 difference).
14. **Cross PE tile alignment** — Host fallback for `D_half < 64`; both Q and K SP-sharded for A↔V context.

## Key Measurements

### Per-layer PCC (1 block, 1×1 mesh)

| Configuration | PCC | Notes |
|---------------|-----|-------|
| Gated ref + gated TT | 0.999954 | `apply_gated_attention=True` on reference |
| Ungated ref + gated TT (wrong) | 0.989 | Test bug |
| Video-only per-layer | 0.998 | SPLIT RoPE |
| AV per-layer video | 0.998 | With xattn context modulation |
| AV per-layer audio | 0.999 | SPLIT RoPE |
| AV 1-layer + cross PE video | 0.999031 | |
| AV 1-layer + cross PE audio | 0.998776 | |

### End-to-end quality

| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| Video-only e2e (5 steps, 256×256) | PSNR | 23–24 dB | vs CPU video-only ref |
| AV e2e video (5 steps, 256×256) | PSNR | 21–24 dB | vs CPU AV ref, after xattn fix |
| AV e2e BEFORE xattn fix | PSNR | 12–14 dB | |
| Video latent PCC (40 steps, gate fix) | PCC | 0.997 | Was 0.922 with bf16 device gate |
| Audio latent PCC (40 steps, gate fix) | PCC | 0.998 | Was 0.970 |
| Transformer block 4×8 BH | PCC | 99.88% | Real 22B weights |
| VAE decoder 4×8 | PCC | 99.9964% | |

### Denoise timing (512×768, 121 frames, 30 steps, Pro AV)

| System | Denoise | s/step | Notes |
|--------|---------|--------|-------|
| WH 2×4 | ~166–795s | ~20–29s | Range reflects cross-PE host fallback vs optimized path |
| BH 4×8 | ~843s | ~28s | Post sigma/ge_gamma fixes |
| WH + cross PE host fallback | 858.8s | ~28.6s | 2.4× overhead from split-rope host path |
| WH no cross PE | 353.5s | ~11.8s | Post context-modulation fix |

Gate values (checkpoint `ltx-2.3-22b-dev`): attn1 range [0.04, 1.98], attn2 [0.06, 1.88] — non-trivial, not near-identity.

## Per-op Error Budget

Largest delta contributors per block: self-attention (~0.28 video, ~0.33 audio) and feedforward (~0.23 video, ~6.59 audio). A↔V cross-attention delta is negligible (~0.001 video). Audio FF delta is larger because audio std grows ~4× through the block. Error is accumulated bf16 SDPA/matmul precision, not input quantization.

## CCL Fusion (TP all-gather → matmul)

On Ring topology, `ColParallelLinear` can fold the TP all-gather of its input into the
matmul via `all_gather_minimal_matmul_async` (input sharded on contraction dim K is
gathered across the TP axis during the matmul). On Linear topology that op is unavailable,
so an explicit AG + plain matmul is used. WanAttention/transformer already do this; the LTX
port was carrying redundant standalone all-gathers and has been brought in line:

| Site | Before | After (Ring) |
|------|--------|--------------|
| `attn._to_out_fused_addcmul` | explicit AG + `dit_minimal_matmul_addcmul_fused` | `all_gather_minimal_matmul_async` (AG + matmul + bias + addcmul fused) |
| transformer FFN (video-only, AV video, AV audio) | unconditional AG + `forward_fused_addcmul` (no `parallel_config`) | `forward_fused_addcmul(parallel_config=...)` — AG folds into `ff1` |
| A↔V cross-attn KV (`audio_kv_a2v`, `video_kv_v2a`) | explicit TP AG in transformer before cross-attn | AG folds into cross-attn `to_kv` (`kv_parallel_config` on Ring); padding mask still applied before `to_kv` |

Linear topology and TP=1 behavior are unchanged (explicit AG / plain matmul fallbacks).

### `proj_out` stays explicit-AG (matches Wan) — deliberately *not* fused

`proj_out`/`audio_proj_out` are replicated `Linear` (full weight on every TP device),
preceded by an explicit `dim=3` TP all-gather. This matches Wan's `transformer_wan.py`
final projection and the pipeline's `device_to_host(tp_already_gathered=True)` contract
(output is replicated-by-identical-compute across TP).

An AG-fused variant (call `all_gather_minimal_matmul_async` with the replicated weight →
output stays replicated) was prototyped and **reverted**. It is *correct* — the op computes
`K = K_local·ring_size` and only requires `K == weight_K`, which holds for a replicated
weight — but:
- it is a once-per-step cold path (not the 48-block hot loop), so ROI is negligible;
- it keeps the **redundant full-`[M,4096]@[4096,128]` matmul on every TP device** (TP× the
  FLOPs), same as the explicit-AG path — fusing only overlaps the gather, not the compute.

The compute-optimal fix is **RowParallel** `proj_out` (weight sharded on K): no input gather
and 1/TP the matmul FLOPs, then a cheap reduce of the 128-wide output. That changes output
sharding (N-fractured, needs the `device_to_host`/`tp_already_gathered` contract updated) and
is tracked under Performance below.

## Open Work

### Correctness
- [ ] TTNN Gemma numerical stability (separate from DiT bringup)
- [ ] Validate `ge_gamma=2.0` gradient estimation quality vs reference (currently `ge_gamma=0` in some paths)
- [ ] `fp32_dest_acc_en` on SDPA compute kernel config
- [ ] Embedding cache key should include Gemma/checkpoint paths (prompt-only key risks stale cache)

### Performance (ordered by impact)
1. Batched/fused MultiModalGuider passes (up to 4 serial forwards/step today)
2. Device-resident denoise loop: keep latents on device, device Euler + CFG (`ttnn.lerp`)
3. Tune matmul and SDPA blocking for LTX shapes
4. On-device per-head gate (or async batched host readback)
5. Native TTNN split/interleaved RoPE (eliminate host roundtrips for audio `D_half=32`)
6. Conv3D blocking sweep for LTX VAE (128ch latent)
7. On-device Gemma encoding
8. Caption projection on device (19B distilled)
9. RowParallel `proj_out`/`audio_proj_out` (eliminate input AG + redundant full matmul; needs `device_to_host` contract update)

See [`HOST_OPS.md`](HOST_OPS.md) for operations that must move on-device to unlock the above.

## Debug Utilities

CPU reference runner for comparing against official `ltx_pipelines`:

```bash
cd $TT_METAL_HOME
python models/tt_dit/tests/models/ltx/reference_cpu_pipeline.py \
  --prompt "A cat playing piano in a cozy room" \
  --output ref_cat.mp4 --steps 30 --seed 42
```

Requires cloned `LTX-2/` beside `tt-metal` and checkpoint/Gemma paths (see `LTX2.md`).
