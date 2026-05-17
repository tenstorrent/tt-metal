# WAN 2.2 S2V Performance — BH-LB (2×4)

## Setup

- HW: BH Loud Box, mesh **(2, 4)** = 8 chips, sp_factor=4, tp_factor=2, topology=Linear
- Resolution: 832×480 (480p), 81 video frames
- Reference inputs: bundled `models/tt_dit/tests/models/wan2_2/assets/pose.png` (human
  portrait) + `talk.wav`. Prompt "a person is talking".
- Config aligned with reference `s2v_14B`: `motion_frames=73/19`, `guidance_scale=4.5`,
  `drop_first_motion=True`, `FlowUniPCMultistepScheduler(sample_shift=3)`, s2v-specific
  negative prompt.

## Reproduce

```bash
cd /home/kevinmi/tt-metal && source python_env/bin/activate
export PYTHONPATH=$(pwd) TT_DIT_CACHE_DIR=/home/kevinmi/.cache PYTHONUNBUFFERED=1

# 5 steps (~2:30 wall — quick localization of stage costs)
pytest "models/tt_dit/tests/models/wan2_2/test_performance_wan_s2v.py::test_s2v_pipeline_performance[blackhole-steps5-resolution_480p-bh_2x4sp1tp0]" -s -v

# 40 steps (production)
pytest "models/tt_dit/tests/models/wan2_2/test_performance_wan_s2v.py::test_s2v_pipeline_performance[blackhole-steps40-resolution_480p-bh_2x4sp1tp0]" -s -v
```

Stage timings come from `BenchmarkProfiler` hooks in `pipeline_wan_s2v.py:prepare_latents`
and the cumulative audio-injection counter on `transformer_wan_s2v.after_transformer_block`.

## Pre-/post-cleanup comparison

The cleanup plan (`PLAN_WAN_S2V_CLEANUP.md`) landed four perf-relevant changes:

- **M4** Skip motion-VAE encode of 73 zero frames when `drop_first_motion=True` (the
  transformer ignores `motion_latents` on that path).
- **M5** Cache audio K/V projections inside `AudioInjector_WAN` across diffusion steps —
  the audio embedding is constant per clip, so `to_kv` + `norm_k` + V head-split run once
  per inject layer per clip instead of every step.
- **M6** Move `MotionEncoder_tc` stages 2 + 3 LayerNorm onto device — the conv → LN → SiLU
  pipeline runs without bouncing the activation back to host between conv stages.
- **VAE decode** picked up an unrelated upstream win during the cleanup window.

### 5-step run

| Stage | Pre-cleanup | Post-cleanup | Δ |
|---|---|---|---|
| Text encoder (UMT5) | 0.4s | 0.5s | ~same |
| prepare_latents (total) | 25.7s | **8.8s** | **−66%** |
| ↳ VAE encode (ref) | 1.4s | 1.4s | ~same |
| ↳ wav2vec2 + bucketing | 1.6s | 1.6s | ~same |
| ↳ prepare_audio_emb | 4.9s | 4.6s | −5% (M6 partial) |
| ↳ VAE encode (motion 73f) | 16.5s | **0.002s** | **M4 — −100%** |
| ↳ prepare_cond_emb | 1.4s | 1.2s | ~same |
| Denoising loop | 59.7s | 58.7s | −2% |
| ↳ Audio cross-attn cumulative | 23.9s | 23.5s | −2% (M5 saves K/V proj only) |
| ↳ Block-stack (non-audio) | 35.8s | 35.1s | ~same |
| VAE decoder | 9.1s | 4.8s | −47% |
| **TOTAL** | **94.9s** | **72.8s** | **−23%** |

### 40-step run

| Stage | Pre-cleanup (proj.) | Post-cleanup (measured) | Δ |
|---|---|---|---|
| Text encoder (UMT5) | 0.4s | 0.7s | ~same |
| prepare_latents (total) | 25.7s | **8.9s** | **−65%** |
| ↳ VAE encode (ref) | 1.4s | 1.3s | ~same |
| ↳ wav2vec2 + bucketing | 1.6s | 1.7s | ~same |
| ↳ prepare_audio_emb | 4.9s | 4.7s | −5% |
| ↳ VAE encode (motion 73f) | 16.5s | **0.005s** | **M4 — −100%** |
| ↳ prepare_cond_emb | 1.4s | 1.2s | ~same |
| Denoising loop | ~478s | **303.5s** | **−37%** |
| ↳ Audio cross-attn cumulative | ~192s | **31.7s** | **M5 steady-state — −83%** |
| ↳ Block-stack (non-audio) | ~286s | 271.8s | −5% |
| VAE decoder | 9.1s | 5.2s | −42% |
| **TOTAL** | **~513s ≈ 8:30** | **318.3s ≈ 5:20** | **−38%** |

Per-step breakdown: 7.59s/step total (303.5/40); audio injection 0.79s/step;
block-stack 6.80s/step.

**Why M5 looks much better at 40 steps than at 5 steps.** The K/V cache misses once
per inject layer (12 misses on step 1) and hits on every subsequent step. At 5 steps the
12 misses + 48 hits give roughly 4 cached steps' worth of savings; at 40 steps the same
12 misses are amortized over 39 cached steps. Step-1 audio injection ≈ 4.7s (cold; full
projection); steady-state ≈ 0.69s/step (cached; just Q-proj + SDPA + output-proj).

## Open perf gaps

### Block-stack (non-audio) per-step time — 6.80s/step vs T2V's 6s/step target

S2V's per-device sequence length at (2,4) 480p is **8608** (= padded `(N_noisy + N_ref) / sp_factor`)
vs T2V's **8192**. The extra ~5% in token count plus ~8% from untuned matmul shapes (no
`grid_13_10` entries exist in `models/tt_dit/utils/matmul.py` for S2V's shapes — verified by
`get_matmul_config` falling through to the 8x8x8 default with a one-time warning per shape)
accounts for the 13% gap.

Closing this is **shared with the T2V perf team**: tuning matmul block sizes for the 13x10
core grid benefits both T2V (currently also untuned at 8192) and S2V (at 8608). Tracked as a
follow-up since it requires an empirical sweep across the matmul shape table rather than an
S2V-specific code change.

### Audio cross-attn cost — minor at production step count

At 5 steps the K/V projection cache (M5) only nets ~2% because the cold first-step amortizes
poorly over only 4 cached steps. At 40 steps the same 12 K/V projections amortize over 39
cached steps, dropping audio injection from ~192s projected to **31.7s measured (−83%)** —
0.79s/step total, ≈0.69s/step at steady-state. Audio injection is no longer the bottleneck
at production step count.

### `prepare_audio_emb` host roundtrips — ~4.6s clip-level

M6 moved LayerNorms inside `_conv_stage_BTC` onto device, but two roundtrips remain:

- **Local-branch stage 1 head split** — `[B, T, num_heads * head_dim] → [B*num_heads, T, head_dim]`
  is a reshape + permute + reshape; on TILE 5D tensors this requires layout-aware on-device
  reshape+permute. Doable but ~1 day of work for ~0.3s savings; deferred.
- **Global-branch stage 1 LayerNorm** — could use the same on-device pattern as
  `_conv_stage_BTC`; ~0.2s savings; deferred for the same trade-off reason.

Both are listed in `PLAN_WAN_S2V_CLEANUP.md` as future work, not blockers.

## Notes for future measurement

- The 5-step run localizes stage costs well (all clip-level stages are step-independent;
  denoising is linear in step count). The 40-step projection is reliable to within a few
  percent.
- The audio-injection accumulator (`transformer._audio_inject_sec_accum`) is a manual
  `time.perf_counter()` sum because the profiler API can't accumulate same-name buckets.
  Reset by `prepare_audio_emb` on each clip.
- First step typically takes ~3x steady-state (kernel compile / cache warm), so per-step
  numbers reported at 5 steps are slightly inflated vs the 40-step steady-state.
