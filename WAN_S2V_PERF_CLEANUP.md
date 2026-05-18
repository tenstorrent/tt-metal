# WAN 2.2 S2V Perf Cleanup

## Goal
Drive WAN 2.2 S2V on BH-LB (2x4, 480p) toward production-grade per-second-of-video latency.

## Baseline timeline

| Stage | Pre-cleanup (337.0 s) | After Item 2 (303.7 s) | Δ |
|---|---|---|---|
| Text encoder (UMT5) | 0.5 s | 1.3 s | +0.8 |
| **prepare_latents** | **38.5 s** | **5.0 s** | **-33.5** |
|   ↳ VAE encode (ref) | 1.4 s | 1.3 s | -0.1 |
|   ↳ **wav2vec2 + bucketing** | **37.0 s** | **3.7 s** | **-33.3** |
| Sum denoise (4×5 steps) | 234.8 s | 235.0 s | +0.2 |
| Sum VAE decode | 18.3 s | 17.4 s | -0.9 |
| Sum VAE motion-encode | 20.9 s | 20.9 s | 0 |
| **TOTAL** | **337.0 s** | **303.7 s** | **-33.3** |

Repro: `pytest models/tt_dit/tests/models/wan2_2/test_performance_wan_s2v.py::test_s2v_pipeline_performance[blackhole-clips4-steps5-resolution_480p-bh_2x4sp1tp0]` with `S2V_AUDIO=warm_weird_22s.mp3 S2V_REF_IMAGE=girl-hoodie-glasses.png`.

## Landed
- [x] Encoder conv3d sweep (12/13 shapes + 1 DEFAULT) — `_BLOCKINGS` in `models/tt_dit/utils/conv3d.py`.
- [x] **Item 2 — wav2vec2 feature-extractor chunking with on-device trim+concat** (`models/tt_dit/encoders/wav2vec2/model_wav2vec2.py`). Slice + concat happen via `ttnn` on the per-chunk TILE-layout output; no host roundtrip per chunk. **Measured: -33 s on the 4-clip × 5-step × 22 s-audio run.**
- [x] Item 4 (partial) — eliminated wasted `padd_lat` zero-alloc in `FramePackMotionerWan.forward` (~5 ms × 4 clips = 20 ms saved).

## Priority queue

### 1. DiT denoise matmul tuning (~70 % of runtime, biggest win)
- 11.7 s/step × 20 steps = 235 s. Audio cross-attn = 10 % of denoise; **block stack = 90 %**.
- Untuned `(M, K, N)` matmuls on the 13×10 BH 2x4 grid log "No known best blocking… using default 8×8×8". Observed M=6240, 9184, etc.
- **Approach**: enumerate the (M, K, N) values that hit the warning during a 5-step run (grep `get_matmul_config` warnings); register optimal blockings in `models/tt_dit/utils/matmul.py`.
- **Estimated impact**: 2-4× speedup on block stack → **100-150 s saved per 4-clip run**.

### 2. VAE motion-encode trace mode
- Each per-clip motion encode is ~6.5 s × 3 active clips = 20 s; estimated 1-2 s/clip is program build/dispatch overhead.
- **Approach**: wrap encoder forward in `ttnn.Trace`, reuse program across clips (encoder shapes are constant per pipeline instance).
- **Estimated impact**: **3-6 s saved per run**.

### 3. wav2vec2 transformer per-clip cost (4.6 s × 4 = 18 s)
- `prep_audio` is ~4.6 s per clip. Likely the wav2vec2 24-layer transformer (1100 features × 1024 dim, bf16). Has its own matmul-blocking warnings.
- **Approach**: sweep `_BLOCKINGS_MATMUL` for wav2vec2 transformer shapes; potentially share QKV projection state across clips since audio is the same input.
- **Estimated impact**: 1-2 s/clip × 4 clips = **4-8 s saved per run**.

### 4. MotionEncoder on-device conv3d patchify (deferred — needs sweep)
- Three `WanPatchEmbed` projections in `FramePackMotionerWan` still patchify on host (CPU reshape+permute), then upload, then matmul. Could replace with `ttnn.experimental.conv3d` (stride=kernel=patch_size).
- **Blocker**: the three shapes `(16, 5120, (1,2,2))`, `(16, 5120, (2,4,4))`, `(16, 5120, (4,8,8))` aren't in `_BLOCKINGS` or `_DEFAULT_BLOCKINGS` → fallback to `(in_c=16→32, 32, 1, 1, 1)` which is much slower than the current host-patchify + matmul.
- **Verdict**: not worth pursuing without a sweep first. Direct conv3d-fallback path would regress ~0.5-2 s/clip.
- **Approach if pursued**: add 3 shapes to `bruteforce_conv3d_sweep.py`, sweep, then port `WanPatchEmbed` to a conv3d-native path. Sweep ~5-10 min device time.
- **Estimated impact**: 30-50 ms/clip × 4 = ~150 ms saved (small).

### 5. Two wav2vec2 conv3d shapes still `[NONE]`
- `(1024, 5120, (3,1,1))` and `(1024, 1280, (3,1,1))` in `_BLOCKINGS`. Fall back to hardcoded `(in_c, 32, 1, 1, 1)`.
- **Approach**: extend `bruteforce_conv3d_sweep.py` with a wav2vec2 sweep variant; parse winners.
- **Estimated impact**: **~1-2 s saved** on wav2vec2 transformer per run.

### 6. `LAT_TARGET=20` binary_ng workaround (precision, not throughput)
- Ship with `LAT_TARGET=21` (one extra latent frame); reference uses 20. The ttnn `binary_ng` op asserts at `Sq=8224` (257 tiles).
- **Approach**: file ttnn ticket OR host-pad operands in `prepare_cond_emb` before upload to dodge the broadcast classifier.
- **Impact**: lip-sync precision, not wall-clock.

### 7. Encoder `compute_encoder_dims.T_tconv` convention cleanup (NOT a bug, doc-only)
- `T_tconv = cur_T` but the actual conv input is `cur_T + 1` (cache concat prepends 1 frame). Lookup convention is internally consistent (sweep + runtime both use cur_T as the key); the comment is misleading but the blocking is correct.
- **Approach**: rename to `T_tconv_lookup_key` or add a comment documenting the cache-concat semantics.
- **Impact**: 0. Future-proofing only.

## Open Questions
- [ ] Block-stack matmul sweep: does M vary across clips, or is it constant? Check before launching the sweep.
- [ ] Can wav2vec2 transformer reuse K/V cache across clips? Audio is identical for all clips; only the position window changes. K/V values are position-dependent (RoPE-ish in wav2vec2's conv_pos_embed), so direct reuse may not work — needs investigation.
- [ ] Trace-mode for motion encoder: do shapes change between clip 0 (initial zero motion) and clips 1+ (real motion latents)? If yes, two traces needed.

## State
- [x] Encoder conv3d sweep (12/13 layers).
- [x] Item 2: wav2vec2 chunking on-device trim+concat (-33 s measured).
- [x] Item 4a: padd_lat zero-alloc cleanup (cosmetic ~20 ms).
- [ ] Item 1: DiT denoise matmul tuning.
- [ ] Item 2.5 / Item 5: wav2vec2 transformer matmul + conv3d tuning.
- [ ] Item 4b: motioner on-device conv3d (deferred until sweep).
- [ ] Item 6: LAT_TARGET=20 ttnn workaround.
- [ ] Item 7: T_tconv comment cleanup.
