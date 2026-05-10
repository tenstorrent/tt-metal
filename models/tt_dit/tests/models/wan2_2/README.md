# Wan2.2 — running tests and generating videos

End-to-end image-to-video pipelines for the Wan2.2 family on Blackhole Galaxy.

This directory contains:

| File | Purpose |
|---|---|
| `test_pipeline_wan.py` | Base Wan2.2 T2V inference |
| `test_pipeline_wan_i2v.py` | Base Wan2.2 I2V inference |
| `test_pipeline_wan_distill_i2v.py` | Wan2.2-Distill (lightx2v 4-step) I2V inference |
| `test_pipeline_anisora.py` | Index-AniSora V3.2 I2V inference |
| `test_pipeline_lora.py` | LoRA-on-base inference |
| `test_performance_wan.py` | Per-stage perf tests for all of the above |
| `test_transformer_wan.py`, `test_attention_wan.py`, `test_vae_wan2_1.py`, `test_rope.py`, … | Sub-module unit tests |

The instructions below cover the **inference** and **perf** entry points. Sub-module tests are run the same way; pick the test with `pytest <path> -v -k <id>`.

## Hardware

Canonical config: **Blackhole Galaxy, 4×8 ring** (`bh_4x8sp1tp0_ring` for distill / AniSora, `ring_bh_4x8_sp1tp0` for the base perf test — yes, the parametrize ids use different orderings; pytest `-k` filters are not interchangeable across tests).

Smaller meshes (2×2, 2×4) are available in the parametrize lists but treat them as dev-only.

## One-time setup

```bash
cd $TT_METAL_HOME                                # this is the tt-metal checkout
source python_env/bin/activate

export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Persistent caches on a big disk (any path with ~100 GB free).
# Skip the export and the defaults under ~/.cache work, but they're tight on /.
export HF_HOME=/storage/sdawle/huggingface
export TT_DIT_CACHE_DIR=/storage/sdawle/tt_dit_cache

# Allow HuggingFace downloads on first run for each weight file. Once cached,
# you can drop this to enforce no-network.
export TT_DIT_ALLOW_HF_DOWNLOAD=1

# Skip the interactive prompt loop in inference tests (use the hardcoded /
# env-var prompt and exit after one generation).
export NO_PROMPT=1
```

Approx download sizes (first run):
- Base Wan2.2-I2V-A14B-Diffusers: ~70 GB (text encoder + tokenizer + VAE + two ~28 GB transformer subfolders).
- lightx2v distill safetensors: 57 GB (two 28.6 GB BF16 experts).
- After both runs once, total HF cache ≈ 130 GB.

---

## Generate a new video — base Wan2.2 I2V (40 steps, CFG)

```bash
# Pick a seed image + prompt.
export PROMPT_IMAGE=$TT_METAL_HOME/racing.jpg
export PROMPT="A race car speeds along the track, kicking up dust, cinematic motion blur, dynamic camera following the action"

pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance \
  -v -k "i2v and ring_bh_4x8_sp1tp0 and resolution_480p" \
  --timeout 1800 -s
# Output: ./wan_output_video_i2v.mp4   (832×480, 81 frames @ 16 fps, ~54 s on BH 4×8 ring)

# Rename before the next run so it isn't overwritten:
mv wan_output_video_i2v.mp4 wan_base_$(date +%Y%m%d_%H%M%S).mp4
```

| Knob | How |
|---|---|
| Resolution | swap `resolution_480p` → `resolution_720p` (output ≈ 147 s) |
| Prompt | edit `PROMPT="..."` |
| Seed image | edit `PROMPT_IMAGE=/path/to/img.{png,jpg}` (any aspect — pipeline resizes) |

---

## Generate a new video — Wan2.2-Distill (lightx2v 4-step, no CFG)

```bash
export PROMPT_IMAGE=$TT_METAL_HOME/racing.jpg
export PROMPT="A race car speeds along the track, kicking up dust, cinematic motion blur, dynamic camera following the action"

pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_distill_i2v.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p and not random_weights" \
  --timeout 1500 -s
# Output: ./wan_distill_i2v_832x480_0.mp4   (~8 s on BH 4×8 ring with warm cache)
```

For 720p use `resolution_720p`. Output file gets the resolution in its name (`wan_distill_i2v_<W>x<H>_0.mp4`), so distill outputs don't overwrite each other.

### Smoke test without HF downloads (random weights)

If you want to validate compile / mesh / shape / CCL plumbing without pulling the 57 GB lightx2v safetensors:

```bash
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_distill_i2v.py::test_pipeline_inference_random_weights \
  -v --timeout 1500 -s
```

Output frames are garbage by design.

---

## Generate a new video — Index-AniSora V3.2 I2V (anime domain, 40 steps)

Anime-domain Wan-derived model: subclass of `WanPipelineI2V` that swaps in `IndexTeam/Index-anisora` V3.2 weights. Same architecture, same parameter count, anime-style training data. Uses 40 inference steps by default but `NUM_STEPS` is configurable.

```bash
export PROMPT_IMAGE=$TT_METAL_HOME/some_anime_frame.png
export PROMPT="An anime girl smiling, soft lighting, cinematic"
# Optional: shorten for faster preview (16 or 8 is common for quick iteration)
export NUM_STEPS=40

pytest models/tt_dit/tests/models/wan2_2/test_pipeline_anisora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p and not random_weights" \
  --timeout 1800 -s
# Output: ./wan_anisora_i2v_832x480_0.mp4
```

| Knob | How |
|---|---|
| Inference steps | `export NUM_STEPS=16` (or 8 for very fast previews; quality drops) |
| Weights from a custom location (skip HF cache) | `export ANISORA_LOCAL_DIR=/path/to/weights` containing `high_noise_model/diffusion_pytorch_model.safetensors` and `low_noise_model/diffusion_pytorch_model.safetensors` |
| Random-weights smoke test | `pytest models/tt_dit/tests/models/wan2_2/test_pipeline_anisora.py::test_pipeline_inference_random_weights -v --timeout 1500 -s` |

The AniSora weights are at `IndexTeam/Index-anisora` on HF (V3.2 subfolder, ~28 GB BF16 per expert). First-time download adds ~57 GB on top of the base diffusers cache.

`boundary_ratio` for AniSora is **0.9** (vs 0.5 for distill, 0.5 for base) — most of the trajectory uses the high-noise expert, with the low-noise expert kicking in only at the end. This is hardcoded in `AniSoraPipeline.ANISORA_BOUNDARY_RATIO` and not overridden by env vars.

---

## Per-stage performance test

Same code path as inference but wraps the pipeline in `BenchmarkProfiler` and prints encoder / image-encode / denoising / VAE-decode / total timings, plus runs an explicit warmup iteration so the timed run is steady-state.

```bash
# Distill perf (warm cache → ~70 s total per resolution)
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance_distill \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" \
  --timeout 1500 -s

# Base i2v perf (~55 s @ 480p, ~150 s @ 720p, no warmup on 4×8 ring → first-run includes compile)
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance \
  -v -k "i2v and ring_bh_4x8_sp1tp0 and resolution_480p" \
  --timeout 1800 -s

# AniSora perf (40-step anime, similar wall-time to base)
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance_anisora \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" \
  --timeout 1800 -s
```

The base perf test exports a video to the cwd (`wan_output_video_i2v.mp4`). Distill / AniSora perf tests do not currently export video — use the corresponding inference test if you want frames.

---

## Reference: per-stage perf on BH Galaxy 4×8 Ring (2026-05-08)

81 frames, single timed iteration:

| Stage | Base 480p (40 steps, CFG) | Base 720p (40 steps, CFG) | Distill 480p (4 steps, no CFG) | Distill 720p (4 steps, no CFG) |
|---|---:|---:|---:|---:|
| Text encoding (UMT5) | 0.136 s | 0.133 s | 0.134 s | 0.138 s |
| Image encoding (CLIP+VAE encode) | 5.020 s | 13.024 s | 4.949 s | 12.577 s |
| Denoising | 48.806 s | 133.243 s | 2.436 s | 6.553 s |
| VAE decoding | 0.426 s | 0.711 s | 0.424 s | 0.746 s |
| **Total** | **54.4 s** | **147.1 s** | **7.96 s** | **20.0 s** |
| Speedup vs base | — | — | 6.84× | 7.34× |

Per-forward denoise cost is identical between base and distill at each resolution; distill's win is "10× fewer steps × 2× from no-CFG" with kernel performance unchanged.

---

## Common gotchas

- **Output filename collisions.** Base test always writes `wan_output_video_i2v.mp4` regardless of resolution — rename between runs or the 720p clobbers the 480p. Distill inference test includes resolution in the filename and doesn't collide.
- **Different test-id naming.** Base test uses `ring_bh_4x8_sp1tp0`; distill / AniSora use `bh_4x8sp1tp0_ring`. Pytest `-k` filters cannot be reused across tests.
- **No warmup on the base perf test for 4×8 ring.** `if traced:` guard means base 4×8 numbers include first-time kernel compile (~3–5 s extra). Distill perf has an explicit warmup pass.
- **Encoder perf gate fails harmlessly.** Base perf test asserts `encoder < 0.1 s`, measured is ~0.13–0.14 s. The pytest reports FAIL but all other numbers (denoising, total, etc.) are valid.
- **`prompt_image.png` is the default seed.** If you don't set `PROMPT_IMAGE`, the inference tests open `./prompt_image.png` from the cwd. A 832×480 placeholder ships in the repo; the perf test falls back to a procedurally-generated fractal.
- **`imageio_ffmpeg` is required for MP4 export.** If it's missing the test logs "Could not export video" and silently skips. Install with `python -m pip install imageio-ffmpeg` (in the venv).
- **Don't `git push`** from this checkout without explicit instruction. Local commits and `git add` are fine.

---

## Files of interest

| Path | What's there |
|---|---|
| `models/tt_dit/pipelines/wan/pipeline_wan.py` | Base T2V pipeline (parent of i2v / distill / anisora subclasses) |
| `models/tt_dit/pipelines/wan/pipeline_wan_i2v.py` | I2V subclass — the pattern to follow |
| `models/tt_dit/pipelines/wan/pipeline_wan_distill.py` | Distill subclass: lightx2v weight load + CFG-off override |
| `models/tt_dit/pipelines/wan/pipeline_anisora.py` | AniSora V3.2 subclass: anime-domain weight load, `boundary_ratio=0.9` |
| `models/tt_dit/utils/lightx2v_loader.py` | lightx2v ↔ diffusers key remap (reused by AniSora) |
| `models/tt_dit/models/Wan2_2.md` | Wan2.2 model card |
| `models/tt_dit/models/Wan2_2_Distill.md` | Wan2.2-Distill model card |
