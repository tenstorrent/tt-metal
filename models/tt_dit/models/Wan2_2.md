# Wan2.2-A14B

## Introduction

[Wan2.2](https://huggingface.co/Wan-AI) is a state-of-the-art open-weights video generative model supporting both text-to-video (T2V) and image-to-video (I2V) generation.

This model is implemented in the TT-DiT library to enable inference on Wormhole and Blackhole multi-chip systems.

## Details

The architecture is described in the paper [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314).

The model consists of a text encoder ([UMT5-XXL](https://huggingface.co/google/umt5-xxl)), a scheduler, two WanTransformer3DModel transformers, and a VAE. Each transformer has 40 blocks of self-attention, cross-attention, and feedforward layers, with RoPE positional embeddings.

Wan2.2 uses a Mixture-of-Experts (MoE) architecture with two-stage denoising. A high-noise expert handles the first 87.5% of timesteps and a low-noise expert handles the remainder, giving 27B total parameters with 14B active per step. Classifier-free guidance (CFG) is used with separate guidance scales per stage.

## Performance

Current T2V (text-to-video) performance for supported systems is detailed below. Performance is measured in total seconds per video, with 81 frames and 40 denoising steps.

### 480p (832x480)

| System           | Arch | SP | TP | Current Performance |
|------------------|------|----|----|---------------------|
| Loud Box (2x4)   | WH   | 2  | 4  | 735s                |
| Quiet Box (2x2)  | BH   | 2  | 2  | 466s                |
| Loud Box (2x4)   | BH   | 4  | 2  | 207s                |

### 720p (1280x720)

| System           | Arch | SP | TP | Current Performance |
|------------------|------|----|----|---------------------|
| Galaxy (4x8)     | WH   | 8  | 4  | 354s                |
| Galaxy (4x8)     | BH   | 8  | 4  | 168s                |

Performance work is ongoing to improve these numbers:
- increased matmul utilization
- increased SDPA utilization
- overlapped AllGather-Matmul and Matmul-ReduceScatter
- fused binary ops
- overlapped weight AllGather with compute

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Text-to-Video (T2V)

# Run T2V on Blackhole Quiet Box (2x2 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x2sp0tp1"

# Run T2V on Wormhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x4sp0tp1"

# Run T2V on Blackhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "bh_2x4sp1tp0"

# Run T2V on Wormhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "wh_4x8sp1tp0"

# Run T2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "bh_4x8sp1tp0"

# Image-to-Video (I2V)

# Run I2V on Blackhole Quiet Box (2x2 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "2x2sp0tp1"

# Run I2V on Wormhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "2x4sp0tp1"

# Run I2V on Blackhole Loud Box (2x4 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "bh_2x4sp1tp0"

# Run I2V on Wormhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "wh_4x8sp1tp0"

# Run I2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py -k "bh_4x8sp1tp0"
```

## Scalability

Wan2.2 has been implemented to support execution on the following systems:

**Wormhole:**
- 8-chip (Loud Box with 2x4 mesh topology)
- 32-chip (Galaxy with 4x8 mesh topology)

**Blackhole:**
- 4-chip (Quiet Box with 2x2 mesh topology)
- 8-chip (Loud Box with 2x4 mesh topology)
- 32-chip (Galaxy with 4x8 mesh topology)

The DiT model can be parallelized on 2 main axes:
1. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. Self-attention is implemented with ring attention, overlapping KV all-gather with computation.
2. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

A parallel config is defined by the mesh shape and the axis assignments for `sp` and `tp`. For example, on a 2x4 mesh with `sp_axis=0, tp_axis=1`, we get `sp` parallelism with factor 2 on axis 0 and `tp` factor 4 on axis 1. On a 4x8 mesh with `sp_axis=1, tp_axis=0`, we get `sp` factor 8 on axis 1 and `tp` factor 4 on axis 0.

The text encoder (UMT5) is parallelized with tensor parallelism. The VAE is parallelized with height/width spatial parallelism across the mesh.

## Model Variants

### Text-to-Video (T2V)
- Generates video from a text prompt and an optional negative prompt
- Uses [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) checkpoint
- Supports 480p (832x480) and 720p (1280x720) resolutions
- Default: 81 frames, 40 denoising steps

### Image-to-Video (I2V)
- Generates video conditioned on one or more input images and a text prompt
- Uses [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) checkpoint
- Input images are encoded through the VAE encoder and concatenated with the latent noise
- Supports the same resolutions and frame counts as T2V

Both variants use the same MoE two-stage denoising architecture with separate high-noise and low-noise expert transformers.

## Step caching (DBCache)

The pipeline runs [cache-dit](https://github.com/vipshop/cache-dit) style **Dual Block Cache** step
skipping **by default** (pass `cache_config=None` to `WanPipelineConfig.default` or to a call to disable it),
ported to TT-NN in `models/tt_dit/utils/dbcache.py` (decision logic) and
`models/tt_dit/pipelines/wan/dbcache.py` (device buffers / tracers). On every denoising step the first
`Fn` blocks are computed and the change they make to the hidden state is compared with the previous
computed step (relative mean-L1). If the change is below `residual_diff_threshold`, the remaining
blocks are skipped and the residual they produced on the last computed step is re-applied instead.
Each expert (high-noise / low-noise) and each CFG branch (conditional / unconditional) is cached
independently, mirroring cache-dit's `has_separate_cfg=True` dual-transformer setup.

```python
from models.tt_dit.pipelines.wan.dbcache import WanDBCacheConfig
from models.tt_dit.utils.dbcache import DBCacheConfig

# Default (nothing to pass): cache-dit's Wan 2.2 preset, F1B0, threshold 0.08, <= 2 consecutive
# cached steps, high-noise expert: 4 warmup steps / max 8 cached, low-noise expert: 2 warmup /
# max 20 cached. See `WanDBCacheConfig.default` for the rationale.
frames = pipeline(prompts=[...], num_inference_steps=40)

# Uncached (the original single-trace path):
frames = pipeline(prompts=[...], num_inference_steps=40, cache_config=None)

# Conservative preset (recommended with flow_shift=12):
frames = pipeline(prompts=[...], num_inference_steps=40, flow_shift=12.0,
                  cache_config=WanDBCacheConfig.default(residual_diff_threshold=0.05))

pipeline.cache_summary()  # cached steps and residual diffs per expert / branch for the last call
```

The pipeline's scheduler defaults to `flow_shift=5.0` (cache-dit's and vLLM's 720p setting) rather than the
official Wan 2.2 A14B T2V value of 12.0, because the flatter low-noise tail is what makes 15-16 of 40 steps
cacheable (measurements below). `flow_shift=12.0` is available per call. An uncached pipeline can be built with
`WanPipelineConfig.default(..., cache_config=None)`.

A/B test (same seed, baseline vs. split-without-caching vs. DBCache, with PSNR against the baseline):

```bash
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_dbcache.py -k "bh_4x8sp1tp0nl2_ring"
```

Measured on Blackhole Galaxy (4x8, sp=8, tp=4, ring), 832x480, 81 frames, 40 steps, one prompt, seed 0:

| Run | Denoising | Cached steps (of 40, per CFG branch) | PSNR vs. baseline | PCC vs. baseline |
|---|---|---|---|---|
| baseline (no cache) | 45.3s | 0 | - | - |
| split path, caching disabled | 46.2s | 0 | inf (bit-exact) | 1.0000 |
| DBCache default preset (threshold 0.05) | 37.2s (1.22x) | 8 | 23.3 dB | 0.974 |
| DBCache, cache-dit's threshold 0.08 | 32.7s (1.38x) | 12 | 15.7 dB | 0.849 |

Same mesh at 1280x720 (81 frames, 40 steps): baseline 142.4s, split path bit-exact (PCC 1.0), default preset
129.9s (1.10x) with 4 cached steps per branch, PSNR 26.7 dB / PCC 0.991. At 720p most high-noise residual
diffs sit just above the 0.05 threshold (0.05-0.07), so fewer steps cache than at 480p. Threshold sweep at 720p
(`cache_config=WanDBCacheConfig.default(residual_diff_threshold=...)`):

| Threshold | Denoising | Cached steps per branch | PSNR vs. baseline | PCC vs. baseline |
|---|---|---|---|---|
| 0.05 (default) | 129.9s (1.10x) | 4 | 26.7 dB | 0.991 |
| 0.06 | 110.3s (1.30x) | 10 | 18.5 dB | 0.943 |
| 0.07 | 105.4s (1.36x) | 11 | 12.3 dB | 0.748 |
| 0.08 | 101.9s (1.40x) | 12 | 12.3 dB | 0.748 |
| vLLM recipe: F8, 0.12, warmup 4, no cap, 3 consecutive | 119.3s (1.20x) | 9 | 23.0 dB | 0.980 |

The cliff between 0.06 and 0.07 is about *when* caching starts rather than how many steps cache: at 0.07 the
high-noise expert starts caching at step 5, at 0.06 at step 10. Steps before ~10 change the trajectory
strongly, so 0.06 is the recommended F1 setting for 720p. The vLLM-Omni Wan 2.2 recipe (F8 / 0.12, see
`recipes.vllm.ai`) sits between the two at 720p: its 8-block residual is a smoother signal, so it starts caching
later (step 13) and keeps PCC 0.98 at 1.20x, but each cached step still computes 8 of 40 blocks.

For comparison, a vLLM-Omni style recipe (`Fn_compute_blocks=8`, threshold 0.12, warmup 4, no cached-step cap,
3 consecutive) at 480p gives 1.22x with PCC 0.944 / 20.1 dB, i.e. the same speed as the default preset at lower
fidelity: computing 8 of 40 blocks on every cached step costs 20% of a step, so it needs 10 cached steps per
branch (starting at step 9) to match what F1 reaches with 8. Thresholds are not comparable across `Fn` values
(the residual after 8 blocks is roughly twice as large, so 0.08 never caches with F8). TaylorSeer order 1 on
top of it: 1.20x, PCC 0.934.

**The schedule decides how much can be cached.** With `flow_shift=12` (the official Wan 2.2 A14B T2V setting,
the pipeline's previous default), the schedule puts 26 of 40 steps on the high-noise expert and compresses the low-noise expert
into the fast-changing end of the trajectory, where its residuals are never stable enough to cache. cache-dit's
own Wan 2.2 example runs `flow_shift=3` (480p) / `5` (720p). Re-running 480p with those schedules
(`pipeline(..., flow_shift=...)`, `WAN_DBCACHE_FLOW_SHIFT` in the A/B test), 40 steps, same prompt/seed:

| flow_shift | Config | Denoising | Cached steps per branch (high / low) | PSNR vs. baseline | PCC vs. baseline |
|---|---|---|---|---|---|
| 12 (ours) | F1, 0.08 | 32.7s (1.38x) | 8 / 4 | 15.7 dB | 0.849 |
| 5 | F1, 0.05 (default) | 39.4s (1.15x) | 1 / 5 | 31.6 dB | 0.995 |
| 5 | F1, 0.08 | 28.2s (1.60x) | 6 / 10 | 18.6 dB | 0.909 |
| 3 | F1, 0.05 (default) | 39.8s (1.14x) | 0 / 6 | 35.6 dB | 0.998 |
| 3 | F1, 0.08 | 28.4s (1.61x) | 4 / 12 | 21.5 dB | 0.952 |
| 5, **720p** | F1, 0.08 | 93.8s vs 142.9s (1.52x) | 6 / 9 | 17.7 dB | 0.894 |

End-to-end wall clock per video on BH Galaxy 4x8 with the defaults (`flow_shift=5`, threshold 0.08): 480p
29s (uncached 46s), 720p 95s (uncached 144s). Model load from the weight cache (~3 min) and mp4 export excluded.

With cache-dit's schedule and threshold the low-noise expert caches 10 to 12 steps (including consecutive
pairs) and the pipeline reaches 1.6x while staying near or above PCC 0.9, i.e. the GPU-class result. These
are now the pipeline defaults; note that `flow_shift=5` changes the generated video relative to the official
`flow_shift=12` schedule independently of caching.

The traced path (`traced=True`, three traces per expert) makes the same cache decisions and produces the same
video as the untraced path (PCC 0.974 vs. the untraced baseline); the split-without-caching traced run is
bit-exact as well.

At threshold 0.08 the video is still clean and coherent but follows a different trajectory (different
choreography); at 0.05 it is visually the same video as the uncached run. TaylorSeer forecasting
(`taylorseer_order=1/2`) did not improve agreement with the baseline in this setup and is off by default.
The cached-step pattern is alternating (compute, cache, compute, ...) because the residual diff is measured
against the last *computed* step. Multi-host (4x32) meshes are not supported yet: the residual-diff readback
needs a host-side all-reduce.

## Limitations

While output videos look good, we have many items of work in progress to improve correctness.
Performance optimization is in progress.
