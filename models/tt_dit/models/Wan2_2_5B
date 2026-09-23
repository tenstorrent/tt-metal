# Wan2.2 TI2V-5B

## Introduction

[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) is a 5B dense video generation
model that handles text-to-video and image-to-video in a single checkpoint. Unlike the 27B MoE
members of the Wan2.2 family, it is dense and 720P-only, trading parameter count for a
high-compression VAE. Apache-2.0.

This model is implemented in the TT-DiT library to enable inference on Tenstorrent Blackhole
Galaxy systems, in both T2V and I2V modes.

## Details

Wan2.2 TI2V-5B consists of:

- a 5B dense diffusion transformer — 24 heads x head_dim 128 (3072 hidden), ffn_dim 14336,
  patch size (1,2,2), 48 latent input channels
- Wan2.2-VAE, a video VAE with a T x H x W compression ratio of 4 x 16 x 16; with the
  transformer's patchification layer the total compression reaches 4 x 32 x 32
- umT5-XXL as the text encoder (4096-dim conditioning). There is no CLIP image encoder

I2V conditions by **pinning latent frame 0**: the seed image is VAE-encoded on host, written into
latent frame 0, and re-pinned after every solver step, with a per-token timestep giving frame-0
tokens t=0 and all other tokens the current t. This is architecturally unlike the 14B I2V, which
concatenates the conditioning frame onto the input channels and uses CLIP.

| Mode | Resolution | Frames | Latent T x H x W | Tokens | M per device (SP=8) |
|---|---|---|---|---|---|
| 720p | 1280x704 | 81 | 21 x 44 x 80 | 18480 | 2336 |
| 480p | 832x480 | 81 | 21 x 30 x 52 | 8190 | 1024 |

## Performance

Measured on one Blackhole Galaxy (4x8), SP=8 / TP=4, Ring, FSDP off. 40 steps, warm-traced, mean
of 3 invocations; denoise and total spread <= 0.8%.

| Mode | Resolution |TP|  Total |
|------|------------|-|-------|
| T2V | 1280x704 (720p)| 4| **11.77s** |
| I2V | 1280x704 (720p)| 4| **13.96s** |
| T2V | 832x480 (480p) |  4| **6.34s** |


Against the initial bring-up, on a different host: 720p T2V 16.78s -> 11.77s (-29.8%), 720p I2V
18.86s -> 13.96s (-26.0%), 480p T2V 8.87s -> 6.34s (-28.5%). VAE decode 4.632s -> 0.959s (-79.3%)
and denoise 12.045s -> 10.701s (-11.2%). Cross-host, so read these as directional.

Denoise is ~91% of the run and is compute-bound: ring SDPA ~28%, fused ff2 matmul ~27%, AGMM ~17%,
norms ~11%. VAE decode is now dominated by conv3d (42% of 1.25s device total).

## Accuracy

| gate | result |
|---|---|
| transformer vs torch, scalar timestep | PCC **99.9893%** |
| transformer vs torch, per-token timestep | PCC **99.9894%** |
| per-token (2-row) vs scalar path | PCC **100.0000%** |
| VAE chunked decode | PCC **1.0**, max_abs_diff 0.0 |
| VAE upsample rewrite | bit-exact, max_abs_diff == 0.0 at production shapes |
| I2V conditioning math vs reference | 20/20 exact |
| I2V end-to-end, frame 0 vs seed image | PCC **0.9984** |
| CLIP prompt similarity, 121f 720p | mean 40.38 (gate 36.00) |

bf8 weights (opt-in, `WAN5B_QUANT_CONFIG=all_weights_bf8`) hold PCC at 99.9885% with CLIP 40.20 and
no visible degradation across 121 frames; bf8 perf is not yet measured.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `imageio-ffmpeg` to write mp4 files from the output frames

## How to Run

```bash
# Set the directory to cache the weights to speed up future runs.
export TT_DIT_CACHE_DIR=/your/cache/path

# T2V / I2V performance, 720p and 480p, traced
pytest models/tt_dit/tests/models/wan/test_pipeline_performance_ti2v_5b.py --timeout=0
pytest models/tt_dit/tests/models/wan/test_pipeline_wan_ti2v_5b_i2v.py --timeout=0

# Generate an mp4
pytest models/tt_dit/tests/models/wan/test_pipeline_ti2v_5b.py -k generate --timeout=0
```

`pytest.ini` pins a 300s timeout, so long device runs need `--timeout=0`. All 32 chips are claimed
by every run; check the box is free first.

I2V prompting matters more than for T2V, since only frame 0 is pinned. Describe the seed image plus
the motion you want; asking for content absent from the seed will change the scene.

## Limitations

- 720P only, at 1280x704 or 704x1280. 832x480 runs but is out of distribution for this checkpoint
  and looks soft.
- Frame counts must be 4k+1, following the VAE's temporal compression of 4.
- Validated on a 4x8 Blackhole Galaxy only; smaller meshes are untried.
- 121 frames has one unrepeated latency measurement and no tuned matmul blockings.
- I2V at 480p is unmeasured.
- `flow_shift` currently carries the 14B value (12.0) rather than this checkpoint's 5.0, so
  quality comparisons against upstream are not valid until that is aligned.
