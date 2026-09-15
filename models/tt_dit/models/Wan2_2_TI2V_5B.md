# Wan2.2-TI2V-5B

## Introduction

[Wan2.2](https://huggingface.co/Wan-AI) is a state-of-the-art open-weights video generative model. TI2V-5B is its high-compression dense variant: a single 5B transformer that natively serves **both** text-to-video (T2V) and image-to-video (I2V) from one checkpoint, targeting 720P at 24 fps.

This model is implemented in the TT-DiT library to enable inference on Blackhole multi-chip systems.

## Details

The architecture is described in the paper [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314).

The model consists of a text encoder ([UMT5-XXL](https://huggingface.co/google/umt5-xxl)), a scheduler, a single WanTransformer3DModel transformer, and the Wan2.2-VAE. The transformer has 30 blocks of self-attention, cross-attention, and feedforward layers, with RoPE positional embeddings.

Unlike Wan2.2-A14B, TI2V-5B is **dense, not Mixture-of-Experts** — there is one transformer and no high-noise/low-noise split (`boundary_ratio=None`), so a single guidance scale applies. Its Wan2.2-VAE compresses `4x16x16` and an additional patchify layer takes the total to `4x32x32`, which is what allows a 5B model to serve 720P.

| | Wan2.2-A14B | Wan2.2-TI2V-5B |
|---|---|---|
| Experts | 2 (MoE, `transformer` + `transformer_2`) | 1 dense (`boundary_ratio=None`) |
| dim / ffn / layers / heads | 5120 / 13824 / 40 / 40 | 3072 / 14336 / 30 / 24 |
| VAE | Wan2.1, 16-channel | Wan2.2-VAE `4x16x16`, **48-channel** |
| Native resolution | 480p and 720p | 720P (`1280x704` or `704x1280`) |
| Conditioning (I2V) | channel-concat + CLIP image embeds | latent pinning + per-token timestep mask |

## Performance

Measured on a single Blackhole Galaxy (4x8), 81 frames and 40 denoising steps, warm-traced. Performance is total seconds per video.

### 720p (1280x704)

| Mode | System       | Arch | SP | TP | Text enc | Image enc | Denoise | VAE dec | **Total** |
|------|--------------|------|----|----|----------|-----------|---------|---------|-----------|
| T2V  | Galaxy (4x8) | BH   | 8  | 4  | 0.089s   | —         | 12.045s | 4.632s  | **16.78s** |
| I2V  | Galaxy (4x8) | BH   | 8  | 4  | 0.090s   | 1.379s    | 12.728s | 4.649s  | **18.86s** |

### 480p (832x480)

| Mode | System       | Arch | SP | TP | **Total** |
|------|--------------|------|----|----|-----------|
| T2V  | Galaxy (4x8) | BH   | 8  | 4  | **8.87s** |

> 480p is **out of distribution** for this checkpoint. The `4x32x32` total compression leaves only 15x26 tokens at 832x480 versus 22x40 at 1280x704, and output is visibly soft. Run quality and correctness work at 720p.

I2V costs **+2.08s** over T2V. That is 1.379s of host-side VAE encode of the conditioning frame plus 0.68s of per-token AdaLN in the denoise loop; text encode and VAE decode are identical to T2V, so the T2V conv3d blockings carry over with no I2V-specific work.

Performance work still open:
- `_register_5b_matmul_tables` in `pipeline_wan_ti2v_5b.py` registers M values the model never requests, so every transformer matmul falls back to the default `8x8x8` blocking (see Limitations)
- Residual VAE encoder on device, which would remove most of the 1.379s host image encode
- `flow_shift` is currently forced to the A14B value (see Limitations)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/path/to/cache

# Text-to-Video (T2V)

# Run T2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py -k "generate and bh_4x8"

# Image-to-Video (I2V)

# Run I2V on Blackhole Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b_i2v.py -k "generate"

# Animate your own image and write an mp4 plus first/mid/last PNG previews
I2V_IMAGE=/path/to/photo.jpg I2V_PROMPT="..." I2V_OUT=/path/to/out.mp4 \
  pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b_i2v.py -k demo

# Performance. Note the T2V gate must be selected by node id: "-k performance_ti2v_5b"
# also matches the I2V test, because that name contains the T2V one as a substring.
pytest "models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance_ti2v_5b"
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k performance_ti2v_5b_i2v
```

Demo knobs: `I2V_STEPS`, `I2V_FRAMES`, `I2V_SEED`, `I2V_GUIDANCE`, `I2V_FLOW_SHIFT`, `I2V_FPS`.
Omit `I2V_IMAGE` for a synthetic fractal seed. `PYTHONPATH` must include the user site
(`$(python3 -m site --user-site)`) or mp4 export silently degrades to PNGs only.

## Scalability

TI2V-5B has been brought up on:

**Blackhole:**
- 32-chip (Galaxy with 4x8 mesh topology)

The DiT model is parallelized on 2 axes, as for A14B:
1. `sp` (sequence parallel) — the input sequence is fractured across a mesh axis, factor 8 on axis 1.
2. `tp` (tensor parallel) — weights are fractured across a mesh axis, factor 4 on axis 0.

There is **no data parallelism and no CFG parallelism**: batch size is 1 and classifier-free guidance runs as two sequential forwards per step, so all 32 chips are consumed by `sp x tp`. The text encoder (UMT5) uses tensor parallelism; the VAE uses height/width spatial parallelism across the mesh.

Sequence parallelism pads the token count to a multiple of `TILE_SIZE * sp`, so per-device `M = N/8` — 2336 at 1280x704 with 81 frames, 3424 with 121 frames.

## Model Variants

### Text-to-Video (T2V)
- Generates video from a text prompt and an optional negative prompt
- Uses [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) checkpoint
- `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b.py`
- Default: 81 frames, 40 denoising steps

### Image-to-Video (I2V)
- Generates video conditioned on a single input image and a text prompt
- Same checkpoint as T2V — TI2V-5B serves both from one set of weights
- `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b_i2v.py`
- **Conditioning differs fundamentally from A14B I2V.** The checkpoint has `in_channels == out_channels == 48` and `image_dim: null` and ships no image encoder, so there is no channel-concat and no CLIP embedding. Instead latent frame 0 is **pinned** to the conditioning latent at every step and once more after the denoise loop, while its tokens see timestep 0 and all other tokens see `t`. The reference is diffusers `WanImageToVideoPipeline` with `expand_timesteps=True`.
- The conditioning image is encoded on the **host** with the torch `AutoencoderKLWan`; the device VAE encoder does not support the Wan2.2 residual encoder (`WanEncoder3D` asserts `not is_residual`). `WAN5B_I2V_ENCODE_COMPILE=0` disables the `torch.compile` wrapper, `WAN5B_I2V_ENCODE_FP32=1` forces the bit-faithful fp32 path.
- Only `frame_pos=0` is representable; conditioning on a later frame is rejected rather than silently ignored.

## Prompting I2V

This matters more than it does for T2V. TI2V-5B pins **only latent frame 0**, so nothing constrains the rest of the clip and the model will follow a prompt away from the seed image. A prompt asking for a subject that is not in the seed reads as an instruction to change the scene.

The model card's own I2V example prompt is purely **descriptive of the seed image** plus motion. Following that pattern keeps the subject intact for the whole clip.

Controlled A/B, identical image / seed / 40 steps / 1280x704, prompt the only variable:

| prompt style | `mean_frame_delta` | final frame |
|---|---|---|
| introduces a new subject ("…add sharks") | 19.62 | subject gone, scene replaced |
| descriptive of the seed + motion | 16.29 | subject preserved, coherent motion |

So apparent "melting" or subject loss in an I2V clip is usually prompt-driven drift, not a pipeline defect. Check the prompt style before investigating the implementation.

## Validation

| check | result |
|---|---|
| Conditioning math vs diffusers reference | 17 exact-equality tests |
| Transformer vs torch, scalar timestep | PCC 99.9893 % |
| Transformer vs torch, per-token timestep | PCC 99.9894 % |
| Two-row timestep == scalar timestep | PCC 100.0000 % (bit-exact) |
| E2E decoded frame 0 vs seed image | PCC 0.9984 |
| T2V CLIP correctness gate | 40.45 (threshold 36.00) |

T2V is unaffected by the I2V work: 3 runs of the pre-I2V commit versus 3 runs after, at 720p, give total 16.9072s -> 16.8848s, **-0.13 %** (0.13 sigma, ranges fully overlapping). The measurement noise floor on this system is ~1.8 % on total, so single-sample comparisons below that are not resolvable — use 3 repeats.

## Limitations

1. **`_register_5b_matmul_tables` never hits.** It registers M values `(2368, 2720, 3488, 9472, 13952)`, but the model requests 1024 at 480p and 2336 at 720p; `N=64` is copied from A14B where TI2V-5B `proj_out` emits `48*2*2 = 192`; and only `"11x10"` is registered when four of five shapes are looked up on `"12x9"`. Every transformer matmul therefore runs on the default `8x8x8`. Measured for ff1 `(2336, 3072, 3584)`: 401.2 us default versus 351.3 us swept (12.4 %), but only ~2-3 % of denoise end to end.
2. **`flow_shift` is the A14B value.** `pipeline_wan.py` passes `flow_shift=12.0` to the scheduler, overriding this checkpoint's own `scheduler_config.json` value of **5.0**.
3. **The CLIP gate cannot detect quality regressions.** A visibly soft 480p render scored 37.24 while a crisp 720p render scored 36.63 — CLIP measures semantic alignment, not fidelity, and the threshold is calibrated to one prompt at one resolution.
4. **The per-token timestep path is L1-sensitive.** The overflow reported during bring-up is the **timestep MLP**, not the AdaLN. Sharding the per-token timestep on the `sp` axis brings M from 18688 to 2336, which is necessary but not sufficient. The pipeline instead embeds only the two distinct timestep values (0 on the conditioned frame, `t` elsewhere) and expands through a mask — mathematically identical because the embedder is pointwise in the token axis, runs at M=32 like the scalar path, and needs no blocking entries at any resolution.
5. **True 720p (1280x720) is unsupported**; use `1280x704`, since `720/16 = 45` is odd and breaks the `patch_size=2` patchify.
6. Residual VAE **encoder** (`AvgDown3D` / `WanResidualDownBlock`) is not implemented on device; only the decoder was ported.

---

## Appendix: original bring-up notes

Kept verbatim from the manual bring-up on a single BH Galaxy.

> Parallel track to `tt_hw_planner auto-up` on `/home/ttuser/tt-metal-hw-planner`.
>
> **Target**
> - Hardware: DC16-2-BG-0105-u20-48, 32-chip BH Galaxy, mesh 4x8
> - Checkpoint: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
> - Same Galaxy preset as 14B: SP=8 axis1, TP=4 axis0, Ring, FSDP off, 11x10 matmul grid
>
> **Code landed in this clone**
> 1. `model_type=ti2v` allows `in_channels=48`
> 2. `WanCheckpoint.build` now passes `num_layers` from HF config (30 vs hardcoded 40)
> 3. Pipeline skips `transformer_2` when `boundary_ratio is None`
> 4. Variant: `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b.py`
> 5. Test: `models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py`
>
> **VAE (landed)** — Wan2.2-VAE residual decoder path in `vae_wan2_1.py`:
> - `WanResidualUpBlock` + `WanDupUp3D` shortcut (`is_residual=True`)
> - `decoder_base_dim` from HF (256, not encoder `base_dim=160`)
> - `patch_size=2` host `unpatchify` after decode; decoder planned at H/2 x W/2
> - `first_chunk` threaded (full-T and cached frame 0)
>
> **Still TODO (manual)**
> - Conv3d blocking sweep for 5B VAE channel counts (160/256/512/1024/48)
> - Sweep 5B 11x10 matmul / fused-MMRS (first-cut tables registered in pipeline_wan_ti2v_5b.py)
> - ~~I2V path~~ **done** — see the I2V sections above
> - Residual VAE encoder (`AvgDown3D` / `WanResidualDownBlock`) — I2V encodes the conditioning
>   frame on the host instead; worth ~1.4 s if ported
> - Smoke run: `WAN5B_SMOKE=1 pytest ... -k bh_4x8` once ttnn is built on this tree
