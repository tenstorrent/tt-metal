# FLUX.1 Kontext [dev]

## Introduction

[FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is a
12B rectified-flow transformer from Black Forest Labs for **instruction-based image editing**:
given an input (reference) image and a text instruction (e.g. *"Add a hat to the cat"*), it
produces an edited image.

This port reuses the tt-metal FLUX.1 [dev] implementation. Kontext shares the exact same
`FluxTransformer2DModel`, CLIP-L + T5-XXL text encoders, `AutoencoderKL` VAE, and
`FlowMatchEulerDiscreteScheduler` as FLUX.1 [dev]; the only differences are in the pipeline.

## How Kontext differs from FLUX.1 [dev]

All conditioning happens in the pipeline layer — the transformer weights and forward are
unchanged (see [Flux1.md](Flux1.md) for the base architecture):

1. **Reference image → latents.** The input image is VAE-encoded (deterministic mode),
   normalized `(z - shift_factor) * scaling_factor`, and packed into tokens.
2. **RoPE id offset.** Generated (noise) tokens keep position-id channel 0 = `0`; reference
   image tokens set channel 0 = `1`. Row/column coordinates deliberately overlap — the two
   token sets are separated purely by channel 0.
3. **In-context concatenation.** Each denoising step feeds the transformer the concatenated
   sequence `[noise_tokens, image_tokens]` (noise first) with the matching concatenated RoPE.
4. **Output slice.** After the forward, only the noise tokens are kept for the scheduler step;
   the reference-image tokens are discarded.

Default `guidance_scale = 3.5` (guidance-distilled, no CFG), same as [dev].

## Implementation

- Pipeline: `models/tt_dit/pipelines/flux1/pipeline_flux1_kontext.py`
  (`Flux1KontextPipeline`, `Flux1KontextPipelineConfig`)
- Transformer / text encoder / VAE decoder / scheduler: reused unchanged from FLUX.1 [dev].
- Reference-image VAE **encoder**: host-side torch (`diffusers.AutoencoderKL`) in the initial
  port — one encode per request is negligible next to the denoising loop. A future
  optimization can move it on-device (see `models/demos/stable_diffusion_xl_base/vae/tt`).

### Sequence-parallel handling

With sequence parallelism the noise and image sequences are fractured *separately* on the sp
axis and concatenated *on device* so each shard holds `[noise_shard | image_shard]`; slicing
the first `n // sp` tokens per shard then recovers the global noise sequence. For `sp == 1`
this reduces to a plain concat + `[:, :n]` slice. **Bring up `sp == 1` (tp-only) first**, then
enable sp after verifying ring-attention position handling over the combined sequence.

## Prerequisites
- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) + installed TT-Metalium / TT-NN
- HuggingFace access to gated weights:
  1. Grant access at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
  2. `huggingface-cli login`

## How to Run

```bash
export TT_DIT_CACHE_DIR=/your/cache/path
# optional: a reference image (otherwise a synthetic gradient is used)
export KONTEXT_INPUT_IMAGE=/path/to/input.png

# Start with sp=1 (tp only) to validate correctness
pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1_kontext.py -k "1x2sp0tp1"

# 2x4 mesh (QuietBox)
pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1_kontext.py -k "2x4sp0tp1"
```

## Status / roadmap

- [x] Pipeline scaffolding (host VAE encode, id offset, concat/slice, SP-safe layout)
- [ ] Numerical bring-up on Wormhole (`sp=1` tp-only) — PCC vs diffusers reference
- [ ] Sequence-parallel enablement (verify ring attention over combined sequence)
- [x] Preferred-resolution snapping / variable aspect ratio (`PREFERRED_KONTEXT_RESOLUTIONS`)
- [ ] On-device VAE encoder
- [ ] Performance tuning + tracing
- [ ] tt-inference-server registration (`TTFluxKontextRunner`, `/v1/images/edits`)

## Scalability

Same parallelism model as FLUX.1 [dev] (SP + TP; no CFG parallelism). See
[Flux1.md](Flux1.md) for mesh/parallel-config details.
