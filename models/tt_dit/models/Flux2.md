# FLUX.2

## Introduction

[FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) is Black Forest Labs' image generation
and editing model family. Two variants run here:

- [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev): 32B transformer with a guidance
  embedding, Mistral 3 (24B) text encoder.
- [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): the same transformer
  architecture at 4B (3072 wide, 24 heads, 5 double-stream and 20 single-stream blocks), without the
  guidance embedding, with a Qwen3 text encoder. Distilled to 4 steps.

`Flux2Pipeline` reads the transformer's sizes and the text encoder from the checkpoint, so both run
through the same code: `pipelines/flux2/pipeline_flux2.py`, `models/transformers/transformer_flux2.py`
and `models/vae/vae_flux2.py`.

## Performance

Klein 4B, 4 steps, traced, total time per image:

| Case                      | T3K (2x4 WH) | QuietBox 2 (2x2 BH) | Galaxy (4x8 BH) |
| ------------------------- | ------------ | ------------------- | --------------- |
| 1024x1024, no input image | 1.59 s\*     | 0.86 s\*            | 0.34 s\*        |
| 1024x1024, 1 input image  | 2.84 s       | 1.40 s\*            | 0.61 s\*        |
| 2048x2048, no input image | 5.68 s       | 3.35 s              | 1.15 s          |
| 2048x2048, 1 input image  | 6.86 s       | 4.17 s\*            | 1.30 s          |

\* Mean of runs 2 and 3, with a slow first run dropped. The other cells are the mean of all three
runs, which had no outlier.

## Sizes and editing

`resolutions` (by default 1024x1024 and 2048x2048) are chosen per call with `width` and `height`.
Every size is warmed up untraced, with and without a reference image, before the first trace is
captured, and each has its own denoising and VAE traces.

`images=[image]` edits one reference image. It is center-cropped to the output's aspect ratio,
resized to the output size capped at one megapixel, and encoded by `Flux2VaeEncoder`. Each device
runs its latents shard followed by its reference shard; the reference tokens' position ids have time
10, as in diffusers.

The initial noise is drawn as diffusers draws it, so the same seed gives the same starting latents.

## Parallelism

`Flux2Pipeline.create_pipeline` takes the parallel layout from a preset per mesh shape, for any
argument left as `None`:

| Mesh                | Transformer | VAE       |
| ------------------- | ----------- | --------- |
| WH 1x8              | SP 1, TP 8  | TP 8      |
| WH 2x4 (T3K)        | SP 4, TP 2  | TP 2, H 4 |
| BH 2x2 (QuietBox 2) | SP 2, TP 2  | TP 2, H 2 |
| BH 2x4 (LoudBox)    | SP 2, TP 4  | TP 4, H 2 |
| BH 4x8 (Galaxy)     | SP 4, TP 8  | TP 4, H 8 |

The prompt is sharded across the SP devices (`shard_prompt=True`). FLUX.2-dev on a 2x2 mesh needs
`is_fsdp` and `dynamic_load`; with `dynamic_load`, reloading a component releases the traces that
depend on it, so they are captured again on the next traced call.

## Trace region

The device's `trace_region_size` must hold all traces at once: for Klein's default resolutions, four
denoising traces and the VAE traces. They take 42.2 MB on a BH QuietBox 2 and fit in 31 MB on a T3K.
The Klein tests reserve 50 MB (`line_params_flux2_klein`), which is enough on a BH Galaxy.

## Tests

In `tests/models/flux2/`:

- `test_transformer_flux2.py`, `test_vae_flux2.py`, `test_prompt_encoder.py`: against diffusers and
  transformers. `test_vae_flux2_encoder` covers the VAE encoder.
- `test_pipeline_flux2.py::test_pipeline`: one image per prompt, FLUX.2-dev and Klein.
- `test_pipeline_flux2.py::test_pipeline_sizes_and_editing`: Klein at every default resolution, with
  and without a reference image; every traced image must match its untraced one.
- `test_performance_flux2.py`: timing per stage.
