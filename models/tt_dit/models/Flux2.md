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

Klein 4B, 4 steps, traced, total time per image, median of five runs after two unmeasured ones:

| Size      | Input images | T3K (2x4 WH) | QuietBox 2 (2x2 BH) | LoudBox (2x4 BH) | Galaxy (4x8 BH) |
| --------- | ------------ | ------------ | ------------------- | ---------------- | --------------- |
| 1024x1024 | 0            | 1.55 s       | 0.86 s              | 0.68 s           | 0.32 s          |
| 1024x1024 | 1            | 2.55 s       | 1.38 s              | 1.10 s           | 0.50 s          |
| 1024x1024 | 2            |              | 2.06 s              | 1.49 s           | 0.67 s          |
| 1024x1024 | 3            |              | 2.73 s              | 1.85 s           | 0.82 s          |
| 1024x1024 | 4            |              | 3.52 s              | 2.44 s           | 0.97 s          |
| 1024x1024 | 5            |              | 4.48 s              | 2.92 s           | 1.15 s          |
| 1024x1024 | 6            |              | 5.46 s              | 3.62 s           | 1.30 s          |
| 1024x1024 | 7            |              | 6.53 s              | 4.21 s           | 1.47 s          |
| 1024x1024 | 8            |              | 7.75 s              | 5.01 s           | 1.86 s          |
| 1024x1024 | 9            |              | 9.04 s              | 5.59 s           | 2.06 s          |
| 1024x1024 | 10           |              | 10.43 s             | 6.52 s           | 2.26 s          |
| 2048x2048 | 0            | 5.65 s       | 3.41 s              | 2.27 s           | 1.07 s          |
| 2048x2048 | 1            | 6.84 s       | 4.21 s              | 2.89 s           | 1.22 s          |
| 2048x2048 | 2            |              | 5.16 s              | 3.39 s           | 1.39 s          |
| 2048x2048 | 3            |              | 6.16 s              | 4.07 s           | 1.55 s          |
| 2048x2048 | 4            |              | 7.24 s              | 4.64 s           | 1.71 s          |
| 2048x2048 | 5            |              | 8.45 s              | 5.42 s           | 2.13 s          |
| 2048x2048 | 6            |              | 9.77 s              | 6.05 s           | 2.31 s          |
| 2048x2048 | 7            |              | 11.18 s             | 6.91 s           | 2.52 s          |
| 2048x2048 | 8            |              | out of DRAM         | 7.61 s           | 2.70 s          |
| 2048x2048 | 9            |              |                     | 8.50 s           | 2.88 s          |
| 2048x2048 | 10           |              |                     | 9.31 s           | 3.12 s          |

## Sizes and editing

`resolutions` (by default 1024x1024 and 2048x2048) are chosen per call with `width` and `height`.
Every size is warmed up untraced, with and without a reference image, before the first trace is
captured, and each has its own denoising and VAE traces.

`images=[image, ...]` edits with one or more reference images. Each is center-cropped to the
output's aspect ratio and resized to the output size capped at one megapixel. Each number of
reference images gets its own denoising trace and compiles on its first call.

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
