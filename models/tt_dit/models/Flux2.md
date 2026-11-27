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

Klein 4B, 4 steps, traced, total time per image, median of three runs after the trace capture:

| Size      | Input images | T3K (2x4 WH) | QuietBox 2 (2x2 BH) | LoudBox (2x4 BH) | Galaxy (4x8 BH) |
| --------- | ------------ | ------------ | ------------------- | ---------------- | --------------- |
| 1024x1024 | 0            | 1.61 s       | 0.88 s              | 0.69 s           | 0.33 s          |
| 1024x1024 | 1            | 2.93 s       | 1.42 s              | 1.18 s           | 0.59 s          |
| 1024x1024 | 2            |              | 2.11 s              | 1.61 s           | 0.77 s          |
| 1024x1024 | 3            |              | 2.78 s              | 2.06 s           | 0.91 s          |
| 1024x1024 | 4            |              | 3.59 s              | 2.70 s           | 1.19 s          |
| 1024x1024 | 5            |              | 4.61 s              | 3.21 s           | 1.49 s          |
| 1024x1024 | 6            |              | 5.59 s              | 4.06 s           | 1.61 s          |
| 1024x1024 | 7            |              | 6.70 s              | 4.61 s           | 1.66 s          |
| 1024x1024 | 8            |              | 7.92 s              | 5.41 s           | 2.11 s          |
| 1024x1024 | 9            |              | 9.18 s              | 6.10 s           | 2.52 s          |
| 1024x1024 | 10           |              | 10.54 s             | 7.00 s           | 2.75 s          |
| 2048x2048 | 0            | 5.67 s       | 3.44 s              | 2.33 s           | 1.07 s          |
| 2048x2048 | 1            | 6.87 s       | 4.25 s              | 2.96 s           | 1.31 s          |
| 2048x2048 | 2            |              | 5.21 s              | 3.48 s           | 1.55 s          |
| 2048x2048 | 3            |              | 6.25 s              | 4.23 s           | 1.63 s          |
| 2048x2048 | 4            |              | 7.34 s              | 4.86 s           | 1.99 s          |
| 2048x2048 | 5            |              | 8.57 s              | 5.65 s           | 2.30 s          |
| 2048x2048 | 6            |              | 9.85 s              | 6.42 s           | 2.62 s          |
| 2048x2048 | 7            |              | 11.35 s             | 7.40 s           | 2.76 s          |
| 2048x2048 | 8            |              | out of DRAM         | 8.12 s           | 3.04 s          |
| 2048x2048 | 9            |              |                     | 9.06 s           | 3.15 s          |
| 2048x2048 | 10           |              |                     | 10.10 s          | 3.43 s          |

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
