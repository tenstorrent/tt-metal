# FIBO

## Introduction

[FIBO](https://huggingface.co/briaai/FIBO) is Bria's text-to-image model. It is trained on
structured JSON captions rather than free text, so a generation starts by expanding the user's
prompt into that JSON with [FIBO-vlm](https://huggingface.co/briaai/FIBO-vlm), a Qwen3-VL
derivative. The pipeline therefore runs four models: the VLM, the SmolLM3 text encoder, the
transformer, and the VAE decoder.

The VLM can also start from an image. Given an image alone, it describes the image as a structured
prompt. Given an image with a prompt, it treats the prompt as instructions for how to change the
image.

[FIBO Edit](https://huggingface.co/briaai/fibo-edit) is FIBO's image editing variant. It edits an
image according to a structured prompt with an `edit_instruction` field, which
[FIBO-edit-vlm](https://huggingface.co/briaai/FIBO-edit-vlm) writes from the image and editing
instructions. The image is encoded by the VAE, and its tokens are appended to the transformer's
sequence, which doubles its length.

## Details

- Transformer: `models/tt_dit/models/transformers/transformer_fibo.py`
- VAE decoder and encoder: `models/tt_dit/models/vae/vae_wan_2d.py` (the Wan 2.2 VAE on a single frame)
- Prompt expansion: `models/tt_dit/pipelines/fibo/vlm.py`
- Pipeline: `models/tt_dit/pipelines/fibo/pipeline_fibo.py`

## Performance

1024x1024, 30 denoising steps, guidance scale 5.0 with CFG on (two transformer forwards per step),
traced, starting from a natural-language prompt. Each figure is the median of 8 generations after 2
untimed ones, as `models/tt_dit/tests/models/fibo/test_performance_fibo.py` reports them.

Prompt expansion generates a different number of tokens on each system and its runtime is
proportional to that count, so the figures here are normalized to 600 tokens using the measured time
per token.

| System                     | CFG | SP  | TP  | Image   |
| -------------------------- | --- | --- | --- | ------- |
| Galaxy (4x8 Blackhole)     | 2   | 4   | 4   | 10.63 s |
| QuietBox 2 (2x2 Blackhole) | 2   | 1   | 2   | 24.63 s |
| T3000 (2x4 Wormhole)       | 2   | 2   | 2   | 36.63 s |

Per stage, in seconds:

| Stage                | Galaxy                | QuietBox 2            | T3000                  |
| -------------------- | --------------------- | --------------------- | ---------------------- |
| VLM (600 tokens)     | 6.27 (10.46 ms/token) | 7.58 (12.63 ms/token) | 10.56 (17.60 ms/token) |
| encoder (SmolLM3)    | 0.43                  | 0.18                  | 0.26                   |
| prepare              | 0.09                  | 0.09                  | 0.09                   |
| denoising (30 steps) | 3.78 (7.93 it/s)      | 16.50 (1.82 it/s)     | 25.36 (1.18 it/s)      |
| VAE                  | 0.06                  | 0.28                  | 0.36                   |
| **total**            | **10.63**             | **24.63**             | **36.63**              |

## How to Run

```bash
# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/path/to/cache

# Generate images from structured JSON prompts
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_pipeline

# Generate images from natural-language prompts, expanded into JSON by the VLM
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_pipeline_vlm

# Generate images from an image, alone or with editing instructions, turned into JSON by the VLM
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_pipeline_vlm_image

# Edit an image with FIBO Edit, from editing instructions turned into JSON by the edit VLM
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_edit_pipeline_vlm
```
