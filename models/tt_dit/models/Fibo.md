# FIBO

## Introduction

[FIBO](https://huggingface.co/briaai/FIBO) is Bria's text-to-image model. It is trained on
structured JSON captions rather than free text, so a generation starts by expanding the user's
prompt into that JSON with [FIBO-vlm](https://huggingface.co/briaai/FIBO-vlm), a Qwen3-VL
derivative. The pipeline therefore runs four models: the VLM, the SmolLM3 text encoder, the
transformer, and the VAE decoder.

## Details

- Transformer: `models/tt_dit/models/transformers/transformer_fibo.py`
- VAE decoder: `models/tt_dit/models/vae/vae_fibo.py`
- Prompt expansion: `models/tt_dit/pipelines/fibo/vlm.py`
- Pipeline: `models/tt_dit/pipelines/fibo/pipeline_fibo.py`

## Performance

1024x1024, 30 denoising steps, guidance scale 5.0 with CFG on (two transformer forwards per step),
traced, starting from a natural-language prompt. Each figure is the median of 8 generations after 2
untimed ones, as `models/tt_dit/tests/models/fibo/test_performance_fibo.py` reports them.

| System                     | CFG | SP  | TP  | Image   | Throughput      |
| -------------------------- | --- | --- | --- | ------- | --------------- |
| Galaxy (4x8 Blackhole)     | 2   | 4   | 4   | 10.35 s | 0.0966 images/s |
| QuietBox 2 (2x2 Blackhole) | 2   | 1   | 2   | 25.16 s | 0.0397 images/s |
| T3000 (2x4 Wormhole)       | 2   | 2   | 2   | 41.16 s | 0.0243 images/s |

Per stage, in seconds:

| Stage                  | Galaxy           | QuietBox 2        | T3000             |
| ---------------------- | ---------------- | ----------------- | ----------------- |
| vlm (prompt expansion) | 5.81             | 8.44              | 13.69             |
| encoder (SmolLM3)      | 0.59             | 0.22              | 0.66              |
| prepare                | 0.08             | 0.07              | 0.19              |
| denoising (30 steps)   | 3.81 (7.87 it/s) | 16.15 (1.86 it/s) | 26.15 (1.15 it/s) |
| vae                    | 0.07             | 0.28              | 0.39              |
| **total**              | **10.35**        | **25.16**         | **41.16**         |

## How to Run

```bash
# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/path/to/cache

# Generate images from structured JSON prompts
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_pipeline

# Generate images from natural-language prompts, expanded into JSON by the VLM
pytest models/tt_dit/tests/models/fibo/test_pipeline_fibo.py::test_fibo_pipeline_vlm
```
