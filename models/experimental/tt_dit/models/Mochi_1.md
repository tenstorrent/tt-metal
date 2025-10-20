# Mochi-1

## Introduction

[Mochi-1](https://huggingface.co/genmo/mochi-1-preview) is a leading video generation model.

This model is implemented in the TT-DiT library to enable inference on Wormhole LoudBox and Galaxy systems.


## Details

Mochi-1 consists of:
 -a 10B parameter diffusion model with an Asymmetric Diffusion Transformer (AsymmDiT) architecture
 -a 362M parameter video VAE model with an asymmetric encoder-decoder structure that causally compresses videos by 128x, with an 8x8 spatial and 6x temporal compression to a 12-channel latent space
 -a T5-XXL language model to encode text prompts


## Performance

Current performance for two systems are detailed below. Performance is measured in seconds per video, where the video size is 824x480px and 168 frames.

| System    | CFG | SP | TP | Current Performance |
|-----------|-----|----|----|---------------------|
| QuietBox  | 1   | 2  | 4  | 1260s               |
| Galaxy    | 1   | 8  | 4  | ___s                |



## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

To generate mp4 files from the output frames the imageio-ffmpeg pip package is needed.


## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs.
export TT_DIT_CACHE_DIR=/your/cache/path

# Cache the transformer weights. On a LoudBox/QuietBox, use the "2x4sp0tp1" option. On Galaxy, use the "4x8sp1tp0" option.
pytest models/experimental/tt_dit/tests/models/mochi/test_transformer_mochi.py::test_mochi_transformer_model_caching -k "2x4sp0tp1"

# Generate a video with the pipeline test. Use the dit_2x4sp0tp1_vae_1x8sp0tp1 option on 8-chip systems and 4x8sp1tp0 on 32-chip systems.
TT_MM_THROTTLE_PERF=5 pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "dit_2x4sp0tp1_vae_1x8sp0tp1"
```

## Limitations

We currently need to set `TT_MM_THROTTLE_PERF=5` when running inference in order to avoid a certain set of hangs.
As of now, running on systems smaller than a 2x4 Wormhole mesh is not well supported. The model is large and requires 8-chips worth of memory to run.
