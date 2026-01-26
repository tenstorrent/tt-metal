# CLIP + IP-Adapter Resampler Codegen

This directory contains generated code for the CLIP Vision Encoder + IP-Adapter Plus Resampler pipeline, compiled for Tenstorrent hardware.

## Model Components

### CLIP Vision Encoder
- **Model**: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- **Architecture**: ViT-H/14 (Vision Transformer Huge with 14x14 patches)
- **Source**: [HuggingFace](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)

### IP-Adapter Plus Resampler
- **Weights**: `ip-adapter-plus_sdxl_vit-h.bin`
- **Source**: `h94/IP-Adapter` (subfolder: `sdxl_models`)
- **Architecture**: Perceiver Resampler from IP-Adapter Plus for SDXL
- **Source**: [HuggingFace](https://huggingface.co/h94/IP-Adapter)

## Pipeline

The combined module (`CLIPResamplerModule`) performs:
1. CLIP Vision Encoder processes input image `[batch, 3, 224, 224]`
2. Extracts penultimate hidden layer `[batch, 257, 1280]`
3. Resampler produces IP-Adapter tokens `[batch, 16, 2048]`

## Compile Options

- `optimization_level`: 1
- `codegen_try_recover_structure`: True
- `dtype`: bfloat16
