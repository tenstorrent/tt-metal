# Stable Diffusion 3.5 Medium

## Introduction

[Stable Diffusion 3.5 Medium](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for text-guided image synthesis.
The Medium version is a lighter and faster variant of SD3.5, tuned for performance on Wormhole systems using TT-Metal.

## Details

The architecture follows the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of a single text encoder with its tokenizer, a scheduler, a compact MMDiT transformer and a VAE.
The MMDiT is built from spatial, prompt and time embeddings together with a stack of transformer blocks.
Attention layers operate either on the spatial embedding alone or jointly on the spatial and prompt embeddings.

Compared to SD3.5-Large, the Medium variant is smaller, faster, and requires less memory while preserving the same core architecture.

## Performance

Performance is measured in seconds per 1024×1024 image.

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | _   | _  | _  | ~_._s               | ~_._s              |
| Galaxy    | _   | _  | _  | ~_._s               | ~_._s              |


## Prerequisites

- Clone the [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Install TT-Metalium™ / TT-NN™
  → https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md
- Request access to SD3.5-Medium on HuggingFace

## How to Run

1. Visit the model page on HuggingFace:
   https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

2. Login using HuggingFace token:

```bash
huggingface-cli login
