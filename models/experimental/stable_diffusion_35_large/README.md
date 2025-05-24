# Stable Diffusion 3.5

## Introduction

[Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for image synthesis guided by text prompts.

## Details

The architecture is described in the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of two different text encoders together with their tokenizers, a scheduler, a trasformer and a VAE. The core component is the transformer, called MMDiT (Multimodal Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks mainly contain attention layers, that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.

## Implementation Status

- All operations of MMDiT are implemented using `ttnn`.
- Almost all tensors have data type bfloat16 and reside on DRAM.
- The VAE, the scheduler, the text encoders and tokenizers are taken from the `diffusers` library.
  - An update of the `diffusers` library was required.
- The T5 text encoder takes several seconds to encode a prompt on the CPU. It could be ported to `ttnn` to improve performance.

## Running the Tests

The tests are run using the following command:

```sh
pytest models/experimental/stable_diffusion_35_large/tests
```

## Running the Demo

The demo is run using the following command:

```sh
pytest models/experimental/stable_diffusion_35_large/demo.py
```
