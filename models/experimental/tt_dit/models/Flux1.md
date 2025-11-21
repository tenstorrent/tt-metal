# Flux 1

## Introduction

[Flux](https://blackforestlabs.ai/flux) is a state-of-the-art generative model for text-to-image synthesis.

This version of Flux 1 is tuned for inference performance, achieving competitive results on Wormhole Galaxy Systems.

## Details

The architecture is described in the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of two different text encoders together with their tokenizers (CLIP-L and T5-XXL), a scheduler, a transformer, and a VAE. The core component is the transformer, called Flux DiT (Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks contain attention layers that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.

Unlike traditional diffusion models, Flux uses learned guidance embeddings rather than classifier-free guidance (CFG). This means the model doesn't require running conditional and unconditional paths separately.

## Performance

Current performance and target performance for two systems are detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px.

Flux comes in two variants:
- **schnell**: Fast variant optimized for speed (4 inference steps)
- **dev**: Development variant with higher quality (28 inference steps)

> **Note**: Flux.1 doesn't use traditional CFG; guidance is handled via learned guidance embeddings. The CFG column in the table reflects mesh allocation only (always 1 for Flux).

### Schnell Variant (4 steps)

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | 1   | 2  | 4  | 4.70s               | 2.00s              |
| Galaxy    | 1   | 4  | 8  | 2.40s               | 1.00s              |

### Dev Variant (28 steps)

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | 1   | 2  | 4  | 21.60s              | 6.50s              |
| Galaxy    | 1   | 4  | 8  | 8.00s               | 6.50s              |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

1. Visit [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) to grant access to the model weights
2. Login with the HuggingFace token: `huggingface-cli login`

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the schnell variant on QuietBox (2x4 mesh)
pytest models/experimental/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "schnell and 2x4sp0tp1 and traced and encoder_device"

# Run the schnell variant on Galaxy (4x8 mesh)
pytest models/experimental/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "schnell and 4x8sp0tp1 and traced and encoder_device"

# Run the dev variant on QuietBox (2x4 mesh)
pytest models/experimental/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "dev and 2x4sp0tp1 and traced and encoder_device"

# Run the dev variant on Galaxy (4x8 mesh)
pytest models/experimental/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "dev and 4x8sp0tp1 and traced and encoder_device"
```

## Scalability

Flux1 has been implemented to support execution on 4-chip (1x4 mesh), 8-chip (QuietBox and LoudBox with 2x4 mesh topology), as well as 32-chip (Galaxy with 4x8 mesh topology) systems.
The model has only been tested on Wormhole. Blackhole support is coming soon.

The DiT model can be parallelized on 2 main axes (note: no CFG parallelism for Flux):
1. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
2. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((2, 0), (4, 1))`. This gives us `sp` parallelism with factor 2 on axis 0, and `tp` factor 4 on axis 1, meaning weights are tensor-fractured on the mesh on axis 1.

Another example parallel config on a 4x8 mesh is `((4, 0), (8, 1))`. `sp` is on axis 0 with factor 4, and `tp` is on axis 1 with factor 8.

The text embedding models (CLIP-L and T5-XXL) and the VAE decoder are parallelized with tensor parallelism.

## Model Variants

### Flux.1 [schnell]
- Optimized for speed with only 4 inference steps
- Suitable for rapid prototyping and real-time applications
- Generates high-quality 1024x1024 images in under 5 seconds

### Flux.1 [dev]
- Higher quality output with 28 inference steps
- Better for final production-quality images
- More detailed and refined results

Both variants use the same architecture but are trained with different timestep samplers and optimizations.
