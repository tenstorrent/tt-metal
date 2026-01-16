# Motif

## Introduction

[Motif](https://huggingface.co/Motif-Technologies/Motif-Image-6B-Preview) is a text-to-image generative model.


## Details

The architecture follows a Rectified Flow Diffusion Transformer design inspired by [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of three text encoders (CLIP-L, CLIP-G, and T5-XXL) together with their tokenizers, a scheduler, a transformer, and a VAE. The core component is the transformer, called Motif DiT (Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of 30 transformer blocks. Transformer blocks contain attention layers that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.


## Performance

Current performance and target performance for two systems are detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px with 20 inference steps.

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | 2   | 2  | 2  | 10.5s               | TBD                |
| Galaxy    | 2   | 4  | 4  | 10.5s               | 6.5s               |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the pipeline test on QuietBox (2x4 mesh)
pytest models/experimental/tt_dit/tests/models/motif/test_pipeline_motif.py -k "2x4cfg1sp0tp1 and traced and encoder_device"

# Run the pipeline test on Galaxy (4x8 mesh)
pytest models/experimental/tt_dit/tests/models/motif/test_pipeline_motif.py -k "4x8cfg1sp0tp1 and traced and encoder_device"
```


## Scalability

Motif has been implemented to support execution on 8-chip (QuietBox and LoudBox with 2x4 mesh topology) as well as 32-chip (Galaxy with 4x8 mesh topology) systems.
The model has only been tested on Wormhole. Blackhole support is coming soon.

The DiT model can be parallelized on 3 axes:
1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((2, 1), (2, 0), (2, 1))`. This gives us `cfg` parallelism with factor 2 on axis 1, yielding 2 2x2 submeshes. `sp` is factor 2 on axis 0, meaning that activations are sequence-fractured on the `2x2` submesh on axis 0. `tp` is factor 2 on axis 1, meaning weights are tensor-fractured on the `2x2` submesh on axis 1.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 and `tp` is on axis 1, giving us `sp` factor 4 and `tp` factor 4.

The text embedding models (CLIP-L, CLIP-G, and T5-XXL) and the VAE decoder are parallelized with tensor parallelism on one or both of the cfg submeshes.
