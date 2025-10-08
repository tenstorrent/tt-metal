# Stable Diffusion 3.5 Large

## Introduction

[Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for image synthesis guided by text prompts.

This version of Stable Diffusion 3.5 Large is tuned for inference performance, achieving competitive performance on Wormhole Galaxy Systems.


## Details

The architecture is described in the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).


The model consists of two different text encoders together with their tokenizers, a scheduler, a transformer and a VAE. The core component is the transformer, called MMDiT (Multimodal Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks mainly contain attention layers, that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.


## Performance

Current performance and target performance for two systems are detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px.

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | 2   | 2  | 2  | 12.2s               | 14.4s              |
| Galaxy    | 2   | 4  | 4  | 5.6s               | 3.6s               |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

1. Visit [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) to grant access to the model weights
2. Login with the HuggingFace token: `huggingface-cli login`

Finally, run the demo
```
pytest models/experimental/stable_diffusion_35_large/fun_demo.py -k "t3k_cfg2_sp2_tp2 and yes_trace"

# On Galaxy
TT_MM_THROTTLE_PERF=5 pytest models/experimental/stable_diffusion_35_large/fun_demo.py -k "tg_cfg2_sp4_tp4 and yes_trace"
```


## Scalability

SD3.5-Large has been implemented to support execution on 8-chip (LoudBox and QuietBox) as well as 32-chip (Galaxy) systems.
The model has only been tested on Wormhole. Blackhole support is coming soon.

The DiT model can be parallelized on 3 axes:
1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((2, 1), (2, 0), (2, 1))`. This gives us `cfg` parallelism with factor 2 on axis 1, yielding 2 2x2 submeshes. `sp` is factor 2 on axis 0, meaning that activations are sequence-fractured on the `2x2` submesh on axis 0. `tp` is factor 2 on axis 1, meaning weights are tensor-fractured on the `2x2` submesh on axis 1.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 and `tp` is on axis 1, giving us `sp` factor 4 and `tp` factor 4.

The text embedding models and the VAE decoder are parallelized with tensor parallelism on one or both of the cfg submeshes.
