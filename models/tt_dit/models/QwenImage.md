# Qwen-Image

## Introduction

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is an image generation model developed by Alibaba's Qwen team, with strong general capabilities in image generation and exceptional performance in text rendering, especially for Chinese.


## Details

The architecture is described in the [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324).

The model consists of a Qwen2.5-VL text encoder together with its tokenizer, a scheduler (FlowMatchEulerDiscreteScheduler), a transformer, and a VAE (AutoencoderKLQwenImage). The core component is the transformer, called QwenImage DiT (Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks contain attention layers that operate on both the spatial and prompt embeddings jointly.



## Performance

Current performance for four systems is detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px with 50 inference steps.

| System                  | CFG | SP | TP | Encoding | Denoising (50 steps) | VAE   | Total |
|-------------------------|-----|----|----|----------|----------------------|-------|-------|
| WH QuietBox / T3K (2x4) | 1   | 2  | 4  | 0.55s    | 67.2s                | 0.14s | 68.0s |
| BH QuietBox 2 (2x2)     | 2   | 1  | 2  | 0.10s    | 27.2s                | 0.11s | 27.5s |
| BH LoudBox (2x4)        | 2   | 1  | 4  | –        | –                    | –     | 19.0s |
| WH Galaxy (4x8)         | 2   | 4  | 4  | 0.35s    | 25s                  | 0.4s  | 26s   |


## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)


## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future startup times.
# Note: the first run will populate the cache and will not benefit from faster startup.
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the pipeline test on a 4-chip box such as BH QuietBox 2 (2x2 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_pipeline -k 2x2

# Run the pipeline test on QuietBox (2x4 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_pipeline -k 2x4

# Run the pipeline test on Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_pipeline -k 4x8

# Run the performance test on a 4-chip box such as BH QuietBox 2 (2x2 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k 2x2

# Run the performance test on QuietBox (2x4 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k 2x4

# Run the performance test on Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k 4x8
```


## Image Editing

The same pipeline runs [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit),
[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) and
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), chosen by `checkpoint_name`. The text
encoder sees the input images through the Qwen2.5-VL vision tower, and the transformer sees their latents appended to
the generated image's. Pass `image=` one image, or up to three for Edit-2509 and 2511; the output takes the
supported resolution closest in aspect ratio to the last one. With `mask_image=`, Qwen-Image-Edit inpaints a single image.

Only WH QuietBox / T3K (2x4) has been run so far, at 3.4–4.2 s per step.

```bash
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit -k 2x4
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit_multiple_images -k 2x4
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit_inpaint -k 2x4
```


## Scalability

Qwen-Image has been implemented to support execution on 4-chip (BH QuietBox 2 with 2x2 mesh topology), 8-chip (QuietBox and LoudBox with 2x4 mesh topology) as well as 32-chip (Galaxy with 4x8 mesh topology) systems.
The 2x4 and 4x8 configurations have been tested on Wormhole. The 2x2 configuration targets Blackhole (BH QuietBox 2) and is the configuration exercised in CI; it runs 1024x1024 at 50 steps in ~27.5s end to end.

The DiT model can be parallelized on 3 axes:
1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((1, 0), (2, 0), (4, 1))`. `cfg` is factor 1, so the conditional and unconditional passes run batched on the whole mesh. `sp` is factor 2 on axis 0, and `tp` is factor 4 on axis 1, meaning weights are tensor-fractured across each row of 4 devices.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 with factor 4, and `tp` is on axis 1 with factor 4.

The text encoder (Qwen2.5-VL) and the VAE decoder are parallelized with tensor parallelism. On QuietBox, the encoder and transformer are dynamically loaded (swapped in and out of device memory) due to memory constraints.
