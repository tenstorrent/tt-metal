# Qwen-Image

## Introduction

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is an image generation model developed by Alibaba's Qwen team, with strong general capabilities in image generation and exceptional performance in text rendering, especially for Chinese.

## Details

The architecture is described in the [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324).

The model consists of a Qwen2.5-VL text encoder together with its tokenizer, a scheduler (FlowMatchEulerDiscreteScheduler), a transformer, and a VAE (AutoencoderKLQwenImage). The core component is the transformer, called QwenImage DiT (Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks contain attention layers that operate on both the spatial and prompt embeddings jointly.

## Performance

Total time per image, with classifier-free guidance, at each model card's step count. Each time is
the mean of 3 runs. Text-to-image and [inpainting](#inpainting) run Qwen-Image with the prompt from
the Qwen-Image-2512 model card (142 tokens, which the pipeline pads to its 256-token sequence length
bucket). [Editing](#image-editing) runs Edit-2511. Each cell is the time without → with
[DBCache](#dbcache).

| Case                           | Steps | T3K (2x4)     | BH QuietBox 2 (2x2) | BH LoudBox (2x4) | BH Galaxy (4x8) |
| ------------------------------ | ----- | ------------- | ------------------- | ---------------- | --------------- |
| 1024x1024                      | 50    | 64.9s → 30.8s | 26.0s → 12.6s       | 19.0s → 9.2s     | 8.0s → 4.1s     |
| 1344x1344                      | 50    | -             | 46.2s → 22.1s       | 30.3s → 14.7s    | 10.8s → 5.4s    |
| 1664x928                       | 50    | -             | 39.7s → 19.8s       | 27.1s → 13.7s    | 9.9s → 5.1s     |
| inpaint 1024x1024              | 50    | 65.0s → 32.2s | 26.6s → 13.3s       | 19.1s → 9.4s     | 8.1s → 4.0s     |
| edit 1024x1024, 1 input image  | 40    | -             | 57.2s → 31.1s       | 37.1s → 20.4s    | 12.7s → 7.1s    |
| edit 1024x1024, 2 input images | 40    | -             | 90.8s → 49.1s       | 56.8s → 30.9s    | 19.9s → 11.2s   |
| edit 1024x1024, 3 input images | 40    | -             | 129.4s → 69.7s      | 77.6s → 42.1s    | 25.8s → 14.1s   |

Almost all of the time is the transformer's denoising steps. The stages of 1024x1024 text-to-image:

| System              | Encoding | Denoising | VAE   | Total |
| ------------------- | -------- | --------- | ----- | ----- |
| T3K (2x4)           | 0.41s    | 66.4s     | 0.12s | 67.0s |
| BH QuietBox 2 (2x2) | 0.10s    | 25.9s     | 0.11s | 26.1s |
| BH LoudBox (2x4)    | 0.10s    | 18.8s     | 0.08s | 19.0s |
| BH Galaxy (4x8)     | 0.10s    | 7.8s      | 0.05s | 8.0s  |

T3K only runs 1024x1024. The aspect ratio of an edit hardly matters, since every edit resolution has
about the same number of pixels, and an edit with a mask takes as long as one without. Edit and
Edit-2509 take as long as Edit-2511.
The parallelism of each system is listed under [Scalability](#scalability).

The DBCache times use cache-dit's Qwen-Image setting, F1B0 with threshold 0.12, 8 warm-up steps and
at most 3 skipped steps in a row. It skips 26-27 of 50 steps for text-to-image and 19 of 40 for
edits, the same on every system. Against the same image without it, the PSNR is 26-29 dB for
text-to-image and 21-23 dB for edits. Edits with DBCache run out of DRAM on T3K.

## Resolutions

The pipeline generates 1024x1024, 1344x1344, 1664x928 (16:9), 1472x1120 (4:3) and 1600x1056 (3:2),
and the transposes of the last three. They are multiples of 32, so the VAE can split the latents
across devices without padding. Each preset warms up and traces all of them when the pipeline is
created, except WH QuietBox / T3K (2x4), which only has 1024x1024: the larger sizes run out of DRAM
there with the text encoder on the device. Pass `width` and `height` to the pipeline call to choose
one.

The edit checkpoints have their own set: diffusers' sizes of about 1024² pixels for the same aspect
ratios, 1024x1024, 1376x768 (16:9), 1184x896 (4:3) and 1248x832 (3:2), and the transposes of the
last three. Without `width` and `height`, an edit's output takes the one closest in aspect ratio to
the last input image.

The transformer pads the image's sequence to one of a few lengths, so sizes of similar area share a
trace.

## Inpainting

Pass `image=` and `mask_image=` to regenerate the white part of the mask and keep the rest.
`strength` sets how much the white part may change: 1 ignores what was there, lower values stay
closer to it. Without `width` and `height`, the output takes the supported resolution closest in
aspect ratio to the image.

```bash
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_inpaint -k 2x4
```

## DBCache

The pipeline supports [cache-dit](https://github.com/vipshop/cache-dit)'s DBCache.

```bash
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_dbcache -k 2x2
```

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

# Measure every resolution, here on BH QuietBox 2 (2x2 mesh)
QWENIMAGE_PERF_ALL_RESOLUTIONS=1 pytest models/tt_dit/tests/models/qwenimage/test_performance_qwenimage.py -k "all_resolutions and 2x2"
```

## Image Editing

The same pipeline runs [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit),
[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) and
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), chosen by
`checkpoint_name`. The text encoder sees the input images through the Qwen2.5-VL vision tower, and
the transformer sees their latents appended to the generated image's. Pass `image=` one image, or up
to three for Edit-2509 and 2511; the output takes the supported resolution closest in aspect ratio
to the last one. With `mask_image=` and a single input image, all three inpaint: only the white part
of the mask is edited.

The edit tests pass on BH LoudBox (2x4). On BH QuietBox 2 they all pass except Edit-2511, which has
not been run there with the current code. On Wormhole, edits run out of DRAM: the transformer's
weights take about 10 GB of each chip's 12 GB, which leaves too little for the text encoder and the
longer image sequences. With the text encoder on the CPU (`use_torch_text_encoder=True`) and
`max_images=1`, inpainting with Edit-2509 and 2511 passes on WH 2x4.

```bash
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit -k 2x4
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit_multiple_images -k 2x4
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py::test_qwenimage_edit_inpaint -k 2x4
```

## Scalability

Qwen-Image has been implemented to support execution on 4-chip (BH QuietBox 2 with 2x2 mesh topology), 8-chip (QuietBox and LoudBox with 2x4 mesh topology) as well as 32-chip (Galaxy with 4x8 mesh topology) systems.
The 2x4 configuration runs on Wormhole (QuietBox / T3K) and Blackhole (LoudBox). The 2x2
configuration targets Blackhole (BH QuietBox 2) and is the configuration exercised in CI. The 4x8
configuration does not currently run on Wormhole Galaxy: the first generation after the warm-up
hangs.

The DiT model can be parallelized on 3 axes:

1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((1, 0), (2, 0), (4, 1))`. `cfg` is factor 1, so the conditional and unconditional passes run batched on the whole mesh. `sp` is factor 2 on axis 0, and `tp` is factor 4 on axis 1, meaning weights are tensor-fractured across each row of 4 devices.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 with factor 4, and `tp` is on axis 1 with factor 4.

The pipeline picks its configuration from the mesh shape and architecture:

| System        | Transformer       | Text encoder | VAE      |
| ------------- | ----------------- | ------------ | -------- |
| WH T3K        | CFG 1, SP 2, TP 4 | TP 4, FSDP 2 | H 4, W 2 |
| BH QuietBox 2 | CFG 2, SP 1, TP 2 | TP 2, FSDP 1 | H 2, W 1 |
| BH LoudBox    | CFG 2, SP 1, TP 4 | TP 4, FSDP 1 | H 4, W 1 |
| WH Galaxy     | CFG 2, SP 4, TP 4 | TP 4, FSDP 4 | H 4, W 4 |
| BH Galaxy     | CFG 2, SP 4, TP 4 | TP 4, FSDP 1 | H 4, W 4 |

H and W are the VAE's height and width splits. All models stay on the device, except the text
encoder's vision tower on WH T3K, which is loaded for each edit call.
