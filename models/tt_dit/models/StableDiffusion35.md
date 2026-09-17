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

| System                     | CFG | SP | TP | Current Performance | Target Performance |
|----------------------------|-----|----|----|---------------------|--------------------|
| QuietBox                   | 2   | 2  | 2  | 12.2s               | 14.4s              |
| Galaxy                     | 2   | 4  | 4  | 5.6s                | 3.6s               |
| QuietBox (Blackhole, 4-chip) | 1   | 1  | 4  | 6.8s (20 steps)      | -                  |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

1. Visit [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) to grant access to the model weights
2. Login with the HuggingFace token: `huggingface-cli login`

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the pipeline test on QuietBox (2x4 mesh)
pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "2x4cfg1sp0tp1 and yes_traced"

# Run the pipeline test on Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "4x8cfg1sp0tp1 and yes_traced"
```

### Blackhole (4-chip QuietBox, 1x4 mesh)

The 4-chip Blackhole path runs on a 1x4 mesh with parallel config `((1, 0), (1, 0), (4, 1))` (cfg=1, sp=1, tp=4; encoder/VAE on the same native 1x4 mesh - no reshape needed, since the parent mesh already is 1x4; T5 auto-disabled). The mesh uses `Ring` CCL topology (not `Linear`) - this is what unlocks the fused-kernel optimizations below. The model runs in plain bf16; the bf8 weight/activation quantization path previously available on this config (`SD35_QUANT=bf8`) has been reverted.

```bash
# NOTE: on this setup TT_DIT_CACHE_DIR caused a hang during on-device weight distribution,
# so we run with it unset (weights load directly from the HF/torch state dict).
unset TT_DIT_CACHE_DIR

NO_PROMPT=1 pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py \
  -k "1x4cfg0sp0tp1 and yes_traced"
```

Performance (1024x1024, CFG on, bf16) scales as ~0.30s/step + ~0.8s fixed overhead (text encode + VAE decode):

| Denoising steps | Total time |
|-----------------|------------|
| 10              | 3.8s       |
| 20 (default)    | 6.8s       |
| 28              | 9.1s       |

To change the step count, edit `num_inference_steps` (the last field) in the parametrize at `models/tt_dit/tests/models/sd35/test_pipeline_sd35.py:33`:

```python
("large", 1024, 1024, 3.5, 20),   # change 20 -> 10 / 28 / etc.
```

#### Implementation notes

- **Grid**: switched from a 2x2 mesh (`cfg=1, sp=2, tp=2`) to a 1x4 line (`cfg=1, sp=1, tp=4`). `sp` and `tp` can't share a single mesh axis here - a single tensor can't be sharded along two independent dims on one physical axis - so the 1x4 grid goes fully tensor-parallel instead of splitting sequence- and tensor-parallelism across two axes.
- **Topology**: switched from `Linear` to `Ring`. The fused AGMM/MMRS kernels below gate on `Topology.Ring`; requires `device_params` with `fabric_config=FABRIC_1D_RING` (see `ring_params_req_exact_devices` in `models/tt_dit/utils/test.py`) rather than the `Linear`-topology `line_params`.
- **Fused MM+RS+addcmul**: the FFN's `ff2` (row-parallel matmul + reduce-scatter + gate-multiply + residual-add) runs as one fused op instead of three. This device's compute grid (11x10) isn't covered by the op's built-in swept configs or rule engine (both assume a 12-wide Blackhole grid), so a manual blocking is registered for this grid in `transformer_sd35.py`. The fused kernel also requires batch size 1, but this model batches CFG cond+uncond as batch=2, so the caller flattens the batch dim into the token dim before the call and reshapes back after.
- **Fused AGMM** (`to_out`/`to_add_out`) was investigated but is not used on this config - it hits a NOC-assignment error in the underlying kernel that appears to be a genuine gap for 1-row mesh shapes (every other usage of that op in this codebase is on a real 2D mesh, e.g. 4x8/8x8/12x9). Left unwired pending kernel-level investigation.


## Scalability

SD3.5-Large has been implemented to support execution on 8-chip (LoudBox and QuietBox) as well as 32-chip (Galaxy) systems.
On Wormhole it runs the 8-chip and 32-chip `cfg=2` configs above. On Blackhole it runs on a 4-chip QuietBox (1x4 mesh, `cfg=1`, fully tensor-parallel); see the Blackhole run section above.

The DiT model can be parallelized on 3 axes:
1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((2, 1), (2, 0), (2, 1))`. This gives us `cfg` parallelism with factor 2 on axis 1, yielding 2 2x2 submeshes. `sp` is factor 2 on axis 0, meaning that activations are sequence-fractured on the `2x2` submesh on axis 0. `tp` is factor 2 on axis 1, meaning weights are tensor-fractured on the `2x2` submesh on axis 1.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 and `tp` is on axis 1, giving us `sp` factor 4 and `tp` factor 4.

The text embedding models and the VAE decoder are parallelized with tensor parallelism on one or both of the cfg submeshes.
