# Qwen-Image-Edit-2511

## Introduction

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is an image editing model developed by Alibaba's Qwen team. It takes an input image and a text instruction, producing an edited image that follows the instruction while preserving relevant elements from the original.

It is an enhanced version of Qwen-Image-Edit-2509 with improved consistency, identity preservation, and geometric reasoning.


## Architecture

The model shares the same core components as [Qwen-Image](QwenImage.md):

| Component | Class | Notes |
|-----------|-------|-------|
| Transformer | `QwenImageTransformer2DModel` | MMDiT, 24 heads × 128 dim, 60 layers |
| VAE | `AutoencoderKLQwenImage` | 16-channel latent, shared with Qwen-Image |
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL-7B vision-language model |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Dynamic shifting |

**Key differences from Qwen-Image (generation):**
- **Image conditioning:** Input images are encoded through the VAE and concatenated with noise latents along the sequence dimension
- **Vision-language prompts:** Source images are also fed to the VL encoder for cross-modal understanding
- **True CFG:** Uses classifier-free guidance with norm preservation
- **`zero_cond_t`:** Transformer config parameter for conditional timestep zeroing
- **Edit-specific prompt template:** System prompt designed for image editing instructions

### Transformer Config

```
attention_head_dim: 128
num_attention_heads: 24
inner_dim: 3072  (24 × 128)
num_layers: 60
in_channels: 64  (16 × 4, after 2×2 packing)
out_channels: 16
joint_attention_dim: 3584
patch_size: 2
```


## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)


## How to Run

```bash
# Set cache directory for faster subsequent runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the pipeline test on QuietBox (2x4 mesh)
pytest models/tt_dit/tests/models/qwenimageedit/test_pipeline_qwenimageedit.py -k "2x4sp1tp4 and traced and encoder_device"

# Run the pipeline test on Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/qwenimageedit/test_pipeline_qwenimageedit.py -k "4x8sp4tp4 and traced and encoder_device"

# Run the performance test on QuietBox (2x4 mesh)
pytest models/tt_dit/tests/models/qwenimageedit/test_performance_qwenimageedit.py -k "2x4cfg2sp1tp4"

# Run the performance test on Galaxy (4x8 mesh)
pytest models/tt_dit/tests/models/qwenimageedit/test_performance_qwenimageedit.py -k "4x8cfg2sp4tp4"
```


## Scalability

Qwen-Image-Edit-2511 supports execution on 8-chip (QuietBox/LoudBox with 2×4 mesh) and 32-chip (Galaxy with 4×8 mesh) systems.

The parallelization strategy is the same as Qwen-Image:
1. `cfg` (classifier-free guidance) — execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) — input sequence fractured across a mesh axis
3. `tp` (tensor parallel) — weights fractured across a mesh axis


## Component Reuse

This implementation maximally reuses existing tt-dit components:

| Component | Source |
|-----------|--------|
| `QwenImageTransformer` | `models/transformers/transformer_qwenimage.py` |
| `QwenImageVaeDecoder` | `models/vae/vae_qwenimage.py` (wraps Wan 2.1 decoder) |
| `Qwen25VlTokenizerEncoderPair` | `encoders/qwen25vl/encoder_pair.py` |
| `TransformerBlock` / `Attention` | `blocks/transformer_block.py`, `blocks/attention.py` |
| `ParallelFeedForward` | `layers/feedforward.py` |
| Parallel config utilities | `parallel/config.py`, `parallel/manager.py` |


## Weight Key Mapping

The Qwen-Image-Edit-2511 HuggingFace checkpoint uses the same key structure as Qwen-Image. The existing `_prepare_torch_state` methods in `QwenImageTransformerBlock` and `QwenImageTransformer` handle the mapping:

| HuggingFace Key | tt-metal Key |
|----------------|-------------|
| `transformer_blocks.{i}.img_mod.1.*` | `transformer_blocks.{i}.norm1.linear.*` |
| `transformer_blocks.{i}.img_norm1.*` | `transformer_blocks.{i}.norm1.norm.*` |
| `transformer_blocks.{i}.img_norm2.*` | `transformer_blocks.{i}.norm2.*` |
| `transformer_blocks.{i}.txt_mod.1.*` | `transformer_blocks.{i}.norm1_context.linear.*` |
| `transformer_blocks.{i}.txt_norm1.*` | `transformer_blocks.{i}.norm1_context.norm.*` |
| `transformer_blocks.{i}.img_mlp.*` | `transformer_blocks.{i}.ff.*` |
| `transformer_blocks.{i}.txt_norm2.*` | `transformer_blocks.{i}.norm2_context.*` |
| `transformer_blocks.{i}.txt_mlp.*` | `transformer_blocks.{i}.ff_context.*` |
| `norm_out.linear.*` | `time_embed_out.*` |
| `norm_out.norm.*` | `norm_out.*` |
