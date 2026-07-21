# Index-AniSora V3.2 (I2V)

## Introduction

[IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora) V3.2
is an anime-domain image-to-video model derived from
[Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers).
It preserves the two-expert (high-noise / low-noise) MoE structure of Wan2.2
and ships a finetuned weight pair specialized for anime-style motion.

This is a **full checkpoint replacement** (not a LoRA adapter or distilled model).
The model graph is identical to base Wan2.2 I2V — see [Wan2_2.md](Wan2_2.md) for
the underlying architecture.

## Details

The two-stage MoE schedule is preserved: the high-noise expert handles the
first 90% of the denoising trajectory and the low-noise expert handles the
final 10% (`boundary_ratio=0.9`, vs 0.5 for base Wan2.2).

The AniSora safetensors use the original-Wan key naming
(`blocks.X.self_attn.q.weight`, etc), identical to the lightx2v distill
checkpoints. The same `wan_lightx2v_to_diffusers_key` rename function in
`models/tt_dit/utils/lightx2v_loader.py` is reused.

`NUM_STEPS` is configurable (default 40). Lower step counts (16, 8) can produce video at decent quality.

## Weights

### AniSora V3.2 experts (DiT only)

- **Source:** [IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora)
- **Files:**
  - `V3.2/high_noise_model/diffusion_pytorch_model.safetensors` (~28 GB, BF16)
  - `V3.2/low_noise_model/diffusion_pytorch_model.safetensors` (~28 GB, BF16)
- **Total download:** ~57 GB (auto-downloaded on first run with `TT_DIT_ALLOW_HF_DOWNLOAD=1`)

### Base Wan2.2 components (shared, not re-downloaded if already cached)

- **Source:** [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- **Components:** tokenizer, UMT5 text encoder, VAE, scheduler config
- **Total download:** ~12 GB (shared with base Wan2.2)

### Download weights manually (optional)

Weights are auto-downloaded via HuggingFace Hub on first run. To pre-download or use a local copy:

```bash
# Option 1: Pre-download to HF cache
python -c "
from huggingface_hub import hf_hub_download
for f in ['V3.2/high_noise_model/diffusion_pytorch_model.safetensors',
          'V3.2/low_noise_model/diffusion_pytorch_model.safetensors']:
    hf_hub_download('IndexTeam/Index-anisora', f)
"

# Option 2: Point to a local directory
# Layout must be:
#   $ANISORA_LOCAL_DIR/V3.2/high_noise_model/diffusion_pytorch_model.safetensors
#   $ANISORA_LOCAL_DIR/V3.2/low_noise_model/diffusion_pytorch_model.safetensors
export ANISORA_LOCAL_DIR=/path/to/index-anisora-weights
```

## Inference defaults

Mirrors `anisoraV3.2/wan/configs/wan_i2v_A14B.py`:

| Parameter | Value | Note |
|---|---|---|
| `num_inference_steps` | 40 | UniPC sampler (configurable via `NUM_STEPS` env var) |
| `boundary_ratio` | 0.9 | High-noise covers first 90% of timesteps |
| `guidance_scale` | 3.5 | High-noise stage CFG |
| `guidance_scale_2` | 3.5 | Low-noise stage CFG |
| `sample_shift` | 5.0 | Inherited from base Wan2.2 scheduler config |

## Performance

BH Galaxy 4x8 Ring (sp=8, tp=4), 81 frames @ 16fps (~5s video):

| Stage | Base Wan2.2 I2V 480p | AniSora V3.2 I2V 480p | AniSora V3.2 I2V 720p |
|---|---|---|---|
| Text Encoding | 0.13s | 0.14s | 0.14s |
| Image Encoding | 5.03s | 5.02s | 12.84s |
| Denoising | 48.92s (40 steps, 1.22s/step) | 48.29s (40 steps, 1.21s/step) | 133.61s (40 steps, 3.34s/step) |
| VAE Decoding | 0.43s | 0.43s | 0.70s |
| **Total** | **54.52s** | **53.90s** | **147.32s** |

### Reduced-steps perf (AniSora V3.2)

| Steps | 480p (832x480) | 720p (1280x720) |
|---|---|---|
| 40 | 53.90s | 147.32s |
| 16 | 25.0s | 67.2s |
| 8 | 15.3s | 40.4s |

## Supported configurations

| System | Mesh | SP | TP | Topology | Test ID |
|---|---|---|---|---|---|
| BH Galaxy | 4x8 | 8 (axis 1) | 4 (axis 0) | Ring | `bh_4x8sp1tp0_ring` |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# Environment setup
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_DIT_ALLOW_HF_DOWNLOAD=1

# Set input image and prompt
export PROMPT_IMAGE=$TT_METAL_HOME/dog.jpg
export PROMPT="An anime girl smiling, soft lighting, cinematic"
export NUM_STEPS=40       # 40 default, or 16/8 for faster previews
export NO_PROMPT=1

# Run AniSora inference (480p)
pytest models/tt_dit/experimental/tests/test_pipeline_anisora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p and not random_weights" \
  --timeout 1800 -s
# Output: ./wan_anisora_i2v_832x480_0.mp4

# Run AniSora inference (720p)
pytest models/tt_dit/experimental/tests/test_pipeline_anisora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_720p and not random_weights" \
  --timeout 1800 -s
# Output: ./wan_anisora_i2v_1280x720_0.mp4

# Random-weights smoke test (no large download needed)
pytest models/tt_dit/experimental/tests/test_pipeline_anisora.py::test_pipeline_inference_random_weights \
  -v --timeout 1500 -s
```
