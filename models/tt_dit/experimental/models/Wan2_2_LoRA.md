# Wan2.2 I2V — LoRA Adapter Pipeline

## Introduction

`WanLoraPipelineI2V` fuses [LoRA](https://arxiv.org/abs/2106.09685) adapters
into the base [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
weights on CPU before pushing to device. This enables camera control, style
transfer, distill acceleration, and other adapter-based effects without
retraining or replacing the full checkpoint.

See [Wan2_2.md](Wan2_2.md) for the base model architecture.

## Details

- **CPU-side LoRA fusion:** each LoRA weight pair (`lora_down` × `lora_up`) is
  multiplied and added to the corresponding base weight (`W' = W + scale × up @ down`).
  Optional per-module alpha scaling is applied when present.
- **L2 norm verification:** after fusion, the pipeline checks that fused weights
  differ from base weights (catches silent load failures from key mismatches).
- **Key format auto-detection:** supports Wan/lightx2v-style module keys under
  `blocks.<i>...`, including checkpoints that prefix those keys with
  `diffusion_model.`, `transformer.`, `unet.`, or `model.`, as well as
  kohya/A1111-style keys such as
  `lora_unet_blocks_0_cross_attn_k.lora_down.weight`.
- **Two LoRA modes:**
  - **Split LoRA** — separate adapters for high-noise and low-noise experts
    (set both `LORA_HIGH_PATH` and `LORA_LOW_PATH`)
  - **Single-file LoRA** — one adapter for the high-noise expert only; the
    low-noise expert uses unmodified base weights (set only `LORA_HIGH_PATH`)

## Weights

### Earth zoom-out LoRA

- **Source:** [wangkanai/wan22-fp16-i2v-loras](https://huggingface.co/wangkanai/wan22-fp16-i2v-loras)
- **File:** `loras/wan/wan22-camera-earthzoomout.safetensors` (293 MB)
- **Format:** Single-file, kohya/A1111-style keys
- **Effect:** Adds a dramatic earth zoom-out perspective transition to any I2V generation

### Camera rotation LoRA

- **Source:** [wangkanai/wan22-fp16-i2v-loras](https://huggingface.co/wangkanai/wan22-fp16-i2v-loras)
- **File:** `loras/wan/wan22-camera-rotation-rank16-v2.safetensors` (293 MB)
- **Format:** Single-file, kohya/A1111-style keys
- **Effect:** Rotates the camera around the subject
- [More camera LoRAs on HuggingFace](https://huggingface.co/wangkanai/wan22-fp16-i2v-loras/tree/main/loras/wan)

### Base Wan2.2 components (shared, not re-downloaded if already cached)

- **Source:** [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- **Components:** tokenizer, UMT5 text encoder, VAE, scheduler config, transformer base weights
- **Total download:** ~70 GB (shared with base Wan2.2)

### Download weights

```bash
pip install huggingface_hub   # if not already installed

# Camera control LoRAs (~293 MB each)
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/path/to/lora-camera', exist_ok=True)
for f in ['loras/wan/wan22-camera-earthzoomout.safetensors',
          'loras/wan/wan22-camera-rotation-rank16-v2.safetensors']:
    hf_hub_download('wangkanai/wan22-fp16-i2v-loras', f,
                    local_dir='/path/to/lora-camera')
"
```

## Supported configurations

| System | Mesh | SP | TP | Topology | Test ID |
|---|---|---|---|---|---|
| BH Galaxy | 4x8 | 8 (axis 1) | 4 (axis 0) | Ring | `bh_4x8sp1tp0_ring` |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- LoRA weight files downloaded (see above)

## How to Run

### Earth zoom-out LoRA (40 steps)

```bash
# Environment setup
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_DIT_ALLOW_HF_DOWNLOAD=1

# LoRA config
export LORA_HIGH_PATH=/path/to/lora-camera/wan22-camera-earthzoomout.safetensors
unset LORA_LOW_PATH          # single-file: low-noise expert uses base weights
export LORA_SCALE=1.0

# Inference config
export PROMPT_IMAGE=$TT_METAL_HOME/dog.jpg
export PROMPT="A golden retriever running on a sandy beach, waves in the background"
export NUM_STEPS=40
export GUIDANCE_SCALE=3.5
export BOUNDARY_RATIO=0.875
export NO_PROMPT=1

pytest models/tt_dit/experimental/tests/test_pipeline_lora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" \
  --timeout 1800 -s
# Output: ./wan_lora_i2v_832x480_0.mp4
```

### Camera rotation LoRA (40 steps)

```bash
export LORA_HIGH_PATH=/path/to/lora-camera/wan22-camera-rotation-rank16-v2.safetensors
unset LORA_LOW_PATH
export LORA_SCALE=1.0

export PROMPT_IMAGE=$TT_METAL_HOME/dog.jpg
export PROMPT="A golden retriever running on a sandy beach, waves in the background"
export NUM_STEPS=40
export GUIDANCE_SCALE=3.5
export BOUNDARY_RATIO=0.875
export NO_PROMPT=1

pytest models/tt_dit/experimental/tests/test_pipeline_lora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" \
  --timeout 1800 -s
# Output: ./wan_lora_i2v_832x480_0.mp4
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LORA_HIGH_PATH` | Yes | — | Path to high-noise expert LoRA `.safetensors` |
| `LORA_LOW_PATH` | No | None | Path to low-noise expert LoRA (omit for single-file LoRAs) |
| `LORA_SCALE` | No | 1.0 | Blend strength (0.0 = no effect, 1.0 = full effect) |
| `NUM_STEPS` | No | 40 | Inference steps (40 for camera/style LoRAs, 4 for distill) |
| `GUIDANCE_SCALE` | No | 3.5 | CFG scale (3.5 for camera/style, 1.0 for distill) |
| `BOUNDARY_RATIO` | No | 0.875 | High/low expert split ratio (0.875 for camera, 0.5 for distill) |
| `PROMPT_IMAGE` | No | `./prompt_image.png` | Seed image path |
| `PROMPT` | No | golden retriever prompt | Text prompt |

## Limitations / open items

- LoRA fusion happens on CPU before device push — adds ~5–10 s to pipeline
  initialization (one-time cost, not per-frame).
- Only `bh_4x8sp1tp0_ring` is exercised so far.
- Dedicated perf test function (`test_pipeline_performance_lora`) not yet added.
- Any community LoRA trained for Wan2.2 I2V should work if its keys follow
  diffusers or kohya/A1111 naming. Untested LoRAs may need key remap additions.
