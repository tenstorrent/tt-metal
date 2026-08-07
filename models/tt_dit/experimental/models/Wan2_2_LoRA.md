# Wan2.2 I2V — LoRA Adapter Pipeline

## Introduction

`WanPipelineI2VLora` fuses [LoRA](https://arxiv.org/abs/2106.09685) adapters
into the base [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
weights on CPU before pushing to device. Inference uses vanilla I2V
machinery with no LoRA-specific runtime cost.

Each expert transformer (high-noise + low-noise) accepts an ordered LoRA
stack — multiple adapters can be combined on the same expert. Later entries
see earlier ones' deltas already folded into the base, which matches the
training-time expectation for adapters that were fine-tuned on top of
another adapter (e.g. an SVI LoRA on top of a LightX2V LoRA).

See [../../models/Wan2_2.md](../../models/Wan2_2.md) for the base model architecture and
[Wan2_2_SVI.md](Wan2_2_SVI.md) for long-video generation that builds on this pipeline.

## Details

- **CPU-side fusion:** `W' = W + sum_i scale_i * (B_i @ A_i)` for low-rank
  adapter pairs, with per-module alpha scaling applied when an `.alpha` key
  is present. The `fp32` promote → add → cast-back uses a fresh allocation
  per pair to avoid mutating the base when a stack folds in multiple
  adapters touching the same target.
- **Single aggregate verification:** after the full stack is fused, an L2
  norm check confirms the base weights actually changed (catches silent
  load failures from key mismatches).
- **Cache namespacing:** the TT cache is keyed by a SHA1 hash of the ordered
  `(resolved_path, scale)` tuples per expert. Two stacks with the same files
  in different order get different namespaces.

### Supported adapter key formats

`fuse_lora_state_dict` auto-detects these key shapes per adapter entry:

1. **Low-rank pairs** at one of:
   - `<base>.lora_A.weight` / `<base>.lora_B.weight` (lightx2v native)
   - `<base>.lora_down.weight` / `<base>.lora_up.weight` (legacy / kohya)
   - PEFT-style `<base>.lora_A.default.weight` (the trailing `.default`
     adapter-name segment is tolerated; SVI uses this convention)
2. **Bias deltas** at `<base>.diff_b` — added to the base `.bias`.
3. **Full-parameter deltas** at `<base>.diff` — added to the base
   `.weight`. Some LoRAs ship RMSNorm gammas via this channel.
4. **Per-module alpha** at `<base>.alpha` — applied as
   `effective_scale = scale * alpha / rank` for the low-rank pair under
   the same `<base>`.

### Supported key namespaces (auto-stripped or remapped)

- **lightx2v native:** `blocks.<i>.attn.q.lora_A.weight` — passes through.
- **diffusers prefixes:** `diffusion_model.`, `transformer.`, `unet.`,
  `model.` — stripped automatically before matching.
- **kohya / A1111:** `lora_unet_blocks_<i>_cross_attn_k.lora_down.weight`
  — remapped to lightx2v-style.

Anything outside the above is logged as a skipped-unknown warning rather
than erroring, so legitimate-but-unsupported keys don't break the load.

## API

```python
from models.tt_dit.experimental.pipelines.pipeline_wan_lora import LoRASpec, WanPipelineI2VLora

# Single-LoRA, both experts
pipe = WanPipelineI2VLora.create_pipeline(
    mesh_device=...,
    lora_high="/path/high.safetensors",
    lora_low="/path/low.safetensors",
)

# Multi-LoRA stack (LightX2V applied first, then SVI on top, half strength)
pipe = WanPipelineI2VLora.create_pipeline(
    mesh_device=...,
    lora_high=[
        LoRASpec("/path/lightx2v_high.safetensors", scale=1.0),
        LoRASpec("/path/svi_high.safetensors", scale=0.5),
    ],
    lora_low=[
        LoRASpec("/path/lightx2v_low.safetensors", scale=1.0),
        LoRASpec("/path/svi_low.safetensors", scale=1.0),
    ],
)
```

Either `lora_high` or `lora_low` may be `None` — that expert uses unmodified
base weights. At least one must be set.

## Weights

### Earth zoom-out LoRA

- **Source:** [wangkanai/wan22-fp16-i2v-loras](https://huggingface.co/wangkanai/wan22-fp16-i2v-loras)
- **File:** `loras/wan/wan22-camera-earthzoomout.safetensors` (293 MB)
- **Format:** Single-file, kohya/A1111-style keys

### Camera rotation LoRA

- **Source:** [wangkanai/wan22-fp16-i2v-loras](https://huggingface.co/wangkanai/wan22-fp16-i2v-loras)
- **File:** `loras/wan/wan22-camera-rotation-rank16-v2.safetensors` (293 MB)
- **Format:** Single-file, kohya/A1111-style keys

### Base Wan2.2 components (shared)

- **Source:** [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- **Components:** tokenizer, UMT5 text encoder, VAE, scheduler config, transformer base weights (~70 GB)

### Download

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/path/to/lora-camera', exist_ok=True)
for f in ['loras/wan/wan22-camera-earthzoomout.safetensors',
          'loras/wan/wan22-camera-rotation-rank16-v2.safetensors']:
    hf_hub_download('wangkanai/wan22-fp16-i2v-loras', f, local_dir='/path/to/lora-camera')
"
```

## Supported configurations

| System | Mesh | SP | TP | Topology | Test ID |
|---|---|---|---|---|---|
| BH Loud Box | 2x4 | 4 (axis 1) | 2 (axis 0) | Linear | `bh_2x4sp1tp0` |
| BH Galaxy | 4x8 | 8 (axis 1) | 4 (axis 0) | Ring | `bh_4x8sp1tp0_ring` |

## How to Run

### Single-LoRA (40 steps)

```bash
export TT_DIT_CACHE_DIR=/your/cache/path
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export TT_DIT_ALLOW_HF_DOWNLOAD=1

export LORA_HIGH_PATH=/path/to/lora-camera/wan22-camera-earthzoomout.safetensors
unset LORA_LOW_PATH          # single-file: low-noise expert uses base weights
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
```

### Multi-LoRA stack

```bash
# Comma-separated path[:scale] entries. Order matters -- earlier entries are
# fused first.
export LORA_STACK_HIGH="/path/lightx2v_high.safetensors:1.0,/path/style_high.safetensors:0.5"
export LORA_STACK_LOW="/path/lightx2v_low.safetensors:1.0,/path/style_low.safetensors:1.0"

pytest models/tt_dit/experimental/tests/test_pipeline_lora.py \
  -v -k "bh_4x8sp1tp0_ring and resolution_480p" \
  --timeout 1800 -s
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LORA_HIGH_PATH` | Either this or stack | — | Path to high-noise expert LoRA `.safetensors` |
| `LORA_LOW_PATH` | No | None | Path to low-noise expert LoRA |
| `LORA_SCALE` | No | 1.0 | Blend strength applied to single-LoRA entries |
| `LORA_STACK_HIGH` | Either this or single | — | Comma-separated `path[:scale]` list for the high-noise expert |
| `LORA_STACK_LOW` | No | None | Comma-separated `path[:scale]` list for the low-noise expert |
| `NUM_STEPS` | No | 40 | Inference steps |
| `GUIDANCE_SCALE` | No | 3.5 | CFG scale |
| `BOUNDARY_RATIO` | No | 0.875 | High/low expert split ratio |
| `PROMPT_IMAGE` | No | `./prompt_image.png` | Seed image path |
| `PROMPT` | No | golden retriever prompt | Text prompt |

## Limitations / open items

- LoRA fusion happens on CPU before device push — adds ~5–10 s per expert to
  pipeline initialization (one-time cost, scales with stack depth).
- Only `bh_4x8sp1tp0_ring` is exercised in CI.
- Any community LoRA trained for Wan2.2 I2V should work if its keys follow
  lightx2v native, diffusers, or kohya/A1111 naming. Untested LoRAs may need
  key remap additions.
