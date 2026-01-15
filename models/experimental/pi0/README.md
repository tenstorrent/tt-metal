# PI0 Model for Tenstorrent

PI0 (Physical Intelligence Zero) is a vision-language-action model for robotics
that combines a vision encoder, language model, and action expert for end-to-end
robot control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PI0 Model                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────┐│
│  │         PREFIX EMBEDDING            │   │    SUFFIX EMBEDDING      ││
│  │                                     │   │                          ││
│  │  ┌───────────┐   ┌───────────────┐  │   │  ┌────────┐  ┌────────┐ ││
│  │  │  Images   │   │ Language      │  │   │  │ State  │  │ Noisy  │ ││
│  │  │  (224x224)│   │ Tokens        │  │   │  │ (32)   │  │Actions │ ││
│  │  └─────┬─────┘   └───────┬───────┘  │   │  └───┬────┘  └───┬────┘ ││
│  │        │                 │          │   │      │           │      ││
│  │        ▼                 │          │   │      └─────┬─────┘      ││
│  │  ┌───────────┐           │          │   │            │            ││
│  │  │  SigLIP   │           │          │   │   ┌────────▼─────────┐  ││
│  │  │  Vision   │           │          │   │   │ Action+Time MLP  │  ││
│  │  │  Tower    │           │          │   │   │ (fuse_action_    │  ││
│  │  │(27 blocks)│           │          │   │   │  time)           │  ││
│  │  └─────┬─────┘           │          │   │   └────────┬─────────┘  ││
│  │        │                 │          │   │            │            ││
│  │        ▼                 │          │   └────────────┼────────────┘│
│  │  ┌───────────┐           │          │                │             │
│  │  │Projector  │           │          │                │             │
│  │  │(1152→2048)│           │          │                │             │
│  │  └─────┬─────┘           │          │                │             │
│  │        │                 │          │                │             │
│  │        ▼                 ▼          │                │             │
│  │  ┌───────────────────────────────┐  │                │             │
│  │  │  Image Embeds + Lang Embeds   │  │                │             │
│  │  │  (Gemma 2B embedding)         │  │                │             │
│  │  └───────────────┬───────────────┘  │                │             │
│  │                  │                  │                │             │
│  └──────────────────┼──────────────────┘                │             │
│                     │                                   │             │
│                     ▼                                   ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │               DUAL-EXPERT TRANSFORMER (18 layers)                │ │
│  │  ┌────────────────────────┐    ┌────────────────────────┐        │ │
│  │  │     Gemma 2B VLM       │    │   Gemma 300M Expert    │        │ │
│  │  │   (processes prefix)   │◄──►│  (processes suffix)    │        │ │
│  │  │                        │    │                        │        │ │
│  │  │  Q_vlm ──┐             │    │  Q_exp ──┐             │        │ │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ K_exp   │             │        │ │
│  │  │  V_vlm ──┘   ATTN      │    │  V_exp ──┘             │        │ │
│  │  │                        │    │                        │        │ │
│  │  │  MLP_vlm               │    │  MLP_exp               │        │ │
│  │  └────────────────────────┘    └────────────────────────┘        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                   │                                   │
│                                   ▼                                   │
│                    ┌──────────────────────────────┐                   │
│                    │     FLOW MATCHING DENOISER   │                   │
│                    │     (10 denoising steps)     │                   │
│                    │                              │                   │
│                    │  for t in [1.0 → 0.0]:       │                   │
│                    │    noise_pred = expert_out   │                   │
│                    │    actions = euler_step()    │                   │
│                    └──────────────┬───────────────┘                   │
│                                   │                                   │
│                                   ▼                                   │
│                         ┌───────────────────┐                         │
│                         │   Action Output   │                         │
│                         │ [batch=1, 50, 32] │                         │
│                         └───────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs
- **Flow Matching**: Iterative denoising from pure noise to actions over 10 steps
- **Dual Experts**: VLM (2B) processes images+language, Expert (300M) processes actions

## Directory Structure

```
pi0/
├── common/                     # Shared configs and utilities
│   ├── configs.py              # Model configurations
│   ├── weight_loader.py        # Checkpoint loading
│   └── utils.py                # Common utilities
├── reference/                  # PyTorch reference implementation
│   ├── torch_pi0_model.py      # Main PI0 model
│   ├── torch_paligemma.py      # PaliGemma backbone
│   ├── torch_siglip.py         # SigLIP vision tower
│   ├── torch_gemma.py          # Gemma attention/MLP
│   ├── torch_prefix.py         # Prefix embedding
│   ├── torch_suffix.py         # Suffix embedding
│   └── torch_denoise.py        # Denoising logic
├── tt/                         # TTNN implementation
│   ├── ttnn_pi0_model.py       # Main PI0 model (TTNN)
│   ├── ttnn_paligemma.py       # PaliGemma backbone (TTNN)
│   ├── ttnn_siglip.py          # SigLIP vision tower (TTNN)
│   ├── ttnn_gemma.py           # Gemma attention/MLP (TTNN)
│   ├── ttnn_prefix.py          # Prefix embedding (TTNN)
│   ├── ttnn_suffix.py          # Suffix embedding (TTNN)
│   └── ttnn_common.py          # Common TTNN utilities
├── tests/
│   ├── pcc/                    # PCC (accuracy) tests
│   ├── perf/                   # Performance benchmarks
│   ├── demo/                   # Demo scripts with ALOHA/LIBERO datasets
│   └── download_pretrained_weights.py
└── weights/                    # Pretrained checkpoints
    └── pi0_base/               # Base model checkpoint
```

## Quick Start

### 1. Environment Setup

```bash
# Set required environment variables
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Activate virtual environment
source $TT_METAL_HOME/python_env/bin/activate
```

### 2. Download Pretrained Weights

The model requires pretrained weights to run. Download them using the provided script:

```bash
# Automatic download (requires gdown)
python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py

# Or with custom output directory
python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py \
    --output-dir /custom/path/weights
```

**Manual Download (if automatic download fails):**

1. Open: https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN
2. Download the folder
3. Extract to: `$TT_METAL_HOME/models/experimental/pi0/weights/`

**Alternative: Using command-line tools:**

```bash
# Using gdown
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN \
    -O $TT_METAL_HOME/models/experimental/pi0/weights/

# Using rclone (works with private folders)
rclone config  # Setup Google Drive remote named 'gdrive'
rclone copy gdrive:pi0_base $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base
```

After download, verify the structure:
```
$TT_METAL_HOME/models/experimental/pi0/weights/
└── pi0_base/
    ├── model.safetensors
    └── config.json
```

## Running Tests

### PCC Tests (Accuracy Validation)

PCC (Pearson Correlation Coefficient) tests compare TTNN outputs against PyTorch reference.

**Full Model PCC Test:**

```bash
# Using pytest
pytest models/experimental/pi0/tests/pcc/test_pcc_ttnn_pi0_model.py -v
```

**Code Flow (what gets tested):**

```
PI0ModelTTNN.sample_actions()
│
├─► self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
│   └─► PrefixEmbeddingTTNN.embed_prefix()
│       │
│       ├─► self.embed_image_fn(img)  [backbone.embed_image]
│       │   └─► PaliGemmaBackboneTTNN.embed_image()
│       │       ├─► SigLIPVisionTowerTTNN.forward()
│       │       │   └─► SigLIPBlockTTNN.forward() × 27 layers
│       │       │       ├─► SigLIPAttentionTTNN.forward()
│       │       │       └─► SigLIPMLPTTNN.forward()
│       │       │
│       │       └─► MultiModalProjectorTTNN.forward() [1152 → 2048]
│       │
│       └─► self.embed_language_fn(tokens)  [backbone.embed_language_tokens]
│           └─► ttnn.embedding(tokens, vlm_embed_tokens)
│
├─► self.backbone.forward_vlm(prefix_embs, use_cache=True)
│   └─► PaliGemmaBackboneTTNN.forward_vlm()
│       └─► GemmaBlockTTNN.forward() × 18 layers (VLM blocks)
│           ├─► rms_norm_ttnn()
│           ├─► GemmaAttentionTTNN.forward()
│           │   ├─► ttnn.linear() for fused QKV
│           │   ├─► ttnn.experimental.rotary_embedding() for RoPE
│           │   └─► ttnn.transformer.scaled_dot_product_attention()
│           │
│           └─► GemmaMLPTTNN.forward()
│               ├─► ttnn.linear() for gate_proj, up_proj
│               ├─► ttnn.gelu()
│               └─► ttnn.linear() for down_proj
│
├─► [DENOISING LOOP × 10 steps]
│   │
│   ├─► self.embed_suffix(state, x_t, timestep)
│   │   └─► SuffixEmbeddingTTNN.embed_suffix()
│   │       └─► fuse_action_time MLP + action_embed + state_embed
│   │
│   └─► self.backbone.forward_expert(suffix_embs, past_key_values=prefix_kv_cache)
│       └─► PaliGemmaBackboneTTNN.forward_expert()
│           └─► GemmaBlockTTNN.forward() × 18 layers (Expert blocks)
│
└─► return denoised_actions [batch=1, 50, 32]
```

**Component PCC Tests:**

```bash
# Run all component tests
python models/experimental/pi0/tests/pcc/run_all_pcc_tests.py

# Individual component tests
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_prefix.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_gemma.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_siglip.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_paligemma.py -v
```

**Test with Random vs Pretrained Weights:**

```bash
# Run with pretrained weights only (full validation)
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v -k "pretrained_weight_true"

# Run with random weights only (fast CI)
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v -k "pretrained_weight_false"
```

### Performance Tests (Benchmarking)

```bash
# Full model performance test
pytest models/experimental/pi0/tests/perf/test_perf_ttnn_pi0_model.py -v -s

# Direct execution
python models/experimental/pi0/tests/perf/test_perf_ttnn_pi0_model.py
```

## Demo Scripts

Demo scripts visualize model inference on robotics datasets.

**Extract Sample Images (required first):**

```bash
# Extract ALOHA simulation samples (downloads from HuggingFace)
python models/experimental/pi0/tests/demo/extract_aloha_samples.py

# Extract LIBERO samples (downloads from HuggingFace)
python models/experimental/pi0/tests/demo/extract_libero_samples.py
```

This creates sample images in `tests/demo/sample_images/`:
```
sample_images/
├── aloha_sim/
│   ├── sample_0_top.png
│   ├── sample_1_top.png
│   └── metadata.txt
└── libero/
    ├── sample_0_main.png
    ├── sample_0_wrist.png
    └── metadata.txt
```

**Run Demos:**

```bash
# ALOHA simulation demo
python models/experimental/pi0/tests/demo/run_aloha_sim_demo.py

# LIBERO demo
python models/experimental/pi0/tests/demo/run_libero_demo.py

# Visualize results
python models/experimental/pi0/tests/demo/visualize_demo.py
```

## Troubleshooting

### `Checkpoint not found`

Download weights using the script:
```bash
python models/experimental/pi0/tests/download_pretrained_weights.py
```

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision Encoder | SigLIP (27 transformer blocks, 1152 hidden dim) |
| VLM Backbone | Gemma 2B (18 transformer blocks) |
| Action Expert | Gemma 300M (18 transformer blocks) |
| Image Size | 224×224 |
| Action Dimension | 32 |
| Action Horizon | 50 |

## Future Optimizations

### Trace + 2CQ (Two Command Queue)

**Status:** Not yet implemented

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
