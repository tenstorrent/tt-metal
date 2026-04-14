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
│   ├── perf/                   # Performance benchmarks and e2e (2CQ + Trace)
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

### Performance Test (end-to-end (2CQ + Trace))
```bash
# Direct execution
pytest models/experimental/pi0/tests/perf/test_perf_e2e.py
```

## Demo Scripts

Demo scripts visualize model inference on robotics datasets.

**Extract Sample Images (required first):**

```bash
# ImageIO python library plugin PyAv is needed to extract images from videos
python -m ensurepip --upgrade && python -m pip install imageio[pyav]

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

## Closed-Loop LIBERO Evaluation

Pi-0.5 can be evaluated closed-loop inside the LIBERO simulator via
`tests/pcc/test_rollout_libero.py`. The rollout harness resets the env to
a canonical LIBERO initial state, runs the policy in the loop (replanning
every `--chunk` steps), and reports per-task success rates. Both the
PyTorch reference (`--backend torch`) and the TTNN implementation
(`--backend ttnn`) are supported with identical preprocessing, so their
success rates can be compared directly.

### One-time setup

Pi-0.5 closed-loop evaluation requires the LIBERO simulator repo. Install
it once in the same Python environment you use for tt-metal:

```bash
# Clone upstream LIBERO (tested against commit 8f1084e3)
cd <your workspace root>
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# Install as editable (brings in robosuite, bddl, mujoco, gym, etc.)
pip install --no-deps -e .
pip install robosuite==1.4.0 bddl==1.0.1 mujoco==3.6.0 easydict pyserial gym==0.25.2

# Pre-create the LIBERO config file to skip its interactive setup prompt.
# Point the paths at LIBERO's own `libero/libero/` subdirectory:
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<'YAML'
benchmark_root: <workspace>/LIBERO/libero/libero
bddl_files: <workspace>/LIBERO/libero/libero/bddl_files
init_states: <workspace>/LIBERO/libero/libero/init_files
datasets: <workspace>/LIBERO/libero/libero/datasets
assets: <workspace>/LIBERO/libero/libero/assets
YAML
```

The rollout harness pulls per-task LIBERO init states from LIBERO's
`init_files/` directory via the upstream `libero.libero.benchmark` API.
It does NOT need the raw training demo `.hdf5` files.

Additional Python deps used by the harness (install in the tt-metal venv
if not already present):

```bash
pip install 'lerobot[all]==0.4.4' --no-deps
pip install datasets 'huggingface_hub[hf_transfer]' deepdiff orderly-set
```

### Pi-0.5 LIBERO weights

The harness loads the LeRobot `lerobot/pi05_libero` checkpoint (a ~14.5 GB
pi-0.5 fine-tune for LIBERO). Download it via Hugging Face and symlink
into the expected location:

```bash
export HF_TOKEN=<your huggingface token>
huggingface-cli download lerobot/pi05_libero

# Point the pi0 weights dir at your local HF cache
ln -s ~/.cache/huggingface/hub/models--lerobot--pi05_libero/snapshots/<snapshot-hash> \
      models/experimental/pi0/weights/pi05_libero
```

(The `pi05_base` checkpoint used by the numerical-fidelity tests can be
fetched the same way from `lerobot/pi05_base`.)

### Environment variables

```bash
source <tt-metal venv>/bin/activate
export TT_METAL_HOME=<tt-metal root>
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME
export HF_TOKEN=<your huggingface token>

# Rendering backend:
# - headless (for CI / scripts): MUJOCO_GL=egl
# - on-screen viewer (requires an X display): MUJOCO_GL=glfw + DISPLAY=:0
export MUJOCO_GL=egl
```

### Running the rollout

The harness takes `--suite`, `--tasks` (task IDs), `--n-inits`, and
`--max-steps`. The `lerobot/pi05_libero` fine-tune targets the
`libero_10`, `libero_spatial`, and `libero_goal` suites (matching the
suites used in the openpi eval):

```bash
# TTNN backend, libero_spatial task 0, 1 init, headless
python models/experimental/pi0/tests/pcc/test_rollout_libero.py \
    --backend ttnn --suite libero_spatial --tasks 0 \
    --n-inits 1 --max-steps 220 --chunk 10

# PyTorch reference on the same input (much slower on CPU)
python models/experimental/pi0/tests/pcc/test_rollout_libero.py \
    --backend torch --suite libero_spatial --tasks 0 \
    --n-inits 1 --max-steps 220 --chunk 10

# With on-screen MuJoCo viewer (requires DISPLAY set)
export MUJOCO_GL=glfw
export DISPLAY=:0
python models/experimental/pi0/tests/pcc/test_rollout_libero.py \
    --backend ttnn --suite libero_spatial --tasks 0 \
    --n-inits 1 --max-steps 220 --chunk 10 --render
```

Recommended `--max-steps` per suite (matching openpi's LIBERO eval):
`libero_spatial` 220, `libero_object` 280, `libero_goal` 300,
`libero_10` 520, `libero_90` 400.

### Numerical-fidelity tests (no simulator needed)

Three additional tests live under `tests/pcc/` and validate the TTNN
implementation against the PyTorch reference WITHOUT launching LIBERO.
Run them after any change to `tt/ttnn_gemma.py`, `tt/ttnn_paligemma.py`,
`tt/ttnn_pi0_model.py`, `tt/ttnn_suffix.py`, or
`reference/torch_pi0_model.py`:

```bash
# Per-denoise-step velocity PCC (threshold: per-step ≥ 0.99, aggregate ≥ 0.93)
python models/experimental/pi0/tests/pcc/test_pcc_pi05_per_step.py

# N-run determinism check (max|Δ| must be 0 across repeated inferences)
python models/experimental/pi0/tests/pcc/test_determinism_pi05.py --runs 5

# Action-space divergence on held-out LeRobot samples (real LIBERO / ALOHA inputs)
python models/experimental/pi0/tests/pcc/test_action_divergence_lerobot.py --n 4
```

All three tests load the `pi05_base` checkpoint from
`models/experimental/pi0/weights/pi05_base`; override with
`PI0_CHECKPOINT=/path/to/checkpoint` if you keep weights elsewhere.

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

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
