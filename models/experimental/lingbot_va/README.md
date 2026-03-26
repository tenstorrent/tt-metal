# Lingbot-VA for Tenstorrent

Lingbot-VA is a vision–language–action stack built on the Wan family of models: it encodes multi-camera video with a causal VAE, conditions on UMT5 text embeddings, and runs a 3D Wan transformer for video latents and action tokens under flow-matching schedulers. This tree contains a **PyTorch reference**, the **TTNN implementation** used on device, **PCC / perf tests**, and **demo scripts** that mirror the server inference API without a WebSocket.

## Architecture

At a high level (no GPU path; reference and prep run on **CPU**, model math on **Tenstorrent** via TTNN):

- **Inputs:** Three RGB camera streams (RobotWin naming), task prompt, and optional proprioceptive state; observations match the Lingbot-VA server contract.
- **VAE:** `AutoencoderKLWan`-compatible encoder/decoder; TT path uses causal 3D convs, residual down/up blocks, and (where applicable) Lingbot-specific blocking configs.
- **Text:** UMT5 encoder (HF checkpoint) with a TTNN port for embeds used in cross-attention.
- **Backbone:** `WanTransformer3DModel` with self-attention + cross-attention blocks, RoPE over a 3D grid, dual paths for **video latents** vs **action** tokens, and patch embedding for `(C, F, H, W)`.
- **Schedulers:** Flow-matching style stepping for video and action branches (configurable step counts in demo/tests).
- **Outputs:** Per-chunk **actions** (infer mode) or decoded **`demo.mp4`** (multi-chunk generate mode). TT demo writes video next to the demo script by default unless `--save-dir` is set.

**Key details:**

- Transformer and VAE TT code live under `tt/`; weights are mapped from the reference checkpoints.
- Demo entrypoint is **`tests/demo/inference_ttnn.py`** (TTNN path); `tests/demo/inference_torch.py` is a CPU PyTorch reference runner for comparison/debug.

## Directory Structure

```
lingbot_va/
├── reference/                  # PyTorch reference (diffusers-compatible Wan pieces, utils, configs)
│   ├── transformer_wan.py      # WanTransformer3DModel (reference)
│   ├── utils.py                  # Loaders, schedulers, RobotWin config, VAE streaming helper
│   └── checkpoints/            # Local HF layout: vae/, tokenizer/, text_encoder/, transformer/
├── tt/                         # TTNN implementation
│   ├── transformer_wan.py       # Wan transformer (TTNN)
│   ├── attention_wan.py
│   ├── utils.py                 # TT loaders (transformer, text encoder, VAE encode/decode wrappers)
│   ├── vae_encoder.py / vae_decoder.py
│   ├── wan_rotary_pos_embed.py
│   ├── residual_*.py, avg_down_wan.py, dup_up_wan.py
│   └── conv3d_configs.py
├── tests/
│   ├── pcc/                    # PCC (accuracy) tests vs reference
│   ├── perf/                   # Device perf + E2E timing (pytest)
│   ├── demo/                   # inference_ttnn.py, inference_torch.py, sample_images/
│   └── download_pretrained_weights.py
└── README.md
```

## Quick Start

### 1. Environment Setup

Run from the **tt-metal** repo root and put the tree on `PYTHONPATH`:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

source $TT_METAL_HOME/python_env/bin/activate
```

Install Python deps required by Lingbot-VA (diffusers, transformers, torch, ttnn stack, etc.) per your tt-metal environment.

### 2. Download Pretrained Weights

Checkpoints should follow the usual Wan layout under `reference/checkpoints/` with `vae/`, `tokenizer/`, `text_encoder/`, and `transformer/`.

```bash
# From tt-metal repo root (script writes under models/experimental/lingbot_va/reference/checkpoints)
python3 models/experimental/lingbot_va/tests/download_pretrained_weights.py
```

The script uses Hugging Face `snapshot_download`; ensure you have network access and `huggingface_hub` installed. You can also populate `reference/checkpoints/` manually from a compatible model repo.

## Running Tests

Run pytest from the **tt-metal** repo root with `PYTHONPATH=$TT_METAL_HOME` (or equivalent).

### PCC Tests (Accuracy)

| File | Test function(s) | What it checks |
|------|------------------|----------------|
| `tests/pcc/test_lingbot_va.py` | `test_lingbot_va` | End-to-end smoke via `run_inference` (checkpoint + device) |
| `tests/pcc/test_transformer_wan.py` | `test_wan_transformer_model`, `test_wan_transformer_model_action_mode` | TT vs reference transformer (video and action paths) |
| `tests/pcc/test_encoder_wan.py` | `test_umt5_encoder_comparison` | HF UMT5 vs TT encoder |
| `tests/pcc/test_vae_encoder.py` | `test_encode_one_video_pcc` | Torch encoder vs TT `WanVAEEncoder` |
| `tests/pcc/test_vae_decoder.py` | `test_decode_one_video_pcc` | Torch decode vs TT `WanVAEDecoder` |
| `tests/pcc/test_causal_conv_3d.py` | `test_wan_causal_conv3d` | Diffusers causal conv vs TT `WanCausalConv3d` (encoder `conv_in`) |

```bash
# One file
pytest models/experimental/lingbot_va/tests/pcc/test_lingbot_va.py -v

# All PCC tests in this model
pytest models/experimental/lingbot_va/tests/pcc/ -v

# Single test by node id (examples)
pytest models/experimental/lingbot_va/tests/pcc/test_transformer_wan.py::test_wan_transformer_model -v
pytest models/experimental/lingbot_va/tests/pcc/test_encoder_wan.py::test_umt5_encoder_comparison -v
```

### Performance Tests

| File | Test function | Notes |
|------|----------------|-------|
| `tests/perf/test_perf_ttnn_lingbot_va.py` | `test_perf_device_bare_metal_lingbot_va` | Device profiler; internally runs `pytest …/test_lingbot_va.py::test_lingbot_va` |
| `tests/perf/test_perf_e2e.py` | `test_e2e_perf` | End-to-end wall time via spawn-isolated `run_inference` |

```bash
pytest models/experimental/lingbot_va/tests/perf/test_perf_ttnn_lingbot_va.py -v -s
pytest models/experimental/lingbot_va/tests/perf/test_perf_e2e.py -v -s
```

Markers: `test_perf_ttnn_lingbot_va` uses `@pytest.mark.models_device_performance_bare_metal`; `test_perf_e2e` uses `@pytest.mark.models_performance_bare_metal`. Your CI may filter on these.

## Demo Scripts

Sample RobotWin camera PNGs live under `tests/demo/sample_images/robotwin/` (three files: `observation.images.cam_high.png`, `...cam_left_wrist.png`, `...cam_right_wrist.png`).

### Generate multi-chunk video (`demo.mp4`)

From the **tt-metal** repo root:

```bash
python3 models/experimental/lingbot_va/tests/demo/inference_ttnn.py \
  --checkpoint models/experimental/lingbot_va/reference/checkpoints/ \
  --images-dir models/experimental/lingbot_va/tests/demo/sample_images/robotwin/ \
  --prompt "Lift the cup from the table" \
  --generate
```

By default, `demo.mp4` is written under `tests/demo/` (see `--save-dir` to override). Use `--num-chunks` to change how many chunks are stitched into the video.

### Inference (single-chunk action)

Omit `--generate` to run reset + one infer chunk and print the action shape; optional `--output action.npy` saves the array.

```bash
python3 models/experimental/lingbot_va/tests/demo/inference_ttnn.py \
  --checkpoint models/experimental/lingbot_va/reference/checkpoints/ \
  --images-dir models/experimental/lingbot_va/tests/demo/sample_images/robotwin/ \
  --prompt "Lift the cup from the table"
```

You can set `LINGBOT_VA_CHECKPOINT` instead of `--checkpoint` when convenient.

## Troubleshooting

### Checkpoint not found

Download or sync weights into `reference/checkpoints/` (see **Download Pretrained Weights**), or point `LINGBOT_VA_CHECKPOINT` / `--checkpoint` at a directory that contains `vae`, `tokenizer`, `text_encoder`, and `transformer`.

### Missing sample images

Ensure `--images-dir` contains the three `observation.images.*.png` files, or set `LINGBOT_VA_E2E_IMAGES_DIR` for perf tests that use the same layout.

## Model Notes

| Area | Notes |
|------|--------|
| Transformer | Wan-style 3D blocks; Lingbot `in_channels=48`, action head, UMT5 conditioning |
| VAE | Wan 2.x causal encoder/decoder; TT path uses BTHWC layouts and conv blocking tuned for Wormhole |
| Text | UMT5 encoder; TT port in `models.tt_dit.encoders.umt5` |
| Demo TT entry | `tests/demo/inference_ttnn.py` |

## License

SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
