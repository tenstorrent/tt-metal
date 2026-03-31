# Lingbot-VA for Tenstorrent

### Platforms: Wormhole (n150)

### Supported input — vision (RobotWin): `(256, 320)` = (Height, Width) **per camera**, RGB **uint8**; three fixed keys `observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist` (e.g. `observation.images.<key>.png` under `--images-dir`; layout matches `VA_CONFIGS["robotwin"]` in `reference/utils.py`).

### Supported input — language: Natural-language **task prompt** (instruction string); **512** token limit after preprocessing (tokenizer runs on CPU — see **PyTorch / CPU components**).

## Introduction

Lingbot-VA is a vision–language–action stack built on the Wan family of models: it encodes multi-camera video with a causal VAE, conditions on UMT5 text embeddings, and runs a 3D Wan transformer for video latents and action tokens under flow-matching schedulers. This tree contains a **PyTorch reference**, the **TTNN implementation** used on device, **PCC / perf tests**, and **demo scripts** that mirror the server inference API without a WebSocket.

## Architecture

At a high level (no GPU path; reference and prep run on **CPU**, model math on **Tenstorrent** via TTNN):

- **Inputs:** Three RGB camera streams (RobotWin naming), task prompt, and optional proprioceptive state; observations match the Lingbot-VA server contract.
- **VAE:** `AutoencoderKLWan`-compatible encoder/decoder; TT path uses causal 3D convs, residual down/up blocks, and (where applicable) Lingbot-specific blocking configs.
- **Text:** **UMT5** encoder (HF checkpoint) on TTNN for embeddings used in cross-attention; **tokenization** stays on CPU (see **PyTorch / CPU components**).
- **Backbone:** `WanTransformer3DModel` with self-attention + cross-attention blocks, RoPE over a 3D grid, dual paths for **video latents** vs **action** tokens, and patch embedding for `(C, F, H, W)`.
- **Schedulers:** Flow-matching style stepping for video and action branches (configurable step counts in demo/tests).
- **Outputs:** Per-chunk **actions** (infer mode) or decoded `**demo.mp4`** (multi-chunk generate mode). TT demo writes video next to the demo script by default unless `--save-dir` is set.

**Key details:**

- Transformer and VAE TT code live under `tt/`; weights are mapped from the reference checkpoints.
- Demo entrypoint is `**tests/demo/demo.py`** (TTNN path); `tests/demo/inference_torch.py` is a CPU PyTorch reference runner for comparison/debug.

## Performance


| Kind                                                   | Measured value |
| ------------------------------------------------------ | -------------- |
| **Device** (`tests/perf/test_perf_ttnn_lingbot_va.py`) | 0.5            |
| **End-to-end** (`tests/perf/test_perf_e2e.py`)         | 5.68           |


## PyTorch / CPU components

Some steps stay on **CPU via PyTorch / Hugging Face** while the TT demo runs the backbone on Tenstorrent (similar to other `models/experimental/`** stacks that call out non-TTNN paths explicitly):


| Component          | Runtime                      | Notes                                                                                                                                                                                                               |
| ------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Text tokenizer** | **PyTorch (`transformers`)** | `**T5TokenizerFast`** from the checkpoint `tokenizer/` directory, loaded by `load_tokenizer()` in `reference/utils.py`. Runs on **CPU** to produce `input_ids` and attention masks for the **UMT5** encoder (TTNN). |


## Directory Structure

```
lingbot_va/
├── reference/                  # PyTorch reference (diffusers-compatible Wan pieces, utils, configs)
│   ├── transformer_wan.py      # WanTransformer3DModel (reference)
│   ├── utils.py                  # Loaders, schedulers, RobotWin config, VAE streaming helper
│   └── checkpoints/            # Weights directory
├── tt/                         # TTNN implementation
│   ├── attention_wan.py         # Self-/cross-attention for Wan blocks
│   ├── avg_down_wan.py          # Avg-pool downsample helper
│   ├── conv3d_configs.py        # VAE conv3d blocking overrides (Wormhole L1)
│   ├── dup_up_wan.py            # Duplicate upsample helper
│   ├── residual_block.py        # Residual block (mid)
│   ├── residual_down_block.py   # Residual down block (encoder path)
│   ├── residual_up_block.py     # Residual up block (decoder path)
│   ├── transformer_wan.py       # Wan transformer (TTNN)
│   ├── utils.py                 # TT loaders (transformer, text encoder, VAE encode/decode wrappers)
│   ├── vae_decoder.py           # Wan VAE decoder (TTNN)
│   ├── vae_encoder.py           # Wan VAE encoder (TTNN)
│   └── wan_rotary_pos_embed.py  # 3D RoPE for Wan
├── tests/
│   ├── pcc/                    # PCC (accuracy) tests vs reference
│   ├── perf/                   # Device perf + E2E pipeline perf (pytest)
│   ├── demo/                   # demo.py, inference_torch.py, sample_images/
│   └── download_pretrained_weights.py # Script to download the pretrained weights
├── PR_SUMMARY.md               # PR / release template (problem, PCC table, host paths)
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


| File                                | Test function(s)                                                       | What it checks                                                    |
| ----------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `tests/pcc/test_transformer_wan.py` | `test_wan_transformer_model`, `test_wan_transformer_model_action_mode` | TT vs reference transformer (video and action paths)              |
| `tests/pcc/test_encoder_wan.py`     | `test_umt5_encoder_comparison`                                         | HF UMT5 vs TT encoder                                             |
| `tests/pcc/test_vae_encoder.py`     | `test_encode_one_video_pcc`                                            | Torch encoder vs TT `WanVAEEncoder`                               |
| `tests/pcc/test_vae_decoder.py`     | `test_decode_one_video_pcc`                                            | Torch decode vs TT `WanVAEDecoder`                                |
| `tests/pcc/test_causal_conv_3d.py`  | `test_wan_causal_conv3d`                                               | Diffusers causal conv vs TT `WanCausalConv3d` (encoder `conv_in`) |


#### PCC scores

TTNN vs PyTorch reference; values are **PCC × 100** (%).


| Path        | PCC (%) |
| ----------- | ------- |
| Action path | 99.9966 |
| Video path  | 99.9684 |


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


| File                                      | Test function                            | Notes                                                                                                              |
| ----------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `tests/perf/test_perf_ttnn_lingbot_va.py` | `test_perf_device_bare_metal_lingbot_va` | Device profiler (Tracy); nested run of `test_lingbot_va_ttnn_forward_run`                                          |
| `tests/perf/test_perf_e2e.py`             | `test_perf_lingbot_va_e2e_2cq_no_trace`  | `TtLingbotVA` + `tt_cnn` pipeline, 2 command queues, no trace; requires checkpoints under `reference/checkpoints/` |


**Device perf (Tracy / bare-metal):**

```bash
pytest models/experimental/lingbot_va/tests/perf/test_perf_ttnn_lingbot_va.py::test_perf_device_bare_metal_lingbot_va -v -s
```

**End-to-end perf (pipeline wall-clock, `prep_perf_report`):**

```bash
pytest models/experimental/lingbot_va/tests/perf/test_perf_e2e.py::test_perf_lingbot_va_e2e_2cq_trace -v -s
```

## Demo Scripts

Sample RobotWin camera PNGs live under `tests/demo/sample_images/robotwin/` (three files: `observation.images.cam_high.png`, `...cam_left_wrist.png`, `...cam_right_wrist.png`).

### Generate multi-chunk video (`demo.mp4`)

From the **tt-metal** repo root:

```bash
python3 models/experimental/lingbot_va/tests/demo/demo.py \
  --checkpoint models/experimental/lingbot_va/reference/checkpoints/ \
  --images-dir models/experimental/lingbot_va/tests/demo/sample_images/robotwin/ \
  --prompt "Use an arm to place the smooth blue drinking cup on the wooden coaster" \
  --generate
```

By default, `demo.mp4` is written under `tests/demo/` (see `--save-dir` to override). Use `--num-chunks` to change how many chunks are stitched into the video.

## Troubleshooting

### Checkpoint not found

Download or sync weights into `reference/checkpoints/` (see **Download Pretrained Weights**), or point `LINGBOT_VA_CHECKPOINT` / `--checkpoint` at a directory that contains `vae`, `tokenizer`, `text_encoder`, and `transformer`.

### Missing sample images

Ensure `--images-dir` contains the three `observation.images.*.png` files, or set `LINGBOT_VA_E2E_IMAGES_DIR` for perf tests that use the same layout.

## Known limitations

1. **PyTorch reference runtime:** Running the full **PyTorch reference** stack to completion can take a long time, so it is not always practical to drive bit-for-bit comparisons from an on-demand reference run on the same box.
2. **PCC and intermediate dumps:** For several checks, **TT outputs and PCC are validated against intermediate tensors** produced by the PyTorch reference on a **separate host** (saved dumps from that run), rather than from a freshly executed reference path collocated with every TT invocation.
3. **tt-perf-report:** When generating reports from the device perf test, some versions of tt-perf-report crash in evaluate_fidelity with KeyError: 'FLOAT32' because the matmul advice path does not list FLOAT32 in its internal datatype → mantissa lookup.

## Model Notes


| Area          | Notes                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------- |
| Transformer   | Wan-style 3D blocks; Lingbot `in_channels=48`, action head, UMT5 conditioning                   |
| VAE           | Wan 2.x causal encoder/decoder; TT path uses BTHWC layouts and conv blocking tuned for Wormhole |
| Text          | UMT5 encoder; TT port in `models.tt_dit.encoders.umt5`                                          |
| Demo TT entry | `tests/demo/demo.py`                                                                            |


## License

SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
