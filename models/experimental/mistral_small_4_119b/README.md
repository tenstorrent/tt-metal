# Mistral-Small-4-119B (Multimodal) on Tenstorrent

TTNN implementation of [`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) — a 119B-parameter (6.5B active/token) Mixture-of-Experts, vision-language model — running end to end on Tenstorrent Blackhole hardware.

The model takes **text + image** input and generates text. It pairs a Pixtral vision tower with a Multi-head Latent Attention (MLA) + MoE language decoder.

## Model architecture

```
Image ─▶ Pixtral Vision Tower (24 layers) ─▶ Multimodal Projector ─┐
                                              (spatial-merge ×2,    │
                                               GELU)                ▼
                                                         ┌──────────────────────┐
Text tokens ────────────────────── embed ──────────────▶│  Language Decoder     │──▶ logits ─▶ token
                                                         │  36 layers:           │
                                                         │   • MLA self-attn     │
                                                         │   • MoE (128 experts, │
                                                         │     4 active + 1      │
                                                         │     shared)           │
                                                         └──────────────────────┘
```

| Component | Details |
|-----------|---------|
| **Language decoder** | 36 layers, hidden size 4096, 32 attention heads, vocab 131072 |
| **Attention** | MLA — `q_lora_rank=1024`, `kv_lora_rank=256`, `qk_rope=64`, `qk_nope=64`, `v_head_dim=128`; latent (compressed) KV cache |
| **MoE** | 128 experts, 4 active + 1 always-on shared expert; expert intermediate size 2048 |
| **Vision tower** | Pixtral-style ViT, 24 layers, hidden 1024, 16 heads, patch size 14, 2D RoPE |
| **Projector** | Spatial-merge (size 2) + GELU projection to the text hidden dim |
| **RoPE θ** | text `1e6`, vision `1e4` |
| **License** | Apache 2.0 |

Architecture constants live in [constants.py](constants.py).

## Supported hardware

Blackhole only. The demo targets **P150×8** or a **Blackhole Loud Box**; the test suite also runs on smaller meshes. Select the mesh via the `MESH_DEVICE` env var (see [tests/mesh_param.py](tests/mesh_param.py)):

| `MESH_DEVICE` | Mesh shape | Notes |
|---------------|-----------|-------|
| `P300x2` | (1, 2) | test default |
| `P150x4` | (1, 4) | |
| `P150x8` | (1, 8) | demo target |
| `single` | (1, 1) | single chip; smoke tests only (fabric disabled) |
| `T3K` / `N300` | (1, 8) / (1, 2) | Wormhole, test-only |

## Setup

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

The model is gated on Hugging Face — request access on the model page, then authenticate so weights download on first run:

```bash
huggingface-cli login
```

Weights, config, tokenizer, and image processor are pulled from `HF_MODEL_ID` automatically (no manual repack step).

## Running the demo

[demo/demo_multimodal.py](demo/demo_multimodal.py) drives the full vision → projector → prefill → decode pipeline. Loading is unified: vision + projector + text are lazy-loaded together on the first `encode_image` call.

```bash
export MESH_DEVICE=P150x8

# Full model (36 text + 24 vision layers) on the default sample image + prompt
python models/experimental/mistral_small_4_119b/demo/demo_multimodal.py \
    --n-text-layers 36 --n-vision-layers 24

# Custom image (URL or local path) and prompt
python models/experimental/mistral_small_4_119b/demo/demo_multimodal.py \
    --n-text-layers 36 --n-vision-layers 24 \
    --image /path/to/image.jpg \
    --prompt "Describe the scene."
```

> The defaults run only **2 text + 2 vision layers** — fast (~minutes) but a plumbing check only, so the output is gibberish. Pass the full-layer flags above for real generation.

Key flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--image` | sample battle scene | Image path or URL; `""` → random tensor (pipeline-only) |
| `--prompt` | (built-in) | Text prompt accompanying the image |
| `--max-new-tokens` | 16 | Tokens to generate after prefill |
| `--n-text-layers` | 2 | Text decoder layers (1–36) |
| `--n-vision-layers` | 2 | Pixtral vision layers (1–24) |
| `--image-max-side` | 224 | Max image H/W (px) before the processor; larger = better, slower |
| `--no-chat-template` | off | Use raw `[image]+[prompt]` layout (plumbing only) |
| `--no-trace` | off | Disable decode-step trace capture (eager path; slower, for debugging) |
| `--backend` | `ttnn` | `ttnn`, `hf` (Torch reference), or `both` (compare) |

## Tests

PCC and smoke tests are gated behind per-test env flags (they need hardware and download weights). Each parametrizes the `mesh_device` fixture, so set `MESH_DEVICE` too.

**Correctness (PCC vs HF Torch reference):**

```bash
MISTRAL4_VISION_PCC=1  pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_pcc.py
MISTRAL4_MMP_PCC=1     pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_projector_pcc.py
MISTRAL4_PREFILL_PCC=1 pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_pcc.py
MISTRAL4_DECODE_PCC=1  pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_pcc.py
MISTRAL4_MM_PCC=1      pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc_unified.py
```

**Smoke / plumbing (no PCC, fast):**

```bash
MISTRAL4_VISION_SMOKE=1    pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_smoke.py
MISTRAL4_PREFILL_SMOKE=1   pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_smoke.py
MISTRAL4_DECODE_SMOKE=1    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py
MISTRAL4_LANG_DEMO_SMOKE=1 pytest models/experimental/mistral_small_4_119b/tests/test_language_demo_smoke.py
```

**Performance:**

```bash
# End-to-end wall-clock (TTFT, prefill tok/s, steady-state decode tok/s/user)
pytest models/experimental/mistral_small_4_119b/tests/perf/test_e2e_performant.py -m models_performance_bare_metal

# Per-op device performance (single layer)
pytest models/experimental/mistral_small_4_119b/tests/perf/test_perf.py -m models_device_performance_bare_metal
```

The e2e perf test isolates steady-state timing (compile/load passes excluded). Decode is trace-captured and replayed over 2 command queues (`decode_next_token_2cq`); prefill is not trace-captured.

## Directory layout

```
mistral_small_4_119b/
├── README.md                       # this file
├── constants.py                    # architecture dimensions / HF model id
├── demo/
│   └── demo_multimodal.py          # end-to-end multimodal generation demo
├── tt/                             # TTNN (Tenstorrent) implementation
│   ├── mistral3_for_conditional_generation_unified.py  # top-level orchestrator (unified load)
│   ├── mistral3_for_conditional_generation_helpers.py
│   ├── mistral4_text_model.py      # text decoder: embed, decode path, lm_head
│   ├── mistral4_text_prefill.py    # text prefill path
│   ├── mistral4_self_attention.py  # MLA self-attention (latent KV cache)
│   ├── mistral4_moe.py             # MoE: 128 experts, 4 active + shared
│   ├── mistral4_multimodal_projector.py
│   ├── mistral4_vision_tower.py    # Pixtral vision tower
│   ├── mistral4_vision_attention.py
│   ├── mistral4_vision_mlp.py
│   ├── mistral4_vision_rope.py     # vision 2D RoPE
│   └── vision_matmul_config.py     # vision matmul program configs
└── tests/
    ├── mesh_param.py               # MESH_DEVICE → mesh-shape resolution
    ├── test_*_pcc.py               # PCC correctness vs HF Torch
    ├── test_*_smoke.py             # fast plumbing checks
    └── perf/                       # e2e + device-perf tests
```
