# Mistral-Small-3.1-24B (Vision + Text) on TT

Mistral-Small-3.1-24B-Instruct-2503 is a multimodal model that accepts text prompts and optional images and generates text responses token-by-token. It is a 24B-parameter instruction-tuned model built on the Mistral architecture with an integrated Pixtral vision encoder.

HuggingFace model card: [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

---

## Supported Device

| Hardware | Mesh Shape | Config |
|---|---|---|
| Blackhole QuietBox-2 (BH QB-2) | `P150x4` — logical `(1, 4)` |

> The full end-to-end pipeline and all performance numbers are measured on BH QB-2.

---

---

## File Structure

```
models/experimental/mistral_24b/
├── README.md
├── tt/
│   ├── model.py                        # E2E pipeline implementation
│   ├── generator.py                    # MistralGenerator — prefill/decode orchestration
│   ├── rmsnorm.py                      # TT RMSNorm
│   ├── vision_attention.py             # Vision Attention block
│   ├── vision_conv2d.py                # Patch embedding conv2d
│   ├── vision_mlp.py                   # Vision MLP block
│   ├── vision_mmp.py                   # Multimodal projector (vision → text dim)
│   ├── vision_pixtral_image_block.py   # Single Pixtral image transformer block
│   ├── vision_pixtral_transformer.py   # Full Pixtral vision transformer
│   ├── vision_rope.py                  # Vision RoPE block
│   └── pipeline/
│       ├── mistral_vision_tower.py     # Vision tower module
│       └── vision_model.py             # TtMistralVisionTransformer
└── tests/
    ├── pipeline_tests/
    │   ├── test_end2end.py                     # End-to-end vision-text demo
    │   ├── test_isl_sweep.py                   # Context-window sweep (1k–128k)
    │   ├── test_text_decoder.py                # Decode logits PCC
    │   ├── test_text_prefill_logits.py         # Prefill last-token logits PCC
    │   ├── test_vision_model.py                # Vision model PCC
    │   └── test_vision_tower.py                # Vision tower PCC
    └── (unit tests)
        ├── test_conv2d.py                      # Vision module unit tests
        ├── test_patch_rot_emb.py
        ├── test_pixtral_transformer.py
        ├── test_vision_attention.py
        ├── test_vision_mlp.py
        ├── test_vision_rms.py
        └── test_device_perf_single_layer.py    # Single-layer prefill/decode test
```

---

## Installation

```bash
# 1. Clone tt-metal with submodules
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal

# 2. Install TT-Metalium / TT-NN
# Follow: https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md

# 3. Set model environment variable
export HF_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503

# 4. Set mesh device (for BH QB-2)
export MESH_DEVICE=P150x4
```

---

## Tests

### Demo — End-to-End Vision-Text Pipeline

Runs a full prefill (image + text) followed by autoregressive decode. Validates that the pipeline produces tokens; logs TTFT and throughput.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py
```

---

### Accuracy — Text Decoder Logits PCC

Compares full decode logits (40 layers + norm + lm_head) against HF using synthetic random activations for 32 decode steps. PCC threshold: **≥ 0.98**.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_text_decoder.py
```

---

### Accuracy — Text Prefill Logits PCC

Full-depth prefill last-token logits (40 layers + norm + lm_head) vs HuggingFace over
increasing sequence lengths from *Tale of Two Cities* (128 → 128k tokens).
PCC threshold: **≥ 0.97**.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_text_prefill_logits.py
```

---

### Accuracy — Vision Model PCC

Validates the on-device Pixtral vision transformer output against HF reference.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_model.py
```

---

### Accuracy — Vision Tower PCC

Validates the full vision tower (Conv2D patch embed + Pixtral transformer + MMP projection) against HF reference.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_tower.py
```

---

### Device profiling — Tracy CSV

Same as the text prefill/decode tests with `n_layers=1`.

```bash
HF_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503 python -m tracy -p -r -v -m pytest models/experimental/mistral_24b/tests/pipeline_tests/test_device_perf_single_layer.py::test_single_layer_prefill -s

HF_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503 python -m tracy -p -r -v -m pytest models/experimental/mistral_24b/tests/pipeline_tests/test_device_perf_single_layer.py::test_single_layer_decode -s
```

---

### Performance — ISL Sweep (1k–128k context window)

Sweeps input sequence length through the full vision-text pipeline. Text prompt is sourced from the *Tale of Two Cities* corpus; one image is always included. Output is fixed at **200 tokens** per sweep point.

The sweep starts at **1k**: every run includes the vision tower, which produces ~1024 image tokens plus chat-template overhead. A context window below ~1k cannot fit image + text, so those points are skipped in `test_isl_sweep.py`.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_isl_sweep.py
```

---

### Unit Tests — Vision Submodules

Individual op-level PCC tests for vision submodules:

```bash
pytest models/experimental/mistral_24b/tests/test_conv2d.py
pytest models/experimental/mistral_24b/tests/test_patch_rot_emb.py
pytest models/experimental/mistral_24b/tests/test_vision_rms.py
pytest models/experimental/mistral_24b/tests/test_vision_mlp.py
pytest models/experimental/mistral_24b/tests/test_vision_attention.py
pytest models/experimental/mistral_24b/tests/test_pixtral_transformer.py
```

---

## Performance

Measured on **Blackhole QuietBox-2 (BH QB-2)** with mesh device `P150x4` (logical shape `(1, 4)`).

### End-to-End Demo

| Metric | Without trace | With trace |
|---|---|---|
| **prefill_tokens** | 522 | 522 |
| **inference_prefill** (s) | 2.82139 | 3.29152 |
| **prefill_t/s** (tokens/s) | 185.015 | 148.59 |
| **TTFT** (prefill to first token, ms) | 2821.39 | 3291.52 |
| **inference_decode** (s) | 27.0651 | 13.9218 |
| **decode_t/s/u** (per-user) | 17.4025 | 33.63 |
| **decode_t/s** (aggregate, batch=1) | 17.4025 | 33.63 |
| **num_decode_tokens** | 471 | 471 |
| **Full demo runtime** (s) | 29.9914 | 17.4228 |

### ISL (Context Window) Sweep
| Config | Batch | ISL (max_seq_len) | Prefill tokens | decode_t/s/u (tok/s) | decode_t/s (tok/s) |
|---|---|---|---|---|---|
| b1_isl1k | 1 | 1k (1024) | 506 | 34.05 | 34.05 |
| b1_isl2k | 1 | 2k (2048) | 1058 | 33.82 | 33.82 |
| b1_isl4k | 1 | 4k (4096) | 3106 | 33.29 | 33.29 |
| b1_isl8k | 1 | 8k (8192) | 7202 | 32.66 | 32.66 |
| b1_isl16k | 1 | 16k (16384) | 15394 | 31.44 | 31.44 |
| b1_isl32k | 1 | 32k (32768) | 31778 | 29.18 | 29.18 |
| b1_isl64k | 1 | 64k (65536) | 64546 | 25.52 | 25.52 |
| b1_isl128k | 1 | 128k (131072) | 130082 | 20.41 | 20.41 |

> Note: decode throughput varies from ~33 tok/s/u to ~20 tok/s/u as context length increases (1k → 128k).

---

## Prefill Logits PCC — Last Token

Results from `test_text_prefill_logits.py` on **BH QB-2 (`P150x4`)**. *Tale of Two Cities* tokens; full model (40 layers + norm + lm_head). PCC threshold: **≥ 0.97**.

| seq_len | PCC |
|---:|------:|
| 128 | 0.996038 |
| 256 | 0.998660 |
| 1024 | 0.994131 |
| 3072 | 0.995028 |
| 4096 | 0.991853 |
| 8192 | 0.981217 |
| 16384 | 0.985105 |
| 32k, 64k, 128k | TBD |

> Note: larger context lengths take longer to verify because the HuggingFace reference runs a full 24B forward pass over the entire sequence.

---

## Decode Logits PCC — 32 Generation Steps

Results from `test_text_decoder.py` on **BH QB-2 (`P150x4`)**. Synthetic random activations; full model (40 layers + norm + lm_head). Per-step PCC threshold: **≥ 0.98**.

| Step | PCC |
|---:|------:|
| 0 |  0.998570 |
| 1 |  0.998120 |
| 2 |  0.998177 |
| 3 |  0.998114 |
| 4 |  0.997983 |
| 5 |  0.998259 |
| 6 |  0.998331 |
| 7 |  0.998045 |
| 8 |  0.998336 |
| 9 |  0.997582 |
| 10 | 0.997241 |
| 11 | 0.997598 |
| 12 | 0.997913 |
| 13 | 0.997852 |
| 14 | 0.997882 |
| 15 | 0.997420 |
| 16 | 0.997740 |
| 17 | 0.997481 |
| 18 | 0.998059 |
| 19 | 0.998072 |
| 20 | 0.997810 |
| 21 | 0.997656 |
| 22 | 0.997532 |
| 23 | 0.997628 |
| 24 | 0.997396 |
| 25 | 0.997667 |
| 26 | 0.996565 |
| 27 | 0.996754 |
| 28 | 0.997992 |
| 29 | 0.997651 |
| 30 | 0.997734 |
| 31 | 0.997757 |

---

## Vision PCC

Results from vision pipeline tests on **BH QB-2 (`P150x4`)**. Random `vision_chunk_size` image input vs HuggingFace reference.

| Component | Test | PCC |
|---|---|---:|---:|
| Vision model | `test_vision_model.py` | 0.990847 |
| Vision tower | `test_vision_tower.py` | 0.996050 |

---

## Open items

1. Prefill logits PCC verification at context lengths ≥32k (32k, 64k, 128k).
2. ISL sweep performance analysis at 64k and 128k context lengths (~25 and ~20 tok/s/u decode throughput, respectively).
