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
    │   ├── test_isl_sweep.py                   # Context-window sweep (4k–128k)
    │   ├── test_text_decoder.py                # Decode hidden-states PCC (32 steps)
    │   ├── test_text_decoder_decode_logits.py  # Decode logits PCC — 32 steps vs HF reference
    │   ├── test_vision_model.py                # Vision model PCC
    │   └── test_vision_tower.py                # Vision tower PCC
    └── (unit tests)
        ├── test_conv2d.py                      # Vision module unit tests
        ├── test_patch_rot_emb.py
        ├── test_pixtral_transformer.py
        ├── test_vision_attention.py
        ├── test_vision_mlp.py
        └── test_vision_rms.py
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

### Accuracy — Decode Logits PCC (32 generation steps)

After a 128-token real-text prefill, compares per-step decode logits (full model: transformer layers + norm + lm_head) against HuggingFace reference for **32 consecutive generation steps**.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_text_decoder_decode_logits.py
```

---

### Accuracy — Text Decoder Hidden-States PCC

Compares all 40 decoder layer hidden-state outputs (before norm + lm_head) against HF using synthetic random activations for 32 decode steps. PCC threshold: **≥ 0.98**.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_text_decoder.py
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

### Performance — ISL Sweep (4k–128k context window)

Sweeps input sequence length through the full vision-text pipeline. Text prompt is sourced from the *Tale of Two Cities* corpus; one image is always included. Output is fixed at **200 tokens** per sweep point.

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

---

## Decode Logits PCC — 32 Generation Steps

Results from `test_text_decoder_decode_logits.py`. Prefill: 128 tokens from *Tale of Two Cities*. Prefill PCC threshold: ≥ 0.99. Per-step decode PCC threshold: ≥ 0.97.

> Values below are placeholders — update after running the test.

| Step | Position | PCC |
|---:|---:|---:|
| 0 | 128 | 0.994161 |
| 1 | 129 | 0.987024 |
| 2 | 130 | 0.993026 |
| 3 | 131 | 0.996192 |
| 4 | 132 |  0.996960|
| 5 | 133 |  0.997819|
| 6 | 134 |  0.996799|
| 7 | 135 |  0.997560|
| 8 | 136 |  0.996585|
| 9 | 137 |  0.997587|
| 10 | 138 | 0.998238|
| 11 | 139 | 0.997792|
| 12 | 140 | 0.997527|
| 13 | 141 | 0.997662|
| 14 | 142 | 0.998111|
| 15 | 143 |  0.993457|
| 16 | 144 |  0.992316|
| 17 | 145 |  0.982824|
| 18 | 146 |  0.994470|
| 19 | 147 |  0.996441|
| 20 | 148 |  0.998085|
| 21 | 149 |  0.997162|
| 22 | 150 |  0.997821|
| 23 | 151 |  0.998436|
| 24 | 152 |  0.996853|
| 25 | 153 |  0.996763|
| 26 | 154 |  0.997244|
| 27 | 155 |  0.997038|
| 28 | 156 |  0.996458|
| 29 | 157 |  0.997501|
| 30 | 158 |  0.997934|
| 31 | 159 |  0.996043|
| **Prefill last-token** | **127** | **—** | **—** |

---

## Torch Fallbacks

The following operation run on host CPU (PyTorch) rather than on-device (TTNN):
`torch.nn.Unfold` — Patch extraction in `tt/vision_conv2d.py`
