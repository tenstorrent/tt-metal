# CLIP-ViT

## Platforms
Wormhole (n300)

## Introduction
CLIP (ViT-B/32) was developed by OpenAI. This directory contains a TTNN implementation
(`tt/`) with both a baseline and an optimized variant (block-sharded, bfloat8_b, per-op
program configs), PCC tests against HuggingFace, end-to-end latency/throughput
benchmarks, and a demo.

## How to Run

### PCC Tests

All PCC tests live in a single file: `tests/pcc/test_pcc.py`. Base model tests use a
threshold of 0.98, optimized model tests use 0.90.

Run all PCC tests:
```bash
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -sv
```

Run a single test by name:
```bash
pytest models/demos/clip_vit/tests/pcc/test_pcc.py::test_pcc_vision_model -sv
pytest models/demos/clip_vit/tests/pcc/test_pcc.py::test_pcc_opt_vision_model -sv
```

Run only base or only optimized tests:
```bash
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -k "not opt" -sv
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -k "opt" -sv
```

Filter by component:
```bash
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -k "vision_mlp" -sv
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -k "text and not full" -sv
pytest models/demos/clip_vit/tests/pcc/test_pcc.py -k "full_model" -sv
```

Available test functions:

| Base | Optimized |
|------|-----------|
| `test_pcc_vision_embeddings` | `test_pcc_opt_vision_embeddings` |
| `test_pcc_vision_mlp` | `test_pcc_opt_vision_mlp` |
| `test_pcc_vision_attention` | `test_pcc_opt_vision_attention` |
| `test_pcc_vision_encoder_layer` | `test_pcc_opt_vision_encoder_layer` |
| `test_pcc_vision_model` | `test_pcc_opt_vision_model` |
| `test_pcc_text_embeddings` | `test_pcc_opt_text_embeddings` |
| `test_pcc_text_mlp` | `test_pcc_opt_text_mlp` |
| `test_pcc_text_attention` | `test_pcc_opt_text_attention` |
| `test_pcc_text_encoder_layer` | `test_pcc_opt_text_encoder_layer` |
| `test_pcc_text_model` | `test_pcc_opt_text_model` |
| `test_pcc_full_model` | `test_pcc_opt_full_model` |

### Performance Benchmarks

Latency (median/p95 for the full model, base + optimized):
```bash
pytest models/demos/clip_vit/tests/perf/test_e2e_latency.py -sv
```

Throughput (images/sec at various total-image counts):
```bash
pytest models/demos/clip_vit/tests/perf/test_e2e_throughput.py -sv
```

### Demo

Runs CLIP on real images + candidate labels, prints top-1 predictions, and asserts
TTNN top-1 matches HuggingFace:
```bash
pytest models/demos/clip_vit/demo/demo.py -sv
```

## Details
- TTNN model (baseline): `tt/tt_clip_model.py`, `tt/tt_clip_text.py`, `tt/tt_clip_vision.py`
- TTNN model (optimized): `tt/tt_clip_model_optimized.py`, `tt/tt_clip_text_optimized.py`,
  `tt/tt_clip_vision_optimized.py`
- PCC tests: `tests/pcc/test_pcc.py` (all base + optimized component tests in one file)
- Reference: HuggingFace `openai/clip-vit-base-patch32` (loaded via `transformers`)

## References
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIPModel](https://huggingface.co/openai/clip-vit-base-patch32)
