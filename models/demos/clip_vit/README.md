# CLIP-ViT

## Platforms
Wormhole (n300)

## Introduction
CLIP (ViT-B/32) was developed by OpenAI. This directory contains a TTNN implementation
(`tt/`) with both a baseline and an optimized variant (block-sharded, bfloat8_b, per-op
program configs), PCC tests against HuggingFace, throughput
benchmarks, and a demo.

## How to Run

### PCC Tests

All PCC tests live in a single file: `tests/test_pcc.py`. Base model tests use a
threshold of 0.98, optimized model tests use 0.90.

Optimized model passes pcc with using bfloat16 dtype. There is an average 5% dropoff
of pcc when using bfloat8_b, but you gain ~2x throughput.

Run all PCC tests:
```bash
pytest models/demos/clip_vit/tests/test_pcc.py -sv
```

Run a single test by name:
```bash
pytest models/demos/clip_vit/tests/test_pcc.py::test_pcc_vision_model -sv
pytest models/demos/clip_vit/tests/test_pcc.py::test_pcc_opt_vision_model -sv
```

Run only base or only optimized tests:
```bash
pytest models/demos/clip_vit/tests/test_pcc.py -k "not opt" -sv
pytest models/demos/clip_vit/tests/test_pcc.py -k "opt" -sv
```

Filter by component:
```bash
pytest models/demos/clip_vit/tests/test_pcc.py -k "vision_mlp" -sv
pytest models/demos/clip_vit/tests/test_pcc.py -k "text and not full" -sv
pytest models/demos/clip_vit/tests/test_pcc.py -k "full_model" -sv
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

Throughput (images/sec at various total-image counts):
```bash
pytest models/demos/clip_vit/tests/test_e2e_throughput.py -sv
```

### Demo

Runs CLIP zero-shot classification on 7 COCO images against 7 candidate labels,
prints top-1 predictions, and asserts TTNN top-1 matches HuggingFace.

Base model (bfloat16), the default:
```bash
pytest models/demos/clip_vit/demo.py -sv
```

Optimized model (choose one dtype):
```bash
pytest models/demos/clip_vit/demo.py -sv --opt --b16
pytest models/demos/clip_vit/demo.py -sv --opt --b8
```

## Details
- TTNN model (baseline): `tt/tt_clip_model.py`, `tt/tt_clip_text.py`, `tt/tt_clip_vision.py`
- TTNN model (optimized): `tt/tt_clip_model_optimized.py`, `tt/tt_clip_text_optimized.py`,
  `tt/tt_clip_vision_optimized.py`
- PCC tests: `tests/test_pcc.py` (all base + optimized component tests in one file)
- Reference: HuggingFace `openai/clip-vit-base-patch32` (loaded via `transformers`)

## Optimizations

A throughput-oriented strategy focused on maximizing batch size and keeping activations
resident on-chip across the full encoder stack.

- **Large L1-resident batches.** Activations are block-sharded across the full compute grid
  and sized so the working set fits in Tensix L1. This lets batches of 35 (bfloat16) or 63
  (bfloat8_b) stay on-chip across all 12 encoder layers without eviction.
- **Weights stream from DRAM per op.** Only weights traverse the DRAM↔L1 bus; activations
  never leave L1 mid-encoder.
- **Per-op program configs.** Each matmul, layernorm, and SDPA has a tuned block-shard +
  subblock configuration matched to its shard layout.
- **Fused ops.** QKV is a single fused matmul; GELU is fused into the first MLP matmul;
  attention uses flash-style SDPA.
- **bfloat8_b mode.** Halving activation/weight precision roughly doubles the usable batch
  and halves DRAM traffic, trading ~5% PCC for ~2x throughput.
- **Sequential text-then-vision.** Each modality processes all its batches back-to-back
  before the other starts, then similarity scores are computed in one pass.


## References
- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIPModel](https://huggingface.co/openai/clip-vit-base-patch32)
