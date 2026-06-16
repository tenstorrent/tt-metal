# Mistral-24B (Vision + Text) on TT

Mistral-24B is a multimodal model that takes text prompts and optional images, then generates text responses token-by-token. On TT, the pipeline first converts images into visual features (vision tower + patch merger + multimodal projector), fuses those features with text embeddings during prefill, and then runs autoregressive decode to generate output tokens.

The model is validated on **Blackhole QuietBox-2** (`P150x4` / shape `(1, 4)`).

## Performance measurements

Measured on **Blackhole QuietBox-2 (BH QB-2)** with mesh device `P150x4` (logical shape `(1, 4)`).

End-to-end demo metrics:


| Metric                                | Without trace | With trace |
| ------------------------------------- | ------------- | ---------- |
| **prefill_tokens**                    | 522           | 522        |
| **inference_prefill** (s)             | 2.82139       | 3.29152    |
| **prefill_t/s** (tokens/s)            | 185.015       | 148.59     |
| **TTFT** (prefill to first token, ms) | 2821.39       | 3291.52    |
| **inference_decode** (s)              | 27.0651       | 13.9218    |
| **decode_t/s/u** (per-user)           | 17.4025       | 33.63     |
| **decode_t/s** (aggregate, batch=1)   | 17.4025       | 33.63     |
| **num_decode_tokens**                 | 471           | 471        |
| **Full demo runtime** (s)             | 29.9914       | 17.4228    |


## ISL (context window) sweep

Sweeps the input sequence length / context window through the end-to-end vision-text
pipeline (image prefill via the vision tower + autoregressive decode). The text prompt is
sourced from the *Tale of Two Cities* corpus (the same long-context input the tt_transformers
prefill tests use) and sliced so the input length scales with the swept context window; the
image is always included. Output length is fixed at **200** tokens per sweep point.

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_isl_sweep.py
```

Each sweep point logs throughput under `=== Performance metrics ===`. After all points
complete, the test logs an `ISL sweep decode throughput summary` table.

Measured on **Blackhole QuietBox-2 (BH QB-2)** with mesh device `P150x4` (`bfloat8_b`,
accuracy decoder precision, paged attention, `block_size=32`). The paged KV-cache block pool
is scaled per sweep point to hold the full context plus decode headroom (e.g. 4103 blocks × 32
= **131296 positions** at ISL 128k).


| Config     | Batch | ISL (max_seq_len) | prefill tokens | decode_t/s/u (tok/s) | decode_t/s (tok/s) |
| ---------- | ----- | ----------------- | -------------- | -------------------- | ------------------ |
| b1_isl4k   | 1     | 4k (4096)         | 3106           | 33.29                | 33.29              |
| b1_isl8k   | 1     | 8k (8192)         | 7202           | 32.66                | 32.66              |
| b1_isl16k  | 1     | 16k (16384)       | 15394          | 31.44                | 31.44              |
| b1_isl32k  | 1     | 32k (32768)       | 31778          | 29.18                | 29.18              |
| b1_isl64k  | 1     | 64k (65536)       | 64546          | 25.52                | 25.52              |
| b1_isl128k | 1     | 128k (131072)     | 130082         | 20.41                | 20.41              |


Decode throughput decreases with ISL (33.29 → 20.41 tok/s/u from 4k → 128k) as the
KV-attention cost grows with context length. For batch 1, `decode_t/s/u` and `decode_t/s` are
equal.

Notes:

- ISL **> 32k** requires scaling `page_max_num_blocks` above the default 1024 (the test does
  this automatically per sweep point).

## Prerequisite

```bash
export HF_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

## End-to-end demo test

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py
```

## Pipeline tests

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_model.py
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_tower.py
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_text_decoder.py
```

## Vision unit tests

```bash
pytest models/experimental/mistral_24b/tests/test_conv2d.py
pytest models/experimental/mistral_24b/tests/test_patch_rot_emb.py
pytest models/experimental/mistral_24b/tests/test_vision_rms.py
pytest models/experimental/mistral_24b/tests/test_vision_mlp.py
pytest models/experimental/mistral_24b/tests/test_vision_attention.py
pytest models/experimental/mistral_24b/tests/test_pixtral_transformer.py
```
