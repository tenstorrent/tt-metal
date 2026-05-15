# Mistral-24B (Vision + Text) on TT

Mistral-24B is a multimodal model that takes text prompts and optional images, then generates text responses token-by-token. On TT, the pipeline first converts images into visual features (vision tower + patch merger + multimodal projector), fuses those features with text embeddings during prefill, and then runs autoregressive decode to generate output tokens.

## Performance measurements

Measured on **Blackhole QuietBox-2 (BH QB-2)** with mesh device **`P150x4`** (logical shape `(1, 4)`.

End-to-end demo metrics:

| Metric                                | Without trace | With trace |
| ------------------------------------- | ------------- | ---------- |
| **prefill_tokens**                    | 522           | 522        |
| **inference_prefill** (s)             | 2.82139       | 3.29152    |
| **prefill_t/s** (tokens/s)            | 185.015       | 158.59     |
| **TTFT** (prefill to first token, ms) | 2821.39       | 3291.52    |
| **inference_decode** (s)              | 27.0651       | 13.9218    |
| **decode_t/s/u** (per-user)           | 17.4025       | 33.832     |
| **decode_t/s** (aggregate, batch=1)   | 17.4025       | 33.832     |
| **num_decode_tokens**                 | 471           | 471        |
| **Full demo runtime** (s)             | 29.9914       | 17.4228    |

## Command to run the end-to-end test

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py
```

## Commands to run vision pipeline tests

```bash
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_model.py
pytest models/experimental/mistral_24b/tests/pipeline_tests/test_vision_tower.py
```
