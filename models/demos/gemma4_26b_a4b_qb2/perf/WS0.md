# WS0 · Baseline (A, 16 Nov 2026)

Branch: main of 2026-11-16 (e27c36d2a67) + Gemma 26B port synced to `models/demos/gemma4_26b_a4b_qb2` (dad6199475a). Precision `bfp8_release_selected`.

**Accuracy** (`tests/test_device_numerics.py`, real checkpoint vs HF, 65 tokens):

| Layer | Prefill PCC | Traced decode PCC |
|---|---|---|
| 0 sliding | 0.99882 | 0.99964 |
| 5 full | 0.99906 | 0.99988 |

**Native traced decode** (`tools/decode_timing.py`, whole run ~9 min):

| Batch | Step ms | tok/s/user | tok/s total |
|---|---|---|---|
| 1 (128 prompt) | 18.6 | 53.8 | 53.8 |
| 1 (4K prompt) | 19.0 | 52.5 | 52.5 |
| 8 | 129.4 | 7.7 | 61.8 |
| 16 | 254.7 | 3.9 | 62.8 |
| 32 | 501.0 | 2.0 | 63.9 |

TTFT 92 ms (128) / 1716 ms (4K). Batching gains nothing: every user runs through each layer separately.

**Reference on the same QB2:** `models/demos/gemma4` 26B native demo, batch 1: 47.7 tok/s/user (20.95 ms).
Its batch-32 demo decodes untraced (~100 ms/step), so it is not a serving comparison.
