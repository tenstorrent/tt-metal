# pplx-embed-v1-0.6B on Tenstorrent Blackhole

Text-embedding inference for [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)
on Tenstorrent Blackhole (P150 single device and 32× P150 Blackhole Galaxy via
data parallelism).

This README covers **how to run every script, what each one produces, the
measured performance/accuracy, and the optimizations applied.** For the detailed
engineering write-up (every experiment, rejected ideas, op-level analysis) see
[`README2.md`](README2.md).

---

## Model

pplx-embed-v1-0.6B is a Perplexity AI text-embedding model on the Qwen3-0.6B backbone.

| Property      | Value                       |
|---------------|-----------------------------|
| Parameters    | 0.6B                        |
| Hidden dim    | 1024                        |
| Layers        | 28                          |
| Q / KV heads  | 16 / 8                      |
| Head dim      | 128                         |
| FFN dim       | 3072                        |
| Max context   | 32K                         |
| Attention     | Bidirectional (non-causal)  |
| Pooling       | Mean over real tokens       |
| Output        | L2-normalized 1024-d vector |

It differs from Qwen3-Embedding-0.6B in three ways: **bidirectional attention**
(no causal mask), **mean-token pooling** (not last-token), and it requires
`trust_remote_code=True` for HuggingFace loading.

---

## 1. Setup

```bash
# From the tt-metal repo root, with the Python env active:
source python_env/bin/activate

export HF_MODEL=perplexity-ai/pplx-embed-v1-0.6b
export HF_HOME=$HOME/.cache/huggingface   # must be writable + have network access
```

- The model config/weights are pulled from the HuggingFace hub on first run and
  cached locally; subsequent runs load from cache.
- On a **single P150**, scripts run on device 0 by default — just run them.
- On a **multi-chip host** (e.g. the Galaxy), select a chip with
  `TT_VISIBLE_DEVICES=<chip>` and keep `--device-id 0` (the chip is remapped to
  logical device 0). The multi-process scripts (`dp32_multiprocess.py`,
  `live_demo.py --dp N`, `eval_accuracy_tt.py --num-devices N`) handle this
  isolation automatically.

---

## 2. Scripts — how to run them and what they produce

| Script | What it does / produces |
|--------|--------------------------|
| `demo/demo_bs{1,8,32}_isl{512,1024,2048}.py` | Single-device **latency benchmark** for one (batch, ISL). Prints avg/best prefill time, embeddings/s and tokens/s. Add `--full-pipeline` for end-to-end latency (H2D + replay + post-proc + D2H). |
| `demo/dp32_multiprocess.py` | **Data-parallel benchmark** across N chips (one resident model per chip). Prints per-chip latency (mean/median/min/max), slowest-chip latency and aggregate throughput. `--mean-pool` runs the real serving post-processing (RMSNorm + mean-token pooling folded in-trace). |
| `demo/live_demo.py` | **Resident encoder** — loads the model once and keeps it up. Embed your own text interactively, from a file (one text/line), or from a folder (one doc/file). `--fast` = low-latency traced serving; `--mask` = accurate for short/variable inputs; `--dp N` = serve across N chips; `--bench N` = report per-request latency. |
| `demo/eval_accuracy.py` | **CPU fp32 reference** accuracy (STS-B Spearman). The ground-truth baseline the device is compared against. |
| `demo/eval_accuracy_tt.py` | **On-device accuracy** (STS-B Spearman) with all perf flags on. `--pool {fast,masked,masked-attn}`, `--num-devices N` (DP), `--no-bucket` to disable sequence-length bucketing. |
| `tests/perf/new_perf_bs{1,8,16,32}_isl512.py` | Tracy profiling signpost tests for op-level device timing (developer profiling). |

### 2.1 Latency benchmarks (single device)

```bash
# bs=1, ISL=512 (pure device trace replay)
python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py

# end-to-end latency (H2D + replay + post-processing + D2H)
python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py --full-pipeline

# other shapes
python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs8_isl512.py
python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs32_isl2048.py
# ...one file per (batch ∈ {1,8,32}) × (ISL ∈ {512,1024,2048})

# pick a chip on a multi-chip host
TT_VISIBLE_DEVICES=5 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py --device-id 0
```

### 2.2 Data parallelism across the Galaxy

```bash
# 32 chips, bs=1 ISL=512, real serving post-processing
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 20 --mean-pool
```

### 2.3 Running the model on your own inputs (resident serving)

```bash
# Interactive: model stays loaded; type text, press Enter to embed it.
# --fast replays a captured trace per request and prints live latency.
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast

# Accurate for short / variable-length inputs (adds SDPA padding mask):
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast --mask

# A file of texts (one per line) -> .npy of embeddings
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --fast --mask --input my_texts.txt --output embeddings.npy

# A folder (each file = one document) -> JSONL {name, embedding}
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --fast --mask --input ./docs/ --output embeddings.jsonl

# Serve across all 32 chips at once
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --dp 32 --mask

# Measure per-request latency (per sequence-length bucket)
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast --bench 30 --max-length 512
```

Useful `live_demo.py` flags: `--max-length` (max tokens/text, default 512),
`--no-normalize` (skip L2 norm), `--no-bucket` (single fixed-ISL trace instead of
length buckets), `--metrics` (see below).

**Per-request metrics (`--metrics`).** By default the interactive prompt prints
only the embedding (dimension, norm, token/chip count, first values, and cosine
similarity to the previous input). Add `--metrics` to also print the per-request
performance breakdown — device (prefill + norm + pool), D2H, host, and H2D times,
total replay latency, and, with `--dp`, the aggregate throughput per round:

```bash
# Interactive serving with the timing breakdown shown per prompt
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast --metrics
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --dp 32 --mask --metrics
```

```
text> Hello how are you
  dim=1024  |emb|=1.0000  chips=32/32
  first 8 dims: [-0.0498, +0.0207, ...]
  per-chip step medians (across chips):
    device (prefill+norm+pool): 4.9ms  [min 4.9, max 4.9]
    D2H  (pooled vector copy):  0.06ms
    host (to_torch + L2-norm):  0.12ms
    H2D  (tokens + mask/pool):  0.05ms
    total replay:               5.1ms
  DP round: 32 embeddings in 16.9ms  → 1895 emb/s aggregate
  cross-chip max |Δ| (parity): 0.00e+00
```

### 2.4 Accuracy

```bash
# CPU fp32 reference (STS-B Spearman)
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --dataset stsb

# On-device, single chip (accurate masked path)
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy_tt.py --pool masked-attn

# On-device across 32 chips
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy_tt.py --pool masked-attn --num-devices 32
```

---

## 3. Performance

All numbers measured on Blackhole P150. Optimizations (Section 5) are on by
default. "Best" = fastest measured iteration; "Avg" = mean over the run.

### 3.1 Single-device direct trace replay

Pure device execution (`ttnn.execute_trace` + sync) — the model compute itself,
excluding host overhead, post-processing and D2H copy.

| Batch | ISL  | Avg       | Best      | Embeddings/s | Tokens/s | Activations |
|------:|-----:|----------:|----------:|-------------:|---------:|-------------|
|     1 |  512 |     7.3ms |     7.2ms |        138.1 |   70,718 | L1 + big grid |
|     8 |  512 |    44.0ms |    43.9ms |        182.2 |   93,275 | L1 batched  |
|    32 |  512 |   197.9ms |   197.7ms |        161.8 |   82,856 | DRAM        |
|     1 | 1024 |    18.3ms |    18.2ms |         55.1 |   56,397 | L1          |
|     1 | 2048 |    33.3ms |    33.2ms |         30.1 |   61,661 | L1          |
|     8 | 1024 |   134.7ms |   134.6ms |         59.4 |   60,860 | DRAM        |
|     8 | 2048 |   270.3ms |   270.2ms |         29.6 |   60,633 | DRAM        |
|    32 | 1024 |   524.9ms |   524.7ms |         61.0 |   62,449 | DRAM        |
|    32 | 2048 |  1027.4ms |  1027.1ms |         31.2 |   63,805 | DRAM        |

### 3.2 Full pipeline (end-to-end)

End-to-end latency including H2D, trace replay, post-processing, and D2H +
torch conversion. The `--full-pipeline` flag folds the post-processing device ops
into the trace, so dispatch overhead over the pure replay is negligible.

| Batch | ISL | Direct (best) | Full pipeline (best) | Overhead |
|------:|----:|--------------:|---------------------:|---------:|
|     1 | 512 |         7.2ms |                7.4ms |  ~0.2ms  |

### 3.3 Resident serving latency (`live_demo.py --fast`)

Per-request trace-replay latency (device + D2H + pool), measured with
`--bench 30`. With sequence-length bucketing on (default), each request runs on
the smallest padded-length trace that fits, so short inputs are much faster.

| Bucket ISL | No mask (avg / best) | With mask (avg / best) |
|-----------:|---------------------:|-----------------------:|
|        128 |       5.5ms / 5.5ms  |        5.6ms / 5.5ms   |
|        256 |       6.0ms / 5.9ms  |        6.4ms / 6.0ms   |
|        512 |       7.4ms / 7.3ms  |        8.1ms / 7.8ms   |

The mask adds ~0.5–0.7ms at ISL=512 (28 layers read an S×S additive mask) and is
needed only for accurate short/variable-length inputs (see Section 4).

### 3.4 Data parallelism (32× P150 Galaxy)

DP is embarrassingly parallel: each chip runs an independent resident model, so
**per-chip latency equals the single-device latency** and aggregate throughput
scales with chip count. Measured at bs=1 ISL=512 with the live mean-pooled
serving post-processing (`dp32_multiprocess.py --mean-pool`), all 32 chips active:

| Metric                 | Per-chip latency |
|------------------------|-----------------:|
| Best (min)             | 7.3ms |
| Median                 | 7.4ms |
| Slowest chip (median)  | 7.5ms |

Aggregate throughput: ~4,240 embeddings/s (median), up to ~4,380 best, across all
32 chips — consistent with the single-device 7.2ms direct / 7.4ms full-pipeline
latency above. Accuracy is identical to single device (DP=32 `masked-attn` +
bucketing STS-B Spearman = **0.8488**), confirming every chip computes correctly.

---

## 4. Accuracy (STS-B)

Spearman correlation between model cosine similarities and human scores on the
STS-B test set (1,379 pairs), TT device vs. the CPU fp32 reference. The TT device
runs the performance config (BFP4 weights, BFP8 activations, LoFi math).

| Path | Bucketing | STS-B Spearman | Per-text latency (median / best) |
|------|-----------|---------------:|---------------------------------:|
| CPU fp32 reference | — | **0.8518** | — |
| `fast` (mean over padded ISL, no mask) | off | 0.1286 | 7.9ms / 7.6ms |
| `masked` (real-token pool, no attention mask) | off | 0.5340 | 8.2ms / 7.5ms |
| `masked-attn` (SDPA padding mask + real-token pool) | off | 0.8487 | 8.4ms / 7.9ms |
| `fast` | on | 0.5232 | 6.2ms / 5.5ms |
| **`masked-attn`** (recommended) | **on** | **0.8488** | **6.4ms / 5.6ms** |

> Per-text latency here is end-to-end per text (tokenize + prep + replay + pool),
> so it is ~0.5ms above the pure trace-replay numbers in Section 3.3.

**What this means.** The bidirectional attention has no padding mask by default, so
when a short input is padded to a large ISL, every real token attends to the
padding and the embedding degrades — this is why `fast` (maskless) scores low on
the mostly-short STS-B texts. Two ways to get near-reference accuracy:

1. **Full-length inputs (~512 real tokens):** there is no padding to contaminate,
   so the maskless `fast` path is both near-reference *and* fastest (7.2ms).
2. **Short / variable-length inputs:** enable the SDPA padding mask + real-token
   pooling (`--mask` / `--pool masked-attn`). This recovers **0.8488** vs. the
   0.8518 reference for any input length.

**Sequence-length bucketing (on by default).** Instead of padding every input to
the max ISL, the serving path captures one trace per length tier (128 / 256 /
512 …) and routes each request to the smallest one that fits. This keeps full
reference accuracy while cutting ~2ms for short texts (the **`masked-attn` +
bucketing** row is the recommended general-purpose operating point). Accuracy is
identical single-device and across all 32 chips (DP=32 `masked-attn` = 0.8488).

---

## 5. Optimizations

Applied by default across all workloads (centralized in
`demo/_common.py::WORKLOAD_CONFIGS`, shared by every demo and the DP scripts):

- **BFP4 weights, BFP8 activations** — quantized formats for compute/bandwidth.
- **LoFi math fidelity** — for matmuls, SDPA, and RoPE. RoPE is a cos/sin rotation
  (operands in [-1,1]), so its stock HiFi4 setting is wasted precision; LoFi is
  accuracy-neutral and saves ~0.4ms at bs=1/ISL=512:

  | RoPE math fidelity | STS-B Spearman | Best replay (no mask) |
  |--------------------|---------------:|----------------------:|
  | HiFi4 (op default) |         0.8481 |                 7.7ms |
  | **LoFi (default here)** | **0.8487** |             **7.3ms** |

- **Head-split QKV / concat-heads** — native head-split program variants of
  `nlp_create_qkv_heads` / `nlp_concat_heads` (≈ −1.3ms in the masked path).
- **Workload-tuned memory placement** — L1 activations + large core grid for
  bs=1; batched L1 for bs=8/ISL=512; DRAM for the large shapes.
- **Block-sharded LayerNorm**, **KV-cache fill skip** (prefill-only, no decode),
  and **bidirectional SDPA** (`is_causal=False`).
- **Hardware trace capture** with an *extended trace* that folds the RMSNorm +
  mean-token pooling post-processing into the replay (only the pooled 1024-d
  vector is copied back).
- **Sequence-length bucketing** in the resident serving path (Section 4).

Two optimizations were evaluated and **not enabled**: skipping the Q/K RMSNorm
(saves ~0.5ms but collapses STS-B 0.848 → 0.236 — the trained norm is
load-bearing) and a 2-command-queue input/compute overlap (no throughput benefit
here — the per-request token input is tiny and the model is compute-bound). Both
are documented in [`README2.md`](README2.md).

---

## Implementation notes

- All changes are localized to this directory (plus a one-line `trust_remote_code`
  fix in `tt_transformers/.../model_config.py`); no other `tt_transformers` edits.
- `PplxBidirectionalAttention` (`tt/attention.py`) wraps SDPA with `is_causal=False`
  and applies the LoFi RoPE kernel config.
- `PplxModelArgs` loads weights directly from safetensors, avoiding the custom HF
  `modeling.py` that requires a newer `transformers`.
- `dp32_multiprocess.py` spawns one process per chip (`TT_VISIBLE_DEVICES`
  isolation + CPU-affinity pinning + a barrier for synchronized measurement).
