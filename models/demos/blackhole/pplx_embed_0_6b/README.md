# pplx-embed-v1-0.6B on Tenstorrent Blackhole

Text-embedding inference for [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)
on Tenstorrent Blackhole (P150 single device and 32× P150 Blackhole Galaxy via
data parallelism).

This README covers **how to run every script, what each one produces, the
measured performance/accuracy, and the optimizations applied.**

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

## Embedding API

To call the model from your own code, build it once with
`build_single_device_model()` and wrap it in a resident encoder. The encoder
captures the bidirectional prefill as a hardware trace and replays it per
request, folding the final RMSNorm + mean-token pooling onto the device, so each
`encode()` returns a post-processed `[1024]` embedding at the benchmarked latency.

```python
import ttnn
from models.demos.blackhole.pplx_embed_0_6b.demo._common import (
    apply_workload_env,
    build_single_device_model,
)
from models.demos.blackhole.pplx_embed_0_6b.demo.live_demo import (
    BucketedEncoder,
    TracedEncoder,
    bucket_lengths,
    encode_one,
    _extract_final_norm,
)

apply_workload_env(1, 512)  # enable the perf flags for bs=1, ISL=512

device = ttnn.open_device(
    device_id=0,
    l1_small_size=32768,
    trace_region_size=200_000_000,
    num_command_queues=1,
)

# Build the resident model once (weights + KV cache + page table).
generator, model_args, kv_caches, page_table = build_single_device_model(
    device, batch_size=1, seq_len=512,
)
model = generator.model[0]
norm_weight, eps = _extract_final_norm(model)

# Low-latency encoder. Two encoder classes are available:
#   TracedEncoder   - one fixed-ISL trace (every input padded to seq_len).
#   BucketedEncoder - one trace per length tier (128/256/512...); each request
#                     routes to the smallest bucket that fits (faster for short text).
encoder = BucketedEncoder(
    generator, model, kv_caches[0], page_table, model_args.tokenizer,
    norm_weight, eps, bucket_lengths(512), device,
    pool="masked", use_mask=True,   # accurate "masked-attn" path (any length)
)
```

### Pooling / accuracy options

The `pool` and `use_mask` arguments select the embedding path (same trade-offs as
the `eval_accuracy_tt.py --pool` modes):

| `pool`     | `use_mask` | Behavior | Use when |
|------------|------------|----------|----------|
| `"fast"`   | `False`    | Device mean over the full padded ISL. Lowest latency. | Full-length (~512-token) inputs. |
| `"masked"` | `True`     | Real-token mean pooling + SDPA padding mask (the `masked-attn` path). Near-reference accuracy for any length. | Short / variable-length inputs (recommended default). |

`encode_one(generator, model, kv_caches[0], page_table, tokenizer, norm_weight, eps, text, max_length)`
runs the same forward **eagerly** (no trace, minimal padding to the nearest 128) —
handy for one-off calls or debugging.

## Run inference (Example)

`encoder.encode(text, normalize=True)` returns an L2-normalized `[1024]` torch
tensor. Stack a batch of texts and take a dot product for cosine-similarity
(dense retrieval) scoring:

```python
import torch

sentences_1 = ["What is pplx-embed?", "Definition of BM25"]
sentences_2 = [
    "pplx-embed-v1-0.6B is a bidirectional text-embedding model from Perplexity AI.",
    "BM25 is a bag-of-words retrieval function that ranks documents by query-term matches.",
]

def encode(sentences):
    # Each call returns an L2-normalized [1024] tensor (normalize=False to skip).
    return torch.stack([encoder.encode(s, normalize=True) for s in sentences])

embeddings_1 = encode(sentences_1)
embeddings_2 = encode(sentences_2)

# Vectors are already L2-normalized, so the dot product is the cosine similarity.
similarity = embeddings_1 @ embeddings_2.T
print(similarity)            # [2, 2]: diagonal pairs score highest

ttnn.close_device(device)
```

To serve embeddings across all 32 chips of a Galaxy from your own code, use the
`_DPServer` class in [live_demo.py](live_demo.py) (one resident encoder per chip;
`broadcast(text)` runs on every chip, `map(texts)` distributes a batch
round-robin), or run `live_demo.py --dp 32` directly.

---

## Serving (HTTP, DP=32)

[`demo/serve.py`](demo/serve.py) puts a lightweight async **FastAPI + uvicorn**
layer on top of the 32 resident chip workers and exposes an **OpenAI-compatible**
`/v1/embeddings` API. It is built for high request throughput at low per-request
latency.

**Install** (one-time):

```bash
pip install fastapi "uvicorn[standard]" orjson
```

**Run** (loads one encoder per chip; ~1-2 min build, then resident):

```bash
python models/demos/blackhole/pplx_embed_0_6b/demo/serve.py --dp 32
# CLI: --dp N (chips, default 32), --host (default 0.0.0.0), --port (default 8000),
#      --server-phys S    (physical cores reserved for the server; default 4 = recommended)
#      --cores-per-worker K (pin K physical cores per worker; default 0 = OS-float; NOT recommended)
#      --fastokens / --server-tokenize  (experimental; NOT recommended — measured to
#                                         REGRESS DP=32 throughput; see "Tokenization" below)
```

**Fixed worker configuration** (tuned for lowest per-request latency, not
per-request toggles): ISL 512, `BucketedEncoder` ON (short text routes to the
faster 128/256 trace), masked-attn ON (real-token mean-pool + SDPA padding mask,
near-reference accuracy at any length), batch 1 per chip, 1 command queue.

**Core allocation — isolate the server, OS-float the workers.** The single
biggest scaling lever is *which process gets dedicated CPU*. Each worker is
device-bound (its ~7.6 ms is on-chip) and only needs short, un-starved host bursts
for tokenize + dispatch, so the 32 workers are left **OS-scheduled** — pinning a
worker to dedicated cores starves its tt-metal dispatch/completion threads and
causes catastrophic tails (e.g. e2e p99 jumps to ~128 ms). The HTTP **server** is
the opposite: a *single* process (event loop + result collector + HTTP) serving
all 32 chips, i.e. the one shared host serialization point. By default
(`--server-phys 4`) it gets 4 dedicated physical cores while workers float on the
rest. Measured effect at full load (vs. everything OS-floating): **+46 % throughput
(1,777 → 2,589 req/s) and −42 % p99 latency at 64 in-flight**, with on-device
replay still flat at ~7.6 ms. Set `--server-phys 0` to disable, or
`--cores-per-worker K>0` to dedicate cores per worker (measured to be *worse* —
provided only for experimentation). Concurrency itself comes from the 32 chips
running in parallel — the scheduler hands each input to the next idle chip (up to
32 in flight), and a batched request (`input: [...]`) fans out across free chips.

### API

`POST /v1/embeddings` (OpenAI schema). `input` is a string or list of strings;
`encoding_format` is `"float"` (JSON floats) or `"base64"` (float32
little-endian, ~6× smaller — recommended for high-RPS clients).

```bash
# Single input, float vector
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "pplx-embed-v1-0.6b", "input": "hello world"}'

# Batch input, compact base64 payload (fans out across free chips)
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "pplx-embed-v1-0.6b",
       "input": ["what is a tensor?", "define BM25"],
       "encoding_format": "base64"}'

# Liveness / chip count, and model list
curl http://localhost:8000/health      # {"status":"ok","chips_ready":32,"chips_total":32}
curl http://localhost:8000/v1/models

# Per-request latency breakdown (on-device replay vs. server overhead), ms.
# Add ?reset=true to clear the rolling window.
curl http://localhost:8000/metrics
```

Response (OpenAI schema):

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [/* 1024 floats */]}],
  "model": "pplx-embed-v1-0.6b",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

Because the API is OpenAI-compatible, the `openai` Python SDK works pointed at
the local base URL. A dependency-light async client + load test is provided in
[`demo/embed_client.py`](demo/embed_client.py):

```bash
# Embed inline texts
python models/demos/blackhole/pplx_embed_0_6b/demo/embed_client.py "hello world" "what is a tensor?"

# Concurrent load test: 1500 requests, 64 in flight, full ISL-512 inputs, base64.
# After the run it also prints the server-side device-vs-overhead breakdown
# (pulled from GET /metrics).
python models/demos/blackhole/pplx_embed_0_6b/demo/embed_client.py \
    --load 1500 --concurrency 64 --encoding base64 --long
```

### Load benchmark (DP=32, ISL 512, base64)

Measured on the Blackhole Galaxy with all 32 chips resident, 1200 requests per
concurrency level, each input padded to the full 512-token bucket (worst case).
"In-flight" is the number of concurrent requests; "on-device replay" is the
per-chip prefill+RMSNorm+pool latency reported by the worker (`GET /metrics`).

Default core allocation (`--server-phys 4`, workers OS-float):

| In-flight | Throughput (req/s) | Throughput (tok/s) | On-device replay p50 / p99 (ms) | Client e2e p50 / p99 (ms) |
| --------: | -----------------: | -----------------: | :-----------------------------: | :-----------------------: |
|         1 |                 81 |              41.4k |          7.81 / 7.92            |        12.5 / 13.2        |
|         8 |                636 |               326k |          7.66 / 7.70            |        12.2 / 18.8        |
|        32 |              2,383 |              1.22M |          7.64 / 7.68            |        11.9 / 41.0        |
|        64 |              2,589 |              1.33M |          7.63 / 7.67            |        22.2 / 57.0        |

**Key result — per-chip latency is flat under full load.** On-device replay stays
at ~7.6–7.8 ms (p99 < 8 ms) from 1 to 64 concurrent requests: a request served
while all 32 chips are busy costs the same on-device as one served in isolation.

**Why the server gets dedicated cores (CPU-allocation sweep).** Client e2e p50/p99
and throughput at 64 in-flight, same workload, three allocations:

| Allocation | e2e p50 / p99 (ms) | Throughput (req/s) |
| --- | :---: | ---: |
| Everything OS-floats | 31.0 / 98.7 | 1,777 |
| 1 dedicated core per worker | 32.8 / 76.7 (p99 **128** at c=1) | 1,805 |
| **Server isolated (4 phys), workers float — default** | **22.2 / 57.0** | **2,589** |

Workers are device-bound, so dedicating cores to them only starves their tt-metal
dispatch threads (huge tails); the single server process is the shared host
bottleneck, so isolating *it* is the win (+46 % throughput, −42 % p99).

Sweeping `--server-phys` shows a clear optimum at **4** (c=64 throughput): 2 phys
→ 2,372 req/s (the single event loop occasionally starves under load), **4 phys →
2,589 req/s**, 8 phys → 2,255 req/s (steals cores from the 32 workers). The server
is one event loop + one collector thread, so ~4 physical cores saturate it; more
just costs worker throughput.

**Per-request overhead (concurrency 1, p50 ms).** `GET /metrics` decomposes each
request into its hops. The end-to-end latency is the on-device replay plus a small
fixed host/IPC/HTTP cost:

| Hop | p50 (ms) | Notes |
| --- | -------: | ----- |
| tokenize | 1.3 | host CPU, HF Rust `backend_tokenizer`, in the worker (overlaps across all 32 chips) |
| build host tensors | 0.14 | token tensor only; page table is cached, RoPE is baked into the trace |
| task_q (dispatch→worker) | 0.15 | mp.Queue transit |
| **on-device replay** | **7.5** | prefill + RMSNorm + mean-pool (device-bound floor) |
| (H2D/D2H/host finalize) | 0.20 | inside worker_total (~7.6) |
| result (worker→future) | 0.36 | mp.Queue transit + event-loop wakeup |
| HTTP (parse + base64 resp) | ~0.8 | uvloop + httptools, ~5.5 KB base64 |
| **client e2e** | **~10.4** | at the c=32 operating point |

The host overhead beyond the device replay is the tokenizer plus a fixed
IPC/HTTP cost. An earlier version spent an extra ~5 ms in `prepare` because it
re-tokenized twice and rebuilt the (request-independent) RoPE matrices and page
table every call; those are now cached/skipped, cutting concurrency-1 e2e from
~16 ms to ~10–12 ms. Tokenization runs **in the workers** by default, where it is
distributed across the 32 processes and overlaps device work — so its ~1.3 ms
does not limit throughput (the system is device-bound at ~2,760 req/s).

#### Tokenization: why fastokens / server-side did *not* help here (`--fastokens`, experimental)

The [fastokens](https://github.com/Atero-ai/fastokens) Rust engine
(Crusoe/NVIDIA-Dynamo) is genuinely faster in isolation — ~0.09 ms vs HF's
~0.75 ms for a 512-token input, byte-identical ids on this `Qwen2TokenizerFast`.
But on this DP=32 server it does **not** improve the operating point, and both
placements measured *worse* than the default worker-side HF path:

| Config (c=32, ISL 512) | Throughput | e2e p50 |
| --- | ---: | ---: |
| **worker-side HF (default)** | **~2,760 req/s** | **~10.4 ms** |
| server-side + fastokens | ~2,240 req/s | ~13.5 ms |
| worker-side + fastokens (×32, even `RAYON_NUM_THREADS=1`) | ~510 req/s | ~60 ms |

Why:

* **The single server process is the throughput bottleneck**, not the tokenizer.
  Moving tokenization onto its 4 dedicated cores adds work to the bottleneck and
  slows the event loop's dispatch/collect hops (≈ −19 % throughput).
* **fastokens replicated across 32 workers is pathological** — each instance
  spins up its own thread pool, oversubscribing the host (tokenize p50 jumps to
  ~2.6 ms with a ~26 ms tail), regardless of `RAYON_NUM_THREADS`.
* **At the throughput operating point the tokenizer is off the critical path** —
  the system is device-bound (~7.5 ms replay × 32 chips), so worker-side HF's
  ~1.3 ms overlaps and is effectively free. fastokens only helps single-stream
  (c=1) latency by ~1 ms, which is not the regime that matters here.

**Conclusion: leave tokenization on the worker with the HF tokenizer (the
default).** `--fastokens` / `--server-tokenize` remain as opt-in experimental
flags but are **not recommended** for DP=32 serving.

**Operating point.** Throughput peaks around **~2,600 req/s (~1.3M tokens/s)** at
64 in-flight requests (two per chip, pipelined). Beyond that the chips are
saturated, so extra concurrency only adds queuing delay (client e2e tail grows
while on-device stays flat). The host overhead overlaps across the 32 workers, so
it costs throughput but not device time. Keep 32–64 requests in flight and prefer
`encoding_format: "base64"` to minimize host serialization.

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
|    16 |  512 |    95.0ms |    94.8ms |        168.7 |   86,393 | DRAM resid + L1 intermediates |
|    32 |  512 |   192.4ms |   192.3ms |        166.4 |   85,216 | DRAM resid + L1 intermediates |
|     1 | 1024 |    18.3ms |    18.2ms |         55.1 |   56,397 | L1          |
|     1 | 2048 |    33.3ms |    33.2ms |         30.1 |   61,661 | L1          |
|     8 | 1024 |   134.7ms |   134.6ms |         59.4 |   60,860 | DRAM        |
|     8 | 2048 |   270.3ms |   270.2ms |         29.6 |   60,633 | DRAM        |
|    32 | 1024 |   524.9ms |   524.7ms |         61.0 |   62,449 | DRAM        |
|    32 | 2048 |  1027.4ms |  1027.1ms |         31.2 |   63,805 | DRAM        |

> **bs=8/ISL=512 is the single-chip throughput peak** (93.3k tok/s): it is the
> largest batch whose full activation stream fits in P150 L1. For bs≥16 the
> residual (`bs·ISL·dim·2 B`) spills to DRAM, but the short-lived per-layer
> matmul outputs are still pinned to L1 (`DRAM resid + L1 intermediates`),
> recovering **+6.8 %** at bs=16 (80.9k→86.4k) and **+2.2 %** at bs=32
> (83.4k→85.2k) vs. the all-DRAM placement. See Section 5.

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
- **Per-op L1 intermediates for batched prefill (bs≥16, ISL≤512)** — at bs≥16 the
  persistent residual stream no longer fits L1 and falls back to DRAM, but the
  short-lived matmul outputs inside each layer (produced, consumed, and
  immediately deallocated) are still pinned to L1 so they skip the DRAM
  round-trip. Which outputs are pinned is shape-specific: the big FF-gate/up and
  QKV tensors fit at bs=16 but clash with the matmul static circular buffers at
  bs=32, so only SDPA/FF-down/concat-heads are pinned there. Net: **+6.8 %** at
  bs=16, **+2.2 %** at bs=32 (ISL=512); embeddings are bit-identical (memory
  location only, no math change). Gated by `TT_PREFILL_INTERMEDIATE_L1` /
  per-op `TT_PREFILL_<OP>_L1` env vars and wired per shape in `demo/_common.py`.
- **Block-sharded LayerNorm**, **KV-cache fill skip** (prefill-only, no decode),
  and **bidirectional SDPA** (`is_causal=False`).
- **Hardware trace capture** with an *extended trace* that folds the RMSNorm +
  mean-token pooling post-processing into the replay (only the pooled 1024-d
  vector is copied back).
- **Sequence-length bucketing** in the resident serving path (Section 4).

Two optimizations were evaluated and **not enabled**: skipping the Q/K RMSNorm
(saves ~0.5ms but collapses STS-B 0.848 → 0.236 — the trained norm is
load-bearing) and a 2-command-queue input/compute overlap (no throughput benefit
here — the per-request token input is tiny and the model is compute-bound).

---

## Implementation notes

- Changes are localized to this directory, plus two additions in
  `tt_transformers/.../model_config.py`: the one-line `trust_remote_code` fix and
  the env-gated per-op prefill L1 placement (`_prefill_op_mem_config` /
  `_use_prefill_intermediate_l1`, default-off so all other models are unaffected).
- `PplxBidirectionalAttention` (`tt/attention.py`) wraps SDPA with `is_causal=False`
  and applies the LoFi RoPE kernel config.
- `PplxModelArgs` loads weights directly from safetensors, avoiding the custom HF
  `modeling.py` that requires a newer `transformers`.
- `dp32_multiprocess.py` spawns one process per chip (`TT_VISIBLE_DEVICES`
  isolation + CPU-affinity pinning + a barrier for synchronized measurement).
