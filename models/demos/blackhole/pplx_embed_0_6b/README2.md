# pplx-embed-v1-0.6B on Blackhole (P150)

Performance demo for [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) running on Tenstorrent Blackhole P150, supporting single-device and DP=32 multi-process Galaxy deployment.

## Model Overview

pplx-embed-v1-0.6B is a text embedding model from Perplexity AI built on the Qwen3-0.6B backbone with:

| Property          | Value                     |
|-------------------|---------------------------|
| Parameters        | 0.6B                      |
| Hidden dim        | 1024                      |
| Layers            | 28                        |
| Q / KV heads      | 16 / 8                    |
| Head dim          | 128                       |
| FFN dim           | 3072                      |
| Max context       | 32K                       |
| Pooling           | Mean                      |
| Attention         | Bidirectional (non-causal) |

### Key differences from Qwen3-Embedding-0.6B

1. **Bidirectional attention** — every token attends to every other token (no causal mask).
2. **Mean-token pooling** — embeddings are produced by averaging hidden states over all real tokens, rather than extracting the last token.
3. Requires `trust_remote_code=True` for HuggingFace model loading.

## Quick Start

```bash
# Single device (P150), bs=1 ISL=512
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py

# Other batch/ISL combinations
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs8_isl512.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs32_isl512.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl1024.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl2048.py

# Full pipeline latency (includes Generator overhead, post-processing, D2H)
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py --full-pipeline

# Via pytest
pytest models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py -sv

# DP=32 multi-process on Galaxy (32x P150)
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 10

# DP=32 with the live mean-pooled embedding post-processing (real serving path)
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 20 --mean-pool
```

## Running the Model on Your Own Inputs (live / on-demand)

The `demo_*` scripts above run a fixed synthetic input to benchmark latency.
To **load the model once and keep it resident** while you feed it your own text
on demand (i.e. call `model.forward` manually before any serving-stack
integration), use `demo/live_demo.py`.

The model is loaded onto a single P150 and stays up between requests. Each
request runs a real forward pass — tokenize → bidirectional prefill → final
RMSNorm → mean-token pooling → L2 normalization — and returns a 1024-dim
embedding.

```bash
export HF_MODEL=perplexity-ai/pplx-embed-v1-0.6b

# 1) Interactive: the model stays loaded; type text and press Enter to embed it.
#    Prints the embedding + cosine similarity to your previous input. Ctrl-D / /quit to exit.
#    Add --fast to replay the prefill trace per prompt and print the live
#    trace-replay latency (device+D2H+pool) next to each embedding.
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast

# 2) A file of texts (one per line) -> save embeddings to .npy
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --input my_texts.txt --output embeddings.npy

# 3) A folder (each file = one document) -> save embeddings to JSONL ({name, embedding})
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --input ./docs/ --output embeddings.jsonl
```

Flags: `--max-length` (max tokens/text, default 512), `--device-id` (default 0),
`--no-normalize` (skip L2 normalization), `--fast` (low-latency traced serving
path, see below), `--mask` (SDPA padding mask + real-token pool for accurate
variable-length inputs), `--bench N` (benchmark per-request latency over N
iterations), `--dp N` (data-parallel serving across N chips, e.g. `--dp 32` on a
galaxy — see below).

#### Low-latency serving (`--fast`) — 7.2ms/request at ISL=512

By default `live_demo.py` runs the variable-length, minimally-padded path (best
accuracy for short inputs, ~9–20ms/request). For production-style serving where
inputs are full-length, pass `--fast`: it captures the prefill **as a hardware
trace once** at a fixed ISL and replays it per request, folding the final
RMSNorm + mean-token pooling onto the device so only the pooled `[1024]` vector
is copied back. This matches the benchmarked device latency:

```bash
# Resident, low-latency serving (fixed ISL = nearest 128-multiple of --max-length)
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast

# Measure per-request latency
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --fast --bench 30 --max-length 512
# -> Trace replay (device+D2H+pool): best 7.1ms, avg ~7.3ms; end-to-end ~7.7ms
```

In `--fast` mode every input is padded/truncated to the fixed ISL and pooled
over the full (padded) length, so it is intended for full-length inputs; use the
default mode (or `--fast --mask`) for accurate short-text embeddings. In the
interactive prompt, `--fast` prints the **live trace-replay latency
(device+D2H+pool)** per prompt next to the accurate post-processed embedding:

```
text> deep learning is a subset of machine learning
  dim=1024  |emb|=1.0000  tokens=8
  first 8 dims: [-0.0123, +0.0457, ...]
  replay steps: device(prefill+norm+pool)=7.6ms  D2H=0.06ms  host(to_torch+norm)=0.12ms  H2D=0.03ms
  latency: replay=7.8ms  | tokenize+prep=0.4ms  | total=8.2ms
  cosine similarity to previous input: 0.3744
```

The per-step split shows the device trace (prefill + RMSNorm + pool, all folded
into the trace) dominates; D2H of the pooled `[1024]` vector and host finalize
(to_torch + L2-norm) are negligible. With `--mask`, the first request at a new
token count pays a one-time `H2D` spike (rebuilding + copying the SxS padding
mask); repeated lengths are cached.

**DP=32 holds the same per-request latency.** Data parallelism is embarrassingly
parallel across chips, so running 32 resident instances does not degrade
per-chip latency. Verified with the live mean-pooled post-processing across all
32 P150s (`dp32_multiprocess.py … --mean-pool`): per-chip median **7.3ms**
(min 7.1ms, slowest-chip median 7.4ms) at bs=1 ISL=512, ~4350 embeddings/s
aggregate.

#### Live serving across the galaxy (`--dp 32`)

`live_demo.py --dp N` runs the resident serving path across `N` chips at once
(use `--dp 32` on a Blackhole Galaxy). Each chip runs in its own CPU-pinned
subprocess with its own resident `TracedEncoder` (same per-chip model as
`dp32_multiprocess.py`), and the parent dispatches real user inputs across all
chips:

```bash
export HF_MODEL=perplexity-ai/pplx-embed-v1-0.6b
export HF_HOME=$HOME/.cache/huggingface

# Interactive: each prompt is embedded on ALL 32 chips in parallel; prints
# per-chip replay latency, aggregate throughput, and cross-chip parity.
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --dp 32 --mask

# Batch a file across 32 chips (round-robin) and save embeddings:
python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \
  --dp 32 --input my_texts.txt --output embeddings.jsonl
```

```
text> deep learning is a subset of machine learning
  dim=1024  |emb|=1.0000  chips=32/32
  per-chip step medians (across chips):
    device (prefill+norm+pool): 7.6ms  [min 7.6, max 7.7]
    D2H  (pooled vector copy):  0.06ms
    host (to_torch + L2-norm):  0.12ms
    H2D  (tokens + mask/pool):  0.04ms
    total replay:               7.8ms
  DP round: 32 embeddings in 9.5ms  → ~3400 emb/s aggregate
  cross-chip max |Δ| (parity): 0.00e+00
```

Notes: `--dp` always uses the fast traced path (add `--mask` for accurate
short/variable-length inputs, drop it for the lowest-latency full-length path).
The parent process does **not** open a device; the model loads once per chip and
stays resident, so first launch spends ~1–2 min building 32 encoders, then serves
on demand. Cross-chip `|Δ|=0` confirms every chip computes the identical
embedding (a faulty chip would show non-zero drift).

> **Note on the HF cache.** The model config/weights are pulled from the
> HuggingFace hub on first run. Make sure `HF_HOME` points to a writable
> directory with network access (e.g. `export HF_HOME=$HOME/.cache/huggingface`).

### Accuracy / parity notes

- `live_demo.py` runs the **same performance-optimized model** as the benchmarks
  (BFP4 weights, BFP8 activations, LOFI math fidelity). Embeddings preserve
  semantic ranking (e.g. two paraphrases score ~0.75 cosine vs ~0.29 for an
  unrelated sentence) — appropriate for retrieval / similarity use. Use
  `eval_accuracy.py` for the fp32 CPU reference and STS-B / SciFact scoring.
- The TT prefill path requires the sequence length to be a multiple of 128. By
  default the bidirectional SDPA applies no padding mask, so short inputs padded
  to a large ISL lose accuracy (see [Accuracy](#accuracy-sts-b-isl512-all-perf-flags-on)).
  Pass `--fast --mask` to inject an additive SDPA padding mask + real-token
  pooling, which restores near-reference accuracy (STS-B 0.848 vs 0.852) for any
  input length at ~7.9ms; without it, use full-length inputs to stay at 7.1ms.

## Directory Structure

```
models/demos/blackhole/pplx_embed_0_6b/
├── __init__.py
├── README.md
├── tt/
│   ├── __init__.py
│   └── attention.py              # PplxBidirectionalAttention (is_causal=False)
├── demo/
│   ├── __init__.py
│   ├── _common.py                # Model builder, WORKLOAD_CONFIGS, run_perf
│   ├── live_demo.py             # Live / on-demand encoder (interactive prompt + file/folder)
│   ├── demo_bs1_isl512.py        # BS=1,  ISL=512  (L1 activations)
│   ├── demo_bs8_isl512.py        # BS=8,  ISL=512  (L1 batched activations)
│   ├── demo_bs32_isl512.py       # BS=32, ISL=512  (DRAM optimized)
│   ├── demo_bs1_isl1024.py       # BS=1,  ISL=1024 (DRAM optimized)
│   ├── demo_bs1_isl2048.py       # BS=1,  ISL=2048 (DRAM optimized)
│   ├── demo_bs8_isl1024.py       # BS=8,  ISL=1024 (DRAM optimized)
│   ├── demo_bs8_isl2048.py       # BS=8,  ISL=2048 (DRAM optimized)
│   ├── demo_bs32_isl1024.py      # BS=32, ISL=1024 (DRAM optimized)
│   ├── demo_bs32_isl2048.py      # BS=32, ISL=2048 (DRAM optimized)
│   ├── dp32_multiprocess.py      # DP=32 multi-process Galaxy benchmark
│   ├── eval_accuracy.py          # CPU fp32 reference accuracy (STS-B / SciFact)
│   └── eval_accuracy_tt.py       # On-device accuracy (STS-B), single + DP=32
└── tests/
    └── perf/
        ├── __init__.py
        ├── new_perf_bs1_isl512.py   # Tracy signpost test (bs=1, ISL=512)
        ├── new_perf_bs8_isl512.py   # Tracy signpost test (bs=8, ISL=512)
        ├── new_perf_bs16_isl512.py  # Tracy signpost test (bs=16, ISL=512)
        └── new_perf_bs32_isl512.py  # Tracy signpost test (bs=32, ISL=512)
```

## Performance

### Single Device (P150) — Direct Trace Replay

Direct trace replay measures pure device execution via `ttnn.execute_trace` + sync, bypassing Generator Python overhead, post-processing, and D2H copy.

| Batch | ISL  | Prefill (avg) | Prefill (best) | Embeddings/s (best) | Tokens/s (best) | Memory   |
|------:|-----:|--------------:|---------------:|--------------------:|----------------:|----------|
|     1 |  512 |         7.2ms |          7.1ms |               140.6 |          71,991 | L1 + big grid |
|     8 |  512 |        43.8ms |         43.7ms |               182.9 |          93,651 | L1       |
|    16 |  512 |        95.0ms |         94.8ms |               168.7 |          86,393 | DRAM resid + L1 intermediates |
|    32 |  512 |       192.4ms |        192.3ms |               166.4 |          85,216 | DRAM resid + L1 intermediates |
|     1 | 1024 |        18.0ms |         17.9ms |                55.9 |          57,207 | L1       |
|     1 | 2048 |        33.4ms |         33.4ms |                30.0 |          61,377 | L1       |
|     8 | 1024 |       134.4ms |        134.4ms |                59.5 |          60,968 | DRAM     |
|     8 | 2048 |       269.6ms |        269.5ms |                29.7 |          60,801 | DRAM     |
|    32 | 1024 |       520.4ms |        520.0ms |                61.5 |          63,019 | DRAM     |
|    32 | 2048 |      1045.4ms |       1045.1ms |                30.6 |          62,706 | DRAM     |

### Workload Optimization Strategy

| Workload    | Strategy     | Key Knobs |
|-------------|-------------|-----------|
| bs=1  ISL=512  | L1 activations (1 MB) + big grid | `QWEN_SDPA_BIG_CHUNK_BS1=1`, `QWEN_MM_BIG_GRID_BH=1` |
| bs=1  ISL=1024 | L1 activations (2 MB) | `QWEN_SDPA_BIG_CHUNK_BS1=1` |
| bs=1  ISL=2048 | L1 activations (4 MB) | `QWEN_SDPA_BIG_CHUNK_BS1=1` |
| bs=8  ISL=512  | L1 batched (8 MB)     | `TT_BATCHED_L1_PREFILL=1` |
| bs=8  ISL=1024 | DRAM (16 MB)          | `QWEN_MM_BIG_GRID_BH=1` (80-core matmul) |
| bs=8  ISL=2048 | DRAM (32 MB)          | `QWEN_MM_BIG_GRID_BH=1` |
| bs=16 ISL=512  | DRAM resid + L1 intermediates | `QWEN_MM_BIG_GRID_BH=1`, `TT_PREFILL_{SDPA,FF2,CONCAT,FF13,QKV}_L1=1` |
| bs=32 ISL=512  | DRAM resid + L1 intermediates | `QWEN_MM_BIG_GRID_BH=1`, `TT_PREFILL_{SDPA,FF2,CONCAT}_L1=1` |
| bs=32 ISL=1024 | DRAM (64 MB)          | `QWEN_MM_BIG_GRID_BH=1` |
| bs=32 ISL=2048 | DRAM (128 MB)         | `QWEN_MM_BIG_GRID_BH=1` |

All workloads additionally use: BFP4 weights, BFP8 activations, LOFI math fidelity, head-split TMs, L1 RoPE, LoFi RoPE math, block-sharded LayerNorm, KV cache fill skip, and SDPA LOFI.

**Per-op L1 intermediates (bs≥16, ISL≤512).** At bs≥16 the residual stream
(`bs·ISL·dim·2 B`) is too large for P150 L1 and falls back to DRAM (the
all-or-nothing `use_short_seq_l1_prefill` gate returns DRAM). But the short-lived
matmul outputs *inside* each layer (QKV / create-heads / SDPA / concat / FF
gate-up / FF down) are produced, consumed, and immediately `ttnn.deallocate`d, so
they can still live in L1 — saving the DRAM round-trip on the hottest tensors.
This mirrors the BGE-M3 bs32 strategy (residual in DRAM, op-outputs in L1). The
big FF-gate/up and QKV outputs fit alongside the matmul static circular buffers
at bs=16 but clash at bs=32 (the `minimal_matmul` CB region leaves no L1
headroom, and a wider core grid does not shrink it), so only the smaller
SDPA/FF-down/concat outputs are pinned at bs=32. Implemented as the env-gated
`_prefill_op_mem_config` / `_use_prefill_intermediate_l1` helpers in
`tt_transformers/.../model_config.py` (default off; per-op `TT_PREFILL_<OP>_L1`
knobs allow A/B isolation) and wired per shape via `WORKLOAD_CONFIGS` in
`demo/_common.py`. Measured: **+6.8 %** at bs=16, **+2.2 %** at bs=32 (ISL=512),
with bit-identical embeddings (cosine 1.0, max abs diff 0).

### Full Pipeline (Extended Trace)

Full pipeline measures end-to-end latency including H2D, trace replay, post-processing (slice + norm + to_layout), and D2H + torch conversion. The `--full-pipeline` flag captures an *extended trace* that folds post-processing device ops into the trace replay, reducing Python dispatch overhead from ~0.7ms to ~0.1ms:

| Batch | ISL  | Direct (best) | Full Pipeline (best) | Overhead |
|------:|-----:|--------------:|---------------------:|---------:|
|     1 |  512 |         7.1ms |                7.2ms |    0.1ms |

### DP=32 Multi-Process (Galaxy, 32x P150)

Each chip runs as an independent subprocess with `TT_VISIBLE_DEVICES` isolation, dedicated CPU cores, and barrier-synchronized measurement. Supports all `(batch_size, seq_len)` combinations:

```bash
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32

# Real serving post-processing (full-seq RMSNorm + mean-token pooling in-trace)
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 20 --mean-pool

python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 32 --seq-len 2048 --num-devices 32
```

DP is embarrassingly parallel across chips, so per-chip latency is unchanged vs single device. Measured at bs=1 ISL=512 with the live mean-pooled post-processing (`--mean-pool`), all 32 chips active:

| Metric | Per-chip latency |
|--------|-----------------:|
| Best (min) | 7.1ms |
| Median | 7.3ms |
| Mean | 7.3ms |
| Slowest chip (median) | 7.4ms |

Aggregate throughput: ~4350 embeddings/s (best ~4490). This matches the single-device 7.1–7.2ms direct trace — confirming the DP=32 live serving path holds the target latency.

### Accuracy (STS-B, ISL=512, all perf flags ON)

Measured with `demo/eval_accuracy_tt.py` (TT device, BFP4 weights / BFP8 activations / LOFI math / bidirectional SDPA) vs the CPU fp32 reference (`eval_accuracy.py`). Metric: Spearman correlation on the STS-B test set (1379 pairs).

| Config | STS-B Spearman | Per-chip latency (device replay) | Notes |
|--------|---------------:|---------------------------------:|-------|
| CPU fp32 reference | 0.8518 | — | maskless, minimal padding |
| TT `fast` (mean over ISL, no mask) | 0.10–0.13 | **7.1ms** | accurate only for full-length (~512-tok) inputs |
| TT `masked` (real-token pool, no mask) | 0.58 | ~7.5ms | fixes pooling, not attention |
| TT `masked-attn` (SDPA padding mask + real-token pool) | **0.8481** | 7.9ms | near-reference for any input length |

Verified identical accuracy single-device and across all 32 chips (DP=32): `masked-attn` Spearman = **0.8481** in both cases (a faulty chip would corrupt its shard and drop the score).

```bash
# CPU fp32 reference
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --dataset stsb

# TT device, single chip and all 32 chips
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy_tt.py --pool masked-attn
python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy_tt.py --pool masked-attn --num-devices 32
```

**Key finding — the accuracy/latency trade-off at ISL=512 is driven entirely by padding.** The bidirectional SDPA has no padding mask, so when a short input is padded to 512 every real token attends to ~500 padding tokens and the embedding collapses (the CPU reference is also maskless but pads only to ~30 tokens, hence 0.85). Two ways to get near-reference accuracy:

1. **Full-length inputs (~512 real tokens):** the `fast` path is both near-reference *and* 7.1–7.2ms — there is no padding to contaminate. This is the intended ISL=512 operating point.
2. **Arbitrary-length inputs:** enable the additive SDPA padding mask (`live_demo.py --fast --mask`, or `eval_accuracy_tt.py --pool masked-attn`). This recovers 0.848 (vs 0.852 reference) for any length, costing ~0.8ms of device time (28 layers reading an SxS mask) → **7.9ms**.

The mask + pooling are folded into the trace; the pooling selector and mask are cached by token count so repeated-length requests skip the host rebuild.

**Sequence-length bucketing (default `--fast` path) — full accuracy at ~2ms lower latency.** Rather than padding every input to ISL=512, the live path captures one trace per padded-length tier (128/256/512…) and routes each request to the smallest bucket that fits — the standard prefill pattern from `tt_transformers` (`Generator._easy_trace_prefill`, keyed by `get_padded_prefill_len`). A short input runs a smaller/faster trace *and* a smaller mask. On STS-B (mostly short texts) at ISL=512:

| Pooling | Bucketing | STS-B Spearman | Per-text median | Best |
|---------|-----------|---------------:|----------------:|-----:|
| `fast` (no mask) | off | 0.139 | 8.1ms | 7.5ms |
| `fast` (no mask) | on | 0.527 | 6.2ms | 5.5ms |
| `masked-attn` | off | 0.8481 | 8.6ms | — |
| **`masked-attn`** | **on** | **0.8488** | **6.5ms** | **5.6ms** |

Bucketed `masked-attn` keeps full reference accuracy while cutting ~2ms — short texts route to the 128 bucket (≈5.4ms replay vs ≈7.6ms@512). The maskless `fast` path improves with bucketing (0.14 → 0.53) but intra-bucket padding still contaminates the bidirectional attention, so the mask remains necessary; bucketing just makes it cheap. Bucketing is on by default for `--fast` (and `eval_accuracy_tt.py`); pass `--no-bucket` to force a single fixed-ISL trace. Per-bucket device replay (ISL→latency): **128→5.4ms, 256→6.2ms, 512→7.5ms**.

**Optimizing the masked path (toward 7.1–7.2ms).** The `masked-attn` overhead (~0.8ms over the 7.1ms `fast` path) splits roughly evenly between (a) the masked bidirectional softmax across 28 layers and (b) the real-token pooling reduction. Things tried, informed by [`gtobarTT/bge_m3_2cq`](https://github.com/tenstorrent/tt-metal/compare/main...gtobarTT/bge_m3_2cq):

- **SDPA program-config retune** (bge_m3 B1/S512 uses `q_chunk=32, k_chunk=512, exp_approx_mode=False`, 8×8 grid): *slower* here (8.0–9.3ms). The existing pplx config (`q=k=128, exp_approx_mode=True`, already 8×8 at bs=1) is optimal for this head/dtype shape.
- **Pooling op:** `matmul` selector (7.5ms no-mask) beats `to_memory_config+matmul` and `mul+mean` (7.6ms); kept `matmul`.
- **Mask + full-ISL `mean` pooling** (skip the real-token matmul): no latency win (8.0ms) and worse accuracy (0.814), because the mask-attended padded rows still corrupt a full mean.

- **RoPE math fidelity (applied, accuracy-neutral −0.4ms).** `ttnn.experimental.rotary_embedding_llama` defaults to `MathFidelity::HiFi4` (4 math passes). RoPE is a cos/sin rotation (operands in `[-1, 1]`), so HiFi4 is wasted precision. Dropping to LoFi via `QWEN_ROPE_FIDELITY=lofi` (default in `_common.py`; see `tt/attention.py:_mllama_rope_prefill`) is accuracy-neutral and shaves device time:

  | RoPE fidelity | STS-B Spearman | Best replay (no mask) |
  |---|---|---|
  | HiFi4 (stock op default) | 0.8481 | 7.7ms |
  | HiFi2 | 0.8476 | 7.5ms |
  | **LoFi (default here)** | **0.8487** | **7.3ms** |

  Q/K-norm skip was also evaluated and **rejected**: it saves ~0.5ms but collapses STS-B from 0.848 → 0.236 — the trained per-head Q/K RMSNorm is load-bearing (gated off behind `QWEN_SKIP_QK_NORM` for reference only).

The masked softmax cost is irreducible with the current op set, so **~7.5ms is the optimized floor for the always-masked path** (7.9ms − ~0.4ms from LoFi RoPE).

**bge_m3's "fused QKV/concat-heads" custom ops are already native in our build and enabled here.** Their `tt/custom_ops/fused_qkv_heads` and `fused_concat_heads` `generic_op` kernels are explicitly *"adapted from the Qwen3-Embedding head-split concat reader"* — i.e. reimplementations of the head-split program variants that already ship in `ttnn.experimental.nlp_create_qkv_heads` / `nlp_concat_heads`, gated by `QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1` / `QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1` (both set by default in `demo/_common.py`). Measured contribution in the masked path: **head-split ON = 7.9ms vs OFF = 9.2ms (−1.3ms)** — already realized. Porting bge_m3's versions yields no further gain. The remaining 0.8ms gap vs the 7.1ms maskless path is the masked SDPA softmax (~0.4ms) + real-token pooling (~0.4ms), which the head ops do not touch. For full-length (~512-token) inputs the maskless `fast` path already delivers 7.1ms at near-reference accuracy.

## Implementation Notes

- **No `tt_transformers` modifications**: all changes are localized to this directory (except one-line `trust_remote_code` fix in `model_config.py`).
- **Bidirectional SDPA**: `PplxBidirectionalAttention` wraps the SDPA call in `forward_prefill` to inject `is_causal=False`.
- **Weight loading**: `PplxModelArgs` subclass loads weights directly from safetensors, bypassing the custom HF `modeling.py` that requires a newer `transformers` version.
- **Centralized optimization config**: `WORKLOAD_CONFIGS` in `_common.py` maps every `(batch_size, seq_len)` pair to its optimized env-var settings. Both individual demo files and `dp32_multiprocess.py` use `apply_workload_env()` for consistent tuning.
- **Direct trace replay**: Benchmark loop uses `ttnn.execute_trace` directly for lowest-overhead measurement. Pass `--full-pipeline` to measure end-to-end latency with an *extended trace* that includes post-processing ops (slice + norm + to_layout) inside the trace replay, reducing dispatch overhead to ~0.1ms.
- **DP=32 multi-process**: `dp32_multiprocess.py` spawns one `multiprocessing.Process` per chip with `TT_VISIBLE_DEVICES` isolation, CPU affinity pinning, and a `Barrier` for synchronized measurement.
- **2-CQ overlapped-input serving (opt-in, `live_demo.py --fast --cq2`)**: implements the standard multi-command-queue trace structure (cf. `models/tt_cnn/tt/executor.py:MultiCQTracedModelOverlappedInputExecutor` and the vision `performant_runner`): input H2D on CQ1 into a DRAM staging buffer, copied into the trace input on CQ0, with `op_event`/`write_event` coordination so the next request's H2D overlaps the current compute. **Measured to give no benefit for pplx-embed** — pipelined throughput is identical to 1-CQ (≈197 vs 198 emb/s @128, 140 vs 140 @512) because the per-request input is just the token tensor (~tiny): its H2D is fixed latency, not bandwidth, and the 5–7ms device compute dominates with nothing to overlap. Kept off by default (1 CQ); the flag exists for completeness. (2-CQ helps when the per-request input transfer is large, e.g. image tensors in the vision demos.)
