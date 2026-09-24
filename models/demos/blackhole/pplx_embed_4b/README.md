# pplx-embed-v1-4B on Tenstorrent Blackhole

Text-embedding inference for [perplexity-ai/pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b)
on Tenstorrent Blackhole (P150 single device, and multi-chip via data
parallelism).

This directory holds the optimized pplx-embed-v1-4B stack: live serving,
sequence-length bucketing, masked pooling, a data-parallel harness, and the
memory-placement and kernel optimizations for the
[Qwen3-Embedding-4B](../qwen3_embedding_4b/README.md) backbone (same 2560-d,
36-layer Qwen3-4B architecture). This README covers how to run every script,
what each produces, and the optimizations applied.
The same stack runs the causal `Qwen/Qwen3-Embedding-4B` checkpoint unchanged
(`HF_MODEL=Qwen/Qwen3-Embedding-4B`; numbers in [PERF.md](PERF.md), instructions in
[../qwen3_embedding_4b/README.md](../qwen3_embedding_4b/README.md)).

---

## Model

pplx-embed-v1-4B is a Perplexity AI text-embedding model on the **Qwen3-4B**
backbone (diffusion continued pre-training).

| Property      | Value                       |
|---------------|-----------------------------|
| Parameters    | 4B                          |
| Hidden dim    | 2560                        |
| Layers        | 36                          |
| Q / KV heads  | 32 / 8  (GQA 4:1)           |
| Head dim      | 128                         |
| Intermediate  | 9728                        |
| Max context   | 32K                         |
| Attention     | Bidirectional (non-causal)  |
| Pooling       | Mean over real tokens       |
| Output        | (optionally L2-normalized) 2560-d vector |

It differs from Qwen3-Embedding-4B in three ways: **bidirectional attention**
(no causal mask), **mean-token pooling** (not last-token), and it requires
`trust_remote_code=True` for HuggingFace loading.

### Placement at a glance

| Aspect            | 4B            |
|-------------------|---------------|
| bs=1 ISL=512 activation  | 2.5 MB, L1-resident   |
| bs=32 ISL=512 activation | 80 MB, DRAM-resident  |
| DRAM matmul grid  | full worker grid: 12×10 = 120 here (13×10 on p150a) |
| RMSNorm           | block-sharded at bs=1 (10×8); fused residual add + RMSNorm at bs≥8 |

---

## 1. Setup

```bash
# From the tt-metal repo root, with the Python env active:
source python_env/bin/activate

export HF_MODEL=perplexity-ai/pplx-embed-v1-4b
export HF_HOME=$HOME/.cache/huggingface   # must be writable + have network access
export MESH_DEVICE=P150
```

- The model config/weights are pulled from the HuggingFace hub on first run and
  cached locally; subsequent runs load from cache. The 4B checkpoint is sharded
  across multiple safetensors files — both the sharded (`index.json` weight map)
  and single-file layouts are handled by `PplxModelArgs.load_state_dict`.
- On a **single P150**, scripts run on device 0 by default — just run them.
- On a **multi-chip host**, select a chip with `TT_VISIBLE_DEVICES=<chip>` and
  keep `--device-id 0`. The multi-process scripts (`dp32_multiprocess.py`,
  `live_demo.py --dp N`, `eval_accuracy_tt.py --num-devices N`) handle this
  isolation automatically.

---

## 2. Memory placement — making the most of L1 + DRAM

The single biggest performance lever on the 4B model is **where the activations
live and how wide the matmul grid is**. The P150 single-user prefill keeps
activations in **L1** when the per-user sequence is ≤ 512
(`TT_SHORT_SEQ_L1_PREFILL_MAX`, default 512). Activation bytes (bf16) =
`bs × seq × 2560 × 2`:

| Workload        | Activation | Placement | Matmul kernel / grid | Sustained latency (median of it 15–29 of 30) |
|-----------------|-----------:|-----------|----------------------|----------------------------------------------|
| bs=1  ISL=512   |  2.5 MB    | **L1** (single-user) | legacy 2D multicast, 12×8 (96 cores) | **17.6 ms · 29.1k tok/s** |
| bs=4  ISL=512   | 10.5 MB    | DRAM ᵃ    | `minimal_matmul`, 12×10 (120 cores) | **69.0 ms · 29.7k tok/s** |
| bs=8  ISL=512   | 20 MB      | DRAM      | `minimal_matmul`, 12×10 | **120.9 ms · 33.9k tok/s** |
| bs=16 ISL=512   | 40 MB      | DRAM      | `minimal_matmul`, 12×10 | **227.6 ms · 36.0k tok/s** |
| **bs=32 ISL=512** | **80 MB** | DRAM    | `minimal_matmul`, 12×10 | **446.4 ms · 36.7k tok/s** |
| bs=1  ISL=1024 / 2048 | 5 / 10 MB | DRAM | `minimal_matmul`, 12×10 | not re-measured since the 2026-09-22 baseline |

ᵃ bs=4 moved to the DRAM path on 2026-09-24: the batched-L1 placement now clashes with the fused ops'
circular buffers, and DRAM is faster with them anyway (65 ms best of 10 vs the 75 ms the L1 path read before).
Measured 2026-09-24 on a Galaxy P150 exposing **12×10 = 120 worker cores** (a p150a card exposes 13×10);
"sustained" is after the board's power manager has settled the clock at ≈1.1–1.3 GHz under continuous
load, which is the like-for-like comparison against a steady-state H200 (bs=1 5.44 ms, bs=8 33.08,
bs=16 67.23, bs=32 139.15): **3.24× / 3.65× / 3.39× / 3.21×**. Qwen3-Embedding-4B through the same
stack (`HF_MODEL=Qwen/Qwen3-Embedding-4B`): 18.4 / 120.8 / 228.2 / 445.1 ms. Baseline, same method:
28.8 / 190.3 / 372.7 / 720.8 ms — the full path is in [PERF.md](PERF.md).

### Optimization history

Every landing from the 2026-09-22 baseline to the numbers above, with its mechanism and measured effect, is in
[PERF.md](PERF.md). Most of the early wins were inherited configuration constants that did not fit the
4B shapes; re-check them before reusing this config on another model in the family.

- **L1 path (bs=1, ISL≤512):** activations stay resident in L1, so the residual stream never
  round-trips DRAM; the matmuls are the legacy 2D-multicast kernel on 12×8 with DRAM-width-sharded
  bfp4 weights (at M=512 it beats `minimal_matmul` by 53–65%).
- **DRAM path (bs≥4):** activations live in DRAM, which frees the per-core L1 for `minimal_matmul` on
  the full worker grid (12×10 here). Matmuls are ≈65% of device time at bs=32, the fused SwiGLU product
  13%, the fused residual add + RMSNorm 8%, SDPA 8%.

### What the per-op SQLite memory report showed

The placement above was tuned by reading the ttnn-visualizer memory report (the
per-op live-buffer snapshot in `generated/ttnn/reports/<name>/db.sqlite` — the
`buffers` table: `operation_id, address, max_size_per_bank, buffer_type` with
0=DRAM, 1=L1). Regenerate it with
`tests/perf/gen_mem_report.py` (runs one eager prefill and snapshots
`ttnn._ttnn.reports.get_buffers` after every op):

| Path | Peak L1 / core | of 1464 KB | Idle L1 / core | Peak DRAM |
|------|---------------:|-----------:|---------------:|----------:|
| bs=1 (L1)   | 170 KB | 11.6 % | 1293 KB | 4.3 GB |
| bs=8 (DRAM) |   8 KB |  0.6 % | 1456 KB | 5.7 GB |

The DRAM path leaves **~1.45 MB of L1 per core idle** while activations
round-trip through DRAM — the 4.3–5.7 GB of resident DRAM is almost entirely
**weights** (BFP4/BFP8), which can never fit in L1 (130×1.46 MB ≈ 190 MB total),
so each weight streams from DRAM once per prefill. The actionable win is to put
the *activations* back into that idle L1: bs=4 (10.5 MB) fits and goes
**90.8 → 74.8 ms (+21 %)**. bs=8 (20 MB) does **not** — its 9728-wide FF
intermediate overflows L1 (`static CB region clashes with L1 buffer`), so bs≥8
stays on DRAM. The batched-L1 cap is therefore set to **12 MiB**
(`TT_BATCHED_L1_PREFILL_MAX_BYTES`): admits bs≤4, excludes bs≥8.

All of this is wired up automatically per workload in
`demo/_common.py::WORKLOAD_CONFIGS` / `apply_workload_env`, so every demo and the
DP scripts pick the right placement + grid for their shape with no extra flags.

---

## 3. Scripts — how to run them

| Script | What it does / produces |
|--------|--------------------------|
| `demo/demo_bs{1,4,8,32}_isl{512,1024,2048}.py` | Single-device **latency benchmark** for one (batch, ISL). Times the extended trace (forward + pooling + I/O in one replay) by default and prints avg/best time, embeddings/s and tokens/s; `--no-full-pipeline` times the bare forward replay. |
| `demo/dp32_multiprocess.py` | **Data-parallel benchmark** across N chips (one resident model per chip). Prints per-chip latency (mean/median/min/max), slowest-chip latency and aggregate throughput. `--mean-pool` runs the real serving post-processing (RMSNorm + mean-token pooling folded in-trace). |
| `demo/live_demo.py` | **Resident encoder** — loads the model once and keeps it up. Embed your own text interactively, from a file (one text/line), or from a folder (one doc/file). `--fast` = low-latency traced serving; `--mask` = accurate for short/variable inputs; `--dp N` = serve across N chips; `--bench N` = report per-request latency. |
| `demo/eval_accuracy.py` | **CPU fp32 reference** accuracy (STS-B Spearman / SciFact nDCG@10). The ground-truth baseline the device is compared against. |
| `demo/eval_accuracy_tt.py` | **On-device accuracy** (STS-B Spearman) with all perf flags on. `--pool {fast,masked,masked-attn}`, `--num-devices N` (DP), `--no-bucket` to disable sequence-length bucketing. |
| `tests/perf/new_perf_bs{1,8,16,32}_isl512.py` | Tracy profiling signpost tests for op-level device timing (developer profiling). |

### 3.1 Latency benchmarks (single device)

```bash
# bs=1, ISL=512 (L1-resident; the extended trace is the default)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py

# bare forward replay only (device time without pooling / I/O)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py --no-full-pipeline

# bs=4, ISL=512 (DRAM-resident activations, minimal_matmul on the full grid)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs4_isl512.py --full-pipeline

# DRAM-resident shapes on the full worker grid
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs8_isl512.py
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs32_isl512.py

# pick a chip on a multi-chip host
TT_VISIBLE_DEVICES=5 python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py --device-id 0

# pytest form (CI dashboards)
MESH_DEVICE=P150 pytest models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py -sv
```

### 3.2 Data parallelism

```bash
# 32 chips, bs=1 ISL=512, real serving post-processing
python models/demos/blackhole/pplx_embed_4b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 20 --mean-pool
```

### 3.3 Running the model on your own inputs (resident serving)

```bash
# Interactive: model stays loaded; type text, press Enter to embed it.
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py --fast

# Accurate for short / variable-length inputs (adds SDPA padding mask):
TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py --fast --mask

# A file of texts (one per line) -> .npy of embeddings
python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py \
  --fast --mask --input my_texts.txt --output embeddings.npy

# A folder (each file = one document) -> JSONL {name, embedding}
python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py \
  --fast --mask --input ./docs/ --output embeddings.jsonl

# Serve across multiple chips at once
python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py --dp 32 --mask

# Measure per-request latency (per sequence-length bucket)
python models/demos/blackhole/pplx_embed_4b/demo/live_demo.py --fast --bench 30 --max-length 512
```

Useful `live_demo.py` flags: `--max-length` (max tokens/text, default 512),
`--no-normalize` (skip L2 norm), `--no-bucket` (single fixed-ISL trace instead of
length buckets), `--metrics` (per-request device/D2H/host/H2D breakdown).

### 3.4 Accuracy

```bash
# CPU fp32 reference (STS-B Spearman)
python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy.py --dataset stsb

# On-device, single chip (recommended accurate masked-attn path)
python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_tt.py --pool masked-attn

# On-device across 32 chips
python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_tt.py --pool masked-attn --num-devices 32
```

---

#### Accuracy through the batched paths (bs8/16/32)

`eval_accuracy_tt.py` encodes one text per forward, so it never runs the batched kernels and program
configs. `demo/eval_accuracy_batched.py` runs STS-B through the perf demo's exact batch-B
configuration (`apply_workload_env(B, 512)`, eager forward, final RMSNorm on host, masked mean over
the real tokens; fixed ISL 512, no bucketing, no attention mask, so the numbers compare across batch
sizes but sit below the bucketed bs1 script's 0.8161):

```bash
TT_VISIBLE_DEVICES=10 python models/demos/blackhole/pplx_embed_4b/demo/eval_accuracy_batched.py --batch 8 --save /tmp/embs_B8.pt
```

| path | STS-B Spearman (masked mean) | per-text cosine vs batch 1 (mean / p1) | 1379 pair-cosines vs batch 1 (Pearson) |
|---|---|---|---|
| batch 1 | 0.8121 | — | — |
| batch 8 | 0.8123 | 0.983 / 0.917 | 0.9946 |
| batch 16 | 0.8140 | 0.983 / 0.912 | 0.9943 |
| batch 32 | 0.8159 | 0.982 / 0.908 | 0.9933 |

Measured 2026-09-24 with the shipped defaults (row-split add+RMSNorm, SDPA 12×8 at bs8, interleaved
weights at bs32, bfp4 weights, bfp8 activations). The per-text spread is the bfp8 pipeline's normal
sensitivity (a benign kernel change moves per-token cosines by a similar amount after 36 layers, see
the negatives file §16/§33); the task metric is unchanged or better at every batch size.

## 4. Embedding API

Build the model once with `build_single_device_model()` and wrap it in a
resident encoder. The encoder captures the bidirectional prefill as a hardware
trace and replays it per request, folding the final RMSNorm + mean-token pooling
onto the device, so each `encode()` returns a post-processed `[2560]` embedding
at the benchmarked latency.

```python
import ttnn
from models.demos.blackhole.pplx_embed_4b.demo._common import (
    apply_workload_env,
    build_single_device_model,
)
from models.demos.blackhole.pplx_embed_4b.demo.live_demo import (
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

| `pool`     | `use_mask` | Behavior | Use when |
|------------|------------|----------|----------|
| `"fast"`   | `False`    | Device mean over the full padded ISL. Lowest latency. | Full-length (~512-token) inputs. |
| `"masked"` | `True`     | Real-token mean pooling + SDPA padding mask (the `masked-attn` path). Near-reference accuracy for any length. | Short / variable-length inputs (recommended default). |

`encode_one(...)` runs the same forward **eagerly** (no trace, minimal padding to
the nearest 128) — handy for one-off calls or debugging.

### Example: cosine-similarity scoring

```python
import torch

sentences_1 = ["What is pplx-embed?", "Definition of BM25"]
sentences_2 = [
    "pplx-embed-v1-4B is a bidirectional text-embedding model from Perplexity AI.",
    "BM25 is a bag-of-words retrieval function that ranks documents by query-term matches.",
]

def encode(sentences):
    return torch.stack([encoder.encode(s, normalize=True) for s in sentences])

embeddings_1 = encode(sentences_1)
embeddings_2 = encode(sentences_2)

# Vectors are L2-normalized, so the dot product is the cosine similarity.
similarity = embeddings_1 @ embeddings_2.T
print(similarity)            # [2, 2]: diagonal pairs score highest

ttnn.close_device(device)
```

---

## 5. Profiling

```bash
# bs=1 (L1 path) Tracy device profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/blackhole/pplx_embed_4b/tests/perf/new_perf_bs1_isl512.py -sv

# bs=32 (DRAM-resident, full worker grid) Tracy device profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/blackhole/pplx_embed_4b/tests/perf/new_perf_bs32_isl512.py -sv
```

Filter the resulting `ops_perf_results_*.csv` to ops between the `start` and
`stop` signposts. On the Qwen3-Embedding-4B backbone (same compute), matmuls
dominate ≈60% of device time (FF2 > FF1/FF3 > QKV > WO), with SDPA+RoPE next,
then norms and element-wise — so the BFP4 weight quantization and the wide
matmul grid target the largest cost first.

### Per-op L1/DRAM memory report (ttnn-visualizer)

To see *where every op puts its buffers* (the data behind Section 2's placement
decisions), generate the per-op memory snapshot. It reproduces the
ttnn-visualizer `db.sqlite` `buffers` table by snapshotting
`ttnn._ttnn.reports.get_buffers([device])` after each op during one eager prefill:

```bash
python models/demos/blackhole/pplx_embed_4b/tests/perf/gen_mem_report.py --batch 1 --seq 512 --out /tmp/mem_bs1.csv
python models/demos/blackhole/pplx_embed_4b/tests/perf/gen_mem_report.py --batch 8 --seq 512 --out /tmp/mem_bs8.csv
```

It prints peak L1 per core, L1 headroom, peak DRAM, and the top ops by live L1,
and writes a per-op CSV (`op_idx, op_name, l1_per_bank_bytes, dram_total_bytes,
n_l1_bufs, n_dram_bufs`). This is exactly how the bs=4 batched-L1 win was found
(see Section 2 — the DRAM path was leaving ~1.45 MB/core of L1 idle).

---
