# pplx-embed-v1-4B on Tenstorrent Blackhole

Text-embedding inference for [perplexity-ai/pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b)
on Tenstorrent Blackhole (P150 single device, and multi-chip via data
parallelism).

This is the 4B sibling of [`pplx_embed_0_6b`](../pplx_embed_0_6b/README.md): it
reuses that model's serving/eval tooling (live serving, bucketing, masked
pooling, DP harness) and the memory-placement optimizations validated on the
[Qwen3-Embedding-4B](../qwen3_embedding_4b/README.md) backbone (same 2560-d,
36-layer Qwen3-4B architecture). This README covers how to run every script,
what each produces, and the optimizations applied.

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

It differs from Qwen3-Embedding-4B in three ways (identical to the 0.6B pplx
model): **bidirectional attention** (no causal mask), **mean-token pooling**
(not last-token), and it requires `trust_remote_code=True` for HuggingFace
loading.

### Key differences from the 0.6B pplx model

| Aspect            | 0.6B   | 4B            |
|-------------------|--------|---------------|
| Hidden size       | 1024   | 2560          |
| Layers            | 28     | 36            |
| Q-heads           | 16     | 32            |
| bs=1 ISL=512 act  | 1 MB   | 2.5 MB (L1)   |
| bs=32 ISL=512 act | 32 MB  | 80 MB (DRAM)  |
| DRAM matmul grid  | 80-core (8×10) | **130-core (13×10)** |
| LN block sharding | active | auto-disabled (dim 2560 > per-core cap) |

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

| Workload        | Activation | Placement | Matmul grid | Best (full pipeline) |
|-----------------|-----------:|-----------|-------------|----------------------|
| bs=1  ISL=512   |  2.5 MB    | **L1** (single-user) | standard (8×8) | 30.4 ms · 16.4k tok/s |
| bs=2  ISL=512   |  5 MB      | **L1** (batched) | 130-core | 67.4 ms · 15.2k tok/s |
| **bs=4  ISL=512** | **10.5 MB** | **L1** (batched) | **130-core (13×10)** | **74.8 ms · 27.4k tok/s** |
| bs=8  ISL=512   | 20 MB      | DRAM | 130-core | 172 ms · 23.8k tok/s |
| bs=32 ISL=512   | 80 MB      | DRAM | 130-core | ~686 ms · 23.9k tok/s |
| bs=1  ISL=1024  |  5 MB      | DRAM | 130-core | — |
| bs=1  ISL=2048  | 10 MB      | DRAM | 130-core | — |

- **L1 path (bs≤4, ISL≤512):** activations stay resident in L1, eliminating DRAM
  round-trips for the residual stream. **bs=4 batched-L1 is the throughput-optimal
  config** (27.4k tok/s — higher than bs=8/bs=32) once the matmul grid is widened.
- **DRAM path (bs≥8):** activations spill to DRAM, which *frees the per-core L1
  budget*. We spend that freed budget by widening the MinimalMatmul grid to the
  **full 130-core (13×10) Blackhole grid** (`QWEN_MM_GRID=13,10`). Matmuls dominate
  ≈60% of device time, so this is the dominant DRAM-path win (the analogous
  80→130-core change gave ≈18% on Qwen3-Embedding-4B).

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
| `demo/demo_bs{1,4,8,32}_isl{512,1024,2048}.py` | Single-device **latency benchmark** for one (batch, ISL). Prints avg/best prefill time, embeddings/s and tokens/s. Add `--full-pipeline` for end-to-end latency (H2D + replay + post-proc + D2H). **`demo_bs4_isl512.py` is the throughput-optimal config (27.4k tok/s, batched-L1).** |
| `demo/dp32_multiprocess.py` | **Data-parallel benchmark** across N chips (one resident model per chip). Prints per-chip latency (mean/median/min/max), slowest-chip latency and aggregate throughput. `--mean-pool` runs the real serving post-processing (RMSNorm + mean-token pooling folded in-trace). |
| `demo/live_demo.py` | **Resident encoder** — loads the model once and keeps it up. Embed your own text interactively, from a file (one text/line), or from a folder (one doc/file). `--fast` = low-latency traced serving; `--mask` = accurate for short/variable inputs; `--dp N` = serve across N chips; `--bench N` = report per-request latency. |
| `demo/eval_accuracy.py` | **CPU fp32 reference** accuracy (STS-B Spearman / SciFact nDCG@10). The ground-truth baseline the device is compared against. |
| `demo/eval_accuracy_tt.py` | **On-device accuracy** (STS-B Spearman) with all perf flags on. `--pool {fast,masked,masked-attn}`, `--num-devices N` (DP), `--no-bucket` to disable sequence-length bucketing. |
| `tests/perf/new_perf_bs{1,8,16,32}_isl512.py` | Tracy profiling signpost tests for op-level device timing (developer profiling). |

### 3.1 Latency benchmarks (single device)

```bash
# bs=1, ISL=512 (L1-resident, pure device trace replay)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py

# end-to-end latency (H2D + replay + post-processing + D2H)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs1_isl512.py --full-pipeline

# bs=4, ISL=512 — throughput-optimal (batched-L1 activations + 130-core grid)
python models/demos/blackhole/pplx_embed_4b/demo/demo_bs4_isl512.py --full-pipeline

# DRAM-resident shapes with the 130-core matmul grid
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

## 5. Optimizations

Applied by default across all workloads (centralized in
`demo/_common.py`, shared by every demo, the live serving path, and the DP
scripts):

- **Memory placement** (Section 2): L1-resident activations for bs≤4/ISL≤512
  (batched-L1, 12 MiB cap — bs=4 is the throughput-optimal config at 27.4k tok/s);
  DRAM + the **130-core (13×10)** MinimalMatmul grid for every larger shape.
- **BFP4 weights** for QKV + WO **and FF2 (down_proj)** projections. FF2 BFP4 is
  on by default after on-device validation on this P150 (ISL=512, masked-attn
  pool): STS-B Spearman `0.8287` (FF2 BFP8) → `0.8276` (FF2 BFP4) — a 0.0011 delta
  within run-to-run noise — while latency drops bs1 `31.2→30.4 ms` (−2.6%) and
  bs32 `727.8→685.6 ms` (−5.8%, 22.5k→23.9k tok/s). FF2 was the single most
  expensive matmul in the profile (224 µs at HiFi2/BFP8 vs ~100 µs for the
  BFP4/LoFi matmuls). Opt back out with `QWEN_FF2_BFP4=0`.
- **Full BFP8 residual stream** — FF1/FF3 output, FFN-norm input, and the post-FFN
  residual add all in BFP8.
- **LoFi math fidelity** for matmuls, SDPA, and RoPE. RoPE is a cos/sin rotation
  (operands in [-1,1]), so HiFi4 is wasted precision — LoFi is accuracy-neutral
  and cheaper (see `tt/attention.py::_mllama_rope_prefill`).
- **Head-split QKV / concat-heads** — native head-split program variants of
  `nlp_create_qkv_heads` / `nlp_concat_heads` (n_kv_heads=8 → 128 work units at
  bs=1/ISL=512).
- **RoPE cos/sin tables in L1** (128 KB, well within budget).
- **KV-cache fill skip** (prefill-only, no decode) and **bidirectional SDPA**
  (`is_causal=False`).
- **Hardware trace capture** with an *extended trace* that folds the RMSNorm +
  mean-token pooling post-processing into the replay (only the pooled 2560-d
  vector is copied back).
- **Sequence-length bucketing** in the resident serving path: one trace per
  padded-length tier (128/256/512…); short inputs run a smaller/faster trace
  with less padding.

Two optimizations are intentionally **not enabled** (documented in
`tt/attention.py`): skipping the trained Q/K RMSNorm (load-bearing — collapses
retrieval accuracy when removed) and `QWEN_LN_BLOCK_SHARDED` is set but inert for
the 4B model (dim=2560 exceeds the 16-tile per-core LN budget, so it auto-disables).

All env vars use `os.environ.setdefault`, so any single knob can be overridden
from the shell for A/B comparisons, e.g.:

```bash
# Disable the wide matmul grid for a baseline comparison at bs=32
QWEN_MM_GRID= python models/demos/blackhole/pplx_embed_4b/demo/demo_bs32_isl512.py
```

---

## 6. Profiling

```bash
# bs=1 (L1 path) Tracy device profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/blackhole/pplx_embed_4b/tests/perf/new_perf_bs1_isl512.py -sv

# bs=32 (DRAM + 130-core grid) Tracy device profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
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

## Implementation notes

- All changes are localized to this directory; the bidirectional attention and
  weight-loading live in `tt/attention.py` and `demo/_common.py::PplxModelArgs`.
- `PplxBidirectionalAttention` (`tt/attention.py`) wraps SDPA with
  `is_causal=False`, applies the LoFi RoPE kernel config, and is shared verbatim
  with the 0.6B model (only scale dimensions differ).
- `PplxModelArgs` loads weights directly from (sharded) safetensors, avoiding the
  custom HF `modeling.py` that requires a newer `transformers`.
- `dp32_multiprocess.py` / `live_demo.py --dp N` spawn one process per chip
  (`TT_VISIBLE_DEVICES` isolation + CPU-affinity pinning).
