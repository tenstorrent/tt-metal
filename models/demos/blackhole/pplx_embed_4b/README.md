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

| Workload        | Activation | Placement | Matmul grid | Best prefill |
|-----------------|-----------:|-----------|-------------|--------------|
| bs=1  ISL=512   |  2.5 MB    | **L1** (single-user) | standard (8×8) | **25.9 ms · 19.8k tok/s** |
| bs=2  ISL=512   |  5 MB      | **L1** (batched) | 130-core | 67.4 ms · 15.2k tok/s ᵃ |
| bs=4  ISL=512   | 10.5 MB    | **L1** (batched) | 130-core (13×10) | 74.8 ms · 27.4k tok/s ᵃ |
| bs=8  ISL=512   | 20 MB      | DRAM | 130-core | **157.6 ms · 26.0k tok/s** |
| bs=16 ISL=512   | 40 MB      | DRAM | 130-core | **291.4 ms · 28.1k tok/s** |
| **bs=32 ISL=512** | **80 MB** | DRAM | 130-core | **558.0 ms · 29.4k tok/s** |
| bs=1  ISL=1024  |  5 MB      | DRAM | 130-core | — |
| bs=1  ISL=2048  | 10 MB      | DRAM | 130-core | — |

ᵃ bs=2 / bs=4 rows are the pre-optimization figures; not re-measured since.
All other rows measured on a **harvested P150 exposing 12×10 = 120 worker
cores** (nominal 13×10 = 130), so they are roughly 8% pessimistic against a
full part. For reference, H200 FP8 at the same shapes: bs=1 5.44 ms,
bs=8 33.08, bs=16 67.23, bs=32 139.15 — i.e. 4.0-4.8x.

### Optimization history

Against the BFP8 / FF13-BFP4 reference configuration
(bs=1 44.08 ms, bs=8 185.01, bs=16 366.23, bs=32 690.53) the current defaults
are **-41% / -15% / -20% / -19%**. STS-B Spearman 0.8125 throughout
(pre-optimization 0.8116). What moved, in order of size:

| Change | Env | Effect |
|---|---|---|
| MinimalMatmul output subblock (was 1x1) | `QWEN_MM_SUBBLOCK=1,8` | bs8 -12%, bs16 -13%, bs32 -18% |
| Head-split QKV + concat (`tt/custom_ops`) | `QWEN_NLP_*_HEAD_SPLIT=1` | bs1 -2.7 ms, bs32 -8.8 ms |
| `in0_block_w` cap 8 -> 38 (FF2 was pinned at 2) | `QWEN_MM_MAX_DIVISOR=38` | bs1 -4.4 ms |
| Fused SwiGLU MLP (`tt/mlp.py`) | `QWEN_FUSE_SWIGLU` | bs16 -5.7%, bs8 -1.6%, **off at bs32** |
| Block-sharded LayerNorm (was silently inert) | `QWEN_LN_GRID_MAX_X=10` | bs1 -1.9 ms |
| SDPA q/k chunk | `QWEN_SDPA_Q_CHUNK=512` / `_K_CHUNK=256` | bs8/16/32 -2..-4% |

Most of these were constants tuned for the 0.6B sibling that silently
mis-applied to 4B, which has 2.5x the hidden size and 2x the head count.
Re-check them before reusing this config on another model in the family.

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

## 5. Optimizations

A concise record of every landed optimization and its measured effect per batch size is kept in
`perf_csv/POSITIVE_RESULTS.md` next to the repo (with `perf_csv/NEGATIVE_RESULTS.md` for what did not work); the
sections below carry the details.

Applied by default across all workloads (centralized in
`demo/_common.py`, shared by every demo, the live serving path, and the DP
scripts):

- **Memory placement** (Section 2): L1-resident activations for bs≤4/ISL≤512
  (batched-L1, 12 MiB cap — bs=4 is the throughput-optimal config at 27.4k tok/s);
  DRAM + the **full-device MinimalMatmul grid** for every larger shape. A P150
  is nominally 13×10 but ships harvested: this board reports 12×10 = **120**
  workers, and the profile confirms `cores=120` on every batched matmul.
  `_clamp_grid_to_device` keeps the request portable across harvest configs.
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

### Tested and rejected (2026-09-23)

Measured end-to-end on this P150, ISL=512, 10 iterations, best-of. Baseline
reproduced the committed FINAL numbers exactly, so these deltas are real.

| config | bs1 | bs8 | bs16 | bs32 |
|---|---|---|---|---|
| **baseline (shipping)** | **25.9 ms** | **157.1 ms** | **290.9 ms** | **557.8 ms** |
| SDPA `grid=12x10` | 26.0 (+0.4%) | 156.3 (−0.5%) | 291.2 (+0.1%) | 555.3 (−0.4%) |
| SDPA `10x10` + `q_chunk=128` | 26.0 (+0.4%) | 170.4 (+8.5%) | 326.9 (+12.4%) | 617.4 (+10.7%) |

**Block-sharded activations.** Suggested on the theory that sharding reduces
data movement within an op. Tested tuned-vs-tuned (both arms swept over grids
8×8→12×10, every legal `in0_block_w` divisor, and every `out_subblock_h*w ≤ 8`):

- bs1 path (`MatmulDeviceOperation`, 2D mcast, in0 already in L1): block-sharded
  is **+1.0% to +3.5%** across QKV/FF13/FF2/WO — a wash. The mcast broadcasts in0
  along each core row regardless, so sharding changes where in0 *starts*, not how
  many bytes cross the NoC.
- batched path (`MinimalMatmulDeviceOperation`, in0 DRAM-interleaved bfp8, 120
  cores): block-sharded is −2.6% / +0.3% / −2.4% / −3.7% on the four shapes that
  are 93% of batched matmul time. Share-weighted **−2.0% of matmul**, and matmul
  is ~39% of bs32 → **<1% e2e**, smaller than the spread between shard configs.

Contrast with LayerNorm, where sharding *did* pay (−1.9 ms at bs1): LN has a real
gather to remove, a mcast matmul does not.

**SDPA core utilization.** SDPA runs on 64 (bs1) / 80 (batched) of 120 cores, and
at bs1 `q_chunk=512` over `Sq=512` yields only `1*32*1 = 32` work units. An
isolated sweep suggested 94.6 → 76.5 µs. **This did not survive e2e** — the
microbenchmark was run at `HiFi2` while the model uses LoFi, so its 94.6 µs
"baseline" was already slower than the model's real 75.2 µs. The 76.5 µs "win"
only recovered ground the harness had lost. Shrinking `q_chunk` to create work
units actively hurts at batch (+8.5% to +12.4%), because per-chunk softmax and CB
overhead outweighs the extra parallelism.

`QWEN_SDPA_GRID=x,y` is retained as a default-off probe knob (clamped to the real
device grid) so the grid can be re-swept if the fidelity or shapes change.

**Lesson for future sweeps:** an isolated op microbenchmark must replicate the
model's dtype, memory config *and* compute-kernel config, or its baseline can be
slower than production and manufacture a win that does not exist.

**Custom prefill matmul op (matmul_decode analogue).** Suggested after
`smanoj/ds_v4_flash`, whose `matmul_decode` removes in0 movement and streams
weights through the DRAM prefetcher. Neither half transfers to this workload:

- *Weight prefetch.* Only pays when weight DRAM bandwidth is the limit, which
  `tech_reports/LLMs/llms.md` scopes explicitly to **decode**. Measured here by
  holding fidelity at LoFi and varying only weight dtype: bfp4 -> bfp8 doubles
  weight bytes but costs just **+12-15%** time (and **0%** for QKV). Measured
  DRAM is ~416 GB/s; the hottest matmul needs 177 GB/s. Weights are already
  overlapped.
- *in0 movement.* The op constrains in0 tile heights to `{1,2,4,8}` rows, so a
  512-row prefill activation needs a rewrite, not a port.

More decisively, the matmuls have little left to give. Measured in the **4-D
Z-batched form the model actually uses** (`[1, nchunks, cutoff, dim]`, one
launch for all chunks), at the bs32 row count of 16384:

| shape | ns/row | % of LoFi peak (580.9 TFLOPS) |
|---|---|---|
| FF13 `Z=32,M=512` / `Z=8,M=2048` / `Z=4,M=4096` | 106.8 / 106.5 / 106.6 | **80.3 / 80.5 / 80.5%** |
| FF2 `Z=32,M=512` / `Z=8,M=2048` / `Z=4,M=4096` | 97.2 / 96.9 / 96.6 | **88.2 / 88.5 / 88.8%** |

Only the **total row count** matters; the Z/M split does not. Matmul is ~33% of
bs32 device time at ~84% efficiency, so even a *perfect* custom matmul op is
worth **<6% e2e**. The remaining ~67% is non-matmul (LayerNorm, BinaryNg,
rotary, SDPA, layout conversions, dispatch) -- that is where the headroom is.

**`prefill_len_cutoff`.** Follows directly: the MLP reshape chunk size is
literally M for FF1/FF3/FF2, and since only total rows matter, changing it is a
no-op. Swept e2e and confirmed flat, so the default stays at 512:

| cutoff | bs1 | bs8 | bs16 | bs32 |
|---|---|---|---|---|
| **512 (shipping)** | **25.9** | **157.1** | **290.9** | **557.8** |
| 1024 | 25.9 | 156.2 | 290.9 | 557.3 |
| 2048 | 25.9 | 158.0 | 290.9 | 559.7 |
| 4096 | 25.9 | 157.9 | 290.9 | 558.2 |

`QWEN_PREFILL_LEN_CUTOFF` is retained as a default-off probe knob.

### L1-resident residual stream at batch — measured, net regression (2026-09-23)

Question (Sankar): per-op, block-sharding is a wash, but keeping activations
sharded/L1-resident *across the model* removes DRAM round-trips — does full-model
latency improve? Measured, same invocation (`TT_VISIBLE_DEVICES=0`), ISL=512,
10 iterations, best-of.

| batch | current best (DRAM) | residual L1 + block-sharded LN | Δ |
|---|---|---|---|
| bs1 | **25.9 ms** | 25.9 ms | 0 — already on the L1 path |
| bs8 | **156.4 ms** | 161.1 ms | **+3.0%** |
| bs16 | **290.9 ms** | does not fit — 43 KB/core short | — |
| bs32 | **557.8 ms** | does not fit — 43 KB/core short | — |

**What fits and what does not.** The wide transients cannot be resident: at bs8
Q and the SDPA output are ~280 KB/core each and the 9728-wide MLP intermediate
is 353 KB/core (1.4 MB/core at bs32). Only the 2560-wide residual (91 / 181 /
363 KB/core bfp8 at bs8/16/32) is in scope, so the measured config spills every
wide tensor to DRAM via the per-op `TT_PREFILL_{QKV,HEADS,SDPA,CONCAT,FF13}_L1=0`
knobs and keeps the residual plus the 2560-wide WO/FF2 outputs in L1.

**The cross-op effect is real but small, and what it costs to fit is larger.**
Decomposed at bs8, all at SDPA `q_chunk=256 / k_chunk=128`:

| config | bs8 | vs DRAM at the same SDPA chunks |
|---|---|---|
| DRAM activations | 166.3 ms | — |
| residual L1, all wide transients spilled | 163.7 ms | −1.6% |
| + WO/FF2 outputs also L1 | **161.1 ms** | **−3.1%** |

So the L1 residual stream buys −3.1% at matched chunks. But it only fits after
shrinking SDPA from the shipping 512/256 to 256/128, which alone costs **+6.3%**
(156.4 → 166.3): SDPA's static CBs at 512/256 are ~1.2 MB/core (q 256 KB, k/v
256 KB, a 512 KB q×k buffer, outputs), leaving ~300 KB for every L1 tensor
combined, which is why the very first attempt failed inside SDPA by 82 KB and a
later one by 1196 KB. Net vs shipping: +3.0%.

**bs16/bs32 stop at a `minimal_matmul` dataflow region 43 KB/core short**, at
identical addresses (`buf@820992`, region end `865408`) for both batch sizes —
so the blocking L1 tensor is fixed-size, not the residual (which doubles between
them). `QWEN_MM_BLOCK` at 4,8,8 and 2,8,8 did not move it (the same plateau seen
earlier at 53 KB); the RoPE tables are ~1 KB/core and not it. Unidentified; a
per-op L1 report (`ttnn-visualizer`, Section 6) would name it. Not pursued
because the bs8 result already shows the approach is net negative.

Two dead ends recorded so they are not repeated: `QWEN_MM_BLOCK` does not shrink
SDPA's circular buffers at all (region end unchanged to the byte), and it hits
a floor on the matmul dataflow region after the first M_block halving.
`N_block_size % subblock_w == 0`, so N reductions need a paired
`QWEN_MM_SUBBLOCK`.

**Knob semantics fix that made this measurable:** `TT_PREFILL_<op>_L1=0` now
forces DRAM. Previously "0" only skipped the intermediate-L1 branch and fell
through to the activation placement — which is itself L1 under
`TT_BATCHED_L1_PREFILL` — so the knobs could move an op output *into* L1 but
never *out*. Identical behaviour in the DRAM-activation regime.

### Where the time goes — steady state, per iteration (2026-09-23)

From the FINAL profiles, **warmup excluded** (the profiler CSVs also contain the
Generator's warmup passes, which run the LM head over the full vocabulary — 10
`[32×2560]×[2560×16032/7648]` matmuls plus S2Is per pass; those are *not* in the
traced iteration and must be filtered out before computing shares). Rescaled to
e2e so the columns sum to the measured latency.

| op (calls/layer) | bs1 | bs32 | note |
|---|---|---|---|
| Matmuls QKV+WO+FF13+FF2 (5) | 10.4 ms · 40% | 228 ms · 41% | bs32 at 80–89% of peak; **bs1 ~40%** |
| LayerNorm (3.9) | 1.3 · 5% | 84 · 15% | bs32 ≈ 2.7× off DRAM roofline |
| GenericOp head-split + concat (2) | 2.2 · 8.5% | **73 · 13%** | our custom op, 2.7× off roofline at bs32 |
| BinaryNg: 2 residual adds + SwiGLU mul (3) | **7.8 · 30%** | 72 · 13% | bs32 at ~DRAM roofline; **bs1 see below** |
| Rotary (2) | 1.9 · 7% | 46 · 8% | 3.3× off roofline |
| SDPA (1) | 1.9 · 7% | 37 · 7% | swept every axis; tapped |
| Typecast Q→bfp8 (1, batch only) | — | 18 · 3% | see below: not removable for free |

Gap to 3× H200: bs1 25.9 → **16.3** (−37%); bs32 557.8 → **417.5** (−25%). At bs32
the matmuls are near roofline so the −140 ms has to come from the non-matmul 59%;
at bs1 the lever is op count (~21 ops/layer at ~30 µs each).

**BinaryNg at bs1 is the op, not the model.** Standalone `ttnn.add` on
`[1,1,512,2560]` bf16 takes **150 µs from L1-interleaved inputs and 92 µs from
DRAM** (roofline ~19 µs); the model's 51–109 µs matches. Interleaved-L1 reads go
tile-by-tile over the NoC from other cores' L1, which is worse than DRAM bursts.
The fix under test is a block-sharded residual on the LayerNorm grid (each core
reads its own shard; also deletes the I2S before every LN).

**Q→BFP8 typecast before SDPA (batch only): tested, neutral-to-negative, kept.**
The cast exists to match Q to K/V, but with `skip_kv_cache_fill` K and V reach
SDPA as bf16 anyway, so it looked like 18 ms/iter of pure overhead at bs32.
Skipping it (`_prepare_q_for_sdpa` returning Q unchanged, gate confirmed active):
bs8 156.6 vs 156.4, bs16 290.8 vs 290.9, **bs32 563.7 vs 557.8 (+1.1%)**. SDPA on
bf16 Q (2× the Q bytes and Q-chunk CB) costs what the cast saved. Reverted; the
real fix is emitting Q in bfp8 from the producer (part of the fused QKV epilogue).

### SwiGLU product as a model-local op at bs32 (2026-09-23)

On the unfused path (bs32) the SwiGLU product is `ttnn.mul(a, b,
input_tensor_a_activations=[SILU])`; the multiply alone is at 95% of DRAM roofline but the
SiLU adds 333 µs per call (1258 → 1591 µs) — 12 ms per forward. `custom_ops/silu_mul`
streams a and b tile-by-tile, runs `silu_tile` on the DST tile and multiplies b in with a
dest-reuse FPU multiply, so the SFPU work overlaps the reads: **1373 µs (−13.7%)** at the
bs32 shape, and closer to torch than the stock path (PCC 0.99941 vs 0.99897; the stock
activation path is the less precise one). Same-chip A/B (chip 6): **bs32 443.8 → 438.1 ms
(−1.3%)**. At bs1 the op is slower (62 vs 59 µs — 40 tiles per core), so it applies from
8192 rows (`QWEN_SILU_MUL=1`, `QWEN_SILU_MUL_MIN_ROWS=8192`); bs8/bs16 run the fused SwiGLU
kernel and never reach it. Rejected variants: x·sigmoid_fast(x) (slower, PCC 0.9985) and
the Blackhole `clamped_silu_glu` SFPU op (slower and clamps at |x| = 10 — DeepSeek-V4
semantics, wrong for this model). `QWEN_SILU_MUL_VERIFY=1` prints per-call PCCs against
the stock op on live tensors.

### bs16 fused-SwiGLU matmul blocks 4,8,8 / 1×4 (2026-09-23)

A `minimal_matmul` block/subblock sweep with model-faithful weights (bfp4, DRAM
width-sharded; packed w13 interleaved as the model stores it) found the fused-SwiGLU
kernel at M=8192 prefers `M,K,N = 4,8,8` with `subblock 1×4`: 2945 vs 3133 µs for the
default 8,8,8 / 1×8 (−6.0%). Same-chip A/B (chip 8): bs16 **237.0 → 231.8 ms (−2.2%)**.
`apply_workload_env` sets `QWEN_MM_BLOCK_FF13=4,8,8` and `QWEN_MM_SUBBLOCK_FF13=1,4` for
bs16/ISL512 only: at bs8 the default is already the best of the sweep, and bs32 runs the
unfused path, where the defaults win for every projection. The same sweep says the plain
bs8 matmuls want other blocks (FF2 16,8,8 −16%, QKV 8,4,8 −15%, WO 16,8,8 −12%
standalone) — same-chip A/B (chip 7): bs8 **126.4 → 123.4 ms (−2.4%)**, now the bs8
default (`QWEN_MM_BLOCK_FF2=16,8,8`, `QWEN_MM_BLOCK_QKV=8,4,8`, `QWEN_MM_BLOCK_WO=16,8,8`).
A wider fused-kernel sweep then found K_block 20 for bs16 (`4,20,8` / 1×4): 2988 → 2874 µs
standalone, **232.8 → 228.3 ms e2e (−1.9%)**, now the bs16 default. Larger K steps are slower
for every *plain* projection at every batch (K10…K40: +2…+20%), so only the fused kernel takes
it. Full tables: `perf_csv/NEGATIVE_RESULTS.md` §29.

### Fused residual add + RMSNorm (bs16+) — landed (2026-09-23)

`custom_ops/fused_add_rmsnorm`: one `generic_op` computes ``sum = a + b`` (the next
residual, in the residual dtype) and ``rmsnorm(sum) * gamma`` in a single pass over
`[M, dim]` — a and b read once, both outputs written once (4 DRAM passes instead of
the stock add's 3 + the norm's 2). Per tile-row the compute adds, packs the sum twice
(residual dtype + a bfp8 working copy, so the norm sees exactly what the stock
add→norm path saw), squares, row-reduces with a 1/W scaler tile, rsqrt(+eps), then
scales and applies gamma on the DST tile. Rows are split with the fewest cores that
keep the same per-core maximum. `tt/decoder_fusion.py` installs it by wrapping each
layer's `forward`: the two residual-shaped `ttnn.add`s of a layer are intercepted and
the following `ff_norm` / the *next* layer's `attention_norm` hand back the
precomputed tensor (so both pairs per layer fuse, including the cross-layer one).

Standalone (bfp8 in/out, W=2560): M=16384 add+rms_norm 595.8 → fused 529.9 µs
(−11%, roofline 415), M=4096 183 → 177, M=512 53 → 63 (only 16 rows → 16 cores).
Accuracy vs torch fp32 at M=16384: PCC 0.99890 (fused) vs 0.99879 (stock path) —
both bfp8-limited, fused marginally closer. Same-chip A/B (chips 4/6/7/8):

| batch | H200 | before | **after** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 23.3  | 23.3  | (stock ops) | 4.29× |
| bs8  | 33.081  | 126.2 | 126.2 | (stock ops; fused measured +1.7%) | 3.81× |
| bs16 | 67.225  | 239.6 | **234.8** | **−2.0%** | 3.49× |
| bs32 | 139.150 | 450.6 | **443.5** | **−1.6%** | 3.19× |

`QWEN_FUSED_ADD_NORM=1` (default), `QWEN_FUSED_ADD_NORM_MIN_ROWS=8192` (flattened
rows below which the stock ops are kept). A row-split variant for bs1
(`fused_add_rmsnorm_split`: each of the 16 tile-rows over R=5 cores with a semaphore
partial-sum exchange, 80 cores) beats add + interleaved rms_norm standalone (35 vs 49 µs)
but loses to the model's block-sharded LN chain e2e (23.2 → 24.4 ms, +5%): at bs1 every op
is latency-bound and the fused op's fixed cost (~35 µs) exceeds the four stock kernels'.
Kept as a probe (`QWEN_FUSED_ADD_NORM_SPLIT=1`); `perf_csv/NEGATIVE_RESULTS.md` §34. `QWEN_FUSED_ADD_NORM_VERIFY=1` runs the stock
add + norm next to every fused call on the live model tensors and prints the PCCs: at bs16
all 72 calls of a forward gave sum ≥ 0.99988 and norm ≥ 0.9993 (most 1.0000).

A measurement caveat found on the way: comparing raw last-token hidden states between two
runs is *not* a usable equivalence metric on this bfp8 pipeline — a known-benign kernel
change (the v2 head-split compute, per-op PCC 0.9995, STS-B unchanged) moves them to
cos 0.87–0.97 after 36 layers. Compare final-normed, L2-normalised embeddings and the
agreement of the sentence similarity matrix instead (what retrieval depends on).

### Batched SDPA on all 120 workers with one K chunk (2026-09-23)

Standalone at the model's exact SDPA config (LoFi, exp approx, Q/K/V bfp8, non-causal,
q512/k256): grid 8×10 → 12×10 is −10.6% at B=32 (932 → 833 µs) and −11% at B=8;
k-chunk 512 another −3% (806 / 262 µs). bs1 cannot take it — its activations are
L1-resident and the k512 SDPA CBs clash with them (TT_THROW), and 32 work units
cannot fill 120 cores — so `apply_workload_env` sets `QWEN_SDPA_GRID=12,10` and
`QWEN_SDPA_K_CHUNK=512` for batch > 1 only (`QWEN_SDPA_BATCHED_WIDE=0` opts out).

Same-chip sequential A/B (chips 4/6/7/8; chip-to-chip e2e variation is ≈1.5%, so
only same-chip comparisons count from here on):

| batch | H200 | before | **after** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 23.3  | **23.3**  | — | 4.29× |
| bs8  | 33.081  | 126.7 | **126.2** | −0.4% | 3.81× |
| bs16 | 67.225  | 239.6 | **239.6** | — | 3.56× |
| bs32 | 139.150 | 455.1 | **450.6** | **−1.0%** | 3.24× |

Two things tried in the same session that did *not* pay, both recorded in
`perf_csv/NEGATIVE_RESULTS.md` (§24, §25): widening the bs1 legacy matmul grids
(the DRAM width-sharded weights pin the kernel to 8 core columns — 10×8/12×8 return
inf, and the decode-style DRAM-sharded matmul only supports M == 1 tile), and a
DST-reuse rewrite of the fused head-split compute plus cos/sin caching (standalone
−27%, in the model 0: the RoPE matrices already live in L1).

### generic_op launch cost: merged core ranges + fewest-cores split (2026-09-23)

The three model-local `generic_op`s (head-split+norm+RoPE, concat heads, plain
head-split) built their `CoreRangeSet` as one single-core `CoreRange` per worker.
The dispatcher then unicasts the kernel binaries to each of the 120 cores on every
launch instead of one multicast per kernel. Measured in trace replay at bs1
(`[1,1,512,6144]`, kernel ≈ 20 µs): 55.9 µs/op at 120 cores, 30.6 at 64, 19.7 at
32 — i.e. ≈0.4 µs per core of launch cost that a native op (`ttnn.add` of the same
size: 8.4 µs/op) does not pay. With merged rectangles: 13.0 µs/op at 120 cores,
concat heads 7.9 µs/op. The work split now also uses the fewest cores that keep
the same per-core maximum (bs1: 128 units → 64 cores × 2 instead of 120 cores of
which 8 carry 2; 60.3 → 51.7 µs/op for the norm+RoPE variant).

In the model the gain is smaller than standalone because the dispatcher writes the
next program's binaries while the previous (longer) op runs: bs1 23.7 → **23.4**,
bs8 127.5 → **126.7**, bs16 240.9 → **240.6**, bs32 456.7 → **455.8**; STS-B 0.8161
(unchanged — no numerics involved). Lesson for any further generic_op: never pass
per-core ranges; use `_core_ranges(per_core)` from `custom_ops/fused_qkv_heads/op.py`.

### Fused op emits Q and K/V in bfp8 — Typecast deleted (2026-09-23)

`fused_qkv_heads_norm` now packs Q into its own output CB in SDPA's operand dtype
(`QWEN_FUSED_Q_BFP8=force`) and K/V in bfp8 (`QWEN_FUSED_KV_BFP8=1`), both default
on. Rationale: at bs≥8 SDPA already consumed a bfp8 Q, produced by a separate
Typecast (469 µs/layer at bs32 — 17 ms per iteration, a pure DRAM pass over
134 MB in / 67 MB out); the packer can convert on the way out of the fused op for
free. Packing uses `bfp8_pack_precise` (identical kernel time; mean |err| vs the
un-quantised bf16 output 0.00601 — the same as the stock Typecast — versus
0.00654 in approximate mode, which is what `ttnn.generic_op` defaults to).
`_prepare_q_for_sdpa` returns Q untouched when it already has the target dtype.

| variant (B=8, traced, standalone) | µs | Δ |
|---|---|---|
| fused (bf16) + Typecast Q | 457.7 | — |
| fused emitting Q bfp8 | 321.9 | −29.7% |
| fused (bf16) + Typecast Q,K,V | 519.6 | — |
| fused emitting Q,K,V bfp8 | 312.0 | −39.9% |

E2E (extended trace, 10 iters, best-of), and STS-B Spearman at bs1, where the eval
runs — so it exercises the bfp8 Q/K/V path directly:

| batch | H200 | before | Q bfp8 | **Q + K/V bfp8 (shipped)** | Δ | × H200 |
|---|---|---|---|---|---|---|
| bs1  | 5.437   | 23.9  | 23.7  | **23.7**  | −0.8% | 4.36× |
| bs8  | 33.081  | 143.4 | 137.6 | **135.3** | **−5.6%** | 4.09× |
| bs16 | 67.225  | 263.5 | 252.6 | **250.4** | **−5.0%** | 3.72× |
| bs32 | 139.150 | 495.0 | 480.1 | **474.3** | **−4.2%** | 3.41× |

STS-B: 0.8134 (before) → 0.8164 (Q bfp8) → **0.8190** (Q + K/V bfp8). bs1 gains
little because the bs1 path never had the Typecast (bf16 Q went to SDPA directly);
its 0.2 ms is SDPA reading half the Q/K/V bytes.

**Follow-on, same day — `QWEN_QKV_OUT_BFP8=1` (default on):** the QKV projection
itself now writes bfp8. Upstream pins its output to bf16 only because the stock
rotary op asserted bf16; with norm + RoPE fused, the head-split op is the
projection's only reader and unpacks bfp8 directly, so the projection writes half
the bytes and the fused op reads half (201 MB → 100 MB per layer at bs32). The
subclass wraps `minimal_matmul`/`ttnn.linear` for the call whose weight is
`self.wqkv` and switches `dtype` to bfp8; Q/K/V are then already quantised, so the
fused op emits all three in bfp8 (single output CB).

| batch | H200 | Q+K/V bfp8 | **+ QKV out bfp8 (shipped)** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 23.7  | **23.7**  | 0.0%  | 4.36× |
| bs8  | 33.081  | 135.3 | **127.5** | **−5.8%** | 3.85× |
| bs16 | 67.225  | 250.4 | **240.9** | **−3.8%** | 3.58× |
| bs32 | 139.150 | 474.3 | **456.7** | **−3.7%** | 3.28× |

STS-B 0.8190 → **0.8161** (still above the 0.8134 this day started at). bs1 is
unchanged: its QKV output is 6 MB, so the bytes saved are ~15 µs per layer against
a launch-bound op sequence.

### Fused head-split + Q/K RMSNorm + RoPE — landed (2026-09-23)

The same `fused_qkv_heads_norm` op now also applies RoPE to Q and K
(`QWEN_FUSED_ROTARY=1`, default on). The model's prefill rotary is a single
32×32 tile-local rotation `T` applied to every tile, so per head the kernel
does `rot = x @ T`, `out = x·cos + rot·sin` right after the gamma phase, with
the unit's cos/sin tiles read once per seq-tile and `T` resident. Five ops per
layer (heads, q_norm, k_norm, rotary×2) are now one. Standalone vs the device
`rotary_embedding_llama` reference: PCC q 0.99997 / k 1.00000; B=8 traced
670.1 → 334.5 µs (−50.1%). Upstream `forward_prefill` deallocates the
"pre-rotary" tensors after rotary; with RoPE fused those are the tensors SDPA
reads, so the wrapper skips exactly one deallocation of each (later frees still
happen).

| batch | H200 | before | **after** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 25.0  | **23.9**  | **−4.4%** | 4.40× |
| bs8  | 33.081  | 144.7 | **143.4** | −0.9% | 4.33× |
| bs16 | 67.225  | 276.8 | **263.5** | **−4.8%** | 3.92× |
| bs32 | 139.150 | 519.2 | **495.0** | **−4.7%** | 3.56× |

STS-B Spearman **0.8134** (0.8135 before). Next in the same op: emit Q in bfp8
(SDPA validates operand dtypes independently, so Q bfp8 with K/V bf16 is what
runs today after a separate cast) to remove the per-layer Typecast.

### Fused head-split + Q/K RMSNorm — landed (2026-09-23)

`tt/custom_ops/fused_qkv_heads_norm/`: a model-local `ttnn.generic_op` with a
compute kernel that splits the fused QKV activation into heads **and** applies
the per-head RMSNorm to Q and K in the same pass (V copied through). It replaces
three ops per layer — `nlp_create_qkv_heads` + `q_norm` + `k_norm`, 41 + 24 + 6 ms
of kernel time per iteration at bs32, each a DRAM-bound pass over the same
tensors. Constants (row-replicated gamma tiles, a `1/head_dim` reduce scaler, eps)
are built once per layer from the checkpoint. Per head the kernel does
x² → row-reduce → +eps → rsqrt → bcast-column scale → gamma, ≤4 DST tiles per
phase so fp32 accumulation fits. Standalone: PCC 1.00000 (bf16) / 0.99900 (bfp8)
vs torch; B=8 traced 275.1 → 193.8 µs (−29.6%). Default on
(`QWEN_FUSED_HEADS_NORM=0` reverts to the separate ops).

| batch | H200 | before | **after** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 25.2  | **25.0**  | −0.8% | 4.60× |
| bs8  | 33.081  | 155.6 | **144.7** | **−7.0%** | 4.37× |
| bs16 | 67.225  | 288.7 | **276.8** | **−4.1%** | 4.12× |
| bs32 | 139.150 | 543.5 | **519.2** | **−4.5%** | 3.73× |

STS-B Spearman 0.8125 → **0.8135**. Next steps in the same op: rotary (the
kernel applies a single 32×32 tile-local rotation, `rotated = x @ T`, then
`x·cos + rotated·sin`; −2 ops/layer, 27 ms at bs32) and emitting Q in bfp8
(removes the per-layer Typecast, 15 ms at bs32).

### Measurement fix: the extended trace is now the timed path (2026-09-23)

The demo looked up the Generator's prefill trace with a stale 3-part key
(`f"{seq_len}_0_{batch_size}"`; the Generator keys it 4-part with an `_sp0/_sp1`
suffix), so `use_direct_trace` was always False and every timed iteration fell
back to the Generator path: eager post-processing ops (slice + norm + to_layout)
dispatched outside any trace, four H2D copies and a blocking readback per call.
`serve.py` calls `execute_trace` directly and never paid this; only the demo's
numbers were inflated. The demos now default to `--full-pipeline` (forward +
pooling + I/O in one traced replay — the mode the README already described as
the optimised path; `--no-full-pipeline` times the bare forward replay), the
lookup matches on the 3-part prefix, and the printed mode label reports the
branch actually taken.

| batch | H200 | before (Generator fallback) | **now (extended trace)** | Δ | × H200 |
|---|---|---|---|---|---|
| bs1  | 5.437   | 25.9  | **25.2**  | −2.7% | 4.63× |
| bs8  | 33.081  | 156.4 | **155.6** | −0.5% | 4.70× |
| bs16 | 67.225  | 290.9 | **288.7** | −0.8% | 4.29× |
| bs32 | 139.150 | 557.8 | **543.5** | −2.6% | 3.91× |

Per-iteration host cost in this mode at bs1 (`QWEN_ITER_TIMING=1`): h2d 0.08 +
trace issue 0.02 + readback enqueue 0.04 + to_torch 0.16 = **0.3 ms**; the
remaining 25.05 ms is the device. bs1 is device-bound end to end.

### Correction: the loaded chip runs at ≈1.1 GHz, so profiler µs are ~20% optimistic (2026-09-23)

`tt-smi -s` sampled every 4 s during a 150-iteration bs32 run: the loaded P150 sits at
**1087–1143 MHz, 166–170 W, 66 °C** (idle chips 800 MHz / 37 W). The profiler converts
device cycles at the nominal 1.35 GHz, which is why the eager bs32 pass shows Σkernel
384.5 ms with no inter-op gaps while the same forward takes 446–451 ms (eager and trace
replay alike: 451.3 vs 451.5 ms). Every per-op µs and roofline % quoted from the
profiler in this README is therefore ~20% low for sustained batched runs: compute-bound
ops scale with AICLK, DRAM-bound ones do not. The chip is power-limited, so lowering
power per FLOP (fewer active cores on DRAM-bound ops, narrower operands) buys clock as
well as bytes. E2E numbers are unaffected. Details: `perf_csv/NEGATIVE_RESULTS.md` §30.

### Correction: use DEVICE KERNEL DURATION, not FW DURATION, for op shares (2026-09-23)

`DEVICE FW DURATION` starts when a core receives the launch, which on a traced
run is *before* the previous op has finished — it includes the wait for GO. Its
per-iteration sum exceeds wall time by 46% at bs1 (32.7 vs 25.9 ms) and 13% at
bs32. `DEVICE KERNEL DURATION` is the op's own work. Re-ranked on kernel time:

| op | bs1 kernel ms (% of 21.0) | bs32 kernel ms (% of 497) |
|---|---|---|
| Matmuls QKV+WO+FF13+FF2 | **13.0 (62%)** — ~42% of device peak on 64 cores | 243 (49%) — 80–89% of peak |
| SDPA | 2.2 (10.6%) | 39.9 (8.0%) |
| BinaryNg (2 adds + SwiGLU mul) | 2.0 (9.6%) — *not* 30% | 78.5 (15.8%) — at DRAM roofline |
| LayerNorm ×3.9 | 1.2 (5.7%) | 51.7 (10.4%) — q_norm alone 24 ms |
| Rotary ×2 | 1.4 (6.6%) | 27.2 (5.5%) |
| GenericOp head-split+concat | 0.9 (4.2%) | 41.3 (8.3%) — at DRAM roofline |
| inter-op gaps (e2e − kernel sum) | **4.9 (19% of e2e)** | 60.6 (11% of e2e) |

The bs1 matmuls are at ~79% of the peak of the 64 cores the legacy 2D path can
use (Mt=16 requires gy | 16); the loss is the grid cap, not the kernel.

### Tested and rejected, round 2 (2026-09-23)

| experiment | result | why |
|---|---|---|
| `minimal_matmul` at bs1 on 120 cores, 1×8 subblocks (`QWEN_FORCE_MINIMAL_MM=1`) | 45.6 ms vs 25.9; 2×4: 45.6; + fused SwiGLU: 55.6 | still far slower than legacy 2D at M=512 |
| SiLU moved into FF1's matmul epilogue at bs32 (`QWEN_SILU_IN_FF1=1`) | 569.8 vs 557.8 (+2.2%) | SFPU epilogue serialises with a matmul already at 80% of peak |
| head-split kernels: 1 barrier per Q/K/V unit instead of 3 | bit-exact; bs1 60.5→69.2 µs, B=8 137.4→135.9 µs | op is at DRAM roofline (53 MB in 136 µs = 390 GB/s); only fusion removes the pass |
| standalone eager microbenchmarks of small ops | invalid | eager timing is host-dispatch-bound (I2S 85 µs eager vs 3.4 µs device); use trace replay |

Knobs retained default-off as probes: `QWEN_FORCE_MINIMAL_MM`, `QWEN_FUSE_SWIGLU_BS1`,
`QWEN_SILU_IN_FF1`.

**Operational note — device resets on this host.** Use **`tt-smi -r` only**.
Never run `tt-smi -glx_reset`: this is a shared 32-chip Galaxy and `-glx_reset`
issues an IPMI reset of the whole tray, disrupting every chip on the box, not
just the one in use. tt-smi's own output suggests `-glx_reset` as a fallback —
ignore that suggestion on this host.

Do not `kill -9` a run that is mid-device-operation; stop runs with SIGTERM and
let them close the device. A `kill -9` here left the board unable to initialise
firmware ("Device 0 init: failed to initialize FW"). Recovery notes:

- `tt-smi -r` restores PCIe and device enumeration (`tt-smi -ls` healthy again).
- Stale `/dev/shm` state (`sm_segment.*`, `tt_device_*_memory`,
  `TT_UMD_LOCK.*`, including `CHIP_IN_USE_<n>_PCIe`) is left behind by killed
  runs and can block later runs on a lock. Safe to delete only once no
  tt-metal process is running under any user.
- If after `-r` `tt-smi -ls` is healthy but every ttnn device open fails with
  `IndexError: unordered_map::at` (thrown after "Starting devices in cluster
  completed", before any tt-metal device-init log), the board is fine — the
  failure is in tt-metal's cluster construction over **all 32 chips**. Restrict
  to the chip you use: `TT_VISIBLE_DEVICES=0` opens it at the expected 12x10
  grid. The demos now set this by default. Do not read tt-smi's
  `HARVESTING_STATE 0x0` as evidence of a bad re-init; it is not the tensix
  harvesting tt-metal uses (chip 0 reports 12x10 with it set to 0x0).

**Second lesson:** benchmark the op in the *rank and batching* the model uses.
A `[1,1,M,K]` sweep of the MLP matmul showed 34% -> 57% "efficiency scaling"
with M that vanished entirely in the real `[1,Z,M,K]` form, because the 2-D form
starves the 120-core grid while the 4-D form does not.

All env vars use `os.environ.setdefault`, so any single knob can be overridden
from the shell for A/B comparisons, e.g.:

```bash
# Disable the wide matmul grid for a baseline comparison at bs=32
QWEN_MM_GRID= python models/demos/blackhole/pplx_embed_4b/demo/demo_bs32_isl512.py
```

---

### bs1: 12×8 matmul grids, coalesced weight reads, SDPA q256 — landed (2026-09-24)

bs1 went **21.5 → 17.7 ms** (chip 4, same-chip A/B; 23.3 at the start of the day). STS-B 0.8161
unchanged. All three changes are gated to `batch_size == 1` in `apply_workload_env`; bs8/16/32 are
untouched (123.0 / 228.7 / 438.0 ms after the change, within noise of 123.4 / 228.3 / 438.1).

**1. The four bs1 matmuls run on 12×8 = 96 cores (−3.7 ms).** The bs1 path uses the legacy
`MatmulMultiCoreReuseMultiCastProgramConfig` with DRAM width-sharded bfp4 weights in the 8 DRAM banks,
and every grid wider than 8 columns returned inf. The cause was in the 2D program factory
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp`): the per-column bank walk sets
`worker_core_stride = per_core_N_storage - storage_core_stride`, i.e. a column takes a whole bank
stripe even when its `per_core_N` is smaller, so the L1 block is overrun. Capping it at `per_core_N`
makes every wide grid bit-identical to 8×8 (PCC vs torch unchanged to 5 digits). Standalone, traced,
M=512, chip 3:

| projection | 8×8 (was) | 12×8 (now) | Δ |
|---|---|---|---|
| QKV 512×2560×6144 | 67.5 µs (1×4) | **49.8** (per_core_N 16, 1×4) | −26% |
| WO 512×4096×2560 | 48.7 (1×2) | **40.1** (per_core_N 7, 2×1) | −18% |
| FF1 / FF3 512×2560×9728 | 110.9 (1×2) | **79.5** each (per_core_N 26, 2×2) | −28% |
| FF2 512×9728×2560 | 101.8 (1×2) | **78.8** (per_core_N 7, 2×1) | −23% |

`per_core_N` 7 needs an explicit 2×1 subblock (the derived 1×1 is slower than 8×8). 12×10 is within
1 µs of 12×8 (M pads 512 → 640 over 10 rows); 10×8 / 11×8 sit between. Knobs: `QWEN_QKV_GRID_X=12`,
`QWEN_LEGACY_GRID_{FF13,FF2,WO}=12,8`, `QWEN_LEGACY_TIGHT_PER_CORE_N=1` (callers pass a per_core_N
sized to the 8 shards; this sizes it to the grid), `QWEN_LEGACY_SUBBLOCK_K<k>_N<n>=h,w` per shape.
Opt out with `QWEN_LEGACY_BS1_WIDE=0` (falls back to the tuned 8×8 blocks, e2e 21.4).

**2. DRAM-sharded weight reader: one NoC read per block-row segment (−0.8 ms).**
`reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (`IN1_DRAM_WIDTH_SHARDED`) issued one
576-byte read per bfp4 tile — 240 requests per QKV block. The tiles of a block row inside one bank
are contiguous in DRAM and in the L1 block, so each row segment is now a single multi-burst read
(13.8 KB QKV, 21.9 KB FF1). Same bytes to the same addresses, numerics identical. Standalone
QKV 70.3 → 67.8, WO 51.3 → 49.0, FF1 114.0 → 110.5, FF2 111.6 → 102.8 µs; e2e 22.3 → 21.5 (kernel
file swapped between runs on chip 4). Only the bs1 path reaches this kernel here.

**3. SDPA q_chunk 256 at bs1 (−0.9 ms).** With q512 the bs1 SDPA has 32 work units for 64 cores.
Standalone at the model's exact config (LoFi, `fp32_dest_acc_en=False` = the streaming kernel, exp
approx, bfp8 Q/K/V in L1): q512/k256 79.1 → **q256/k256 55.0 µs**; q256/k512 71.5, q128/k128 72.2,
12×10 grids 67.6, fp32 acc on (legacy kernel) 109.3. e2e 23.3 → 22.4. Batched sizes keep 12×10/k512.
Opt out: `QWEN_SDPA_BS1_Q256=0`.

**Also in this landing (≈0 e2e, kept):** per-shape in0_block_w / subblock overrides for the legacy
matmuls (`QWEN_LEGACY_IN0_BW_K<k>_N<n>`, `QWEN_LEGACY_SUBBLOCK_K<k>_N<n>`; guarded to skip shapes
they do not divide, e.g. shorter warm-up seq_lens). On 8×8 the sweep gave FF1 −3.7%, WO −2.0%,
FF2 −1.9%, QKV −1.2% standalone but only −0.1 ms e2e.

**What bounds the bs1 matmuls now.** At 8×8, HiFi2 costs +71% and interleaved weights +38% (even
from L1: it is the tile-granular request pattern, not DRAM bandwidth), so the kernel was ≈70% FPU
time and ≈30% weight delivery; 96 cores take it to ≈60% of the 96-core LoFi peak. The 1D-multicast
kernel (`mcast_in0`, every core reading its own N-slice) is +16% and flat across grids: its single
in0 sender caps at ≈17 GB/s. bs8/16 op outputs in L1 (the BGE-M3 B8/B16 win) still fail trace
capture here by 42 KB/core, with or without the FF intermediates.

**bs1 host path.** `QWEN_ITER_TIMING=1`: h2d 0.04 ms, trace issue 0.02, readback enqueue 0.02, device
sync = the rest, to_torch 0.1 — the bs1 number is device time; the 3.4 ms host bubble of an earlier
profile is gone since the extended trace landed.

### bs>1: what transfers from the BGE-M3 P150 branch, and three probes (2026-09-24)

Reviewed `gtobarTT/bge_m3_p150_optimizations` (25 commits; B1 4.31 → 3.73 ms, B8 23.7 → 11.0,
B16 39.0 → 20.5, B32 65.7 → 42 ms on a 13×10 p150a — Galaxy chips expose 12×10, as here). The
batched wins there are: bf8 Q/K/V into the streaming SDPA kernel at LoFi with q256/k512 (already
ours), the `no_padding` mask skip (no mask here), and moving the QKV output, the Q/K/V heads, the
matmul outputs and the LayerNorm output into L1. Each L1 placement was tried here with the existing
`TT_PREFILL_<op>_L1` knobs plus two new ones, same-chip A/B, per-chip logs:

| placement | bs8 | bs16 |
|---|---|---|
| QKV out + heads + SDPA out + concat in L1 (also with SDPA k-chunk 256) | trace-capture clash, 42 KB/core short (28 KB with k256) | clash |
| WO / FF2 outputs in L1 | 122.8 → 122.3 (noise) | — |
| LayerNorm output in L1 (`TT_PREFILL_LN_L1=1`, new) | 121.5 → 123.2 (+1.4%) | 227.7 → 227.9 |
| LayerNorm fp32 accumulation off (`QWEN_NORM_FP32_ACC=0`, new) | 123.4 → 122.5 | inside the bs16 band; bs32 434.3 → 439.6; STS-B 0.8161 → 0.8159 |

Nothing lands: BGE-M3 has dim 1024 and 16 heads of 64, so its per-core L1 footprint at B8 is 2.5×
smaller than ours, and its LayerNorm feeds a 2D-multicast matmul that benefits from an L1 in0;
ours feeds `minimal_matmul`, which streams in0 from DRAM at full rate. The legacy 2D kernel itself
(now with the coalesced reader, §37 of the negatives file) was also re-measured against
`minimal_matmul` at bs8/bs16 shapes and loses by 3–60%; conversely `minimal_matmul` at bs1 loses
by 53–65% to the legacy 12×8 kernel. Both knobs stay in the tree, default off, for future probes.

**bs16 measurement note.** The shipped bs16 configuration measured 216.8 / 228.7 / 229.1 / 229.4 /
227.7 / 230.5 ms across today's runs and drifts 217 → 235 ms inside one 10-iteration run as the
chip warms; a single bs16 A/B cannot resolve less than ≈ 5%.

**Fused heads op, 2-way Q split (`QWEN_HEADSPLIT_Q_SPLIT=2`, default 1).** A unit becomes half the
group's Q heads plus K *or* V (bs1: 256 units of 12 tiles instead of 128 of 24). Output is
bit-identical; standalone 48.4 → 44.5 µs at bs1 (−8%), 185.0 → 183.3 at bs8; e2e bs1 17.7 → 17.6
(noise). Opt-in.

### bs>1 round: row-split add+RMSNorm on every core, SDPA 12×8 at bs8, interleaved weights at bs32 (2026-09-24)

**Multi-wave row-split fused add + RMSNorm (`fused_add_rmsnorm_split`, `tt/custom_ops/fused_add_rmsnorm/op_split.py`).**
The row-granular fused kernel (landed for bs16+) assigns one tile-row per core, so 128 / 256 / 512
tile-rows on 120 cores run in 2 / 3 / 5 waves with most cores idle in the last one — at bs8 it was
only −3% vs the stock two ops and was never enabled there. The row-split kernel (built for bs1,
where it lost to sync overhead) now takes any number of units: unit u = (row, slice k) runs on core
u mod C in wave u div C with C a multiple of R, so every core keeps one slice index and the R cores
of a row form a fixed group that walks rows together; each wave's R partial mean-squares land in
their own CB 8 slot and the semaphore only counts up (`noc_semaphore_wait_min`). Standalone, DRAM
bfp8, PCC vs stock ≥ 0.9996:

| M | stock add + rms_norm | fused, 1 row/core | R=2 | R=4 | R=5 | R=10 |
|---|---|---|---|---|---|---|
| 4096 (bs8) | 184 µs | 178 | 165 | 144 | **141** | 136 |
| 8192 (bs16) | 320 | 299 | 281 | 250 | **242** | — |
| 16384 (bs32) | 597 | 530 | 491 | **459** | 463 | — |

e2e: **bs8 123.1 → 119.9 (−2.6%, chip 7)** with `QWEN_FUSED_ADD_NORM_MIN_ROWS=4096` and
`QWEN_FUSED_ADD_NORM_R=5`; **bs32 449.4 → 443.2 (−1.4%, chip 10)** with R=4. Per-call PCC vs the stock
ops in the model at bs8 (`QWEN_FUSED_ADD_NORM_VERIFY=1`): sum ≥ 0.99988, norm ≥ 0.99980. R=20 is
exchange-bound (+24%).

**SDPA 12×8 q512/k512 at bs8 (−0.9%).** The batched model runs q512/k512 (the q256 override never
reached these shapes). Standalone bs8 12×10 272 → 12×8 234 µs: 256 work units take 3 waves on 96
cores as on 120, with less DRAM contention. e2e bs8 122.8 → 121.7 (chip 7); bs32 is neutral
(435.1 → 436.4) and keeps 12×10. Opt out: `QWEN_SDPA_BS8_12X8=0`.

**Interleaved bfp4 weights for QKV / WO / W1 / W3 at bs32 (−1.1%).** `minimal_matmul` streams
interleaved weights 2–3% faster than the width-sharded layout from M=8192 up (QKV −3.4%, WO −2.0%,
FF1 −1.6% at M=16384; FF2 prefers sharded, +1.4%); at M=4096 the sharded layout wins (+1.5…+2.3%),
and the bs1 legacy kernel needs it. Knob `QWEN_WEIGHT_INTERLEAVED_K<k>_N<n>=1` in
`create_dram_sharded_mem_config`; e2e bs32 441.9 → 436.9 (chip 6).

**bs16, and what "best of 10" measures.** tt-smi sampled during bs16 runs shows the two recurring
readings are the clock: 216.7 ms at AICLK 1281–1350 MHz, 232–243 ms at 1112–1162 MHz, the board's power
manager pulling the clock down ≈ 0.6 s into a sustained load and deeper as the chip warms (no power or
clock setting was changed; `AICLK_LIMIT_MAX` 1350). "Best of 10" is therefore the cold-chip number.
Sustained (median of iterations 5–9) with these defaults: bs1 17.8, bs8 123.8, bs16 232, bs32 452.6 ms
(previous defaults 18.0 / 126.7 / 235 / 452.3): the bs32 changes only help the cold iteration, bs8 holds
−2.3% sustained, bs16 ≈ −1.5%. The alternating multi-launch A/Bs for bs16 are in the negatives file §46.

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
