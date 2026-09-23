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
