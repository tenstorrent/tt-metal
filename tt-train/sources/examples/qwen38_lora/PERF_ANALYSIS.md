# Qwen3.8-27B LoRA (TP=4, 1x4 Blackhole) — performance analysis

Setup for every number below unless stated otherwise: batch 1, seq 1024, TP=4,
`--delta-rule fused --conv fused --recompute-deltanet`, LoRA on all linears,
peak 594 TFLOP/s across the 4 chips.

## MFU history

| Configuration | ms/step | MFU |
|---|---|---|
| Baseline (after skipping grads of frozen operands in linear/swiglu/binary) | 1828.8 | 9.89% |
| Composite conv (control rerun) | 1819.7 | 9.94% |
| Fused causal conv1d + SiLU fwd, fused conv backward | 1450.8 | 12.47% |
| + flat/GQA layout in `gated_delta_net_backward` (no head-axis reshapes; L2 norm and GVA sum as 0/1 matmuls) | 1212.6 | 14.92% |
| **+ grouped gated RMSNorm fw/bw kernel (`gated_rmsnorm_*`), two-phase `rmsnorm_bw` on all cores, no dgamma for frozen gamma** | **1157.8** | **15.62%** |

Logs: `/localdev/umales/scratch/mfu_fused_conv.log`, `mfu_composite_conv.log`, `mfu_a3_flat.log`,
`mfu_a3_c_d.log`.

## Where the time goes (Tracy, 4 layers = 3 DeltaNet + 1 attention)

### Current profile (2026-10-06, after flat GQA layout + gated_rmsnorm + two-phase rmsnorm_bw)

Profile: `/localdev/umales/scratch/tracy_4layers_a3cd/reports/2026_10_06_18_32_07/ops_perf_results_2026_10_06_18_32_07.csv`
(analysis: `scratch/tracy_a3cd_analysis.txt`, op sequence: `scratch/tracy_a3cd_seq.txt`,
tt-perf-report: `scratch/tracy_a3cd_perf_report.txt`, `scratch/tracy_a3cd_summary.{csv,png}`).
Device time per 4-layer step 97.7 ms with 817 ops (was 135.3 ms / 1238 ops in the 2026-10-01
frozen profile below). Per section:

| Section | ops | µs | Full model |
|---|---|---|---|
| DeltaNet layer fwd | 64 | 4 892 | ×48 |
| DeltaNet layer bwd (incl. mixer recompute) | 130 | 10 597 | ×48 |
| Attention layer fwd | 57 | 5 970 | ×16 |
| Attention layer bwd | 82 | 9 050 | ×16 |
| Embedding: `untilize` of the 248320×5120 table + lookup | 2 | 12 903 | ×1 |
| Final norm + LM head fwd (`[1024,5120]×[62080,5120]`) | 2 | 7 434 | ×1 |
| Cross-entropy fwd+bwd on `[1024,62080]` | 30 | 7 662 | ×1 |
| LM head dX + final norm bw | 5 | 6 986 | ×1 |
| AdamW (26 LoRA tensors) | 26 | 241 | ×1 |

Projected full-model device time: 48×15.5 + 16×15.0 + 36.2 = **1020 ms/step** (DeltaNet 73%,
attention 24%, fixed 3.5%) vs 1157.8 ms measured wall time, i.e. ~12% of the step is host /
dispatch gaps.

Inside a DeltaNet layer (fwd + bwd = 15.5 ms, ×48 = 744 ms):

| Item | per layer | full model | Note |
|---|---|---|---|
| MLP matmuls (`4352` gate/up/down, fwd + 3 bwd; 6 × ~490 µs) + swiglu fwd/bwd elementwise | 3.3 ms | 160 ms | the useful FLOPs; 88 cores |
| `gated_delta_net_backward` (GenericOp) | 1.87 ms | 90 ms | 110 cores; was 2.6 ms before the flat layout |
| in_proj / z_proj / out_proj base matmuls (fwd, recompute, dX; 9 × 190–300 µs) | 2.0 ms | 98 ms | frozen base weights, no dW |
| LoRA matmuls (30 per layer; e.g. `[1024,5120]×[32,5120]` 78 µs on 32 cores, `[1024,32]×[32,5120]` 92–100 µs) | 1.97 ms | 94 ms | each ~1–2% utilised; as expensive as the base projections |
| `ReduceScatter` + `AllGather` on `[1024,5120]` (8 pairs) | 1.34 ms | 64 ms | 95–110 µs + 65–75 µs per pair; RS on 24 cores |
| `BinaryNg` on `[1024,2560]` (17/layer: residual adds, LoRA scale/add, grad accumulation) | 1.0 ms | 48 ms | 60–70 µs each |
| `ChunkGdnPrep` + `ChunkGdnScan` (fwd, run twice with recompute) | 0.74 ms | 36 ms | |
| Block `RMSNormForward` (158 µs, 32 cores) ×3 + `rmsnorm_bw` partial+apply (207 µs) ×2 | 0.9 ms | 43 ms | fwd still limited to 32 cores |
| `QkvCausalConv1dSilu` (150 µs ×2) + `DepthwiseConv1dK4` bw (188+138 µs) | 0.63 ms | 30 ms | |
| GDN output plumbing: untilize/permute/reshape/tilize/typecast `[12,16,64,128]→[1024,1536]` (×2) | 0.42 ms | 20 ms | last remaining head-layout cost in DeltaNet |
| `GatedRmsNorm` fwd ×2 + bwd | 0.29 ms | 14 ms | |

Inside the attention layer (fwd + bwd = 15.0 ms, ×16 = 240 ms): MLP ~3.9 ms, `SDPA` fwd 388 µs +
bwd 303+820 µs (KV bwd on 32 cores), and **~2.9 ms of head plumbing**: `ReshapeView` on
`[1,1024,32[6],256]` style shapes (662 + 366 + 2×105 µs fwd, 881 + 337 + 2×81 µs bwd), `Concat`
`[1,1024,32,256]` (175 + 178 µs), `Transpose`/`Slice` (~500 µs). This is item 5 below and is now
the largest remaining layout cost (~46 ms/step).

Fixed costs (36 ms/step, 3.5%): the embedding `untilize` alone is 12.8 ms — `ops/embedding_op.cpp`
untilizes the whole 2.5 GB table on every forward although the weight is frozen.

### Previous profile (2026-10-01)

Profile: `/localdev/umales/scratch/tracy_4layers_frozen/reports/2026_10_01_14_38_24/ops_perf_results_2026_10_01_14_38_24.csv`.
It predates the fused conv, so the composite-conv ops in it (shift/untilize/concat/tilize on
`[1024,2560]`) are gone now; everything else listed below is unaffected. Per-layer costs are scaled
to the full model (48 DeltaNet + 16 attention layers, ~1450 ms/step).

### Ranking (excluding gated-norm / attention-gate elementwise work)

| # | Item | Per layer | Full model / step | Share |
|---|---|---|---|---|
| 1 | Head-layout plumbing around the delta rule | ~5.8 ms | ~280 ms | ~19% |
| 2 | `gated_delta_net_backward` kernel (GenericOp) | 2.6 ms | ~125 ms | ~8.6% |
| 3 | RMSNorm backward of the `[T,5120]` block norms | 2 × 685 µs | ~88 ms | ~6% |
| 4 | All-reduces of `[1024,5120]` (8 per DeltaNet layer, 186 µs each) | ~1.5 ms | ~90 ms (≈45 recoverable) | ~6% |
| 5 | Attention head split/merge plumbing | ~3.1 ms | ~50 ms | ~3.5% |
| 6 | Untilizing the 248320×5120 embedding table every step | — | 12.8 ms | 0.9% |

Items 1 and 5 together, "preparing heads", are ~330 ms/step (~23%).

### Root cause of the plumbing cost: tile padding on dim -2

TILE layout stores 32×32 tiles over the last two dims; dim -2 is padded to 32.

```
[1, 1, T, 512]  -> last two dims (T, 512): no padding
[1, T, 4, 128]  -> last two dims (4, 128): 4 rows padded to 32 = 8x memory
```

So any reshape that puts heads on dim -2 (`[B,T,H,D]`) is a full data-movement kernel
(`ReshapeView`), and the result is 2.7x (12 heads) to 32x (1 head) larger. Every following op
(l2 norm, concat, slice, transpose) streams the padding too, and each op gets a mirrored op in
backward. Recompute runs the DeltaNet forward plumbing twice.

### 1. DeltaNet head prep — `ttml/models/qwen38/gated_deltanet.py::_fused_mixer`

Per chip: q, k `[1,1,T,512]` (4 key heads × 128), v `[1,1,T,1536]` (12 value heads × 128).

```
q [1,1,T,512] -reshape-> [1,T,4,128]   (4->32 pad)                     125 µs
              -l2_norm-> 6 ops on padded tensor                        192 µs
              -concat x3-> [1,T,4,384]                                  139 µs   (repeat_interleave_token_major)
              -reshape-> [1,T,12,128]  (12->32 pad)                     132 µs
k             same                                                    ~590 µs
v [1,1,T,1536]-reshape-> [1,T,12,128]                                  310 µs
-- inside ttnn.transformer.chunk_gated_delta_rule (chunk_gated_delta_rule.cpp) --
q,k,v         -permute-> [12,T,128]                                3 × 77 µs
              ChunkGdnPrep + ChunkGdnScan  (the actual math)          ~375 µs
out [12,T,128]-untilize, permute, reshape, tilize, typecast->
              [1,1,12T,128]                                            ~150 µs
```

Forward plumbing ≈ 1.9 ms, run twice (forward + recompute). Backward mirror ≈ 2.1 ms:
- dv reshape back to `[1,1,T,1536]` (360 µs);
- dq/dk: undo the GVA repeat with a reshape to `[1,T,4,384]`, 3 slices and 2 adds; then the
  l2-norm backward (6 ops on padded data); then a reshape to `[1,1,T,512]`.

Total ≈ 5.8 ms/layer, against 0.375 ms for the forward kernels.

**Unused fast path:** `chunk_gated_delta_rule` already accepts *rank-3 flat* inputs ("OPT-A": q, k
`[B,T,H*K]`, v `[B,T,HV*V]`). It then skips the head-split permutes and maps value head `hv` to
key head `hv // G` in the reader, so no GVA repeat is needed. At `chunk_size == 32` it also does
the q/k L2 norm in-kernel (OPT-B). Our config uses `delta_chunk_size = 64`.
`gated_delta_net_backward` still needs L2-normalised, GVA-expanded `[B,T,HV,K]` inputs, but using
flat mode in the forward means the plumbing runs once (in backward) instead of three times.

**Full fix:** a fused "GDN prep" kernel. It reads the conv's q|k|v (each head is exactly 4 tiles
wide, so no padded intermediate) and writes L2-normed, head-major q/k/v. Pair it with a
mirror-backward kernel (sum the GVA copies, L2-norm backward, write `[T,2560]` for the conv
backward), or teach `gated_delta_net_backward` the flat/GQA layout.

### 2. Delta-rule backward kernel

`gated_delta_net_backward` (GenericOp, 110 cores) takes 2.6 ms vs 0.375 ms for the forward
prep + scan. That is 7×, where 2–3× is typical. The cost is inside the kernel, not in how work
is distributed across cores.

### 3. RMSNorm backward on `[T,5120]`

`ttml/ops/rmsnorm_op.cpp` → `metal/ops/rmsnorm_bw`:
- **Wasted gamma gradient:** it always writes a `[1024,5120]` dgamma-components tensor and
  reduces it (FastReduceNC ×2 + Reduce ≈ 241 µs), even though gamma is frozen under LoRA. This is
  the same frozen-grad skip already done for linear/swiglu/binary.
- **Low core use:** the kernel splits work by tile-rows; T=1024 gives only 32 rows, so it uses
  32 of 110 cores (444 µs vs 158 µs for the forward).

Skipping dgamma alone should save ~45–55 ms/step.

### 4. All-reduces / input projections

Every `ColumnParallelLinear` / `LoraColumnParallelLinear` calls
`ttml.ops.distributed.broadcast(x)`. That is an identity in forward, but its backward is an
all-reduce of the full `[1024,5120]` input gradient.

- DeltaNet calls 4 projections on the same input (`in_proj_qkv`, `in_proj_z`, `in_proj_b`,
  `in_proj_a`), so it pays 4 all-reduces where 1 would do. `b` and `a` emit only 12 columns per
  chip but still pay a full all-reduce each.
- Attention q/k/v: 3 → 1.
- **Stage 1 (small change):** one broadcast per layer, with the projections skipping theirs. This
  is exact, since the sum of all-reduces equals the all-reduce of the sum. ~33 ms/step.
- **Stage 2 (`mla_kv_assemble`-style):** concatenate base weights at load time (pad b/a to 32
  columns). That gives 1 forward matmul and 1 dx matmul instead of 4, plus 3 fewer `[1024,5120]`
  grad accumulations; the LoRA A matrices for qkv and z can share one matmul. ~15–25 ms more.
- Moving `out_proj` out of the recomputed region avoids recomputing its matmul + all-reduce:
  ~0.4 ms/layer.

### 5. Attention head plumbing — `ttml/models/qwen38/attention.py::forward`

Per chip: 6 query heads × 256 and 1 KV head × 256. `q_proj` emits per-head interleaved
`[q0|g0|...|q5|g5]` (3072 wide).

```
qg [1,1,T,3072]-reshape-> [1,T,6,512]   (6->32 pad)                    661 µs
               -slice-> query, gate [1,T,6,256]                    88 + 80 µs
gate           -reshape-> [1,1,T,1536]                                 374 µs
query          -transpose-> [1,6,T,256]                                 83 µs
k [1,1,T,256]  -reshape-> [1,T,1,256]   (1->32 pad, 32x)               105 µs
               -transpose-> [1,1,T,256]                                 48 µs
v              same                                                    154 µs
```

Backward:
- the backward of each slice is concat-with-zeros on padded data (175 + 178 µs), followed by an
  add (220 µs);
- reshape `[1,T,6,512]→[1,1,T,3072]` (**853 µs**);
- gate reshape (336 µs);
- q/k/v transposes and reshapes back (~430 µs).

Merging heads is already cheap: `heads_fusion` uses `nlp_concat_heads` (24 µs). For comparison,
SDPA forward + backward is 1.5 ms/layer.

**Fix:**
1. At load time, permute `q_proj` rows so the output is `[q0..q5 | g0..g5]`. Query and gate then
   become tile-aligned column slices of `[T,3072]`, and the gate is already `[T,1536]`.
2. Split heads with `nlp_create_qkv_heads` (ttml: `multi_head_utils.heads_creation` /
   `grouped_heads_creation`), whose backward is `nlp_concat_heads`.

Saves ~40 ms/step.

### 6. Embedding table untilize

`ttml/ops/embedding_op.cpp` untilizes the full 248320×5120 weight every forward (12.8 ms/step).
The table is frozen, so store it row-major once.

## Gated norm / attention gate (originally proposed fusions)

| Target | Cost today | Note |
|---|---|---|
| `silu(gate) * x` after DeltaNet RMSNorm | ~0.16 ms/layer → ~8 ms/step | fusing into the norm epilogue alone is ~0.5% |
| Reshapes around that norm: gate `[T,1536]→[12T,128]` and output back, in forward, recompute and backward | ~1.3 ms/layer → ~61 ms/step | the real cost |
| `sigmoid(gate) * attn` | ~0.1 ms/layer → ~2 ms/step | an `sdpa_fw` gate input also needs `sdpa_bw` changes (it needs the ungated O for `rowsum(dO∘O)`); not worth it alone |

**Reshape-free gated RMSNorm:** for `[T, H·128]` in tile layout, tile pages are ordered so that
reading tile-row `r = mt*12 + h` as 4 tiles `r*4 + w` is exactly head `h` of tile-row `mt`. So a
per-head RMSNorm can run directly on `[T,1536]`, either as a group-width mode in
`rmsnorm_fw`/`rmsnorm_bw` or as a zero-copy aliased view (row-wise norms don't care about row
order). Combine that with a `silu(gate)` epilogue, take the gate straight from `in_proj_z`, write
the delta-rule output as `[T, H·V]` (free while still row-major), and feed `out_proj` directly.
Expected ~65–70 ms/step.

## Other findings

- **Possible LoRA correctness issue:** `LoraColumnParallelLinear` keeps `lora_A` replicated, but
  its gradient is computed from a column-sharded `lora_B`. Each chip's dA is therefore a partial
  sum, and nothing all-reduces it, so the 4 copies of A may diverge. It doesn't affect MFU; verify
  separately.
- **Seq 16k (paused):**
  - Weights take 17.33 GB per chip.
  - Saved block inputs take 11.36 GB.
  - Block recompute peaks at about +5 GB.
  - The full model OOMs at ~33.9 GB of 34.18 GB.
  - The `_LAST_SCRATCH` pin (~2 GB) in `gated_delta_net_backward` is fixed in
    `fused_delta_rule.py`.
  - Options: host-side embedding (the untied, frozen 2.5 GB table is replicated on every chip),
    bfp8 frozen weights, or sequence parallelism.

## How to run the Tracy profiler

Tracy must be compiled in (`ENABLE_TRACY=ON`; `tracy-capture` and `tracy-csvexport` are built
with tt-metal). Use a reduced model (`--layers 4` = 3 DeltaNet + 1 attention, the same pattern
as the full model) so the device profiler buffers don't overflow.

```bash
cd /localdev/umales/tt-metal/tt-train
export TT_METAL_HOME=/localdev/umales/tt-metal   # tracy looks for build/tools/profiler/bin under $TT_METAL_HOME (defaults to cwd)
export TT_MESH_GRAPH_DESC_PATH=$PWD/configs/mgd/bh_qb_1_4_ring_ring.textproto
PYTHONUNBUFFERED=1 nohup timeout 3600 /localdev/umales/tt-metal/python_env/bin/python -m tracy \
    -r -p --op-support-count 30000 \
    -o /localdev/umales/scratch/tracy_4layers_<name> \
    sources/examples/qwen38_lora/train.py \
    --random-init --random-data --dp 1 --tp 4 --seq-len 1024 \
    --steps 4 --warmup-steps 2 --layers 4 \
    --delta-rule fused --conv fused --recompute-deltanet --profile \
    > /localdev/umales/scratch/tracy_4layers_<name>.log 2>&1 &
```

- `-r` generates the ops performance report CSV (device kernel durations per op).
- `-p` profiles only enabled Tracy zones (`-l` would instead trace every Python line, which is very slow).
- `--op-support-count` is the maximum number of ops the device profiler can record; raise it
  with more layers or steps.
- `--profile` (train.py flag) emits a `signpost("step N")` at the start of each step.
- The output CSV is in `<-o dir>/reports/<timestamp>/ops_perf_results_<timestamp>.csv`.
- For **MFU runs, do not use Tracy**. Run train.py directly and unset the profiler env vars:
  `env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT ...`.
- If `Tracy tools were not found`: `TT_METAL_HOME` is unset/wrong (see above); the binaries live
  in `tt-metal/build/tools/profiler/bin/`.

### tt-perf-report (stacked device-time chart)

`tt-perf-report` (install once: `python_env/bin/uv tool install /localdev/umales/tt-perf-report`)
gives a per-op table with advice plus a stacked PNG. Select one step with the signposts:

```bash
tt-perf-report <csv> --start-signpost "step 3" --end-signpost "step 4" --arch p150 --no-color \
    --summary-file out_summary.csv --csv out_ops.csv > out_report.txt
# -> out_summary.csv.csv (per-op totals) and out_summary.csv.png (stacked chart)
```

It merges the 4 devices (divide by 4 for per-chip numbers) and does not know the ttml custom ops
(`GenericOp`, `ChunkGdn*`, `RMSNorm*`, `GatedRmsNorm*`, `SwigluElemwiseBw`, …), which it files
under "Other"; the matmul utilisation it prints is not meaningful for the padded LoRA shapes.

### Analysing the CSV

All 4 devices are recorded; use one device. Divide sums by the number of profiled steps.

```python
import pandas as pd

csv = ".../ops_perf_results_<ts>.csv"
steps = 4
df = pd.read_csv(csv, low_memory=False)
df = df[df["DEVICE ID"] == df["DEVICE ID"].min()].reset_index(drop=True)
dur = "DEVICE KERNEL DURATION [ns]"

def shape(i):
    cols = [f"INPUT_{i}_{d}_PAD[LOGICAL]" for d in "WZYX"]
    return df[cols].astype(str).agg("x".join, axis=1)

df["s0"], df["s1"] = shape(0), shape(1)
total_us = df[dur].sum() / steps / 1e3

# Top ops by (op, input shapes), per step
g = df.groupby(["OP CODE", "s0", "s1"])[dur].agg(["count", "sum"])
g["count"] /= steps
g["sum"] /= steps * 1e3          # µs per step
g["pct"] = 100 * g["sum"] / total_us
print(g.sort_values("sum", ascending=False).head(40).to_string())

# Op-by-op sequence of one step (to attribute ops to model code)
n = len(df) // steps
for i, r in df.iloc[2 * n:3 * n].iterrows():
    print(r["OP CODE"], r["s0"], r["s1"], int(r[dur] / 1e3), "µs", int(r["CORE COUNT"]), "cores")
```

The shape strings read `padded[logical]` per dim. For example, `1024x32[12]x128` means 12 heads
padded to 32, which is the signature of the head-layout problem above. Scale per-layer costs to
the full model by ×16 for DeltaNet (48/3) and ×16 for attention (16/1). Fixed costs (embedding,
LM head, cross-entropy, optimizer) count once per step.
