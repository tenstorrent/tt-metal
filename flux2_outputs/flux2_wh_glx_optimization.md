# Flux2 WH Galaxy Optimization Summary

## Overview

This document tracks the optimization of the Flux2 model on WH Galaxy (4x8 mesh, 32 devices).
Starting from the BH Galaxy port, we systematically profiled, swept, and tuned the pipeline to
achieve significant end-to-end throughput improvements.

**Branch**: `tvardhineni/flux2_wh-glx`
**Base**: `origin/friedrich/flux2`

### Hardware Configuration (WH Galaxy)

| Property | Value |
|---|---|
| Mesh shape | 4×8 (32 devices) |
| Topology | Ring |
| SP axis | 0 (SP factor = 4) |
| TP axis | 1 (TP factor = 8) |
| Num links | 4 |
| DRAM per device | ~1 GB |
| L1 per device | ~1464 KB |
| Compute grid | 8×9 (72 cores) |
| FSDP | **Required** (DRAM OOM without it) |

---

## Current Performance

| Resolution | Pipeline Time | Denoising Throughput | Notes |
|---|---|---|---|
| 1024×1024 | **21.87s** | **2.35 it/s** | Shipped: selective MLP-only bf8 FSDP gather (PSNR 38.3 dB) |
| 1024×1024 | 25.71s | 1.99 it/s | bf16 reference (no bf8) |

**Target was 20-24s — met at 21.87s.** The production target resolution is 1024×1024.

### Performance History (1024×1024)

| Milestone | Pipeline Time | Throughput | Delta |
|---|---|---|---|
| Initial port (baseline) | ~32.5s | ~1.54 it/s | — |
| After matmul sweep (MM, AGMM, MMRS, 1D) | ~27.5s | ~1.82 it/s | +15% |
| After Ring SDPA tuning (q=256, k=512) | ~27.38s | ~1.83 it/s | +0.6% |
| After merged spatial+prompt matmuls | 25.67s | 1.95 it/s | +6.6% |
| After swept merged-matmul configs | 25.71s | 1.99 it/s | +2% denoising |
| **After selective bf8 (MLP only, attention bf16)** | **21.87s** | 2.35 it/s | **-15% pipeline, PSNR 38.3 dB** |

### Pipeline Time Breakdown (1024×1024)

| Stage | % of total |
|---|---|
| Encoding | 0.6% |
| **Denoising** | **97.5%** |
| VAE | 1.7% |

**Denoising dominates entirely.** All future optimization must target the denoising loop.
Within a transformer block, FSDP AllGather is ~34% — the single biggest lever.

### Swept Merged-Matmul Configs (proj_mlp, 8×9 grid)

| Shape (M, K, N) | Resolution | Best (M_blk, K_blk, N_blk) | Kernel time | Status |
|---|---|---|---|---|
| (1536, 6144, 4608) | 1024 | (8, 8, 10) | 1.13 ms | applied + validated (in matmul.py) |
| (4608, 6144, 4608) | 2048 | (6, 12, 10) | 2.94 ms | applied, not needed for 1024 target |

These are stored in `grid_89_configs` in `models/tt_dit/utils/matmul.py`. Only the
1024 entry matters for the production target; the 2048 entry is harmless pre-tuned
data used only if 2048×2048 is run.

---

## Optimizations Applied

### 1. Matmul Block Size Sweep (Commit: `df08922`)

Swept optimal `(M_block, K_block, N_block)` configurations for all matmul shapes used by
the Flux2 transformer on WH Galaxy across four operation types:

- **MM** (`ttnn.experimental.minimal_matmul`) — plain matmul
- **AGMM** (`ttnn.experimental.all_gather_minimal_matmul_async`) — fused AllGather+Matmul
- **MM+RS** (`ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`) — fused Matmul+ReduceScatter
- **1D Matmul** (`ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig`) — for small-M ops

**Methodology**: For each `(M, K, N, grid)` shape, enumerate all valid `(M_block, K_block, N_block)`
combinations that fit within the WH L1 budget (~1400 KB), run each on the device with profiling,
and pick the fastest by `device_kernel_duration`.

**Sweep script**:
```
models/tt_dit/utils/sweep_mm_block_sizes.py
```

**Run example** (sweep one shape on WH):
```bash
source python_env/bin/activate
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "1536_6144_4608_8x9_mm_plain-wh_4x8_ring" -x -s
```

**Config storage**: `models/tt_dit/utils/matmul.py`
- `grid_89_configs` — 8×9 grid (WH full compute grid)
- `grid_88_configs` — 8×8 grid (reduced grid for some ops)
- `fused_mmrs_configs` — fused matmul+reduce-scatter configs
- `grid_1d_configs` — 1D matmul configs

**Selected sweep results** (WH Galaxy, 1024 res):

| Shape (M, K, N) | Op | Grid | Best (M, K, N, subblock) | Latency |
|---|---|---|---|---|
| 1024, 6144, 4608 | MM | 8×9 | 4, 8, 10 (2,2) | 818 µs |
| 512, 6144, 4608 | MM | 8×9 | 2, 8, 9 (1,3) | 750 µs |
| 1024, 6144, 2304 | MM | 8×9 | 4, 6, 10 (2,2) | 451 µs |
| 1024, 768, 4608 | AGMM | 8×8 | 20, 3, 6 (2,2) | 355 µs |
| 512, 768, 4608 | AGMM | 8×8 | 10, 3, 12 (1,4) | 242 µs |
| 1024, 768, 768 | AGMM | 8×8 | 4, 1, 16 (1,4) | 190 µs |
| 1024, 2304, 6144 | MM+RS | 8×7 | 12, 6, 8 (1,4) | 953 µs |
| 512, 2304, 6144 | MM+RS | 8×7 | 10, 8, 8 (1,4) | 662 µs |

**Impact**: ~11% denoising improvement on the previous branch (default 8×8×8 blocking → swept).

### 2. Ring SDPA Chunk Size Tuning (Commit: `422191a`)

Swept `(q_chunk_size, k_chunk_size)` for Ring SDPA on WH Galaxy. Worker grid is 8×8
(full 8×9 minus one row reserved for CCL).

**Best config** (1024 res): `q_chunk=128, k_chunk=512`
**Best config** (4096 spatial tokens): `q_chunk=256, k_chunk=512`

**Full sweep** (4096 spatial tokens, denoising/step):

| q_chunk | k_chunk | Per step | Pipeline | Notes |
|---|---|---|---|---|
| 128 | 512 | ~556 ms (1.80 it/s) | ~28.33s | default |
| **256** | **512** | **546.9 ms (1.83 it/s)** | **27.98s** | **best (-1.6%)** |
| 256 | 256 | 549.3 ms (1.80 it/s) | 28.11s | -0.8% |
| 128 | 256 | ~556 ms | ~28.33s | hung mid-run |
| 512 | 512 | — | — | device error (stale state) |
| 256 | 1024 | — | — | L1 OOM (2.1 MB > 1.5 MB limit) |

**Config location**: `models/tt_dit/blocks/attention_opt.py`
```python
ring_sdpa_chunk_size_map = {
    (False, 4, 8): {-1: (128, 512), 4096: (256, 512)},
    ...
}
```

### 3. Merged Spatial+Prompt Matmuls (Commit: `09c752d`)

In single-stream transformer blocks, `proj_mlp` and `proj_out` use the **same weights**
for both spatial and prompt tensors. Instead of running two separate matmuls (one for
spatial, one for prompt), we concatenate along dim=1 and run a single larger matmul.

**Before** (per block):
```
proj_mlp(spatial)  → (1024, 6144, 4608)  # M=1024
proj_mlp(prompt)   → (512, 6144, 4608)   # M=512 (very inefficient)
proj_out(spatial)  → separate matmul
proj_out(prompt)   → separate matmul
```

**After** (per block):
```
proj_mlp(concat(spatial, prompt)) → (1536, 6144, 4608)  # single matmul
proj_out(concat(spatial, prompt)) → single matmul
```

This reduces the number of matmul dispatches and improves core utilization by avoiding
the small-M prompt matmul.

**Code location**: `models/tt_dit/models/transformers/transformer_flux2.py`
- `_forward_merged()` — new merged path
- `_forward_separate()` — original separate path (used when `compute_prompt_output=False`)
- `forward()` dispatches between them

**Impact**: 6.6% pipeline speedup (1.83 → 1.95 it/s)

### 4. Needs-Gather Check (Commit: `09c752d`)

Added a `needs_gather` check in `ColParallelLinear` to skip redundant AllGather operations
when the input tensor already has the full K dimension (i.e., `x.padded_shape[-1] == weight.padded_shape[-2]`).

**Code location**: `models/tt_dit/layers/linear.py`

**Impact**: No measurable change in current critical path (AllGather was already avoided
where unnecessary), but prevents future regressions.

---

## FSDP Analysis

### Why FSDP is Required

WH Galaxy has ~1 GB DRAM per device vs ~8 GB on BH. The Flux2 model weights don't fit
in a single device's DRAM without sharding. FSDP (Fully Sharded Data Parallelism) shards
weights across SP-axis devices and AllGathers them before each layer.

**Without FSDP**: DRAM OOM (only 4 MB free out of 1 GB).

### FSDP Overhead (Single Transformer Block, 1024 res)

| Metric | FSDP Enabled | FSDP Disabled | Delta |
|---|---|---|---|
| Total kernel time | 1517 ms | 1102 ms | -27.3% |
| AllGather time | 511 ms (33.7%) | 97 ms (8.8%) | -81.1% |
| Matmul time | ~same | ~same | 0% |
| SDPA time | ~same | ~same | 0% |

FSDP AllGather accounts for **34% of the single transformer block** execution time.
Reducing this overhead is the single largest remaining optimization opportunity.

### Selective bf8 FSDP Weight Gather — biggest WH win

Since FSDP re-gathers the (constant) weights every denoising step, transferring
them in `bfloat8_b` instead of `bfloat16` halves the gather traffic. The bf8 copy
of each sharded weight is created once during eager warmup and cached on the
module, so the typecast stays out of the captured denoising trace (zero per-step
typecast cost). Activations remain bf16; the matmuls consume the bf8 weight directly.

**Code**: `models/tt_dit/layers/linear.py` — `_fsdp_weight_for_gather()` + the 3
FSDP gather sites (`ColParallelLinear`, `RowParallelLinear`, `forward_fused_addcmul`).
Enabled per-layer via the `fsdp_gather_bf8=True` constructor arg.

**Selective bf8 (shipped default)**: only the feed-forward / MLP weights use bf8
(`proj_mlp`, `proj_out` in the single-stream block; `ff`, `ff_context` in the
double-stream block). Attention QKV/out weights stay bf16, because image quality is
dominated by the attention path. bf8 can also be enabled for *all* FSDP layers
(including attention) via the `TT_FSDP_BF8=1` flag for a bit more speed, but it
noticeably degrades image quality, so it is not the default.

**Quality of selective bf8** (1024×1024, vs bf16 baseline):

| Config | Pipeline | PSNR | Mean pixel diff | Pixels >10 |
|---|---|---|---|---|
| bf16 vs bf16 (noise floor, same seed) | 25.71s | ∞ (identical) | 0.000 | 0.00% |
| bf8 MLP only (attention bf16) | 21.87s | **38.3 dB** | **0.65** | **0.64%** |

The pipeline is **bit-deterministic** (two bf16 runs with the same seed are
pixel-identical, PSNR ∞), so the bf8 delta above is a *genuine* change, not
run-to-run noise. MLP-only bf8 is a *small* real deviation (38.3 dB, <1% of
pixels) — near-baseline quality with most of the speedup.
Baseline images: `flux2_1024_bf16_baseline.png`, `flux2_1024_bf8_mlp_only.png`.

**Per-op weight-gather speedup** (8×9 grid):

| Weight shape | bf16 | bf8 | Speedup |
|---|---|---|---|
| 1536×4608 | 726µs | 443µs | -39% |
| 3072×1536 | 500µs | 309µs | -38% |
| 1536×2304 | 379µs | 237µs | -37% |
| 2304×1536 | 381µs | 239µs | -37% |

**Pipeline impact (1024×1024)**: 25.71s → **21.87s** with selective MLP-only bf8
(1.99 → 2.35 it/s, -15%). The win exceeds the gather savings alone because the
matmuls also run faster with bf8 weights.

**Profiler reports**: `flux2_outputs/wh_perf_reports/`
- `fsdp_1024_tt_perf_report*.csv` — processed report, transformer block with FSDP
- `nofsdp_1024_tt_perf_report*.csv` — processed report, transformer block without FSDP
- `*_stacked.png` — stacked op-time breakdown charts

---

## Topology Comparison

**Ring vs Linear on WH Galaxy**:
- Linear topology **hangs** with fused AGMM/MM+RS operations
- Ring topology works correctly and is the production config
- `shard_prompt=True` also hangs on WH Ring (used only on BH)

---

## Key Differences: WH vs BH Galaxy

| Property | WH Galaxy | BH Galaxy |
|---|---|---|
| DRAM per device | ~1 GB | ~8 GB |
| L1 per core | 1464 KB | ~1500 KB |
| Compute grid | 8×9 | 11×10 / 12×9 |
| FSDP required | Yes | No |
| Num links | 4 | 2 |
| `shard_prompt` | Not supported (hangs) | Supported |
| Linear topology | Hangs with fused ops | Supported |

---

## Scripts and Commands

### Full Pipeline Performance Test (1024×1024 — the production target)

This is the default "just run it" command. With **no env var set** it runs the
shipped config (selective bf8 — see note below) and saves the output image to
`flux2_4x8_1024x1024.png` in the repo root.

```bash
cd tt-metal
source python_env/bin/activate

# 1024×1024 (default config: bf8 MLP-only FSDP gather, bf16 attention)
TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest \
  models/tt_dit/tests/models/flux2/test_performance_flux2.py::test_flux2_performance \
  -k "wh_glx_ring_sp0tp1_fsdp and 1024x1024" -x -s --timeout=3600
# -> output image: flux2_4x8_1024x1024.png  (~21.87s, 2.35 it/s)
```

### bf8 FSDP Weight Gather Control (`TT_FSDP_BF8`, tri-state)

**Default = nothing set.** When `TT_FSDP_BF8` is *unset*, each layer honors its own
`fsdp_gather_bf8` flag, which is `True` only for the MLP / feed-forward weights
(`proj_mlp`, `proj_out` in the single-stream block; `ff`, `ff_context` in the
double-stream block) and `False` for attention QKV/out. So the **default run is
already "bf8 MLP-only, attention bf16"** — no flag needed. The env var only exists
to override this for A/B measurement:

```bash
# (unset / DEFAULT) bf8 on MLP, bf16 on attention -> 21.87s, PSNR 38.3 dB (shipped)
TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest ... -k "...1024x1024" -x -s

# Enable bf8 for ALL FSDP layers incl. attention (a bit faster, but more degraded quality):
TT_FSDP_BF8=1 TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest ... -k "...1024x1024" -x -s

# Force bf8 OFF everywhere (pure bf16 reference, 25.71s):
TT_FSDP_BF8=0 TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest ... -k "...1024x1024" -x -s
```

Control logic lives in `_fsdp_weight_for_gather()` in `models/tt_dit/layers/linear.py`:
`"0"` → always bf16; `"1"` → always bf8; unset → per-layer `fsdp_gather_bf8` flag.

### Higher Resolutions (2048×2048) — DRAM OOM on WH

2048×2048 **does not fit** on WH Galaxy — it OOMs in DRAM during warmup (the
~1 GB/device DRAM is already ~981 MB full at 1024, and 2048 needs ~4× the
activation memory). This is the same DRAM wall that requires FSDP; it is **not** a
regression from any optimization here. Observed failure:

```text
# TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest ... -k "...2048x2048" -x -s
RuntimeError: TT_FATAL @ bank_manager.cpp:462
Out of Memory: Not enough space to allocate 536870912 B (512 MB) DRAM buffer
  across 12 banks, bank size 1070773184 B
  (allocated: 981057280 B, free: 89715904 B, largest free block: 42656320 B)
# raised from ttnn.from_torch -> to_device (core.py:352) during eager warmup
```

Making 2048 fit would require DRAM-sharding the large replicated activation
(the 512 MB buffer) and/or more aggressive bf8 — not pursued since 1024×1024 is
the production target.

### Image Quality Comparison (vs bf16 baseline)

```bash
# Each pipeline run saves flux2_4x8_1024x1024.png; copy it per-config then compare:
python3 -c "
from PIL import Image; import numpy as np
a=np.asarray(Image.open('flux2_1024_bf16_baseline.png').convert('RGB')).astype('float32')
b=np.asarray(Image.open('flux2_1024_bf8_mlp_only.png').convert('RGB')).astype('float32')
mse=((a-b)**2).mean(); print('PSNR', 10*np.log10(255**2/mse) if mse else 'inf')
"
# Side-by-side for slides: flux2_1024_bf16_vs_bf8_sidebyside.png
```

### Single Transformer Block Profiler

```bash
TT_DIT_CACHE_DIR=tt_dit_cache_shard python -m tracy -r -m \
  "pytest models/tt_dit/tests/models/flux2/test_transformer_flux2.py::test_transformer_profile \
  -k 'wh_4x8_ring_fsdp and x_both' -x -s"
```

### Matmul Block Size Sweep

```bash
# Sweep all WH shapes
python models/tt_dit/utils/sweep_mm_block_sizes.py --device-config wh_4x8_ring

# Sweep specific shape
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
  -k "1536_6144_4608_8x9_mm_plain-wh_4x8_ring" -x -s
```

### Generate tt-perf-report from profiler CSV

```bash
tt-perf-report <path_to_ops_perf_results.csv> -o <output_dir>
```

### Debug: Print Missing Matmul Configs

```bash
TT_DEBUG_MM_SHAPES=1 TT_DIT_CACHE_DIR=tt_dit_cache_shard pytest \
  models/tt_dit/tests/models/flux2/test_transformer_flux2.py::test_transformer_profile \
  -k "wh_4x8_ring_fsdp and 1024 and x_both" -x -s 2>&1 | grep "MM_MISS"
```

---

## Remaining Optimization Opportunities

1. ~~Sweep merged matmul shapes~~ — **DONE**: `(1536,6144,4608)` and
   `(4608,6144,4608)` swept and applied to `grid_89_configs`.

2. ~~Reduce FSDP AllGather (weight compression)~~ — **DONE**: selective bf8 weight
   gather on MLP layers (−15% pipeline). Still open within FSDP:
   - Overlap weight AllGather with compute (prefetch next layer's gather).
   - bf8 attention weights too (`TT_FSDP_BF8=1`) — a bit faster but more degraded quality, so not enabled by default.
   - Finer split (keep first/last blocks bf16) to push PSNR > 38.3 dB.

3. **Quality validation at scale** — bf8 quality so far is one image (PSNR 38.3 dB
   vs a bit-deterministic bf16 baseline). A prompt-set eval (FID/CLIP) would harden
   the bf8 decision before broad rollout.

4. **Upstream sync** — Monitor `origin/friedrich/flux2` for new BH optimizations
   that can be adapted for WH. Note: upstream heuristic blocking (`use_heuristic=True`)
   is BH-tuned and causes L1 OOM on WH — don't use directly.

---

## File Map

| File | Purpose |
|---|---|
| `models/tt_dit/utils/matmul.py` | Matmul block config lookup tables |
| `models/tt_dit/utils/sweep_mm_block_sizes.py` | Matmul sweep script |
| `models/tt_dit/blocks/attention_opt.py` | Ring SDPA chunk sizes, attention config |
| `models/tt_dit/blocks/transformer_block_opt.py` | Double-stream transformer block |
| `models/tt_dit/models/transformers/transformer_flux2.py` | Single-stream block + merged path |
| `models/tt_dit/layers/linear.py` | Linear layers: needs_gather fix + bf8 FSDP gather (`_fsdp_weight_for_gather`, `TT_FSDP_BF8`) |
| `models/tt_dit/layers/feedforward.py` | `ParallelFeedForward` — plumbs `fsdp_gather_bf8` |
| `models/tt_dit/tests/models/flux2/test_performance_flux2.py` | Full pipeline perf test |
| `models/tt_dit/tests/models/flux2/test_transformer_flux2.py` | Transformer block profiler test |
| `flux2_outputs/wh_perf_reports/` | Profiler CSV/PNG reports |
| `flux2_outputs/flux2_wh_glx_optimization.md` | This document |
| `flux2_outputs/flux2_1024_bf16_baseline.png` / `flux2_1024_bf8_mlp_only.png` | Quality comparison images |
| `flux2_outputs/flux2_1024_bf16_vs_bf8_sidebyside.png` | Side-by-side for slides |
