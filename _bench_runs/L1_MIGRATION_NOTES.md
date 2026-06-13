# L1 weight migration — pi0.5 BH-Galaxy

**Status as of 2026-06-13 (verified by e2e perf test).** What's on L1 today,
what's verified to crash, and the headroom that's left.

## TL;DR

- **Denoise weights are L1-resident by default** (`PI0_GLX_DENOISE_L1=1`).
- **SigLIP + VLM safe-stack landed (this commit)** — all flags in
  `_bench_runs/pi05_production.env`. Adds ~150 MB to vision chips and
  ~45 MB / 18 prefill chips of L1-resident weights vs. prior production.
- **Hard ceiling found:** the matmul kernel's static CB region clashes with
  interleaved L1 weights at **N ≥ 4608** (SigLIP fc1/wqkv) and at **N=16384**
  (VLM gate_proj / up_proj). All four are verified to crash at compile time.
- **One pleasant surprise:** VLM `down_proj` (K=16384 but N=2048) was flagged
  as "depends on in0 CB sizing" in the prior plan — it actually works
  cleanly with the normal interleaved matmul. That's the biggest single
  weight (~35 MB/chip × 18 chips) of the safe-stack.

## Verified safe-stack (production env)

```bash
# denoise weights (already on L1 — default ON in pipeline.py)
export PI0_GLX_DENOISE_L1=1

# SigLIP small-N matmuls
export PI0_GLX_SIGLIP_L1_TENSORS=wo,fc2

# VLM PaliGemma attention
export PI0_GLX_VLM_ATTN_L1=1

# VLM MLP — down_proj only (gate/up still in DRAM)
export PI0_GLX_PREFILL_MLP_L1=1
export PI0_GLX_PREFILL_MLP_L1_PROJ=down_proj
export PI0_GLX_PREFILL_MLP_L1_LAYOUT=interleaved
```

| Stage | Weight placement | Flag |
|---|---|---|
| Denoise expert (QKV/O, gate/up/down, adaRMS mod, RoPE) | **L1 interleaved** | `PI0_GLX_DENOISE_L1=1` (default ON) |
| KV cache | **L1 interleaved** | (same flag) |
| Suffix MLP, denoise head | **L1 interleaved** | (same flag) |
| SigLIP wo, fc2 | **L1 interleaved** | `PI0_GLX_SIGLIP_L1_TENSORS=wo,fc2` |
| SigLIP fc1, wqkv | DRAM | (CB clash blocks L1) |
| VLM wqkv, o_proj | **L1 interleaved** | `PI0_GLX_VLM_ATTN_L1=1` |
| VLM down_proj | **L1 interleaved** | `PI0_GLX_PREFILL_MLP_L1=1`, `PROJ=down_proj` |
| VLM gate_proj, up_proj | DRAM | (CB clash blocks L1) |
| LN weights (all stages) | L1 (gated) | `PI0_LN_WEIGHTS_L1=1` |
| RoPE tables | L1 (gated) | `PI0_ROPE_TABLES_L1=1` |

## E2E perf — measured

All runs: `test_perf_tt_bh_glx_e2e.py`, 1 warmup + 3 iters, production env
(LIBERO 3-cam, 5 denoise steps). Per-call wall-clock stddev was 2.3–4.4 ms
across all runs — the deltas below sit inside that noise, so this work is
**capacity-unblocking, not a wall-clock win at the e2e level**. The per-matmul
kernel-time drop needs Tracy to measure.

| Configuration | Total (ms) | Vision (ms) | Prefill (ms) | Denoise (ms) | Outcome |
|---|---:|---:|---:|---:|---|
| Baseline (no Tier 1) | 277.94 | 37.50 | 31.10 | 204.59 | PASS |
| + SigLIP wo,fc2 | 273.06 | 36.62 | 31.51 | 199.87 | PASS |
| + VLM wqkv,o_proj | 279.72 | 37.20 | 31.59 | 205.76 | PASS |
| + Both above (stacked) | 280.38 | 38.69 | 31.58 | 205.05 | PASS |
| + VLM down_proj | 278.38 | — | 32.21 | 204.58 | **PASS** ← max safe |
| + VLM up_proj (also) | — | — | — | — | **CRASH** (N=16384) |
| + SigLIP fc1 (only on stack) | — | — | — | — | **CRASH** (N=4608) |

Logs: `_bench_runs/l1_tier1_runs/{00..06}*.log`.

## The blocker — kernel CB vs interleaved L1 weights

The matmul kernel statically reserves a CB region at the bottom of each
core's L1. That region scales with the matmul's `N` dimension (and per-core
`N` block size). Interleaved L1 weights get assigned to a fixed offset across
cores; if that offset falls within the CB region, the program crashes at
compile time. Verified errors (this commit's runs):

```
# SigLIP fc1 (N=4608):
TT_THROW: Statically allocated circular buffers in program 1126 clash with
L1 buffers on core range [0-0 - 11-7].
L1 buffer allocated at 419776 and static circular buffer region ends at 464384

# VLM up_proj (N=16384):
TT_THROW: Statically allocated circular buffers in program 1296 clash with
L1 buffers on core range [0-0 - 11-9].
L1 buffer allocated at 367104 and static circular buffer region ends at 455168
```

**The N threshold sits at exactly 4608.** N=4096 (denoise expert) works;
N=4608 (SigLIP fc1/wqkv) is the first crash. N=16384 (VLM gate/up) has even
larger margin to the L1-buffer slot, hence the wider core-range in the error.

## Per-matmul fitness — final table

### SigLIP (×9 layers per chip on vision chips 1/2/3)

| Op | Shape (bf8) | N | Bytes | Interleaved-L1 status |
|---|---|---:|---:|---|
| wqkv | 1152×4608 | 4608 | 5.4 MB | ❌ CRASH (CB clash) |
| **wo** | 1536×1152 | 1152 | 1.85 MB | ✅ **PASS — in production env** |
| fc1 | 1152×4608 | 4608 | 5.4 MB | ❌ CRASH (CB clash) |
| **fc2** | 4608×1152 | 1152 | 5.4 MB | ✅ **PASS — in production env** |

### VLM PaliGemma (×1 layer per chip × 18 prefill chips)

| Op | Shape (bf8) | N | Bytes | Interleaved-L1 status |
|---|---|---:|---:|---|
| **wqkv** | 2048×2560 | 2560 | 5.4 MB | ✅ **PASS — in production env** |
| **o_proj** | 2048×2048 | 2048 | 4.3 MB | ✅ **PASS — in production env** |
| gate_proj | 2048×16384 | 16384 | 35.4 MB | ❌ CRASH (CB clash) |
| up_proj | 2048×16384 | 16384 | 35.4 MB | ❌ CRASH (CB clash) |
| **down_proj** | 16384×2048 | 2048 | 35.4 MB | ✅ **PASS — in production env** ← surprise |

The conservative prior estimate said `down_proj` was uncertain because of the
large `in0` CB driven by K=16384. In practice the kernel's CB sits within the
per-core budget and the L1 weight slot lands above it without overlap.

## What's left — Tier 2 (width-sharded path)

The remaining four matmuls (SigLIP fc1, wqkv; VLM gate_proj, up_proj) all
share the same blocker: their static CB region overlaps the L1 weight slot at
the per-core offset chosen by the interleaved allocator. The known path:

1. Store the weight **width-sharded** across cores so each core owns a
   `[K, N/cores]` column — the L1 buffer slot no longer sits at a single
   per-core offset.
2. Use a matmul kernel that **consumes width-sharded weights** instead of
   bank-interleaved weights.

The width-shard infrastructure is already scaffolded in
`migrate_prefill_mlp_weights_to_l1` (use `layout='width_sharded'`), but the
default `MinimalMatmul` / `ttnn.linear` path doesn't accept width-sharded
weights — calling it would compile-time error a different way. Wiring this up
needs a kernel-team conversation; outside scope for now.

Estimated upside if Tier 2 lands: SigLIP fc1+wqkv ≈ 60 MB/chip × 3 vision
chips + VLM gate+up ≈ 70 MB/chip × 18 prefill chips ≈ **~1.5 GB more L1
weight residency**. Wall-clock impact dwarfed by trace replay (Phase B.3),
which is the next big lever.

## Files

- `_bench_runs/pi05_production.env` — env stanza activates the safe-stack
- `_bench_runs/l1_tier1_runs/*.log` — verified-run logs (this commit)
- `models/experimental/pi0_5/tt/tt_bh_glx/_l1_migration.py` — migrators +
  env helpers, with verified-behavior docstrings
- `models/experimental/pi0_5/tt/tt_bh_glx/pipeline.py:283-311` — invocation
