# Prefill Single-Layer Profiler Analysis

Analysis of 64K prefill, 1 layer, device 0. Focus on ops targeted by fused RS+MM and AG+MM.

---

## Summary (64K, 1 layer)

| Config | Total (ms) | vs Baseline |
|--------|-------------|-------------|
| Baseline | 135.87 | — |
| AG+MM only | 134.06 | −1.81 ms (1.3%) |
| RS+MM only | 132.75 | −3.12 ms (2.3%) |
| Both (expected) | ~131 | −4.9 ms (3.6%) |

**Baseline source:** `profiler_sweep_results/baseline/64k/prefill.csv`
**Config:** `USE_FUSED_AG_MM=0 TT_LLAMA_USE_FUSED_MM_RS=0`

---

## Op Breakdown by Category

### 1. Attention Block (~37.5% of total)

| OP | Total (us) | % | Notes |
|----|------------|---|-------|
| **SDPA** | 50,949 | **37.50%** | Main attention kernel; not fused |
| Matmul (attn output proj) | ~3,281 | 2.4% | After SDPA |
| ReduceScatter (attn out) | ~3,332 | 2.5% | After attn output proj |
| AllGather (attn) | ~978 | 0.7% | 2× for QKV path |
| NlpCreateHeads, RotaryEmbedding, PagedFillCache, etc. | ~5,500 | 4.0% | Setup/overhead |

**Attention subtotal: ~63 ms (46%)**

### 2. MLP Block – Fused-Op Targets

#### RS+MM (FF1, FF3) – `TT_LLAMA_USE_FUSED_MM_RS=1`

| OP | Total (us) | % | Fused? |
|----|------------|---|--------|
| Matmul (FF1 gate_proj) | 3,281 | 2.4% | ✅ Fused with RS |
| ReduceScatter (FF1) | 3,332 | 2.5% | ✅ Fused with MM |
| Matmul (FF3 up_proj) | 6,065 + 6,345 = 12,410 | 9.1% | ✅ Fused with RS |
| ReduceScatter (FF3) | 5,552 + 5,430 = 10,982 | 8.1% | ✅ Fused with MM |

**RS+MM target subtotal: ~30.0 ms (22%)** — Matmul + ReduceScatter for FF1 and FF3

#### AG+MM (FF2) – `USE_FUSED_AG_MM=1`

| OP | Total (us) | % | Fused? |
|----|------------|---|--------|
| AllGather (FF2 input) | 3,393 | 2.5% | ✅ Fused with MM |
| Matmul (FF2 down_proj) | 9,861 | 7.3% | ✅ Fused with AG |
| ReduceScatter (FF2 output) | 3,323 | 2.4% | ❌ Stays separate (AllReduce) |

**AG+MM target subtotal: ~13.25 ms (9.8%)** — AllGather + Matmul for FF2 (row 35+36)

#### Other MLP

| OP | Total (us) | % |
|----|------------|---|
| BinaryNg (mul FF1×FF3) | 5,214 | 3.8% |
| LayerNorm (×3) | 2,910 + 4,332 | 5.3% |
| AllGather (LayerNorm) | ~565 | 0.4% |

---

## Fused-Ops Summary (Baseline)

| Fusion | Ops Replaced | Baseline Time | Expected Savings |
|--------|--------------|---------------|------------------|
| **RS+MM** (FF1, FF3) | 4× (MM+RS) pairs | ~30.0 ms | Eliminate RS overhead, overlap MM+RS |
| **AG+MM** (FF2) | 1× (AG+MM) | ~13.25 ms | Eliminate AG overhead, overlap AG+MM |
| **Combined** | — | **~43.25 ms** | **~32% of layer time** |

---

## Attention vs MLP Bottleneck

| Block | Time (ms) | % |
|-------|-----------|---|
| **Attention** | ~63 | **46%** |
| **MLP (all)** | ~55 | **40%** |
| Other (Embed, LayerNorm, etc.) | ~18 | 14% |

- **Attention (SDPA)** is the largest single op (37.5%).
- **MLP** is ~40% of the layer.
- When MLP is sped up by fused ops, **Attention can become the dominant bottleneck**.
- Total gain from fusions is capped by the remaining Attention time.

---

## AG+MM Comparison (agmm_only run)

**Fused op:** `AllGatherMinimalMatmulAsyncOp` — appears at **line 39** in `agmm_only/64k/prefill.csv`

| Metric | Baseline | AG+MM Only | Δ |
|--------|----------|------------|---|
| Total prefill time (ms) | 135.87 | _134.06_ | **−1.81 ms (1.3%)** |
| AllGatherAsyncDeviceOperation (us) | 10,601 | 7,248 | −3,353 |
| MatmulDeviceOperation (us) | 30,274 | 20,472 | −9,802 |
| **AllGatherMinimalMatmulAsyncOp** (us) | — | **11,338** | (new fused) |
| ReduceScatterMinimalAsyncDeviceOperation (us) | 19,606 | 19,638 | +32 (noise) |
| SDPA (us) | 50,949 | 50,950 | ~0 (unchanged) |

**FF2 fusion (line 39):**
- Baseline: AG (3,393 us) + Matmul (9,861 us) = **13,254 us**
- AG+MM: AllGatherMinimalMatmulAsyncOp = **11,338 us**
- **Savings: 1,916 us (~14.5%)** from overlapping AG+MM

**Checks:**
- [x] FF2 AllGather + Matmul replaced by single fused op
- [x] AllGatherAsyncDeviceOperation drops by ~3.4 ms
- [x] MatmulDeviceOperation drops by ~9.8 ms
- [x] SDPA unchanged (no fusion in attention)
- [x] Total time drop ~1.8 ms (fusion saves ~1.9 ms; small variance)

---

## RS+MM Comparison (rsmm_only run)

**Fused op:** `MinimalMatmulStridedReduceScatterAsync` — appears at **lines 32–33** in `rsmm_only/64k/prefill.csv` (FF1 + FF3)

| Metric | Baseline | RS+MM Only | Δ |
|--------|----------|------------|---|
| Total prefill time (ms) | 135.87 | **132.75** | **−3.12 ms (2.3%)** |
| MatmulDeviceOperation (us) | 30,274 | 17,821 | −12,453 |
| ReduceScatterMinimalAsyncDeviceOperation (us) | 19,606 | 9,007 | −10,599 |
| **MinimalMatmulStridedReduceScatterAsync** (us) | — | **19,951** | (new fused, 2×) |
| AllGatherAsyncDeviceOperation (us) | 10,601 | 10,630 | +29 (noise) |
| SDPA (us) | 50,949 | 50,948 | ~0 (unchanged) |

**FF1+FF3 fusion (lines 32–33):**
- Baseline: FF1 (Matmul 3,281 + RS 3,332) + FF3 (Matmul 12,410 + RS 10,982) = **30,005 us**
- RS+MM: MinimalMatmulStridedReduceScatterAsync 10,225 + 9,726 = **19,951 us**
- **Savings: ~10.0 ms (~33%)** from overlapping MM+RS for FF1 and FF3

**Checks:**
- [x] FF1 and FF3 Matmul+RS replaced by 2× fused ops
- [x] MatmulDeviceOperation drops by ~12.5 ms
- [x] ReduceScatterMinimalAsyncDeviceOperation drops by ~10.6 ms
- [x] SDPA unchanged (no fusion in attention)
- [x] Total time drop ~3.1 ms (fusion saves ~10 ms; some overlap with other ops)

---

## CSV Row Reference

**Baseline (baseline/64k/prefill.csv):**
- **FF1 RS+MM:** Matmul 3,281 us (row 26), RS 3,332 us (row 27)
- **FF3 RS+MM:** Matmul 6,065 + 6,345 us (rows 31–32), RS 5,552 + 5,430 us (rows 33–34)
- **FF2 AG+MM:** AG 3,393 us (row 36), Matmul 9,861 us (row 37)

**AG+MM (agmm_only/64k/prefill.csv):**
- **Fused op:** `AllGatherMinimalMatmulAsyncOp` 11,338 us at **line 39** (replaces AG + Matmul)

**RS+MM (rsmm_only/64k/prefill.csv):**
- **Fused op:** `MinimalMatmulStridedReduceScatterAsync` 10,225 + 9,726 us at **lines 32–33** (replaces FF1 and FF3 Matmul+RS)

---

## Files

- Baseline: `profiler_sweep_results/baseline/64k/prefill.csv`
- AG+MM: `profiler_sweep_results/agmm_only/64k/prefill.csv`
- RS+MM: `profiler_sweep_results/rsmm_only/64k/prefill.csv`
- Summary: `profiler_sweep_results/<run>/64k/summary.txt`
