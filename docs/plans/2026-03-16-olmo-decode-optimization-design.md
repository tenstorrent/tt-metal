# OLMo Decode Optimization — Design Document

**Date:** 2026-03-16

## Problem

OLMo-3.1-32B decode achieves ~17 tok/s/user on TG (Galaxy 4×8), vs Qwen3-32B at ~3x better performance for the same parameter count. Root cause analysis via `decode.csv` profiler output (1-layer model, 8076 us/step total):

## Root Causes

### 1. Q-norm on 1 core (DRAM) — 98 us/layer

After `llama_rs_create_heads`, Q heads are `[8-batch, 5-real-heads, 128]`. The Q global norm path (slice→reshape→tilize→rms_norm_pre→all_gather→rms_norm_post→untilize) runs entirely on DRAM with **1 core**, causing:
- `TilizeWithValPadding`: 47.40 us at 1 core
- `LayerNormPreAllGather`: 13.17 us at 1 core
- `LayerNormPostAllGather`: 24.39 us at 1 core
- `UntilizeWithUnpadding`: 13.89 us at 1 core

Fix: Flatten to `[32, 640]` (tiled), use **20-core L1 WIDTH_SHARDED** (640/32 = 20 tiles, 1 tile/core). The `rms_norm_pre/post_all_gather` ops are designed for width-sharded distributed norm.

### 2. K-norm on DRAM (L1→DRAM roundtrip) — 31 us/layer

K heads exit `llama_rs_create_heads` as L1_HEIGHT_SHARDED `[8, 128]`, are immediately moved to DRAM for tilize+norm+untilize, then back to L1. Fix: Keep on **L1 HEIGHT_SHARDED** (8 cores, one per batch item), avoid DRAM roundtrip.

### 3. MLP decode DRAM paths — 70 us/layer

OLMo-specific branches in `llama_mlp.py` force things to DRAM:
- `ff1ff3_mem_config = ttnn.DRAM_MEMORY_CONFIG if is_olmo` (SiLU mul output)
- FF2 `line_all_gather(..., memory_config=ttnn.DRAM_MEMORY_CONFIG)` (input to FF2)
- `w2_interleaved` (unoptimized generic matmul, no `program_config`) instead of `self.w2` + `FF2_DRAM_SHARDED_PROGCFG_OLMO`

### 4. WO path: AllBroadcast+Tilize on 1 core (DRAM) — 37 us/layer

OLMo WO path slices 8→5 heads, then reassembles batch via `line_all_gather` (AllBroadcast on DRAM, 1 core, 13 us) + `Concat` + `Tilize` (1 core, 8 us) before DRAM matmul. Fix: Skip the slice — pass full 8-head output (3 zero-padded heads) into standard `all_gather_concat` → `WO_DECODE_RING_PROGCFG` using `wo_ring` weight (already padded to K=1024 with zeros for phantom heads).

## Expected Gains

| Task | us/layer | ms/64 layers |
|---|---|---|
| K-norm L1 | ~15 | ~1.0 |
| Q-norm 20-core L1 | ~93 | ~5.9 |
| MLP L1 | ~70 | ~4.5 |
| WO ring | ~37 | ~2.4 |
| **Total** | **~215** | **~13.8 ms → ~25 tok/s** |

## PCC Gates

| Test | Baseline | Gate |
|---|---|---|
| Decode 1L | 0.9983 | ≥ 0.998 |
| Decode 4L | 0.9963 | ≥ 0.995 |
| Decode 64L | 0.8165 | ≥ 0.80 |

Run after **every task**. Revert immediately on failure.
