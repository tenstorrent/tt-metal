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
# Single device (P150), pytest
HF_MODEL=perplexity-ai/pplx-embed-v1-0.6b MESH_DEVICE=P150 pytest \
  models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py -sv

# Standalone (no pytest)
python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py

# DP=32 multi-process on Galaxy (32x P150)
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --num-devices 32 --iterations 10 --warmup 2
```

## Directory Structure

```
models/demos/blackhole/pplx_embed_0_6b/
├── __init__.py
├── README.md
├── tt/
│   ├── __init__.py
│   └── attention.py          # PplxBidirectionalAttention subclass (is_causal=False)
├── demo/
│   ├── __init__.py
│   ├── _common.py            # Model builder, optimizations, run_perf
│   ├── demo_bs1_isl512.py    # BS=1, ISL=512 perf demo (pytest + standalone)
│   └── dp32_multiprocess.py  # DP=32 multi-process Galaxy benchmark
└── tests/
    └── perf/
        └── __init__.py
```

## Performance

### Single device (P150)

| Batch | ISL | Mode           | Avg Prefill | Best Prefill | Best emb/s | Best tok/s |
|-------|-----|----------------|-------------|--------------|------------|------------|
| 1     | 512 | Direct trace   | 7.1ms       | 7.1ms        | 142        | 72,511     |
| 1     | 512 | Full pipeline   | 8.0ms       | 7.8ms        | 128        | 65,507     |

- **Direct trace**: Pure device execution via `ttnn.execute_trace` + sync (no post-processing overhead).
- **Full pipeline**: Includes Generator loop, RMSNorm post-processing, D2H, and host extraction.

### DP=32 Multi-Process (Galaxy, 32x P150)

Each chip runs as an independent subprocess with `TT_VISIBLE_DEVICES` isolation, dedicated CPU cores, and barrier-synchronized measurement. Timing includes the full pipeline: trace execution, RMSNorm post-processing, D2H copy, and host extraction.

| Chips | Batch/chip | ISL | Per-chip mean | Per-chip min | Slowest median | Throughput (median) | Throughput (best) |
|-------|-----------|-----|---------------|--------------|----------------|--------------------|--------------------|
| 32    | 1         | 512 | 8.0ms         | 7.4ms        | 8.2ms          | 3,886 emb/s (2.0M tok/s) | 4,308 emb/s (2.2M tok/s) |

Key properties:
- **Near-perfect scaling**: all 32 chips achieve ~8.0ms per-chip (same as single-device full pipeline).
- **Full pipeline**: includes Generator prefill trace, RMSNorm post-processing, D2H, and host extraction.
- **CPU pinning**: 2 cores/worker (64 total) prevents OS scheduler contention.
- **Barrier sync**: all workers finish warmup before any measurement begins.

## Implementation Notes

- **No `tt_transformers` modifications**: all changes are localized to this directory.
- **Bidirectional SDPA**: `PplxBidirectionalAttention` wraps the SDPA call in `forward_prefill` to inject `is_causal=False` without duplicating the entire method.
- **Weight loading**: `PplxModelArgs` subclass overrides `get_hf_model_cls()` to use `AutoModel` (no LM head) and enables `trust_remote_code` for the custom HF config.
- **Optimization knobs**: BFP4 weights (all layers), BFP8 activations, head-split TMs, L1 RoPE, block-sharded LayerNorm, KV cache fill skip, SDPA LOFI (safe for embeddings), bigger SDPA chunks for bs=1.
- **Direct trace replay**: Benchmark loop uses `ttnn.execute_trace` directly to eliminate Generator Python overhead and post-trace D2H processing, measuring pure device execution latency.
- **DP=32 multi-process**: `dp32_multiprocess.py` spawns one `multiprocessing.Process` per chip with `TT_VISIBLE_DEVICES` isolation, CPU affinity pinning, and a `Barrier` for synchronized measurement. Direct trace replay ensures near-zero Python overhead per chip.
