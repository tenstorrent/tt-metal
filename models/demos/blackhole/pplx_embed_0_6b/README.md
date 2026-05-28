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
# Single device (P150), bs=1 ISL=512
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py

# Other batch/ISL combinations
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs8_isl512.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs32_isl512.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl1024.py
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl2048.py

# Full pipeline latency (includes Generator overhead, post-processing, D2H)
TT_VISIBLE_DEVICES=1 python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py --full-pipeline

# Via pytest
pytest models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py -sv

# DP=32 multi-process on Galaxy (32x P150)
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32 --iterations 10
```

## Directory Structure

```
models/demos/blackhole/pplx_embed_0_6b/
├── __init__.py
├── README.md
├── tt/
│   ├── __init__.py
│   └── attention.py              # PplxBidirectionalAttention (is_causal=False)
├── demo/
│   ├── __init__.py
│   ├── _common.py                # Model builder, WORKLOAD_CONFIGS, run_perf
│   ├── demo_bs1_isl512.py        # BS=1,  ISL=512  (L1 activations)
│   ├── demo_bs8_isl512.py        # BS=8,  ISL=512  (L1 batched activations)
│   ├── demo_bs32_isl512.py       # BS=32, ISL=512  (DRAM optimized)
│   ├── demo_bs1_isl1024.py       # BS=1,  ISL=1024 (DRAM optimized)
│   ├── demo_bs1_isl2048.py       # BS=1,  ISL=2048 (DRAM optimized)
│   ├── demo_bs8_isl1024.py       # BS=8,  ISL=1024 (DRAM optimized)
│   ├── demo_bs8_isl2048.py       # BS=8,  ISL=2048 (DRAM optimized)
│   ├── demo_bs32_isl1024.py      # BS=32, ISL=1024 (DRAM optimized)
│   ├── demo_bs32_isl2048.py      # BS=32, ISL=2048 (DRAM optimized)
│   ├── dp32_multiprocess.py      # DP=32 multi-process Galaxy benchmark
│   └── eval_accuracy.py          # Accuracy evaluation (STS-B)
└── tests/
    └── perf/
        ├── __init__.py
        ├── new_perf_bs1_isl512.py   # Tracy signpost test (bs=1, ISL=512)
        ├── new_perf_bs8_isl512.py   # Tracy signpost test (bs=8, ISL=512)
        └── new_perf_bs32_isl512.py  # Tracy signpost test (bs=32, ISL=512)
```

## Performance

### Single Device (P150) — Direct Trace Replay

Direct trace replay measures pure device execution via `ttnn.execute_trace` + sync, bypassing Generator Python overhead, post-processing, and D2H copy.

| Batch | ISL  | Prefill (avg) | Prefill (best) | Embeddings/s (best) | Tokens/s (best) | Memory   |
|------:|-----:|--------------:|---------------:|--------------------:|----------------:|----------|
|     1 |  512 |         7.2ms |          7.1ms |               140.6 |          71,991 | L1 + big grid |
|     8 |  512 |        43.8ms |         43.7ms |               182.9 |          93,651 | L1       |
|    32 |  512 |       199.1ms |        198.9ms |               160.9 |          82,366 | DRAM     |
|     1 | 1024 |        18.0ms |         17.9ms |                55.9 |          57,207 | L1       |
|     1 | 2048 |        33.4ms |         33.4ms |                30.0 |          61,377 | L1       |
|     8 | 1024 |       134.4ms |        134.4ms |                59.5 |          60,968 | DRAM     |
|     8 | 2048 |       269.6ms |        269.5ms |                29.7 |          60,801 | DRAM     |
|    32 | 1024 |       520.4ms |        520.0ms |                61.5 |          63,019 | DRAM     |
|    32 | 2048 |      1045.4ms |       1045.1ms |                30.6 |          62,706 | DRAM     |

### Workload Optimization Strategy

| Workload    | Strategy     | Key Knobs |
|-------------|-------------|-----------|
| bs=1  ISL=512  | L1 activations (1 MB) + big grid | `QWEN_SDPA_BIG_CHUNK_BS1=1`, `QWEN_MM_BIG_GRID_BH=1` |
| bs=1  ISL=1024 | L1 activations (2 MB) | `QWEN_SDPA_BIG_CHUNK_BS1=1` |
| bs=1  ISL=2048 | L1 activations (4 MB) | `QWEN_SDPA_BIG_CHUNK_BS1=1` |
| bs=8  ISL=512  | L1 batched (8 MB)     | `TT_BATCHED_L1_PREFILL=1` |
| bs=8  ISL=1024 | DRAM (16 MB)          | `QWEN_MM_BIG_GRID_BH=1` (80-core matmul) |
| bs=8  ISL=2048 | DRAM (32 MB)          | `QWEN_MM_BIG_GRID_BH=1` |
| bs=32 ISL=512  | DRAM (32 MB)          | `QWEN_MM_BIG_GRID_BH=1` |
| bs=32 ISL=1024 | DRAM (64 MB)          | `QWEN_MM_BIG_GRID_BH=1` |
| bs=32 ISL=2048 | DRAM (128 MB)         | `QWEN_MM_BIG_GRID_BH=1` |

All workloads additionally use: BFP4 weights, BFP8 activations, LOFI math fidelity, head-split TMs, L1 RoPE, block-sharded LayerNorm, KV cache fill skip, and SDPA LOFI.

### Full Pipeline (Extended Trace)

Full pipeline measures end-to-end latency including H2D, trace replay, post-processing (slice + norm + to_layout), and D2H + torch conversion. The `--full-pipeline` flag captures an *extended trace* that folds post-processing device ops into the trace replay, reducing Python dispatch overhead from ~0.7ms to ~0.1ms:

| Batch | ISL  | Direct (best) | Full Pipeline (best) | Overhead |
|------:|-----:|--------------:|---------------------:|---------:|
|     1 |  512 |         7.1ms |                7.2ms |    0.1ms |

### DP=32 Multi-Process (Galaxy, 32x P150)

Each chip runs as an independent subprocess with `TT_VISIBLE_DEVICES` isolation, dedicated CPU cores, and barrier-synchronized measurement. Supports all `(batch_size, seq_len)` combinations:

```bash
python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 1 --seq-len 512 --num-devices 32

python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \
  --batch-size 32 --seq-len 2048 --num-devices 32
```

## Implementation Notes

- **No `tt_transformers` modifications**: all changes are localized to this directory (except one-line `trust_remote_code` fix in `model_config.py`).
- **Bidirectional SDPA**: `PplxBidirectionalAttention` wraps the SDPA call in `forward_prefill` to inject `is_causal=False`.
- **Weight loading**: `PplxModelArgs` subclass loads weights directly from safetensors, bypassing the custom HF `modeling.py` that requires a newer `transformers` version.
- **Centralized optimization config**: `WORKLOAD_CONFIGS` in `_common.py` maps every `(batch_size, seq_len)` pair to its optimized env-var settings. Both individual demo files and `dp32_multiprocess.py` use `apply_workload_env()` for consistent tuning.
- **Direct trace replay**: Benchmark loop uses `ttnn.execute_trace` directly for lowest-overhead measurement. Pass `--full-pipeline` to measure end-to-end latency with an *extended trace* that includes post-processing ops (slice + norm + to_layout) inside the trace replay, reducing dispatch overhead to ~0.1ms.
- **DP=32 multi-process**: `dp32_multiprocess.py` spawns one `multiprocessing.Process` per chip with `TT_VISIBLE_DEVICES` isolation, CPU affinity pinning, and a `Barrier` for synchronized measurement.
