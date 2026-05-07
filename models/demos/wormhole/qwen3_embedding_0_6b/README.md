# Qwen3-Embedding-0.6B on Tenstorrent

Optimized inference of [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) on Tenstorrent Wormhole (P150/Blackhole) hardware.

## Model overview

| Property | Value |
|---|---|
| Parameters | 0.6B |
| Hidden size | 1024 |
| Layers | 28 |
| Attention heads (Q / KV) | 16 / 8 |
| Intermediate size | 3072 |
| Head dim | 128 |
| GQA ratio | 2:1 |

## Directory structure

```
qwen3_embedding_0_6b/
├── demo/
│   ├── _common.py                 # Shared optimization knobs, model builder, perf harness
│   ├── demo_bs1_isl512.py         # Perf demo — batch=1, ISL=512
│   ├── demo_bs8_isl512.py         # Perf demo — batch=8, ISL=512
│   ├── demo_bs32_isl512.py        # Perf demo — batch=32, ISL=512
│   └── mteb_evaluation.py         # MTEB accuracy eval (TT vs HF reference)
└── tests/
    └── perf/
        ├── new_perf_bs1_isl512.py   # Tracy profiling — batch=1, ISL=512
        ├── new_perf_bs8_isl512.py   # Tracy profiling — batch=8, ISL=512
        └── new_perf_bs32_isl512.py  # Tracy profiling — batch=32, ISL=512
```

## Prerequisites

```bash
# Build tt-metal and activate the virtual environment
source python_env/bin/activate
export MESH_DEVICE=P150
```

## Running demo files

The demo scripts measure prefill latency over multiple iterations with all optimizations enabled. Each can be run via pytest or as a standalone script.

### Batch size 1

```bash
# Via pytest
pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs1_isl512.py -sv

# Standalone
MESH_DEVICE=P150 python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs1_isl512.py
```

### Batch size 8

```bash
# Via pytest
pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs8_isl512.py -sv

# Standalone
MESH_DEVICE=P150 python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs8_isl512.py
```

### Batch size 32

```bash
# Via pytest
pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs32_isl512.py -sv

# Standalone
MESH_DEVICE=P150 python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs32_isl512.py
```

### Common options (standalone mode)

```bash
python .../demo_bs1_isl512.py --device-id 0 --iterations 20
```

## Running MTEB evaluation

The evaluation script runs both the HuggingFace reference model and the TT model on the same MTEB datasets (ArguAna retrieval + STS-Benchmark by default), then displays a comparison table with Published / HF / TT scores and TT/HF ratio.

```bash
# Default: ArguAna + STS-Benchmark, 100-sample subset, both HF and TT
MESH_DEVICE=P150 python models/demos/wormhole/qwen3_embedding_0_6b/demo/mteb_evaluation.py

# Full ArguAna dataset
MESH_DEVICE=P150 python .../mteb_evaluation.py --datasets mteb/ArguAna --max-samples 0

# Quick sanity check (20 samples)
MESH_DEVICE=P150 python .../mteb_evaluation.py --max-samples 20

# TT-only (skip HF reference for faster runs)
MESH_DEVICE=P150 python .../mteb_evaluation.py --skip-hf-reference

# Custom batch size and sequence length
MESH_DEVICE=P150 python .../mteb_evaluation.py --batch-size 1 --seq-len 1024
```

## Running Tracy profiling tests

These scripts run a single measured iteration with `tracy.signpost("start"/"stop")` markers for clean device-time capture.

```bash
# bs=1 Tracy profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs1_isl512.py -sv

# bs=8 Tracy profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs8_isl512.py -sv

# bs=32 Tracy profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs32_isl512.py -sv
```

Filter the resulting `ops_perf_results_*.csv` to ops between the `start` and `stop` signposts.

## Optimizations enabled

All demos and tests automatically apply these optimizations via `_common.apply_recommended_env()`:

| Env var | Effect |
|---|---|
| `QWEN_QKV_BFP4=1` | QKV projection weights in BFP4 |
| `QWEN_WO_BFP4=1` | Output projection weights in BFP4 |
| `QWEN_FF13_OUT_BFP8=1` | FF1/FF3 output activations in BFP8 |
| `QWEN_FFNORM_IN_BFP8=1` | FFN norm input in BFP8 |
| `QWEN_RESIDUAL_BFP8=1` | Post-FFN residual add in BFP8 |
| `QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1` | Split NlpCreateHeads by head group (16 -> 128 work units) |
| `QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1` | Split NlpConcatHeads by head group (16 -> 128 work units) |
| `QWEN_ROPE_PREFILL_L1=1` | Keep RoPE cos/sin tables in L1 |
| `QWEN_LN_BLOCK_SHARDED=1` | Block-sharded LayerNorm (active for 0.6B) |
| `TT_SKIP_KV_CACHE_FILL=1` | Skip KV cache fill for embedding (prefill-only) |
| `TT_BATCHED_L1_PREFILL=1` | L1-resident activations for batched prefill (bs > 1) |

All vars use `os.environ.setdefault` so you can override any single knob from the shell for A/B comparisons.
