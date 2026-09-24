# Qwen3-Embedding-4B on Tenstorrent

Optimized inference of [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) on Tenstorrent Blackhole (P150) hardware.

The demo, perf tests and MTEB script here run through the optimized
[`pplx_embed_4b`](../pplx_embed_4b/README.md) stack: Qwen3-Embedding-4B is the backbone pplx-embed-v1-4B was
trained from, so the same tuned code path serves both. `demo/_common.py` sets `HF_MODEL=Qwen/Qwen3-Embedding-4B`
and the stack switches to causal attention and last-token pooling from that. Performance from baseline to now:
[`../pplx_embed_4b/PERF.md`](../pplx_embed_4b/PERF.md).

## Model overview

| Property | Value |
|---|---|
| Parameters | 4B |
| Hidden size | 2560 |
| Layers | 36 |
| Attention heads (Q / KV) | 32 / 8 |
| Intermediate size | 9728 |
| Head dim | 128 |
| GQA ratio | 4:1 |

## Directory structure

```
qwen3_embedding_4b/
├── demo/
│   ├── _common.py                 # Shared optimization knobs, model builder, perf harness
│   ├── demo_bs1_isl512.py         # Perf demo — batch=1, ISL=512
│   ├── demo_bs32_isl512.py        # Perf demo — batch=32, ISL=512
│   └── mteb_evaluation.py         # MTEB accuracy eval (TT vs HF reference)
└── tests/
    └── perf/
        ├── new_perf_bs1_isl512.py   # Tracy profiling — batch=1, ISL=512
        └── new_perf_bs32_isl512.py  # Tracy profiling — batch=32, ISL=512
```

## Prerequisites

```bash
# Build tt-metal and activate the virtual environment
source python_env/bin/activate
export MESH_DEVICE=P150
```

## Running demo files

The demo scripts time the extended trace (forward + pooling + I/O in one replay) over 10 iterations with the
tuned per-batch defaults applied by `apply_workload_env`. Each can be run via pytest or as a standalone script.

### Batch size 1

Activation = 512 x 2560 x 2 = 2.5 MB -- L1-resident; legacy 2D-multicast matmuls on a 12x8 grid.

```bash
# Via pytest
pytest models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py -sv

# Standalone
MESH_DEVICE=P150 python models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py
```

### Batch size 32

Activation = 32 x 512 x 2560 x 2 = 80 MB -- DRAM-resident; `minimal_matmul` on the full worker grid (12x10 on a
Galaxy P150, 13x10 on a p150a card).

```bash
# Via pytest
pytest models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs32_isl512.py -sv

# Standalone
MESH_DEVICE=P150 python models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs32_isl512.py
```

### Common options (standalone mode)

```bash
python .../demo_bs1_isl512.py --device-id 0 --iterations 20
python .../demo_bs1_isl512.py --no-full-pipeline      # time only the bare forward replay
```

On a multi-chip host pick the chip with `TT_VISIBLE_DEVICES=<n>` (the process then sees it as device 0).

## Running MTEB evaluation

The evaluation script runs both the HuggingFace reference model and the TT model on the same MTEB datasets (ArguAna retrieval + STS-Benchmark by default), then displays a comparison table with Published / HF / TT scores and TT/HF ratio.

```bash
# Default: ArguAna + STS-Benchmark, 100-sample subset, both HF and TT
MESH_DEVICE=P150 python models/demos/blackhole/qwen3_embedding_4b/demo/mteb_evaluation.py

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
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/blackhole/qwen3_embedding_4b/tests/perf/new_perf_bs1_isl512.py -sv

# bs=32 Tracy profile
MESH_DEVICE=P150 \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  python -m tracy -p -r -v -m pytest \
  models/demos/blackhole/qwen3_embedding_4b/tests/perf/new_perf_bs32_isl512.py -sv
```

Filter the resulting `ops_perf_results_*.csv` to the ops between the `start` and `stop` signposts, in file
order (trace-replayed ops keep their capture-time host timestamps). The 40000 program budget is required or
the batched-shape ops get no device data.

## Performance (P150, sustained)

Sustained latency at ISL 512 (median of iterations 15–29 of a 30-iteration run; the board settles its clock at
≈1.1–1.3 GHz under load), one Galaxy P150 (12x10 = 120 worker cores), measured 2026-09-24:

| batch | Qwen3-Embedding-4B | pplx-embed-4B (same stack) | H200 reference | × H200 |
|---|---|---|---|---|
| 1 | **18.4 ms** | 17.6 ms | 5.44 ms | 3.4× |
| 8 | **120.8** | 120.9 | 33.08 | 3.7× |
| 16 | **228.2** | 227.6 | 67.23 | 3.4× |
| 32 | **445.1** | 446.4 | 139.15 | 3.2× |

The previous demo in this directory measured 32.3 ms at bs=1 and 725 ms at bs=32. STS-B Spearman through the
batched paths (last token + EOS): 0.819 / 0.810 / 0.808 / 0.807 at bs 1 / 8 / 16 / 32
(`../pplx_embed_4b/demo/eval_accuracy_batched.py --pool last --eos`). The H200 reference is the pplx-embed-4B
measurement; the compute is identical.

## Optimizations enabled

Everything comes from the shared stack: `../pplx_embed_4b/demo/_common.py::apply_workload_env(batch, seq)` sets
the tuned defaults per batch size (all `os.environ.setdefault`, so any knob can be overridden from the shell for
an A/B), and the model-local kernels live in `../pplx_embed_4b/tt/custom_ops/`. In short: bfp4 weights with
LoFi matmuls (legacy 2D multicast on 12x8 at bs1, `minimal_matmul` on 120 cores at bs≥8), a fused head-split +
Q/K RMSNorm + RoPE op emitting bfp8 Q/K/V, SDPA on the streaming kernel writing the concatenated-heads layout
directly, a fused residual add + RMSNorm, a fused SwiGLU product, and DRAM-interleaved weights at bs32. The
Qwen3-specific switch is `QWEN_SDPA_CAUSAL=1`, set automatically for a non-pplx `HF_MODEL`. The full list with
the effect of each change is in [`../pplx_embed_4b/PERF.md`](../pplx_embed_4b/PERF.md).
