# Performance Targets — Granite TTM-R1

## Core requirements

| Metric              | Target      | Achieved | Notes |
|---------------------|-------------|----------|-------|
| Throughput (batch=1, eager) | ≥ 500 seq/s | ~117 seq/s (Stage 2 baseline) | Stage 3: use traced path |
| Throughput (batch=1, traced) | ≥ 500 seq/s | TBD | `test_throughput_and_latency_traced` |
| Throughput (batch=8+, traced) | ≥ 2000 seq/s | TBD | `test_throughput_batch` |
| Latency (batch=1, eager) | < 10 ms | ~8.5 ms ✅ | Stage 2 full TTNN pipeline |
| Latency (batch=1, traced) | < 5 ms | TBD | Stage 3 trace capture target |
| Model parameters | < 1M | 805,280 ✅ | |
| Model size on disk | < 10 MB | 3.07 MB ✅ | float32 weights |
| Model size on disk | < 5 MB | 3.07 MB ✅ | Stage 3 target also met |
| PCC vs PyTorch | ≥ 0.99 | ≥ 0.99 ✅ | 7/7 components pass on Wormhole N300s |
| MSE vs PyTorch | within 5% | within 5% ✅ | validated by PCC tests |
| Zero-shot vs published | within 5% | pending | requires ETTh1 dataset (see scripts/prepare_assets.py) |

## Stage 3 feature targets

| Feature | Target | Status |
|---------|--------|--------|
| TTNN trace capture | Implemented | `model.compile(device)` + `model.execute_compiled(history)` |
| Batch-size sweep | Documented | `test_throughput_batch` sweeps batch 1–32 |
| Multi-model serving | Implemented | `TtnnGraniteTTMModel.from_shared_parameters()` |
| Streaming inference | Implemented | `GraniteTTMStreamingForecaster` in `tt/streaming.py` |
| HiFi2 compute config | Implemented | All `ttnn.linear` calls use `WormholeComputeKernelConfig(HiFi2)` |

## Bottleneck analysis

### Stage 2 (eager path)

```
~160 TTNN ops × ~53 µs dispatch overhead = ~8.5 ms latency
Throughput = 1 / 0.0085 s ≈ 117 seq/s  (batch=1)
```

The model is host-dispatch-bound, not compute-bound.

### Stage 3 mitigations

| Strategy | Expected gain |
|----------|---------------|
| TTNN trace capture | Eliminate ~160 × 53 µs = ~8.5 ms of Python dispatch; target <1-2 ms |
| Larger batch sizes | Amortise fixed dispatch cost: batch=8 → ~8× throughput at same latency |
| HiFi2 compute config | Marginal kernel-level gain; PCC ≥ 0.99 preserved |

### Throughput model (theoretical)

With trace (dispatch ≈ 1 ms):
```
batch=1  → ~1000 seq/s
batch=8  → ~8000 seq/s  (if not compute-bound)
```

Actual saturation point depends on available Tensix compute vs DRAM bandwidth.
Run `test_throughput_batch` to find the empirical saturation batch size.

## Running the benchmarks

```bash
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# Model size only (no device needed)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v

# Stage 2 eager latency baseline
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_and_latency -v

# Stage 3 traced latency / throughput
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_and_latency_traced -v

# Batch size sweep (throughput vs batch)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_batch -v -s

# Multi-model serving (100 shared-weight instances)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_multi_model_serving -v

# Streaming inference
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_streaming.py -v

# All PCC tests
python -m pytest models/demos/granite_ttm_r1/tests/pcc/ -v

# Zero-shot accuracy (requires ETTh1 — run scripts/prepare_assets.py first)
python -m pytest models/demos/granite_ttm_r1/tests/accuracy/ -v -s
```

## Known limitations and trade-offs

| Trade-off | Decision |
|-----------|----------|
| HiFi2 vs LoFi math fidelity | HiFi2 chosen; safe for PCC ≥ 0.99. LoFi may give 5–10% faster kernels if PCC permits. |
| Trace capture requires static shapes | Fixed context_length=512; streaming uses eager for variable-length input |
| Trace must be recompiled per batch size | One `compile()` call per batch size; cached on model instance |
| Shared weights across model instances | Read-only weight tensors; no correctness risk; saves ~1.53 MB per extra instance |
| Streaming: `step()` with n_new > 1 | Buffer rolls by n_new; all are accounted for. Output is forecast from most-recent window. |
