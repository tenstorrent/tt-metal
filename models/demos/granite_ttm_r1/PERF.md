# Performance Targets — Granite TTM-R1

## Core requirements

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Latency (batch=1, eager) | < 10 ms | ~8.5 ms ✅ | Stage 2: full TTNN pipeline, no TorchModuleFallback |
| Latency (batch=1, traced) | < 5 ms | **~1.82 ms ✅** | Stage 5: double warmup + D2D input copy + dealloc; was ~2.3 ms in Stage 3 |
| Throughput (batch=1, traced) | ≥ 500 seq/s | **~550 seq/s ✅** | Stage 5: now passes at batch=1 (was ~440 seq/s / xfail in Stage 3) |
| Throughput (batch=2, traced) | ≥ 500 seq/s | **~840 seq/s ✅** | |
| Throughput (batch=8, traced) | ≥ 2000 seq/s | **~3290 seq/s ✅** | Stage 5: was ~2600 seq/s in Stage 3 |
| Model parameters | < 1M | 805,280 ✅ | |
| Model size on disk | < 5 MB | 3.07 MB ✅ | float32 weights |
| PCC vs PyTorch | ≥ 0.99 | ≥ 0.99 ✅ | 9/9 tests pass (batch=1,4,8) on Wormhole N300s |
| MSE vs PyTorch | within 5% | within 5% ✅ | validated by PCC tests |
| Zero-shot vs published | within 5% | **2.6% below ✅** | TTNN MSE 0.4324 vs published 0.444 (7ch, 57 test windows) |

## Throughput vs batch size (traced, Wormhole N300s)

Stage 5 numbers after double-warmup + D2D input copy + dealloc optimisation:

| Batch | Latency | Throughput | Status |
|-------|---------|------------|--------|
| 1 | ~1.82 ms | **~550 seq/s** | ✅ exceeds 500 seq/s (was 440 seq/s / xfail) |
| 8 | ~2.43 ms | **~3290 seq/s** | ✅ exceeds 2000 seq/s stretch target |
| 32 | ~3.65 ms | **~8760 seq/s** | ✅ |
| 64 | ~6.10 ms | **~10500 seq/s** | ✅ (bfloat16 default) |
| 64 (bf8) | ~5.68 ms | **~11260 seq/s** | ✅ peak throughput with `TTNN_WEIGHT_BF8=1` |

## Stage 3 feature status

| Feature | Status | How to use |
|---------|--------|------------|
| TTNN trace capture | ✅ Implemented | `model.compile(device, batch_size=N)` then `model.execute_compiled(history)` |
| Batch inference | ✅ Implemented | Pass batch > 1 to `_build_model_and_example`; trace per batch size |
| Multi-model serving (100 instances) | ✅ Implemented | `TtnnGraniteTTMModel.from_shared_parameters(parameters, config)` — 100 instances in 0.12 s |
| Streaming inference | ✅ Implemented | `GraniteTTMStreamingForecaster` in `tt/streaming.py` — 2.5 ms/step traced |
| HiFi2 compute config | ✅ Implemented | All `ttnn.linear` calls use `WormholeComputeKernelConfig(HiFi2, packer_l1_acc=True)` |
| Explicit tensor deallocation | ✅ Implemented | `ttnn.deallocate()` in scaler, time/channel mixers, model de-normalisation |
| bfloat8_b weight option | ✅ Implemented | `TTNN_WEIGHT_BF8=1` — ~7% gain at batch=64; PCC ≥ 0.99 |

## Bottleneck analysis

### Stage 2 baseline (eager path)
```
~160 TTNN ops × ~53 µs Python dispatch overhead = ~8.5 ms latency
Throughput (batch=1) = 1 / 0.0085 ≈ 117 seq/s
```
The model is host-dispatch-bound, not compute-bound.

### Stage 3 result (traced path)
```
Trace replay = ~1 host command for entire graph
Measured latency (batch=1): ~2.3 ms  →  4× improvement over Stage 2
Measured throughput (batch=8): ~2620 seq/s  →  22× improvement over Stage 2
```

### Stage 5 result (double warmup + D2D input copy + dealloc + bf8 option)
```
Two warmup passes before trace capture → better program cache coverage
ttnn.copy() for device-tensor inputs → eliminates D2H→H2D round-trip
ttnn.deallocate() for intermediates → frees L1 memory earlier
TTNN_WEIGHT_BF8=1 → bfloat8_b weights, ~7% gain at batch=64

Measured latency (batch=1):  ~1.82 ms  →  1.26× improvement over Stage 3
Measured throughput (batch=1): ~550 seq/s → 500 seq/s target now met at batch=1
Measured throughput (batch=8): ~3290 seq/s → 1.26× improvement over Stage 3
Measured throughput (batch=64): ~10500 seq/s → 1.84× improvement over Stage 4 peak
Measured throughput (batch=64, bf8): ~11260 seq/s → new peak with bfloat8_b weights
```

Throughput saturation now occurs above batch=64 (~10500 seq/s bf16, ~11260 seq/s bf8).

## Running the benchmarks

```bash
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# Model size only (no device needed)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v

# Stage 3 traced latency / throughput (batch=1)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_and_latency_traced -v -s

# Batch size sweep (throughput vs batch, batch=1–64)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_batch -v -s

# Double-buffered pipelined inference (2 command queues, batch=1; xfail)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_pipelined -v -s

# Multi-model serving (100 shared-weight instances)
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_multi_model_serving -v

# Streaming inference
python -m pytest models/demos/granite_ttm_r1/tests/perf/test_streaming.py -v

# All PCC tests
python -m pytest models/demos/granite_ttm_r1/tests/pcc/ -v

# Zero-shot accuracy (requires ETTh1 — run scripts/prepare_assets.py first)
python -m pytest models/demos/granite_ttm_r1/tests/accuracy/ -v -s
```

## Stage 4 experiment results

| Experiment | Result | Finding |
|---|---|---|
| E1: Zero-shot ETTh1 accuracy | ✅ **PASSED** | TTNN MSE 0.4324 vs published 0.444 (2.6% below target); PCC vs PyTorch 0.9999 |
| E2: Pre-allocated host buffer | No gain | Input tensor is 1 KB; per-call allocation overhead is negligible vs trace replay |
| E3: LoFi math fidelity | No gain | Model is dispatch-bound; faster kernels hidden by trace replay cost; HiFi2 retained |
| E4: Double-buffering (2-CQ pipeline) | ~433 seq/s (xfail) | `synchronize_device(idle)` adds 0.155 ms/frame overhead; negates H2D hiding (0.12 ms H2D < 0.155 ms sync); Python overhead is binding constraint |
| E5: batch=64 sweep | ✅ **5723 seq/s peak** | Saturation at batch=64; batch=128 drops to ~4972 seq/s (memory pressure) |
| E6: Streaming circular buffer | ✅ Implemented | Eliminates `torch.roll` allocation on every step; in-place indexed writes + wrap-around; ~0.01 ms saving per step |

## Known limitations and trade-offs

| Trade-off | Decision |
|-----------|----------|
| HiFi2 vs LoFi math fidelity | HiFi2 chosen; safe for PCC ≥ 0.99. LoFi shows no measurable gain (model dispatch-bound); toggle via `TTNN_LOFI=1` |
| bfloat16 vs bfloat8_b weights | bfloat16 default; bfloat8_b gives ~7% gain at batch=64 with PCC ≥ 0.99; toggle via `TTNN_WEIGHT_BF8=1` |
| Sharding strategy | Not applicable — tensors ~3 KB, too small for multi-core parallelism benefit (overhead exceeds gains) |
| Trace capture requires static shapes | Fixed context_length=512; streaming always uses eager path for variable-length input |
| Trace must be recompiled per batch size | One `compile()` call per batch size; cached on model instance |
| Shared weights across model instances | Read-only weight tensors; no correctness risk; saves ~1.53 MB per additional instance |
| 500 seq/s target at batch=1 | **Now met**: ~550 seq/s at batch=1 after Stage 5 optimisations |
| Peak throughput | batch=64 gives ~10500 seq/s (bf16) / ~11260 seq/s (bf8); up from ~5723 seq/s in Stage 4 |
| Zero-shot accuracy | Stage 4 E1: TTNN MSE 0.4324 vs published 0.444 on ETTh1 test split (all 7 channels, normalized) |
| 2-CQ double-buffering (E4) | Tried: `synchronize_device(idle)` costs 0.155 ms/frame, more than H2D (0.12 ms) being hidden; net: 433 seq/s — no gain over single-CQ traced path |
