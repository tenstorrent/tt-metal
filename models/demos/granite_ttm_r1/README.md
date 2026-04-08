# Granite TTM-R1

## Overview

TTNN bring-up for [`ibm-granite/granite-timeseries-ttm-r1`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1),
a compact (805K-parameter) multivariate time-series forecasting model from IBM.

**Hardware:** Tenstorrent Wormhole (N150/N300)
**GitHub issue:** https://github.com/tenstorrent/tt-metal/issues/32142

## Architecture Summary

All components run natively on TTNN (no `TorchModuleFallback` in the forward path).

```
TinyTimeMixerForPrediction (805,280 params)
  backbone.scaler              # TinyTimeMixerStdScaler  [B,T,C] → [B,T,C]     TTNN ✓
  backbone.patching            # TinyTimeMixerPatchify   [B,512,1] → [B,1,8,64] TTNN ✓
  backbone.encoder.patcher     # Linear(64→192)                                 TTNN ✓
  backbone.encoder.mlp_mixer_encoder  # 3× AdaptivePatchingBlock                TTNN ✓
  decoder.adapter              # Linear(192→128)                                 TTNN ✓
  decoder.decoder_block        # 2× TinyTimeMixerLayer                          TTNN ✓
  head                         # Flatten + Linear(1024→96)                      TTNN ✓
```

Key dimensions: context_length=512, patch_length=64, num_patches=8,
d_model=192, decoder_d_model=128, forecast_length=96.

## Performance (Wormhole N300s)

| Mode | Latency | Throughput |
|------|---------|------------|
| Eager (batch=1) | ~8.5 ms | ~117 seq/s |
| Traced (batch=1) | **~1.82 ms** | **~550 seq/s** |
| Traced (batch=8) | ~2.4 ms | **~3290 seq/s** |
| Traced (batch=32) | ~3.7 ms | ~8760 seq/s |
| Traced (batch=64) | ~6.1 ms | **~10500 seq/s** |
| Traced (batch=64, bf8 weights) | ~5.7 ms | **~11260 seq/s** (peak) |

See [PERF.md](PERF.md) for the full throughput-vs-batch table and methodology.

## Layout

```
models/demos/granite_ttm_r1/
  common.py                    # HuggingFace model loading helpers
  requirements.txt             # Python dependencies
  PERF.md                      # Performance targets and achieved results
  reference/
    model.py                   # GraniteTTMReferenceModel wrapper
    preprocess.py              # build_reference_inputs, sliding_windows
    eval.py                    # pcc, mse, mae metrics
    model_summary.py           # Architecture inspection script
  tt/
    config.py                  # GraniteTTMModelConfig dataclass
    common.py                  # preprocess_parameters, get_linear_compute_config
    ttnn_granite_ttm_patching.py      # TTNN reshape+permute patchify
    ttnn_granite_ttm_embedding.py     # TTNN Linear (patch projection)
    ttnn_granite_ttm_time_mixer.py    # TTNN PatchMixerBlock (LN+MLP+gate)
    ttnn_granite_ttm_channel_mixer.py # TTNN FeatureMixerBlock (LN+MLP+gate)
    ttnn_granite_ttm_block.py         # TTNN TinyTimeMixerLayer (time+channel)
    ttnn_granite_ttm_adaptive_block.py # TTNN 3-level AdaptivePatchingBlock
    ttnn_granite_ttm_head.py          # TTNN head (flatten+linear)
    ttnn_granite_ttm_model.py         # Full model wiring + trace compile API
    streaming.py                      # GraniteTTMStreamingForecaster
  demo/
    demo.py                    # Benchmark demo: PCC, latency, throughput
  tests/
    conftest.py                # Shared device fixture
    pcc/
      test_pcc_patching.py
      test_pcc_embedding.py
      test_pcc_time_mixer.py
      test_pcc_channel_mixer.py
      test_pcc_block.py
      test_pcc_head.py
      test_pcc_full_model.py   # parametrized over batch_size=[1, 4, 8]
    perf/
      test_perf.py             # Eager + traced latency/throughput, batch sweep,
                               # multi-model serving (100 shared-weight instances)
      test_streaming.py        # Rolling-window streaming inference tests
    accuracy/
      test_accuracy_etthi.py   # Zero-shot ETTh1 benchmark (slow)
  scripts/
    prepare_assets.py          # Download ETTh1 and other datasets
```

## Setup

```bash
source python_env/bin/activate
uv pip install granite-tsfm

# Download ETTh1 for accuracy tests (optional)
python models/demos/granite_ttm_r1/scripts/prepare_assets.py --datasets etthi
```

## Running Tests

```bash
export PYTHONPATH=/root/tt/tt-metal

# All PCC tests (require Wormhole device)
pytest models/demos/granite_ttm_r1/tests/pcc/ -v

# Model size only (no device needed)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v

# Stage 3 traced latency / throughput (batch=1)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_and_latency_traced -v -s

# Throughput vs batch size sweep (batch=1,2,4,8,16,32,64)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_batch -v -s

# Double-buffered pipelined inference (2 command queues, batch=1)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_pipelined -v -s

# Multi-model serving (100 shared-weight instances)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_multi_model_serving -v

# Streaming inference tests
pytest models/demos/granite_ttm_r1/tests/perf/test_streaming.py -v

# Zero-shot accuracy (slow, requires ETTh1 data)
pytest models/demos/granite_ttm_r1/tests/accuracy/ -v -s
```

## Trace-Compiled Inference

For lowest latency, compile the model after construction:

```python
import ttnn
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel

device = ttnn.open_device(device_id=0)

# Build model (once)
model = TtnnGraniteTTMModel(parameters=parameters, config=model_config)
model.compile(device, batch_size=1)   # warm-up + trace capture

# Inference (fast path — single host command per call)
output = model.execute_compiled(history_tensor)

model.release_trace()
ttnn.close_device(device)
```

## Multi-Model Serving

Share a single pre-processed weight tree across many model instances:

```python
parameters = preprocess_parameters(hf_model, device)  # once

instances = [
    TtnnGraniteTTMModel.from_shared_parameters(parameters, model_config)
    for _ in range(100)
]
# All 100 instances share the same device weight tensors.
```

## Streaming Inference

For online (rolling-window) forecasting:

```python
from models.demos.granite_ttm_r1.tt.streaming import GraniteTTMStreamingForecaster

forecaster = GraniteTTMStreamingForecaster(model, model_config, device)

for new_obs in sensor_stream:          # new_obs: [n_new, num_channels]
    forecast = forecaster.step(new_obs)  # [forecast_len, num_channels]
```

## Stage 4 Experiments

Six optimisation experiments were run after Stage 3:

| Experiment | Result |
|---|---|
| E1: Zero-shot ETTh1 accuracy | TTNN MSE 0.4324 vs published 0.444 (2.6% below, all 7 channels) |
| E2: Pre-allocated host buffer | No gain — per-call allocation overhead negligible vs trace replay |
| E3: LoFi math fidelity | No gain — model is dispatch-bound; HiFi2 retained |
| E4: 2-CQ double-buffering | ~433 seq/s (xfail) — `synchronize_device` overhead exceeds H2D saving |
| E5: batch=64 sweep | **~5723 seq/s peak** — saturation at batch=64; batch=128 drops |
| E6: Streaming circular buffer | Eliminated `torch.roll` allocation; in-place indexed writes |

## Stage 5 Optimisations

Applied after reviewing TTNN model bringup guide and YOLOv4 tech report:

| Optimisation | Result |
|---|---|
| Double warmup + D2D input copy | ~1.82 ms / ~550 seq/s at batch=1 (was ~2.3 ms / ~440 seq/s) |
| Explicit `ttnn.deallocate()` for intermediates | Frees L1 earlier in scaler, time/channel mixers, de-normalisation |
| `TTNN_WEIGHT_BF8=1` (bfloat8_b weights) | **~11260 seq/s at batch=64** — 7% gain; PCC >= 0.99 holds |
| LoFi math fidelity | No gain — model dispatch-bound; available via `TTNN_LOFI=1` |
| Sharding | Not applicable — tensors ~3 KB, too small for multi-core benefit |

## Architecture Inspection

```bash
python -m models.demos.granite_ttm_r1.reference.model_summary
```
