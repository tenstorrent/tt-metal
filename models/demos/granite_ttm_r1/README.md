# Granite Timeseries TTM-R1 (Tiny Time Mixer)

## Overview

TTNN bring-up for [`ibm-granite/granite-timeseries-ttm-r1`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1),
a compact (805K-parameter) multivariate time-series forecasting model from IBM.

**Target Hardware:**
* Tenstorrent Wormhole (N300s)

## Architecture Summary

All components run natively on TTNN.

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
      test_accuracy_multi_dataset.py  # Zero-shot ETTh2/ETTm1/ETTm2 (slow)
  scripts/
    prepare_assets.py          # Download ETTh1 and other datasets
```

## Setup
```bash
cd tt-metal/
export PYTHONPATH=`pwd`
source python_env/bin/activate

uv pip install granite-tsfm
```

## Running Tests
```bash
# All PCC tests (require Wormhole device)
pytest models/demos/granite_ttm_r1/tests/pcc/ -v -s

# Model size only (no device needed)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v -s

# Stage 3 traced latency / throughput (batch=1)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_and_latency_traced -v -s

# Throughput vs batch size sweep (batch=1,2,4,8,16,32,64)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_batch -v -s

# Double-buffered pipelined inference (2 command queues, batch=1)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_throughput_pipelined -v -s

# Multi-model serving (100 shared-weight instances)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_multi_model_serving -v -s

# Streaming inference tests
pytest models/demos/granite_ttm_r1/tests/perf/test_streaming.py -v -s
```

### Accuracy Tests (using Multiple External Datasets)
```bash
# Download datasets for accuracy tests (depended on by the accuracy test below)
python models/demos/granite_ttm_r1/scripts/prepare_assets.py --datasets etthi etth2 ettm1 ettm2

# Zero-shot accuracy (slow, requires datasets — run scripts/prepare_assets.py first)
pytest models/demos/granite_ttm_r1/tests/accuracy/ -v -s
```

## Architecture Inspection
```bash
python -m models.demos.granite_ttm_r1.reference.model_summary
```
