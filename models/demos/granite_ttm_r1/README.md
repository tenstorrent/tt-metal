# Granite TTM-R1

## Overview

TTNN bring-up for [`ibm-granite/granite-timeseries-ttm-r1`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1),
a compact (805K-parameter) univariate time-series forecasting model from IBM.

**Hardware:** Tenstorrent Wormhole (N150/N300)
**GitHub issue:** https://github.com/tenstorrent/tt-metal/issues/32142

## Architecture Summary

```
TinyTimeMixerForPrediction (805,280 params)
  backbone.scaler              # TinyTimeMixerStdScaler  [B,T,C] → [B,T,C]
  backbone.patching            # TinyTimeMixerPatchify   [B,512,1] → [B,1,8,64]
  backbone.encoder.patcher     # Linear(64→192)          [B,1,8,192]     TTNN ✓
  backbone.encoder.mlp_mixer_encoder  # 3× AdaptivePatchingBlock    TorchFallback
  decoder.adapter              # Linear(192→128)                    TTNN ✓
  decoder.decoder_block        # 2× TinyTimeMixerLayer              TorchFallback
  head                         # Flatten + Linear(1024→96)          TTNN ✓
```

Key dimensions:
- context_length=512, patch_length=64, num_patches=8
- d_model=192, decoder_d_model=128, forecast_length=96

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
    common.py                  # preprocess_parameters, TTNN tensor utilities
    ttnn_granite_ttm_patching.py    # TorchModuleFallback (unfold op)
    ttnn_granite_ttm_embedding.py   # TTNN Linear (patch projection)
    ttnn_granite_ttm_time_mixer.py  # TTNN PatchMixerBlock
    ttnn_granite_ttm_channel_mixer.py # TTNN FeatureMixerBlock
    ttnn_granite_ttm_block.py       # TTNN TinyTimeMixerLayer (time+channel)
    ttnn_granite_ttm_head.py        # TTNN head (flatten+linear)
    ttnn_granite_ttm_model.py       # Full model wiring
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
      test_pcc_full_model.py
    perf/
      test_perf.py             # Throughput + latency + model size
    accuracy/
      test_accuracy_etthi.py   # Zero-shot ETTh1 benchmark (slow)
  scripts/
    prepare_assets.py          # Download ETTh1 and other datasets
```

## Setup

```bash
# Install dependencies (requires Python 3.10+ with TT-Metal env active)
pip install "granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git" \
    --ignore-requires-python --no-deps

# Download ETTh1 for accuracy tests (optional)
python models/demos/granite_ttm_r1/scripts/prepare_assets.py --datasets etthi
```

## Running Tests

```bash
# PCC tests (require Wormhole device)
pytest models/demos/granite_ttm_r1/tests/pcc/ -v

# Performance tests
pytest models/demos/granite_ttm_r1/tests/perf/ -v

# Model size test (no device needed)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v

# Zero-shot accuracy (slow, requires ETTh1 data)
pytest models/demos/granite_ttm_r1/tests/accuracy/ -v -m slow

# Demo
python -c "
import ttnn
from models.demos.granite_ttm_r1.demo.demo import run_granite_ttm_demo
device = ttnn.open_device(device_id=0)
result = run_granite_ttm_demo(device=device)
ttnn.close_device(device)
"
```

## Architecture Inspection

```bash
python -m models.demos.granite_ttm_r1.reference.model_summary
```
