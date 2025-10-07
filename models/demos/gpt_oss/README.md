# GPT-OSS: batch=1 inference

Inference for GPT-OSS models on Tenstorrent Wormhole devices LoudBox and Galaxy.
This model is under active development.
Currently we have only support prefill upto sequence length 128 and batch=1.

## Quick Start

```bash
# Set model path
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"

# Run demo
cd tt-metal/models/demos/gpt_oss/demo
pytest simple_text_demo.py -k "4x8"
```

## Configuration

### Model Selection
```bash
# GPT-OSS-20B (faster)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"

# GPT-OSS-120B (higher quality)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-120B"
```

## Testing

```bash
# Run all tests
pytest models/demos/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/demos/gpt_oss/tests/unit/test_submodules.py -v  # Utility components
pytest models/demos/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/demos/gpt_oss/tests/unit/test_model.py -v       # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_submodules.py`** | Utility components | • RoPE embeddings<br>• Scaled Dot Product Attention (SDPA) |
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |
