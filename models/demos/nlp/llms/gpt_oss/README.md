# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)

**Current Status**: This model is under active development.
- ✅ Supported: Prefill up to sequence length 128, batch size 1, total sequence length 4096
- 🚧 In Progress: Extended sequence lengths, larger batch sizes

## Quick Start

```bash
# Bump up transformers version
pip install -r models/demos/nlp/llms/gpt_oss/requirements.txt

# Set model path using HF_MODEL environment variable
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# Run text generation demo on Galaxy (4×8 mesh)
cd tt-metal/models/demos/nlp/llms/gpt_oss/demo
pytest text_demo.py -k "4x8"
```

## Configuration

### Model Selection
```bash
# GPT-OSS-20B (faster, recommended for development)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# GPT-OSS-120B (higher quality, requires more memory)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-120b"
```

## Testing

```bash
# Run all tests
pytest models/demos/nlp/llms/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/demos/nlp/llms/gpt_oss/tests/unit/test_submodules.py -v  # Utility components
pytest models/demos/nlp/llms/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/demos/nlp/llms/gpt_oss/tests/unit/test_model.py -v       # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_submodules.py`** | Utility components | • RoPE embeddings<br>• Scaled Dot Product Attention (SDPA) |
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |
