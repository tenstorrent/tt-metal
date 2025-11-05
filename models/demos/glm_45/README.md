# GLM-4.5-Air: Mixture of Experts Language Model

Inference implementation for GLM models on Tenstorrent Wormhole accelerators.

**Model Source**: [GLM-4.5 Air on HuggingFace](https://huggingface.co/zai-org/GLM-4.5-Air) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy Quietbox** Single Blackhole (1x4 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)

**Current Status**: This model is under active development.
- ✅ Supported: batch size 32, high sequence length

## Quick Start

```bash
# Bump up transformers version
pip install -r models/demos/glm_45/requirements.txt

# Set model path using HF_MODEL environment variable
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/tt/GLM-4.5-Air"

# Run text generation demo on Galaxy (4×8 mesh)
cd tt-metal/models/demos/glm_45/demo
pytest text_demo.py -k "1x4"
```

## Testing

```bash
# Run all tests
pytest models/demos/glm_45/tests/unit/ -v

# Run specific test files
pytest models/demos/glm_45/tests/unit/test_submodules.py -v  # Utility components
pytest models/demos/glm_45/tests/unit/test_modules.py -v     # Core components
pytest models/demos/glm_45/tests/unit/test_model.py -v       # Full model accuracy (currently failing with accuracy=0)
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_submodules.py`** | Utility components | • RoPE embeddings<br>• Scaled Dot Product Attention (SDPA) |
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison (currently failing with accuracy=0)|
