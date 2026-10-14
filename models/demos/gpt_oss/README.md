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
# Set model path using HF_MODEL environment variable
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# Run text generation demo on Galaxy (4×8 mesh)
cd tt-metal/models/demos/gpt_oss/demo
pytest text_demo.py -k "4x8 and prefill_128"
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
pytest models/demos/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/demos/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/demos/gpt_oss/tests/unit/test_model.py -v       # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |

### Host sampling readback

GPT-OSS declares `supports_compact_host_logits` and
`supports_selective_host_readback` for the vLLM plugin. The plugin supplies
`sample_rows` in scheduled request order. Host processing preserves this order
and returns independently owned FP32 logits without vocabulary padding.

Model construction prepares the selective readback programs before any trace
is captured. Reads use aligned power-of-two slot ranges, clipped at the batch
boundary. For 32 slots per device, there are 62 slice programs and five reusable
device buffers. The buffers contain 31 rows in total. A full-range read uses
the original decode output directly. Some scattered requests transfer extra
rows because their enclosing range is larger than the exact active interval.

Serving reads use only these prepared programs and buffers. They reject an
input specification that differs from the prepared row-major BF16 DRAM logits.
Keep the program cache enabled throughout model construction and serving.
Slices and transfers use command queue 0, so each transfer completes before
a later slice can overwrite its buffer. Each read has a separate host tensor.
Consumers must wait on all returned events before processing host output.
Inactive generator ranks have no transfer event; the event list is not indexed
by rank. Calls without `sample_rows` retain the full readback path.
