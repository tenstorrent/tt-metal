# GPT-OSS: Mixture of Experts Language Model (Quasar)

> Quasar-specific fork of `models/demos/gpt_oss`, kept separate so it can be adapted to
> the emulator's compute-grid and SRAM constraints. All module paths are rooted at
> `models.experimental.ops.quasar.gpt_oss`, and every path in this README is relative to
> the repo root. Do not run pytest over both this tree and `models/demos/gpt_oss` in a
> single session: their conftests register the same options (`--skip-model-load`,
> `--test-modules`) and pytest errors on the duplicates.

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

# Run text generation demo on Galaxy (4×8 mesh), from the repo root -- the demo's
# sample-prompt paths are repo-root-relative, so running from the demo dir cannot find them
pytest models/experimental/ops/quasar/gpt_oss/demo/text_demo.py -k "4x8 and prefill_128"
```

## Quasar bring-up knob

`QSR_N_LAYERS` (this fork only) builds just the first N decoder layers. Unset, the model is
whole and the demo behaves exactly as upstream.

```bash
QSR_N_LAYERS=2 MESH_DEVICE=N150 HF_MODEL=... \
    pytest models/experimental/ops/quasar/gpt_oss/demo/text_demo.py -k "prefill_128 and 1x1"
```

Why it exists: each layer's experts cost ~448 MB (`down_proj` is 32 x 2880 x 2880 at bf4 =
149 MB, `gate_up_proj` twice that), so the full 24-layer gpt-oss-20b needs ~10.7 GB of expert
weights before attention, embeddings or KV cache — a single device's ~12.75 GB of DRAM runs out
on the last layer. 2-4 layers leaves room to actually run something.

Generated text is meaningless with a truncated model; this is for getting ops onto the device.
The weight cache records the layer count (`tt/model_config.py:349`), so a partial build never
satisfies the full-model completeness check and cannot poison a later full run.

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
pytest models/experimental/ops/quasar/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/experimental/ops/quasar/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/experimental/ops/quasar/gpt_oss/tests/accuracy/test_model.py -v   # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |
