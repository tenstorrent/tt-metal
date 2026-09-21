# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole and Blackhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole or Blackhole mesh (4×8 configuration)
- **Blackhole single card / small mesh**: P150 (1×1), DeskBox (1×2), LLMBox / QuietBox 2 (1×4)

**Current Status**: This model is under active development.
- ✅ Supported: Prefill up to sequence length 128, total sequence length 4096; batch 1 on any
  supported mesh, and batch 128 decode on a 4×8 Galaxy (Wormhole or Blackhole)
- 🚧 In Progress: Extended sequence lengths

### MoE expert paths

Row-sharded batch decode runs the *throughput experts* path. It has two flows:

| Flow | Pipeline | Used on |
|------|----------|---------|
| Fused | `all_to_all_dispatch_metadata` → `moe_gpt` → `selective_reduce_combine` | Wormhole |
| Dense | `all_to_all_dispatch` → `matmul` → `all_to_all_combine` → `all_reduce` | Blackhole |

Both compute the same thing. The fused kernels shard K across the DRAM-bank-aligned matmul
cores and assume Wormhole's 12 DRAM banks (`moe_gpt`'s `tiles_per_core_table[12]` sums to
90 = 2880/32 only at 12 banks; `topk_router_gpt` `TT_FATAL`s below 12). Blackhole has 8, so
it takes the dense flow — selected automatically by
`fused_moe_kernels_supported_on_arch()` in `utils/general_utils.py`. Porting the fused path
to Blackhole means moving it onto `ttnn.experimental.moe_compute`, which derives its ring
size from the live DRAM-bank count.

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
