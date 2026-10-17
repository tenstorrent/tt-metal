# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole and Blackhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)
- **QuietBox 2**: 4 Blackhole devices (1×4 configuration); batch-1 decode uses the fused decode path ([tt/fused_decode](tt/fused_decode/README.md))

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

# Run text generation demo on QuietBox 2 (1×4 mesh)
pytest text_demo.py -k "1x4 and prefill_128"
```

## DFlash speculative decoding (QuietBox 2, gpt-oss-20b)

Greedy batch-1 decode with the [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) drafter
(downloaded from HuggingFace on first use). Each round, the drafter proposes 7 tokens from the target's hidden states;
the target verifies the 8-row block [last token, 7 drafts] in one pass and keeps the drafts up to the first mismatch,
plus its own next token. The output is the target's greedy output (up to BF16 rounding of near-ties).

```bash
export HF_MODEL=openai/gpt-oss-20b
pytest models/demos/gpt_oss/demo/dflash_demo.py -k "isl_1k"   # isl_256, isl_512, isl_1k, ... isl_64k, isl_128k
pytest models/demos/gpt_oss/tests/test_dflash.py               # accuracy vs. the target's own greedy decode
```

All three passes of a round run as device traces, and the host reads one small result per round:
- draft pass: [tt/dflash/fused_draft.py](tt/dflash/fused_draft.py);
- 8-row verify pass: [tt/dflash/verify.py](tt/dflash/verify.py);
- context pass (adds the accepted tokens' target features to the drafter's K / V cache).

The drafter attends to the last 512 positions.

| Prompt length (tokens) | Tokens per round | Round (ms) | DFlash (tok/s/user) |
|---|---|---|---|
| 256 | 4.03 | 4.928 | 799.1 |
| 512 | 2.90 | 4.833 | 602.0 |
| 1,024 | 3.36 | 4.980 | 676.4 |
| 2,048 | 3.39 | 5.042 | 668.1 |
| 4,096 | 3.34 | 5.021 | 662.2 |
| 8,192 | 3.28 | 5.093 | 644.4 |
| 16,384 | 3.27 | 5.243 | 626.0 |
| 32,768 | 3.59 | 5.549 | 649.8 |
| 65,536 | 4.20 | 6.186 | 678.4 |
| 130,752 | 3.41 | 7.259 | 470.2 |

Measured with the demo above (one `-k isl_*` case per row: the chat-template Gutenberg prompt, 256 new tokens). For
comparison, plain batch-1 decode (`pytest models/demos/gpt_oss/demo/text_demo.py -k "1x4 and prefill_128"`) runs at
2.37 ms per token (421 tok/s/user) on the same build.

| File | Role |
|---|---|
| `tt/dflash/runner.py` | `DFlashRunner`: prefill, trace capture, the round loop |
| `tt/dflash/draft.py` | the drafter's weights, K / V cache, prompt context and per-round inputs |
| `tt/dflash/fused_draft.py` | the draft pass and the context pass |
| `tt/dflash/verify.py` | the 8-row verify pass of the target |
| `tt/dflash/qk_prep.py`, `kv_write.py` | the drafter's q / k norm + RoPE, its K / V cache write |
| `tt/dflash/row_attention.py`, `block_attention.py` | verify attention: sliding-window layers, full-attention layers; drafter attention |
| `tt/dflash/moe_fused.py` | the verify MoE and the drafter MLP as one program per device |
| `tt/dflash/lm_head.py`, `lm_rescore.py`, `argmax.py` | the two-stage LM head and the on-device acceptance |
| `tt/dflash/capture.py` | the target's prefill features for the drafter |
| `reference/dflash.py` | torch reference of the drafter and of DFlash greedy decoding |

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
| **`../test_dflash.py`** (`pytest models/demos/gpt_oss/tests/test_dflash.py`) | DFlash speculative decoding (QuietBox 2) | • Teacher-forced top-1 agreement with the target's greedy decode<br>• Mean accepted tokens per round |

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
