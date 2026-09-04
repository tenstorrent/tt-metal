# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)
- **Blackhole**: P150 (1×1), P300 / QuietBox 2 (1×2, 1×4), P150x8 (1×8)

**Current Status**: This model is under active development.
- ✅ Supported: batch size 1 on all meshes; batch 128 (32 users per mesh row, `users_row_sharded`) on Wormhole Galaxy
- ✅ Supported: batch size 32 on single-row 1×8 meshes (TP=8, EP=1), e.g. 8× Blackhole P150 — see *Multi-user decode on single-row meshes*
- 🚧 In Progress: Extended sequence lengths, batched (multi-user) prefill on single-row meshes

## Multi-user decode on single-row meshes

On a 1×8 mesh every device holds a 1/8 slice (TP) of every expert and there is no mesh axis to
expert-parallelize over, so the Galaxy "throughput experts" (`all_to_all` dispatch/combine +
fused `moe_gpt`, which also only exist for Wormhole) cannot be used. Instead the low-latency
experts (`tt/experts/decode.py`) process all users of a decode step as one 32-row tile:

1. the dense per-user routing weights `[users, 128]` are summed over users on device into a
   *union-of-active-experts* mask `[1, 1, 1, 128]`;
2. the fused gate+up projection and the down projection run as `ttnn.sparse_matmul` over that
   mask (`nnz` inferred on device);
3. the per-(user, expert) routing weights zero every unselected pair, the result is reduced over
   experts and the down bias is folded in through the routing weights, followed by the usual TP
   all-reduce.

The cost of `sparse_matmul` is dominated by a fixed per-active-expert overhead (~4 µs per call
per expert on P150), so the number of calls matters more than their width: gate and up are
therefore stored fused (`[gate | up]` per device, each half zero-padded to a tile multiple) and
computed in one call. The activation footprint stays identical to batch 1 (M is a single 32-row
tile either way).

Constraints: users are not row-sharded; the decode batch must equal `max_batch_size` (pad unused
slots); on Blackhole `max_batch_size` must be ≤ 8 or a multiple of 32 so that the per-user
Q/K/V shard grid (8 wide) agrees with the RoPE cos/sin placement; prefill of the 32 users runs
sequentially through the generator (one user per forward).

```bash
export HF_MODEL=/path/to/gpt-oss-120b
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x8 and batch32"
pytest models/demos/gpt_oss/tests/unit/test_modules.py -k "1x8 and decode_b32"
# cross-user isolation check with real weights (teacher-forced logits across slot layouts)
pytest models/demos/gpt_oss/tests/test_multi_user_consistency.py -k 1x8
```

Measured on 8× Blackhole P150 (1×8, greedy, ~128-token prompts, 200 generated tokens):

| Model | Batch | Decode step | Per user | Aggregate | TTFT per user |
|---|---|---|---|---|---|
| gpt-oss-120b | 1 | 36.3 ms | 27.5 tok/s | 27.5 tok/s | 407 ms |
| gpt-oss-120b | 32 | 49.0 ms | 20.4 tok/s | 653 tok/s | 399 ms |
| gpt-oss-20b | 1 | 18.9 ms | 52.9 tok/s | 52.9 tok/s | 125 ms |
| gpt-oss-20b | 32 | 20.8 ms | 48.1 tok/s | 1538 tok/s | 85 ms |

TTFT at batch 32 is the per-user share of 32 sequential prefills (one user per forward); packing
several users into one prefill forward is the main open item for multi-user TTFT.

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
