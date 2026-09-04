# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)
- **Blackhole**: P150 (1×1), P300 / QuietBox 2 (1×2, 1×4), P150x8 (1×8)

**Current Status**: This model is under active development.
- ✅ Supported: batch size 1 on all meshes; batch 128 (32 users per mesh row, `users_row_sharded`) on Wormhole Galaxy
- ✅ Supported: any batch size up to 32 on single-row 1×8 meshes (TP=8, EP=1), e.g. 8× Blackhole P150 — see *Multi-user decode on single-row meshes*
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
slots); any `max_batch_size` ≤ 32 works — the per-user Q/K/V / KV-cache / SDPA core placement
(`ProgramConfig.get_decode_user_grid`) follows RotarySetup's cos/sin placement and the model checks
the two agree at construction; prefill of the users runs sequentially through the generator (one
user per forward).

Prefill runs each 32-token tile through the experts routed to at least one of its tokens
(`experts/prefill.py`, routing-aware `sparse_matmul` mask) rather than through every expert, which
is worth ~1.3× on gpt-oss-120b prefill/TTFT; grouping tokens per expert (one GEMM per expert with
variable M, as the Galaxy throughput path does) is the remaining lever.

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

### Multi-user regression sweep

`tests/test_multi_user_regression.py` runs the tt-inference-server benchmark ISL/OSL pairs
(128/128, 128/1024, 1024/128, 2048/128, 4096/128, 8192/128, 8192/1024, 16384/128, 32768/128) for
batch {1, 2, 4, 8, 16, 32} with a per-user context of min(64K, 512K/B) (pairs above it are skipped,
as the server caps concurrency). Every cell is gated on token ids, on every user's first token being
`<|channel|>`, on QA keyword accuracy at ISL 128 (32 distinct prompts) and on a degeneracy
heuristic; it records prefill, TTFT (first / mean / last user), decode step mean/p50/p99, tok/s
and board telemetry to `generated/gpt_oss_multi_user_regression/<model>_<mesh>.jsonl`.

```bash
export HF_MODEL=/path/to/gpt-oss-120b TT_CACHE_PATH=/path/to/cache/gpt-oss-120b
# one process per batch size (function-scoped mesh device; pytest -k "batch1" also matches batch16)
pytest "models/demos/gpt_oss/tests/test_multi_user_regression.py::test_multi_user_regression[blackhole-1x8-batch32]" \
    --timeout 3600 --timeout-method thread
```

Knobs (env): `GPT_OSS_REGRESSION_PAIRS="128:128,1024:128"`, `_KV_TOKENS` (total KV budget, default
512K), `_PAGE_TABLE_SEED`, `_DECODE_TRACE=0`, `_DECODE_K_CHUNK`, `_TAG` (results file suffix) and
`_COOLDOWN_C` / `_COOLDOWN_TIMEOUT_S`: waits for all boards to cool below the threshold before each
prefill and before the first decode step. On the 8× P150 box the hottest board reaches ~88 °C during
long prefills and throttles to 800 MHz, and a traced decode step launched while a board is
deep-throttled has deadlocked inside paged SDPA decode; `GPT_OSS_REGRESSION_COOLDOWN_C=84` avoided
that for the whole sweep. All prefill lengths are compiled eagerly before the first trace is
captured, because programs compiled while a trace is live can be overwritten by a later replay.

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
