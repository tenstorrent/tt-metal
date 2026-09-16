# Gemma4 31B on QB2

Text generation for `google/gemma-4-31B-it` on one Blackhole QuietBox 2: four chips, tensor parallelism 4, and up to 32 concurrent requests. This implementation specializes the dense 31B decoder for QB2. The broader Gemma4 family implementation remains in [`../gemma4`](../gemma4).

## Design

The 60 decoder layers alternate five sliding-attention layers with one full-attention layer. Each chip holds a quarter of the projection weights. A ring fabric carries tensor-parallel reductions; completed sliding-attention sums retain BF16 precision. Decode replays a device trace with resident token, position, and sampling state. Scheduler updates replace inputs or page tables without resetting unrelated requests.

Projection weights use BF4 and LoFi matmuls. The LM head uses BF8 and HiFi2. Residuals and normalization use BF16; KV storage uses BF8. Prefill computes attention from temporary K/V and saves only each sliding window's live tail. Full-attention history remains paged. This keeps sliding-cache allocation bounded even for long prompts.

The vLLM adapter declares device sampling for greedy decoding and stochastic top-k up to 32. The shared plugin routes larger/unbounded top-k, penalties, logprobs, and other unsupported device parameters through its host sampler. It preserves signed 64-bit request seeds. Host and device samplers need not produce the same random sequence.

## Run

Build TT-Metal with its supported toolchain and install the [vLLM TT plugin](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/install-vllm-tt.sh) in the TT-Metal environment. The plugin must include the companion whole-prompt sliding-cache and sampling changes. Set `HF_HOME` to a cache containing checkpoint revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.

```bash
export TT_METAL_HOME=$PWD
export HF_HOME=/path/to/huggingface
export HF_MODEL
HF_MODEL=$(python -c 'from models.demos.gemma4_31b_qb2.tt.model import checkpoint_path; print(checkpoint_path())')
export MESH_DEVICE=P300x2 EXTRA_MODELS_DIR="$PWD/models/demos"
export TT_METAL_CACHE="$PWD/generated/gemma4_31b_qb2_cache"
# Workaround for large mmap-backed weight uploads: tt-metal issue #56613.
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0

python -m vllm.entrypoints.openai.api_server \
    --model "$HF_MODEL" --served-model-name google/gemma-4-31B-it \
    --hf-overrides '{"architectures":["TTGemma4QB2ForCausalLM"]}' \
    --host 127.0.0.1 --port 8000 --block-size 128 --max-num-seqs 32 \
    --max-model-len 262144 --max-num-batched-tokens 262144 --max-logprobs -1 \
    --async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
    --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}' \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":536870912,"l1_small_size":16384}}'
```

`HF_MODEL`, if set before checkpoint resolution, must name a local checkpoint directory. Use a new cache directory for different weights. The maximum context setting is allocation capacity; it does not establish accuracy at 262,144 tokens. Images, audio, speculative decoding, prefix caching, and externally chunked prefill are unsupported.

## Validation and weekly CI

[`tests/test_decoder.py`](tests/test_decoder.py) compares real checkpoint layers with Hugging Face BF16 outputs. It covers sliding and full attention, partial batches, padding and chunk boundaries, eager decode, and traced decode after changing tokens, positions, and page mappings. Each logical user row must reach PCC 0.995. Replicas and unchanged trace replays must agree exactly.

```bash
python -m pytest models/demos/gemma4_31b_qb2/tests/test_decoder.py --timeout 600
```

The weekly Tier 3 QB2 entry in [`agentic_research_model_tests.yaml`](../../../tests/pipeline_reorg/agentic_research_model_tests.yaml) contains the complete installation, server, API-test, evaluation, reporting, and cleanup commands. It runs **only the first 10 of 198 GPQA Diamond questions**, with one seed and a 32,768-token output budget, to bound weekly runtime. The accuracy gate is 80%. This subset is a regression check; use a full-dataset run for published model-quality comparisons.

Weekly correctness coverage selects the two 1,025-token decoder cases (full and sliding attention, batch 2), all seven client/adapter checks, and five API checks for device sampling/page growth, request isolation, host sampling, mixed penalties and logprobs. The broader decoder matrix and all 37 API checks remain available for model/plugin changes and release validation; their existing publication results are retained.

The scorer uses Gemma4 chat formatting with thinking enabled, temperature 1, top-p 0.95, top-k 20, and seed 42. It scores final answers from the same streamed responses that it times. Inputs, choice permutations, dataset and harness revisions, outputs, usage counts, and timings are saved with the result.

Performance coverage uses 128- and 1,024-token inputs, 128-token outputs, and concurrency 1 and 32. Each shape has one warmup burst and two measured bursts, greedy sampling, and ignored EOS. Client TTFT includes queueing. Per-user decode throughput excludes the first token; aggregate throughput divides all generated tokens by elapsed burst time. These serving measurements differ from kernel-only or native-demo timings.

## Measured serving

Measured on 16 September 2026 with one QB2, the same checkpoint revision, the same plugin, and identical tokenized prompts. Both servers allow 32 requests. Each row uses one warmup burst and two measured bursts, with 128 output tokens and ignored EOS. Values are **this implementation / existing Gemma4 implementation**.

| Input tokens | Concurrency | TTFT (ms) | Decode tokens/s/user | Aggregate output tokens/s |
|---:|---:|---:|---:|---:|
| 128 | 1 | 90.3 / 121.1 | 37.90 / 26.43 | 37.19 / 25.98 |
| 128 | 32 | 1936.3 / 3452.9 | 35.88 / 18.81 | 746.98 / 400.75 |
| 1024 | 1 | 180.2 / 301.8 | 29.76 / 25.61 | 28.78 / 24.33 |
| 1024 | 32 | 4594.7 / 5294.4 | 26.25 / 10.66 | 433.89 / 230.92 |

This compares each implementation’s serving configuration. The existing family implementation uses its default BF8 attention/MLP weights, BF16 KV, BF16 LM head, line fabric, and a 49,152-token context pool. This implementation uses the precision policy above, ring fabric, and 262,144-token context capacity. The comparison does not isolate precision from implementation changes. Both use asynchronous scheduling; the existing adapter uses `decode_only` device sampling and internal chunked prefill, while this adapter uses device sampling for prefill and decode.

The selected experiment source scored **167/198 (84.34%) on full GPQA Diamond**, compared with the [recorded HF/vLLM reference of 83.33%](https://github.com/tenstorrent/tt-inference-server/issues/4176#issuecomment-4715337652). The reference checkpoint revision was not recorded. The maintained source subsequently scored **9/10 on the 10/198 CI subset**, with the weekly protocol above; this is a separate regression result.

The original full validation recipe took **20m23s**, including installation, 15 model tests, 37 API tests, GPQA, performance and cleanup. Weekly CI selects the smaller correctness suite above, targets under 15 minutes, and has an 18-minute timeout. Together with Llama’s 12-minute allowance, the shared QB2 budget is 30 minutes. The [model PR](https://github.com/tenstorrent/tt-metal/pull/56765) and [companion plugin PR](https://github.com/tenstorrent/vllm-tt-plugin/pull/132) track validation and merge order.
