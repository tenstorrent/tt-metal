# Gemma4 31B on QB2

Text generation for `google/gemma-4-31B-it` on one Blackhole QuietBox 2: four chips, tensor parallelism 4, and up to 32 concurrent requests. This implementation specializes the dense 31B decoder for QB2. The broader Gemma4 family implementation remains in [`../gemma4`](../gemma4).

## Design

The 60 decoder layers alternate five sliding-attention layers with one full-attention layer. Each chip holds a quarter of the projection weights. A ring fabric carries tensor-parallel reductions; completed sliding-attention sums retain BF16 precision. Decode replays a device trace with resident token, position, and sampling state. Scheduler updates replace inputs or page tables without resetting unrelated requests.

Projection weights use BF4 and LoFi matmuls. The LM head uses BF8 and HiFi2. Residuals and normalization use BF16; KV storage uses BF8. Scheduler-driven chunked prefill uses the plugin's standard `SlidingWindowSpec` and paged full-attention history. Each sliding layer retains a BF16 K/V tail between scheduler calls, including one extra page for an unaligned continuation. Request completion, preemption and slot moves preserve or release that history through the existing plugin lifecycle. Sliding storage is bounded by the window and in-flight chunk budget.

Greedy decoding and stochastic sampling with `top_k=1..32` are supported. Launch with `--override-generation-config '{"top_k":20}'` so requests that omit top-k use a supported default. Request seeds must be integers in `0..2³¹−1`. Larger/unbounded top-k and wider seeds are outside this serving contract until the independent plugin fixes land. Penalties, logprobs and other host-only parameters use the existing host sampler. Host and device samplers need not produce the same random sequence.

## Run

Build TT-Metal with its supported toolchain and install the [vLLM TT plugin](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/install-vllm-tt.sh) in the TT-Metal environment. Use unmodified plugin `main`; no companion plugin PR is required. Set `HF_HOME` to a cache containing checkpoint revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.

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
    --max-model-len 262144 --max-num-batched-tokens 8192 --max-logprobs -1 \
    --async-scheduling --no-enable-prefix-caching --enable-chunked-prefill \
    --override-generation-config '{"top_k":20}' \
    --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}' \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":536870912,"l1_small_size":16384}}'
```

`HF_MODEL`, if set before checkpoint resolution, must name a local checkpoint directory. Use a new cache directory for different weights. The maximum context setting is allocation capacity; it does not establish accuracy at 262,144 tokens. Keep the default decode interleaving enabled and use a batched-token budget no greater than 8,192. Disabling interleaving exhausted DRAM in the tested 32-user configuration because more BF16 prefill histories remained live. Images, audio, speculative decoding and prefix caching are unsupported.

## Validation and weekly CI

[`tests/test_decoder.py`](tests/test_decoder.py) compares real checkpoint layers with Hugging Face BF16 outputs. It covers sliding and full attention, partial batches, padding and chunk boundaries, eager decode, and traced decode after changing tokens, positions, and page mappings. Each logical user row must reach PCC 0.995. Replicas and unchanged trace replays must agree exactly.

```bash
python -m pytest models/demos/gemma4_31b_qb2/tests/test_decoder.py \
    models/demos/gemma4_31b_qb2/tests/test_chunked_prefill.py --timeout 600
python -m pytest models/demos/gemma4_31b_qb2/tests/test_api.py \
    --gemma-server-url http://127.0.0.1:8000
```

The weekly Tier 3 QB2 entry in [`agentic_research_model_tests.yaml`](../../../tests/pipeline_reorg/agentic_research_model_tests.yaml) contains the complete installation, server, API-test, evaluation, reporting, and cleanup commands. It runs **only the first 10 of 198 GPQA Diamond questions**, with one seed and a 32,768-token output budget, to bound weekly runtime. The accuracy gate is 80%. This subset is a regression check; use a full-dataset run for published model-quality comparisons.

Weekly correctness coverage selects the two 1,025-token decoder cases (full and sliding attention, batch 2), eight continuation/HF checks, all inexpensive client/adapter checks, and nine live API checks. Continuation cases cover aligned and partial pages, repeated ragged boundaries, capped history tails and decode after prefill at unchanged PCC≥0.995. API checks cover the supported sampling limits/default, page growth, slot reordering, parameter isolation, allowed IDs, mixed penalties and logprobs. The broader decoder matrix remains available for model changes and release validation.

The scorer uses Gemma4 chat formatting with thinking enabled, temperature 1, top-p 0.95, top-k 20, and seed 42. It scores final answers from the same streamed responses that it times. Published results contain question IDs, hashes of the inputs (including choice permutations), dataset and harness revisions, correctness, usage counts, and timings. The evaluator scores responses in memory and omits GPQA documents, prompts, generated answers, and reasoning from result artifacts. CI disables vLLM request/output logging and keeps server/evaluator diagnostics outside the public logs, artifact uploads, and AI-summary inputs; these private diagnostics are discarded with the job workspace. This preserves the dataset’s restriction on publishing examples.

Weekly performance coverage uses **128-token inputs and outputs only**, to bound runtime while testing both server configurations alongside GPQA. It starts a dedicated **one-slot server** (`--max-num-seqs 1`) at concurrency 1, then a **32-slot server** at concurrency 1 and 32. Server capacity and active concurrency are recorded separately in raw results and CI benchmark metadata. Each of these three shapes has one warmup burst and two measured bursts, greedy sampling, and ignored EOS; total coverage is 68 measured requests plus 34 warmups. The benchmark's default full sweep also includes 1,024-token inputs; weekly CI explicitly selects `--performance-input-lengths 128`.

Client TTFT includes queueing. Per-user decode throughput excludes the first token; aggregate throughput divides all generated tokens by elapsed burst time. These serving measurements differ from kernel-only or native-demo timings.

## Measured serving

The chunked implementation was measured on 25 September 2026 on one exclusive four-chip QB2 with unmodified plugin main `35090660433d5606957ded97f7130b5cc75f94f7`. Each shape uses one warmup burst, then five measured requests at concurrency 1 or three bursts of 32 requests. Generation is greedy with exactly 128 output tokens and ignored EOS.

| Server capacity / concurrency | Input tokens | Output tokens | TTFT ms | Decode tokens/s/user | Completed requests/s |
|---|---:|---:|---:|---:|---:|
| 1 / 1 | 128 | 128 | 68.28 | **46.50** | 0.357 |
| 1 / 1 | 1024 | 128 | 146.89 | 44.27 | 0.332 |
| 1 / 1 | 32768 | 128 | 5811.08 | 42.41 | — |
| 32 / 32 | 128 | 128 | 1926.30 | 35.89 | **5.847** |
| 32 / 32 | 1024 | 128 | 3109.85 | 21.00 | **3.402** |
| 32 / 32 | 2049 | 128 | 7665.12 | 15.41 | **1.863** |

Request throughput counts completed requests over the complete measured bursts, including queueing. These are burst measurements, not sustained-arrival capacity estimates. Decode/prefill interleaving delivers earlier first tokens while the per-user generation interval includes pauses to prefill other requests. Against the prior whole-prompt serving stack, longer-prompt TTFT improved about 30%, with request-throughput changes of −1.1% at 1,024 input tokens and −6.6% at 2,049. The 128-token case changed −0.4%. Single-user speed was retained through 32,768 input tokens. The chunked path uses more memory for retained BF16 histories and in-flight cache pages; full-context capacity was not stress-tested.

The original experiment source scored **167/198 (84.34%) on full GPQA Diamond**, compared with the [recorded HF/vLLM reference of 83.33%](https://github.com/tenstorrent/tt-inference-server/issues/4176#issuecomment-4715337652). The reference checkpoint revision was not recorded. The chunked implementation subsequently completed **10/10 selected questions, with 9/10 correct**, on the **10/198 CI regression subset** at the unchanged 80% gate. The full 198-question evaluation has not been repeated for this serving revision.

Each server has a **20-minute wall-clock readiness deadline**, within a **40-minute total recipe allowance**. These are safety limits, not measured runtimes. Together with Llama's 12-minute allowance, the shared QB2 budget is **52 minutes**. Public phase timings contain only fixed stage names and elapsed seconds. Startup snapshots report capacity, elapsed time, a fixed stage label, a bounded layer index and an exception flag. Private evaluator/server diagnostics remain excluded. The [model PR](https://github.com/tenstorrent/tt-metal/pull/56765) tracks final integrated validation against plugin main.

Prefill compiles on demand. An unseen logical prefill shape retires decode traces before creating persistent program buffers; decode then warms and captures again. The first request for that shape can therefore include compilation and recapture latency. Reported performance uses a warmup burst per shape and does not measure this cold-request cost. The 512 MiB serving and 32 MB decoder-test trace reservations are validated budgets, not measured trace footprints or quantified safety margins.
