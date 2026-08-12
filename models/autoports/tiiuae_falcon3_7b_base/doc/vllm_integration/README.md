# Falcon3-7B-Base vLLM integration

**Primary single-user serving (128 prompt / 128 output / 1 request / concurrency 1,
greedy): TTFT P50/P99 182.5/182.5 ms; TPOT mean/P99 15.9/15.9 ms;
ITL P50/P99 14.6/15.1 ms; 58.2 output tok/s; 62.9 TPOT-derived
decode t/s/u.**

Secondary CI serving burst (100 prompt / 100 output / 32 requests / unbounded
client concurrency, greedy): TTFT P50/P99 414.4/415.6 ms; TPOT mean/P99
16.9/18.3 ms; ITL P50/P99 15.1/74.7 ms; 1,539.7 output tok/s;
59.3 TPOT-derived t/s/u. This burst TPOT is capacity evidence, not the headline
decode number because admission and interleaved prefill affect it.

## Serving status

`tiiuae/Falcon3-7B-Base` revision
`bf3d7ed586cb22a921520e2d681a9d3d7642cde8` serves through the shared TT vLLM
path on the fixed TP4 `P300x2`/1x4 mesh. The final shared runner completed full
sampling (72 passed, 1 skipped), checkpoint-correct qualitative completions, an
exact 37-token non-aligned request, the primary benchmark, and the CI burst.

The adapter is `tt/generator_vllm.py`. It delegates prefill/decode and sampling
to `tt/generator.py`, passes vLLM's cache through, and uses the full-model
canonical model-trace plus split-sampling-trace token-out path. Steady decode
does not read stale host tokens or positions; the sampled device token is copied
directly into the persistent model input. There is no measured-path host argmax,
full-logits readback, generic top-k greedy fallback, or Python token feedback
loop. `supports_async_decode=True`; the final server log records
`trace_mode=all`, `sample_on_device_mode=all`, and async scheduling enabled.
The runner passes each request's declared prompt-plus-output horizon before
prefill, so Falcon3 grows its shared RoPE table before decode trace capture.

Host sampling is explicit compatibility behavior only. The plugin selects it
for logprobs, logits processors/allowlists, non-default penalties, and stochastic
top-k values beyond Falcon3's dedicated top-32 sampler. Greedy benchmark and
normal supported sampling use the traced on-device path.

## Configuration and context

- Advertised and served `max_model_len`: **32,768**, exactly matching
  `doc/context_contract.json`. No capability reduction was needed.
- `max_num_seqs`: **32**. vLLM allocated 4,128 blocks / 132,096 tokens, reporting
  4.03x concurrency at the full 32K length; shorter requests exercise all 32
  serving slots.
- TT config: `trace_region_size=512000000`,
  `fabric_config=FABRIC_1D_RING`, `sample_on_device_mode=all`.
- Cache: vLLM owns and sizes the BF16 paged cache. Explicit external block counts
  no longer inherit the standalone generator's `batch * max_context` reservation.
- Precision config: `all_bfp4_lofi_bf16_kv` from
  `doc/datatype_sweep/selected_precision_config.json`: BFP4 attention, MLP, and
  LM-head weights; BF16 embedding/norm weights; BFP8 activations, CCL, and logits;
  BF16 residual and KV cache; LoFi attention/MLP/LM-head fidelity; no layer
  exceptions.
- Non-aligned evidence: `non_aligned_prompt_37.json` records a successful exact
  37-token request plus four returned tokens. Thirty-seven is not divisible by
  the 32-token page/tile or 128-token prefill alignment.

The tokenizer has no chat template, so qualitative judgment uses plain
completions. `base_chat_template.jinja` is only an explicit compatibility
template for shared chat-endpoint sampling tests.

## Commands

Final runner command:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 32768 \
  --block-size 32 --sampling-profile full \
  --tt-config '{"trace_region_size":512000000,"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all"}' \
  --additional-server-args '--chat-template models/autoports/tiiuae_falcon3_7b_base/doc/vllm_integration/base_chat_template.jinja' \
  --server-timeout 1200
```

The exact benchmark commands are preserved in `vllm_benchmark.json` and
`vllm_ci_serving_benchmark.json`.

## Evidence and interpretation

- `readiness_vllm/sampling_tests.log`: full profile, 72 passed / 1 skipped. The
  skip is the suite's unsupported all-vocabulary chat-logprobs case.
- `readiness_vllm/vllm_qualitative_outputs.json`,
  `qualitative_prompt_format.json`, and `qualitative_verdict.md`: six greedy and
  six sampled outputs read in full. Coherent English and on-topic starts; no
  repetition loop, gibberish, wrong-language drift, corrupt feedback, or
  cross-request leakage. Long tails sometimes continue into unrelated
  training-document questions/instructions. Exact 256-token HF BF16 controls in
  `hf_exact_qualitative_controls.json` show the same base-checkpoint autocomplete
  tendency; this is request contamination, not a serving-state leak.
- `readiness_vllm/async_overlap_and_determinism.json`: a 300-token target is
  byte-identical alone and under staggered admission/removal churn (300/300
  tokens, crossing position 256); repeated and cross-batch-position top-10
  logprob signatures match exactly. The vLLM first token is also HF BF16's
  rank-1 token.
- `readiness_vllm/async_overlap_qualitative_outputs.json` preserves the exact
  isolated/overlap strings in the shared checker schema;
  `async_overlap_degenerate_output.json` reports no degeneracy finding (213
  measured words each, zero adjacent duplication, trigram-loop fraction 0.0282).
- `readiness_vllm/vllm_result.json` and `vllm_benchmark.json`: primary raw and
  normalized 128/128/1 results.
- `readiness_vllm/vllm_ci_serving_result.json` and
  `vllm_ci_serving_benchmark.json`: secondary raw and normalized 100/100/32
  results; all 32 requests returned all 100 requested tokens.
- `tests/test_generator_vllm_contract.py`: six adapter-boundary tests covering
  stale token/position rejection, current reset state, live-slot remap rejection,
  logical versus padded page-table ownership, and pre-trace RoPE horizon growth.
- Full-model lower bound: optimized teacher-forcing is 110.81 t/s/u and
  caller-visible canonical token-out is 110.38 t/s/u. These are device-only/full-
  model lower bounds, not serving-equivalent measurements. vLLM's 62.9 headline
  includes scheduler, asynchronous boundary, sampling, and serving overhead.

## Limitations and runtime audit

- This is a base checkpoint, not instruction-tuned chat. The compatibility chat
  template concatenates content and must not be interpreted as a quality format.
- The dedicated device sampler supports top-k up to 32. Wider stochastic top-k
  and host-only vLLM features use the explicit compatibility path.
- On this Blackhole host, repeated process lifecycles sometimes leave ERISC core
  29-25 with a stale heartbeat even after the vLLM process reports clean exit.
  Recovery was bounded: confirm no vLLM/EngineCore owner, `tt-smi -r`, `tt-smi -s`,
  then a 1x4 mesh-open/close smoke. The final run itself shut down cleanly; the
  post-run process audit found no vLLM or EngineCore process holding devices.
- No profiler was run in the vLLM serving stage.

Local stage commit SHAs are recorded in `work_log.md`; nothing was pushed.
