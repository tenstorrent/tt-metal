# Llama 3.1 8B Instruct vLLM Integration

Status: complete on 2026-06-15.

| Batch-1 serving metric | Value |
| --- | ---: |
| vLLM sampling status | full profile pass: 41 passed, 22 skipped, 9 xfailed |
| Qualitative verdict | coherent/on-topic; minor lead-token duplication; no degeneracy |
| Prefill TTFT P50 | 81.10 ms |
| Prefill TTFT P99 | 1047.01 ms |
| ITL P50 | 15.77 ms |
| ITL P99 | 25.09 ms |
| Aggregate output throughput | 51.84 tok/s |
| Mean per-user decode | 53.90 t/s/u |

Batch-1 benchmark workload: prompt length 128, output length 128, 32 requests,
concurrency 1, `max_num_seqs=1`, `max_model_len=1024`, T3K 1x8,
`FABRIC_1D_RING`, decode trace enabled, `sample_on_device_mode=all`.

## Serving Commands

Final full sampling command:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --sampling-profile full \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --port 8001
```

Batch-1 benchmark command:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --port 8001
```

Qualitative outputs were produced by the successful smoke/qualitative/benchmark
runner invocation archived under
`artifacts/success_smoke_sampling_qual_bench_20260615/`.

## Artifacts

Current readiness artifacts:

- `readiness_vllm/sampling_tests.log` - final full sampling profile.
- `readiness_vllm/vllm_qualitative_outputs.json` - qualitative completions.
- `readiness_vllm/vllm_benchmark.json` - batch-1 benchmark with concurrency 1.
- `readiness_vllm/check_degenerate_output.log` - vLLM degeneracy check.
- `readiness_vllm/server.log` - latest benchmark server log.

Archived evidence:

- `artifacts/success_full_sampling_20260615/` - final full sampling server and
  pytest logs.
- `artifacts/success_batch1_benchmark_20260615/` - benchmark server log,
  benchmark JSON, and degeneracy log.
- `artifacts/success_smoke_sampling_qual_bench_20260615/` - successful smoke,
  qualitative, and earlier concurrency-8 benchmark artifacts.

## Adapter Contract

Adapter:
`models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator_vllm.py`

Class: `Llama31_8B_InstructForCausalLM`.

The adapter is intentionally thin. It delegates to the full-model generator
low-level methods:

- `prefill_forward_device(...)` for prefill logits consumed by on-device
  sampling.
- `_decode_trace_sample(..., readback=False)` for traced decode token-out.
- `read_decode_output(...)` and `process_decode_output_host(...)` for the vLLM
  async submit/read/host-format split.

It does not implement a separate adapter sampling path, host greedy argmax,
full-logits readback, generic top-k greedy fallback, or Python readback/writeback
token feedback loop. `process_decode_output_host(..., is_tokens=False)` fails
closed.

vLLM owns the serving attention KV cache. `allocate_kv_cache(...)` allocates the
serving cache with `self.model.policy.kv_cache_dtype`, and prefill/decode pass
that cache through the generator/model path. Standalone generator mode continues
to use the model-owned cache.

Plugin registration is in
`/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.
Set `TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct` or
`TT_LLAMA_TEXT_VER=meta_llama_llama_3_1_8b_instruct` to register
`TTLlamaForCausalLM` to this adapter.

## Precision Policy

Serving constructs `Llama31_8B_InstructFullModel.from_pretrained(...)`, which
loads `doc/datatype_sweep/selected_precision_config.json` by default.

Selected policy:

- Weight groups: embedding BF16, norms BF16, attention BFP4 all layers, MLP
  BFP4 all layers, LM head BFP8.
- Runtime dtypes: activation BFP8, residual BFP8, CCL BFP8, KV cache BFP8,
  MLP multiply BFP8, device logits BF16.
- Compute fidelities: attention HiFi2, MLP LoFi, LM head HiFi2, norms HiFi4.
- Layer exceptions: none.

## Sampling Results

Final full profile:

```text
41 passed, 22 skipped, 9 xfailed, 16 warnings in 119.69s
```

The skipped cases are request shapes that require host compatibility sampling or
full/top-k logprob readback: `min_p`, bad words, logit bias, allowed token IDs,
minimum tokens, top-k/all-vocab logprobs. These are intentionally not served
under `sample_on_device_mode=all`.

The xfailed cases are no-seed variety/top-k variety checks. The vLLM integration
skill classifies these reproducibility/variety-only failures as out of scope
when correctness checks pass. Seeded reproducibility, request isolation,
sampled-token logprobs, top1-greedy, and batch-1 penalty smoke checks passed.

## Qualitative Verdict

The qualitative outputs are coherent and generally on topic for haiku,
supervised/unsupervised learning, story continuation, thermodynamics,
translation, and Fibonacci-code prompts. There is minor repeated leading text in
some completions (`a a`, `The The`) and one sampled haiku completion drifts into
meta-discussion/request commentary. There is no gibberish, no wrong-language
drift except the expected French translation, and no cross-request contamination.

Degeneracy check:

```text
No degenerate output detected.
```

## Async Decode

Capabilities:

- `supports_prefix_caching=False`
- `supports_async_decode=True`
- `supports_sample_on_device=True`
- `tt_async_decode_allows_overlap=False`

`supports_async_decode=True` covers split submit/read/host formatting with
decode trace enabled. Scheduler overlap is a separate contract and remains
disabled because next-step token/current-position/page-table inputs are still
refreshed from vLLM host scheduler state at trace/layout changes. No
async-scheduling overlap proof was run.

The full-model trace-feedback evidence remains the lower-level stale-state
proof for the delegated path: unchanged page tables are not recopied, changed
page tables copy once, current positions advance on device, and
`greedy_uses_split_sampling=true`. The vLLM adapter exercised the same delegated
path through full sampling with request isolation, seeded reproducibility, and
top1-greedy passing.

## Performance Comparison

Use full-model teacher forcing only as a lower bound. The selected datatype
stage recorded teacher-forcing decode at `49.49 t/s/u`; optimized full-model
token-out no-readback steady replay recorded `70.55 t/s/u`.

The vLLM batch-1 serving benchmark measured mean per-user decode
`53.90 t/s/u`. This is above the teacher-forcing lower bound and below the
direct no-readback token-out path, as expected for serving orchestration,
streaming response handling, token/logprob readback, and OpenAI API overhead.
No adapter-side host sampling, full-logits readback, or Python token feedback
loop remains in the measured path.

## Cleanup

The successful full sampling and benchmark runs terminated cleanly. Final
process audits found no leftover vLLM, EngineCore, or `run_vllm_server`
processes holding devices, and `tt-smi -ls --local` listed all 8 devices.

Known limitations:

- The adapter currently supports `max_num_seqs=1`.
- Host-compatibility request features are rejected/skipped under strict
  on-device sampling.
- Top-k/all-vocab logprobs are unsupported for this model without a device top-k
  logprob path.
- `tt_async_decode_allows_overlap=False`; async scheduler overlap is disabled
  until a focused stale-state overlap proof exists.
