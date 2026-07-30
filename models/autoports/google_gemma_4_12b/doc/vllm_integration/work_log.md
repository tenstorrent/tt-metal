# Gemma 4 12B vLLM Integration Work Log

## Implementation

- Started from the completed optimized full-model stage in `../optimized_full_model/README.md`: batch-1 T3K TTFT 121.93 ms for the 149-token AIME24 prompt and traced on-device decode replay at 23.08 tokens/s/user.
- Added `tt/generator_vllm.py` as a thin vLLM adapter around `tt/generator.py`.
- Added vLLM-owned KV cache allocation from per-layer specs. Sliding layers use 8 local KV heads with head size 256; full/global layers use 1 local KV head with head size 512.
- Kept cache ownership explicit: vLLM mode constructs `Gemma412BGenerator` with `allocate_standalone_cache=False` and requires `kv_cache`/page tables from the runner.
- Updated `tt/generator.py` and `tt/model.py` for vLLM page-table/cache handoff, per-layer page tables, traced decode page-table refresh, and sharded prefill logits for on-device sampling.
- Registered the model in `/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.
- Updated the TT vLLM plugin model runner for `UniformTypeKVCacheSpecs` per-layer cache specs and for already-finalized device-sampled logprobs.
- Added `simple_chat_template.jinja` for chat endpoint startup.
- Adjusted TT plugin tests so batch-1 servers and tokenizers with low-ID special tokens are handled robustly.

## Validation Commands

Syntax checks:

```bash
python -m py_compile \
  models/common/sampling/tt_sampling.py \
  models/autoports/google/gemma-4-12B/tt/model.py \
  models/autoports/google/gemma-4-12B/tt/generator.py \
  models/autoports/google/gemma-4-12B/tt/generator_vllm.py
python -m py_compile /localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
python -m py_compile /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_host_only_params.py
python -m py_compile /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_seeding_and_variety.py
python -m py_compile /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_tt_penalties.py
```

Smoke readiness passed with:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --sampling-profile smoke \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args '--chat-template models/autoports/google/gemma-4-12B/doc/vllm_integration/simple_chat_template.jinja --generation-config vllm'
```

Final full run used a held server plus external checks:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args '--chat-template models/autoports/google/gemma-4-12B/doc/vllm_integration/simple_chat_template.jinja --generation-config vllm'
```

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages sampling,qualitative,benchmark \
  --server-url http://localhost:8000 \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --sampling-profile full \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}'
```

Final result:

- Sampling: `71 passed, 1 skipped` in `326.45s`.
- Qualitative: artifacts refreshed at `readiness_vllm/vllm_qualitative_outputs.json`; outputs are not coherent enough for production.
- Benchmark: `readiness_vllm/vllm_benchmark.json`.

## Benchmark

Workload:

- `prompt_len=128`
- `output_len=128`
- `num_requests=32`
- `concurrency=8`
- `max_num_seqs=1`
- `max_model_len=4096`

Metrics:

- Requests: 32 completed in 182.3 s
- TTFT P50/P99: 40003.7 ms / 40203.0 ms
- ITL P50/P99: 44.0 ms / 88.2 ms
- Aggregate output throughput: 21.3 tok/s
- Mean per-user decode throughput: 21.8 t/s/u

## Qualitative Verdict

The six qualitative prompts were read manually.

- Coherence: poor. Some outputs contain topical fragments, but most fail to answer.
- Topic: weak. Several completions drift into Python/code, FAQ, or repeated prompt-like text.
- Repetition: severe in greedy outputs and present in sampled outputs.
- Gibberish/blank output: sampled thermodynamics and translation outputs are effectively blank; other outputs contain formatting/code fragments.
- Wrong-language drift: the French translation prompt does not produce a French translation.
- Request contamination: repeated source prompt fragments and unrelated question templates appear.

## Cleanup And Stability Notes

- One cold startup attempt hit `ARC startup error ... Timed out after 300000 ms` before model code ran. Devices were reset with `tt-smi -r` and the run was retried successfully.
- AutoFix was used for stale traced decode behavior; see `AUTODEBUG.md`.
- After the final full pass, the held server was stopped and `pgrep -af 'vllm|EngineCore|api_server'` showed no serving processes. The only remaining match was the unrelated multigoal launcher command line.
