# Qwen3-4B vLLM Integration Work Log

## 2026-07-03

- vLLM plugin local commit: `de6c44fd89154bd800c8c947e7205876b93013e3`
- Implemented `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py`.
- Registered `TT_QWEN3_TEXT_VER=autoport_qwen3_4b` in `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.
- Added Qwen3-4B support for vLLM-owned KV cache allocation, selected datatype-sweep precision, traced decode token-out, async decode capability declaration, page-table generation forwarding, inactive row scratch mapping, and explicit on-device greedy sampling.
- Updated full-model generator/model APIs to expose device-output decode paths, prefill final-row device logits, and the trace-safe TP4 greedy sampler.
- Fixed the TP4 pair-reducer gathered-candidate stride in `qwen_argmax_pair_reduce_kernel.cpp`. Before the fix, local winners selected EOS correctly but the reducer read the wrong gathered row and selected token `25521`; after the fix the same probe selected EOS token `151645`.
- Updated the shared vLLM TT plugin for model registration, sample-on-device gating, async decode overlap, and optional host-compat behavior needed by shared sampler tests.
- Added TT vLLM runner idle trace release: when the scheduler has no active requests and no pending sample/async decode completions, it calls the model `reset_warmup_state()` hook. For Qwen3-4B this releases decode and sampling traces before later requests can recapture or execute traces.
- Updated `models/common/readiness_check/run_vllm_server.py` for `P150x4`, `--additional-config`, final sampling profile execution, and Qwen3 chat-template qualitative requests.

## Commands

Final serving command:

```bash
TT_QWEN3_TEXT_VER=autoport_qwen3_4b \
VLLM_PLUGINS=tt,tt_model_registry \
QWEN3_4B_AUTOPORT_DIR=/home/ubuntu/tt-metal/models/autoports/qwen_qwen3_4b \
PYTHONPATH=/home/ubuntu/vllm:$PYTHONPATH \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/qwen_qwen3_4b \
  --hf-model Qwen/Qwen3-4B \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 40960 \
  --block-size 32 \
  --sampling-profile full \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 134217728, "fabric_config": "FABRIC_1D_RING"}'
```

Final non-aligned prompt check:

```bash
python_env/bin/vllm bench serve --backend vllm --model Qwen/Qwen3-4B --base-url http://localhost:8000 --endpoint /v1/completions --dataset-name random --random-input-len 37 --random-output-len 8 --num-prompts 1 --max-concurrency 1 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename models/autoports/qwen_qwen3_4b/readiness_vllm/non_aligned_prompt_len37_result.json --temperature 0.0
```

Final contract test:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short
```

Degenerate output check:

```bash
python models/common/readiness_check/check_degenerate_output.py --hf-model Qwen/Qwen3-4B --missing-artifacts critical --scope vllm
```

## Validation

- Adapter contract: `20 passed`, `2 warnings` in `6.30s`; includes stale-token/current-position/page-table adapter checks.
- Shared sampling profile: `72 passed`, `1 skipped`, `2 warnings` in `787.14s`; artifact `readiness_vllm/sampling_tests.log`.
- Qualitative: six prompts completed with greedy and sampled outputs; final judgment is coherent, on topic, no material repetition, no gibberish, no wrong-language drift, no request contamination.
- Degenerate-output checker: no findings for final vLLM outputs.
- Non-aligned prompt length: `random_input_len=37`, `random_output_len=8`, completed `1/1`; artifact `readiness_vllm/non_aligned_prompt_len37_result.json`.
- Trace lifecycle: the final non-aligned rerun still logs TT-Metal's active-trace allocation warning once after the final decode signpost, but `readiness_vllm/trace_warning_audit.log` shows zero decode signposts and zero `execute_trace` mentions after that warning; the next server activity is metrics, idle `Running: 0 reqs`, and shutdown.
- Process cleanup: wrapper-launched servers terminated cleanly; final process audit shows no live vLLM server or EngineCore worker holding devices. Old defunct EngineCore children remain under PID 1 only.
- Stage review: independent rereview returned `clean-pass` after refreshing the 20-test adapter artifact and classifying/controlling the active-trace warning.

## Metrics

Primary single-user benchmark, 128/128/1, concurrency 1, temperature 0.0, ignore EOS:

- Raw path: `readiness_vllm/vllm_result.json`
- Summary path: `readiness_vllm/vllm_benchmark.json`
- TTFT P50/P99/mean: `225.6/225.6/225.6 ms`
- TPOT mean/P99: `13.3/13.3 ms`
- ITL P50/P99/mean: `10.3/10.7/13.3 ms`
- Output throughput: `67.0 tok/s`
- TPOT-derived decode: `75.4 t/s/u`

CI serving-burst benchmark, 100/100/32, vLLM burst admission, temperature 0.0, ignore EOS:

- Raw path: `readiness_vllm/vllm_ci_serving_result.json`
- Summary path: `readiness_vllm/vllm_ci_serving_benchmark.json`
- TTFT P50/P99/mean: `1893.0/2869.6/2172.4 ms`
- TPOT mean/P99: `33.2/48.4 ms`
- ITL P50/P99/mean: `17.7/295.8/33.2 ms`
- Output throughput: `586.3 tok/s`
- TPOT-derived decode: `30.2 t/s/u`, secondary only

## Artifacts

- `readiness_vllm/server.log`
- `readiness_vllm/sampling_tests.log`
- `readiness_vllm/vllm_qualitative_outputs.json`
- `readiness_vllm/vllm_benchmark.json`
- `readiness_vllm/vllm_result.json`
- `readiness_vllm/vllm_ci_serving_benchmark.json`
- `readiness_vllm/vllm_ci_serving_result.json`
- `readiness_vllm/non_aligned_prompt_len37.log`
- `readiness_vllm/non_aligned_prompt_len37_result.json`
- `readiness_vllm/adapter_runtime_tests.log`
- `readiness_vllm/trace_warning_audit.log`
- `readiness_vllm/hf_no_think_greedy_256_controls.json`

## Debug Notes

- HF no-thinking greedy controls were captured in `readiness_vllm/hf_no_think_greedy_256_controls.json`.
- Direct host-argmax generator evidence produced a clean haiku and EOS.
- TP4 EOS probe before the reducer fix: local pairs included shard 3 EOS `[16888, 151645]` but global sample selected shard 0 token `25521`.
- TP4 EOS probe after the reducer fix: the same local winners reduced to EOS `151645`.
- The reducer fix uses the gathered tensor's aligned page size for pair-row stride instead of assuming an unpadded active-batch stride.
