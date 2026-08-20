# Qwen3.6-35B-A3B vLLM Integration Work Log

## Scope

Stage: vLLM integration for `Qwen/Qwen3.6-35B-A3B`.

Primary TT repo root: `/localdev/vkovacevic/tt-metal`.
vLLM repo root: `/localdev/vkovacevic/vllm`.

Skills used:

- `/localdev/vkovacevic/tt-metal/.agents/skills/vllm-integration/SKILL.md`
- `/localdev/vkovacevic/tt-metal/.agents/skills/tt-device-usage/SKILL.md`
- `/localdev/vkovacevic/tt-metal/.agents/skills/stage-review/SKILL.md`
- `/localdev/vkovacevic/tt-metal/.agents/skills/qualitative-check/SKILL.md`
- `/localdev/vkovacevic/tt-metal/.agents/skills/autofix/SKILL.md`

## Implementation Notes

- Added `tt/generator_vllm.py` for `Qwen3_5MoeForConditionalGeneration`.
- The adapter delegates to `QwenReadinessGenerator` and `QwenFullModel` low-level prefill/decode methods. Adapter-owned logic is limited to vLLM interface translation, cache/page-table mapping, prompt lengths, sampling parameter formatting, trace input reset, and output formatting.
- Registered the model in `/localdev/vkovacevic/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()`.
- Updated the TT vLLM plugin to make unsupported capabilities explicit: async scheduler overlap is disabled unless the model declares it, host sampling compatibility is optional/test-only, batch rows are propagated for lane scheduling, and on-device token-output sampling remains the measured path.
- Updated common sampling so greedy/top-k1 rows use the TT argmax sentinel instead of a host greedy fallback.
- Kept prefix caching disabled.

## Selected Configuration

Serving uses `doc/datatype_sweep/selected_precision_config.json` `baseline_default`:

- embedding, norms, router: BF16;
- attention, linear attention, shared MoE, LM head: BF8;
- routed MoE: BF8 on linear-attention layers, BF4 on full-attention layers;
- activation, residual, CCL, KV cache, linear state, logits: BF16;
- sampling output: uint32;
- compute fidelity: TTNN defaults;
- layer exceptions: none.

Context comes from `doc/context_contract.json`: `supported_context=262144`. Served `max_model_len=262144`.

## Commands And Evidence

Adapter and sampling-contract tests:

```bash
python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py

python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_generator_vllm.py -q
```

Result: `5 passed`. The generated artifact from the broader stage run is `readiness_vllm/adapter_tests.log`.

Final readiness run:

```bash
env TT_METAL_HOME=/localdev/vkovacevic/tt-metal TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=384000000 \
PYTHONPATH=/localdev/vkovacevic/tt-metal:/localdev/vkovacevic/vllm/plugins/vllm-tt-plugin/src:/localdev/vkovacevic/vllm:${PYTHONPATH:-} \
LD_LIBRARY_PATH=/localdev/vkovacevic/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-} \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --block-size 32 \
  --port 8011 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":384000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args=--async-scheduling \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/run_vllm_server_final_sampling_benchmark_trace_replay_patch.log
```

Result: server reached `/health` after about 360s, full sampling passed, primary benchmark passed, CI serving-burst benchmark passed, and the server terminated cleanly.

Final sampling suite:

- Artifact: `readiness_vllm/sampling_tests.log`
- Result: `72 passed, 1 skipped, 2 warnings in 3579.31s`
- Sampling profile: full
- `tt-max-num-seqs`: `32`

Targeted request-isolation and seeding repro:

- Artifact: `readiness_vllm/targeted_request_isolation_then_seeding_after_lane_greedy_patch.log`
- Result: `2 passed`

Non-aligned prompt check:

- Artifact: `readiness_vllm/non_aligned_prompt_check_trace_replay_patch.json`
- Request type: direct chat completion after the trace-replay/page-table fixes
- Prompt token length: `26`, from `response.usage.prompt_tokens`
- Checked moduli: `16, 32, 64, 128, 1072, 2048`
- Result: HTTP `200`

No-thinking qualitative run:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/scripts/run_no_think_chat_qualitative.py \
  --server-url http://localhost:8014 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/vllm_chat_no_think_qualitative_outputs.json \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/run_no_think_chat_qualitative_trace_replay_patch.log
```

Artifacts:

- `readiness_vllm/vllm_chat_no_think_qualitative_outputs.json`
- `readiness_vllm/run_no_think_chat_qualitative_trace_replay_patch.log`
- `readiness_vllm/qualitative_no_think_control_verdict.json`
- `readiness_vllm/full_model_no_think_greedy_controls.json`
- `readiness_vllm/vllm_chat_no_think_checker_outputs.json`
- `readiness_vllm/vllm_no_think_degenerate_output_report.json`
- `readiness_vllm/check_vllm_no_think_degenerate_output_trace_replay_patch.log`

Verdict: no-thinking chat outputs are coherent, on topic, and not contaminated by reasoning/request-analysis text. No repetition loop, gibberish, or wrong-language drift was observed. The story prompt regression was fixed; greedy output starts `a young shepherd discovered a hidden door...`.

Benchmark artifacts and metrics:

- Primary raw: `readiness_vllm/vllm_result.json`
- Primary normalized: `readiness_vllm/vllm_benchmark.json`
- Primary workload: `128 input / 128 output / 1 request / max concurrency 1 / temperature 0.0 / max_num_seqs 32`
- Primary TTFT P50/P99: `7517.318 ms / 7517.318 ms`
- Primary TPOT mean/P99: `945.279 ms / 945.279 ms`
- Primary ITL P50/P99: `919.151 ms / 1231.973 ms`
- Primary output throughput: `1.003386 tok/s`
- Primary TPOT-derived decode: `1.057889 t/s/u`

- CI burst raw: `readiness_vllm/vllm_ci_serving_result.json`
- CI burst normalized: `readiness_vllm/vllm_ci_serving_benchmark.json`
- CI burst workload: `100 input / 100 output / 32 requests / unbounded concurrency / temperature 0.0 / max_num_seqs 32`
- CI burst TTFT P50/P99: `156247.173 ms / 156248.158 ms`
- CI burst TPOT mean/P99: `1003.533 ms / 2002.874 ms`
- CI burst ITL P50/P99: `919.192 ms / 3851.690 ms`
- CI burst output throughput: `12.753816 tok/s`
- CI burst TPOT-derived decode: `0.996480 t/s/u`

The CI burst profile is recorded for CI parity and serving-capacity context only. The headline decode t/s/u is the primary single-user TPOT-derived value.

## Anomalies And Resolutions

Initial sampling failures and request-isolation issues:

- Evidence: triage files under `doc/vllm_integration/triage/`, `readiness_vllm/targeted_request_isolation_then_seeding_after_lane_greedy_patch.log`, and `readiness_vllm/sampling_tests.log`.
- Resolution: explicit capability gating, batch-row propagation, optional host compatibility canonicalization for shared tests, and TT argmax sentinel handling. Final full sampling suite passed.

No-thinking qualitative story regression:

- Symptom: the story prompt briefly returned `Here are!` after earlier fixes.
- Root causes found: first prefill chunk was reading from newly-filled paged cache instead of local K/V; vLLM physical block IDs needed one-based-to-zero-based normalization for TT cache kernels; first trace-capture decode step returned capture-run output rather than replay output with current inputs.
- Resolution: first prefill chunk uses local attention (`chunk_start_idx=None` for `start == 0`), page tables subtract positive vLLM block IDs, and capture now resets inputs and executes the trace before returning. Final no-thinking story output is coherent.

Raw/default qualitative contamination:

- Evidence: `readiness_vllm/vllm_qualitative_outputs.json`, `readiness_vllm/vllm_chat_qualitative_outputs.json`, `readiness_vllm/qualitative_verdict.json`.
- Resolution: accepted serving qualitative mode is chat completions with `chat_template_kwargs.enable_thinking=false`.

Serving performance gap versus full-model teacher-forcing:

- Evidence: primary vLLM decode `1.057889 t/s/u`; selected datatype-sweep teacher-forcing lower bound `16.378533 t/s/u`.
- Resolution: avoidable adapter fallbacks were removed and scheduler overlap was left disabled because the experiment did not materially improve TPOT and overlap safety was not proven. Remaining gap is recorded as a serving limitation.

Active-ethernet cleanup timeout:

- Evidence: a post-cleanup smoke attempt hit the recurring active-ethernet timeout; `readiness_vllm/tt_smi_reset_after_cleanup_smoke_timeout.log` records the board reset.
- Resolution: final `readiness_vllm/mesh_smoke_after_cleanup.log` after reset completed with `MESH_SMOKE_OK`.

## Cleanup Audit

- `readiness_vllm/process_cleanup_check.log`: `NO_MATCH` for vLLM server, `EngineCore`, and `run_vllm_server`.
- `readiness_vllm/tt_smi_list_after_cleanup.log`: all four P300C chips visible and resettable.
- `readiness_vllm/mesh_smoke_after_cleanup.log`: final 2x2 mesh open/synchronize/close passed with `MESH_SMOKE_OK`.

## Review And Commits

Independent `$stage-review`: clean-pass from subagent `01a01f58-57e6-7970-9e96-e87972e8cacb` (`Laplace`).

Review summary:

- No required work.
- Confirmed shared TT generator/model path, model registration, selected precision config, on-device split token-output sampling path, context length, final sampling/benchmark/qualitative evidence, non-aligned prompt evidence, cleanup audit, and updated docs.
- Noted dirty worktrees only as a post-review checkpointing task.

Local checkpoint commit SHAs are recorded in the final stage handoff after commit creation.
