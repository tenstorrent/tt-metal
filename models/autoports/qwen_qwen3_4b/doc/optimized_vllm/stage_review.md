# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Findings

- The optimized-vLLM stage satisfies the original goal contract. The measured path is the real vLLM TT plugin path through `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py`, with the plugin registration in `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.
- The before/after primary benchmark artifacts use the same single-user workload and serving configuration: random 128 input / 128 output, 1 prompt, max concurrency 1, temperature 0.0, ignore EOS, `max_model_len=40960`, `max_num_seqs=32`, `sample_on_device_mode=all`, and the same TT config. The after result reports TTFT 218.95 ms, TPOT 13.25 ms, ITL p50/p99 10.33/10.62 ms, aggregate throughput 67.30 tok/s, and TPOT-derived decode 75.48 t/s/u.
- The before/after CI serving-burst artifacts also use matching workloads: random 100 input / 100 output, 32 prompts, temperature 0.0, ignore EOS, and the same serving configuration. The after result reports TTFT p50/p99 1868.82/2837.42 ms, TPOT mean/p99 28.92/44.03 ms, aggregate throughput 638.38 tok/s, and TPOT-derived decode 34.58 t/s/u. This is correctly treated as secondary capacity/nightly-parity evidence, not the headline decode number.
- The 40960 context contract is preserved. `doc/context_contract.json`, `doc/optimized_vllm/perf_summary.json`, and the final server log all report `max_model_len=40960`; I found no evidence of lowered benchmark, eval, or serving context.
- Non-aligned prompt serving remains valid. `readiness_vllm/non_aligned_prompt_len37_result.json` completed 1/1 requests with 37 input tokens and 0 failures.
- vLLM sampling and adapter gates passed from existing artifacts. `doc/optimized_vllm/after/sampling_tests.log` reports 72 passed and 1 skipped, and `readiness_vllm/adapter_runtime_tests.log` reports 20 passed.
- Final qualitative outputs are coherent under the chat-formatted prompts, and the stale debug qualitative file is correctly excluded. Direct inspection of `doc/optimized_vllm/after/vllm_qualitative_outputs.json` shows usable answers for the haiku, supervised-vs-unsupervised explanation, story continuation, thermodynamics answer, French translation, and Fibonacci function prompts. Direct inspection of `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json` shows stale pre-final sampler failures such as repeated decorative tokens and repeated "Final Answer" text, matching the exclusion claim.
- Async decode and trace reuse are implemented in the measured path. `generator_vllm.py` advertises async decode capability, calls `decode_forward(..., read_from_device=False)` for the async path, returns device tensors before the plugin async boundary, refreshes cached page-table tensors only when changed, and keeps `force_argmax=False`. `async_decode.py` submits non-DP async decode with `read_from_device=False`, defers reads through `read_decode_output`, and finalizes host conversion after synchronization. `model.py` executes decode traces with `ttnn.execute_trace(..., blocking=False)`.
- On-device greedy sampling is preserved for the measured path. The final server config uses `sample_on_device_mode=all`, `model_runner.py` selects device sampling for greedy requests without host-only sampling features, and the adapter/model code routes through split greedy sampling without host `argmax`, full-logits readback, or force-argmax in the measured decode path.
- I found no existing profiler artifacts or profiler-command evidence for this stage. The only profiler-related references in the optimized-vLLM artifacts are documentation statements saying profiler collection was intentionally skipped.
- Runtime cleanup is adequate for the stage gate. The final server log reaches API shutdown, closes user-mode device drivers, and completes cluster destruction. A process-table check found no live `vllm`, `EngineCore`, `run_vllm_server`, or API server process.

## Other Concerns

- vLLM decode is slower than the optimized full-model traced decode comparison: 75.48 t/s/u from primary vLLM TPOT versus 96.91 t/s/u for optimized full-model steady decode. That is about 22% lower throughput for the serving path, but still close enough for this stage given vLLM/plugin overhead and the absence of a stronger same-model optimized-vLLM reference in the checkout or run roots I inspected.
- Shutdown logs include nanobind leaked-instance diagnostics after both sampling tests and the final server run. This did not leave a live vLLM/EngineCore process, and the final server log shows device-driver and cluster cleanup completing, so I am treating it as non-blocking residual cleanup noise.

## Hard-Check Gaps

- I did not find a standalone saved output artifact for the final vLLM degenerate-output checker under `readiness_vllm` or `doc/optimized_vllm`; the successful checker run is recorded in `doc/vllm_integration/work_log.md`. Direct inspection of the final vLLM qualitative JSON did not show visible degeneracy, so this is an evidence-packaging gap rather than required work.
- The bounded device recovery commands after the active-Ethernet startup failure are recorded in `doc/optimized_vllm/work_log.md`, but their stdout is not preserved as a separate artifact. The later final server run, final benchmark artifacts, and server shutdown evidence are sufficient for this stage review.

## Anomaly Ledger

- Observed anomaly: Stale debug qualitative output contains visibly bad responses.
  Evidence: `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json` includes repeated decorative text and repeated "Final Answer" tokens.
  Affected path: Pre-final TP4 on-device sampler investigation output.
  Control or comparison: `doc/optimized_vllm/after/vllm_qualitative_outputs.json` and `readiness_vllm/hf_no_think_greedy_256_controls.json` use the same rendered chat prompt shape and show coherent final/control outputs.
  Likely subsystem: Historical sampler/debug path before the final serving configuration.
  Investigation performed: Direct JSON inspection of final, stale debug, and HF control qualitative artifacts.
  Resolution: Controlled and correctly excluded from the optimized-vLLM evidence set.

- Observed anomaly: Active-trace allocation warning appears after the final non-aligned rerun.
  Evidence: `readiness_vllm/trace_warning_audit.log` records the warning after the final decode signpost and reports 0 decode or execute-trace calls after the warning.
  Affected path: Trace cleanup at idle/shutdown after non-aligned serving.
  Control or comparison: `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` releases model traces when the scheduler is idle and no pending completions remain; the audit shows no continued decode after the warning.
  Likely subsystem: Trace lifecycle cleanup after serving completion.
  Investigation performed: Read the audit log and inspected the plugin idle-reset path.
  Resolution: Controlled; not a serving correctness or measured-performance failure.

- Observed anomaly: First after-stage server attempt failed during startup on an active Ethernet core heartbeat timeout.
  Evidence: `doc/optimized_vllm/after_failed/server_active_eth_startup_failure.log` fails during mesh-device opening before model serving requests, with a timeout waiting for an active Ethernet core.
  Affected path: Device startup before the final after-stage run.
  Control or comparison: Final after-stage server log and benchmark artifacts complete with the same serving configuration.
  Likely subsystem: Device/firmware startup health, not Qwen adapter logic.
  Investigation performed: Read the failed startup log and compared it to the successful final run artifacts.
  Resolution: Controlled by subsequent successful run; no required stage work remains.

- Observed anomaly: Nanobind leaked-instance diagnostics appear after test/server shutdown.
  Evidence: `doc/optimized_vllm/after/sampling_tests.log` and `doc/optimized_vllm/after/server.log` both contain nanobind leak diagnostics after their useful pass/shutdown records.
  Affected path: Python extension teardown diagnostics.
  Control or comparison: Final server log reports user-mode device-driver close and cluster destructor completion; process-table check found no live vLLM/EngineCore process.
  Likely subsystem: Python/native binding lifetime diagnostics.
  Investigation performed: Inspected test/server tails and ran a read-only process-table check.
  Resolution: Non-blocking residual risk only.

## Scope Inspected

- Stage requirements: `stage-review`, `vllm-integration`, `optimize`, `tt-device-usage`, `tt-enable-tracing`, and `qualitative-check` skill files.
- Stage documents and summaries: `doc/optimized_vllm/README.md`, `doc/optimized_vllm/work_log.md`, `doc/optimized_vllm/perf_summary.json`, and `doc/context_contract.json`.
- Benchmark artifacts: `doc/optimized_vllm/before/vllm_benchmark.json`, `doc/optimized_vllm/before/vllm_result.json`, `doc/optimized_vllm/before/vllm_ci_serving_benchmark.json`, `doc/optimized_vllm/before/vllm_ci_serving_result.json`, and matching files under `doc/optimized_vllm/after/`.
- Serving/test artifacts: `doc/optimized_vllm/after/server.log`, `doc/optimized_vllm/after/sampling_tests.log`, `doc/optimized_vllm/after/vllm_qualitative_outputs.json`, `doc/optimized_vllm/after_failed/server_active_eth_startup_failure.log`, `readiness_vllm/non_aligned_prompt_len37_result.json`, `readiness_vllm/adapter_runtime_tests.log`, `readiness_vllm/trace_warning_audit.log`, `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json`, and `readiness_vllm/hf_no_think_greedy_256_controls.json`.
- TT model code: `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py`, `models/autoports/qwen_qwen3_4b/tt/generator.py`, `models/autoports/qwen_qwen3_4b/tt/model.py`, and `models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py`.
- vLLM plugin code: `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`, and `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py`.
- Read-only checks used: artifact inventory with `find`, JSON rederivation with `jq`, source/log inspection with `rg`, `sed`, `nl`, `head`, and `tail`, git revision/status checks, same-model reference search in the checkout and run roots, and a process-table check for leftover vLLM/EngineCore processes.

## Residual Risk

- This review did not rerun hardware, servers, vLLM experiments, profilers, watcher, or `tt-smi` commands, per instruction. The verdict is based on existing JSON/log/code artifacts and small read-only local analysis.
- The primary vLLM result is a one-request single-user benchmark, as required by the stage contract. It is adequate for the stage gate, while broader latency variance remains better represented by the secondary CI serving-burst artifact.
