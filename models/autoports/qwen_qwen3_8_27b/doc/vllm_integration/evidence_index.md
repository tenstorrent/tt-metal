# Stage09 evidence index

Paths are relative to the model root. Runtime source hashes are in each final
command manifest. Final independent review and local checkpoint records are in
`doc/vllm_integration/stage_review.md` and `work_log.md`.

| Contract | Final evidence |
|---|---|
| Stack and registration | `doc/vllm_integration/serving_stack.json`; `readiness_vllm/registration.log`; `final_b1_server.log`; `full_b32_performance_server.log` |
| Selected precision/native context | `doc/datatype_sweep/selected_precision_config.json`; `doc/context_contract.json`; `full_b32_budget131072_events.jsonl:1` allocation policy; final server policy logs |
| Canonical generator delegation/cache ownership | `tt/generator_vllm.py`; `tt/generator.py`; `tt/serving_state.py`; `tt/vllm_cache_specs.py`; `doc/vllm_integration/runtime_audit.md` |
| Reduced stale-input/state/trace proofs | `readiness_vllm/generator_device_contract.json`; `adapter_device_contract.json`; `generator_b1_device_contract.json`; `continuation_probe.json`; `state_pool_probe.json` |
| Full64 automatic allocator growth | `readiness_vllm/page_growth_async.json` and `.requests.json`; `page_growth_sync_control.json` and `.requests.json`; exact B8 command/source manifests and untruncated event histories |
| Final B32 mixed-shape/scheduler/trace lifecycle | `readiness_vllm/full_b32_budget131072_lifecycle.json`, `.requests.json`, `full_b32_budget131072_events.jsonl`, `full_b32_budget131072_command.json`, server log |
| Prefix hit rules and multi-turn reuse | `doc/agentic_serving.json:serving`; final B32 lifecycle; 32x4130 initial plus33 appended, not measured32x32768 |
| Full sampling profile | `readiness_vllm/full_b8_sampling_tests.log`; `full_sampling_runner.log`; `sampling_verdict.json`;76pass74skip, explicit host compatibility |
| Device sampling/logit reproducibility | `readiness_vllm/sampling_route_controls.json`; `standalone_route_logits.json`; all64/B32 exact top20/greedy controls; short standalone512 cache is correctness-only |
| Qualitative review/controls | `readiness_vllm/vllm_qualitative_outputs.json`; `vllm_qualitative_verdict.json`; `full_model_controls.json`; `full_model_control_verdict.json`; `degenerate_output_check.json` |
| Final primary4096/128/B1/concurrency1 | `readiness_vllm/final_b1_vllm_benchmark.json` paired with `final_b1_vllm_result.json`; `final_b1_command.json`; `final_b1_server.log` |
| Secondary CI100/100/32 | `readiness_vllm/vllm_ci_serving_benchmark.json` paired with `vllm_ci_serving_result.json`; immutable `full_b32_performance_` copies; `full_b32_performance_command.json` |
| Batch16 validation129/8/16/16 | `readiness_vllm/full_b16_vllm_benchmark.json` paired with `full_b16_vllm_result.json`; command/events/server archives; tracking enabled, not headline performance |
| Per-batch settings | `doc/vllm_integration/batch_configurations.json`; native context unchanged at every batch |
| Scheduler before/after latency | `doc/vllm_integration/b32_host_latency_analysis.md`; `readiness_vllm/full_b32_host_timeline.json/.png` and `full_b32_budget131072_host_timeline.json/.png`; host wall only |
| Runtime cleanup | `readiness_vllm/final_cleanup.json`; shutdown audits in final server logs; no live serving/device holders in container |

Normalized runner archives retain their original generic `raw_result_file` field.
The immutable pairings above are authoritative for archived runs; canonical
`vllm_benchmark.json`/`vllm_result.json` now contain the final B1 target.
Earlier `full_b1_` copies are a separate pre-finalization-fix result, not the final
headline. FixedB32 single-request4096/128 is a regression row, not B1 performance.

The exact-identifier compact telemetry packet is
`/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/4362e94e-730c-4b3c-840b-f8f182e1bca1.json`.
It uses final4096/128/B1/concurrency1 vLLM measurements only. Device metrics remain
unknown because live-serving profiling is prohibited.
