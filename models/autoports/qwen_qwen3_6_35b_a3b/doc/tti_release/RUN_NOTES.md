# Qwen3.6-35B-A3B TTI Release Notes

## Scope
- Model: `Qwen/Qwen3.6-35B-A3B`.
- Stage: Stage 11 TTI release handoff.
- Evaluated implementation: `models/autoports/qwen_qwen3_6_35b_a3b`.
- Workflow: no-Docker TTI `release` against an already-running autoport vLLM OpenAI-compatible server.
- Reservation/container host: `qb2-120-p02t01`.
- Device target: `P300X2`; reservation-container server used a 4-chip `P300C` mesh.
- Docker fallback: not used.
- Readiness status: `release-readiness-ci-subset-pass`.

## Source State
- `tt-metal` SHA used by TTI report metadata: `f2bdeec2f41255c106b859601e094ad2cfa9ecca`.
- `vllm` SHA: `b2d90800d77ba04a54462dad1384641d17e1db47`.
- `tt-inference-server` version: `0.19.0`.
- `tt-inference-server` SHA: `872bbdf6563db2c66c4cff7a37b10df218a9b406`.
- Runtime spec: `specs/qwen36_35b_a3b_autoport_release_runtime_spec.json`.
- Final copied runtime spec: `artifacts/final7/runtime_model_spec.json`.
- Final release report: `reports/tti_release_final7_report.md`.
- Final report data: `reports/tti_release_final7_report_data.json`.
- Canonical regenerated report files: `reports/report_id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release_2026-08-21_13-26-34.md` and `reports/data/report_data_id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release_2026-08-21_13-26-34.json`.

## Context Contract
- `doc/context_contract.json` preserves `supported_context=262144`.
- TTI release spec uses `max_context=262144`, `max_tokens_all_users_override=262144`, and vLLM `max_model_len=262144`.
- Final runtime spec `artifacts/final7/runtime_model_spec.json` records `max_model_len=262144`, `max_num_batched_tokens=262144`, and `max_context=262144`.
- No release, benchmark, or eval context/request cap was added to hide model behavior.
- Non-aligned prompt smoke passed with 33 prompt tokens, not divisible by 32, 64, or 128: `artifacts/non_aligned_openai_chat_check.json`.

## Server Mode
- Server mode: external no-Docker autoport vLLM server.
- Service port: `8031`.
- Server command source: optimized-vLLM autoport server, not TTI Docker.
- Key vLLM args: `max_num_seqs=32`, `max_model_len=262144`, `block_size=32`, `--async-scheduling`, `--served-model-name Qwen/Qwen3.6-35B-A3B`, `--enable-auto-tool-choice`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`.
- TT config: `trace_region_size=384000000`, `fabric_config=FABRIC_1D_RING`.
- Prompt/API format: OpenAI chat completions with Qwen chat template and `qwen3` reasoning/parser settings. Prompt-format evidence: `artifacts/qualitative_prompt_format.json`, `artifacts/openai_chat_smoke_response.json`, and `artifacts/non_aligned_openai_chat_response.json`.
- Server import proof: `readiness_vllm/server.log` records `models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm:Qwen3_5MoeForConditionalGeneration`.
- Autoport implementation check: `artifacts/final7/runtime_model_spec.json` records `impl.code_path=models/autoports/qwen_qwen3_6_35b_a3b`, `impl_name=qwen36-autoport`, `server_mode=external_no_docker_autoport_vllm`, `docker_server=false`, `local_server=false`, and `service_port=8031`.
- Stock implementation check: copied final report data and runtime spec do not identify `models/tt_transformers`, `models/demos`, or another packaged implementation as the evaluated path.

## Smoke
- Health check: `artifacts/openai_health_8031.txt`, `logs/openai_health_8031.log`.
- OpenAI chat smoke: `artifacts/openai_chat_smoke_response.json`, HTTP 200, `prompt_tokens=19`, `completion_tokens=8`.
- Stage-review-fix OpenAI smoke summary: `logs/openai_smoke_stage_review_fix.json`, `finish_reason=stop`, `content_chars=22`.
- Non-aligned OpenAI chat smoke: `artifacts/non_aligned_openai_chat_check.json`, HTTP 200, `prompt_tokens=33`, `completion_tokens=8`.
- TTI no-Docker smoke benchmark with `disable_trace_capture=true`: `logs/tti_smoke_benchmarks_retry.log`, `isl=8`, `osl=8`, `concurrency=1`, acceptance PASS.

## Release Command
```bash
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")" \
CACHE_ROOT=/localdev/vkovacevic/tti-release/qwen36_35b_a3b_20260820/release_cache_final7 \
SERVICE_PORT=8031 \
ONLY_BENCHMARK_TARGETS=1 \
TT_METAL_HOME=/localdev/vkovacevic/tt-metal \
vllm_dir=/localdev/vkovacevic/vllm \
python3 run.py \
  --model Qwen3.6-35B-A3B \
  --runtime-model-spec-json /localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/tti_release/specs/qwen36_35b_a3b_autoport_release_runtime_spec.json \
  --tt-device p300x2 \
  --workflow release \
  --service-port 8031 \
  --tools vllm \
  --no-auth \
  --skip-system-sw-validation \
  --limit-samples-mode ci-nightly \
  --disable-trace-capture
```

Key environment notes:
- `HF_TOKEN` was loaded from the local Hugging Face cache, but the token value was not printed or copied.
- `TT_METAL_HOME=/localdev/vkovacevic/tt-metal`.
- `vllm_dir=/localdev/vkovacevic/vllm`.
- `ONLY_BENCHMARK_TARGETS=1` limited benchmark rows to target rows only.
- No caches, weights, Docker layers, persistent TT caches, or raw eval sample JSONL files were copied into this handoff directory.

## Final Release Result
- TTI final run: `release_cache_final7`, completed `2026-08-21T13:26:34Z`, `run.py` rc=0.
- Final report: `reports/tti_release_final7_report.md`.
- Final report data: `reports/tti_release_final7_report_data.json`.
- TTI acceptance: PASS, 0 blockers.
- Benchmarks: PASS, 2/2 target rows.
- Evals: PASS, 1/2 passed and 1/2 issue-waived.
- Spec tests: PASS, `LoggerForkSafetyTest` and `VLLMParamConformanceTest`; 22 pytest cases passed in 3256.83s.

Benchmark rows:
- ISL 128 / OSL 128 / concurrency 1: TTFT 6132.6 ms, decode throughput 9.29 tok/s, target PASS.
- ISL 100 / OSL 100 / concurrency 32: mean TTFT 150266.1 ms, decode throughput 12.82 tok/s, target PASS.

Eval rows:
- `leaderboard_ifeval`: PASS on the `ci-nightly` 0.05 subset. Score 89.285714% prompt-level strict, ratio to published/reference 0.9591, tolerance 0.05. Sample summary: `artifacts/final7/ifeval_sample_summary.json`.
- `r1_gpqa_diamond`: issue-waived because `Idavidrein/gpqa` is gated in this environment and failed before inference. Evidence: `logs/tti_release_ci_nightly_final7.log` lines 123-171, upstream lm-evaluation-harness GPQA README, Hugging Face dataset page, Hugging Face gated-dataset docs, and `WAIVERS.md`.
- No `meta_ifeval` or `meta_gpqa_cot` rows were present in the final release report.

Report regeneration:
- The expensive final7 eval/benchmark/spec-test sections were not rerun.
- After stage-review found stale `final5` waiver text in the generated report data, the final7 report was regenerated with TTI `ReportGenerator` from the final7 raw section data only to refresh GPQA waiver evidence and point report metadata at the exact copied runtime spec.
- Regenerated report metadata records `report_regeneration_reason=Refresh r1_gpqa_diamond gated-dataset waiver evidence; final7 raw section data unchanged.`

CI-subset note:
- `--limit-samples-mode ci-nightly` propagated to lm-eval as `--limit 0.05`.
- IFEval used 28 sampled rows and took about 3h22m on this server path. Scaling by sample count estimates unrestricted IFEval alone at roughly 65 hours, before GPQA. That made the unrestricted suite impractical for this Stage 11 handoff window.
- Accuracy numbers in this directory are CI-subset results, not unrestricted full-set release accuracy.

## Recovery
- A prior restart hit an active Ethernet timeout while waiting on core `29-25`.
- Recovery was done in the reservation container with `tt-smi -r`, then `tt-smi -ls --local`, then a TTNN mesh open/close smoke.
- Recovery artifacts:
  - `logs/tt_smi_reset_after_stage_review_fix_active_eth.log`
  - `logs/tt_smi_list_after_stage_review_fix_reset.log`
  - `logs/mesh_smoke_after_stage_review_fix_reset.log`
- No Docker fallback or loudbox host-level recovery was used.

## Copied Artifacts
- `reports/tti_release_final7_report.md`
- `reports/tti_release_final7_report_data.json`
- `reports/report_id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release_2026-08-21_13-26-34.md`
- `reports/data/report_data_id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release_2026-08-21_13-26-34.json`
- `artifacts/final7/runtime_model_spec.json`
- `artifacts/final7/release_summary.json`
- `artifacts/final7/ifeval_sample_summary.json`
- `artifacts/final7/leaderboard_ifeval_results_aggregate.json`
- `artifacts/final7/benchmark_isl128_osl128_conc1.json`
- `artifacts/final7/benchmark_isl100_osl100_conc32.json`
- `logs/tti_release_ci_nightly_final7.log`
- `logs/tti_spec_tests_stage_review_fix_pass.log`
- `logs/tti_evals_stage_review_fix.log`
- `logs/pytest_penalty_fix_live.log`

## Cleanup
- The autoport vLLM server was stopped after final release artifact inspection.
- `http://127.0.0.1:8031/health` returned no service after cleanup.
- No TTI release client, `run_workflows.py`, `lm_eval`, `vllm bench`, autoport vLLM server, `api_server`, or `VLLM::EngineCore` process remained after cleanup.
- Final device list check after server shutdown: `logs/tt_smi_list_final_after_release_cleanup.log`, all 4 P300C chips visible.
- No TTI Docker fallback container was created. A pre-existing long-lived `tt-xla-dev:local` container was observed and left untouched because it was not created by this workflow.
- No tmux session was created by this workflow. Pre-existing workspace sessions `0` and `exp24-qwen36-codex` were observed and left untouched.

## Stage Review And Commits
- Stage-review status: clean-pass from subagent `01a02492-8da3-7253-979d-b4f620ff6f2d`.
- Prior stage-review finding from `01a02486-d71a-7002-bbed-7f28fd7228ba` was fixed by refreshing the GPQA waiver evidence and regenerating final7 report artifacts from final7 raw section data.
- `tt-metal` release handoff commit: pending.
- `tt-inference-server` release wiring commit: `4d91237b85a40bc45396cc2b5b2fe8d097277dde`.
