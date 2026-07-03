# Qwen/Qwen3-4B TTI Release Notes

Date: 2026-07-03

Status: blocked before final release report. The no-Docker autoport smoke passed, but the full release workflow could not complete because the release eval attempted to download the gated Hugging Face dataset `Idavidrein/gpqa` and the available token could not download the dataset files. This failed before model generation requests were sent for that eval.

Release-readiness status: `release-workflow-fail`.

## Host And Context

- tt-metal checkout: `/home/ubuntu/tt-metal`
- tt-metal SHA: `affc17f0d3bba0388b27e8fbc8853ea0aef3e421`
- Autoport implementation: `models/autoports/qwen_qwen3_4b`
- HF model: `Qwen/Qwen3-4B`
- Hardware: local reservation container, `P150X4`
- Context contract: `40960`, from `models/autoports/qwen_qwen3_4b/doc/context_contract.json`
- vLLM checkout: `/home/ubuntu/vllm`, SHA `de6c44fd89154bd800c8c947e7205876b93013e3`
- TTI checkout: `/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/tt-inference-server`
- TTI version: `0.17.0`
- TTI SHA: `3a7c6a021dac5bcc94ba09b0bcd52be72615d683`
- Docker: not used

## Server Mode

Mode: external OpenAI-compatible autoport vLLM server on port `8000`; TTI ran client-side with `docker_server=false` and `local_server=false`.

Server command:

```bash
TT_QWEN3_TEXT_VER=autoport_qwen3_4b \
VLLM_PLUGINS=tt,tt_model_registry \
QWEN3_4B_AUTOPORT_DIR=/home/ubuntu/tt-metal/models/autoports/qwen_qwen3_4b \
PYTHONPATH=/home/ubuntu/vllm:$PYTHONPATH \
MESH_DEVICE=P150x4 \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --block_size 32 \
  --max_num_seqs 32 \
  --port 8000 \
  --max_model_len 40960 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 134217728, "fabric_config": "FABRIC_1D_RING"}}'
```

Key environment: `TT_QWEN3_TEXT_VER=autoport_qwen3_4b`, `VLLM_PLUGINS=tt,tt_model_registry`, `QWEN3_4B_AUTOPORT_DIR=/home/ubuntu/tt-metal/models/autoports/qwen_qwen3_4b`, `MESH_DEVICE=P150x4`, `PYTHONPATH=/home/ubuntu/vllm:$PYTHONPATH`. `HF_TOKEN` was present for dataset/model access, but no token value was printed or copied.

## Runtime Specs

Custom runtime specs were used so TTI evaluated the generated autoport instead of stock `tt-transformers`, `models/demos`, or another packaged implementation.

Autoport implementation check: PASS for the smoke and release attempts. The copied TTI specs identify `models/autoports/qwen_qwen3_4b` as the implementation path and do not identify stock `models/tt_transformers`, `models/demos`, or another packaged implementation.

- Smoke input spec: `specs/autoport_smoke_runtime_model_spec.json`
- Release input spec: `specs/autoport_release_runtime_model_spec.json`
- Captured smoke run spec: `specs/smoke_runtime_model_spec_captured.json`
- Captured release run spec: `specs/release_runtime_model_spec_captured.json`

The captured specs prove:

- `runtime_model_spec.impl.code_path = "models/autoports/qwen_qwen3_4b"`
- `runtime_model_spec.device_model_spec.max_context = 40960`
- `runtime_config.docker_server = false`
- `runtime_config.local_server = false`
- `runtime_config.service_port = "8000"`
- `runtime_model_spec.metadata.autoport_dir = "models/autoports/qwen_qwen3_4b"`
- `runtime_model_spec.metadata.context_contract_path = "models/autoports/qwen_qwen3_4b/doc/context_contract.json"`

## Smoke Results

Health check: passed.

OpenAI-compatible request: passed. Log: `logs/external_server_openai_smoke.log`.

Small TTI benchmark with trace capture disabled: passed.

Command:

```bash
ONLY_BENCHMARK_TARGETS=1 CACHE_ROOT=/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/smoke_tti_cache_target_only \
python3 run.py \
  --model Qwen3-4B \
  --runtime-model-spec-json /home/ubuntu/tti-release/qwen_qwen3_4b_20260703/autoport_smoke_runtime_model_spec.json \
  --tt-device p150x4 \
  --workflow benchmarks \
  --service-port 8000 \
  --server-url http://127.0.0.1 \
  --no-auth \
  --skip-system-sw-validation \
  --disable-trace-capture
```

Smoke report: `reports/smoke_benchmark_report.md`.

Smoke benchmark JSON: `reports/smoke_benchmark_isl8_osl8.json`, with `completed=1`, `failed=0`, ISL `8`, OSL `8`, concurrency `1`.

## Release Attempt

Command:

```bash
CACHE_ROOT=/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/release_tti_cache_final \
python3 run.py \
  --model Qwen3-4B \
  --runtime-model-spec-json /home/ubuntu/tti-release/qwen_qwen3_4b_20260703/autoport_release_runtime_model_spec.json \
  --tt-device p150x4 \
  --workflow release \
  --service-port 8000 \
  --server-url http://127.0.0.1 \
  --no-auth \
  --skip-system-sw-validation
```

Release log: `logs/tti_release_final.log`.

Copied release run log: `logs/tti_release_final_run.log`.

Copied release runtime spec: `specs/release_runtime_model_spec_captured.json`.

Release failed in `r1_gpqa_diamond` with:

```text
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub.
```

The failure occurred while `lm_eval` loaded task data, before model execution. `mmlu_pro` began after the first failure and was stopped to avoid a long burn once the release was already blocked. The TTI Qwen3-4B release target in this checkout used `r1_gpqa_diamond` and `mmlu_pro`; no `meta_ifeval` or `meta_gpqa_cot` target was configured for this runtime spec.

No final customer release report was produced.

Fresh access recheck: `logs/gpqa_access_recheck_20260703T074315Z.log`. In the exact TTI eval virtualenv, `datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)` failed in all tested modes: plain environment, `token=True`, and explicit token value. The log records only token presence booleans and does not include token contents.

Local cache audit: `logs/gpqa_local_cache_audit_20260703T074550Z.log`. The Hugging Face hub cache for `datasets--Idavidrein--gpqa` contains only `README.md`, `refs/main`, and one 3.3 KB blob; there are no dataset files under `/home/ubuntu/.cache/huggingface/datasets`. Rerunning release with the current token/cache would hit the same pre-model dataset access failure.

## Context And Prompt Handling

The release path preserved the `40960` context contract. The first full release attempt exposed a TTI harness bug where eval tasks requested `max_length=65536`; this was not a model bug. A local TTI patch clamps eval `max_length` to `device_model_spec.max_context`, and the final release log shows both `r1_gpqa_diamond` and `mmlu_pro` using `max_length=40960`.

The release eval command used `--apply_chat_template`, and the smoke used the OpenAI chat endpoint. The existing optimized-vLLM artifacts already include non-aligned prompt length evidence, so no benchmark prompt was aligned or capped to hide a generator issue.

## Recovery

Initial autoport vLLM startup failed before model execution with an active Ethernet core timeout:

```text
Device 0: Timed out while waiting for active ethernet core 31-25 to become active again
```

Recovery used the reservation container:

- `tt-smi -ls --local`
- `tt-smi -r`
- `tt-smi -ls --local`
- `ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=0)` mesh smoke

Logs: `logs/recovery_reset_1.log`, `logs/recovery_mesh_smoke_1.log`, `logs/autoport_vllm_server.failed_eth_core.log`.

## Harness Fixes

Local TTI checkout fixes were needed for the no-Docker autoport spec path:

- `tt-inference-server-v2/workflow_module/command_factory.py`: v2 workflow now loads `ModelSpec.from_json(args.runtime_model_spec_json)` instead of always querying the catalog, allowing the custom autoport `p150x4` runtime spec.
- `tt-inference-server-v2/llm_module/eval_command.py`: eval `model_kwargs.max_length` is clamped to `device_model_spec.max_context`.

Diff artifact: `diagnostics/tti_local_fixes.diff`.

Autofix/autodebug notes: `diagnostics/AUTODEBUG.md`, `AUTOFIX.md`.

## Cleanup

The external vLLM tmux session and TTI release tmux session were stopped. No Docker containers were left running. Only the pre-existing `auto-0` tmux session remained. Two defunct `lm_eval` processes were left under PID 1; they had already exited and could not be killed directly.

## Stage Review And Commit

`stage-review` was not run because the release stage did not reach a passable final report state. No local tt-metal commit was made because the stage is blocked on external gated dataset access and there is no `clean-pass` review.
