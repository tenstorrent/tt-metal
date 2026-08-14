# Mistral Small 24B Instruct 2501 — TTI release handoff

## Status

- Final classification: `release-workflow-pass/readiness-fail`.
- The generated autoport server, CI-subset eval collection, acceptance-target benchmark, and API/spec conformance all completed operationally.
- The integrated local report is mechanically `PASS` with zero blockers because two task-scoped `known_issues` masks are declared. They are not customer-readiness waivers: no current linked issue proves the canonical implementation fails the same mandatory IFEval and GPQA rows.
- Unwaived quality gaps remain: IFEval 75.6635% versus 78.755%, and GPQA flexible-extract 38.8889% versus 40.3% (35/90, two answers short).

## Autoport implementation check

- Target implementation: `models/autoports/mistralai_mistral_small_24b_instruct_2501` — matched.
- Copied release spec `impl.code_path`: `models/autoports/mistralai_mistral_small_24b_instruct_2501` — matched.
- Server selector: `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport` — matched.
- HF model: `mistralai/Mistral-Small-24B-Instruct-2501`.
- No stock `models/tt_transformers` or `models/demos` implementation was used.

## Topology and versions

- Server mode: external OpenAI-compatible autoport vLLM API; no Docker.
- Reservation host: `qb2-120-p04t04`.
- Server port: 8000. Durable exec session ID for v9: 71818; no tmux session was used.
- tt-metal base: `1529e332a1c37937a682ba04b77e7dc3418f2589` on `mvasiljevic/fast-models/mistralai-mistral-small-24b-instruct-2501`.
- nested vLLM fix commit: `0e5e6495ac0a39e7c16a925140547cfb4a2e3030` (based on slot-release commit `fe721dae82367bb154f9de80f7363264d7b84163`).
- tt-inference-server fix/report commit: `3e933fbbaaf71fd4859b017f8f08570e39834c09` on `main`; checkout description before commit was `v0.10.0-1096-gbfad4a69`.
- Docker image/version: not applicable.

## Context and server configuration

- The serving limit remained 32768, matching `doc/context_contract.json`; no prompt, page, trace, or benchmark alignment reduction was made.
- v9 server: P300X2, fixed batch 32, block size 32, `max_num_seqs=32`, `max_num_batched_tokens=32768`, engine seed 9472, device sampling `all`, 200,000,000-byte trace region, and `FABRIC_1D`.
- Material environment: `HF_HOME=/home/mvasiljevic/hf-cache`, `HF_HUB_CACHE=/home/mvasiljevic/hf-cache/hub`, `HF_HUB_OFFLINE=1`, `VLLM_TARGET_DEVICE=tt`, `ARCH_NAME=blackhole`, `MESH_DEVICE=P300x2`, `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport`, `VLLM_RPC_TIMEOUT=900000`, and `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`.

Server command (shell quoting normalized):

```bash
python_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-24B-Instruct-2501 \
  --block_size 32 --max_num_seqs 32 --max_num_batched_tokens 32768 \
  --max-log-len 32 --port 8000 --max_model_len 32768 --seed 9472 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}}'
```

## Release commands and sampling scope

Release client command:

```bash
CACHE_ROOT=/home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9 \
ONLY_BENCHMARK_TARGETS=1 HF_HOME=/home/mvasiljevic/hf-cache \
HF_HUB_CACHE=/home/mvasiljevic/hf-cache/hub HF_HUB_OFFLINE=1 \
timeout 10800 python run.py \
  --runtime-model-spec-json runtime_specs/mistral_small_24b_2501_autoport_release.json \
  --workflow release --tt-device p300x2 --tools vllm --no-auth \
  --server-url http://127.0.0.1 --service-port 8000 \
  --tt-metal-home /home/mvasiljevic/tt-metal \
  --vllm-dir /home/mvasiljevic/tt-metal/vllm \
  --limit-samples-mode ci-nightly --skip-system-sw-validation \
  --disable-trace-capture
```

- Effective eval limits were 0.2 for IFEval (109 samples) and 0.2 for GPQA (90 samples). Every accuracy result in this directory is a CI-subset result, not unrestricted full-set accuracy.
- Based on measured v9 time (18m32s IFEval, 1h13m04s GPQA, 19m53s spec tests), a linear unrestricted estimate is approximately 8 hours before setup/report overhead. This exceeded the three-hour release watchdog/reservation window, so `ci-nightly` was used.
- v9's initial benchmark failed before requests because the benchmark venv's cached Transformers 5 Mistral tokenizer lacked `is_fast`. After the compatibility fix, the exact acceptance-target benchmark was rerun separately:

```bash
ONLY_BENCHMARK_TARGETS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.workflow_venvs/.venv_llm_vllm/bin/python run_workflows.py \
  --model Mistral-Small-24B-Instruct-2501 --workflow benchmarks \
  --device p300x2 --service-port 8000 --server-url http://127.0.0.1 \
  --tools vllm --runtime-model-spec-json <v9-runtime-model-spec.json> \
  --output-dir <v9-benchmark-fixed-output>
```

It completed 8/8 requests with zero failures: mean TTFT 1272.74ms, mean TPOT 19.19ms, and decode throughput 34.50 tokens/s. All configured functional checks passed.

## Prompt format and qualitative integrity

- Both evals used the HF-declared instruct chat template (`--apply_chat_template`); raw-completion prompting was not used.
- GPQA used five official demonstrations and a held-out question. Few-shot assistant targets minimally demonstrate `The answer is (X).`; the held-out scored target remains bare `(X)`.
- The strict parser accepts only `The answer is ([A-D])`; flexible extraction supplies the multiple-choice accuracy gate. Strict remains a format diagnostic.
- Exact seed-42 prompt metadata records five answer markers, six reasoning prompts, and 1285 tokens (20 tokens / 1.581% above the same-prompt bare-target baseline).
- Private qualitative smokes checked hashes, parser placement, language/control-token integrity, and deterministic repetition. No raw samples or completions are copied here.

## Fixes and verification

- Proven production fixes retained: prior request-slot release; async decode drains pending token/position feedback before preemption layout repack; async scheduling reserves one lookahead KV token; cached Mistral tokenizer exposes the missing `is_fast` compatibility property without tokenizer or chat-template substitution; GPQA few-shot/parser and flexible accuracy key corrections.
- Refuted sorted-allocation and page-map telemetry source candidates and their tests were reverted before commit.
- Focused host tests: TT async scheduler/preemption 5/5; tokenizer adapter 1/1; GPQA/config 3/3; acceptance waiver 2/2; model-spec known-issue tests 11/11.
- v9 GPQA completed 90/90 through 13 preemptions with health HTTP 200 and zero request, transport, page, slot, retry, EngineCore, or fatal errors. This closes the original 32/90 page-boundary crash independently of accuracy.
- Spec/API conformance passed 2/2 report blocks and all 22 parametrized vLLM chat-completion cases.

## Failed rows and classification

- `meta_ifeval`: 75.6635% versus unchanged 78.755% floor — `readiness-fail`.
- `meta_gpqa_cot`: 38.8889% flexible-extract versus unchanged 40.3% floor — `readiness-fail`.
- Runtime-spec `known_issues` entries are task-scoped local acceptance masks with exact measured evidence. No numeric threshold or workflow-wide gate changed. No canonical-control issue URL exists, so neither mask is claimed as a valid customer release waiver.
- Benchmark tokenizer harness failure — `fixed` and rerun successfully.
- GPQA page-boundary/preemption failure — `fixed` and rerun successfully.

## Recovery and cleanup

- Diagnostic forced stops occasionally left ARC/remote-Ethernet state; recovery used bounded `tt-smi -ls --local`, `tt-smi -r`, `tt-smi -ls --local` sequences and same-command relaunches. v8 was discarded as a dead external-endpoint artifact after its server exited cleanly during readiness.
- v9 itself remained device healthy throughout eval, benchmark, and spec traffic. No Docker container was created.
- Final cleanup: server/client processes and `autoport-vllm-mistral-tti` tmux session stopped; TTI `.env` removed. `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls --local`, `tt-smi -r`, then `tt-smi -ls --local` completed successfully; all four Blackhole p300c UMD chips (0--3) were visible and resettable afterward.

## Copied artifacts

- `final_release_report.md` / `.json`: integrated v9 eval/spec plus post-fix benchmark evidence and declared masks.
- `benchmark_report.md` / `.json`, `benchmark_raw_v9_fixed.json`, and `benchmark_smoke_v9_fixed.json`.
- `ifeval_v9_results.json`, `gpqa_v9_results.json`, and `eval_v9_aggregate_metadata.json` (aggregate/config/hash data only).
- `release_spec.json`, `gpqa_prompt_format_v2.json`, and `gpqa_v5_enginecore_failure_metadata.json`.
- `client_release_v9.log` and `server_release_v9.log`.
- No raw eval sample JSONL, caches, weights, tensor dumps, profiler bulk, `.env`, or secrets were copied.

The integrated report was regenerated from the valid v9 eval/spec report blocks and the post-fix standalone benchmark report block; its metadata names both evidence sources. It does not claim that the expensive evals were rerun after the tokenizer-only benchmark-client fix, because that fix cannot affect server model outputs.
