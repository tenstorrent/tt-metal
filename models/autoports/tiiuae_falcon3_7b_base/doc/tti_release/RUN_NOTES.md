# Falcon3-7B-Base TTI release run notes

## Status

Stage 11 has a context-preserving nightly-equivalent no-Docker release PASS
against the generated autoport. The current passing report is
`report_tiiuae__Falcon3-7B-Base_2026-08-13T105032+0000.md`; the detailed final
run is documented in the last section of this file. The first review's
2,047-token lm-eval adapter
finding was fixed: the regenerated release log records `Using max length 32768
- 1` for both mandatory evals and ends with zero blockers. Final closure is
subject only to the independent rereview recorded in `STAGE_REVIEW.md`.

The earlier `release_report_invalid.md` and corrected FAIL render are retained
only as historical diagnostics. They capture the former publisher-reference
and GPQA-access blockers and must not be confused with the replacement nightly
release evidence.

## Evaluated implementation and server

- Model: `tiiuae/Falcon3-7B-Base` (Base completion model).
- Generated implementation: `models/autoports/tiiuae_falcon3_7b_base`.
- vLLM adapter imported by the server:
  `models.autoports.tiiuae_falcon3_7b_base.tt.generator_vllm`.
- Runtime spec proof: `release_runtime_spec.json` records
  `impl.code_path=models/autoports/tiiuae_falcon3_7b_base`, the matching
  `code_link`, and the autoport generator path. No stock `models/tt_transformers`,
  `models/demos`, or packaged implementation was evaluated.
- Autoport implementation check: PASS — `models/autoports/tiiuae_falcon3_7b_base` is selected.
  Copied artifacts identify no stock or different autoport implementation.
- Server mode: external, no Docker, no local TTI server; OpenAI-compatible
  autoport vLLM at `http://127.0.0.1:8000` inside reservation container
  `b30c965c728b`.
- Device: P300X2 / four P300 chips as a 1x4 mesh.
- Context contract: 32,768 tokens from `../context_contract.json`; server and
  specs used `max_model_len=32768`, `max_num_batched_tokens=32768`, block size
  32, and max concurrency 32. No context or request-length cap was introduced.
- Prompt mode: raw `/v1/completions`; the checkpoint has no tokenizer chat
  template. The TTI benchmark client used `/v1/chat/completions` with the
  autoport's passthrough template solely for client compatibility.

## Versions and host configuration

- tt-metal starting SHA: `053fb3f6362189a6fae76632143ec8faa569f532`.
- TTI release-base SHA: `c8509ac20b4cf179710caba94ba34ed54abc6c00`.
- TTI local AutoFix commit: `ca152fe2` (`Support autoport external-server
  release specs`), not pushed.
- TTI describe: `v0.10.0-1092-gc8509ac2`; `run.py` reported version `0.19.0`.
- Docker image: none.
- Key non-secret environment: `SERVICE_PORT=8000`; isolated TTI caches were
  rooted under `/home/mvasiljevic/tti-release-cache/`; local model weights came
  from the existing Hugging Face cache. Credentials, weights, persistent TT
  caches, Docker layers, and raw eval sample dumps were not copied here.

## Hardware recovery

Initial 1x4 mesh open failed with an ERISC Ethernet heartbeat timeout on device
0. A bounded reservation-container `tt-smi -r` reset was followed by device
listing and a successful 1x4 mesh open/close probe. This was the first reset;
no loudbox-host recovery or Docker fallback was used.

Exact bounded recovery sequence (all in the reservation container):

```text
timeout 60 tt-smi -ls --local
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
# rerun the same 1x4 open/close Python probe
```

The first mesh probe failed at ERISC heartbeat initialization; reset and list
completed, and the repeated probe printed `MESH_SMOKE_OK` and exited 0.

Exact server launch command:

```text
python -m models.common.readiness_check.run_vllm_server --stages serve --model-dir models/autoports/tiiuae_falcon3_7b_base --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 --mesh-device P300x2 --port 8000 --max-num-seqs 32 --max-model-len 32768 --block-size 32 --tt-config '{"trace_region_size":512000000,"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all"}' --additional-server-args '--served-model-name tiiuae/Falcon3-7B-Base --chat-template models/autoports/tiiuae_falcon3_7b_base/doc/vllm_integration/base_chat_template.jinja' --server-timeout 1200
```

## Required smoke

Before release:

1. `/health` returned HTTP 200.
2. One raw OpenAI-compatible completion request returned a non-empty completion.
   The prompt was 5 tokenizer tokens and requested 8 completion tokens (13
   total), proving a valid length not divisible by page/tile/trace sizes works.
   `api_smoke_summary.json` retains only response structure/status and token
   counts; generated text was intentionally not copied.
3. The TTI no-Docker benchmark ran with trace capture disabled and passed:
   1/1 successful request, ISL 8, OSL 8, concurrency 1, mean TTFT 194.43 ms,
   TPOT 45.34 ms, and no failed requests. See `smoke_report.md`,
   `smoke_report_data.json`, `smoke_benchmark.json`, and `smoke_run.log`.

Smoke command:

```text
CACHE_ROOT=/home/mvasiljevic/tti-release-cache/falcon3-base-smoke SERVICE_PORT=8000 python3 run.py --workflow benchmarks --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_smoke_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test
```

## Unrestricted release attempt

Command:

```text
CACHE_ROOT=/home/mvasiljevic/tti-release-cache/falcon3-base-release SERVICE_PORT=8000 python3 run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture
```

Performance completed all 13 sweep points with zero request failures. The
graded 128/128/concurrency-1 row passed all configured tiers: mean TTFT 200.8
ms, user throughput 62.1 tokens/s, and aggregate decode throughput 57.0
tokens/s. The sweep included ISL 16,384 and concurrency up to 32 without
reducing the 32K server contract. See `release_target_benchmark.json`.

The release report is nevertheless invalid and blocking:

- `ifeval`: prompt-level strict score 18.67 versus publisher value 34.3. The
  four observed aggregates were prompt strict 18.67, instruction strict 30.94,
  prompt loose 19.96, and instruction loose 32.49. The publisher does not state
  which aggregate, harness revision, BOS policy, or generation recipe produced
  34.3. A same-command HF/GPU control is required before attributing the gap to
  TT or changing the metric/reference; changing the result key merely to pass
  is forbidden.
- `gpqa_diamond_generative_n_shot`: did not run because the current token owner
  lacks authorized file access to gated `Idavidrein/gpqa`. Metadata visibility
  was available, but the eval venv's direct dataset load was denied. No model
  request or score was produced, so this row remains blocking.
- Spec tests: no matching custom-model suite was discovered and the row was NA.
  This does not waive the accuracy failures.

This TTI checkout's `ReleaseWorkflow` explicitly contains only `evals`,
`benchmarks`, and `spec_tests`; it does not emit a separate API-conformance
child. For this custom model, suite discovery also found no matching spec-test
matrix. The retained API coverage is therefore the direct health/completion
smoke (`api_smoke_summary.json`) plus successful OpenAI benchmark requests,
not a hidden or silently dropped report section. A final release should add a
custom spec-test matrix if Stage 11 policy requires parameter-level API rows;
the current NA is disclosed and is not used to excuse accuracy failures.

The copied files are named `release_report_invalid.*` and
`release_run_invalid.log` intentionally. They must not be presented as release
handoff success. A replacement no-Docker release must run after GPQA access and
same-command IFEval reference evidence are available.

## TTI client repairs

Local, hardware-free AutoDebug/AutoFix work in the TTI checkout repaired:

- top-level and nested custom runtime-spec parsing without catalog membership;
- preservation of embedded external-server workflow/port controls;
- remote benchmark and eval readiness using the canonical host-plus-port URL;
- propagation of `disable_trace_capture` to the benchmark runner;
- release acceptance so model maturity never silently waives failed evals.

Focused/relevant evidence includes 90 top-parser tests, 279 nested-workflow
tests, 331 remote/trace-path tests from the repair loop, 27 focused eval and
benchmark caller tests, and 50 report-acceptance tests. The TTI checkout's
`AUTODEBUG*.md` and `AUTOFIX.md` contain the source-level evidence.

Final focused test command at TTI commit `ca152fe2`:

```text
python3 -m pytest -q tests/test_run_arguments.py tests/test_run_workflows_arguments.py tests/report_module/test_acceptance_criteria.py tests/test_module/llm_tests/test_llm_eval_tests.py tests/test_module/llm_tests/test_llm_performance_tests.py
```

Result: 175 passed, one benign pytest collection warning, zero failures.

## Copied artifact inventory

- Inputs/provenance: `autoport_smoke_spec.json`, `autoport_release_spec.json`,
  `smoke_runtime_spec.json`, `release_runtime_spec.json`.
- Smoke: `api_smoke_summary.json`, `smoke_report.md`,
  `smoke_report_data.json`, `smoke_benchmark.json`, `smoke_run.log`.
- Invalid unrestricted diagnostic run: `release_report_invalid.md`,
  `release_report_data_invalid.json`, `release_target_benchmark.json`,
  `release_run_invalid.log`.
- Corrected acceptance rendering of the same measured rows:
  `release_report_corrected_fail.md` and
  `release_report_data_corrected_fail.json` (FAIL, diagnostic only).
- Review/handoff: this file and `STAGE_REVIEW.md`.

No per-sample eval JSONL, generated completion text, credential, cache, weight,
Docker layer, or persistent TT cache is included.

## Cleanup and remaining work

The autoport server and engine were terminated cleanly. Port 8000 is closed;
no TTI Docker container or tmux session remains. The multigoal orchestrator is
not a serving process and remains outside cleanup scope.

Remaining gates before Stage 11 can pass:

1. Grant the active Hugging Face token's individual owner accepted/read access
   to `Idavidrein/gpqa`, validate without printing data, and rerun GPQA.
2. Supply a GPU/HF endpoint or same-command reference aggregates for Falcon3
   Base IFEval; investigate/fix TT correctness if that control proves a gap.
3. Rerun the no-Docker release with corrected acceptance, copy the replacement
   report, obtain an independent `stage-review` `clean-pass`, and then record
   the final local tt-inference-server and tt-metal commit SHAs. Never push.

## Final Stage 11 CI-nightly release — 2026-08-13

The previously listed accuracy gates were resolved and superseded by this
run. GPQA access became available to the active identity. Exact-snapshot BF16
CPU Hugging Face controls were then run for the same deterministic first-5%
documents, installed task versions, raw prompt mode, zero-shot settings, and
seed as TTI. The aggregate-only evidence is
`hf_paired_ci_references.json`; raw samples were not copied.

- IFEval v4 prompt-strict reference: HF 5/28 (17.857%). The final TT release
  run scored 6/28 (21.429%). An earlier paired TT probe scored 4/28; both are
  within the existing sample-count-aware CI policy's one-item quantization
  allowance. Other HF aggregates are retained in the compact reference JSON.
- GPQA generative v2 flexible-extract reference: HF 5/10 (50%). The final TT
  run scored 6/10 (60%), also a one-item difference.
- These values are configured only through `CI_NIGHTLY`
  `ModeReferenceScore`. Unrestricted `gpu_reference_score` remains unset and
  the model-card publisher values remain provenance, so no unrestricted gate
  was weakened or re-keyed.
- The measured CPU IFEval subset took 2h14m51s; a linear full-set estimate is
  about 45 hours. Stage 11 therefore claims nightly-equivalent subset
  readiness, not unrestricted full-set readiness, as allowed by the stage
  contract.

Server mode and context:

- Reservation container hostname: `b30c965c728b`; four P300 chips, mesh shape
  1x4 (`P300x2` product naming). A fresh open/close mesh probe passed before
  serving. No reset was required in this resumed run.
- External no-Docker server, port 8000. TTI used neither Docker nor its local
  server. `disable_trace_capture=true` was preserved.
- Server context remained 32,768 tokens with block size 32 and max sequences
  32. The 13-point release sweep reached ISL 16,384 plus OSL 128 and included
  naturally generated tokenization-length variation. No context cap, prompt
  alignment, request truncation, or invalid-request waiver was introduced.
- `server_autoport_provenance.log`, the runtime spec, and the server log prove
  the backend imported
  `models.autoports.tiiuae_falcon3_7b_base.tt.generator_vllm`. It did not use
  `models/tt_transformers`, `models/demos`, or another packaged model path.

TTI identity and command:

- Checkout: `/home/mvasiljevic/tt-inference-server`
- Tag/version: `0.19.0`; starting SHA `ca152fe223227f85f1a0d86cef7d372023b9de77`
- Final local TTI SHA: `e26e723bf0266cde85f674e381fbee10068ae0ec`
  (`Propagate model context to API evals`), following reference commit
  `bd15f1cdcf1bbb12187bd68b120e814b7e8a1e83`; never pushed.
- Docker image: not used
- Key environment: `CACHE_ROOT=/home/mvasiljevic/tti-release-cache/falcon3-base-stage11-contextfix-final`,
  `SERVICE_PORT=8000`; no token value was printed or copied.

```text
CACHE_ROOT=/home/mvasiljevic/tti-release-cache/falcon3-base-stage11-contextfix-final SERVICE_PORT=8000 python3 run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly --disable-trace-capture
```

The embedded spec itself contains `workflow=release`,
`limit_samples_mode=ci-nightly`, `docker_server=false`,
`local_server=false`, `service_port=8000`, and the autoport implementation and
context-contract paths. The loaded copy is
`runtime_model_spec_ci_nightly.json`.

Final result: PASS, exit code 0, zero blockers and zero waivers. Evals passed
2/2. Benchmarks passed 1/13 graded rows with 12 explicitly NA coverage rows;
all requests succeeded. The graded 128/128/concurrency-1 row measured 206.4 ms
mean TTFT, 56.9 decode tokens/s, and 62.1 user tokens/s against 250 ms / 50 /
50 targets. Spec tests were NA because TTI has no matching custom-model suite;
the prior health/completion smoke and successful OpenAI-compatible eval and
benchmark traffic provide the available API-path evidence.

Final copied artifacts:

- `report_tiiuae__Falcon3-7B-Base_2026-08-13T105032+0000.md`
- `release_report_data_ci_nightly_pass.json`
- `runtime_model_spec_ci_nightly.json`
- `release_run_ci_nightly.log`
- `release_target_benchmark_ci_nightly.json`
- `hf_paired_ci_references.json`
- `server_autoport_provenance.log`

No raw eval JSONL, generated text, secret, model weight, cache, Docker layer,
or persistent TT cache was copied. The final report was skimmed: both formerly
blocking eval rows pass without waivers, the graded benchmark has every metric,
and no failed or missing graded row remains.

The final context-preserving rerun used cache root
`/home/mvasiljevic/tti-release-cache/falcon3-base-stage11-contextfix-final`
at TTI SHA `e26e723bf026`. Its lm-eval commands passed
`max_length=32768`, and the copied run log confirms `Using max length 32768 -
1` for both IFEval and GPQA. It supersedes the earlier passing report that
exposed lm-eval's 2,048-token default. The stage artifact commit before this
rereview cycle was `53521e54ffb856b331a3015ffd6320ed9a1a8412`;
the context-preserving artifact commit is
`2e76369011bbd804609bbadf9c3ab2539de3ae60`. The final review-record commit is
logged after rereview. No stage-owned commit was pushed.
