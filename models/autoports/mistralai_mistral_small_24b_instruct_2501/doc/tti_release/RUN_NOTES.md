# Mistral Small 24B Instruct 2501 — authoritative v10 TTI release

## Final classification

- `release-workflow-fail`
- `readiness-fail`

The native release workflow reached a terminal report and failed two mandatory
quality gates. It also exposed two independent serving failures. AutoFix repaired
both serving defects and the failed benchmark/spec components were rerun through
the native TTI workflows on the corrected exact server.

TTI has no generalized reports-only CLI, but its native `ReportSchema`,
acceptance, and `ReportGenerator` primitives support lossless block aggregation.
`final_reports_merge_v10.md` / `.json` combines the two retained valid eval
blocks with the corrected 13 benchmark blocks and corrected two spec blocks. It
is provenance-stamped with all three source paths and SHA256s. Its honest final
acceptance remains FAIL with exactly the two quality blockers: Benchmarks PASS
(1/13 passed, 12 NA), Spec Tests PASS (1/1), Evals FAIL (0/2). It is a
reports-only native schema merge, not a claim that a second monolithic release
command reran the valid evaluations.

No `known_issues` mask, score override, release waiver, implementation
substitution, reduced context, or threshold change was used.

## Exact scope

- Implementation:
  `models/autoports/mistralai_mistral_small_24b_instruct_2501`
- Selector: `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport`
- Model: `mistralai/Mistral-Small-24B-Instruct-2501`
- Device: P300X2, four local Blackhole p300c chips, `MeshShape(1, 4)`
- Server: external OpenAI-compatible TT vLLM API; no Docker and no stock model
  implementation
- Context/config: block size 32, max model length 32768, max sequences 32,
  max batched tokens 32768, seed 9472, device sampling `all`, trace region
  200,000,000 bytes, `FABRIC_1D`
- TTI workflow checkout:
  `b118a82c59d6a3b682253ff170ac4fe2990a300f`
- tt-metal base: `5bab286dc7fb063f4f435c840af64359fe4bf533`
- final nested vLLM: `aab6d846caf95c5e9cf8038f3338650a9132c383`

`runtime_model_spec_v10_native.json` is the TTI-written immutable spec for the
terminal native release report; it records nested vLLM commit
`971ee6cfcdd97a36a98e26f96ff7dda08441d219`, after the first repair. The current
`release_spec.json` records final nested commit `aab6d846...`, after the second
repair, and contains no masks.

## Pre-release server and native smoke ordering

The exact v10 server reported API startup on port 8000 at
`2026-08-14 19:04:19 UTC`. Its access log then recorded HTTP 200 for `/health`
and `/v1/chat/completions`; the small client response artifact
`/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/logs/chat_v10.json`
is timestamped `2026-08-14 19:04:58 UTC` and contains the expected assistant
response `V10 SMOKE OK`. The response is referenced here but not copied, and no
request headers, tokens, or secrets are included in this handoff.

The authoritative completed native smoke then ran from
`2026-08-14 19:06:40.247 UTC` through `2026-08-14 19:06:46.162 UTC`. Its source
run log is
`/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/smoke_cache_pass/workflow_logs/run_logs/run_2026-08-14_19-06-40_id_autoport_mistral-small-24b-instruct-2501_p300x2_smoke_v10_benchmarks_Aw_ZfxMz.log`.
The native workflow produced one block over one sweep point, Acceptance PASS,
and terminal workflow/command rc=0. The request asked vLLM for random input
length 8 and output length 8; tokenization produced an actual 7-token input,
and the request completed with 8 output tokens (1/1 completed, zero failed).
This smoke therefore precedes the native release command at
`2026-08-14 19:07:16.899 UTC` and release workflow start at
`2026-08-14 19:07:17.986 UTC`.

`runtime_model_spec_v10_smoke.json` is the TTI-written spec from
`/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/smoke_cache_pass/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_19-06-40_id_autoport_mistral-small-24b-instruct-2501_p300x2_smoke_v10_EXIwNfrR.json`.
It records the exact implementation code path, 32768-token context, tt-metal
commit `5bab286dc7fb063f4f435c840af64359fe4bf533`, and then-current nested vLLM
commit `0e5e6495ac0a39e7c16a925140547cfb4a2e3030`. Both its embedded CLI arguments
and runtime config record `docker_server=false`, `local_server=false`, service
port `8000`, `disable_trace_capture=true`, and
`limit_samples_mode=smoke-test`.

Committed smoke artifact SHA256s are:

- `runtime_model_spec_v10_smoke.json`:
  `296cbdbcd3fde42bfb4dbabc78b56c0c3e192b80c856fa4b3982623c5b7b3ed8`
- `smoke_v10.log`:
  `241a3f372ae94b73403ba2861f72c322eec8eaa5b1f90203310205ab50b91401`
- `smoke_v10_benchmark.json`:
  `cc90ca3445e538a63eaf582a743195cb4bb79d4912c36336e8c4e8cb38733e01`

## Native terminal release outcome

`native_release_report.md` and `.json` are the complete terminal report emitted
by the native `release` workflow. Its acceptance status is FAIL:

- IFEval completed 109/109 samples operationally and scored
  72.55740423987976%, below the 78.755% requirement.
- GPQA completed 90/90 samples operationally and scored
  38.88888888888889% flexible extract, below the 43.035% requirement.
- The configured 128-input/128-output, concurrency-1 benchmark point completed
  8/8 and passed all functional performance targets.
- The next 32-concurrency benchmark point killed EngineCore because stale
  device-state slots survived an idle cleanup, so later benchmark/spec rows in
  that terminal report are not valid conformance evidence.

The evaluations used `--limit-samples-mode ci-nightly` (0.2 dataset limit), the
HF-declared instruct chat template, deterministic seed 42, and the full 32768
serving context. These are CI-subset scores, not unrestricted full-dataset
claims. Aggregate evidence is in `eval_v10_aggregate_metadata.json`; raw sample
JSONL and generated completion corpora are excluded.

## AutoFix results

The first failure occurred during GPQA at an exact KV page boundary: host token
2,429 versus device position 2,432. Nested commit `971ee6c...` reserves the full
three-token TT async pipeline lookahead. A 2,399-input/96-output hardware control
crossed the failing boundary, completed 1/1, and the resumed GPQA completed
90/90.

The second failure followed the benchmark's single-request idle transition:
32 historical state-slot owners remained before 31 new prefills. Nested commit
`aab6d846...` reclaims only older off-batch owners, only under impossible slot
pressure, while preserving immediately recent potentially live state. The exact
lifecycle control completed 8/8 at concurrency 1 immediately followed by
256/256 at concurrency 32, with zero failures.

Focused host regressions passed 10 tests. Full reasoning and regression contracts
are in `AUTODEBUG_V10.md` and `AUTOFIX.md`; compact hardware metrics are in
`hardware_controls_v10.json`.

## Corrected native components

The corrected benchmark used the native 13-point TTI sweep from 128 through
16,384 input tokens. Only the spec's 128/128 concurrency-1 reference point is
graded; the other 12 rows are explicitly NA/ungraded information rows. The
graded point completed 8/8 with zero failures, mean TTFT 535.42 ms, mean TPOT
18.94 ms, output throughput 43.53 tokens/s, and passed the configured functional
targets (TTFT 1400 ms, per-user throughput 50 tokens/s, decode throughput 32
tokens/s, tolerance 5%). See `benchmark_report_v10_slotfix.md` / `.json`.

The corrected native spec workflow reran logger fork safety plus all 22
parametrized vLLM chat-completion conformance cases. All 22 passed; the task
emitted two blocks, zero failures, Acceptance PASS, and rc=0 in 1084.8 seconds.
See `spec_report_v10_slotfix.md` / `.json` for the authoritative component
result.

## Cleanup and artifact policy

The production server was stopped after all component work; no API server,
EngineCore, TTI client, benchmark process, or port-8000 listener remained.
Bounded local `tt-smi` list/reset/list succeeded for UMD IDs 0, 1, 2, and 3;
all four Blackhole p300c chips remained visible and resettable. A fresh
`MeshShape(1, 4)` then opened and closed successfully.

Committed v10 artifacts are deliberately small and inspectable:

- reports-only final acceptance: `final_reports_merge_v10.md` / `.json`
- terminal native release: `native_release_report.md` / `.json`
- native provenance: `runtime_model_spec_v10_native.json`, `release_spec.json`
- ordered native smoke: `runtime_model_spec_v10_smoke.json`, `smoke_v10.log`,
  `smoke_v10_benchmark.json`
- corrected component reports and TTI-written specs:
  `benchmark_report_v10_slotfix.md` / `.json`,
  `runtime_model_spec_v10_benchmark_slotfix.json`,
  `spec_report_v10_slotfix.md` / `.json`,
  `runtime_model_spec_v10_spec_slotfix.json`
- aggregate quality/control evidence: `eval_v10_aggregate_metadata.json`,
  `hardware_controls_v10.json`, `remediation_tests_v10.log`
- ordered hardware controls: `boundary_control_v10.log`,
  `slot_control_order1_c1_n8_v10.log`,
  `slot_control_order2_c32_n256_v10.log`
- diagnosis: `AUTODEBUG_V10.md`, `AUTOFIX.md`

Excluded: weights, tensor dumps, caches, raw per-request benchmark JSON other
than the single authoritative 1-request smoke above, raw eval sample JSONL,
generated completion corpora, profiler bulk, `.env`, and secrets.
Older v9-named files in this directory remain historical evidence and are not
the authoritative v10 result.
