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

Excluded: weights, tensor dumps, caches, raw per-request benchmark JSON, raw eval
sample JSONL, generated completion corpora, profiler bulk, `.env`, and secrets.
Older v9-named files in this directory remain historical evidence and are not
the authoritative v10 result.
