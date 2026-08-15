# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The shared `minimal_default_writer.cpp` fix has strong source-level diagnosis and passes the exact fallback-raising greedy-to-sampled watcher repro, but no dedicated generic CCL unit test covering endpoint directions and every 1--4-page packet geometry was added. This is desirable follow-up coverage, not required stage work: the observed one-page failure is fixed, normal model collectives exercise larger packet geometries, and all nearby full-model, sampler, mixed-prompt, batch-32, accuracy, qualitative, and performance paths pass.
- The compact profile intentionally covers one sliding layer, one full-attention layer, and the terminal path rather than all 30 decoder layers. The report correctly keeps its 5.585 ms merged device-op sum separate from the 32.329 ms modeled all-layer decoder-stack lower bound.

## Hard-Check Gaps

- AIME24 prefill and teacher-forcing logs use `throw_exception_on_fallback=false`; separate fallback-raising reduced, mixed-prompt, batch-32, watcher, and exact no-host full-stack probes cover the runtime fallback contract.
- `tt-perf-report` leaves several operations unclassified, including SDPA decode, async collectives, TopK, and ArgMax. Their individual rows and times remain present in the retained CSV, so terminal attribution is still auditable, while the category/roofline summary is less complete.
- The 1.1 GiB raw Tracy capture was removed after compact extraction. The CSV, text report, summary, PNG, profile log, command, signpost scope, and provenance remain durable.
- Static and hardware logs report nanobind shutdown leaks. No runtime failure, increasing per-token allocation, or incorrect result is associated with them in this stage.

## Anomaly Ledger

- Observed anomaly: the historical and current host-visible runs differ by 10.31% although no retained model/generator performance candidate remains.
  Evidence: `performance.json`, `README.md`, `work_log.md`, both candidate JSON files, prior `doc/full_model/autoregressive_perf_128_fixed/autoregressive_meta.json`, and current `autoregressive_perf_128/autoregressive_meta.json`.
  Affected path: full 30-layer host-visible autoregressive decode.
  Control or comparison: inherited 23.7629 t/s/u versus current reproduced 26.2128 t/s/u.
  Likely subsystem: session cache/JIT/runtime-state variance rather than a retained graph optimization.
  Investigation performed: compared prompt/regime metadata, candidate decisions, current implementation diff, and revised performance wording.
  Resolution: controlled. README, work log, and performance JSON now explicitly call the difference observational and use the current reproduced values as the authoritative stage result.

- Observed anomaly: TT and HF free-running AIME24 sequences diverge after early greedy choices.
  Evidence: `autoregressive_aime24/autoregressive_meta.json`, both completion files, and `autoregressive_aime24_degeneracy.json`.
  Affected path: autonomous traced greedy token feedback.
  Control or comparison: same checkpoint revision and rendered chat prompt; both completions are coherent, English, on-topic equation setups.
  Likely subsystem: ordinary logit-order sensitivity after alternate valid greedy choices.
  Investigation performed: inspected token sequences and text directly; separately verified shifted-left ranks through the readiness teacher-forcing source and log.
  Resolution: controlled. Free-running evidence passes qualitative/feedback checks, while the contracted rank gate independently achieves top-5 and top-100 of 100/100.

- Observed anomaly: shared-suite HF/TT token agreement ranges from 1/63 on French to 46/64 on supervised-learning explanation.
  Evidence: `shared_qualitative_degeneracy.json` and all six `shared_qualitative/*/{hf_completion.txt,tt_completion.txt,autoregressive_meta.json}` sets.
  Affected path: free-running chat generation.
  Control or comparison: correctly rendered same-revision HF controls for each prompt.
  Likely subsystem: expected free-running branch divergence, not mechanical corruption.
  Investigation performed: directly inspected every TT and HF completion plus prompt metadata and degeneracy metrics.
  Resolution: controlled. All TT outputs are coherent and on-topic; French is correctly translated, the haiku obeys form, and no wrong-language drift, prompt echo, control-token leakage, adjacent duplication, loop, or gibberish appears.

- Observed anomaly: the first durable no-host remediation attempt exhausted a 128-position allocation and its live triage helper was incompatible with current UMD reads.
  Evidence: remediation section of `work_log.md` and `triage/nohost-bench-summary.txt`.
  Affected path: benchmark harness capacity, before the measured final run.
  Control or comparison: corrected harness allocates `max_seq_len=512` for prompt 128 plus capture, five warmups, and 128 measurements.
  Likely subsystem: test-harness capacity configuration, not model cache correctness.
  Investigation performed: source correction, reset/relist recovery, and exact fallback-raising rerun.
  Resolution: fixed. `no_host_boundary_token_out.log` passes at 28.015097 t/s/u with final position 262 and zero readbacks/refreshes/timed-loop synchronizations.

- Observed anomaly: watcher exposed endpoint-connection and one-chunk scatter-header assertions in the shared direct-fabric writer.
  Evidence: `watcher_full_path.log`, `watcher_full_path_fixed.log`, `AUTOTRIAGE.md`, `AUTOFIX.md`, CCL source diff, and `watcher_full_path_fixed_v2.log`.
  Affected path: model and split-sampler async all-gather collectives.
  Control or comparison: persistent-off reproduced the first failure; focused model-layer probes refuted persistent-resource reuse; the exact original greedy-plus-sampled repro passes after both guards.
  Likely subsystem: generic minimal-default async all-gather writer contracts.
  Investigation performed: source-line matching, call-site bounds analysis, isolated hypothesis tests, and exact watcher rerun.
  Resolution: fixed for the observed contracts, with generic unit-test breadth retained as an other concern.

- Observed anomaly: profiler identifies BF16 interleaved LM head and ArgMax as the largest terminal rows.
  Evidence: `profile/full_model_reduced_decode.csv`, `candidates/dram_sharded_lm_head.json`, and sampler comparison evidence cited in the work log.
  Affected path: terminal logits and greedy sampling.
  Control or comparison: adapted 16-chunk DRAM-sharded LM head regressed to 36.0360 ms/token from 35.7060; shape-faithful force-argmax measured 2.3434 ms versus split top-k=1 at 10.7359 ms with identical tokens.
  Likely subsystem: terminal LM-head/sampler cost.
  Investigation performed: checked runtime rows, dtypes/fidelity, legal adapted candidate, and full-path accounting.
  Resolution: controlled. The selected complete path is 35.6950 ms/token, only 10.41% above the decoder-stack lower bound, so terminal/orchestration cost is closed within the stated 10--15% band.

## Scope Inspected

- Goal/skill paths: supplied optimized-full-model contract; `.agents/skills/stage-review/SKILL.md`, `multichip/SKILL.md`, `optimize/SKILL.md`, `tt-device-usage/SKILL.md`, and `qualitative-check/SKILL.md`.
- Artifact paths: README/work log; performance, fallback, accuracy-contract, qualitative, context-provenance, and no-host JSON; authoritative no-host raw log; AIME24 prefill/teacher-forcing/free-running logs and outputs; all six shared qualitative prompt files, metadata, HF/TT outputs, logs, suite summary, and degeneracy report; mixed-prompt, batch-32, watcher, static, AutoTriage/AutoFix, candidate, profiler, and prior full-model/optimized-multichip evidence.
- Code paths: `models/common/readiness_check/run_teacher_forcing.py`, readiness `teacher_forcing.py`, model `tt/generator.py`, `tt/model.py`, decoder implementations, `tests/test_full_model_contract.py`, and `ttnn/.../minimal_default_writer.cpp`.
- Commands run: read-only `find`, `sed`, `rg`, `git status`, `git diff`, `sha256sum`, and small Python JSON/CSV/metadata analyses. No TT device, server, reset, vLLM, or hardware command was run during review.

## Residual Risk

- The durable 262,111-token context provenance hash matches the currently available `/tmp` source log, and the compact context contract is retained; the verbose original log remains machine-local rather than repository-resident.
- The reduced profile samples representative sliding/full layers rather than every layer role, so preservation of the complete inherited decoder policy also relies on prior optimized-multichip artifacts, current source inspection, and full-stack accuracy/performance runs.
- Stochastic top-k/top-p trace transition is watcher-tested and the sampler contract is statically validated, but qualitative generation is greedy; stochastic output quality remains future serving-stage coverage rather than a blocker for this pre-vLLM optimized-full-model stage.
