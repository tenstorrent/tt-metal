# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- `work_log.md` still says “The adapter remains thin and unchanged,” while this repair added the `_page_table_refreshes` evidence counter to `tt/generator_vllm.py`. The adapter remains thin and the counter does not alter serving semantics, so this wording mismatch is editorial and non-blocking.

## Hard-Check Gaps

- None material. The stage intentionally has no live-serving profiler evidence, as required by the vLLM/optimization/device contracts.

## Anomaly Ledger

- Observed anomaly: the original first-measured optimized TTFT was 1304.75 ms versus 532.77 ms before.
  Evidence: `after/vllm_first_measured_benchmark.json` and `before/vllm_first_measured_benchmark.json`.
  Affected path: cold first request / sampler materialization.
  Control or comparison: the explicit second-run warmed pair is 201.59 ms after versus 214.35 ms before, with identical 128/128/1 settings; after TPOT is 35.67 ms versus 46.48 ms before.
  Likely subsystem: first-use sampler/kernel compilation.
  Investigation performed: inspected both raw/normalized first-measured and warmed artifact pairs and their commands/configs.
  Resolution: controlled. The required warmed comparison improves TTFT 6.0%, TPOT-derived decode 30.3%, and total elapsed time.

- Observed anomaly: P300C Ethernet reactivation failed twice after clean server shutdown.
  Evidence: `work_log.md` records the exact signature and bounded list/reset/list recovery; all four devices and a 1x4 mesh subsequently passed, and final cleanup found no live server process.
  Affected path: infrastructure between server runs, before model execution.
  Control or comparison: post-reset device listing and mesh open/close passed; subsequent serving evidence passed.
  Likely subsystem: recoverable firmware/device lifecycle behavior.
  Investigation performed: reviewed the preserved recovery account against `$tt-device-usage`.
  Resolution: controlled infrastructure event.

## Scope Inspected

- Goal/skills: stage 10 optimized-vLLM contract and `.agents/skills/{stage-review,vllm-integration,optimize,tt-enable-tracing,tt-device-usage,qualitative-check}/SKILL.md`.
- Artifacts: all files under `doc/optimized_vllm/{before,after}`, including warmed and first-measured primary pairs, CI serving pair, sampling log, all twelve qualitative completions, degeneracy report, six-prompt HF/full-model control comparison, overlap smoke, runtime audit, and repaired adapter trace-state probe; plus `readiness_vllm`, `doc/context_contract.json`, selected precision policy, and optimized-full-model performance/control evidence.
- Code: `tt/generator.py`, `tt/generator_vllm.py`, `tt/model.py`, `models/common/modules/sampling/sampling_1d.py`, touched unit/contract tests, and `tests/run_vllm_async_overlap.py`.
- Commands: read-only `git status/diff`, `find`, `rg`, `sed`, `nl`, and `jq`. No server, TT device, reset, profiler, or long test was run.

## Residual Risk

- The repaired hardware probe now exercises all four replays through `Gemma4ForCausalLM.decode_forward` with real host scheduler tables. It records adapter refresh counts 1/1/2/2, stable device addresses, exact mutated contents, zero generator identity recaptures, stale token/position rejection, aliased device feedback, nonblocking replay counters, and deferred-read event synchronization. This closes the prior required-work finding.
- The changed-table assertion proves the exact scheduler mapping reached the stable trace-bound tensors and serving output remained structurally valid. It does not retain a separate golden-token comparison for the deliberately remapped cache, but the mapping is synthetic and the contract-critical address/copy behavior is directly proven; this is not a blocking gap.
- The new two-chunk local-TopK path has broad real serving coverage from 72 passing runnable sampling tests, including greedy, top-k, seeds, mixed batches, penalties, and logprobs. A narrower unit-level chunk-index test would improve fault localization but is not required given the end-to-end evidence.
