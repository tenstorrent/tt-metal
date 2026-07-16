# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The full sampling log has one explicit all-vocabulary logprobs skip
  (`test_chat_logprobs_all_vocab`); the other 72 tests pass. This is disclosed
  capability coverage, not a serving-correctness failure, and bounded logprob,
  greedy, stochastic, mixed-batch, structured-output, and penalty cases pass.
- The final server log ends with nanobind reference-leak diagnostics. It has no
  traceback, TT fatal, OOM, unsafe-allocator warning, or error; UMD closes the
  devices, and the saved requests and benchmarks complete. The anomaly is
  classified and does not contradict the serving result.

## Hard-Check Gaps

- `.agents/scripts/check_context_contract.py` does not consume `--stage` and
  does not interpret `current_supported_context_scope` or
  `supported_context_by_surface`; it checks the single top-level
  `current_supported_context`. The repair therefore depends on the explicit
  scoped top-level value for the current optimized-vLLM surface. This is not
  checker gaming in the inspected tree: `vllm_supported_context` was already
  113280, the adapter consumes that vLLM-specific field first, all standalone
  and full-model fields remain 262144, and no full-model code consumes the
  changed top-level fallback. A future runner improvement could select the
  supported context by stage/surface directly.
- The saved JUnit files prove 49 adapter/full-model and 22 plugin tests passed,
  but their timestamps are the original Stage 10 run rather than a newly saved
  post-metadata-repair run. Since the live repair changes only contract/docs
  and the exact repaired runner check was independently rerun, this does not
  leave a stage-critical correctness gap; the work-log wording that these
  suites were rerun after repair is not independently timestamped by new
  artifacts.

## Anomaly Ledger

- Observed anomaly: The original exact Stage 10 runner check exited 2 on a
  context-cap finding.
  Evidence: `.exp_run/multigoal_logs/07-10-optimized-vllm.check-1.log` shows
  degeneracy passing, then `perf_summary.json:workload.max_model_len=113280`
  being compared against stale top-level `current_supported_context=262144`.
  A fresh execution of the unchanged runner script now exits 0.
  Affected path: Runner-side optimized-vLLM context verification and contract
  metadata; not the serving implementation or measured server configuration.
  Control or comparison: `vllm_supported_context=113280`, the final server
  command/log, and the maximum-length response all predate and agree with the
  repair; full-model surface fields remain 262144.
  Likely subsystem: Cross-surface context-contract metadata.
  Investigation performed: Inspected the checker source, pre/post diff,
  contract consumers, exact failed output, repaired runner output, capacity
  audit, failure/pass logs, and recomputed both boundary candidates.
  Resolution: fixed.

- Observed anomaly: Optimized-vLLM serving advertises 113280 rather than the HF
  262144-token context.
  Evidence: The 262144 server log reaches full-depth HMA KV allocation and
  fails with a TT DRAM OOM while allocating a 1,569,017,856-byte buffer. The
  direct 113279-input plus one-output request returns HTTP 200 with a real
  choice and `usage.total_tokens=113280`. Recomputed source accounting gives
  1,157,824 bytes/bank margin at 113280 and a 148,800-byte/bank shortfall at
  the next 64-token candidate, 113344.
  Affected path: Full-depth HMA vLLM serving on a 1x4 P150b mesh only.
  Control or comparison: Standalone/full-model, decoder, and datatype-selected
  surfaces retain 262144 in both the legacy fields and
  `supported_context_by_surface`.
  Likely subsystem: Device DRAM capacity and largest-contiguous allocation
  under the hybrid vLLM cache plus mandatory long-prefill live set.
  Investigation performed: Inspected source allocation formulas and adapter
  pool sizing, the failed and passing logs/responses, and independently
  evaluated `B(C)`, `P(C)`, `T(C)`, and `A(C)` for 113280 and 113344.
  Resolution: controlled.

- Observed anomaly: Several raw qualitative prompts produce repetitive
  continuation-corpus text rather than direct instruction answers.
  Evidence: The supervised-learning greedy output repeats the related question;
  thermodynamics and translation continue related-question datasets.
  Affected path: Raw-continuation qualitative generation.
  Control or comparison: The saved datatype-selected standalone controls begin
  with the same tokens and the same repetition/continuation patterns; the
  tokenizer has no native chat template. Fibonacci and story outputs remain
  coherent, and the degeneracy runner exits 0.
  Likely subsystem: Base-checkpoint continuation behavior, not vLLM async token
  feedback or cache state.
  Investigation performed: Read all six greedy and six sampled outputs and the
  matching standalone controls; inspected exact logit determinism across runs
  and batch positions.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: Stage 10 goal prompt; exact Stage 10 check script;
  `stage-review`, `vllm-integration`, `optimize`, and `tt-device-usage` skills.
- Artifact paths: `doc/context_contract.json`; optimized-vLLM README, work log,
  runtime audit, anomaly ledger, perf summary, before/after benchmark and raw
  result JSON, non-aligned/max-context/logit JSON, sampling log, all qualitative
  outputs/controls, JUnit XML, and final server log; vLLM integration capacity
  audit JSON/markdown and 262144-failure/113280-pass evidence; original failed
  runner log.
- Code paths: `tt/generator_vllm.py`, Stage 10 changes in `tt/generator.py` and
  `tt/model.py`, adapter contracts, and the nested plugin async controller,
  model input, input-batch/model-runner propagation, scheduler tests, plus the
  context checker source.
- Commands run: Read-only `git status/diff/show`, `rg`, `jq`, `sed`, `tail`,
  XML parsing, formula recomputation, artifact-existence/consumer searches, and
  the exact offline/non-hardware Stage 10 runner check (exit 0). No server,
  device, profiler, Tracy, watcher, or hardware command was run.

## Residual Risk

- The source ceiling has only about 1.16 MB/bank runtime margin, so allocator
  lifetime or persistent-buffer changes could lower the feasible serving
  boundary. Such a change must rerun the maximum-length request and capacity
  derivation; the present implementation and evidence are internally
  consistent.
- The main worktree contains unrelated dirty/untracked state, including a
  deleted requirements file and historical experiment artifacts. The live
  Stage 10 repair is isolatable by explicit paths: its modified files are the
  context contract, optimized-vLLM README, perf summary, and work log, plus
  this review report. The nested vLLM repository is clean at `44b7853`.
