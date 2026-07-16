# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The live sampling suite has one expected skip:
  `test_chat_logprobs_all_vocab`. The test explicitly permits this when the
  server's configured `max_logprobs` cap rejects an all-vocabulary response.
  The other 72 sampling/logprob cases passed, and this compatibility case is
  outside the measured greedy device-sampling policy.
- Firmware 19.9 is newer than the latest fully tested 19.5 bundle. The final
  run itself completed, closed all devices, and left no serving process, so
  this is an environment qualification risk rather than contradictory stage
  evidence.
- Nanobind reports reference leaks during interpreter teardown. The diagnostic
  follows successful requests and benchmarks; the final log then records
  `Closing devices in cluster completed` and `Application shutdown complete`.
  It did not select a fallback or leave the device/runtime live.

## Hard-Check Gaps

- Device-only decode latency and a roofline estimate are intentionally null in
  `perf_summary.json`. Stage 10 forbids Tracy, `tt-perf-report`, live-server
  device profiling, adapter profiling, and `ReadDeviceProfiler`; serving JSON,
  trace contracts, and no-fallback evidence are the required substitutes.
- The available full-model split-sampling control is comparable rather than an
  exact workload match: standalone uses prompt 149 / generation 100 / batch 1,
  while the vLLM primary uses 127 actual input / 128 output / batch 1. The
  distinction is disclosed. Standalone reports 24.787 end-to-end and 34.256
  steady t/s/u; vLLM reports 26.645 TPOT-derived and 34.095 median-ITL t/s/u,
  which satisfies the required "about as fast" comparison without claiming
  identity.

## Anomaly Ledger

- Observed anomaly: unsafe allocator warning at the top-k 32 to greedy top-1
  transition in an earlier run.
  Evidence: `anomaly_ledger.md`, the sampler capture diff in
  `tt/generator.py`, `evidence/adapter_contract.xml`, and the superseding
  `after/sampling_tests.log`.
  Affected path: transition from host-compatible sampling to the traced greedy
  TP4 sampler while the model trace is live.
  Control or comparison: the exact superseding suite sequence
  `test_topk[32]` then `test_top1_is_greedy` passed; the final server log has
  zero `Allocating device buffers is unsafe` matches.
  Likely subsystem: first-use sampler program/resource allocation after model
  trace registration.
  Investigation performed: source-order audit found a redundant eager sampler
  dispatch immediately before capture. Exact sampler prewarm is now before
  model capture, sampler capture contains only capture dispatch, and the first
  replays are model then sampler, both nonblocking.
  Resolution: fixed.

- Observed anomaly: unsafe allocator warning at the first 64-token KV block
  growth boundary after the sampler repair.
  Evidence: `anomaly_ledger.md`, persistent-copy initialization in
  `tt/model.py`, page-table change handling in the nested plugin, focused XML
  contracts, long qualitative outputs, and `evidence/final_server.log`.
  Affected path: scheduler-owned page-table refresh into persistent trace input
  tensors while model and sampler traces remain live.
  Control or comparison: the final full runner repeated long batch-1
  generations across block boundaries, completed the primary and CI workloads,
  and emitted zero unsafe allocator warnings or traceback/error matches.
  Likely subsystem: first-use `ttnn.copy(source, target)` program allocation.
  Investigation performed: two source audits isolated the warning from async
  readback and placed it at the first `new_block_ids` boundary. Initialization
  now prewarms every distinct persistent source/target copy pair before any
  trace registration; scheduler `new_block_ids` separately disables overlap
  and stale-input reuse for the boundary step.
  Resolution: fixed.

- Observed anomaly: some greedy qualitative outputs continue question corpora
  or repeat a related question instead of following the instruction directly.
  Evidence: `qualitative/vllm_qualitative_outputs.json`,
  `qualitative/standalone_control_outputs.json`, prompt-format metadata,
  `qualitative/verdict.md`, and the zero-finding degeneracy JSON.
  Affected path: visible generated text.
  Control or comparison: this tokenizer has no native chat template and the
  checkpoint is evaluated as raw continuation. Every vLLM greedy output starts
  with the selected standalone TT control for the same prompt; the repeated
  supervised-learning continuation also matches HF.
  Likely subsystem: inherited base-checkpoint continuation behavior, not stale
  serving state or sampler feedback.
  Investigation performed: six greedy and six sampled outputs were reviewed,
  compared with prompt-correct standalone controls, and scanned by the scoped
  degeneracy checker.
  Resolution: controlled.

- Observed anomaly: two device-0 Ethernet resume failures before model code.
  Evidence: `work_log.md` and `anomaly_ledger.md`.
  Affected path: device open before serving initialization.
  Control or comparison: bounded list/reset/list recovery and corrected 1x4
  mesh smokes passed; the complete final runner then passed and shut down.
  Likely subsystem: recoverable device/firmware infrastructure state.
  Investigation performed: only failed-run processes were stopped; reset and
  mesh-open recovery followed the device-usage contract before resuming.
  Resolution: controlled.

- Observed anomaly: nanobind reference-leak diagnostics at interpreter exit.
  Evidence: tail of `evidence/final_server.log`.
  Affected path: process teardown after serving completion.
  Control or comparison: device-cluster close and application shutdown are
  recorded after the diagnostic; all runner gates and HTTP requests completed.
  Likely subsystem: binding teardown/reference accounting.
  Investigation performed: ordering and cleanup state were checked against the
  final log and process/device-holder audit recorded in `work_log.md`.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `.agents/prompts/model_bringup_multigoal/10-optimized-vllm.txt`;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/vllm-integration/SKILL.md`;
  `.agents/skills/tt-enable-tracing/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`; and
  `.agents/skills/qualitative-check/SKILL.md`.
- Artifact paths: all files under `doc/optimized_vllm/`, the active
  `doc/context_contract.json`, and
  `doc/datatype_sweep/post_selection_token_out.json`.
- Code paths: main-repo changes in `tt/generator.py`, `tt/generator_vllm.py`,
  `tt/model.py`, and `tests/test_vllm_adapter_contract.py`; nested-vLLM changes
  in `async_decode.py`, `input_batch.py`, `model_input.py`, `model_runner.py`,
  and `tests/test_lane_model_runner.py`; plugin registration and shared runner
  behavior were also traced to their source.
- Snapshot: main branch `odjuricic/agentic-research/graph-rewrite-skill` at
  `a010c05f387d47c9e87c1cb076a949d5bc97a95d`; nested vLLM branch `dev` at
  `91c467d6fc18c4386eda14360baf0bee0e0f684c`. Stage-owned changes are still
  uncommitted, as expected before the post-review checkpoint commits.
- Commands run: scoped `git status`, `git diff`, `git diff --check`, and nested
  equivalents; `jq` and small Python scripts to recompute percentages, compare
  before/after command/config parity, inspect raw metrics, validate qualitative
  prefixes, and count XML cases; `rg` scans for allocator/error/trace lifecycle,
  async configuration, sampling, profiler, and cleanup signatures; `cmp` checks
  for copied final server and qualitative artifacts.

## Residual Risk

- The page-table boundary proof is end-to-end serving evidence plus source and
  focused contract tests, not a forbidden live-serving device-profiler trace.
  This is the evidence regime required by Stage 10.
- The async fast path intentionally trusts an explicit plugin proof that token,
  position, layout, cache identity, sampling key, and page tables are stable.
  The new `page_tables_changed` default is conservative, both scheduler states
  are tested, and long final-run output matches standalone prefixes, which
  materially bounds stale-state risk.
- Local checkpoint commits and replacement of `pending_stage_review` status are
  post-verdict orchestration steps; they were not present in this review
  snapshot and do not alter the technical verdict.
