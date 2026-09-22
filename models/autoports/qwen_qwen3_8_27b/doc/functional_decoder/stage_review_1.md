# Stage Review

Verdict: more-work-needed

Stage 1 functional decoder, `Qwen/Qwen3.8-27B`. Independent inspection of the
live workspace on branch `mvasiljevic/qwen38-full-bringup`, base
`a3a9fb4229a045ad9361b4e39ad854b491346ea9`. Evidence cutoff:
2026-09-11 17:52 UTC. Context runs and synthetic pytest were still pending;
this is an interim review, not a finding that those pending runs failed.

No additional concrete implementation defect was found in the inspected
decoder/cache/trace paths. The required work below consists of unfinished
contract gates and their documentation. A local stage checkpoint belongs
after clean-pass, so its current absence is not a review failure.

## Required Work

- P1: Complete the advertised-context evidence for both layer kinds.
  Evidence: `linear_context.json` currently records prefill 262143 with PCC
  0.9990367322 and traced decode at context 262144 with PCC 0.9997068644;
  its exact-262144 prefill run is still in progress. There is no
  `full_context.json` at this cutoff; full-attention completed evidence reaches
  prefill 257 / decode context 258. `doc/context_contract.json` still has
  `supported_context: null` and smoke-era tested maxima 32/33.
  Why this matters: the goal requires prefill and decode at context 262144,
  including a long non-aligned case, or a demonstrated hard physical limit.
  No capability reduction is claimed or justified.
  Required next step: finish exact-262144 linear prefill and full-attention
  near-limit/exact-limit prefill plus traced decode at context 262144. Keep
  passing results already produced; no short-shape rerun is requested solely
  because the long runs were pending during review. Populate the context
  contract from completed results, keeping per-kind coverage clear.

- P2: Execute the real-shape synthetic pytest suite and retain its result.
  Evidence: `tests/test_functional_decoder.py:15` defines both layer kinds
  using deterministic statistics-derived weights and the sequence
  `1,31,32,33,127,128,129,257,31`, batch 2, and continuation. The README and
  work log explicitly say execution is pending; no result is present at the
  cutoff. Host AST parsing succeeds, but it does not exercise this suite.
  Why this matters: the skill requires usable CI tests without checkpoint
  downloads; the authored synthetic path has not yet been shown to execute.
  Required next step: run the existing suite, investigate any failure, and
  retain its exact command and result. This finding does not request a larger
  test matrix or new synthetic acceptance threshold.

- P2: Finish the capability table and reproducible command record.
  Evidence: `README.md` still ends its evidence section with unfinished
  context/pytest gates. `work_log.md` ends at profiler-tool installation;
  it has no collection/report-generation commands for the existing two
  `*_profile.json` runs or commands/results for the context tests. The API
  contract is documented, but the requested claim/evidence/remaining-risk
  capability table is absent.
  Why this matters: the stage requires README/work-log commands, metrics,
  limitations, and a capability table. Existing performance artifacts are
  real and internally consistent; this is a documentation gap, not a demand
  for fresh profiling.
  Required next step: record the actual profiling, context and pytest commands
  and outcomes, and add the compact capability table once the pending gates
  resolve. Preserve the existing four profiler windows.

## Other Concerns

- The implementation's full-attention prefill batch requires equal logical
  lengths and prefix positions; this is explicitly documented at
  `tt/functional_decoder.py:290`. It is not silently advertised as ragged
  batching. Downstream orchestration must respect that boundary.
- The replay checks restore the prefix state between variants. An
  uninterrupted multi-step decode check would strengthen later generator
  integration evidence, but inspection shows persistent state copies and
  stable captured inputs, so its absence is not an extra Stage 1 blocker.
- Logs and CSVs are ignored by repository defaults. The eventual checkpoint
  must retain the required compact evidence deliberately; no checkpoint was
  yet attempted, so this is not currently a missing-commit finding.

## Hard-Check Gaps

- The reviewer ran no device tests, TTNN imports, hardware commands, resets,
  servers, or profiler collection. Runtime conclusions below are rederived
  from saved runner logs/JSON, watcher logs, profiler files, and source.
- No new runtime counters, long soak, per-token PCC threshold, or hardware
  matrix is required by this review. Aggregate PCC meets the stated bar in
  every completed reviewed result.

## Anomaly Ledger

- Observed anomaly: first batch-32 linear scan failed with
  `num_heads 1536 exceeds compute cores 110`.
  Evidence: `linear_b32_watcher.log`, native
  `chunk_gdn_phased_program_factory.cpp`, and the current decoder lines 254–270.
  Affected path: multi-request linear prefill.
  Control or comparison: `linear_b32_watcher_retry.json` passes batch 32,
  continuation, traced decode, and changed-input replay; both watcher logs
  inspected for the final runs contain no suspicious diagnostics.
  Likely subsystem: native scan's one-head-per-core allocation constraint.
  Investigation performed: checked that the repair splits only the independent
  batch axis, then concatenates outputs and recurrent state in request order.
  Resolution: fixed; no native source changes are needed for this workaround.

- Observed anomaly: initial mesh open failed on a frozen ERISC heartbeat.
  Evidence: `AUTOTRIAGE.md`, `mesh_smoke.log`, `reset_1.log`,
  `device_list_after_reset_1.log`, `mesh_smoke_after_reset_1.log`.
  Affected path: infrastructure initialization before model execution.
  Control or comparison: same firmware 19.8.0 opens/closes after one reset,
  followed by successful model runs.
  Likely subsystem: stale Ethernet firmware execution state; its originating
  workload is not established.
  Investigation performed: read the independent triage report and recovery
  evidence; no hardware action by this reviewer.
  Resolution: controlled. The report also correctly retains the discrepancy
  between firmware 19.8.0 and the checkout's documented 19.8.1 as residual risk.

- Observed anomaly: compiler unused-variable warnings and device topology
  warnings are present in successful runs.
  Evidence: final batch-32 console logs contain `dfb_x_id` unused warnings,
  unknown-motherboard tray fallback, and subset-of-MMIO-device warnings.
  Affected path: existing JIT compilation / topology initialization.
  Control or comparison: both layer kinds complete, watcher attaches/polls/
  detaches, and PCC gates pass. Profiler reports identify one measured device.
  Likely subsystem: compiler diagnostics and host topology descriptions.
  Investigation performed: separated console diagnostics from actual watcher
  messages and checked final watcher files for fatal, corrupt, invalid,
  overflow, out-of-bounds, sanitize, and error signatures.
  Resolution: controlled for this single-device stage; no source fix justified.

## Scope Inspected

- Goal/skill paths: supplied Stage 1 contract; installed
  `tt-model-bringup/0.1.4/skills/{stage-review,functional-decoder,tt-device-usage}/SKILL.md`;
  installed model-bringup startup; repository
  `.agents/skills/{stage-review,functional-decoder,tt-enable-tracing}/SKILL.md`;
  user-supplied repository AGENTS instructions.
- Code paths: the complete `tt/functional_decoder.py`, `tests/reference.py`,
  `tests/run_decoder.py`, and `tests/test_functional_decoder.py`; imported
  rotary helper in `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`;
  native delta-rule adapter and scan constraint; installed Transformers
  `models/qwen3_5/modeling_qwen3_5.py` attention, norms, convolution,
  delta recurrence, and decoder sections.
- Artifact paths: README/work log/context contract; pinned HF config and
  tensor statistics; both smokes; boundary and continuation JSON/logs;
  both final batch-32 watcher runs; `linear_context.json/log`; profile JSON/logs;
  all four human tables and filtered CSVs; copied and original Tracy ops CSVs;
  recovery/triage report and referenced logs.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `head`, `tail`, `git status`,
  `git diff --name-only`, and `git log`; standard-library `python3` JSON/CSV,
  SHA256 and AST analysis. One initial artifact-analysis invocation of
  `python` failed because that host executable was absent; it was rerun with
  `python3`. Only this review report was written.
- Source inspected: decoder SHA256
  `d42c25abaf2dafa6f96b17f9a28a47f88dd126fbc5ee2c1c71169413104b850a`;
  runner SHA256
  `f8942f0b67dd89ff484299155ecdd4616d34ca3eae1de25151be02ac98afcff2`.
- Re-derived findings: the pinned config has 48 linear and 16 full-attention
  layers at the real shapes, and its SHA256 matches the statistics manifest.
  Residual order, SwiGLU, zero-centered norms, GQA, gated attention and partial
  RoPE agree with HF source. Linear history retains the three needed prior
  projected tokens; native chunk padding zeros beta/g to preserve state.
  Full attention routes page-aligned chunks through paged fill/causal SDPA and
  unaligned-prefix tokens through positioned paged updates, preserving prefixes.
  Forward helpers are TTNN device operations; supplied constants avoid native
  host-built delta constants. Capture/replay binds the refreshed tensors and
  restores state after warmup/capture. Completed real-weight results pass
  PCC >= 0.995, including independent page ownership and changed replay inputs.

The reported profiler sums exactly match filtered CSV `Device Time` values
in microseconds, and both kinds' copied ops CSVs byte-match their originals:

| Kind | Warmed prefill 128 | Traced decode context 129 |
| --- | ---: | ---: |
| Linear attention | 4623.045 µs | 3009.414 µs |
| Full attention | 3407.822 µs | 2378.159 µs |

These are kernel-time sums, not end-to-end generation latency. Each report's
signposts delimit the corresponding warmed path in the runner.

## Residual Risk

- The live workspace can advance after this cutoff; this verdict applies to
  the files and completed results identified above. Pending gate completion
  requires independent follow-up inspection before clean-pass.
- Representative layers 0 and 3 use real weights; other layer weights and
  accumulated full-model/generation quality belong to subsequent stages.
- Full advertised context is not yet fully validated for both kinds. The
  later full model's combined weight/state capacity is a separate question.
- Reset recovery demonstrates reversibility, not the unknown ERISC failure's
  exact origin or absence of future infrastructure recurrence.
