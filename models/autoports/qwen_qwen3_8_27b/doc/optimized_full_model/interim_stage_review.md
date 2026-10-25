# Stage Review

Verdict: more-work-needed

Independent interim review of stage07, Qwen/Qwen3.8-27B, 2026-10-25. Live branch
`exp26-qwen38-bringup`, base `6bcbce2d6e26a194e58be1fe23eef340d1a08dff`.
This reviews work in progress, not a claim that the README or stage is complete.
No TT devices, servers, resets, implementation edits, or subagents were used.

## Required Work

- P1: Repair the validation wrapper's public signature, then finish the interrupted gates.
  Evidence: `full_validation.log` ends with `TypeError: Generator.generate() must
  explicitly declare an enable_trace keyword`. The `--buffered-output` branch in
  `tests/run_full_validation.py` replaces the method with
  `buffered_generate(*positional, **kwargs)` without `functools.wraps`, hiding the
  explicit parameter that exists at `tt/generator.py:547`.
  Why this matters: readiness teacher forcing did not execute; AR/shared-suite/
  long-case validation later in that same process also did not execute.
  Required next step: remove the redundant wrapper (the generator already defaults
  to buffered autoregression), or preserve the original signature. Retain valid
  completed prefill evidence if its source provenance is unchanged, then complete
  the missing gates. This is a harness defect, not evidence of model inaccuracy.

- P2: Complete tracking-enabled batch32 all-active/masked collector transitions.
  Evidence: existing `buffered_final_b32.json` records concurrency3 in every row;
  only `MeshTraceId(1)` (masked decode) feeds buffered sampler `MeshTraceId(3)`.
  `buffered_final_b1.json` cannot cover the batch32 async gather or unmasked trace.
  The planned all-batch benchmark disables allocation tracking.
  Why this matters: the added sampling trace must be safe against every live model
  trace in its actual replay order, including switching back to an earlier mode.
  Required next step: a reduced final-default B32 tracker-on process, with program
  cache included, should run all-active -> mixed -> all-active, greedy and sampled,
  checking exact streamed agreement and stable trace IDs/capture counts. The owner
  has added these cases to `probe_buffered_output.py`; execution and
  `buffered_final_b32_transition.json` were pending at review cutoff. A full-stack
  rerun is not required solely to close this focused gap.

- P2: Measure and resolve the default collector's EOS-latency tradeoff.
  Evidence: `tt/generator.py:647-672` submits a whole128-step tail window before
  inspecting EOS. The accepted `full_model/default_long_quality_cases.json`
  factual4096 control ends at output47; with its1024 cap this algorithm schedules
  129 outputs, including82 unnecessary full-model decode steps. The initial paired
  fixed128 B1 artifact improves throughput only35.1787 ->35.2523 tokens/s/user
  (about0.21%).
  Why this matters: the agentic workload waits for short complete replies. Exact
  returned tokens do not establish that making buffering the default improves
  request latency. The docs disclose the policy, but no paired EOS-latency control
  yet quantifies its cost.
  Required next step: measure the same EOS-terminated factual request on streamed
  and buffered paths, recording returned/scheduled tokens and total wall time;
  select or adjust the default from that evidence. The owner has agreed to add
  the paired control. Any collector policy change needs affected boundary/EOS
  checks, not a new broad accuracy sweep by default.

## Other Concerns

- Copied `qualitative_prompt_format.json` lists EOS IDs248044 and248046, while the
  new reduced trace artifact reports generation EOS248046. Final metadata should
  distinguish tokenizer/control IDs from the actual TT generation stop set.
- Final-default performance must use the new final measurements. The initial
  `buffered_full_b1.json` is correctly labeled a candidate; both saved source
  snapshots match its recorded hashes. No stale-source performance claim was found.
- No definite token-feedback, packed-prefill, cache-ownership, or sampler-adapter
  implementation bug was established by this inspection. The inherited decoder
  dtype/fidelity/layout implementation is unchanged in the inspected diff.

## Hard-Check Gaps

These are acknowledged pending final gates, not undisclosed completed claims:

- Final all64 AIME prefill/teacher-forcing acceptance, refreshed AR/shared-suite
  and long controls, direct output review, and degenerate-output result.
- Final same-regime4096/128 warmed TTFT/token-out rows at B1/32/8/16, separate
  teacher-forcing timing, agent-target context row, and before/after accounting.
- Reduced-only prefill/decode profiles and advice, runtime dtype/fidelity rows,
  terminal/collector attribution, and layer-stack gap accounting.
- Separate watcher/fallback audit; updated allocator/context/agentic ledger,
  including collector and its additional trace; final documentation and telemetry.
- Independent final clean-pass, then stage-owned local checkpoint commit and
  recorded SHA. Interim review does not authorize stage completion.

## Anomaly Ledger

- Observed anomaly: readiness aborts on `enable_trace` signature validation.
  Evidence: current `full_validation.log` and the optional wrapper above.
  Affected path: teacher-forcing and downstream validation orchestration.
  Control or comparison: original generator explicitly declares `enable_trace`.
  Likely subsystem: Python wrapper introspection.
  Investigation performed: source and exception compared; root cause sent to owner.
  Resolution: more-work-needed, narrow harness repair.

- Observed anomaly: buffered default keeps executing after an early EOS.
  Evidence: collector loop plus prior47-token factual control.
  Affected path: high-level EOS-enabled request latency; fixed slots remain valid.
  Control or comparison: existing streamed generation stops when all rows finish.
  Likely subsystem: output-window scheduling.
  Investigation performed: exact overshoot derived from loop bounds; paired control
  requested. No incorrect returned token sequence was inferred.
  Resolution: more-work-needed pending measured policy selection.

- Observed anomaly: early head candidates hit `bad optional access` and L1 limits.
  Evidence: head logs,19-row `head_candidates.csv`, individual candidate JSONs.
  Affected path: rejected terminal experiments.
  Control or comparison: adapted DRAM-sharded and split-interleaved trials run
  correctly but lose to the same BF16/HiFi4 baseline; larger K16/20/32 were exercised.
  Investigation performed: checked candidate code, policy, timings, and log narrative.
  Resolution: controlled; not an unsupported first-error rejection.

## Scope Inspected

- Goal/skills: parent's stage07 contract; installed0.1.21 stage-review,
  model-bringup startup, multichip, optimize, full-model, agentic-serving,
  tt-enable-tracing, tt-device-usage, and qualitative-check; repository
  `TraceCorrectness.md`.
- Artifacts: current README/work log/topology audit; head and sampling candidate
  JSON/CSV/logs; final reduced buffered/trace/prefill artifacts; initial all64
  paired candidate and saved source snapshots; copied HF provenance/outputs;
  current validation log; prior full-model README/rejection/long-quality controls.
- Code: generator/model terminal and sampler adapter; common sampler callback;
  head candidate helper; changed probes, validation/benchmark/profile scripts,
  memory inventory/audit and pending summary writer.
- Commands: read-only `git status/diff/rev-parse`, `rg`, `cat`, `sed`, `nl`,
  `tail`, `sha256sum`, and small standard-library JSON/hash/median analyses.
  One hash lookup initially used the wrong relative benchmark key; it was
  corrected to the artifact's `/workspace/...` key and both snapshots verified.
- Inspected generator SHA256:
  `db4710bf7dc2c367013b41e0d8b66a88dc0552efd72663c589ce4596631d7433`.
  Sampler adapter SHA256:
  `1c652512bb91cec3692d95211c49808fbe1553173155365bf3c73d4414421694`.

## Residual Risk

Live work continues after this snapshot. Reduced checks establish the tested
mechanisms, not final all64 quality/performance. Native262144 capacity evidence
is inherited and unchanged in scope; no full-model native-context HF accuracy is
claimed. A later review must inspect the repaired validation results, final
collector policy, measured defaults, and final documentation before clean-pass.
