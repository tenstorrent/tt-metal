# Stage Review

Verdict: **more-work-needed**

Independent round 1 review of stage 7, optimized-full-model, for
`Qwen/Qwen3.8-Flash-Next`, on the live `main` worktree based on
`de432e04c21c3ea53ffe13d9c6e78a96f7acb704`.

The final implementation is a materially faster, numerically validated
checkpoint, but it does not satisfy the required FUNCTIONAL performance floor.
The latest user instruction permits carrying nonblocking work at the cap;
it does not waive this mandatory gate. This verdict does **not** request another
unchanged gate run, new optimization experiment, or additional review round in
the capped turn. Preserve the selected version and the failed-gate status in
the authorized local checkpoint.

## Required Work

- **P1: The mandatory batch-1 FUNCTIONAL floor remains failed.**
  Evidence: `final_resident_v1.json::performance.resident_token_out` records
  13.089249865151942 seconds for 127 model-plus-sampler decode steps. Independent
  division gives **103.06495956812553 ms/token**, or **9.702618660991217
  tokens/s/user**, for all 48 layers, 128 input / 128 output, capacity 262144,
  serial 1x4 Blackhole execution. The source-audited prepared active-weight
  subtotal is 3,281,879,040 bytes/rank and the selected-KV payload is 2,359,296
  bytes/rank. At 512,000,000,000 bytes/s/rank, their bound is **6.414528 ms**.
  Thus the measured fraction is **6.2237719074%**, below **10%**; acceptance
  requires at most **64.145280 ms/token**. The remaining deficit is
  **38.919679568 ms/token**. These numbers agree with `perf_summary.json` and
  the inventory in `performance_accounting_plan.md`; intermediate product
  traffic and capacity-sized scorer outputs are correctly excluded from the
  acceptance denominator.

  Why this matters: `$optimize` explicitly retains this floor after the Stop
  Rule fires. Improvement over baseline, successful accuracy checks, and a
  small orchestration gap do not satisfy a failed required performance gate.
  `AUTOFIX_functional_floor.md` records actual discriminating experiments and a
  bounded repair failure, not a successful repair or proof of physical
  impossibility.

  Required next step: retain `stage_accepted: false` and the bounded AutoFix
  failure in the local checkpoint/handoff. A later authorized continuation
  must produce a correctly validated path meeting the floor before stage 7
  can be accepted. No additional device work is requested under the current
  cap, and there is no basis to repeat already completed unchanged gates.

## Other Concerns

- The seven untried families in `work_log.md::Not tried (stop rule)` are
  legitimate handoff work: exact scorer/selector epilogues, selected-KV gather,
  indexer/GDN scheduling, an alternative GR-down schedule, smaller matmul
  configurations, a safe persistent-output sampler gather, and multirow
  prefill work. Their contained-operation ceilings overlap and are not
  achievable-gain forecasts. I do not treat these untried candidates as
  additional closure blockers after the cap.
- Batch 32 preserves functionality but has limited performance evidence:
  `batch32_resident_v1.json` gives **2558.592232 ms/step**, **0.390839927
  tokens/s/user**, **12.506877649 aggregate tokens/s**, with 1617 trace
  segments per model step. Attention and the head still execute Q1 per slot;
  MoE uses packed 32-row execution. There is no matched prior B32 performance
  baseline, so no B32 speedup is established. The HF gate uses staggered
  histories of one correlated AIME reference, as disclosed, while the clean
  timing uses prompt lengths 125..128. This is not evidence for 32 independent
  long-answer workloads.
- The qualitative evidence establishes comparative behavior under its frozen
  degeneration rubric and 128-token cap. It does not establish solved AIME
  answers, completed Fibonacci functions, poetic meter, or full-answer quality.
  Directly inspected HF/TT text is coherent but frequently capped during
  reasoning or an incomplete answer. The matched-control limitation is
  recorded with `original_model: "same"` in
  `../eval_readiness/limitations.json`.
- Several historical/in-progress sentences remain in planning artifacts:
  `terminal_profile_analysis_v2.md` ends by calling the now-completed prefill
  recipe pending; `optimization_checklist.md` and a preceding work-log paragraph
  still mention B32 as running; `../context_contract.json` calls performance
  remediation active. The later timestamped completion and final numerical
  artifacts settle these states. These are documentation cleanup notes, not
  reasons to repeat a review or hardware gate. The top-level README correctly
  discloses the failed floor and completed device gates.
- Experimental `compact_gr_down` code remains in the patch, but it is not
  imported or dispatched by the selected decoder. It must remain disabled
  unless a future continuation resolves the integrated failure. Component
  exactness or speed alone is insufficient for promotion.

## Hard-Check Gaps

- The installed stage-7 shell check primarily checks autoregressive
  degeneration; its passing exit is not a FUNCTIONAL-floor check.
  `stage07_check_v2.log` includes the new v2 output as well as historical
  output, and the direct v2 degeneration report also passes. The floor failure
  is independently visible rather than hidden by the shell-check result.
- Reduced eager profiler attribution and reduced frozen-input component
  traces are not a simultaneous all48 device-time measurement. The final
  artifacts label this correctly. Re-summing the retained report CSVs gives
  **99.939983500 ms** weighted device work; re-summing component medians with
  multiplicities 35/1/12 gives **104.083627679 ms** including entry, terminal
  and sampling. They adequately explain the 103.064959568 ms full-model
  resident result within their stated limits, but do not prove an exact
  physical lower bound or isolate host overhead.
- Full48 prefill device-time reconstruction is absent because the six-bucket
  profile lacks a validated layer-kind partition. Per-bucket kernel tables,
  advice, raw CSV, coverage, and separate full48 before/after TTFT are present.
  Stronger attribution would be useful later; its absence is not an additional
  missing required gate for this capped review.
- No additional missing or failed correctness gate was identified in the
  inspected final evidence. No server/vLLM behavior is claimed or required in
  this stage. The existing full-model serving contract pins the plugin/vLLM
  pair and remains generator-scope evidence.

## Anomaly Ledger

- Observed anomaly: resident decode is below the mandatory performance floor.
  Evidence: final resident timing and independently recomputed useful-weight/KV
  bound above.
  Affected path: all48 B1 token-out decode.
  Control or comparison: same-harness baseline 138.618792796 ms/token;
  final 103.064959568 ms/token, with all 128 caller-control greedy tokens equal.
  Likely subsystem: retained device computation/data movement, chiefly exact
  projection and recurrent/indexed-attention work, rather than an unaccounted
  sampler/host boundary.
  Investigation performed: source audit, isolated AutoDebug/AutoFix record,
  component experiments, final default reproduction, reduced profiler and
  traced component reconciliation.
  Resolution: **more-work-needed**; bounded AutoFix failed and the cap ends
  current experiments. This is the Required Work item.

- Observed anomaly: native GR-down compaction passes component controls but
  fails integrated mixed-prefill equality.
  Evidence: `final_serving_watcher_v1.json`,
  `compaction_serving_repro_v1.json`,
  `compaction_eager_operands_v1.json`, and
  `compaction_integration_diagnosis_v2.md`.
  Affected path: experimental compaction dispatch.
  Control or comparison: restored production source passes all 16 reduced
  serving checks in `final_serving_watcher_v2.json`, followed by the all48
  validation and mixed-prefill gates. Selected raw profiler rows contain no
  compaction kernel.
  Likely subsystem: candidate integration/trace/state behavior; root cause is
  not established by conditioned eager comparisons.
  Investigation performed: canonical bit controls, repeated native-only
  reproduction, real-operand comparisons, and rollback.
  Resolution: **controlled** for the selected checkpoint by removing dispatch;
  the candidate remains rejected, with no claimed gain.

- Observed anomaly: reduced batched-vs-solo decode is not bit-exact.
  Evidence: `final_serving_watcher_v2.json` reports whole-output PCC
  0.9980032125, minimum row PCC 0.9934975416 and top-1 agreement 1.0.
  Affected path: existing two-row decode geometry.
  Control or comparison: these exceed the stage-review whole-output 0.995 and
  diagnostic-row 0.99-plus-top1 criteria. Fresh all48 B32 HF top5/top100 and
  reordered matched-history state/logit controls also pass.
  Likely subsystem: inherited active-width numerical differences.
  Investigation performed: inspected the explicit per-row/whole-output
  distinction and the fresh larger-batch evidence.
  Resolution: **controlled**; not a failed PCC gate.

- Observed anomaly: full-model latency is about 1.019 ms below the weighted
  component estimate, while some prefill raw descriptors retain a logical
  `32[2]` label for larger buckets.
  Evidence: `component_trace_v2.json`, `terminal_profile_analysis_v2.json`,
  `prefill_profile_analysis_v1.json` and the raw/report CSVs.
  Affected path: performance attribution and metadata interpretation.
  Control or comparison: reduced frozen-input residency/history differs from
  the full48 evolving workload; prefill driver logical counts and final device
  positions cover all six buckets, and independent full48 mixed/resume
  correctness passes.
  Likely subsystem: measurement-regime differences and profiler descriptor
  metadata; descriptor-reuse explanation remains unproven.
  Investigation performed: re-summed component/report timings, checked native
  source dispatch and runtime precision rows, and checked coverage and driver
  scope.
  Resolution: **controlled** by explicit limits on conclusions; no invented
  host-overhead or full48 prefill number is used.

- Observed anomaly: generation often reaches the cap before an answer finishes,
  and serialized thinking/EOS boundaries appear in text.
  Evidence: v2 AIME HF/TT completions and the six-prompt qualitative outputs.
  Affected path: generated-text interpretation.
  Control or comparison: same pinned checkpoint, chat template, prompts,
  sampling recipe and ten seeds per prompt; label-hidden review records 0/60
  rubric failures for each backend.
  Likely subsystem: matched model behavior and token cap.
  Investigation performed: direct reading of AIME outputs, all six greedy
  prompt pairs and seed-0 sampled pairs; checked control byte identity,
  paired prompt/seed keys, prompt metadata, rubric and verdict.
  Resolution: **controlled** under the stated comparative rubric, with the
  model limitation retained; no full-answer quality claim.

## Scope Inspected

- Goal/skill paths: `stage_review_contract.md`; installed package
  `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/`
  `stage-review`, `optimize`, `multichip`, `tt-device-usage`,
  `tt-enable-tracing`, `qualitative-check`, model-bringup startup, and optimize
  evidence/performance-accounting references. Latest user cap instruction
  takes precedence over further optimization and permits carrying nonblocking
  items without forcing three expensive reviews.
- Artifact paths: stage README/work log/manifest/budget; source manifests;
  baseline/final resident JSON and harness; final validation v2 and its accuracy,
  resume, context, autoregressive and qualitative outputs; raw B32 HF JSON and
  compact summary; B32 resident JSON/log; final full48 prefill signatures;
  restored-source serving Watcher JSON/log; component traces; final decode and
  prefill report tables, raw CSVs and coverage; operation-topology/checklist and
  AutoFix/compaction records; representative causal/exact-kernel/head/CCL probe
  artifacts; final hook logs and tile-guard patch; context/limitations and
  full-model serving contract.
- Code paths: tracked model/decoder diff; the selected causal scorer, exact Q1
  projection, exact residual writer and exact rank reducer, including their
  native kernels; generator preparation/replay/sampling boundaries; resident
  benchmark and validation runtime guard; batch/signature provenance changes;
  profile analyzer and selected runtime geometry. Experimental compaction was
  inspected for dispatch exclusion, not accepted as production behavior.
- Commands run: read-only `git status`, `git diff`, `rg`, `sed`, `cat`, and
  small standard-library Python artifact inspections/recomputations. No TTNN
  import, device open, hardware test, server, long test, or secondary reviewer
  was launched. This report is the only file written by the reviewer.

Independent checks include:

| Evidence | Inspection result |
|---|---|
| Source provenance | All 72 model and three shared source hashes in v4 match the live tree. Validation/B32 HF differ only in the scorer's documented standard-tile rejection guard. Post-guard resident, full48 mixed-prefill, profiler/component and B32 resident evidence bind the final source. |
| Accuracy | Prefill 97/100 top1, teacher forcing 96/100 top1; both 100/100 top5 and top100. Raw B32 records independently recount to 2645/2704 top1 and 2704/2704 top5/top100 in each of two reordered runs. |
| Resume and mixed prefill | Resumed length257 controls at starts32/68 pass; no-history and fresh-suffix controls fail as expected. Full48 mixed/reversed logits are exact; resumes68..257 and31..265 are exact; no late compilation or request-time release. |
| Trace/state and Watcher | Tracking enabled with program-cache allocations included; positive control rejects a live allocation and clean replay follows. Restored-source battery passes16 checks, prepared/final cache count2142, zero request-time releases; full48 mixed-prefill separately passes after the guard. |
| Context | Capacity262144 and BF16 page32 policy are preserved. Final32769 prompt smoke plus extra device-feedback replay passes; max context is inherited because KV layout/dtype did not change. |
| Runtime/feedback | Full48 guarded prefill/decode/sampling/capture bodies pass, with positive controls. B1 resident counters show127 model plus127 sampler replays and zero interior read/sync/refresh/capture/release. B32 has stable2150 program entries,4816 bindings and zero interior metadata uploads. |
| Performance | Baseline/final greedy token arrays are exactly equal,128 tokens. TTFT7035.653051→4705.950625ms; resident throughput7.214029064→9.702618661 tokens/s/user. Runtime rows confirm inherited FP32/BF16 exact paths, BF16/HiFi4 head and BFP4 routed experts; no new precision policy. |
| Local checkpoint/push | Checkpoint is planned after this review under explicit user instruction, even though acceptance fails. No push is planned or performed according to the run plan/manifest; no push was attempted by this reviewer. Unrelated `doc/fused_decoder` artifacts remain outside stage scope. |

## Residual Risk

- The mandatory performance floor is the only Required Work finding identified
  in this review. It prevents a passing stage verdict despite successful final
  correctness evidence and a valid capped checkpoint.
- Inspection cannot prove arbitrary-input numerical equivalence of all custom
  kernels. Finite real-model behavior, component arithmetic, repeated replay,
  Watcher, final top-k accuracy, mixed-prefill and longer-context controls
  provide the relevant current evidence. Nonfinite payload semantics of the
  exact projection are explicitly outside the established component proof.
- Larger-batch long-context throughput, independent long-answer workloads,
  full-answer quality, and actual vLLM integration remain outside the measured
  scope. No production serving or maximum feasible performance claim follows
  from this checkpoint.
- The local checkpoint SHA is not yet available at review time. The stage
  owner must record the actual SHA and retain this unaccepted verdict without
  including unrelated dirty artifacts or pushing.
