# Stage7 follow-up review brief, round2

Original contract: stage_review_contract.md plus the explicit2026-11-16 owner amendment. All hardware gates finished Nov14. No new tuning or hardware runs.

## Previous findings, verbatim

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


## Changes since round1

- Commit9799a084 applied only the clang-format delta documented in checkpoint_sources_v5.json after numerical measurements/review; source remains unchanged since that checkpoint.
- Commit3e05e46c recorded evidence and resolved prior documentation notes.
- Current documentation patch records the owner decision and performance limitation; no implementation, tests, raw benchmarks, thresholds or original review were modified.
- Current changed paths:

- `models/autoports/qwen_qwen3_8_flash_next/doc/context_contract.json`
- `models/autoports/qwen_qwen3_8_flash_next/doc/eval_readiness/limitations.json`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/README.md`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/manifest.json`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/operation_topology_audit.md`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/optimization_checklist.md`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/perf_summary.json`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/stage_review_contract.md`
- `models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/work_log.md`
- `doc/optimized_full_model/owner_decision_2026_11_16.json` (new, relative to model directory).

## Review scope

Check each prior finding closed/open under the explicit amended owner contract. Read current limitation entry and recorded owner decision. Verify metrics remain honest (numerical floor still false), unchanged checkpoint hashes, scope of styling-only source delta, and no new issue introduced by the docs patch. Do not re-review the whole stage, run hardware or spawn another reviewer. The historical round1 review is not rewritten. A clean-pass is valid if no required work remains under the amended contract; do not infer a claim that the original numeric floor now passes. May write only stage_review_round2.md.
