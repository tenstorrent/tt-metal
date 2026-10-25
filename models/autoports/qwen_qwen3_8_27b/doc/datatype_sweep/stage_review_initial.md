# Stage Review

Verdict: more-work-needed

Independent interim review, 2026-10-25. Stage08 datatype sweep for
Qwen/Qwen3.8-27B, live `exp26-qwen38-bringup` worktree starting at
`af283dd6d375f8f7ffe057b2c1e271f9e4aef3e9`. The stage owner is still running
the serialized selected-policy benchmarks. This verdict identifies completion
work, not an observed new implementation defect or a terminal blocker.

## Required Work

- P1: Finish the selected native-context and agent-batch evidence.
  Evidence: `selected/benchmark_b1_perf.json` currently contains three warmed
  autoregressive 4096/128/B1 records, but no completed `context_validation`,
  `allocation_audit`, or `trace_allocation`. Selected B8/B16/B32 results are not
  yet available. The B1 median is 36.17090288005509 tokens/s/user and
  431.26330710947514 ms TTFT. Each record has 127 model/sampling replays, zero
  host refreshes/captures, and one final readback.
  Why this matters: the selected policy must preserve native262144 and valid
  nonaligned prompts, fit the32x32768 target working set, and have measured
  batch1/8/16/32 behavior. Earlier-stage memory figures do not establish the
  selected policy's final ledger.
  Required next step: complete the already-running benchmark wrapper, inspect
  native262141/262144 and final-position checks, mixed/masked replay checks,
  allocator audits and batch32 long-prefix result; rerun only failed or missing
  work. Do not repeat the completed candidate matrix or quality suite absent a
  relevant implementation change.

- P2: Reconcile final selected-policy artifacts and telemetry after those runs.
  Evidence: `selection_state.json` still says benchmarks pending;
  `../agentic_serving.json` identifies stage07; selected memory/performance and
  propagation summaries are pending. The exact telemetry packet
  `72c6bef3-7aa8-4a33-8578-ceab07255f0d.json` still has null token-out metrics and
  baseline accuracy provenance. README/context statements also await the final
  BFP8-KV result.
  Why this matters: later construction and stage reporting need one consistent
  policy, capacity ledger and measured performance result.
  Required next step: run the existing offline reconciliation/propagation
  checks, update the context and agentic contracts, README/work log, selection
  state and exact packet from completed raw evidence, then request this review's
  final follow-up. Keep teacher forcing as the selection metric and
  autoregressive4096/128/B1 as the performance headline. Leave unavailable device
  time/roofline metrics null with reasons. Local checkpoint commits follow a
  clean independent verdict; no push or vLLM work is required here.

## Other Concerns

- No additional implementation defect was found in the precision-loader,
  model/head construction, decoder geometry adaptations or BF4 cache-reset
  changes inspected. The selected validation's recorded source hashes match
  all current model sources and the recorded rotary source.
- Recomputed all14 completed candidate medians and verified all64 layer indices,
  three traced repetitions,99 decode model replays and zero captures per run.
  The JSON and CSV each retain19 rows:14 evaluated policies plus five historical
  failures. The failures do not contribute invented throughput points.
- Actual selected kernel arguments show BFP4/LoFi ordinary/fused decoder
  projections and BFP8/HiFi2 head. Complete fixed representations are validated
  by the loader, rather than silently accepting unsupported residual, norm,
  recurrence or sampling changes. The pending batch audit should complete the
  same proof for native batched paths.
- The required material BFP4 LoFi and BFP8 LoFi/HiFi2 comparisons are present.
  The grouped decoder comparison is accurately described as a full-model
  comparison, not isolated per-group timing. The two pyplot charts contain all
  measured policies, correct threshold lines, numerical frontier and selected
  red point; the fastest numerical frontier is correctly quality-rejected.
- Selected default-path readiness is99/100/100% over100 generated positions.
  The independently checked B1 autoregressive result improves on the prior
  same-regime35.25243314779324 tokens/s/user. No observed teacher-forcing versus
  token-out selection conflict requires a new comparison.

## Hard-Check Gaps

- Presence/schema gates alone cannot establish that the narrative responses are
  coherent. This review read the selected long responses and relevant rejected
  outputs directly, and reran the existing exact-prompt/control provenance
  assertions without writing their outputs.
- The first two matrix runs lack captured kernel-argument inventories; their
  runtime layer policies/weight inventories and source hashes, followed by later
  concrete same-policy kernel inventories, support their stated scope. This is
  not a request to repeat those full-model runs.
- Native capacity/finite logits are not native-context HF accuracy. The existing
  contract distinguishes those claims; final reconciliation must preserve that
  distinction.

## Anomaly Ledger

- Observed anomaly: five policies with passing AIME token agreement fail a
  fixed-budget free-running task.
  Evidence: `quality_candidates/*`, `selection_state.json`, actual completion
  text and exact-prompt HF/baseline controls.
  Affected path: full64-layer autoregressive structured or narrative generation.
  Control or comparison: structured HF128 and baseline114 complete exact4/3
  JSON; HF narrative512 and prior TT1024 reach final text. Different long control
  caps are explicitly recorded.
  Likely subsystem: accumulated precision/fidelity effects on token choices.
  Investigation performed: inspected actual truncated JSON/continued thinking;
  checked matched prompts and unchanged candidate budgets. Selected
  `head_bfp8_hifi2` gives structured JSON114/EOS, coherent4096/4095 narratives
  1024/760 with EOS, and exact4096/32769 retrieval47/33 with EOS.
  Resolution: controlled; faster failing policies rejected and selected quality
  passes. This does not establish broad task accuracy.

- Observed anomaly: selected short explanation/code do not finish at128 tokens.
  Evidence: `selected_validation/qualitative/*_tt.json` and corresponding
  `../optimized_full_model/qualitative/*_hf.json`.
  Affected path: shared short suite with thinking enabled.
  Control or comparison: same rendered prompts, token IDs, greedy setting and
  128-token budget; HF also truncates the explanation and code.
  Likely subsystem: bounded output budget and model reasoning behavior.
  Investigation performed: read both controls and selected output; explanation
  remains correct and code outlines an ordered deduplication implementation.
  Resolution: controlled budget limitation; no completed-code claim is justified.

- Observed anomaly: BFP8 weights, BF16 CCL and native BF16 KV initially exceed
  live L1; BF4 KV reset initially performs costly host packing.
  Evidence: preserved failure JSON, AutoDebug/AutoFix reports, geometry controls,
  native BF16 replay, BF4 exact-shape reset primitive and real-generator replay.
  Affected path: candidate preparation/projection/SDPA/reset.
  Control or comparison: real target weights with matching precision, legal
  geometry alternatives, affected lengths1/33/4095/4096/4097 and native/batched
  controls. Invalid cross-precision PCC controls remain preserved and corrected.
  Likely subsystem: physical circular-buffer/workspace budgets and BF4 creation
  fallback.
  Investigation performed: inspected narrow dtype-dependent geometry branches,
  in-place reset and proof artifacts; successful full64 reruns complete all
  previously failing policies. Selected BF8 reset/geometry is unchanged.
  Resolution: fixed/controlled; candidates are rejected on measured speed or
  quality, not their original runtime errors.

## Scope Inspected

- Goal/skill paths: parent-provided stage08 contract; installed0.1.21
  `stage-review`, `datatype-sweep`, `agentic-serving`, `qualitative-check`,
  `tt-device-usage` and model-bringup startup instructions.
- Artifact paths: datatype-sweep README/work log/search/policy/selection/schema;
  all candidate result summaries and preserved failure rows; baseline and
  selected readiness/provenance; selected and rejected short/long output text;
  corresponding HF controls; AutoDebug/AutoFix/reset/geometry evidence; both
  Pareto images; partial selected B1 benchmark; context/agentic contracts and the
  exact stage telemetry packet supplied in the review request.
- Code paths: `tt/precision.py`, changes to `model.py`, `generator.py`,
  `multichip_decoder.py`; candidate/full validation and benchmark runners;
  selected benchmark wrapper; quality, propagation and final reconciliation
  scripts. Inspected inherited preparation/native-context paths as needed to
  assess changed behavior.
- Commands run: read-only `rg`, `cat`, `sed`, `git status/diff`; small Python
  standard-library JSON/CSV/hash/median/provenance analyses; image viewing.
  No TT imports, device access, servers, resets, profiler runs or target model
  experiments. Only this review report was written.

## Residual Risk

- One AIME24 problem/100 generated positions and a small qualitative suite do not
  establish dataset-level accuracy or broad agent-task reliability.
- Pending benchmark/native/memory results remain genuine stage closure gaps;
  this interim report is not a conditional clean pass.
- Tested batch/context pairs do not imply every batch at native262144 context.
  Prefix-cache serving behavior and vLLM integration belong to later stages.
