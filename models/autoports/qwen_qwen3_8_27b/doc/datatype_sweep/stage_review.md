# Stage Review

Verdict: clean-pass

Independent final review, 2026-10-25. Stage08 datatype sweep for
Qwen/Qwen3.8-27B, live `exp26-qwen38-bringup` worktree starting at
`af283dd6d375f8f7ffe057b2c1e271f9e4aef3e9`. This final review supersedes the
interim more-work-needed report archived as `stage_review_initial.md`. All its
required measurement and reconciliation work is now complete. Implementation
and runner source hashes in the final benchmarks match the inspected worktree.

## Required Work

None. The stage satisfies the reviewed datatype-selection contract. Local
stage-owned checkpoint commits and receipt recording follow this independent
clean pass; no push or vLLM work is authorized by this review.

The selected `head_bfp8_hifi2` policy passes99/100/100% readiness, the shared
qualitative suite with documented HF-controlled truncation, and all four long
quality cases. It is the fastest evaluated policy satisfying those gates.
Selection uses36.3809209 traced teacher-forcing tokens/s/user; the separate
normal-default4096/128/B1 autoregressive median is36.17090288005509 tokens/s/user
and431.26330710947514 ms TTFT, reproduced from three raw records. Each record
contains127 model/sampler replays, zero host refreshes/captures and one final
readback.

Recomputed selected4096-input autoregressive medians are21.33486695,13.85968783
and8.89952050 tokens/s/user at batches8,16 and32. The warmed32768-input/B32
single-repeat result is7.46256615 tokens/s/user. Native262141 nonaligned prefill,
exact262144 prefill and explicit final decode input262143 pass finite/capacity
checks with retained trace IDs. B8/B16/B32 mixed prompts33/65/1 repeat exactly,
then return to all-active execution with unchanged traces/capture counts.

All four actual policy audits pass. The final ledger recomputes8704 KV
bytes/token/device and a1,056,768-token B32 pool, sufficient for32x32768. I
replayed every recorded allocation/deallocation event from each hashed compressed
raw audit. All reconcile exactly with zero new live allocations at the end. The
largest measured peak is28,142,107,136 bytes/device, below34,178,731,008 capacity.
Final context, agentic, performance, selection and telemetry artifacts agree.
The exact supplied packet uses only the measured autoregressive B1 headline;
unmeasured device-time/roofline fields remain null with reasons.

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
  recurrence or sampling changes. Final inventories and captured actual arguments
  establish the same policy for all four measured batches and native fused paths.
- The required material BFP4 LoFi and BFP8 LoFi/HiFi2 comparisons are present.
  The grouped decoder comparison is accurately described as a full-model
  comparison, not isolated per-group timing. The two pyplot charts contain all
  measured policies, correct threshold lines, numerical frontier and selected
  red point; the fastest numerical frontier is correctly quality-rejected.
- Selected default-path readiness is99/100/100% over100 generated positions.
  The independently checked B1 autoregressive result improves on the prior
  same-regime35.25243314779324 tokens/s/user. No observed teacher-forcing versus
  token-out selection conflict requires a new comparison.

- The agentic checker exits1 for six documented representation-copy advisories;
  no critical finding is present. The inventory counts each physical tensor's
  bytes once, and inherited separate prefill/packed decode and norm layouts have
  measured reasons. These advisory copies do not prevent the target pool fitting.

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

- Observed anomaly: selected clean B32 timing emits the generic active-trace
  allocation warning.
  Evidence: `selected/benchmark_b32_perf.log`, allocator.cpp
  `verify_safe_allocation`, `anomaly_ledger.md`, prior
  `optimized_full_model/runtime_audit.md`, selected tracked default validation
  and prior tracked B32 full/mixed/full controls.
  Affected path: eager request setup/prefill while traces exist, with tracking
  disabled for clean timing.
  Control or comparison: source emits once/thread for any allocation while the
  unsafe flag is set, excluding trace-storage context; it is not a specific
  tracker-detected lifetime violation. The selected head weights are allocated
  before capture. Head output dtype/shape and persistent request/bucket/cache
  ownership are unchanged. Eager temporaries are retired and logits copied into
  persistent storage before replay. No new broad acknowledgment was introduced.
  Likely subsystem: generic allocator warning.
  Investigation performed: inspected warning source and changed head/reset/
  preparation paths, checked exact current source hashes, selected tracking-on
  B1 validation (program-cache allocations included), prior tracked batched
  controls, and final selected all-active/masked replay plus raw allocation
  reconciliation. All new audit allocations retire; repeated outputs and trace
  IDs remain consistent.
  Resolution: controlled. The inspected precision change creates no concrete
  missing representative batched tracker check, and no extra hardware run is
  required by this warning alone.

## Scope Inspected

- Goal/skill paths: parent-provided stage08 contract; installed0.1.21
  `stage-review`, `datatype-sweep`, `agentic-serving`, `qualitative-check`,
  `tt-device-usage` and model-bringup startup instructions.
- Artifact paths: datatype-sweep README/work log/search/policy/selection/schema;
  all candidate result summaries and preserved failure rows; baseline and
  selected readiness/provenance; selected and rejected short/long output text;
  corresponding HF controls; AutoDebug/AutoFix/reset/geometry evidence; both
  Pareto images; complete selected B1/B8/B16/B32 benchmarks and hashed raw
  allocation events; final propagation/memory/performance/check summaries;
  context/agentic contracts and the exact stage telemetry packet supplied in the
  review request.
- Code paths: `tt/precision.py`, changes to `model.py`, `generator.py`,
  `multichip_decoder.py`; candidate/full validation and benchmark runners;
  selected benchmark wrapper; quality, propagation and final reconciliation
  scripts. Inspected inherited preparation/native-context paths as needed to
  assess changed behavior.
- Commands run: read-only `rg`, `cat`, `sed`, `git status/diff`; small Python
  standard-library JSON/CSV/hash/median/provenance and raw allocation-event
  analyses; read-only execution of the existing policy/provenance assertions
  without their report writes; image viewing.
  No TT imports, device access, servers, resets, profiler runs or target model
  experiments. Only this review report was written.

## Residual Risk

- One AIME24 problem/100 generated positions and a small qualitative suite do not
  establish dataset-level accuracy or broad agent-task reliability.
- Native context evidence proves capacity and finite outputs, not full-model HF
  accuracy at262144. The B32 long-prefix timing has one warmed measured repeat;
  the ordinary4096-input rows each have three.
- Tested batch/context pairs do not imply every batch at native262144 context.
  Prefix-cache serving behavior and vLLM integration belong to later stages.
