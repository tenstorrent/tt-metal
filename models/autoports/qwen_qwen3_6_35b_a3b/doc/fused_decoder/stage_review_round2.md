# Stage Review

Verdict: clean-pass

Independent follow-up, round 2 of 3, for stage 2 fused-decoder,
`Qwen/Qwen3.6-35B-A3B`. Reviewed the live worktree on base checkpoint
`ad48340200720f4c24d49a97321bd4cd2c5a5922`, with selected runtime SHA256
`d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`.
Scope was the prior findings and changed implementation/evidence. The explicit
owner wall/time/token cap governs completion; no further tuning is required.

## Required Work

None. The two prior required-work findings are closed:

- **Round 1 P1, layer-34 accuracy: closed.** Independently scanned every result
  in `stress_representative_34_seed154.json`: 24,576 decode rows, three repeats
  covering the same 8,192 positions, and 33 continuation rows, with no nonfinite
  or below-0.995 output/contribution metric. Minimum decode output PCC is
  `0.9999666394231101`; minimum contribution PCC is `0.999962901548466`.
  Minimum continuation contribution PCC is `0.9989890356985085`.
  The run uses fresh HF (`cpu_oracle_cache: null`), one trace capture,
  allocation tracking, included program-cache allocations, and bit-equal
  repeated outputs. The raw execution log records all three completed repeats
  and clean device closure; its launcher records exit 0. Current runtime,
  stress harness and reference hashes match their provenance. This is the
  retained failing regression rerun on the repaired device implementation,
  with unchanged strict checks, rather than an HF waiver or weakened oracle.
- **Round 1 P2, optimization stop condition: closed by explicit owner amendment.**
  The brief quotes the owner's instruction to end tuning at the stage cap,
  finish remaining gates once, and hand off untried ideas. Current README and
  work log identify this condition without claiming eight hours of active work
  or 70% roofline. The work log's `Not tried (stop rule)` section records the
  remaining selector/softmax, recurrence, direct-reduction, geometry/precision,
  and matched long-context performance ideas. These are authorized handoffs,
  not unfulfilled accuracy requirements.

The affected broader evidence also passes: I scanned the four additional
8192-step endurance JSONs (layer 0 seeds 913/42/154 and layer 3 seed 913), each
with three repeats and 33 continuation tokens. All output and contribution
metrics are finite and at least 0.995. The weakest decode contribution PCC is
`0.9969259572575533` on layer 3. Current-source context artifacts cover
262143/262144 prefill and deterministic decode at position 262143 for both
kinds. Current-source batch, boundary and trace-reuse artifacts are present.
Trace reuse records one capture, no request release, one teardown release,
the same trace ID across requests/resume cases, changed-input sensitivity,
and a rejected allocation-tracker negative control. Watcher reports are clean.

## Other Concerns

- **Prior historical-qualification wording: closed.** `pattern_assessment.json`
  identifies preceding records as historical and supplies a selected d810
  disposition pointing to `endurance/v6_d810_pattern_reconciliation.md`.
  Old V4/V5 results are not used as current accuracy acceptance.
- **Prior convolution-state description: closed.** The current
  `fallback_audit.json` says `BF16 ROW_MAJOR backing [1,3,8192]`, consistent
  with runtime allocation and the context contract.
- **Prior conditional performance evidence: closed.** Final pure measurements
  are bound to d810, while final correctness is established separately by the
  fresh acceptance suite. I recomputed medians from the 10 prefill and 60
  decode samples: layer 0 `478.1315684995 / 1.4475849998 ms`; layer 3
  `8.9377445001 / 0.5831014996 ms`. Final decode beats all nine manifest
  comparison rows. Linear comparators now include their own layer-34
  correctness evidence; eligibility is not inferred from old-suite names.
  The restored comparator-input manifest hash matches the hash consumed by
  `hard_gates.json`.
- **Performance wording should be precise, without another experiment or review.**
  README/work-log language that the selected implementation "beats every"
  comparator should specify **decode**. Final linear prefill is slightly slower
  than two repeated 85ad runs (`477.8659645001` and `478.0330579997 ms`), by
  about 0.056% and 0.021%. The checker explicitly requires faster decode and
  reports prefill deltas; it does not require an unchanged prefill path to win
  timing noise. The main tables are accurate, and the substantial regression
  against the original functional linear-prefill baseline is openly disclosed.
- **Historical tensor dump has an ambiguous adjacent filename.**
  `stress_representative_34_seed154.json.tensors.pt` is the preserved round-1
  failure, not the current passing JSON's tensor output. Its SHA256 is exactly
  the prior review's `2b08e72dbe08fc524a19195986d8f53740c7d45467b5ff354c644f888122d936`.
  The unchanged runner saves this file only on failure and leaves it on disk
  after a subsequent pass. A short historical label in the current index would
  prevent confusion; no deletion, archival pass, new tensor export or hardware
  rerun is needed. Current acceptance rests on the indexed current JSON,
  provenance, source and execution logs, as authorized by the owner.
- **Checkpoint/push state:** the supplied run plan authorizes a local checkpoint
  after this review and prohibits push. At review time the final work remains
  staged; the base checkpoint alone is not labeled final stage completion.

## Hard-Check Gaps

- The prior one-repeat `repeated_bit_equal` bookkeeping and local stress NaN
  predicate limitations remain in the unchanged harness. Final acceptance
  requires three repeats and explicitly checks finite metrics. Every current
  endurance row inspected satisfies those checks, so these limitations do not
  hide a current failure.
- The snapshot benchmark's generic `acceptance_eligible: false` and the profile
  validator's stage-ineligible labels correctly mean those artifacts alone do
  not establish stage acceptance. The aggregate checker binds selected-source
  timing to the production hash and separately requires correctness. Likewise,
  `raw_capture_validation.json` is an intermediate report-pending artifact;
  the final per-kind `profiler_completeness.json` says
  `complete_capture_and_reports_pass` and confirms generated phase tables.
- I inspected the checker instead of rerunning it, since it writes evidence.
  `logs/final_hard_gates.log` contains `FUSED_DECODER_HARD_GATES_PASS`;
  `logs/final_precommit_after_newlines.log` records the completed repository
  checks. No additional evidence format or archival gate is imposed.
- Independently invoked the host-only interleaved-tail auditor's `audit()`
  function without writing its result. It passed on d810, confirmed the exact
  inverse transformation and current context-source identity, and verified that
  neither TTNN nor Torch was imported in that audit process.

## Anomaly Ledger

- **Observed anomaly:** round-1 layer-34 drift and tied expert mismatch.
  **Evidence:** prior review, current retained regression, CPU selector proof,
  `endurance/AUTOFIX_v6_dynamic_selector.md`, and current `fused_decoder.py`.
  **Affected path:** linear attention and routing.
  **Control/comparison:** unchanged actual-model HF stream and strict PCC bars.
  **Investigation:** inspected the scalar device selector, ordered/corrected
  arithmetic, current provenance and all result rows.
  **Resolution:** repaired; no host fallback or accuracy waiver. The selector
  proof explicitly limits direct-logit versus FP32-softmax equivalence for
  artificial underflow/tiny-value distributions; universal bit equivalence is
  not claimed.
- **Observed anomaly:** recomputing the adjacent tensor dump reproduces the old
  `0.927889633267 / 0.912541466536` minima rather than the current JSON.
  **Evidence:** exact old tensor hash, older modification time, and
  `stress_fused_decoder.py`'s failure-only save condition.
  **Affected path:** artifact identification, not the final execution.
  **Control/comparison:** final log gives all three new passing minima and exit 0.
  **Investigation:** CPU-only tensor recomputation and source inspection.
  **Resolution:** identified as preserved historical failure; excluded from
  current acceptance and recorded above for indexing clarity.
- **Observed anomaly:** four full-attention `to_layout` placement warnings.
  **Evidence:** fresh capture's `layout_warning_review.json`, raw `ops.csv`,
  and native `slice.cpp`.
  **Affected path:** native router score/index slices.
  **Control/comparison:** all 44 actual output metadata records plus three
  rejected metadata mutations.
  **Investigation:** independently checked all 44 raw output-memory fields:
  every one is `DEV_0_DRAM_INTERLEAVED`; output types are BFLOAT16 or UINT16.
  Native slice passes the requested memory configuration to its final primitive.
  **Resolution:** classified composite advisory; no missing final transfer.
- **Observed anomaly:** tiny linear-prefill ranking reversals between repeated
  candidate runs.
  **Evidence:** recomputed sample medians and all nine comparison rows.
  **Resolution:** disclose the precise decode win; no material new prefill
  regression, tuning or gate repetition is warranted under the owner cap.

## Scope Inspected

- **Goal/skills:** complete `stage_review_round2_brief.md`; installed
  tt-model-bringup 0.1.24 `stage-review`, `graph-fusing`, `optimize`,
  `tt-device-usage`, and model-bringup startup/validation requirements.
- **Artifacts:** final README, manifest, work log, hard gates, comparison
  manifest, all five final endurance streams and their provenance; final pure
  sample arrays; final context, batch, trace-reuse and source/Watcher audits;
  final profile validation, completeness, raw router metadata and table
  existence; pattern reconciliation and selector repair/proof records.
- **Code:** staged runtime/audit changes; `InlineSelector`, corrected recurrence
  call sites, `TransposeCorrectedMatmul`, its reader, `WeightedNormTail` and its
  interleaved reader; stress and acceptance harnesses; native slice memory path.
  The transpose reader preserves increasing-K arithmetic while mapping physical
  tile/face coordinates; the tail reader publishes its input buffers after the
  NoC barrier and removes the former compute-side publication. No new persistent
  tensor ownership or runtime host transfer was introduced by these two changes.
- **Commands:** read-only `git diff/status`, `rg`, `sed`, and small Python
  JSON/hash/median/CSV analyses; the host-only source auditor; CPU-only
  `torch.load` and PCC calculation over
  the preserved historical tensor dump. No TTNN import, device open, reset,
  hardware test, server or implementation mutation was performed.

## Residual Risk

- This is decoder-stage validation, not full-model generation or serving
  qualification. Layer 0/3/34 coverage does not claim every model layer was
  independently exercised.
- The latest gains are small and the scalar/corrected numerical paths retain
  substantial linear-prefill cost. Untried performance ideas are handed off
  under the explicit owner stop condition.
- Large current raw artifacts remain local under the owner's logs-plus-manifest
  policy. Historical bundles must not be restored over current acceptance paths.
- No blocking correctness, contract or missing-evidence issue remains in this
  follow-up scope. Routine documentation clarification does not require another
  review round.
