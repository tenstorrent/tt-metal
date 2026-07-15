# Stage Review

Verdict: clean-pass

## Required Work

- None.

The prior P1 candidate-matrix finding is closed. The review-added
`attention_output_bfp8_hifi2` row is a genuine full-60-layer, batch-1,
149+100-token traced run with 99 model replays; it passes at 90%/100%/100% but
regresses to 22.463623753965983 t/s/u versus the 23.139082344511007 baseline.
The selected BFP8/HiFi2 LM head combined with the independently passing
BFP4/HiFi2 attention output is likewise a genuine full run; it passes at
91%/99%/100% but regresses to 22.289408789244337 t/s/u. Both candidate JSONs
contain physical tensor-dtype summaries for all 60 layers and the raw K/L logs
close all four devices normally. Because isolated BFP8-output/HiFi2 did not
improve the baseline, the documented conditional selected-LM-head plus
BFP8-output/HiFi2 candidate was not triggered. This is an earned conditional
screen after direct isolated coverage, not the component-screening assumption
rejected by the prior review.

The prior P2 plot finding is also closed. Direct visual inspection of both
final 2540x1455 PNGs confirms a readable decision-region panel plus an explicit
all-policy overview. Both plots show all 21 exact policies, the global Pareto
frontier, the selected red star, the vertical dotted accuracy threshold,
passing/rejected marker separation, and collision-spaced labels for the
selection, nearest alternatives, frontier, and material failures. The 1%
residual-activation outlier remains visible without compressing the useful
decision region.

## Other Concerns

- The 19 original aggregate rows retain null
  `runtime_consumption.physical_weight_dtypes` fields because those runs
  predate the physical-summary enhancement. This is accurately disclosed in
  README/work-log, and it does not leave the selected path unproven: the
  selected normal-path token-out and qualitative artifacts physically report
  all 60 layers, while both new review rows also contain constructed tensor
  dtypes. Policy propagation is additionally covered by runtime/source tests.

- `lm_head_bfp8_hifi2` leads the next passing row by only 0.07280262382219416
  t/s/u, but it also has 92% rather than floor-level 90% top-1. The compatible
  attention-output refinements and the independently improving BF16-decode-CCL
  combination were run and regressed materially, so the small headline gap is
  not being protected by an obvious missing compatible candidate.

## Hard-Check Gaps

- No blocking hard-check gap remains. Candidate JSONs still record the
  immutable Stage 07 parent `727b333b7bf0a62cebcd01afcc9ff64c796deffa`
  beneath the live Stage 08 worktree rather than a nonexistent future Stage 08
  checkpoint. The work log now explicitly classifies that field, ties all
  numeric groups to the documented Stage 08 source set, identifies unrelated
  dirty paths to exclude, and gives the force-add plan for ignored CSV/log
  evidence. Per the stage-review ordering contract, the owner must now make a
  local Stage 08 checkpoint without pushing, update
  `context_contract.json` from `complete_pending_stage_review`, and append the
  branch and final commit SHA to the work log. This is the required post-review
  handoff, not missing pre-review evidence.

## Anomaly Ledger

- Observed anomaly: Metal warns that allocating device buffers while a trace
  is active may corrupt those buffers.
  Evidence: the warning appears once in each full candidate-group log and in
  `post_selection_token_out.log` and `qualitative.log`.
  Affected path: cooperating model/sampler trace setup.
  Control or comparison: the Stage 07 optimized-full-model anomaly ledger
  classifies the same warning on the unchanged trace lifecycle. Every one of
  the 21 numeric rows records exactly 99 model trace replays; the five
  token-out samples record 99 replays, zero token refreshes, one final sampled
  token readback, and zero full-logits readbacks; qualitative output has no
  TT-only corruption signature.
  Likely subsystem: conservative trace-allocation registration warning during
  setup of the cooperating traces.
  Investigation performed: compared full-run logs, trace counters, generator
  release/preallocation source, prior controlled evidence, and output behavior.
  Resolution: controlled.

- Observed anomaly: `residual_activation_bfp8` collapses to 1% top-1, 1%
  top-5, and 5% top-100 while remaining traced and relatively fast.
  Evidence: candidate JSON, group-A full log, aggregate row, and AutoFix
  reports.
  Affected path: full-stack BFP8 inter-layer residual policy.
  Control or comparison: BF16 residual baseline is 91%/100%/100%; isolated
  BFP8 embedding activation with BF16 residual reaches 89%/100%/100% and is
  independently rejected by the declared gate.
  Likely subsystem: accumulated residual precision loss after the separate
  cache-update and QKV-head-split kernel boundaries were repaired.
  Investigation performed: AutoFix diagnosed the two kernel contracts, proved
  repaired traced runtime through reduced hardware smokes, separated the
  embedding boundary, and then ran full-model accuracy.
  Resolution: controlled accuracy rejection.

- Observed anomaly: three full-group logs emit the legacy sharded `tilize`
  warning that the input shard spec is used for the output tensor.
  Evidence: `group_a_full.log`, `group_h_full.log`, and `group_j_full.log`.
  Affected path: candidate setup/tilization for those grouped policies.
  Control or comparison: the warning explicitly describes shard-spec reuse,
  not a host fallback; every affected group completes its full numeric rows,
  trace counts, accuracy checks, and normal device close. Selected normal-path
  token-out and qualitative logs do not contain this warning.
  Likely subsystem: legacy sharded optimized tilize factory diagnostics.
  Investigation performed: searched all full, post-selection, and qualitative
  logs for warning/error/fallback/corruption signatures and compared affected
  run completion.
  Resolution: controlled; it does not touch the selected default evidence.

- Observed anomaly: conservative capacity accounting is negative for the
  canonical BFP8-MLP/BF16-KV control.
  Evidence: `context_capacity.md` computes -860,726,664 bytes/device under the
  retained 12 GiB general reserve.
  Affected path: conservative accounting for the canonical control, not the
  selected BFP8-KV policy.
  Control or comparison: the canonical 60-layer run physically constructs the
  advertised-context cache, executes 100 tokens with 99 trace replays, and
  closes normally. The selected policy has 6,883,007,096 bytes/device accounted
  margin and retains the batch-3 physical upper bound.
  Likely subsystem: conservative reserve overlap rather than a hard DRAM limit.
  Investigation performed: checked the policy-specific weight/KV arithmetic,
  candidate runtime, and updated context contract.
  Resolution: controlled; no capability reduction is justified.

- Observed anomaly: completion prompt 1 repeats mechanically, and prompt 2
  repeats corpus-style instruction phrases.
  Evidence: direct inspection of
  `qualitative/vllm_qualitative_outputs.json` and degeneracy metrics.
  Affected path: base-checkpoint completion-mode qualitative suite.
  Control or comparison: the exact-revision `GemmaTokenizer` has no chat
  template; prompt 1 repeats identically in HF, prompt 2 is token-identical to
  the passing Stage 07 TT control, two prompts match HF exactly, and four of six
  TT outputs match Stage 07 exactly.
  Likely subsystem: base-checkpoint corpus-completion behavior, not selected
  precision, token feedback, cache state, or trace replay.
  Investigation performed: checked prompt metadata/rendered tokens, direct
  HF/TT output, Stage 07 controls, and the scoped degeneracy artifact.
  Resolution: controlled.

- Observed anomaly: the first Pareto artifacts were unreadable because the 1%
  outlier compressed the useful region and labels overlapped.
  Evidence: historical review and plot AutoFix report.
  Affected path: required Pareto reporting.
  Control or comparison: direct visual inspection of both regenerated final
  PNGs and source tests covering all-point overview/frontier/annotations.
  Likely subsystem: plotting layout.
  Investigation performed: inspected the final images at native resolution and
  reran the plot/source contract tests.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: supplied Stage 08 goal contract;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/datatype-sweep/SKILL.md`;
  `.agents/skills/qualitative-check/SKILL.md`.

- Artifact paths: datatype-sweep README/work log and prior review; all 21
  candidate JSONs, 22 configs, grouped raw logs, regenerated JSON/CSV, both
  PNGs, selected config, context-capacity report and context contract;
  post-selection token-out JSON/log; qualitative format/rendered prompts,
  HF/TT outputs, degeneracy result, verdict, and log; AutoDebug/AutoFix reports
  and repaired smoke logs; Stage 07 readiness/reference and qualitative
  controls.

- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`,
  `tt/multichip_decoder.py`, `tt/optimized_decoder.py`; candidate/grouped
  runners, normal-path qualitative/token-out runner, aggregate generator, and
  precision/cache/geometry/artifact/source/full-model contract tests.

- Commands run: read-only `sed`, `rg`, `find`, `jq`, `git status/diff/log`,
  `git check-ignore`, `stat`, `file`, artifact-invariant scripts, direct PNG
  visual inspection, and non-device static verification. Static result:
  `57 passed, 3 warnings in 15.13s`; separate source-contract result:
  `Ran 7 tests ... OK`. No TT device, reset, server, profiler, or vLLM run was
  started.

## Residual Risk

- The full candidate rows were measured from a documented live Stage 08
  worktree above a Stage 07 parent, not an immutable Stage 08 commit. The
  immediate post-review checkpoint and SHA/branch log update are needed to
  make that source set durable; unrelated dirty paths must remain excluded.

- The selected BFP8 LM-head policy has strong physical/default-path evidence,
  but the 19 older rows rely on resolved policy summaries plus source/static
  consumption evidence rather than per-row physical dtype maps. This is
  explicitly disclosed and does not change the verified winner.

- No vLLM adapter/integration source or serving run is present. The filename
  `vllm_qualitative_outputs.json` and `--scope vllm` are shared qualitative
  checker schema conventions used by the full-model harness, not Stage 08
  vLLM work.
