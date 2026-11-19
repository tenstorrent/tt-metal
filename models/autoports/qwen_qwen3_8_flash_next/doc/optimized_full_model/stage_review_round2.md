# Stage Review

Verdict: **clean-pass**

Independent follow-up review, round 2 of 3, of stage 7, optimized-full-model,
for `Qwen/Qwen3.8-Flash-Next`. Reviewed the live documentation patch on `main`
at `3e05e46cc7096fa27dd86019844d058e60603788`, with implementation checkpoint
`9799a08439f6405e224ca30db7526d317fd6f86b`. This is a pass under the explicitly
amended owner contract. The original numerical FUNCTIONAL floor still fails.

The authoritative instruction supplied to this reviewer is:

> Owner decision: accept the current decode speed (about 9.7 tok/s/user vs the 15.6 FUNCTIONAL floor) as a known limitation for this run. Record it in limitations with this decision and the evidence, finish the review and complete stage 7. No further tuning for this floor.

`owner_decision_2026_11_16.json` and `stage_review_contract.md` preserve that
decision. Its authority is the explicit owner message delivered through the
runner, as supplied in the review task; it is not inferred from an agent-written
document or from the original goal prompt. Only this run's performance-floor
acceptance blocker is superseded. Accuracy, correctness, capability, trace,
state, policy-preservation and local-checkpoint requirements remain intact.

## Required Work

- None under the amended contract.
- **Previous P1, mandatory batch-1 FUNCTIONAL floor: closed as an acceptance
  blocker by explicit owner decision; numerical failure remains.** Independent
  recomputation from `final_resident_v1.json` gives 127 steps in
  13.089249865151942 seconds, or **103.06495956812553 ms/token** and
  **9.702618660991217 tokens/s/user**. The retained prepared-weight plus
  selected-KV denominator, `(3281879040 + 2359296) / 512000000000`, gives
  **6.414528 ms**, hence **0.062237719074250665** of the roofline. The original
  10% floor still requires **64.145280 ms/token**, or
  **15.58961158170952 tokens/s/user**. The new decision, limitation entry,
  manifest and performance summary agree with these numbers and retain false
  numerical-pass flags. `AUTOFIX_functional_floor.md` remains a bounded repair
  failure, not a successful repair or proof of physical impossibility.
  `stage_review_round1.md` is unchanged. No further tuning or device rerun is
  required or authorized for this closure.

## Other Concerns

Each prior Other Concern is disposed of separately:

- **Seven untried optimization families: open as nonblocking handoff notes.**
  `work_log.md::Not tried (stop rule)` and `optimization_checklist.md` retain
  scorer/selector epilogues, selected-KV gather, indexer/GDN scheduling,
  alternative GR-down scheduling, smaller matmul configurations, safe
  persistent-output sampler gather and multirow prefill work. Their ceilings
  remain overlapping contained-work bounds, without an achievable-gain claim.
  The owner decision does not authorize further floor tuning in this run.
- **Batch-32 evidence limits: open as a disclosed limitation.**
  `batch32_resident_v1.json` still records 127 steps in 324.941213469021 seconds:
  **2558.592232 ms/step**, **0.390839927 tokens/s/user** and
  **12.506877649 aggregate tokens/s**, with 1617 segments per model step.
  The README retains Q1 attention/head per slot, packed32 MoE, the absent
  matched prior B32 baseline, and the correlated AIME-history scope. It claims
  neither B32 speedup nor 32 independent long-answer workloads.
- **Qualitative output-window limits: open as a controlled known limitation.**
  `../eval_readiness/limitations.json` retains
  `full_model_shared_thinking_128_token_cap`, including the
  `optimized_full_model_refresh`, `original_model: "same"` and its matched
  evidence paths. The README continues to restrict conclusions to the frozen
  comparative rubric and 128-token cap. No solved-AIME, completed-code,
  poetic-meter or full-answer quality claim was introduced. Historical
  limitation entries are unchanged; the new performance exception is a
  separate entry and does not pretend to be reproduced HF behavior.
- **Stale planning status: closed for current-state interpretation.**
  `terminal_profile_analysis_v2.md` now calls the six-bucket prefill analysis
  complete. The checklist and later timestamped work-log completion record
  settle B32 completion. The context contract now records the accepted
  performance limitation and pending follow-up. Some older planning and
  capped-disposition paragraphs still say running/unaccepted; the dated
  completion and owner-amendment text explicitly supersede them. These
  historical wording remnants are not a new gate or review blocker.
- **Experimental GR-down compaction: closed for the selected path; candidate
  remains rejected.** The selected decoder/model sources contain no import
  or dispatch of `compact_gr_down`, and all current model/shared hashes match
  the checkpoint. The README/checklist retain the integrated numerical
  rejection and prohibition on promotion from component speed/exactness
  alone. The experimental helper's formatter delta does not enable it.
- Current `followup_review_pending` fields and false stage-acceptance fields
  correctly describe the state before this verdict. The stage owner should
  record this review, finalize those administrative fields and create the
  local documentation checkpoint as the normal post-review closure. Preserve
  the false numerical floor result and the historical round-1 verdict. This
  does not require another review or hardware run.

## Hard-Check Gaps

- No new missing or failed required check was introduced by this documentation
  patch. The original stage shell check is not a performance-floor check;
  acceptance of this run follows the explicit owner amendment, not that
  script's exit code.
- The prior limits on reduced profiler/component attribution, the lack of a
  validated full48 prefill layer partition, finite qualitative coverage and
  the absence of B32 before/after performance evidence remain. This follow-up
  did not rerun or re-review unchanged gates.
- Independently checked all **72 model and 3 shared checkpoint SHA-256 hashes**:
  all match the live files. The v4-to-v5 manifest difference is exactly the
  ten documented native files, and the shared hashes are identical.
- Recovered the ten measured pre-format files from existing staged Git blobs
  using read-only `git fsck`/`git cat-file`. Each blob's SHA-256 matches the
  recorded measured hash. Inspected every unified diff: only whitespace,
  line wrapping and braces around existing single return, continue,
  assignment or function-call statements changed. No declaration moves
  scope; no operand, operator, statement order, API or configuration changes.
  Non-whitespace/non-brace text is identical in all ten pairs. This supports
  the documented formatting-only scope, without claiming binary equivalence
  or a post-format hardware measurement. The owner subsequently retained
  `checkpoint_clang_format_delta.patch` and
  `checkpoint_format_delta_verification.json`; their blob identities match
  this independent recovery, and the patch equals the independently
  regenerated complete unified diff.
- Existing numerical leaves in the changed context, limitations, manifest and
  performance JSON are unchanged. All new owner-decision evidence paths
  exist. `final_resident_v1.json`, the round-1 review, AutoFix record and v4/v5
  source manifests are byte-identical to the evidence checkpoint.
  `owner_closure_hooks_v1.log` reports passing applicable hooks, and this
  review's `git diff --check` passes.
- `closure_audit_2026_11_16.json` is explicitly a CPU audit of retained
  evidence, with pending review/commit fields and false numerical-floor pass.
  Its floor, source-count and resident-boundary claims agree with the direct
  checks here; its remaining unchanged-gate mappings retain round-1 scope
  and are not new device measurements.

## Anomaly Ledger

- Observed anomaly: resident batch-1 decode remains below the original floor.
  Evidence: unchanged `final_resident_v1.json`, `perf_summary.json` and the
  independent calculation above.
  Affected path: all48 batch-1 resident split-sampling token-out, 128 input /
  128 output, capacity 262144, serial 1x4 Blackhole.
  Control or comparison: original 15.589611582 tokens/s/user floor versus
  measured 9.702618661; the prior matched-baseline and AutoFix evidence remains.
  Likely subsystem: retained device computation/data movement, as classified
  in round 1; this follow-up introduces no new root-cause claim.
  Investigation performed: checked the explicit amendment, rederived the
  metric, verified unchanged raw evidence and inspected the scoped limitation.
  Resolution: **controlled by explicit owner acceptance for this run**;
  performance is not repaired and the numerical predicate stays false.

- Observed anomaly: ten native checkpoint hashes differ from measured v4
  hashes because the staged commit hook applied formatting after measurement.
  Evidence: `checkpoint_sources_v5.json`, `source_commit_hooks.log`,
  `source_commit_hooks_v2.log` and matching measured Git blobs.
  Affected path: six causal-score kernels, three exact-residual-write kernels
  and the disabled experimental compaction writer.
  Control or comparison: measured versus checkpoint source bytes, plus all
  75 live checkpoint hashes.
  Likely subsystem: repository formatter applied to newly staged files.
  Investigation performed: independently matched both sides' hashes and
  inspected all ten diffs, including each added brace's statement scope.
  Resolution: **controlled**; source-level formatting delta only, with
  original measurement provenance and post-format measurement caveat retained.

The unchanged controlled anomalies in round 1 retain their scoped dispositions;
the owner exception does not extend to them or waive a correctness defect.

## Scope Inspected

- Goal/skill paths: original runner prompt
  `/home/ttuser/qwen38-flash/multigoal_logs/07-07-optimized-full-model.prompt.txt`;
  `stage_review_contract.md`; `stage_review_round2_brief.md`; installed
  `stage-review/SKILL.md`, model-bringup startup and the applicable optimize
  FUNCTIONAL-floor/Stop-Rule clauses under
  `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/`.
  Enabled dependency configuration and package environment setup were checked.
- Artifact paths: all nine tracked documentation changes listed in the brief;
  `owner_decision_2026_11_16.json`; round-1 review; source manifests and hook
  logs; final resident and B32 timing fields; matched qualitative limitation
  entry; relevant completion/handoff paragraphs; `checkpoint_commits.log`;
  the retained formatter patch/verification and CPU closure audit.
  The brief's prior Required Work and Other Concerns are verbatim copies of
  the round-1 sections.
- Code paths: all 75 checkpoint files by hash; exact v4-to-v5 diffs of the ten
  files listed in `checkpoint_sources_v5.json::changed_files`; selected
  decoder/model files for compaction-dispatch exclusion. There is no current
  implementation/test delta and no non-documentation change between the
  implementation and evidence checkpoints.
- Commands run: read-only `cat`, `sed`, `rg`, `git status`, `git diff`,
  `git show`, `git rev-parse`, `git fsck --unreachable --no-reflogs --no-progress`,
  `git cat-file`, `git diff --check`, package environment validation and small
  standard-library Python JSON/hash/diff/calculation scripts. No TTNN import,
  device open, reset, server, hardware test, tuning or secondary reviewer.
  This report is the only file written by this reviewer.
- Push state: the supplied run plan, manifest, work log and checkpoint receipt
  record no push planned or performed. This reviewer performed no commit or
  push. The stage owner retains responsibility for the final local docs commit.

## Residual Risk

- Decode still delivers about 9.7 tokens/s/user in the stated regime, below
  the original 15.6 floor. This is an accepted performance limitation for
  this stage-7 run, not evidence of meeting the floor or a blanket exception
  for later stages.
- Correctness evidence is the unchanged finite evidence accepted in round 1;
  no new device behavior is asserted by this follow-up. There is no binary
  equivalence proof or post-format measurement.
- B32 independent long-answer performance, full-answer quality, full48
  prefill device attribution and production serving remain outside the
  established evidence. Experimental compaction remains disabled.
