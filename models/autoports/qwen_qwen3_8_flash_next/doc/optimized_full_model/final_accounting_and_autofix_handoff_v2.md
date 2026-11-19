# Final accounting preparation and bounded performance hand-off

CPU-only review, 2026-11-14. Final terminal/profile and component-trace v2
measurements are pending. This document does not change production, README,
perf_summary, the floor denominator, or overall stage acceptance.

## The gate remains failed

The last completed all48 result before compaction is
`exact_integrated_resident_v1.json`: **103.068540624 ms/token**, 9.702282 tokens/s/user,
TTFT 4700.293363 ms, and all128 greedy tokens equal the original baseline.
The exact fusions improved the previous 113.881454 ms result by 9.495%.
The useful-weight/KV floor remains **6.414528 ms** and requires
**<=64.145280 ms/token**. That completed candidate achieves only **6.223556%**
and still needs **38.923260624 ms** removed.

Compaction is **rejected and reverted from production**: despite isolated
all-pattern/all-Q checks and a faster component, `final_serving_watcher_v1.json`
failed integrated mixed-prefill row0-versus-solo equality. Its isolated saving
cannot be credited or extrapolated into the retained full-model result.
`final_serving_watcher_v2.json` passes all16 serving checks after the revert.
Executable sources are frozen in `final_sources_v2.json`; final all48 validation
is running separately. The retained measured deficit is38.923260624 ms.

Keep the denominator as 3,281,879,040 useful prepared major-weight bytes plus
2,359,296 short-window selected KV bytes per rank, divided by512 GB/s per rank.
The selected exact fusions remove intermediate materialization, not required
weight/KV payload. Do not add FP32 product traffic, padded score writes, repeated
NoC reads, measured kernel durations, or inactive expert scans to this useful
work denominator. A reduced-profile kernel sum is not a substitute roofline.

## What has been tested, and what those tests refute

| Experiment | Measured evidence | Disposition / limit |
|---|---|---|
| Exact local residual writer | `exact_writer_clean_v1.json`: Q1 0.039336→0.012933 ms; tracked/Watcher changed-input controls pass | Kept, with explicit intermediate BF16 rounding; integrated baseline-token equality passes. |
| Hyperconnection norm RHS fusion | `exact_norm_clean_v1.json`: three real norm weights, Q1 ~0.0689→0.0632 ms | Kept. Add/rsqrt variant ~0.0672 ms is slower than RHS and not selected. This does not validate a different reduction tree or full native RMSNorm. |
| Exact GR-up R16 projection | `exact_projection_real_clean_v3.json`: 0.060828→0.047846 ms including query widening | Kept; R8 0.062021 ms is slower than its 0.060549 ms control. |
| Exact GDN-output R16 projection | Same artifact: 0.181423→0.119231 ms | Kept; R8 0.144928 ms is correct but slower than R16. |
| Exact GR-down native projection | R8 0.127198→0.189479 ms; R16 0.127393→0.208758 ms | Rejected as slower. Exactness succeeds, so the issue is kernel geometry/load/compute cost. Neither result proves every possible exact kernel impossible. |
| Exact router native projection | R8 0.046075→0.055927 ms; R16 0.046014→0.053529 ms | Rejected as slower; selected projection retained. |
| Local exact post-gather rank sum | `exact_rank_sum_clean_v2.json`: Q1 0.051082→0.020202 ms, Q32 0.155644→0.020196 ms, unchanged downstream scaling/cast | Kept, unchanged synchronous transport. All Q1..32, poisoned padding and changed-input controls pass. |
| Canonical GR-down compaction | `compact_gr_down_clean_v1.json`: Q1 0.040585→0.006272 ms; Q32 0.040602→0.013979 ms | Rejected after integrated mixed-prefill equality failed in final_serving_watcher_v1. Production reverted; v2 passes16 serving checks. Isolated exhaustive65536-pattern and all-Q controls were insufficient integration evidence. |
| Larger fixed-policy head blocks | `head_larger_blocks_v2.json`/`v3.json`: adapted20-core block8,40/80-core block10 correct but slower than20/3/block2 for Q1/Q32 | Rejected as slower after adaptation. Original8/1/1 and10/3/block8 L1 collisions remain specific physical blockers, not family-wide claims. |
| Sampler async barrier gather | `sampling_ccl_v1.json` plus tracked counterpart | Correct but slower; synchronous sampling retained. Persistent-output-buffer variant is unmeasured. |
| Force-argmax | `terminal_clean_v1.json`: 2.765952 ms versus split0.481979 ms | Correct but slower; full-vocabulary gather is not selected. |

These are discriminating experiments under the installed AutoDebug/AutoFix
diagnosis. Unsupported precision changes, reassociated FP32 reductions, a
replicated residual stream, or a changed cache layout are not admissible fixes.

## Final profile v2 reconciliation

Use the already-prepared versioned recipes. Do not profile all48 layers or
reuse v1 CSV row numbers after fusion changes the operation sequence.

CPU analyzer: `../../tests/analyze_full_model_profile.py`. From the checkout
root, run the following once the named v2 files exist. The resident argument
below intentionally names the last completed retained candidate; replace it
with the final matched-source resident report if a fresh run is completed.
The analyzer lists source differences instead of calling mismatched runs a
matched experiment, and refuses to overwrite any existing output pair.

```bash
D=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model
python models/autoports/qwen_qwen3_8_flash_next/tests/analyze_full_model_profile.py \
  --profile-dir "$D/terminal_final_perf_v2" \
  --driver "$D/terminal_final_ops_v2.json" \
  --component "$D/component_trace_v2.json" \
  --resident "$D/exact_integrated_resident_v1.json" \
  --floor "$D/functional_floor_audit_v1.json" \
  --source-manifest "$D/final_sources_v2.json" \
  --output-prefix "$D/terminal_profile_analysis_v2"
```

CPU regression against v1 reproduces the112.435613500 ms weighted eager sum,
the38.923260624 ms retained-candidate floor deficit, the sum of weighted kernel
groups, explicit source mismatches and overwrite refusal. Compilation and
format checks pass. Generated reports retain semantic advice review as pending;
raw kernel names/shapes/configs replace fragile historical row offsets.

1. Verify v2 driver source hashes against `final_sources_v2.json` and the final
   resident/validation source manifests. Resolve documentation-only versus
   executable differences explicitly. Require natural128-prefix staging,
   measured position132→133 on all ranks, C262144, B1, layers0/1/3, warmup4,
   no traces in the op profile, and strict host/device/final key coverage.
2. Confirm all24 markers, equal per-rank region counts, leaf partition of the
   model, and terminal parent=mixer+head. The old1,247 ops/rank count is not a
   required v2 invariant: kept fusions legitimately reduce it.
3. Rebuild each region from its final tt-perf-report CSV. Sum ordinary op
   max-rank timings and collective rank-mean timings as the tool does. Also
   retain independently summed raw per-rank durations. Never add enclosing
   MODEL/TERMINAL regions to their leaves.
4. Compute device estimate as35×L0 +1×L1 +12×L3 +token entry +embedding +final
   mixer +LM head +row assembly +split sampler. Preserve force-argmax as a
   separate alternative. Report the difference from clean all48 wall time as
   a reduced-eager/full-trace reconciliation residual, not directly measured
   host work.
5. Independently take `component_trace_v2.json::weighted_estimate`, without
   reconstructing clean trace latency from eager gaps. Keep its reduced
   residency, frozen component inputs and advancing-state caveats. Confirm
   first replay, input integrity, program-cache stability and position checks.
   Compare stack+terminal/sampler with final all48 token-out; a >10–15%
   positive residual requires investigation. The prior v1 estimate115.048771 ms
   versus full113.881454 ms already refuted a large orchestration gap for that
   source; it cannot be relabeled as final-source proof.
6. Match native operations by raw `COMPUTE KERNEL SOURCE`, data-movement kernel,
   shapes and dtypes, not only `GenericOpDeviceOperation` labels. Expect
   exact GR-up/GDN-output, residual-write and local rank-sum kernels to replace
   their old subgraphs. Verify rejected GR-down/router variants and compaction
   are absent from the measured graph; legacy compaction remains selected.
7. Read back actual fidelity, FP32 destination, BF16 KV/activation/CCL and
   head20/3/block2 geometry. The report's8-core DRAM heuristic can still produce
   >100% modeled FLOP utilization; raw program/shard geometry is authoritative.
8. Refresh the sampler domination check: expected candidate-table gathers,
   local TopkLargeIndices, no selected force-argmax/full-vocabulary gather or
   generic TopK. Keep teacher-forcing and token-out timing boundaries separate.
9. Compare v1/v2 attribution by semantic subgraph, recalculate new weighted
   budgets, and classify each advice row kept/rejected/exact blocker/untried.
   Fresh matched traces do not waive final AIME, qualitative, serving, B32,
   context, allocation and runtime-guard evidence.

## Highest justified remaining fixed-policy candidate

**First new experiment, only if a bounded optimization window remains:** fuse
the Q1 QSA score epilogue (ReLU, exact four-head H-reduction and scale) into the
existing causal scorer, retaining the current score shape and selector for the
first version. This targets a well-localized remaining sequence without
touching CCL, precision, cache ownership or token feedback.

In v1 layer3, post-dot rows231:236 cost
44.207+46.307+46.307+44.013+47.202=228.036 us: **2.736432 ms/full token** across
12 layers is the impossible zero-cost ceiling for those five operations.
The larger scorer-through-selector budget3.837540 ms includes native dot and
selector work that still must execute. Require v2 to confirm this remains
material before implementing it. Source still contains
`multiply(sum(relu(dots), dim=2), 128**-0.5)` for packed Q1.

Preserve the selected per-head dot order, ReLU exceptional-value semantics,
exact H-reduction tree including padded zeros, and scalar-rounding behavior.
Reuse the existing complete-group boundary/near-max score+ordered-ID probes,
changed-position traces, Q1/3/17/32 and watcher/tracker controls. Keep Q>1 on
the existing correct path initially. A compact row-major score representation
is a separate, larger selector-contract change and must not be mixed into the
first experiment. Do not skip score ordering merely because <=512 groups are
visible: selected-ID order affects subsequent SDPA arithmetic.

**Largest residual mathematical budget, lower confidence:** GR-down precise
projection. V1 down product+reduce costs10.383131 ms weighted; selected
reduction uses only3 cores. Existing native R8/R16 attempts are conclusively
slower, so repeating that geometry is not justified. A different reduction-only
decomposition with partial-row reads could preserve the same K-tile fold and
row tree, but would increase NoC request count, needs a new kernel and may still
lose. It is a research hand-off, not a quick untested promised10 ms win.

Other historical upper bounds, pending v2 confirmation: QSA ordered indexer
5.264052 ms, GDN QKV4.779234 ms, GDN recurrence2.549591 ms, MoE SwiGLU5.552740 ms,
sparse KV gather2.865492 ms and SDPA1.662144 ms. These are already-native or
mandatory arithmetic/data-movement groups. They are not mutually free savings,
and several require careful numerical-tree or cache/selection validation.
The ~6.26 ms CCL transport and sub-ms head/sampler cannot individually close
the38.923260624 ms gap. No demonstrated fixed-policy candidate closes it.

## Stop Rule and a truthful AutoFix outcome

Installed `optimize/SKILL.md` says optimization stops when (1) all checklist
items have measurements/decisions/blockers, (2) the last two kept candidates
each gain <3%, or (3) the stage wall cap is reached, default4h from stage start.
It also explicitly says that the Stop Rule **never waives** correctness or the
10% FUNCTIONAL bar.

Do not claim rule1 while exact small-matmul geometry or the new score-epilogue
candidate remains untried and lacks an exact blocker. Do not claim rule2 by
splitting the retained9.495% integrated fusion result into convenient small
sub-candidates: use measured kept-candidate boundaries. Compaction is rejected
and cannot count as either kept candidate. Current evidence does not establish
rule2.

The authoritative `stage_budget.json` records `get_goal.createdAt=1794654561`:
start **2026-11-14 11:09:21 UTC**, default four-hour deadline **15:09:21 UTC**.
`baseline_perf.log` begins at11:10:52.980 UTC and independently corroborates
that baseline execution followed the start. Invoke rule3 against goal creation,
not accumulated device time or file modification timestamps.
At the cap, retain the frozen executable source, finish required final evidence,
and list untried items under the exact work-log heading
`Not tried (stop rule)`. Do not launch another speculative kernel whose
integration would invalidate the ongoing all48 validation.

If final measurements remain above64.145280 ms after that bounded repair loop,
the defensible outcome is:

> AutoFix repaired the demonstrated avoidable costs and preserved the selected
> model contracts, but the bounded attempt **failed to meet the mandatory
> FUNCTIONAL performance gate**. The final measured candidate remains below10%
> of the strict useful-weight/KV floor. Stage7 is not accepted. Remaining exact
> kernel work is recorded as hand-off research; no denominator or precision
> policy was changed to manufacture a pass.

This is failure to attain the gate in the bounded attempt, not proof that the
hardware has a fundamental100 ms lower bound, not a relaxed acceptance, and not
a claim that untried kernels failed. Final review must retain that distinction.
Suggested hand-off entries are score-epilogue fusion (<=2.736432 ms historical
zero-cost ceiling), alternative exact GR-down reduction geometry (<=10.383131
ms product/reduce ceiling, prior fused variants slower), remaining fixed-policy
small-matmul L1/block/subblock configs (gain unknown), native QKV/indexer read
reuse, sparse-KV reader optimization, and recurrent value-row partitioning
(ceilings above, actual gains unknown). Larger head blocks are now measured
rejections and should be removed from the untried list.
