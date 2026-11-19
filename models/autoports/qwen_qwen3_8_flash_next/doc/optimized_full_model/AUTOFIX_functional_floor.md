# AutoFix: full-model FUNCTIONAL performance floor

Status: **bounded AutoFix failed to restore the required FUNCTIONAL floor**.
The selected exact-fusion path passes refreshed numerical and serving checks,
but final resident decode is103.064959568ms versus64.145280ms required. The stage
cap has fired. Experimental compaction remains disabled after reproduced
mixed-prefill failure. Remaining required evidence and independent review are
being completed; the optimized stage is not accepted.

## Initial failing contract (historical causal/head candidate)

The then-selected full48, batch1, 128-input/128-output resident token-out window takes
113.881454ms per step. The unchanged prepared-major-weight plus selected-KV
roofline is 6.414528ms at512GB/s per rank:5.632636% efficiency, below the skill's
10% floor. Passing requires at most64.145280ms, another49.736174ms reduction.
See [functional_floor_audit_v1.json](functional_floor_audit_v1.json) and
[perf_summary.json](perf_summary.json). Intermediate product/scorer traffic is
excluded from that denominator.

The selected change already improves resident throughput21.72% and TTFT31.70%
against the matched baseline, with all128 greedy tokens unchanged. Those gains
do not waive the remaining floor or other stage gates.

## Verified diagnosis

The installed isolated AutoDebug launcher completed successfully; its report is
[AUTODEBUG_functional_floor.md](AUTODEBUG_functional_floor.md). Root source
inspection confirms:

- Exact GR/GDN projections materialize full FP32 products before a separate
  accurate reduction. GR-down reduction exposes only three32-row tasks.
- The selected sum seeds from K tile0, folds later tiles in order, then performs
  the existing SFPU row reduction. Split-K, FMA and native matmul are not exact
  substitutes. FP32 GR weights and the explicit BF16 rounding must remain.
- Local residual write uses eight operations around two mandatory BF16
  roundings; the inherited native writer has the correct arithmetic but a
  different four-stream geometry.
- Rank sums retain synchronous BF16 transport, with avoidable local formatting
  around the selected FP32 reduction. A direct reducer needs an independent
  proof of the rank tree and padding treatment.

The report's31.338994ms expanded projection/compaction/GR-sum budget is a
historical zero-cost ceiling. Even its42.004552ms broader budget does not promise
the required49.736174ms. Fresh selected profiling and actual experiments are
required; the diagnosis does not establish a physical impossibility.

## Initial discriminating repair experiments (historical setup)

1. **Exact Q1 projection fusion.** New test-local `exact_projection_q1` streams
   unchanged weights into R4/R8 independent output-row workers. Compact scratch
   and a uniquely owned assembler avoid partial-output races. First compare raw
   FP32 and final BF16 bits on small mixed-format controls, then real GR-down,
   GR-up, GDN-output and router matrices. Compare warmed complete projector
   latency including assembly/cast. Conservative per-tile zeroing remains a
   suspected cost; any zero-once variant requires ownership proof and the same
   checks. Q>1 and production dispatch remain unchanged during this experiment.
2. **Local2560 exact residual writer.** Adapt inherited native arithmetic to one
   whole residual stream per rank. Compare all four distinct ranks, Q1/3/17/32,
   rounding/cancellation controls, changed-input traces, allocation tracking and
   separate Watcher execution. The formula remains
   `BF16(FP32(residual) + FP32(BF16(FP32(block)*FP32(injection))))`.
3. **Exact norm epilogue.** Compare the existing merged-RHS helper against the
   original local2560 norm using real FP32 norm weights. Keep square/mean, eps,
   rsqrt, both multiplies and final BF16 rounding; require exact scale/output
   bits. Historical native RMSNorm remains rejected. Earlier merged-RHS gain
   was negligible; current mesh timing, not that history, decides selection.
4. **Post-gather rank sum.** Source investigation only until the exact H-reduce
   tree, padded-zero additions and input immutability are established. Keep
   synchronous BF16 all-gather unchanged. Poison padding and test rank-distinct
   cancellation/zero cases before considering a model-local implementation.

At this initial experiment point all candidate modules were opt-in test paths.
No production dispatch changed while `final_validation_v1` was running. Device jobs remain bounded and serial.
Commands are in the two fusion plans and probe docstrings. This file will record
each actual result, rejection or promotion; an unrun idea is not a failed fix.

## Promotion and failure criteria

Keep only exact, trace-safe component improvements, then measure complete layers
and full48 token-out. Refresh affected AIME, qualitative, serving-state, B32,
nonaligned and context gates after production changes. Compute the selected
weighted layer stack plus terminal/sampler work separately from the bandwidth
floor. Retain precision, rejection ledger, cache layout, four-stream residual
ownership and split sampling throughout.

At that initial experiment point no AutoFix result was declared. The final
bounded repair outcome below supersedes that interim status. The optimize Stop
Rule never turns an unmet FUNCTIONAL floor into a pass.

## Component experiment results, 13:12 UTC

- Exact residual writer passes tracked/Watcher Q1/3/17/32, four distinct ranks,
  cancellation and changed replay controls (`exact_writer_watcher_v1.json`).
  Clean Q1 median0.039336→0.012933ms, Q32 0.039434→0.013041ms
  (`exact_writer_clean_v1.json`). Candidate integration pending.
- Norm RHS fusion preserves exact FP32 scale/BF16 output on three real norm
  weights, Q1/3/17/32 and near-epsilon controls (`exact_norm_tracked_v1.json`).
  Clean Q1 local norm~0.0689→0.0632ms; Q32~0.0592→0.0536ms
  (`exact_norm_clean_v1.json`). Add+rsqrt is slower than RHS; native RMS remains
  rejected. Candidate integration is limited to hyperconnection norms.
- Projection R8/R4/R2 zero-once passed tracking/Watcher mixed BF16/FP32 controls
  (`exact_projection_smoke_v4.json`). Real-weight tests found GR-down and router
  slower, so these roles retain selected dispatch. R16 and input-precast were
  tested separately (`exact_projection_real_clean_v3.json`): GR-up including
  exact query widening0.060828→0.047846ms; GDN-output0.181423→0.119231ms.
  GR-down widening remains slower0.127198→0.189479ms. R16 smoke plus GR-up/GDN
  real weights pass tracked Watcher (`exact_projection_watcher_v5.json`).
  These use deterministic synthetic inputs, not captured full-model residuals;
  integrated correctness and latency still decide selection.
- Rank-sum v1 padding-fixture assertion, v2 unsupported one-shape device reshape,
  and v3 missing fabric initialization are harness failures, not numerical
  rejections. Each exited cleanly. v3 passed physical padding and baseline invariance checks but failed before
  the native kernel, at the first baseline collective. The corrected fixture proves physical poison and
  same-address explicit padded shape; selected fabric setup repair is pending.
- Fresh selected reduced eager profile and independent traced components both
  account for full-model latency (`terminal_profile_analysis_v1.md`,
  `component_trace_v1.json`). Representative traced weighted stack113.436162ms
  plus entry/terminal/sampler1.612609ms=115.048771ms versus full113.881454ms.
  Frozen reduced residency is an estimate, not an exact all48 lower bound.
  There is no measured >10% host/orchestration gap. The strict floor remains
  unmet; this accounting cannot substitute for the bandwidth floor.

Projection v2 command omitted required snapshot and exited before hardware;
corrected command is embedded in v3 JSON. No result from a failed harness is
counted as a performance rejection. Current component gains are not full-model
claims and do not establish AutoFix success or failure.

13:14UTC rank-sum tracked_v4 and clean_v2 pass every Q1..32, four rank-distinct
inputs, cancellation/negative-zero, poisoned physical padding, changed traces
and runtime guard. Clean Q1 post-gather baseline0.051082→0.020202ms; Q32
0.155644→0.020196ms including unchanged downstream scale and BF16 cast.
Clean_v1 omitted --clean-timing and exited before mesh creation; corrected v2
passed. Synchronous transport is retained. Eligible Collectives.sum dispatch
also reaches QSA/MoE reductions, so integrated correctness must cover all kinds.

## Integrated measurement, 13:22UTC

Combined exact candidate passes reduced16-check serving Watcher/tracking and
all48 clean128/128 resident benchmark:103.068541ms (9.702282t/s/u),
TTFT4700.293363ms, all128 tokens equal original baseline. This is another
9.495% decode-latency improvement over causal/head-only selected_v1, but the
strict fraction is only6.2236%;38.923261ms remains to64.145280ms. Fresh full
AIME/qualitative/context/B32 evidence and final layer accounting remain pending.
Artifacts: exact_integrated_serving_watcher_v1.json,
exact_integrated_resident_v1.json, exact_integrated_sources_v1.json.

## Compaction semantic diagnosis, 13:30UTC

The unintegrated raw-bit compaction was not accepted from normal-valued tests.
The existing four-slice/concat path invokes untilize/retilize and transforms
BF16 exceptional encodings. v1/v2 tests failed the legacy-vs-host control before
native execution; finite negative zero/subnormals were not waived.
`gr_compaction_canonicalization_v1.json` exhaustively checks all65536 encodings
and518 Q1 controls per rank: all exponent-zero values map to positivezero,
exponent255 clears the mantissa while retaining sign (NaN becomes signedInf),
and all other bits remain unchanged.509 inputs change. The candidate was then adapted to reproduce this exact observed baseline
mapping, with strict baseline and native equality for all encodings; subsequent
results follow. This is preservation of selected
behavior, not a new precision/NaN policy.

## Canonical compaction component pass; integration gate failed, 13:37UTC

[Watcher v3](compact_gr_down_watcher_v3.json) passes every Q1..32 underWatcher10
and allocation tracking, including program-cache allocations. Exhaustive Q32
families cover all65536 patterns across the four ranks; Q1 covers518 special/
boundary controls per rank. Native/baseline/independent canonical-host equality
is strict for finite, subnormal, signed-zero, Inf and NaN inputs, with native
physical padding checked as positive zero. Changed-input trace replay and
runtime/allocation guard positive controls pass. [Clean v1](compact_gr_down_clean_v1.json)
passes Q1/3/17/32 and exhaustive controls with tracking/Watcher disabled:
Q1 .040585247916→.006272341125ms; Q32 .040602310037→.013978860807ms.
These are compaction-only traces, not complete model gains.

The attempted `GatheredProjection` integration selected native compaction for valid
width80, BF16 TILE/DRAM [1,1,Q,384], Q1..32 onBlackhole1x4. Other shapes retain
the original path. Projection, gather, explicit BF16 boundaries, downstream
arithmetic and residual ownership are unchanged. The successful native
operation deliberately preserves legacy exp0→+0 and exp255→signedInf; all
other storage bits are unchanged. [Failed v1](compact_gr_down_watcher_v1.json)
and [failed v2](compact_gr_down_watcher_v2.json) remain failures before native
execution, not numerical exemptions. [Compaction record](compact_gr_down_proposal.md)
contains source invariants, complete clean rows and exact log/provenance links.
**Integrated gate failed:** [final_serving_watcher_v1.json](final_serving_watcher_v1.json)
reports mixed-prefill logits row0 differ from the solo control, with clean
close. The earlier exact-fusion candidate passed serving controls and
compaction is the only new production behavior in this attempt. This is a
discriminating lead, not a proven cause. Compaction is **rejected at the
integrated numerical gate**: the owner reverted only its production dispatch
and started the serving Watcher control v2. The prior exact-fusion candidate
remains selected; helper/kernel/probe remain experimental reproduction
artifacts. Restored-source control and final full-model validation are required. No full-model
performance claim or AutoFix completion/failure follows from component PASS.

## Optimize Stop Rule audit, 13:38UTC

The selected [Optimize skill](../../../../../../codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/optimize/SKILL.md#stop-rule)
says to stop when any checklist-complete, last-two-kept<3%, or stage-wall-cap
condition holds. The last-two-kept condition is **not established**: the latest
measured integrated exact-fusion bundle gained9.495% in decode latency; the
compaction has component measurements but failed its integrated gate and is
not a kept candidate. Slower head/
projection/CCL trials are rejections, not kept candidates. Component timings
cannot be divided by full-model latency to invent two measured sub3% gains.

Checklist work and fresh integration evidence remain pending, so checklist
completion is not established either. The owner confirms stage start11:09:21UTC
from the early environment/goal records; default4h wall deadline is15:09:21UTC.
At this13:41UTC update the wall cap has **not** been reached. Neither the
last-two-kept condition nor the default wall cap explains a stop at this time.
The owner has explicitly cut off further optimization experiments and proceeds
with correctness/control and final evidence work only. Record untried items
under `Not tried (stop rule)` when wrapping up, distinguishing this owner cutoff
from an elapsed default cap. Bounded optimization effort does not waive the
still-failed10% FUNCTIONAL floor or any correctness/review gate.

## Selected-source validation, 14:24 UTC

`final_sources_v2.json` binds the selected exact-fusion path with compaction
dispatch reverted. `final_serving_watcher_v2.json` passes all16 serving checks.
`final_validation_v2.json` passes AIME24 prefill97/100 top1 and teacher-forcing
96/100 top1; both are100/100 top5 and top100. Resume257, the32769-token
non-aligned context smoke, device feedback, all48 QSA/rank position checks and
the complete runtime fallback audit pass. Fresh blinded qualitative comparison
passes0/60 failures for HF and0/60 for TT under the shared rubric, with explicit
128-token-cap limitations. These correctness results do not waive the remaining
FUNCTIONAL performance floor. Batch32 and final selected-source performance
accounting are pending.

## Bounded AutoFix outcome after the stage cap

**Failed to restore the required FUNCTIONAL floor.** The user confirmed the
stage time/token cap; optimization experiments are closed. Exact causal scoring
and the head configuration improved the original full-model path; subsequent
exact projection, norm, residual-write and rank-sum repairs improved it further.
All kept changes passed the affected numerical and serving checks. Native
GR-down/router alternatives and larger head geometries were measured slower.
Compaction passed component and conditioned real-operand checks but failed
native-only serving traces and remains disabled.

The final default reproduces 103.064959568 ms/token (9.702618661 tokens/s/user),
versus the required <=64.145280 ms/token. The unchanged useful-weight-plus-KV
floor is 6.414528 ms; achieved fraction is about6.2238%, leaving38.919679568 ms.
Representative component traces total104.083627679 ms including terminal work;
weighted device profiling gives99.939984 ms. Thus an unaccounted host/sampler
gap does not explain the shortfall: split greedy contributes only0.426224 ms.
See `final_resident_v1.json`, `component_trace_v2.json` and
`terminal_profile_analysis_v2.json`. No denominator inflation, rejected
precision policy, residual replication or force-argmax substitution was used.

This is a failed bounded repair, not proof that the floor is physically
impossible. Further algorithm/kernel work is a handoff. The cap does not waive
the failed gate; the optimized-full-model stage remains unaccepted. Remaining
required evidence and independent review are being completed once.

Compaction's final reproduction is scoped precisely in
`compaction_integration_diagnosis_v2.md`: native-only traced serving has a finite
row1 logit mismatch, while row0 logits/cache pass; earlier integrated failure
was row0. Eager dual comparisons pass28 real calls and Q1/2/3/32 final states,
but baseline-before-native and readbacks can change state/lifetimes. No fixed
slot cause or LLK root cause is claimed. The candidate stays out of production.

Final review disposition: all remaining device gates finished once, including `batch32_resident_v1.json`. `stage_review_round1.md` independently reproduces the floor failure and identifies no additional correctness blocker. The prior bounded AutoFix failure remains unresolved; no new experiments are launched after the user cap. This is an unaccepted checkpoint, not a pass-with-carried verdict.
