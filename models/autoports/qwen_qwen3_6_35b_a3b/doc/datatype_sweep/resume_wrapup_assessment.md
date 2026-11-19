# Resumed-prefill wrap-up assessment

Assessment of retained evidence at the user-directed stage cap (2026-11-16),
using [work_log.md](work_log.md), [AutoDebug](AUTODEBUG_resume.md), and
[AutoFix](AUTOFIX_resume.md). **No general numerical repair was found.**
No new experiment, gate change, selected-policy change, or device operation was
performed for this assessment. Historical diagnostic outcomes below do not
replace the once-only final selected-path acceptance result. At this snapshot,
the work log records that result as pending; production is checkpoint
`c873e26136c8140fe419e523525265ee7d3939dc`.

## What still fails, including matched HF

The unchanged default gate uses length 2560, block 32, a 256-position window
plus the final position, mean PCC >= 0.99, minimum PCC >= 0.98, top-1 agreement
>= 0.98, final argmax equality, and applicable negative controls. Original TT
start 32 passes; starts 68, 2528, and 2559 fail PCC. Top-1, final argmax, and
applicable negative controls satisfy the original check. Fresh final-position
logits repeat bitwise across cases.

The pinned HF BF16 controls use identical tokens and scored positions, model
revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. These are actual HF split
controls, not a claim that TT matches HF at every position.

| Start | TT mean / minimum PCC | HF mean / minimum PCC | Unresolved distinction |
| --- | --- | --- | --- |
| 68 | 0.994176 / 0.968224 | 0.988563 / 0.497114 | Both fail minimum; HF also fails mean. Failing positions only partly overlap. |
| 2528 | 0.989605 / 0.973851 | 0.991039 / 0.977859 | Both fail minimum at different positions; TT alone fails mean. |
| 2559 | 0.986428 / 0.986428 | 0.992442 / 0.992442 | TT alone fails mean; matched HF passes. |

At start 68, TT-only positions below minimum PCC are **105, 292, 300**;
shared positions are **118, 257, 304**; HF alone fails 13 other positions.
TT-only top-1 flips are 132 and 172, HF-only flips 113 and 130, and both flip
304. Aggregate top-1 is 0.988327 for each implementation. At start 2528,
**TT fails minimum at 2528; HF fails at 2531**, with no shared minimum failure.
All four paths choose the same top-1 at these TT-only low-PCC positions.
Starts 2528 and 2559 have perfect within-implementation top-1 agreement.
Shared case-level failures therefore do not explain the remaining TT-only
failures or justify a waiver. See [HF summary](hf_resume_control/summary.json)
and [six-pair position classification](hf_resume_control/failure_classification.json).

## What the investigation established

Source inspection found no demonstrated page-indexing, missing-history,
position-offset, or transport defect. Queue ordering and readback evidence are
consistent, without proving the absence of every possible race. FP32 recurrent
storage has a BF16 public-call boundary, matching the measured HF cached-state
semantics. Removing that boundary changes policy; it is not an established
porting correction.

The full-depth start-2559 capture reproduces original logits bitwise. Prefix
conv/recurrent state is preserved before the public boundary cast. Target 2559
first differs at layer-0 attention (PCC 0.99999682), then layer-0 MoE
(0.99991106). Common prefix 2558 remains exact through layer-1 attention and
first differs after layer-1 MoE. Later expert layers amplify differences.
CPU routing analysis predicts expert substitutions at five inspected later
MoE drops; actual TT expert IDs were not captured, and later input differences
are substantial. See [localization](resume_localize/L2560_s2559/statistics.json)
and [routing sensitivity](resume_localize/L2560_s2559/router_sensitivity.json).

Same-operand attention controls found no supported reader or SiLU repair.
K512 versus K1024 differences were tiny; the later recurrence probe found
identical raw recurrence, state, and final output for that K change. Identical
q/k/v/beta/state passed through chunk-n1 versus singleton arithmetic first
differed within recurrence/decay, but BF16-boundary final attention RMS was
only 1.65e-7. FP32 boundary carry changed output more (1.42e-5 RMS), without
establishing either result as ground truth. See
[attention control](attention_localize/L2560_s2559_l0/result.json) and
[recurrence analysis](linear_singleton_localize/L0_s2559_v1/analysis.md).

Frozen-input grouped-MoE tests establish real padding-dependent stochastic
rounding: padding 32 versus 512 changes output with SR enabled and becomes
bitwise identical without SR. Nevertheless, disabling SR increases the
EP-versus-indexed discrepancy for this operand. Public BF16 activations also
pass through hard-coded native BFP8 internal buffers; BF16 expert storage does
not make the operation internally all-BF16. No existing legal BF16-intermediate
toggle was identified. See [MoE control](moe_localize/l0_suffix2559_v2/result.json).

## Tested changes did not repair the complete requirement

The [completed diagnostic matrix](resume_variants/completed_diagnostics_summary.json)
records individual passing points, with no negative-control or selected-policy
acceptance claim:

- FP32 boundary carry, SR removal, and their combination do not pass all
  required splits. Grouped singleton alone passes 2559, but combinations can
  regress it; no accepted complete-gate result follows.
- Legal q32/k32 bounded attention with grouped singleton passes 68, fails
  2528 and 2559. Adding no-SR passes 2559 and PCC at 2528, but top-1 at 2528
  falls to 31/32 (0.96875); start 68 still fails minimum PCC (0.580381).
- Normal `accuracy_bf16_dense` passes 2559 but fails 68 and 2528. The paired
  all-40-layer BF16 grouped-expert-storage variant fails all three. Neither
  is a general repair or a completed ranked candidate.

No-SR's start-68 worst position 118 overlaps an HF outlier, but its split
change points in the opposite direction (centered-logit cosine -0.968139).
Matched HF/TT pairs have PCC 0.587208 and 0.559811, while crossed pairs have
0.992368 and 0.980353. All outputs retain token 220 with large margins.
No-SR has seven TT-only minimum failures. Position overlap and top-1 stability
cannot establish shared cause or acceptability; see the
[matched no-SR comparison](resume_variants/no_sr_s68_hf_comparison.json).

## Deferred handoff, not authorized new experiments

The following remain untried or incomplete and require a later work scope:

1. Localize start 68 at identical common prefix rows, especially layer-0
   post-MoE versus pre-MoE, with actual expert IDs, scores, counts, and row
   mapping. Freeze one captured input across fresh/prefix schedules to separate
   changed inputs from counts/padding sensitivity. The existing same-input MoE
   control covers one start-2559 operand, not this complete contrast.
2. Capture actual TT routes at the earliest relevant MoE divergence and compare
   router margins against the existing CPU predictions. Test route-conditioned
   expert outputs before attributing later-layer amplification to routing.
3. If those captures justify it, evaluate a row-independent rounding/padding
   design or an explicit native internal-precision design. Neither exists as
   an accepted runtime toggle; native changes need their own correctness and
   performance evidence. Do not describe a BF16 weight promotion as an internal
   activation promotion.
4. Test terminal physical-row placement or transport bytes only if localization
   points there. Current offsets agree; arbitrary lane changes have no
   demonstrated repair basis. Additional-layer recurrence controls may test
   generality, but repeating the already-tested layer-0 K/SiLU contrasts alone
   would not close the unresolved gate.

The retained `prefill_k32_grouped_singleton_proposal.patch` is unapplied and
its tested composite does not solve all splits. Recomputing prefixes, changing
thresholds, or making position-specific branches is not an accepted repair.
Any later justified numerical change needs all four unchanged default starts,
applicable controls, normal selected-policy construction, both full-model
accuracy references, P1, and fresh performance/provenance evidence. Current
full-model accuracy/timing passes do not imply resumed-prefill acceptance.
The remaining precision-search candidates are separately listed in
[the capped work log](work_log.md); no global optimum or lower-precision win
has been established.


## 2026-11-16 owner acceptance and final context acceptance

Owner decision: accept the resumed-prefill PCC miss (start 68 minimum PCC 0.968 vs 0.98; the matched HF BF16 control also fails minimum PCC on the same case, and top-1, final argmax and negative controls pass) as a known limitation, the same as the stage 6 decision. Keep the selected precision configuration, record this in limitations, finish the stage review and complete stage 8. No further resumed-prefill investigation.

Applied to closing stage 8 with the existing selected configuration and recorded resumed-prefill PCC failures. The owner specifically cites start 68; acceptance does not assert that HF reproduces every TT-only failure at starts 68, 2528 or 2559. Raw thresholds, failed results and HF comparisons remain unchanged. No further resumed-prefill investigation is authorized in this stage. See `owner_decision.md` and `owner_acceptance.json`; `doc/eval_readiness/limitations.json` now records the explicit acceptance while retaining raw FAIL and the HF distinctions. Earlier unresolved-status entries below/above are historical. No production or selected precision policy change was made.

The final context artifact passes: logical32769 in65536 allocation, next position32771, two model/sampler replays, zero input refresh/readback/recapture, stable native/Python/input/selected hashes and allocator totals. Free DRAM12,954,371,584bytes/device; advertised262144 context is preserved. The interrupted wrapper did not persist its exit status; the artifact and log show success and normal device closure at10:57:23UTC. The recovered attempt explicitly uses exit_code=null. No hardware job remains or was rerun.
