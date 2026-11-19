# AutoFix: resumed-prefill baseline PCC investigation (active)

Contract: retain the installed default resumed-prefill gates (length 2560, block 32, window 256; mean PCC >= .99, minimum >= .98, top-1 >= .98, final argmax equal and applicable negative controls fail). No gate or prompt change is a repair.

The exact baseline passes start 32 and fails PCC at starts 68, 2528 and 2559. Top-1 and negative controls pass. All devices closed normally. Saved fresh full-prefill last-position logits are bitwise identical across three cases, ruling out uncontrolled run-to-run randomness for that row.

The matched pinned HF BF16 controls reproduce minimum-PCC failures at 68 and 2528, but pass 2559. The additional TT singleton failure remains work. See doc/datatype_sweep/hf_resume_control/summary.json and the full pairwise per-position records. Neither inherited owner acceptance nor the new HF evidence waives this case.

Diagnosis: AUTODEBUG.md and doc/datatype_sweep/AUTODEBUG_resume.md. Source inspection found no established page, lost-history or transport defect. It identified counts-dependent grouped-MoE stochastic rounding/internal BFP8 activations, public BF16 recurrent-cache rounding shared with HF, and singleton changes in projection grouping, recurrent arithmetic, gated SILU and sparse expert kernels.

Experiments in priority order:
1. Capture actual all-40-layer before/after attention and post-MoE values plus relevant state for fresh versus split start 2559. Separate prefix differences from singleton execution. No production policy change before localization.
2. At the earliest divergence, compare identical operands through the differing singleton/chunk helper; assess against the matching arithmetic reference.
3. If grouped MoE owns the early prefix divergence, compare identical real operands under changed grouping, with stochastic rounding independently enabled/disabled. Treat any accepted rounding change as a measured precision policy.
4. Apply only a supported minimum repair, then rerun the unchanged complete default gate, accuracy references, traced timing baseline and eventual selected-policy gates.

Current status: investigation active. No production runtime change or passing-gate claim. The datatype ranking queue is paused to avoid binding measurements to a runtime that may change.

## Full-model localization result

`tests/datatype_resume_localize.py --start 2559` completed normally; `doc/datatype_sweep/resume_localize/L2560_s2559/statistics.json` reproduces both original logits bitwise. Every prefix recurrent/convolution state is bitwise preserved up to the public boundary cast. At target2559 the first difference is layer0 attention (PCC0.99999682, RMS0.000055), then layer0 MoE (PCC0.99991106, RMS0.000381). Common prefix position2558 is exact through layer0 and layer1 attention, then differs after layer1 MoE. Larger differences accumulate at later expert layers. This excludes an observation perturbation and supports numerical-path investigation; it does not yet establish a repair.

Diagnostic-only same-input MoE and attention controls are prepared. `tests/datatype_resume_variants.py` can independently test uniform FP32 prefill boundary carry and disabled grouped stochastic rounding without production edits. Any resulting candidate still requires unchanged default negative controls, both full-model accuracy references, P1, and performance measurement.

## Same-input attention control

`tests/datatype_attention_localize.py` completed on layer0 captured target2559 with normal device closure. Cached512 and narrow512 return bit-identical FP32 partials and BF16 final outputs. Narrow1024 differs only by at most9.536743e-7 in17 rank0 elements plus denormals on rank2. SFPU SiLU gating, production lookup, and CPU FP64 control produce identical BF16 output on all four ranks. No supported reader or SiLU repair follows from this operand. Artifact: `doc/datatype_sweep/attention_localize/L2560_s2559_l0/result.json`.

The first MoE probe stopped on a diagnostic host reshape of a physical32-row reduce result. This is a harness issue; original model explicitly assigns one logical row. Normal device closure; failed result retained. Fix the diagnostic extraction and rerun in a new directory.

## Same-input MoE and routing controls

`moe_localize/l0_suffix2559_v2/result.json` completes with frozen actual singleton routing and exact normalized inputs. CPU FP64 sums of all rank contributions distinguish TP4 and EP4 ownership. 32 versus512 grouped padding with SR enabled changes real output (PCC0.999999332, RMS1.1117e-5), whereas disabling SR makes both padding shapes bit-identical. Same-shape repeats are bit-identical with either setting. EP-versus-indexed RMS is0.000222 with SR and0.000423 without; unquantized checkpoint-reference RMS is0.000398 indexed,0.000263 EP+SR,0.000182 EP withoutSR. Thus SR padding sensitivity is real, but disabling it alone need not align singleton indexed execution with grouped prefill. This is a one-operand diagnostic, not a full-model recommendation.

`resume_localize/L2560_s2559/router_sensitivity.json` finds one CPU-predicted top8 expert substitution at each of the five largest inspected MoE drops (10,14,28,30,39). FP32/FP64 predictions agree; exact input changes explain crossings. Later inputs differ by up to14.94% relativeL2, so later divergences are not called infinitesimal. Actual TT expert IDs were not captured in the all-layer run.

Next full-model diagnostic adds uniform grouped execution for singleton *public prefill* (pad only within MoE, slice output), alongside FP32 public boundary carry and noSR. It leaves decode unchanged and introduces no position-specific special case. No production change has been made.

## First full-model policy factorial

`resume_variants/s2559_v1/results.json` completed (918 seconds launcher elapsed). Baseline parity is exact. PCC at2559: baseline0.986428019; FP32boundary0.988450697; noSR0.984980587; FP32+noSR0.984673379; grouped singleton0.991211126 PASS; allthree0.989482285 FAIL. All top1 tokens agree. No negative controls run in this diagnostic. Only grouped singleton passes this point; uniform SR removal is not by itself a supported full-model repair.

Next diagnostic covers remaining factor pairs and non-singleton splits: FP32/noSR/both at68,2528; FP32+grouped and noSR+grouped at2559. The source remains unchanged. A policy that only improves one recorded point is not yet accepted.

## Remaining split classification and attention constraints

`hf_resume_control/failure_classification.json` identifies TT-only low-PCC
positions 105, 292 and 300 for start 68, and position 2528 for start 2528.
At start 2528, HF fails a different position (2531) and passes mean PCC; TT
fails mean PCC. All four paths choose the same top-1 at those TT-only low
positions. Shared failing position names do not prove matched causes, so no
blanket HF waiver is supported.

The current q32/k256 chunked SDPA requires starts divisible by 256, for both
the scalar API and the documented tensor API. Scalar validation enforces both
q/k divisibility. The absence of tensor-value validation is not evidence of
support for unaligned starts. A q32/k32 configuration permits 32-aligned starts
but requires device evidence. Paged fill copies whole pages and cannot directly
fill at start 68 without overwriting rows 64–67. Padding a partial segment to
32 rows and cropping can be causally valid only with correct page alignment,
prefix preservation, and logical valid-length handling. Singleton and bulk
normalization, projection, and head layouts also differ before SDPA.
No attention policy change has been made.

## No-SR matched-HF audit

`resume_variants/no_sr_s68_hf_comparison.json` compares all 257 positions in
float64. No-SR's minimum at position 118 overlaps an HF outlier but reverses
the HF split direction (cosine -0.968139): HF-fresh/no-SR-resumed PCC is
0.992368 and HF-resumed/no-SR-fresh is 0.980353; matched pairs are only
0.587208 and 0.559811. All six outputs select token 220 with margins
9.44–11.25. No-SR has twelve positions below 0.98, seven TT-only, compared
with six failures for the original TT policy. This does not support accepting
SR removal as a general repair despite its passing 2528 split.

The legal q32/k32 attention diagnostic is prepared in
`tests/datatype_resume_attention_variants.py`. It changes bounded-prefill
attention uniformly, uses bulk fills only on 32-token page boundaries, retains
singletons otherwise, and leaves ordinary decode outside its scope unchanged.
It will test grouped singleton MoE at starts 68, 2528, and 2559 after the current
factorial job closes. Source checks and mocked call contracts passed; hardware
legality and full-model accuracy remain unverified.

## Completed remaining factorial

`resume_variants/remaining_factorial_v1/results.json` completed normally
(1161 seconds launcher elapsed). The final no-SR + grouped singleton point
also fails 2559, PCC 0.987118940. FP32 + no-SR at 2528 passes mean but fails
minimum PCC 0.979680406. Of these eight cases, only no-SR at 2528 passes; the
same policy fails 68. This eliminates all tested boundary/SR combinations as
a general repair. The q32/k32 experiment is now running, with unchanged
production source and ordinary decode.

## q32/k32 interim evidence

The legal q32/k32 bounded attention diagnostic passes start68 at unchanged
thresholds: mean PCC0.994631967, minimum0.980190649, top1=0.992217898.
It fails start2528: mean0.989181713, minimum0.979848194, top1=1. Grouped
singleton MoE is inactive in both cases (counterdelta0). Thus the improvement
at68 is attributable to the changed attention execution/planning policy as a
whole; it does not isolate SDPA accumulation from altered bulk/singleton
projection geometry. It is not a general repair. Start2559 remains running.

## Completed q32/k32 diagnostic

`resume_variants/attention_k32_v1/results.json` completed normally in616s
launcher elapsed. Start2559 also fails (PCC0.986857114, top1=1), so the
K32+grouped-singleton combination is not accepted. Native legality, page-fill
alignment, and unchanged ordinary decode scopes are recorded in
`attention_q32_k32_diagnostic.json`. The composite schema2 patch is retained
UNAPPLIED in `prefill_k32_grouped_singleton_proposal.patch`.

The next paired run changes only grouped stochastic rounding to disabled,
retaining the same K32 planner and grouped-singleton policy at all three starts.
Command: `timeout3600 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_resume_attention_variants --variants no_sr_grouped_singleton --starts 68 2528 2559 --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/resume_variants/attention_k32_no_sr_v1`, wrapped by installed wait_for timeout3720.
No production change or threshold relaxation has occurred.

## K32 with no-SR interim result

The paired no-SR run still fails start68: mean PCC0.992491375, minimum
0.580380595, top1=0.988326848. The two remaining starts continue under the
original bound; no mid-dispatch termination is used. This excludes no-SR as a
general repair with either tested attention planner. Next is the paired existing
BF16 dense policy versus that same policy with all40 grouped expert weights
promoted to BF16. Those tests keep the original public-prefill execution policy
and unchanged installed gate, including negative controls.

## Linear recurrence source audit

The bulk Chunk64StateGraph performs cumulative-decay/triangular-solve/matmul
factorization and one FP32 state update per64 coefficients; public splits change
floating-point grouping. Per-token BF16 recurrence is a different nonlinear
policy and cannot be obtained by adding one cast to the bulk factorization.
The layer0 captured normalization is bit-identical, excluding it as the first
observed divergence at2559. A remaining supported control is the existing
Chunk64OutputGraph at one real token versus singleton recurrence, on identical
normalized operands and cloned states. That graph pads coefficients internally,
not real convolution/cache tokens. A diagnostic is being prepared; no source
change or repair claim follows yet.

## Completed K32/no-SR and next weight control

`resume_variants/attention_k32_no_sr_v1/results.json` completed normally in600s.
Start2559 passes (PCC0.996100514, top1=1). Start2528 passes PCC
(mean0.994261890/min0.987347426) but fails top1 (31/32=0.96875 <0.98).
Start68 fails minimum PCC as recorded above. This remains a rejected general
repair; different failing metrics are not substituted for the required gate.

The first weight control now uses the explicit existing `accuracy_bf16_dense`
config through normal construction and no diagnostic math overrides, at
starts68/2528/2559. The configured diagnostic retains exact baseline token IDs,
all40 layers, and the unchanged comparison thresholds; negative controls are
reserved for the normal full gate if a policy survives. The harness gained an
explicit `--precision-config` and `configured` variant; the original `baseline`
variant still requires bitwise parity and rejects alternate precision configs.
Command and captured actual policy are recorded in
`resume_variants/accuracy_bf16_dense_v1/results.json`.

## 08:58 UTC — dense control and same-input recurrence probe

The explicit normal `accuracy_bf16_dense` policy completed all three resume
splits in 591 s. Start68 fails minimum PCC (0.833366257; mean0.992920894,
top1=0.984435797); start2528 fails mean/minimum PCC (0.988953334/0.977610482,
top1=1); start2559 passes (0.992480746, top1=1). It is not a general repair.
Artifact: `resume_variants/accuracy_bf16_dense_v1/results.json`.

The same-input layer0 singleton/chunk-n1 probe completed normally in15s:
`linear_singleton_localize/L0_s2559_v1/result.json`. The first launch stopped
before TTNN import because the command used the wrong trace-tracking variable;
its assertion log is retained. The retry set `TT_METAL_TRACE_ALLOC_TRACKING=1`
and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; no reset or production edit.
Exact command is in the result and retry log. CPU interpretation is pending.

Paired `accuracy_bf16_dense_ep40` now runs all three splits with unchanged
execution policy and thresholds. It differs only in grouped expert storage,
BFP8 to BF16 across all40 layers. Full-capacity fit is not yet established.
After this control, resume the current-source baseline and coarse matrix;
no baseline diagnostic failure is silently waived for the eventual selection.

## Same-input recurrence interpretation

CPU analysis (`linear_singleton_localize/L0_s2559_v1/analysis.json` and
`analysis.md`) finds K512→K1024 leaves raw recurrence, new recurrent state,
gated values and final attention bit-identical under both boundary policies.
Tiny projection/conv-history differences remain; later effects are untested.
At fixedK1024, chunk-n1 and singleton have exactly equal q/k/v/beta/initial
state and first differ inside recurrence/decay. BF16-boundary final attention
RMS difference is1.65e-7; with FP32 carry final output is bit-identical.
Boundary carry itself changes final attention RMS1.42e-5 (~86x larger), but
neither is established as ground truth and the full-model carry variants
already failed. No isolated K-grouping/engine production repair is supported.

## 09:08 UTC — resume controls concluded; ranking resumes

The normal `accuracy_bf16_dense_ep40` diagnostic completed in603s, without
reset. It fails minimum PCC at68 (mean0.990693280/min0.569636479), minimum at2528
(mean0.990324362/min0.976647091), and mean at2559 (0.985254239). Top1 is0.992217898,
1,1 respectively. Raising grouped expert storage alone is not a repair.
`resume_variants/accuracy_bf16_dense_ep40_v1/results.json` retains the exact
policy, commands, all40-layer evidence and failed tensors. Production math is
unchanged; the eventual selected policy must still undergo the original gate.

The pre-matrix audit found stale-native P1-cache relabeling and source mutation
risks. Fixed before any P1 scores existed: `datatype_p1_runner.py` preserves
installed scoring but binds each newly created score to actual-process native
identity, source/input bytes, text/settings and exact reference-score identity.
Missing/mismatched bindings fail closed. `run_p1.sh` archives the process and
per-score evidence. Candidate measurements now compare Python/source/input
identities before and after, including installed readiness and TTNN Python;
native coverage includes `ttnn/api`. Report noise controls require successful
completion, normal closure and valid reference/native/Python identities; entry
coverage retains row indices. CPU rejection/syntax checks pass.

STOP was removed between jobs. Independent current-source BFP8 processes v3
and v4 now run serially via `run_candidate.sh`, each under3600s timeout and
installed wait_for3720s. Their source snapshot includes the finished P1 wrapper
and identity helper; these files stay frozen through the sweep.

## 10:26 UTC — AutoFix outcome at user-directed cap

AutoFix did not find a general passing resumed-prefill repair in the completed
experiments. No proposed numerical patch was accepted. The same-input controls
and full-model diagnostic matrix narrowed the numerical sensitivity but did
not close all unchanged default PCC/top1 criteria together. This is a failed
repair outcome for the current investigation, not proof that no future repair
is possible. Matched HF controls do not explain every TT-only failure.

The user capped stage time/tokens and instructed remaining acceptance once,
with untried ideas handed off. Further search is therefore deferred. The
retained BFP8 model passes full-model accuracy/P1 and final token-out/nonaligned
checks; the once-only default resumed-prefill acceptance is now active. See
models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/resume_wrapup_assessment.md
for exact failed controls and deferred hypotheses. No waiver or lowered gate
threshold is introduced. This entry supersedes earlier plans to resume a broad
matrix; only final acceptance, review and local checkpoint work remain.

## 10:36 UTC — final default resume gate FAIL, unchanged baseline behavior

Once-only default run completes normally in925s, wrapper exit1. Start32PASS;
start68 minimumPCC0.968224176FAIL; start2528 mean0.989604529/min0.973850864FAIL;
start2559 mean0.986428019FAIL. All four top1 and final-argmax criteria pass;
all four applicable negative controls fail as expected. No thresholds changed.
Artifact final_acceptance/resume_default.failed.json; both native and complete
Python/input identities remain stable, actual runtime policy equals selection.

All fresh/resumed logit tensors for all three failed cases are bitwise equal
to baseline_resume_diagnostics tensors. CPU torch.equal comparison is saved
in final_acceptance/resume_baseline_equivalence.json. Therefore the prior exact
HF per-position comparison applies directly; HF passes2559 and fails different
positions in other cases. This is unchanged baseline numerical sensitivity,
not a new selected-policy regression, but the required PCC gate still fails.
No blanket HF waiver or previous stage6 owner acceptance is extended. The
installed limitations.items records it as unresolved-after-autofix. No further
repair experiment under the user cap. Qualitative and context gates continue
once; the independent review will receive the raw failed gate and this evidence.


## 2026-11-16 owner acceptance and final context acceptance

Owner decision: accept the resumed-prefill PCC miss (start 68 minimum PCC 0.968 vs 0.98; the matched HF BF16 control also fails minimum PCC on the same case, and top-1, final argmax and negative controls pass) as a known limitation, the same as the stage 6 decision. Keep the selected precision configuration, record this in limitations, finish the stage review and complete stage 8. No further resumed-prefill investigation.

Applied to closing stage 8 with the existing selected configuration and recorded resumed-prefill PCC failures. The owner specifically cites start 68; acceptance does not assert that HF reproduces every TT-only failure at starts 68, 2528 or 2559. Raw thresholds, failed results and HF comparisons remain unchanged. No further resumed-prefill investigation is authorized in this stage. See `owner_decision.md` and `owner_acceptance.json`; `doc/eval_readiness/limitations.json` now records the explicit acceptance while retaining raw FAIL and the HF distinctions. Earlier unresolved-status entries below/above are historical. No production or selected precision policy change was made.

The final context artifact passes: logical32769 in65536 allocation, next position32771, two model/sampler replays, zero input refresh/readback/recapture, stable native/Python/input/selected hashes and allocator totals. Free DRAM12,954,371,584bytes/device; advertised262144 context is preserved. The interrupted wrapper did not persist its exit status; the artifact and log show success and normal device closure at10:57:23UTC. The recovered attempt explicitly uses exit_code=null. No hardware job remains or was rerun.
