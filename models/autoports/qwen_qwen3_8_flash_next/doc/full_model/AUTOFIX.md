# Full-model repair evidence (in progress)

## Resumed prefill

Contract: retained-state prefill must match an uninterrupted prefill at every
tested split, including 68, under the installed runner's unchanged PCC and
top-1 agreement floors. The full48 run passes top-1 agreement but fails minimum
PCC at split68. Splits32/256 pass. See `resume_AUTODEBUG.md` for source diagnosis.

Reduced real layers0/1/3, capacity384, reproduce a failure with different
sensitivity: `reduced_resume_baseline.json` has top-1 agreement0.92248 at68,
PCCmin0.99625. The absolute32-boundary chunk planner improves agreement to
0.94574 and PCCmin0.9966 (`reduced_resume_aligned.json`) but does not pass0.98.
This refutes chunk alignment alone as a sufficient repair. Precision, cache,
CCL and decoder policies have not changed. `resume_chunk_split_v1.json` compares identical-state/input32 versus4+28
chunks at64. GDN recurrence and local QSA state/output match exactly; the
first substantial divergence is MoE output, including layer1 where every
pre-MoE input and routing weight/ID matches. Fixed-residual terminal logits
are exact. The propagated32-token terminal comparison has top1=.96875 and
PCCmin=.996637. This localizes the next investigation to expert output,
routed reduction and shared projection; v2 instruments those boundaries.
Native fused MoE internal BFP8 intermediates are part of the accepted policy,
not an authorized precision increase. No further full-stack run is justified yet.

## Seeded split-sampling first capture

`seeded_sampling_device_v1.json/log` fails on the first draw with
`Cannot load new binaries during trace capture` from GenericOpDeviceOperation.
`DeviceSeedState.prepare` used `prepare_generic_op`, which compiles/finalizes
without dispatch (`ttnn/api/ttnn/device_operation.hpp:650`). The earlier seed-only
probe included an eager draw before capture, so it did not test this boundary.

Candidate repair: enqueue both seed programs once during initial setup with
zero update/active masks, thereby loading binaries without changing request
state. `tt/device_seed.py` implements this. The generator now closes and releases
a failed capture before propagating its exception; it never replays a partial
capture. Validation: syntax passes; twelve seed and slot-state CPU tests pass.
Device verification PASSES in `seeded_sampling_device_v2.json/log` (9s),
including first capture, reset/reorder/migration/pause and sampled prefill.
Live failure evidence is preserved in `seeded_sampling_triage/`. The original
bounded job timed out124; subsequent all4-rank mesh/add/read/close smoke passed
without reset (`recovery3_mesh_smoke.log`).

The seed setup repair is verified. Resumed prefill remains unresolved and the
stage remains incomplete.

### Expert rounding-history discriminator

`resume_chunk_split_v2.json/log` completes in154s. Isolated bit-identical
MoE inputs for0/1/3 produce exact router logits, IDs, scores and shared outputs,
but differing owned expert lanes (max .015625/.0078125/.005859375 respectively).
The fused kernel enables FPU stochastic rounding and seeds once per launch;
active-expert sets alter the arithmetic preceding a token. LLK confirms FPU
rounding at every accumulation and a single PRNG seed register, without a
per-row counter API. Candidate v3 reseeds with the same seed at each active
expert M-block, retaining all dtype/fidelity and stochastic-rounding choices.
This is a discriminator, not a verified repair: hardware row relocation may
still change draws. `resume_chunk_split_reseed_v3.json/log` tests it. The source
change is in native `moe_fused_swiglu_compute.cpp` and must be reverted or
completed if it does not establish the intended invariant.

V3 completes160s: reseeding makes the first4 isolated expert rows exact in
all three layers, but suffix relocation still changes21/23/28 rows. Propagated
terminal top1=.9375, so reseeding alone is rejected; its patch is retained as
`moe_reseed_v3.patch`, not selected production behavior. This proves a rounding
history dependency but does not prove hardware PRNG advance timing.

V4 tests native deterministic FPU rounding (documented RNE in Blackhole
`llk_defs.h:152`) with repeated-identical-token controls. BFP4/BFP8, LoFi,
BF16 DEST, grouped dispatch and all cache/CCL policies remain unchanged. The
old kernel comment warns of2–5% accumulation bias, but source history places
it in import commitd7e788cd without a located measurement artifact. No bias
correction or higher-precision workaround is justified. V4 must remove the
row dependency and pass fresh HF accuracy before retention; source inspection
alone is not a quality claim. Exact candidate: `moe_deterministic_v4.patch`.

### Diagnosis addendum: completed rounding controls

CPU inspection of the completed JSON artifacts confirms the v3/v4 causal
discriminator. In `resume_chunk_split_reseed_v3.json`, the first four owned
expert-output rows are exact in layers0/1/3; relocation of the suffix still
changes21/23/28 rows, respectively. Its propagated terminal top-1 agreement
is0.9375. Per-block reseeding is insufficient and is rejected.

In `resume_chunk_split_deterministic_v4.json`, bit-identical mixed inputs give
exact owned expert lanes, routed reductions and final MoE outputs for all32
rows in all three layers across32 versus4+28. The repeated-identical-token
controls also have exact expert outputs across all row positions, across
repeated launches, and across the split. Thus disabling only FPU stochastic
rounding eliminates the observed expert row/history dependency; this is not
evidence for a dispatch, cache-offset or terminal-head repair. The candidate
kernel SHA256 is
`023239cd82cd46bafd3c0efef0388176a3314ecf8e5671a8bab133d873280979`.

Propagated terminal top-1 agreement improves to1.0, with minimum logits
PCC0.998517514351999. Remaining mixer shape sensitivity means this is not a
claim of whole-stack bitwise equivalence. Commands and bounded eager probe
scope are in `tests/diagnose_resume_chunk_split.py`; v4 enables
`--repeat-token-rows`. Exact outputs are the corresponding v3/v4 JSON and
`.log` artifacts. These are reduced-stack producer diagnostics, not HF
accuracy acceptance. Deterministic FPU rounding remains a candidate pending
the fresh HF and standard resumed-prefill gates; it must not be retained on
this numerical-invariance evidence alone.

### Native residual RMS row association (2026-11-14 02:42 UTC)

`resume_chunk_split_mixer_v5.json/log` localizes the first remaining layer0
mismatch to `mlp_mix.norm`: inputs exact, BF16 outputs differ by0.0078125.
`rms_rows_v1.json/log` captures the actual layer0 positions64..95 input once
in `rms_layer0_fixture.pt`, then taps unchanged native RMS. FP32 inputs and
squares are exact; mean first differs by5.82e-11, eventually crossing one BF16
rounding boundary at row18. No cache or router executes in the fixture replay.

Source: Blackhole `ckernel_sfpu_reduce.h` combines A/B row groups using the
recorded right-associated add chain, C/D using balanced additions. The merged
butterfly also selects group results at different columns without aligning
the original operands. FP32 associativity thus depends on row placement.
Candidate v2 fixes only the chain: hardware still fails identically. Candidate
v3 also pre-rotates B/C/D by1/2/3 so all selected columns represent the same
addition order. `rms_rows_v3.json/log` completes10s: every intermediate and final
BF16 result is bitwise equal for32 versus4+28, and repeated runs are exact.
The change preserves FP32 arithmetic, existing formats and merged reduction;
integer and min/max branches are unchanged. General row-permutation regression,
reduced resume and final all-layer accuracy remain required.

`reduced_resume_rms_v5.json/log` now PASSES (229s,208.8s setup), real0/1/3,
capacity384, scoped native RNE plus corrected RMS reduction, allocation tracker
on including program cache. Installed resume starts32/68/256 all have top1=1
and PCCmean/min reported1.0000; missing-history controls reject as required.
The exact four-step reduced HF smoke reports top1/top5/top100 all1. Native
build and both tar/tt_pybinds installations succeeded; logs are `native_*_v1.log`.
This closes the reduced failure and justifies a fresh all48 accuracy/resume run.

Final all-layer numerical verification: `full_accuracy_v2.json/log` PASSES775s,
all48 layers, advertised262144 allocation, trace/program-cache tracking enabled.
AIME24chat100 prefill and traced teacher forcing each97/100/100% top1/5/100.
Installed `readiness/prefill_resume_check.json` passes32/68/256, top1agreement1,
PCCmean/min1.0000, with both missing-history controls rejecting. Scoped native
RNE plus row-invariant FP32 reduction therefore fixes the resumed-prefill
contract without relaxing a gate or changing dtype/fidelity/cache policy.
Performance and remaining serving/max-context/qualitative acceptance are still
separate work. Rejected reseeding and partial reduction fixes remain recorded.

## Shared-pool migrated decode discrepancy (new gate)

`shared_pool_generator_v4.json/log` exits1 cleanly after202s. First pooledB2
prefill/decode and every logical state match privateB2 exactly. First B2vsB1
decode meets frozen accuracy (minPCC.998263,top1/5/100=1). After remapping and
forced token43, pooled and privateB2 remain exact, but comparison to uninterrupted
solo gives minPCC.970202,top1/top5=.5,top100=1. The private control also remaps,
so this does NOT yet distinguish common migration damage from accumulated
B1/B2 numerical divergence. Next discriminator: uninterruptedB2 against remapped
B2 with identical token/position history. Preserve the failed frozen gate;
finish lifecycle diagnostics before its final accumulated assertion. No source
repair or threshold reduction justified yet. Ordinary serving gate runs in
parallel with CPU-only diagnosis, serialized on the device lane.

### Arithmetic discrimination, 2026-11-14

`shared_pool_generator_v5` proves uninterrupted/migrated B2 and pooled/private
lifecycle outputs and state exact. `moe_decode_capture_v1` locates the first
real divergence at layer0 routed reduction, with identical inputs, IDs, scores,
owned expert values and shared branch. `batch_reduction_AUTODEBUG.md` explains
all61440 saved native output bit patterns with exact ISA arithmetic. No native
indexing or cache defect was found; B1 uses FP32 DEST and batched native uses
BF16 DEST plus serial BF16 pack accumulation, as inherited from stage5.

The isolated BF16 batched-matmul candidate passes B2/B32 duplicates, swaps,
repeats and all20 tracked traces (`moe_reduction_candidate_bf16_tracked_v1`).
Clean timing (`moe_reduction_candidate_bf16_v1`) is about3.85x faster atB2,
but only0.62–0.75x native speed atB32. It is therefore not selected. The reduced
model counterfactual improves token43 whole PCC from0.97965 to0.99012 but
still fails the unchanged frozen comparison (minimum rowPCC0.97881,
top1agreement0.5; top5/top100agreement1). Survivor44 now passes. Artifacts:
`batch_reduction_counterfactual_bf16_v1.json/log`, clean process exit1 in50s.

The precision-changing FP32 diagnostic makes all three solo/batch/migration/
survivor outputs bitwise exact and passes the underlying diagnostic checks
(`batch_reduction_counterfactual_fp32_v1.json/log`,46s). Its public artifact
deliberately remains `passed:false, acceptance_eligible:false`. This confirms
causality but does not authorize widening the accepted batched precision.
Neither counterfactual changes production reduction or weakens a threshold.
Next work must reconcile the inherited arithmetic with independently measured
HF-referenced batched accuracy and the original preservation contract; the
failed cross-geometry diagnostic remains visible and unresolved.


### Matched original-model control and current sampler, 2026-11-14

The previous paragraph records the investigation state before the HF controls.
`shared_pool_hf_control_v2.json` and the exact tie-policy rescore establish that
original HF also fails the frozen cross-width PCC contract at43 and changes its
unique survivor44 greedy maximum. `current_geometry_sampler_v1.json` exercises
the selected production sampler, not historical ranked-ID inference: all62
actual samples (31 history/position rows repeated twice) are TT maxima and lie
in their matched HF top5/top100. Each prefill/decode family passes the unchanged
HF bar. Current43 still fails PCC (whole.990041606/min.988163811), with top1/5/100
agreement1; current survivor44 passes the frozen comparison. Both current and
historical raw failures remain unchanged.

`batch_geometry_disposition.md` records the owner's controlled-limitation
decision under the user's matched-HF instruction, pending independent review.
It preserves the inherited native reduction, exact same-history relocation and
state requirements, and full-model HF bars. No precision fallback or threshold
change is selected. Earlier B1/B2 wording means one versus two active rows in
configuredB2, not two separately configured TT generators. The controls do not
claim identical TT/HF discrepancy magnitude or a universal backend property.

The separate full48 B32v1 final-state failure was diagnosed as a probe seed
contract error: common scalar seed0 seeds only the first request; remaining
rows receive entropy. Explicit per-request seeds now pass the reduced real0/1/3
B32 diagnostic (`batch32_seed_state_reduced_v1.json`) with empty per-field final
and dormant mismatch dictionaries and exact logits/tokens across migration.
The subsequent full48 B32v2 rerun passed in1296.826094s with clean exit0
(`batch32_hf_factored_v2.json/log`). Both2704-prediction histories pass all
per-request and per-phase HF gates; final and dormant state mismatch maps
are empty, and matching-history logits/tokens/state are exact. This harness
fix does not change production seeding or retrospectively pass B32v1.
