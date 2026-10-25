# AutoFix — BFP8 and BF16-CCL projection geometry

Status: repaired and verified by reduced boundary/replay checks and full64 runs.
BFP8 LoFi/HiFi2 and BF16 CCL now complete at the unchanged native context with
99%/100%/100% agreement. See the final closure and independent follow-up in
`geometry_alternative_review.md`. No device reset or capability reduction.

The following chronological investigation preserves original hypotheses and
intermediate results. Pending statements below describe their historical point.

Contract: legal BFP8 LoFi/HiFi2 full-model candidates must prepare and execute all
supported prefill/decode paths without reducing context or silently changing dtype.
Observed first failure: linear mix static CB endpoint1,405,568 exceeds live L1
frontier1,212,032. BFP4 controls pass. AUTODEBUG.md independently derives the exact
193,536-byte overlap from the actual triple-buffered weight reader and physical
4608-column padding. Source verification confirmed the triple-buffer formula and
explicit DRAM-sharded program config; no allocator/kernel bug is required.

Hypotheses in priority order:
1. Preserve readers1 and reduce mix K20->10; predicted endpoint777,088. Real layer0
   output comparison and traced timing distinguish this from the original failure.
2. Preserve K20 and use readers2; predicted endpoint801,664. Compare with option1
   on the same real weights/recorded activations. Independent source assessment is
   delegated to bfp8_geometry_check; it owns no implementation or hardware.
3. Down readers1->2 preserves legal K17 and physical5120 columns. Linear mix
   prefillK16->8 and separate gate/up prefillK20->10 reduce BFP8 buffers below the
   known BFP4 footprints. Validate affected short/large/one-row-tail and batched
   paths; do not assume fixing the first projection validates siblings.

At the initial BFP8-only hypothesis stage, no source repair had been verified
and no BFP4 geometry change was proposed. Subsequent observed BF16-CCL failures
required the targeted BFP4 geometry adaptations recorded below. Fused-row
geometry, allocator checks, device state and capability limits remain unchanged.

Additional observed failure: BF16 decode CCL with BFP4 weights fails ordinary down
on its second invocation/cache hit: static endpoint811,904, live frontier798,336.
Captured kernel args identify BF16[1,1,1,4352] x BFP4[4352,5120],LoFi. Independent
assessment in geometry_alternative_review.md derives two-reader BFP4 endpoint
495,616, leaving302,720 bytes below the recorded frontier. Preserve both shared
workspace synchronization slots; adapt down readers only. Initial failure JSON/log
is retained in failures/decode_ccl_bf16_initial.*. This is a compile/allocation
failure, not a measured precision-quality rejection.

Source change: multichip_decoder.py resolves geometry after final dtype/activation
overrides and before packing/phase copies. BFP8 attention uses mixK10, linear
prefillK8; BFP8 gate/up prefill usesK10; BFP8 down or explicit BF16 CCL uses two
ordinary down readers withK17. The default BFP4/BFP8-CCL path is unchanged. Explicit
legal mix readers2/K20 remains available to the reduced discriminating check.

First real layer0 projection probe passes repeated exact trace replay. Same-dtype
BFP8 mix agreement betweenK10/readers1 andK20/readers2 is PCC0.999958; down outputs
match exactly. Alternating host trace timings overlap, so keep the lower static
footprint and existing reader placement (K10/readers1) pending full-model checks.
This is reduced real-weight evidence, never a full-model performance result.

`validate_precision_geometry.sh` now checks real layer0/3 short, one-row,
4095/4096/4097 boundaries, batches8/16/32, and repeated BF16 CCL decode. No
full-model BFP8 retry is accepted until these affected-path checks complete.
The failed CCL process imported the pre-geometry source while the file was later
edited; Python traceback display lines consequently reflect newer text. Its
recorded source hash matches sources/pre_geometry/multichip_decoder.py, independently
verified. Diagnosis uses that snapshot and captured actual kernel arguments.

Reduced repair validation completed: both real layer types pass lengths1,33,
4095,4096,4097 and batches8/16/32; both pass repeated BF16 CCL checks. All use
TT_METAL_TRACE_ALLOC_TRACKING=1 and include program-cache allocations. Full-model
retries remain necessary. Exact commands: validate_precision_geometry.sh, then
resume `LAYERS=3 SKIP_MICRO=1 SEQUENCES='4095 4096 4097'` after the control fix below.

A diagnostic mismatch was investigated rather than waived: layer3 seq33 compared
new BFP8 weights to the runner's default BFP4 single-chip weights. Outputs were
PCC>=.99977, but K/V cache states were .992429/.994390. Preserved under
failures/geometry_l3_s33_vs_bfp4.*. Added --baseline-policy to the existing runner,
leaving its default and .995 threshold unchanged. With BFP8 weights on both sides,
K/V PCC=.999486/.999429, output PCC=.999955 and decode PCC>=.99998. This controls
the geometry path while the upcoming full-model AIME gate judges precision quality.

Full64 BFP8/LoFi retry now passes native-size preparation, prefill and traced
teacher forcing at top1=.99,top5=1,top100=1. Median traced decode27.010588t/s/u,
TTFT119.301873ms. This closes the observed BFP8 L1 crash with a verified repair;
the candidate is slower than the BFP4 control and is rejected on measured speed.
The original failure remains separately preserved. BFP8/HiFi2 and full64 BF16
CCL validation are still running; no complete AutoFix closure is claimed yet.

Full64 BF16 CCL after down repair reaches a later mix, then fails with852,608
static endpoint against833,152 live frontier. A four-real-layer generator with
full head/request owner and8192-token pool reproduces these exact numbers
(failures/ccl_four_layer_before.json). Reducing its ordinary mix K20->10 fixes
that observed decode deficit; predicted endpoint500,608. Same dtype/reader/layout
is retained. The following probe then reaches large prefill and exposes a separate
measured linear-mix endpoint1,195,136 against1,140,096 live frontier
(ccl_four_layer_after_mix.json). This confirms a phase-local prefill deficit;
it is not inferred by reusing the decode frontier.

Adapted only the failing linear-attention prefill mix K16->8 for BF16 CCL. Its
source-derived endpoint is839,808. Gate/up prefill remains unchanged because no
failure there is observed; the independent assessment identified it as only a
precautionary possibility. Final four-layer/real-owner replay and full64 checks
are pending. Source snapshots for both prior failures are under sources/.

The final four-layer/full-head/full-request-owner probe passes all prepared
prefill buckets and repeated lengths1,33,4095,4096,4097 with exact output tokens
and unchanged three trace IDs/capture count. Evidence:
`ccl_four_layer_repaired.json`. Full64 BF16 CCL retry is now running; this reduced
probe is runtime evidence only and supplies no sweep accuracy/performance point.

Full64 BF16 CCL final retry passes all preparation, prefill and three traced
teacher-forcing repeats: top1=.99, top5=1, top100=1, median34.1809491t/s/u,
TTFT120.675606ms. Evidence `results/decode_ccl_bf16.json` includes actual runtime
kernel arguments and source hashes. This closes both observed ordinary-decode
and phase-local prefill CCL allocation failures. BFP8/HiFi2 also completes at
26.5816364t/s/u and the same accuracy. AutoFix outcome for the observed BFP8/CCL
allocation blockers: repaired and verified; no reset or capability reduction.

A separate later BFP4-KV reset performance defect is tracked in
`kv_reset_autofix.md` and `kv_reset_review.md`. It is independent of the repaired
geometry failures above. Primitive and real-layer reset/replay checks pass;
full-model retry is recorded there when complete.
