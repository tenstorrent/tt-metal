# Prefill trace capacity and nonaligned-resume diagnosis

CPU/source review; no device run by this author. Original evidence remains unchanged.

`batch32_hf_v1.json` exhausts DRAM during the first multi-row public prefill call, not necessarily its first trace capture. After prepare, free DRAM is 521,230,528 bytes/bank ×8 =4,169,844,224 bytes/rank. At the failed capture, allocated memory has grown by4,001,431,552 bytes/rank; the next command buffer needs143,278,080 bytes/rank. The old `(count,slot)` cache requires63 full-stack prefill signatures for prefixes203..234:32 full32 keys and31 non32 tails. Roughly143MB each would require about9GB, beyond that measured headroom. Exact per-key sizes were not collected in v1. Its OOM is not evidence that B32 weights/state or context are physically unsupported.

The old decode cache also keys the complete slot tuple, and sampling keys contain the active tuple. Those independent signature spaces remain to be bounded; a prefill-only correction does not establish general serving trace capacity.

The tracing skill requires lifetime retention and at-most-once capture per execution signature. Eviction/recapture is therefore not an accepted repair. The current candidate uses a canonical prefill state/output, device state copies at request-row boundaries, and dyadic chunk sizes1/2/4/8/16/32, retaining the absolute32-token grid. This bounds prefill signatures without changing the decoder's named dtype/fidelity/cache/CCL settings. However, it changes chunk segmentation, invokes the single-row reduction at different token positions, and adds state-copy/tail dispatch work. Numerical and timing validation remain necessary; no acceptance follows from source inspection.

`prefill_signatures_original_v1.json` independently exposes an existing resume failure in the frozen original (`generator.py` SHA256 `8da9d84e637f2e7db1bc210cc1f050a7afd6efdfd95095cbf94d46c1ee4a6233`):

| Case | PCC mean/min | Top1 agreement | Verdict |
|---|---:|---:|---|
| start68 → end257 | 1 /1 | 100% | pass |
| start31 → end265 | 0.9997294294 /0.9967314352 | 94.444444% (221/234) | fail |

The unchanged installed resume requirements are PCC mean≥.99, minimum≥.98 and top1≥.98. The start31 failure is specifically top1, with maximum logit delta0.5341796875; strong PCC does not waive it.

A source-supported mechanism is the shape-dependent routed-expert reduction in `tt/multichip_decoder.py::MeshMoe._reduce`: `n==1` calls `_reduce_single` with the accepted FP32-DEST matmul config, whereas `n>1` calls the native BF16-DEST reducer. A full32-token chunk uses the latter at absolute token31. Resuming at31 uses a one-token chunk to reach the next absolute32 boundary and selects the former. The dyadic candidate also ends a31-token prefix with a singleton at token30, so it cannot be presumed to fix this mechanism. Routing counts/expert packing also depend on logical `n`; the first real divergent producer has not been captured for this exact resume case. This is a hypothesis localized by source, not proof of a kernel defect or permission to widen/change precision.

`tests/probe_prefill_trace_signatures.py` now retains selected logit snapshots even when a gate fails and completes independent comparisons before its final aggregate assertion. Failed thresholds remain failed. It records net allocator growth for each newly captured prefill key, isolated from timing-only runs. The source archive is a diagnostic baseline only, never a production fallback or acceptance implementation.

## Native-single prefill causal control

Parent device-owner run `prefill_native_single_counterfactual_v1.json` now reports `diagnostic_checks_passed:true`, while deliberately retaining `passed:false` and `acceptance_eligible:false`. Both resume cases become exact: start68→257 and start31→265 have top1=1 and maximum absolute delta0. The scoped hook records6 Python warmup/capture overrides,9 unchanged calls outside prefill, and0 decode overrides; these counts are not trace-replay execution counts. Mixed/reordered all-logits and retained-signature checks pass in this control.

This control is strong evidence that the n1-versus-grouped reduction boundary causes the observed reduced-model resume31 discrepancy. It does not justify production selection: selecting native BF16 accumulation for prefill n1 changes the established FP32 accumulation policy. The control stays unselected. The alternative prefill n>1 BF16 batched-matmul scheduling control keeps n1 FP32 and decode unchanged; its result remains separate. A requested unmodified-policy full48 confirmation is also separate from this reduced causal finding.

The harness now supports `--all-layers` only for unchanged-policy correctness, with explicit layer indices and `stage_acceptance_claim:false`. Snapshot v2 embeds the exact layer list; cross-depth or old unlabelled baseline fixtures are rejected before any device/model setup. Old artifacts are preserved.

## BF16 batched prefill scheduling control

Parent device-owner run `prefill_bf16_batched_counterfactual_v1.json` fails the unchanged resume gate. Start68→257 stays exact; start31→265 has top1 agreement225/234=96.153846%, PCC mean0.9997680474482344/minimum0.9964045202905336, and maximum absolute delta0.4677734375. The report preserves `diagnostic_checks_passed:false`, `passed:false`, and `acceptance_eligible:false`. It records30 Python warmup/capture prefill overrides,9 unchanged calls outside prefill, and0 decode overrides. The BF16 n>1 scheduling alternative is rejected for this resume failure; its stronger PCC does not satisfy the98% top1 requirement.

Together these controls support a shape-dependent accumulation explanation while leaving the accepted production arithmetic unchanged. The native-single control closes the reduced failure but changes the accepted n1 accumulation policy; the BF16 batched control preserves n1 policy and does not close it. Neither control is selected. Parent has launched `prefill_signatures_full48_v1` with the unmodified policy, shared KV and matched-reset decode submission-order checks. Its result is pending here; reduced random-head top1 behavior is not substituted for full48 evidence, and no gate is waived.

## Full48 v1 harness setup failure

`prefill_signatures_full48_v1.json` failed after approximately166 seconds during harness pool binding: the initial full48 adaptation still supplied one reduced-model pool spec and asserted one indexed-attention layer. This is a harness defect, not a model or resume result. The original artifact is preserved. The harness now enumerates indexed-attention states from `gen.caches[0].layers`, assigns a separate tensor index to each pool (1 reduced;12 full48), and checks every layer's K/V alias across both physical caches and canonical scratch. It records pool geometry and helper source hash. Page counts derive from context capacity, independently of layer count; a common page table remains valid because each attention layer owns a distinct pool.

The layout arithmetic is extracted unchanged from the existing batch-HF dependency into CPU-only `tests/shared_pool_layout.py`; the old full-batch wrapper retains its original full48/B2..32 validation. CPU tests exercise actual fake-cache enumeration for reduced and full48, unique tensor indices, per-slot page disjointness, scratch last-layer alias rejection, missing-full48-spec rejection, and B1/nonaligned capacity rounding. Three tests pass; affected modules compile. Other prefill/decode helper constants were reviewed: B2 slots are intentional test geometry,32/256 are physical page/head dimensions, and private page counts now derive from the table width. No device run or production change was made by this author.

## Full48 v2 and policy-preserving checkpoint diagnostic

`prefill_signatures_full48_v2.json` completes in269.231 seconds and confirms a real full48 resume31→265 failure: top1=223/234=95.299145%, PCC mean0.9894826780/minimum0.6825643852, maximum absolute delta8.1044921875. Resume68→257 is exact. Mixed/reordered prefill outputs, retained traces, bounded signatures and no late compilation pass. Decode submission-order tokens, logits, feedback and trace IDs are exact; its only unequal request-state hashes are seed-state rows because the old helper seeded only slot0. The helper now sets explicit seed rows `[0,0]` and slots `(0,1)`. A real AST-extracted setter/formatter CPU regression proves scalar seed0 leaves slot1 unset even when merely adding slots; explicit seed rows pass. The numerical resume failure is independent and not excused. Original v2 is unchanged; its source sidecar records current model/decoder hashes with the parent's unchanged-since-run attestation. Future reports hash both files directly.

A separate reduced-only `--prefill-tail-checkpoint-diagnostic` preserves all decoder arithmetic. It snapshots each request's compact state immediately before its final odd-length singleton: GDN convolution/recurrent state, PLE convolution/token history, and QSA raw history/device position. The committed odd prefix still executes the accepted n1 FP32 path. On an immediate matching continuation, it restores the pre-tail checkpoint and replays the old token together with new input using the ordinary n≥2 path, omitting the replayed row from public all-logits output. No host tensor state, KV cache, compressed-index cache or page-table snapshot is used.

Why QSA does not need an index backup here: odd exclusive end E places the final singleton at even absolute position E−1 (mod4=0 or2). QSA writes compressed keys only at mod4=3; otherwise native paged update receives−1 and skips the write. Its K/V row is overwritten by replay before gathering, while selected IDs are causally masked. This reasoning is specific to the final odd tail, not arbitrary singletons. The diagnostic preserves currently bound page tables and restores only matching unchanged mappings/owner/end/token; missing or invalid snapshots retain the original path. Reset/decode invalidate snapshots. Slot migration and externally imported checkpoint provenance are not implemented or claimed by this diagnostic.

`tests/prefill_tail_checkpoint_diagnostic.py` allocates all snapshot buffers before preparation and warms both copy directions. It reports exact allocator growth across allocation plus tensor inventory/helper hash. Source banked arithmetic estimates34,478,592 bytes/rank/request for full48 (about1.103GB at B32); actual measured allocation and context-capacity impact remain pending. The reduced driver separately invalidates slot1's checkpoint and requires the unchanged resume31 failure to persist as a negative control. Three CPU orchestration tests pass (replay slicing/committed prefix, missing/decode/wrong-end negatives, compact inventory), and all seven sampling parameter tests pass. No TT execution by this author; no production selection.

## Checkpoint diagnostic evidence and production integration

Parent-run `prefill_tail_checkpoint_diagnostic_v1.json` completes in about43
seconds with `diagnostic_checks_passed:true`. Both resume68 and resume31 are
bitwise exact. Intentionally missing slot1 checkpoint preserves the original
resume31 failure; fixed two-row seeds make every decode submission-order check
pass. All six prefill trace IDs are retained with no late compilation. The
artifact remains explicitly diagnostic, with no stage acceptance claim.
Measured reduced B2 snapshot allocation is4,163,584 bytes/rank.

The corrected full48 banked snapshot arithmetic is34,576,896 bytes/rank/request:
36×(163,840 GDN convolution +786,432 recurrence),163,840 PLE convolution,512
PLE token history, and12×(16,384 QSA raw history +512 device position).
A raw-history payload has four BF16 tiles but reserves eight tiles across eight
banks, explaining the earlier98,304-byte full48 underestimate. Reduced0/1/3
uses2,081,792 bytes/request, exactly matching the B2 allocation. B32 adds
1,106,460,672 bytes/rank, before runtime overhead; the capacity planning owner
is updating its charge without changing historical measurements.

Production `tt/prefill_checkpoint.py` now implements that compact state. Its
request-owned buffers are preallocated before capture and never referenced by
captured model traces. A fixed-slot permutation moves buffer references and
validity/token metadata together, requiring no additional device scratch.
Reset invalidates only its owned slot; decode invalidates only active slots;
close invalidates all. Dormant requests preserve their checkpoints.

`tt/generator.py` validates all prompt rows, cache ownership, page tables and
sampling parameters before any checkpoint restore. It checks saved end/token
and the still-committed cache position, then binds the current scheduler table,
copies the request into canonical state, and restores only compact state.
It never restores old page ownership. Newly odd tails receive a new snapshot;
public all-logits output excludes the replayed prefix row. No decoder arithmetic,
KV dtype, inter-layer layout or sampling policy changes are part of this repair.

38 CPU tests pass across production checkpoint lifecycle, actual prefill/decode
validation, and shared-pool ownership. They include arbitrary submitted slots,
new odd tails, matching/missing/wrong token/end/position, invalid later rows
without earlier restore mutation, active-only invalidation, request migration,
and a shared-pool copy-on-write fixture that preserves the old physical page,
uses the new scheduler-owned page and rejects overlaps before any snapshot copy.
These CPU fixtures prove orchestration, not TT numerical correctness. Current
production reduced and then full48 gates remain device-owner work. No failed
artifact or threshold was replaced.

The subsequent shared-output change allocates one independent persistent
`prefill_output` for all canonical prefill traces at every batch size. Completed
request rows are written immediately into their physical `sample_logits` slots
with the existing BF16 tiled `write_token_chunk`; inactive canvas rows are filled
with -100 once per device-output submission. The canonical output never aliases
the sampler canvas. Host all-logits reads, resumed-prefix slicing, request state
copies and checkpoint lifecycle are unchanged. Each temporary last-row slice is
released before the next model or sampling replay. Preparation warms the fill
and exact last-row slice/write for all six physical buckets before capture.

Compared with one prefill output per slot, the banked persistent saving is
`(B-1)*3,981,312` bytes/rank: zero at B1 and123,420,672 bytes/rank at B32
(117.703125 MiB). Runtime allocation deltas and performance remain unmeasured.
The prior generator is preserved at
`tests/diagnostic_sources/prefill_workspace_before/generator.py`, SHA256
`f7c458a3fae43f481cca514acb9699efbe3e8991d25ac811f34823ebded16675`;
`prefill_workspace_source_baseline.json` links the parent's already-loaded
full48 checkpoint run to that source. Its hardware result does not validate
this later workspace change.

44 CPU checks pass across actual production prefill/prepare/helper methods,
temporary-slice lifetime (including a deliberately retained-slice negative
control), checkpoint lifecycle, decode validation and shared-pool ownership.
Python compilation also passes for the generator and adapted synthetic seeded
sampling probe. Required device follow-up is tracked reduced mixed/nonaligned
prefill with resumed31/68, arbitrary slot order, inactive rows, sampled and
device-logit returns, six retained trace IDs and no late compilation; then the
same full48 correctness gate. The seeded sampling probe must verify common
sampler reuse after streamed writes. A tracked B32 memory comparison and clean
performance run are needed before claiming the estimated saving or speedup.
