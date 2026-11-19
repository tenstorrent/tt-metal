# Optimization decisions and evidence

This is a CPU extraction of the completed candidate artifacts, not the final
acceptance verdict. Paths below are relative to this directory. Every table
number comes from that JSON's `multichip.prefill_warmed_ms`,
`multichip.decode_traced_warmed_ms`, `prefill_pcc` and minimum `decode_pcc`.
Decode is the whole-batch warmed blocking trace median, prefill is whole batch;
all are serial 1x4 mesh measurements with real checkpoint weights. B32 uses
natural activations from 32 distinct prompts; B1 is a separate regression row.
Exact commands, source identities, geometry and environment are in matching
`.json.provenance.json` sidecars. Screens are cumulative, not an additive
speedup decomposition. Final README numbers must come from `final_*.json`.

Selected defaults enable batched attention reduction, batched direct AR,
batched full-attention input/output projections, packed linear ordered input
projection, and one untilize for batched row extraction. They keep replicated
BF16 residuals, BF16 KV/CCL, BFP8/HiFi2 ordinary full/shared projections,
BFP8/LoFi active experts, and sensitive FP32 router/recurrent arithmetic.
No dense all-expert execution is selected.

## Cumulative topology screens

| Artifact | B | Prefill ms | Decode ms | Prefill PCC | Min decode PCC | Decision |
|---|---:|---:|---:|---:|---:|---|
| [before_l0_b32](before_l0_b32.json) | 32 | 1329.492118 | 21.724882 | 0.999966521 | 0.999902546 | Linear starting control |
| [before_l3_b32](before_l3_b32.json) | 32 | 42.404445 | 6.444862 | 0.999969649 | 0.999854728 | Full starting control |
| [batch_ar_l0_v3](batch_ar_l0_v3.json) | 32 | 1311.888093 | 21.111493 | 0.999975638 | 0.999904061 | Keep batched direct AR |
| [batch_ar_l3_v3](batch_ar_l3_v3.json) | 32 | 45.397688 | 5.962403 | 0.999970131 | 0.999854346 | Keep batched direct AR |
| [batch_proj_l3](batch_proj_l3.json) | 32 | 52.280331 | 5.077451 | 0.999970123 | 0.999854446 | Reject applying decode geometry to prefill |
| [batch_proj_decodeonly_l3](batch_proj_decodeonly_l3.json) | 32 | 43.974094 | 5.073939 | 0.999969649 | 0.999854346 | Keep phase-specific batched projections |
| [packed_linear_l0](packed_linear_l0.json) | 32 | 1158.337290 | 20.590145 | 0.999966521 | 0.999904061 | Keep combined ordered projection |
| [packed_linear_l0_b1](packed_linear_l0_b1.json) | 1 | 52.383858 | 0.999702 | 0.999988819 | 0.999920574 | B1 regression passes |
| [split_once_l0](split_once_l0.json) | 32 | 1155.524781 | 20.466604 | 0.999966521 | 0.999904061 | Keep one untilize before row extraction |
| [split_once_l3](split_once_l3.json) | 32 | 43.718434 | 4.937193 | 0.999969649 | 0.999854346 | Keep one untilize before row extraction |

## Residual/collective/packing families

| Artifact | B | Prefill ms | Decode ms | Prefill PCC | Min decode PCC | Decision |
|---|---:|---:|---:|---:|---:|---|
| [batch_rs_ag_l3](batch_rs_ag_l3.json) | 32 | 43.396538 | 5.015001 | 0.999969649 | 0.999854026 | Reject versus batched direct AR |
| [batch_sharded_l3](batch_sharded_l3.json) | 32 | 44.547269 | 5.831879 | 0.999969649 | 0.999854026 | Reject slower hidden-sharded family |
| [batch_sharded_l0](batch_sharded_l0.json) | 32 | 1156.059481 | 21.585338 | 0.999966521 | 0.999916153 | Reject slower hidden-sharded family |
| [sharded_bfp8_l0](sharded_bfp8_l0.json) | 32 | 1157.464731 | 21.557304 | 0.999936147 | 0.999893917 | Reject slower sharded + BFP8 CCL family |
| [autofix_fused_rs_l3](autofix_fused_rs_l3.json) | 32 | 44.382263 | 5.892128 | 0.999958385 | 0.999850005 | Reject correct repaired fused-output family |
| [repaired_fused_bfp8_l3](repaired_fused_bfp8_l3.json) | 32 | 44.346004 | 5.886388 | 0.999910077 | 0.999764052 | Reject slower fused-output + LoFi/BFP8 CCL |
| [ccl_bfp8_l3](ccl_bfp8_l3.json) | 32 | 42.251538 | 4.956955 | 0.999917806 | 0.999860690 | Reject no decode benefit |
| [split_attn_l3](split_attn_l3.json) | 32 | 44.670344 | 5.024359 | 0.999969649 | 0.999855158 | Reject split Q/K/V/gate |
| [shared_dram_l3](shared_dram_l3.json) | 32 | 42.988619 | 4.946085 | 0.999969649 | 0.999854346 | Reject shared B32 DRAM-sharded projection |

## Precision and geometry screens

| Artifact | B | Prefill ms | Decode ms | Prefill PCC | Min decode PCC | Decision |
|---|---:|---:|---:|---:|---:|---|
| [attention_bfp4_l3](attention_bfp4_l3.json) | 32 | 42.918244 | 5.039884 | 0.994361349 | 0.992968841 | Reject PCC below 0.995 |
| [attention_lofi_l3](attention_lofi_l3.json) | 32 | 42.843701 | 5.059717 | 0.999926020 | 0.999779970 | Pass; later integrated comparison |
| [lofi_split_l3](lofi_split_l3.json) | 32 | 43.839025 | 4.932354 | 0.999926020 | 0.999779970 | Pass; later repeated comparison |
| [lofi_split_l3_b1](lofi_split_l3_b1.json) | 1 | 2.106078 | 0.391804 | 0.999617262 | 0.999278446 | B1 LoFi passes |
| [lofi_integrated_l3](lofi_integrated_l3.json) | 32 | 43.335863 | 4.927228 | 0.999923771 | 0.999779904 | Attention/shared LoFi passes; no phase win |
| [lofi_r1_l3](lofi_r1_l3.json) | 32 | 42.575728 | 5.003930 | 0.999926020 | 0.999779970 | Slower LoFi reader1 |
| [lofi_r2_l3](lofi_r2_l3.json) | 32 | 44.567999 | 4.988035 | 0.999926020 | 0.999779970 | Slower LoFi reader2 |
| [lofi_k8_l3](lofi_k8_l3.json) | 32 | 42.831247 | 4.983045 | 0.999926020 | 0.999779970 | Slower LoFi K-block8 |
| [lofi_k16_l3](lofi_k16_l3.json) | 32 | 43.625365 | 4.979709 | 0.999926020 | 0.999779970 | Slower LoFi K-block16 |
| [lofi_k32_l3](lofi_k32_l3.json) | 32 | 44.696654 | 4.992063 | 0.999926020 | 0.999779970 | Slower LoFi K-block32 |
| [kv_bfp8_l3](kv_bfp8_l3.json) | 32 | 43.811862 | 4.927715 | 0.999957104 | 0.999851237 | Pass; no material decode gain, prefill slower |
| [kv_lofi_l3](kv_lofi_l3.json) | 32 | 44.179815 | 4.979534 | 0.999929831 | 0.999775063 | Pass; integrated KV+LoFi slower |
| [kv_lofi_l3_b1](kv_lofi_l3_b1.json) | 1 | 2.330708 | 0.394560 | 0.999617366 | 0.999362791 | B1 KV+LoFi passes |
| [lofi_repeat_l3](lofi_repeat_l3.json) | 32 | 43.853983 | 4.927429 | 0.999926020 | 0.999779970 | Pass; 0.23% decode gain, 3.27% prefill loss |
| [hifi2_repeat_l3](hifi2_repeat_l3.json) | 32 | 42.464985 | 4.938860 | 0.999969649 | 0.999854346 | Keep HiFi2/BF16 control |

## Coherent topology findings

Batched direct AR replaces 32 independently tile-padded attention collectives
with one M32 carrier while preserving each request's state and positions. The
same persistent scratch serves grouped-MoE output. For BF16 hidden2048 a full
M32 carrier is 128 KiB/rank; nominal four-rank AR traffic is 192 KiB, versus
6 MiB for 32 singleton carriers. FP32 linear partials double those estimates.
They are analytical payload estimates, not fabric counters. FP32 linear partials
remain FP32 until reduction even under the experimental BFP8 CCL knob.

The initial direct-AR attempt failed the following norm's output memory
contract; the next failed arbitrary-row slicing from L1 width shards.
`logs/batch_ar_l3{,_v2,_v3}.log` retains both errors and the adapted passing
consumer. The family was measured after adapting next norm and output layout.
The prefill regression in `batch_proj_l3` was repaired by applying the larger
batch geometry only during decode (`batch_proj_decodeonly_l3`).

The sharded candidates keep hidden512 across the residual add and following
norm; they do not immediately reconstruct the old replicated interface just
for timing. `batch_sharded_l{0,3}` and `sharded_bfp8_l0` therefore compare actual
compatible lower-movement families. The replicated alternative is faster after
including its next consumer and output extraction. There are no inter-layer
CCLs in the selected contract: both kinds produce the same replicated BF16
hidden2048 residual expected by the next layer. B32 retains a tuple of singleton
tensors at the public boundary; conversion is now performed once before row
extraction, rather than 31 repeated untilizes.

Combined distributed norm + fused AG-MM + fused MM-RS ran but failed exact
paired K-cache equality in two scalar cells. This is not waived by passing
whole-output PCC. `AUTODEBUG.md`, `AUTOFIX.md`,
`combined_fused_l3.json.provenance.json`, `combined_fused_l3.pt` and
`logs/combined_fused_l3.log` preserve diagnosis and failure. The adapted control
retains hidden-sharded residual, distributed norm and fused output MM-RS, but
uses canonical separate input projection. It passes exact cache checks in
`autofix_fused_rs_l3.json`; at 5.892128 ms it loses to replicated direct AR.
The compatible reduced-precision fused-output family also passes at 5.886388 ms.
Canonical-order fused AG-MM remains an unimplemented kernel improvement, not a
claim that all fused paths are inherently slower or impossible.

Direct-AR scratch, semaphore and EP output buffers remain preallocated in the
shared serial `CollectiveWorkspace`; their allocations outlive all live traces.
BF16+FP32 direct-AR scratch totals 1.5 MiB/chip. No per-layer persistent copy
was added for batching. B32 grouped experts retain 64 resident experts/rank,
count-controlled active execution, local dispatch and top-8 weighted combine.
B1 retains indexed sparse TP4 GU block64/down block4. Final profile attribution
must separately report B1 and B32 MoE kernel-work fractions.

## Precision, geometry and rejection limits

LoFi is not rejected for synthetic PCC: natural B32 and B1 controls pass.
The repeated integrated attention comparison gives 4.927429 versus 4.938860 ms
(0.23% decode reduction), with prefill 43.853983 versus 42.464985 ms (3.27%
regression). HiFi2 is retained for the measured two-phase tradeoff. The joint
attention/shared LoFi screen gives 4.927228 ms but also higher prefill than the
repeated HiFi2 control. The KV BFP8 screen passes, with 0.23% decode improvement
and 3.17% prefill regression versus that same control; KV remains BF16, so no
new maximum-context claim is made. BFP4/LoFi attention fails the unchanged real
whole-output 0.995 threshold and is never a shipping default from this stage.

The LoFi projection sweep covers readers1/2/3, K-block4/8/16/32, and adapted
K-block64. QKVG has K64 tiles; output has K32, so the larger-block experiment
uses QKVG64/output32. K64 fails with physical L1 allocation1923008 B exceeding
1572864 B, both with input8/output2 and adapted input16/output1. Exact attempts
are `lofi_k64{,_n1}_l3.json.provenance.json` and matching logs; no result JSON
exists because execution failed. This rejects those two physical layouts, not
every possible K64 kernel schedule. Prior-stage `../multichip_decoder/geometry/search.json`
qualifies HiFi2 singleton reader/block/input/output controls, and
`../multichip_decoder/sparse_geometry/search.json` qualifies B1 sparse layouts.
LoFi B32 geometry does not by itself reject equivalent HiFi2 B32 geometry.

## Stop rule and untried handoffs

The last two retained topology changes have less than 3% B32 decode benefit:
packed linear input gives 21.111493 ->20.590145 ms (2.47%); single-untilize row
extraction gives 20.590145 ->20.466604 ms (0.60%) for linear and
5.073939 ->4.937193 ms (2.69%) for full. This satisfies optimize Stop Rule2.
It bounds further optimization effort only; all correctness, watcher, trace,
context and FUNCTIONAL requirements remain required. The owner work log must
retain these untried items under `Not tried (stop rule)`:

- A public batched residual interface carried across stacked mixed layers,
  eliminating row extraction entirely. The earlier profile's total row split
  was about200 us; the new remaining cost must come from the final profile.
- Canonical-accumulation fused AG-MM or pair-owner KV broadcast restoring exact
  duplicated KV equality; no speedup is claimed for this unimplemented repair.
- Shared packed-versus-separate local GU/scalar with fused AG and gathered-input
  reuse for router/experts; the grouped path's small share limits whole-step gain.
- HiFi2 B32 per-role geometry cross-products and residual-grid variants beyond
  inherited singleton controls; current LoFi results are not a precision-locked
  rejection. More explicit 2D prefill/chunk-boundary geometry sweeps remain.
- Grouped-expert core-grid tuning, fused combine+score reduction, and persistent
  dispatch/combine output alternatives. Active-expert execution must remain.
- Further BFP4 expert/shared candidates beyond inherited screens; final datatype
  selection belongs to stage8 and cannot change this stage's BFP8 default.
- BFP8 normalized activations throughout a compatible projection family, beyond
  measured BFP8 CCL and KV candidates; keep FP32-sensitive exceptions explicit.

LM head, sampling, token feedback, TTFT to generated text and qualitative text
checks are not implementable in this decoder-only stage and are reserved for
their full-model/serving stages. Decoder prefill is measured, not relabeled TTFT.

## FUNCTIONAL bar: inherited evidence and applicability

`../optimized_decoder/functional_{0,3}.json` contains actual timed 128-token
prefill /128 changing decode steps at B1, unlike fixed-position replay timing.
`tests/summarize_optimized_perf.py::byte_model,functional_sequence` computes
single-chip optimistic logical DRAM floors at512 GB/s, excluding padding,
rereads and intermediates. Its mean floor/measured ratios are0.203109/1.346337
=15.09% for linear and0.120238/0.428768=28.04% for full. These validate that
single-chip stage; neither the timing nor its roofline is a mesh measurement.

The new `functional_l{0,3}.json` artifacts now provide actual mesh128/128
sequences, three repeats, changing inputs/positions, eager/trace equality and
between-sequence state reset. The per-rank logical traffic model includes
sharded expert/attention weights, replicated router/norm weights, duplicated
local KV heads and local recurrent state; bandwidth remains512 GB/s/rank.
It does not inflate useful bytes with padded intermediates or CCL traffic.
Both performance floors currently fail and require remediation:

| Layer | Mean timed step ms | Logical traffic floor ms | Floor / measured | Required maximum ms at10% |
|---|---:|---:|---:|---:|
| linear | 1.007535 | 0.053876813 | 5.3474% | 0.538768 |
| full | 0.397953 | 0.034443250 | 8.6551% | 0.344433 |

These are failed gates, not stop-rule handoffs. AutoDebug/AutoFix work is owned
by the parent; final completion cannot use the inherited single-chip percentages.

## CPU diagnosis for the FUNCTIONAL remediation

Raw prior-stage B1 `../multichip_decoder/tracy/layer_0/ops.csv`, rank0,
16 replays, has these largest mean individual rows. Indices are zero-based
within a177-op replay. Inline kernels are identified as `Inline source` by CSV;
helper mapping uses operand shapes and forward order in current source, rather
than claiming names are encoded in the CSV.

| Index | Mean us | Cores | Source helper / operation |
|---:|---:|---:|---|
| 32 | 57.892 | 110 | DirectCoefficientStreamingScan, K/V/a/b, `[4,1,512,1600]` |
| 35 | 55.721 | 110 | DirectCoefficientStreamingScan, Q/Z, `[4,1,512,1536]` |
| 99 | 53.477 | 110 | DirectCoefficientStreamingScan, linear output, `[2,1,512,2048]` |
| 145 | 43.512 | 1 | Native TopK,256 logits, internal k32/output k8 |
| 98 | 38.500 | 8 | TP4LocalSiluMultiply |
| 151 | 36.194 | 8 | Indexed sparse expert GU, block64 |
| 44 | 26.824 | 64 | TP4DecodeConvolution |
| 142 | 24.794 | 32 | Ordered router coefficient scan |
| 104 | 17.546 | 34 | FP32 attention all-reduce |
| 156 | 11.629 | 64 | Indexed sparse expert down, block4 |

The two input projection rows have already been packed in the current default;
these historical times cannot be claimed as remaining reducible work. Fresh
packed profiles belong to `functional_debug/tracy/layer_{0,3}`. Prior-stage
input norm rows0:32 cost86.646 us, post-attention norm rows107:142 cost94.764 us,
and recurrent update rows65:89 cost106.808 us. Removing all of those288 us
would still not close the linear469 us deficit. Kernel sums omit dispatch gaps
and may overlap, so this is prioritization evidence rather than an elapsed-time
prediction or proof that the target is impossible.

Smallest source-supported discriminating probes:

1. Pad FP32 router logits256->1024 with negative infinity before native TopK.
   `ttnn/.../reduction/topk/device/topk_constants.hpp` enables low-height
   multicore selection only at width1024; existing width256 runs on one core
   (raw CSV confirms43.47 us full and43.51 us linear). Keep padded IDs impossible
   to select and compare real logits' selected IDs/scores. This needs device
   measurement; more padded work can lose. The generalized MoE gate cannot be
   substituted unchanged because its validator requires BF16 input, while this
   decoder preserves FP32 router ranking. GPT-OSS fused router is128-expert
   and excludes Blackhole, also not a direct replacement.
2. Full B1 input norm costs24.713 us in DRAM interleaved versus the later
   sharded residual norm's about6.11 us. Feed the first norm the already-defined
   L1 width-sharded residual memory and retain it through its next consumer.
   Verify paired KV identity and whole-output PCC; conversion costs belong
   inside timing. TopK plus this norm are plausible full-kind candidates.
3. Linear norms are already inherited from FusedDecoder; they still issue
   explicit compensated root/reciprocal chains. A custom fused scalar tail
   after the existing exact `norm_terminal_scan` could preserve each FP32 store
   and BF16-round boundary while removing launches. Native RMSNorm is not an
   established equivalent: historical `../fused_decoder/work_log.md` records
   near-tie routing changes from norm errors. No ready fused replacement was
   found. Recurrent update fusion similarly must preserve SFPU FP32 arithmetic
   and the exact two-stage reduction tree; current source explicitly rejects
   native FDR's TF32 narrowing. These are larger kernel repairs, not config-only
   fixes or evidence of a470-us improvement.

No hardware was opened and no decoder implementation changed for this CPU
investigation. It supplies hypotheses to AutoFix, not a claim of successful
FUNCTIONAL remediation.
