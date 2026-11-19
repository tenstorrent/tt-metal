> Historical repair chronology. Current resolutions are in [anomaly_ledger.md](anomaly_ledger.md) and [manifest.json](manifest.json); old pending statements below describe their dated experiment state.

# AutoFix: RMSNorm fusion contract

The failing contract is per-row whole-output and residual-subtracted contribution
PCC >=0.995 on full real decoder prefill/decode. Native RMSNorm passed component
PCC but failed complete prefill33. See `AUTODEBUG.md` for independent source analysis.
This is candidate rejection/repair work; it does not lower a gate.

## Verified diagnosis

Current source confirms the original FP32 binary/reduction path uses SFPU arithmetic,
whereas native RMSNorm uses FPU/TF32 operand intake and BF16 epsilon storage. HiFi4
and FP32 destination do not remove the operand discrepancy. Native norm remained
unchanged while other candidate code was added; the new diagnostic reproduces the
same full-layer failure on the current source. Exact argv and source identity:
`norm_diagnosis_v1_provenance.json`, `logs/norm_diagnosis_v1.log`.

The diagnostic deliberately reads router inputs/logits and actual device-selected
expert IDs outside any performance/correctness-audit claim. Baseline row8 selects
[83,310,289,298,471,343,392,272,92,135]. Attention-only native norm selects
[83,310,289,298,471,343,272,392,92,194] and produces contribution PCC0.99164850.
Both native norms give0.99162400 and the same changed membership at row8. MLP-only
native norm changes membership at rows22/27 and minimum contribution PCC0.99914706.
Thus a discrete expert-membership change is observed, not merely inferred from PCC.
A fixed-route recovery has not been measured; no claim attributes every numerical
error exclusively to routing.

## Repair hypotheses and experiments

1. Keep the original reduction and merge only epsilon add plus rsqrt. Implemented
   as `merged_rms` in the stage-owned file. `merged_norm_layer0_v1.json` passes
   exact full output equivalence and repeated trace/eager state. The independent
   diagnostic also has exact outputs and no changed expert memberships. Its first
   tracked timing was slower, so it is not selected on correctness alone.
2. Preserve the original reduction but move epsilon-add/rsqrt into the multiplier's
   RHS activation chain. Implemented as a separate candidate; results pending.
3. Keep original norms and optimize independent subgraphs. This is the reliable
   default policy. Activation fusion, projection packing and fused weighted expert
   reduction have passing device evidence; final cumulative gates are pending.

No router arithmetic, tie behavior, tolerance, functional source or native op source
was changed. Direct native gamma fusion is rejected by its shared-width gamma reader:
it would not apply the distinct four-stream weights. No layout adaptation can repair
that missing per-stream weight contract by passing the same tensor argument.

Status: norm accuracy failure has a verified safe repair (original arithmetic or
exact partial fusion). Native RMSNorm is rejected for this model's sensitive residual
path. Final selection still requires clean same-harness speed evidence; stage
completion is not claimed here.

## Native convolution CI failure and model-local repair (in validation)

Contract: preserve the functional FP32 four-tap products/pairwise sum, BF16
convolution-output boundary, accurate FP32 SiLU and final BF16 boundary, while
keeping the0.995 full-decoder bar. `final_v1_synthetic` failed GDN length1 decode
step1 at0.9946835637. Seven other synthetic cases and all real-HF short cases
passed. No assertion or seed was changed.

Independent source diagnosis: [autofix_ci/AUTODEBUG.md](autofix_ci/AUTODEBUG.md).
The installed AutoDebug launcher completed in a fresh read-only session, with
additional independent source review. Its report hash ledger matches the failed
runtime. Owner controls `ci_control_{functional,best_group64,best_group64_layout}_v1`
pass; `ci_control_best_group64_conv_v1` reproduces the exact failure. This isolates
the native convolution branch. The native false/true compute settings conflate
accumulator precision, activation precision and the absent BF16 boundary. Merely
selecting FP32 destination does not restore the reference contract.

The16 additional real-checkpoint CI-distribution requests in
`real_ci_distribution_conv_v1` all pass. The repair therefore preserves a fused
implementation; it does not justify a slower fallback solely from synthetic weights.
No observed router-ID change is claimed for this failure.

Experiments:

1. `conv_reduction_component_v1`: setup-packed FP32 taps, concatenate windows,
   one product and accurate tap-axis sum, explicit original BF16/SiLU boundaries.
   Bitwise output/replay agreement at1/2/31/32, but substantially slower. Rejected.
2. `conv_exact_component_v1`: first model-local generic SFPU implementation had a
   missing compute/common.h include. Compilation failed before kernel execution;
   device closed normally. Fixed the include.
3. `conv_exact_component_v2`: four shifted BF16 device slices feed one generic
   program. The reader streams original FP32 tap tiles and BF16 input tiles; native
   SFPU row-broadcast multiply plus pairwise add preserve the original order.
   Explicit native typecast rounds/masks to BF16 in FP32 registers; SiLU is
   reinitialized after typecast (which changes SFPU constants), evaluated in FP32,
   then rounded again before packing. Output and replay are bitwise identical to
   original at all tested lengths. Tracked component median1/32 improves
   0.192600/0.265460ms to0.096095/0.149557ms.

Changed production file: `tt/fused_decoder.py` only. The model-local specialization
is embedded there to respect this stage's source boundary; no global native op was
changed. Four device slices remain; there is no host conversion, fallback, format
roundtrip, or native FPU convolution in the selected exact branch. Larger internal
calls chunk into actual1..32 rows and carry real state; public logical lengths are
not restricted. Native failed modes remain explicit experimental policies.

The new default uses `fuse_gdn_conv_exact=True`, native experimental convolution
False. The targeted midpoint/cancellation controls, unchanged full eight-case CI,
real-HF short suites, continuation, Watcher, context and clean complete-decoder
performance are being rerun as `final_v2_*`. Until these gates close, this is a
candidate repair, not stage acceptance.

## QKV reduction-order regression (final-v2 → final-v3)

Fresh AutoDebug report: `autofix_batch/AUTODEBUG.md`. The batch32 real-HF
failure localizes to QKV. QKV accurate-sum with original256-output grouping
still fails; both original QKV and the new grouped exact-tree QKV pass. Thus
changing grouping alone does not repair it. The repaired default uses the same
FP32 products, halves/addition tree and odd tails as the functional implementation,
while concatenating output weight blocks during setup. Other accurate sums remain.
No router membership change is claimed without a fresh intermediate capture.

`final_v3_batch0_32` passes unchanged32-request gates. Correct clean group sweep
`qkv_tree_groups_layer0_clean_v2` verifies explicit FusedPreciseProjection plus
balanced_tree on each candidate: group1/2/4/8/16/32/64 decode medians
21.3982/17.9630/16.3105/15.3251/15.0663/14.8513/14.6362ms;
functional46.7698ms, final functional tail46.7767ms. Group64 retained. Prefill32
292.592ms versus421.901ms functional. Full gates and fresh final-source timing
remain outstanding.

The first group-sweep harness run (`...clean_v1`) did not recognize the new
variant prefix and inadvertently used unfused projections. It is invalid candidate
evidence. The harness now explicitly selects the policy, asserts the QKV type and
balanced-tree flag, and records weight-block geometry. The invalid source is
archived under `sources/qkv_tree_groups_layer0_clean_v1/`.

The profiler host-only unittest suite passes via `python -m unittest`. Invoking
it under repository pytest first failed its no-TTNN-import assertion because
pytest's repository conftest imports TTNN. No parser assertion was weakened;
`profile_validator_cpu_unittest_provenance.json` records the isolated successful
command and the earlier pytest log remains retained.

## Public PLE capacity regression, final-v4

The matched current-environment functional4768 control passed (`capacity_control_baseline_4768_v1.json`). AutoDebug (`autofix_capacity/AUTODEBUG.md`) reconciles the exact first-block live-byte ledger; it excludes previous-product retention as the immediate failure. Same-dtype FP32 typecast allocates1.25MiB and grouping requires another1.25MiB slice; setup regrouping fragments the pool. Narrow repair bypasses only already-FP32 weight casts (no arithmetic/type change) and separately releases consumed accurate-sum products before later blocks. BF16 weights still widen through the original explicit cast. Public4768/4767 gates are rerun unchanged; capacity is not reduced. Runtime SHA256 5b15557a0490bec563bba8c44b86bcf675e5d823b2798b652f991b01237712ba. Validation pending.

Residual-only group1 control advanced past GR, then failed GDN recurrent output allocation25,165,824 bytes (3,145,728 perbank; largest3,006,464). Thus it changed the immediate obstruction but did not restore public4768. Next control retains all original precise projection blocks with projection_decode_group=1; arithmetic fusions remain enabled.

All-group1 control passed both4768 and4767, including prefill/decode PCC, reset/repeated-prefill, changed/repeated replay and complete state equality.4768 prefill global0.9981198714 and contribution0.9980847588 (minimum-row contribution0.9951784297); matched functional global0.9981200046/contribution0.9980848945. All decode contribution PCC>=0.9980316489. This supplies a capacity-preserving configuration; faster grouped layout remains under investigation through recurrence I/O fusion, not an accepted capacity reduction.

Direct packed setup alone with group64 and compact MoE/recurrence passed complete public4768 (`direct_setup_v1_ple_4768.json`). Releasing each original projection block set before direct grouped checkpoint upload restores capacity without reducing projection grouping. This supports placement/fragmentation as the cause rather than insufficient aggregate DRAM or a new physical sequence cap. The combined unpack-widen candidate is undergoing4768/4767 and final ranking.

### Packed expert follow-up (in progress)

The width-adapted packed40 candidate passes synthetic8/8 and short real HF0/1/3, but its prepared public4768 run fails the80MiB MoE down allocation (largest per-bank block10,031,488B versus10,485,760B requested). Aggregate free memory exceeds the request, so this is another layout/allocation-history regression, not a hard capacity reduction.

An activation-reader fusion removes the two packed gate/up slices. Its first test-only prototype incorrectly used padded `Tensor.volume()` with logical height; tail tile counts overran the allocation. Correct explicit tile counting repairs determinism. Matching native binary's SFPU multiply and BF16 unpack-to-FP32 mode repairs the FPU variant's small rounding delta: all12 component shapes are bitwise exact in `packed_silu_component_v6_watcher.json`. Intermediate Python descriptor argument/64-CB-vector errors occurred before kernel execution and are preserved in v4/v5 logs. The production function is re-exercised by the integrated component probe.

First integrated capacity invocation stopped at an overly strict rank4 validation: sparse matmul output combines operand batch axes into rank6. The kernel's flattened tile indexing already supports these leading dimensions. Validation now accepts rank>=2 and the component uses actual rank6 shapes. `packed_silu_integrated_plan_v2.json` is the next capacity/correctness/ranking gate. No failed prototype is a default runtime path; no device reset was needed.

## Packed-expert capacity repair closure (candidate evidence)

Fresh inspection-only diagnosis is [autofix_packed_capacity/AUTODEBUG.md](autofix_packed_capacity/AUTODEBUG.md). The packed late-allocation path had156,113,920bytes free globally but only10,160,512bytes in its largest per-bank block, below the10,485,760bytes required by the80MiB down output. This was fragmentation, not retained packed/slice tensors or aggregate exhaustion. The current-source direct/unpack control passes, ruling out the unrelated shared-buffer ownership timing change as the sole cause.

The minimal repair preallocates the surviving20MiB activation before the temporary40MiB packed result, then passes it as the exact activation kernel destination. The same live-byte count now leaves15,403,392contiguous bytes per bank. `packed_activation_placement_proof_v1.json` hashes the late/early address ledgers and successful public4768 gate. Both4768 and4767 HF/reset/trace gates pass in `packed_early_v2_*`; synthetic8/8, short HF0/1/3, continuation andbatch32/PLEbatch5 pass in the subsequent wide plan. Early placement timing differs from late placement by less than0.04% and is not claimed as a compute speedup. It enables keeping the substantially faster packed-expert graph without reducing capacity.

Rejected alternatives retained in the log include narrower20-core packed gate/up and residual group32, which did not repair capacity. The broader compact-prefill union kernel is bitwise exact, but fixed10*T padded expert slots make longer chunks slower; it is not selected as the capacity repair. A first early-placement attempt failed on unsupported TTNNShape slicing before device kernels; converting the static shape to a Python list repaired that host descriptor error. No reset, gate relaxation or capability reduction was used.

These are candidate repair results. Final default source gates, independent review and stage acceptance remain separately required.
