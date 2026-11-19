> Current stage outcome: [layer34 AutoFix failed to meet the unchanged gate](AUTOFIX_v5_layer34.md). The material below records earlier repair history.

# AutoFix: extended 8192-token recurrent accuracy

Diagnosis: [independent AutoDebug](../autodebug_endurance/AUTODEBUG.md), completed via installed inspection-only launcher. Failing contract: every output and residual-contribution PCC >=0.995 over8192 sequential decode tokens, same seed913/prompt128, deterministic trace repeats, followed by continuation33. No threshold or reference semantics change is permitted.

The current candidate fails seven positions; an older SFPU-control snapshot fails ten, overlapping six. Thus new matmul/carry/sharding is not a necessary cause. Saved incoming-state/current-input probes and independent CPU replay identify inherited recurrent-state differences, amplified in small-RMS heads. Substituting HF incoming state restores attention PCC>=0.999916 at all failing captures; same-input TT recurrence matches canonical CPU locally within~1e-5. Output norm changes alone barely help. Native input RMSNorm, gate rounding, convolution pre-SiLU rounding, and Q/K normalization remain source-supported precision differences.

## Discriminating experiments

1. Decode A/B from existing BF16 packed projection, widen A before dt_bias, BF16-round sigmoid beta. `gates_exact_8192_0_results.json` reduces seven failures to three (3032,5011,6595), minimum output0.994495/contribution0.990714. This supports a real gate contribution, not closure. Prefill unchanged.
2. Add primitive BF16 Q/K square/sum/rsqrt/product to the gate fix. `gates_l2_exact_8192_0_results.json` fails seven positions, minimum contribution0.983661. Reject this implementation as a repair; same-input precision microprobe will check whether its BF16 reduction matches HF before drawing a stronger conclusion about the semantic boundary.
3. Gate fix plus FP32 SFPU input norm, casting only its final weighted output to BF16, is running. Native weighted or unweighted RMSNorm retains FPU normalized multiplication; it is not an exact arithmetic discriminator. Post norm and recurrence remain unchanged.

The stress harness gained optional CPU-only oracle reuse. Cache keys include sequence parameters, pinned revision, Torch version, and SHA256 of the harness, reference, HF model source, and DynamicCache source. Reuse also verifies prefix/tokens/continuation bit-equality. Cache stores canonical outputs only, computed by the unchanged HF forward; device code and measured forward remain independent. This saves repeated canonical CPU work during isolated device experiments. Earlier no-cache artifacts are preserved. Any harness/reference/source change invalidates reuse. Final gates will include fresh-reference runs.

Repair remains open. All failing results and source snapshots are retained. No native/shared code, functional decoder, checkpoint, context limit, or later pipeline stage is changed.

## Same-input precision checks and follow-up controls

`frontend_precision.json` provides a device discriminator at128/3032/3979/5011/6595/7703. Explicit FP32 SFPU input norm matches HF BF16 outputs bit-for-bit at all six. Native norm differs341–607 codes. Native FP32 RMS Q/K differs1169–1665 codes; the primitive BF16 L2 control first diverges in reduction and changes119 output codes at128. Explicit FP32 SFPU arithmetic with BF16 boundaries matches every L2 intermediate/output exactly on all six inputs. These are same-input checks, not whole-decoder acceptance.

The gate+SFPU-inputnorm stream still fails four late positions (7551/7668/7703/8157), worst contribution0.977648. Adding the verified L2 recipe keeps four failures, worst0.978614. Adding FP32 four-tap convolution followed by BF16 preactivation/SiLU gives six failures, worst0.980933. These controls are retained with snapshots; improvement of an individual arithmetic contract does not establish full-stage correctness. The convolution-only extension of the gate patch also fails. The current diagnostic source is `endurance_gates_norm_l2_conv.py.snapshot`.

`prefix_state_probe.py` explicitly substitutes canonical HF prefill state only to isolate its influence. It is marked diagnostic, cannot satisfy acceptance, and is absent from production. With the otherwise corrected decode frontend it removes4904/5011 failures but leaves7551/7668/7703/8157, worst contribution0.982334. Thus prefill mismatch contributes but does not explain all late errors. A second selected-position capture `endurance_exact_frontend_probe` now measures the remaining earliest decode divergence. No production host fallback was introduced.

The reusable CPU oracle is independently checked against the original three-repeat failure: prefix/tokens and all8192 canonical reference outputs are bit-identical (`oracle_cache/stress_8192_original_equivalence.json`). Optional reuse changes experiment cost, not the reference or gate.

## Projection arithmetic and convolution activation localization, 22:58–23:14 UTC

The corrected-frontend captures have bit-exact HF input norm at all13 selected positions, but packed projection still differs from canonical CPU FP32 linear by~0.00033 relativeL2. `projection_precision*.json` compares identical uploaded BF16 inputs/weights; upload roundtrips are bit-equal. Generic and DRAM-pipeline matmul, block1/2/4, packer accumulation on/off, FP32 operand formats, and K2/4/8 outer splits all retain the same error. Fidelity sweep givesLoFi0.026608/HiFi2 0.004219/HiFi3 0.000339859/HiFi4 0.000330360. HiFi3's Wormhole-specific source advisory is not a fix for this Blackhole observation.

`matmul_partial_precision_adapted.json` isolates the native summation boundary: identical two products atK(0,1) differ from CPU by0.000132 relativeL2, while placementK(0,16) and both individual-product controls are exact. Eight-way residue masks plus FP32 SFPU addition reduce full-projection error to5.18e-7;16 masks give3.21e-7. Source identifies16-product MVMUL groups but does not specify their internal mantissa precision; the runtime observation is stronger than an unsupported exact-bit-width claim. The initial mask probe failed cleanly because `fast_and_approximate_mode=False` is only legal for BF16 binary output; FP32 already uses SFPU by default. Corrected only that test argument and retained both logs.

Direct native FP32 elementwise multiply plus accurate sum gives1.83e-7 relativeL2 and effectively exact canonical BF16 output (remaining differences are~1e-37 subnormals). A model-only decode projection control with setup-owned FP32 copies of the same BF16 weights removes the four late failures, leaving only3979, contribution0.992009 (`precise_projection_8192_0_results.json`). Its public cache/layout/context contract is unchanged; timing is not yet ranked. A diagnostic canonical-prefix seed still fails, so inherited prefill state alone does not explain the residual issue.

`frontend_precision_silu.json` identifies another real contract difference: native BF16 SiLU differs in829–878 of8192 output codes on identical BF16 convolution preactivations; FP32 SiLU followed by explicit BF16 conversion matches HF bit-for-bit at all six positions. Source uses different BF16 and FP32 exponent paths; both claim nearest rounding, so this is not attributed merely to truncating the output. The next fullstream control changes only this activation computation on top of precise projection and the verified frontend/recurrence controls. Final stage remains open until the unchanged gate passes and performance is reranked.

## Local recurrence closure and remaining accumulated history, 23:18–23:22 UTC

The precise-projection + FP32-SiLU control still fails seven8192 positions, minimum contribution0.980560. Canonical prefill seed (both convolution history and recurrent state) leaves four late failures, minimum0.975604; these diagnostic substitutions are not acceptance evidence. `endurance_precise_probe.json` now captures each recurrence intermediate. Independent CPU replay finds same-input new-state relativeL2 about1.5e-8, with2–6 BF16 cache-code differences, versus the earlier~1e-5 contraction error. At7703 Q/K/V/beta match HF exactly; substituting only HF incoming state makes the canonical BF16 core exact. Cache casting itself is correct. Reports: `endurance/AUTOFIX_precise_recurrence_followup.md`, `endurance/AUTOFIX_precise_frontend_followup.md`.

The next isolated candidate replaces recurrent addcmul with separate multiply/add, matching HF operation boundaries. A first launch failed before opening hardware because trace-allocation tracking was missing from the environment (`logs/precise_separate_update.log`); retry explicitly sets tracking=1 and skip_program_cache=0 (`logs/precise_separate_update_tracked.log`). No device recovery was needed. A separate rare first-producer discrepancy atposition130 changes one inputnorm BF16 code on a rounding midpoint; rsqrt already defaults to accurate mode, so simply passing false is not a repair. Intermediate diagnostics are pending.

## Reduction-order controls, 23:23–23:28 UTC

Separate recurrent multiply/add still fails six8192 positions, worst contribution0.981106. Position130 norm microprobe localizes first difference to the FP32 variance (one ULP), with rsqrt matching when given canonical shifted variance. A two-level64x32 device sum fixes130 exactly but its integrated stream still fails fivepositions, worst0.977671. Reverting only SiLU to BF16 on that path also fails; retained as a rejected control. No intermediate improvement is treated as acceptance.

Independent CPU emulation identifies the installed Torch reduction schedule: an eight-lane norm schedule matches all8320 input-row means exactly, while FP64 or arbitrary grouping changes dozens of finalBF16 codes. `AUTOFIX_norm_order_prefill_followup.md` contains source anchors and a model-only add schedule. A device-only explicit-add implementation is being checked; its cost has not been ranked. The recurrence reduction order is being independently reconstructed too.

`gate_precision.json` compares identical canonical A inputs atninecaptures. Fused and separate native softplus both differ in54/288 FP32 g values; log1p(exp(a)) differs32/288. Their exp(g) code counts are29/288 and23/288 respectively. The alternative is a candidate, not a bit-exact fix. Exact command is in `logs/gate_precision.log` provenance and `tests/gate_precision_probe.py`; all device operations remain native TTNN, host comparisons occur only in diagnostics.

## Canonical sum schedules and unary follow-up, 23:29–23:32 UTC

Explicit CPU-order input norm reduces8192 failures tofour latepositions but does not pass. Matching both recurrent contractions with eight contiguous16-row sequential sums followed by sequential partial accumulation still fails; the canonical-prefix diagnostic also fails. These candidates remain controls, not selected defaults. Detailed CPU proof of recurrence order is in `AUTOFIX_reduction_order_followup.md`.

`norm_order_precision_device.json` checks all8320 rows. Explicit CPU-order mean has0FP32 differences; native mean differs4006 rows, two-level3581. FinalBF16 norm differs78/118/105 elements respectively, pointing to subsequent unary arithmetic. Scalar-per-token versus batched CPUrsqrt was checked and found bit-identical (`norm_inverse_scalar_cpu_semantics.json`); the broad oracle remains valid. Correctly rounded sqrt followed by correctly rounded reciprocal matches HF all8320 cases, whereas simply computing mathematical inverse-sqrt at higher precision does not. Device probes at430/803/928 confirm native rsqrt can differ byoneULP on exact canonical variance. Simple sqrt→reciprocal fixes430/803 but not928. A refined model-only sqrt/reciprocal control is pending.

A fullstream log1p(exp(a)) gate control is running on the explicit-sum candidate. The direct gate microprobe motivates this experiment but no acceptance or speed claim is made. All original threshold0.995, references, model weights and public context contract are unchanged.

## Reciprocal-square-root refinement localization, 23:33–23:37 UTC

`norm_order_inverse_device.json` compares all8320 canonical means. Native rsqrt differs1760 inversecodes/78finalBF16normcodes; sqrt→reciprocal1297/47; directNewtonrsqrt2192/68; FMA-refinedsqrt697/23; also-refinedreciprocal682/22. The integrated lattercontrol (`refined_inverse_8192_0_results.json`) still fails sixpositions, minimumcontribution0.987528, so it is not accepted. The previous log1p-exp gate control also failed.

`inverse_rounding_device.json` explains incomplete refinement on actual operands: atposition1 rawsqrt is correctly rounded, but its FMA cancellation residual should be−5.0909401694e−11 and device addcmul/mac both return−2.9103830457e−11. Ordinary split multiply/add loses the residual entirely as expected from FP32 rounding. We do not infer an undocumented exact mantissa width from this observation. Independent CPU tests of BF16-high/FP32-low split-product residuals recover the correctly rounded root and reciprocal for all8320 values even with startingestimates shifted±2ULP (`inverse_split_product_cpu.json`, `inverse_split_reciprocal_cpu.json`). Device verification is pending.

`bf16_unary_precision.json` enumerates65280 finiteBF16 inputs and32639 positiveinputs forrsqrt. FP32SiLU→BF16 differs numerically only in tiny/subnormal outputs (maximumabsolute8.76e−37; othercodechangesare signedzero), sigmoid onlythreeunderflowvalues near−88, and rsqrt onlysubnormalinputs/extreme>3e38 values. Thus these BF16 unary boundaries do not explain the ordinary-magnitude endurance failure. All enumeration is diagnostic and absent from production.

## Exact input normalization established, 23:38–23:40 UTC

`inverse_rounding_split_device.json` verifies the split-product refinement at all11 focused values, includingpreviousFMAfailures. `norm_order_split_device.json` then checks the full8320-row stream and reports0mean/0inverse/0BF16norm differences. This is actualdeviceevidence for the isolated normalization path. The integrated candidate `endurance_split_inverse.py.snapshot` still fails8192 (`split_inverse_8192_0_results.json`), so exactinputnormalization is necessary arithmetic progress, not stage closure. A new canonical-prefix seed diagnostic and stream-wide frontend comparison will separate inherited prefill state from remaining decode projection/gate arithmetic.

## Full frontend stream and finite-domain gate folding, 23:43–23:48 UTC

`linear_frontend_stream_probe.json` runs actual F.norm/linear/_linear_attention methods and stops before recurrence, seeded once with canonical convhistory. All8192 inputnorms are exact. Packedprojection differences include793751 QKV codes (mostlytinydeadchannelvalues, butmaximumabsolute0.03125),6889Z,52A,53B; downstreamV differs2723entries andbetaoneentry. The earliest remaining meaningful producer is projection accumulation; CPUhardware/software dot-order analysis is active. This is diagnostic, not full-stage acceptance.

`AUTOFIX_prefix_error_persistence.md` proves onCPU that prefille_error persists with BF16cache rounding: canonical subsequentinputs/arithmetic starting fromTTprefix leaves1.132% finalstate relativeL2, whereas unroundedlineartransport shrinks~0.0007%. Canonical-prefix hardware controls still fail, confirming remainingdecodeerrors also matter.

The BF16 A projection has65536 possible bitpatterns. A checkpoint-specific setup table32x65536 folds dt_bias,softplus,A_log andexp, with nativebitcast<uint16>/gather at runtime. TwoFP32tables occupy16MiB. `gate_lut.json` verifies exactFP32g andexp(g) atallninecapturedinputs; current`endurance_gate_lut.py.snapshot` tests the integrated decoder. This uses no runtimehostdata or per-input oracle; tables cover the wholeBF16domain and depend only oncheckpointconstants. The candidate still has diagnosticunusedg calculation that must be pruned ifselected. No performance claim yet.

## Prefill frontend state and ordered decode projection, 23:50–23:55 UTC

The integrated gate-table control still fails fivepositions (minimumcontribution0.979914). A precise-prefillfrontend control adds8maskednativeprojections, exactinputnorm acrossS, vectorized4tapconv, explicitBF16L2/gates, andrank4 normalizedQ/K to bypass nativeflat normalization. It preserves publicchunk/cacheboundaries. PrefilloutputPCC0.999970passes but8192stillfails ninepositions. `prefill_state_device.json` records exactinputnorm, smallQ/K/Vfrontend differences and0.004202relativeL2 BF16publiccache error; nativechunkrawstateerror0.003451. No acceptance claim.

`AUTOFIX_projection_cpu_dot_order.md` identifies actualoneDNNavx512_core_bf16 decode reduction: fourcontiguousK512 blocks, accumulateoddthen evenproductperpair, combinepartialblocksleft-to-right. CPUemulationmatches allsevenreportedordinaryprojection counterexamples and142additionalA/B cases. Atposition1068QKV2907, mathematicallycloserFP64BF16valueis−4.03125 butHFreturns−4.0, explainingwhy moreprecisionalonecannotmatch. CurrentdevicecontrolreordersKweights/indicesatsetup, usesFP32SFPUproducts +native sequentialcumsum(disable_compensated_sum=True) on4x512blocks, then3SFPUadds. The flagis a documentedtemporaryopt-out in thispinnedcheckout; compatibilitymustberecordedifselected. No512-ophostunroll/sharednativeeditwasintroduced.

## Ordered projection isolates prefill state, 23:57–00:03 UTC

`diagnostic_projection_cumsum_hf_prefix_seed_results.json` passes all8192decode positions, minimumcontributionPCC0.996693, after diagnostic canonicalprefix initialization. This is not acceptance: production must compute its own prefix. `linear_frontend_cumsum_1024.json` proves ordinary-magnitude decodefrontend values exact; remaining QKV errors are <=4.64e-37.

Independent oneDNN CPU audit identifies M128prefill as twoK1024 odd/even dots, unlike decode fourK512. Actualdevice `prefill_state_cumsum.json` now has exactinputnorm,Q,K,V,g,beta, and convolutioncache differs only<=2.83e-37. Nativechunk finalstate remains0.003401relativeL2/raw,0.004222/BF16. A four-row microbatch implements orderedFP32SFPU products+cumsum without changing publiccacheboundary. Its temporaryK1024 rule requires shape-generalization audit before finaldelivery. Commands: `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.prefill_state_precision_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefill_state_cumsum.json`; log`logs/prefill_state_cumsum.log`, provenanceandrawtensorcaptureadjacent.

## First complete8192pass, 00:05 UTC

Exactfrontends plus nativeprefilloutput and model-localFP32state-onlysequentialscan pass8192decode+33continuation with no referencecache substitution. `sequential_prefill_state_8192_0_results.json` minimumcontributionPCC0.9968147574. Snapshot`endurance_sequential_prefill_state.py.snapshot`. Statecorrection uses the samecheckpointdecayLUT and canonical16x8SFPUcontraction asdecode, withoutinternalBF16cacheboundaries; it does not computeunusedoutputcontractions. CPUaudit`AUTOFIX_prefill_state_repair.md` predicts rawstate3.34e-7relativeL2 vsHFchunk64,168BF16codechanges, comparedwith256513before. This is onecorrectcandidate, not finalrepeat/watchergates or performanceacceptance.

`AUTOFIX_projection_public_length_rule.md` establishes K512 forpublicM1 andK1024 forM>1 across15lengths1..4097/allprojectionwidths/JITloops. Explicitprojection_k nowpassesfrompublicprefillthroughinternalone-tokentails, avoidingmutabledecoder-globalstate. Correctcandidatewarmedtiming launched as`correct_precision_baseline_0.json`, length128,trackingdisabled.

## Correct-path native reductions and datatype-boundary fusion, 00:18–00:34 UTC

`last_only_prefill_8192_3_results.json` establishes full-attention8192pass, mincontribution0.9971407755. Linear nativefastreduce recurrence retains16then8sequentialFP32order and passes8192, PF166.439ms/decode4.978ms. Removalofobsoletebuffers/inputnormreshards andTILEconvhistory avoids runtimetilize/untilize, preserves8192results; logical1/31/33/128/4097runnersallpass.

PackedprojectionBF16storage andBF16coefficients feednativeSFPUmultiply with explicit`dtype=FP32,fast_and_approximate_mode=False`, not itsdefaultmixed-dtypeFPU route. `bf16_projection_operands_0.json`:PF147.420ms/decode4.800ms;8192unchangedminimum0.9968147574. `linear_frontend_bf16_operands_8192.json` proves norm/Z/A/B/V/g/beta/decay exactall8192. QKVmax1.408e-36, Q/Kandconvonlysignedzeros. Threshold1e-20isdiagnosticclassificationonly; acceptancePCCisunchangedandnoinputsareexcludedfromstress.

`unary_chain_device.json` tests322884inputs(exhaustivefiniteBF16,middle-roundingneighborsandcapturedvalues), plus161671positiversqrtinputs. Convround/SiLU/roundandround/widenchains differnearunderflow(max1.75e-38), betachainnear−80(max1.18e-38);rsqrtchainbitexact. Actualmodel`unary_chains_8192_0_results.json`passes withsame8192PCCentries. `unary_chains_0.json`:PF146.889ms/decode4.760ms. Noall-domainbit-exactclaimis made.

Specializednativecausalconvhasnoactivation-disable/preactivationoutputAPI, sourceproof`AUTOFIX_native_conv_reassessment.md`. StockFP32depthwiseConv1Dwasactuallyattempted1024/512channelchunks;firstprobe cleanlyfailsL1_SMALLregion0B (`stock_conv_precision.json`). Adapted32768Breservationrunpending;firstAPIerrorisnotarejection.
