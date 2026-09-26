# AutoFix: fusion numerical and contract checks

## Starting evidence

Functional-stage AUTOFIX_decode.md and current AUTODEBUG_precision.md establish
that tiny FP32 boundary changes may alter close top8 expert selection and lower
whole-layer PCC even when attention PCC is near1. Required acceptance remains
PCC>=.995 on every checked layer output, plus deterministic trace replay.

## Hypothesis experiments

- Repeated QKV rows and separate expert gate/up dispatch are unnecessary.
  `probe_fused_components --layer 0` verifies bitwise-equal same-input outputs;
  broadcast product groups256..16384 preserve exact QKV results. Packed expert
  decode is exact and faster. Initial prefill component probe freed its own input
  through the functional baseline's explicit deallocation; the harness now clones
  each component input, consistently for baseline and candidate. This was a probe
  lifetime error, not a fused-runtime failure. Retests in components_sliding_retry.json and components_full.json pass, including all group sizes.
- Native RMSNorm with fused FP32 gamma preserves routed output.
  Headline sliding passes, but norm_batch_sliding.json fails slot5 at .99181415.
  Adaptation: leave gamma as an SFPU multiply, with native FP32 normalization.
  norm_external_batch_sliding.json passes all32 slots (minimum decode .99963224).
  Isolated norm-site controls pass but do not establish combined-path correctness.
- Old native rotary can consume the existing decode logical shape unchanged.
  First probe returned padded logical rows32 rather than16/8; comparison failed
  by shape. Probe adapted output metadata to original logical/padded shape;
  numerical reruns completed: sliding batch fails, full batch passes but full headline fails .9930295. New HF rotary with FP32 tables and interleaved geometry
  passes both real batch32 kinds (rope_hf_fp32_{sliding,full}.json).
- Native paged SDPA with explicit BF16-Q, HiFi4, FP32 destination replaces
  precise FP32 attention. Adapted exact-shape real batch32 checks fail:
  sliding min .98152034, full .98639330. FP32 Q is rejected by native validator;
  native statistics stay BF16 even with FP32 destination. Source evidence in
  AUTODEBUG_precision.md. No kernel changes are authorized in this stage.
- Combining norm/RoPE/batched attention restores the entire contract.
  native_sliding_headline.json disproves this: minimum decode .99339605.
  Native stable-softmax adaptation makes this .98680127.
  Individual norm/RoPE/batched attention controls localize the cause as recorded
  below; neither failing combined candidate is accepted.
- Expanded QKV group width8192 works for full QKV width10240.
  TTNN slice does not clip its stop index; first full run rejected end16384.
  Clamp each group end to the real projection width. Full-kind retest and group-size sweep pass; every group is bitwise equal to the baseline.
- Binary output activation parameter is `activations` in this binding.
  The initial `post_activations` argument was rejected before kernel launch.
  Corrected to the binding's supported spelling; sliding headline then passed.

## Numerical repair outcome

The selected path passes headline HF and direct functional/fused PCC, request
reuse, batch32, continuation, and watcher checks. Incorrect native combinations
remain rejected; no assertion, PCC threshold or context capability is relaxed.

## Combined-path localization

Native HF rotary passes full headline, but the sliding prefill variant
yields decode PCC .9931. Keeping precise sliding prefill rotary and enabling
native rotary only for sliding decode restores the headline contract.

Request reuse then exposed a separate S2049 sliding decode failure (~.9886).
Controls removing RoPE or common norm still failed. Controls removing native
normalization passed. Individual external-gamma norm substitutions isolated the
head norms: input, post-attention and router substitutions pass request reuse;
head substitution fails. `preciseheads` keeps SFPU head normalization for sliding
while allowing native unweighted normalization + external gamma elsewhere.
`guarded_sliding.json` and `guarded_full.json` pass all128 headline decode steps.
The final policy also passes both kinds in batch64_request_reuse_*.json,
batch64_batched_*.json and batch64_prefix_continuation_*.json.

Native softmax is not accepted from isolated near-threshold headline success:
combined guarded sliding fails .993201; precise external centering still fails
.993447 (`centered_softmax_sliding.json`). Full native softmax passes but is slower
(~5688us vs5651us host replay), so the precise softmax is preferable.

An exact SUB+EXP(0.0) postactivation preserves the accurate sum/div path. The first
attempt explicitly set binary `fast_and_approximate_mode=False`, rejected for
FP32 output: this flag only changes BF16 rounding. Removing that flag uses the
FP32 SFPU path; both merged_exp_*_retry.json headline cases pass with unchanged
PCC. Fused residual RMSNorm also passes these runs. These failures were validation
errors, not device hangs; no reset or assertion suppression occurred.

## Shared-down fidelity attribution

Observed anomaly: the initial probe claimed HiFi2, but final native profiler rows show LoFi with program_config. Source inspection identified fidelity inference in matmul_device_operation.cpp:2808–2810. Controlled explicit LoFi/HiFi2 sweeps on both real layer activations (shared_fidelity_commands.json) pass PCC/replay for all legal K divisors; K6 LoFi remains fastest. Documentation and summary metadata now report the actual policy; runtime and original measured values are unchanged. Resolution: controlled and attribution fixed. See PERFORMANCE.md for the matched table.

## Mixed expert-batch tail and inherited cache sensitivity

Fresh logical S65 pads to96 rows and exercises the64+32 expert batches. Both real-weight layer kinds pass FP32-HF prefill. Full attention also passes all8 traced decode outputs. Sliding fails FP32-HF at position68: fused PCC0.9813712959 and functional0.9809274231; other positions pass. This failed row is retained, not relabeled or excluded from the comparison.

The paired complete outputs remain equivalent: prefill PCC0.9991649499 and all8 decode PCC >=0.9997196429, including0.9998401882 at68. The functional stage probe isolates the discontinuity: attention/residual PCC >0.999997, but CPU routing on the TT residual selects expert70 instead of the FP32-HF oracle's eighth expert10. CPU attention with identical QKV reproduces those routes. A pure CPU HF control on the saved identical inputs reproduces the same sole failed position and expert swap by rounding only cached K/V to BF16 (PCC0.9814915924); BF16 cache plus rotary tables gives0.9815510238. No TT kernel or fused rewrite is needed to reproduce it. This is inherited BF16-cache sensitivity at a close MoE rank boundary; cache dtype/context contract remain unchanged.

The durable test_fused_mixed_expert_tail keeps all8 positions, full-HF prefill checks, unchanged0.995 direct functional-equivalence threshold, finite tensors, deterministic replay, runtime audits, program-cache guard and functional-fallback prohibition. It recognizes only the independently controlled sliding position68 FP32-HF failure. Existing33-token and4096/128 FP32-HF gates retain their original threshold without exceptions.

Artifacts: verified_tail65_*.json, tail65_sliding_{functional,fused}_paired.json, tail65_sliding_equivalence.json, tail65_functional_stages.json, tail65_cpu_precision.json, tail65_diagnostic_commands.json, shared_fidelity_commands.json, AUTODEBUG_tail65.md and verified_mixed_tail_pytest.{json,log}. Exact saved tensors/inputs are local .pt artifacts; compact JSON records the comparisons and CPU fixture hashes.
