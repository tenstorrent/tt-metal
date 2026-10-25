# AutoFix — long free-running repetition

Status: active; no root cause or production repair established.

Contract: exact4096-token normal HF chat prompt must yield coherent full-model
free-running behavior with the accepted optimized TP4 policy. Current TT512 quotes
the repeated input through the cap; HF512 exits reasoning and writes prose. Both
share the first56 output IDs. The first differing decision predicts y56 after
consuming y55 at position4151 (position buffers then4152). Current trace, ownership,
short top-k and native-capacity checks pass but do not waive this failure.

Diagnosis: AUTODEBUG_repetition.md. One source-proven conditional discrepancy is
normal top-k's unspecified original-ID tie ordering versus HF argmax's first index.
The initial report did not establish a tie; the measured controls below now
falsify that hypothesis on the failing trajectory. Exact4096 native MM/RS and
batch-dependent decode reductions remain localization candidates. The GDN
control below closes the selected early-prefix fused/phased comparison; no
kernel defect is proven.

Ordered discriminators:
1. Preserve production policy; teacher-force the common history through y55 and
   inspect the same y56 logits, max tie set, host argmax, actual sampler selection,
   token1/token279 ranks and margins. Compare common samplers on that saved vector.
2. Run same-prompt512 with the existing explicit host-sampling compatibility path;
   this is diagnostic only, never a performance or production fallback.
3. If tie semantics is falsified, localize the first numerical mismatch with
   same-history/batch and single-axis operator controls from the AutoDebug report.
   Disabled fusion, larger physical batches, higher precision, penalties or longer
   caps are not accepted final fixes.

The device-owning benchmark pipeline remains serialized. Probe implementation is
being prepared without running another device job. Commands/results follow here
when executed; no experiment outcome is inferred from the source inspection.

## Measured sampler controls

`long_quality_decision.json` reproduces the exact first56 HF/TT IDs and reads the
y56 logits at position4152 after consuming y55 at4151. TT279 is the unique maximum
22.375; TT1 is rank2 at19.125, margin3.25. Both device top-k and host argmax select
279. Full normal512 and explicit host-sampling512 outputs are identical token for
token and reproduce the15-token loop. `long_quality_sampler_only.json` then feeds
the SAME saved BF16 logits to both common device strategies; top-k and force-argmax
both select279. This falsifies sampler tie-breaking as the cause on this trajectory.
No sampler or decoder policy was changed.

Commands: `python -m models.autoports.qwen_qwen3_8_27b.tests.probe_long_quality_decision`
and the same module with `--sampler-only-logits .cache/qwen38-y56-logits.pt --output
models/autoports/qwen_qwen3_8_27b/doc/full_model/long_quality_sampler_only.json`, both
through the established container environment wrapper with tracking1/skip-cache0.
The full-stack run was necessary to capture the actual failing decision; subsequent
sampler localization loads no model. Next are reduced identical-operand GDN and
projection controls before more full-stack math variants. A reduced two-layer
profile is collecting actual selected-kernel identities, not full-model timings.

## Measured same-operand GDN control

`gdn_same_operands.json` reports `complete` for real reduced prefix layers0/1/2/3,
B1, context262144, exact4096 HF prompt. Linear layers0/1/2 each compare AUTO,
explicit fused NV2/NP7, and phased GDN on identical raw flat production Q/K/V,
decay/beta and cloned FP32 initial state. C32, inverse choice, compute policy,
and internal flat-Q/K normalization are unchanged. Every pair is exactly equal
on all four ranks: whole outputs, all128 output chunks, and final states have
max absolute error0, RMS0 and PCC1. The JSON metrics were independently parsed
and asserted, rather than inferred from the completion status.

Command: `python -m models.autoports.qwen_qwen3_8_27b.tests.probe_gdn_same_operands
--layers 0 1 2 3` through the established serialized container wrapper. Artifact
SHA256: `114b5d5aecfa7c931e5168cb431a9fa9347dae55e676b1d4a4eef84d8a71dda5`.

This falsifies a fused-versus-phased GDN arithmetic difference for these actual
early-prefix operands at NC128; it does not certify every later layer or explain
the full-model loop. A full phased-model fallback is neither justified by this
control nor an accepted production repair.

## Native projection probe setup correction

The first native/separate prefill projection probe invocation did not intercept
any projections: the test indexed public layer objects while execution used their
`_prefill_impl` objects. Root corrected the test-only identity map to
`id(getattr(layer, '_prefill_impl', layer))`, archived the failed receipt, and
started the corrected serialized probe. This was a diagnostic harness failure,
not evidence of a production projection failure. Numerical results remain pending.

## Bounded reduced profiler setup

`tests/profile_full_model_reduced.py` now prepares buckets128 and4096, warms only
two decode replays via three generated outputs, and replaces the unmeasured127
decode replays with a public127-token continuation prefill at start4096. The
measured decode still consumes token100 at position4223 with the actual configured
cache pools; history is explicitly synthetic token100 continuation, not an
autoregressive history. Only the Python harness changed; the shell wrapper and
current running profile were left alone. Source compilation and diff checks pass;
this bounded capture version has not yet run on devices or produced a measured
raw-log size. It is reduced-stack diagnostic evidence, never all-layer performance.

## Corrected native projection comparison

`prefill_rs_same_operands.json` completed eight projections (out/down, real
layers0..3) on identical captured operands with all four ranks. All actual MM and
final-output metrics are finite. Native/separate post-RS PCC ranges0.99975255 to
0.99997514. Native direct-BF8 MM has larger tail FP32-reference RMS than the separate
BF16-then-BF8 path; blocking/rounding differ, so this is not proof of a fusion defect.
The corrected probe captures the operation's first returned MM tensor; persistent
intermediate storage is RS scratch and the earlier scratch-MM numbers are invalid.
Next: exact-geometry direct-BF8 standalone matmul, then all64 same-history eager
versus traced decode and diagnostic separate-prefill if needed. Production policy
is unchanged.

## Measured matched-geometry native matmul control (layer0 only)

`prefill_rs_matched_matmul.json` completes real reduced-prefix layer0 out/down
controls at B1/S4096/context262144. The probe captures the fused operation's
actual returned MM tensor (not its persistent reduce-scatter scratch), then runs
standalone matmul on the identical TT operands with the exact program geometry,
compute configuration, DRAM configuration and direct BF8 output dtype. Both
roles are bit-exact on all four ranks: maximum error0, RMS0, PCC1. These values
and the layer/role scope were independently asserted from the saved JSON.

Artifact SHA256: `08106fbc37564e65b4714a9467ac6cb39d164cf2fb19ed8cec63de2570d2b6c5`.
Command: `python -m models.autoports.qwen_qwen3_8_27b.tests.probe_prefill_rs_same_operands
--matched-matmul` through the established serialized container wrapper.

This falsifies a fused-versus-standalone MM arithmetic difference for these
layer0 operands with matched geometry/dtype. It does not establish equivalence
to the separate path's different blocking/BF16-then-BF8 rounding, certify later
layers, or localize the full-model loop. The optional `--matched-reduction`
control now feeds this matched standalone BF8 MM through the original layer's
separate reduce-scatter, preserving policy and workspace ordering, to isolate
reduction differences. Its default artifact is `prefill_rs_matched_reduction.json`;
this newly implemented reduction control has not yet run.

## Source-proven native prefill partial precision

Read-only inspection finds no repeated BF8 partial-storage downgrade in the
accepted native out/down geometries. This conclusion comes from the selected
C++ factory and compute kernel, not merely the Python requested output dtype.

`tt/fused_prefill.py` supplies out K-block48 and down K-block17, BF16 input,
accepted BF4 weights, LoFi/non-FP32 destination configuration and direct BF8 MM
output. In `ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/
ccl_fusion_mcast_2d.cpp:1973–1985`, `num_blocks=K/in0_block_w`. Without bias,
requested packer-L1 accumulation is enabled only when `num_blocks>2`. When
enabled and FP32 destination is disabled, `interm0_data_format` is explicitly
`Float16_b` (BF16), irrespective of BF8 output. When disabled, that field follows
the output dtype. The factory actually installs this format on intermediate
CB5, separately from output CB4 when their formats differ (`:2918–2946`), and
defines `PACKER_L1_ACC` only for the enabled case (`:2490`).

- **Out:** K48/block48 =1 K block. Packer-L1 accumulation is disabled; although
  the intermediate descriptor follows BF8 output, there is no non-final K block
  to spill/reload. The dot products accumulate in the configured non-FP32 DST
  mode and pack once to the BF8 output. A BF8 intermediate descriptor alone does
  not mean repeated BF8 partial rounding.
- **Down:** K136/block17 =8 K blocks. Packer-L1 accumulation is enabled with
  BF16 partial CB5. The kernel packs non-final blocks to that BF16 CB, enables
  accumulation after the first block, reloads partials for the last block, then
  switches the pack format to the BF8 output with L1 accumulation disabled.

The selected compute source is
`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/
bmm_large_block_zm_fused_bias_activation.cpp`, selected by the fusion factory
at line2792. Its inner-K accumulation is at375–394, final output format/pack at
400–433, non-final partial pack at436–454, and final reload scheduling at475–485.
BF16 partial storage is not FP32 accumulation; this inspection does not claim
additional mantissa precision beyond the configured DST/packer modes.

The separate prefill path instead uses K-block4 for both roles (`tt/
multichip_decoder.py:35–36`): out12 and down34 blocks, BF16 matmul output
(`tt/optimized_decoder.py:293`), then explicit BF8 conversion before RS
(`tt/multichip_decoder.py:310–313`). Its standalone mcast2d factory enables
packer-L1 accumulation for more than one block and explicitly selects BF16
partials when FP32 destination is disabled (`ttnn/cpp/ttnn/operations/matmul/
device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:370–380`).
Thus separate versus native changes K grouping/partial-rounding boundaries and
BF16-final-then-BF8 versus direct-BF8 output, as well as reduction scheduling.
The observed accuracy difference cannot be assigned to an unintended BF8
partial downgrade from this source. The measured matched-geometry direct-BF8
standalone equality above further confines the layer0 comparison, without
proving that either numerical policy resolves the full-model quality failure.

No hardware run or production edit was performed for this inspection.

## HF exact-history margin

`hf_long_quality_decision.json` (CPU BF16 HF, exact4096 prompt and first56 HF
outputs) reproduces token1 as unique argmax at21.0 versus token279 at20.125,
margin0.875. TT on the same history instead has279=22.375 and1=19.125,
margin3.25 in the opposite direction. This establishes model-distribution drift,
not a sampler-choice failure. It does not identify the earliest deviating layer.
Command: established wrapper plus `python -m
models.autoports.qwen_qwen3_8_27b.tests.hf_long_quality_decision`;
`hf_long_quality_decision_retry.log`. No HF timing is a performance measurement.

## Full-stack eager and separate-prefill controls

`long_quality_eager.json` completed on all64layers. Replacing exactly56 model
trace dispatches with eager `_decode(masked=False)` leaves all y56 logits BIT-EXACT
(max/RMS0/PCC1), output IDs and post-replay position4152 identical; sampling remains
traced and trace/capture identities stay fixed. This excludes a trace-only model
execution discrepancy on the failing history.

The same loaded model, after retiring traces and rewarming the diagnostic separate
prefill projections, produces coherent512-token reasoning/prose (reasoning closer
at338), with no15-token tail loop. It differs earlier at y55 and still predicts279
at teacher-forced y56 (22.375 versus quote18.875); a single margin is not a complete
free-running quality proxy. Root read the entire completion: it discusses the
research task, exits reasoning and writes substantive English prose, truncated at512.
Separate prefill is diagnostic only, not an accepted final fallback.

`prefill_rs_matched_reduction.json` further shows layer0 out/down native reduction
outputs are BIT-EXACT on all4ranks with the original standalone reduction applied
to the identical matched direct-BF8 MM output. Native MM itself is also exact.
Thus the tested fusion scheduling adds no numerical discrepancy; changed matmul
blocking/final rounding is the remaining projection distinction. Next diagnostic:
change only K-block geometry inside the native fused path, retaining original
BF4weights/BF16inputs/LoFi/BF8MM+CCL/TP4/residual contracts. No production policy
change or performance claim is authorized by these diagnostic results alone.

## Native geometry candidates (not yet adopted)

`native_prefill_geometry.json/.log` completed allfour variants on one all64 model,
with fresh traces and checks preserving every other policy field and64out+64down
native calls per full prefill. BothK4 andoutK16 produce the original512-token loop
exactly. OutK4-only anddownK4-only diverge at55, close the repeated quotation, and
produce relevant English planning without the periodic tail loop. Neither emits
`</think>` or a user-facing answer by512; this is partial improvement, not full
qualitative recovery. Root and independent reviewer read allfour completions.
The nonmonotonic outcome forbids claiming a universal precision improvement.

OutK4-only is the narrower candidate chosen for the next control, still diagnostic:
normalEOS1024 on identical original prompt, preserving the existing512-token prefix,
plus an independent nonrepetitive4096 factual chat and nearby4095 prompt with original
thinking policy. The longer cap tests whether improved planning ends; it cannot by
itself repair or waive the original copy loop. Fresh default-source correctness,
performance, nativecapacity and runtime evidence remain necessary before adoption.

## Narrow default repair selected for final validation

`long_quality_cases_tt_out.json` verifies the outK4 candidate on three fresh exact
chat-template cases. Original4096 first512 tokens match the earlier candidate
exactly; reasoning closes at887 and substantive English prose follows, truncated
at1024 without EOS. Nearby4095 closes reasoning at325 and finishes a coherent
five-paragraph answer at EOS892. Independent4096 distinct-record retrieval returns
strict JSON with allthree expected facts and EOS47. Root and independent reviewer
read the completions. The original prompt's delayed answer versus HF214 remains a
material limitation; no equal-budget quality parity or EOS claim is made.

Selected only `prefill_rs_geometry.out_k:48 ->4` in `tt/multichip_decoder.py` for
final-default validation. Nativefusion/grid/two-linkTP4/requestedLoFi/nonFP32/BF4
weights/BF16activation/BF8MM+CCL/KVsplit/residual/decode policy remain. This changes
effective out accumulation from one K block without partial spill to12blocks with
BF16 packer-L1 partial storage; that distinction is explicit, not hidden. DownK17
is retained. Earlier stage05 ledger entries remain untouched; geometry_selection.json
and rejection_ledger.json document the reopened selection because short-layer PCC
missed full-stack degeneration. Final-default correctness, new HF controls and
B1 performance/capacity/profile evidence are still pending.

## Final-default repair validation

All64 final-default validation04 passes prefill and decode top1=99%, top5/top100=100%
on100 fresh AIME reference positions. All three long outputs exactly reproduce the
reviewed candidate (`default_geometry_reproduction.json`), and the independent
clean benchmark repeats the original1024 output exactly
(`benchmark_quality_reproduction.json`). Paired HF nearby/factual controls are complete.
Final B1 median3 exact4096/128 token-out results are433.812150ms TTFT and35.186165tps;
native262144/nonaligned262141+3 and last-position decode pass. Reduced B1 profiling
and embedded fused-operation policy audit confirm selected outK4/downK17. See
`benchmark_b1_perf.json`, `kernel_policy_audit.json`, and `tracy/b1/scope.json`.
The observed copying failure is repaired; delayed reasoning on the original prompt
remains an explicit limitation. Final stage review is separate from this repair verdict.

Historical `tests.probe_native_prefill_geometry` intentionally asserts the original
outK48 baseline; its diagnostic receipt predates default adoption and is not a
current-default regression command. Current regression uses
`tests.run_full_validation --long-cases`. Exact case construction is in
`tests/build_long_quality_cases.py` and `long_quality_cases.json`.
