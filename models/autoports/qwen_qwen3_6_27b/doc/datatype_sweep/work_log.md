# Datatype sweep work log — Qwen/Qwen3.6-27B

Stage 8 is in progress. No precision config is selected yet. No vLLM integration
is authorized in this stage. Never push stage commits.

## Contract and startup

- Pinned checkpoint: `8f33b98937c6031bb49e713dee3e8e185aafd26f`.
- Starting optimized full-model commit: `5fee59ed` (clean worktree).
- Hardware: four Blackhole p300c chips, mesh 1x4, FABRIC_1D; serial device jobs.
- Initial `fuser /dev/tenstorrent/*` found no holders; bounded `tt-smi -ls --local`
  found all four chips. No reset performed.
- Environment: activate `python_env`, source `/home/ttuser/qwen36-27b/lane-env.sh`.
  Enabled installed skills: tt-model-bringup 0.1.24 and tt-autodebug 0.1.8.
- Thresholds: top1 >=90%, top5 >=98%; preserve top100 expectation, P1 defaults.
  Apply the accuracy bar independently to AIME24 chat100 and real natural text.
- Ranking: traced teacher forcing only; repeated serial baseline spread defines
  noise. Token-out no-readback performance is recorded separately after selection.
- Capability: retain 262144 context, arbitrary logical prompt lengths and B32
  bounded cache ownership. Recompute capacity for each changed KV dtype; run max
  context only if the winning cache dtype/layout changes, otherwise32K–64K smoke.

## Work performed

- Began original optimized default refresh with the installed
  `readiness_check.run_teacher_forcing`, AIME24 chat-template100 reference.
  Exact command, original source identity and exit are in `baseline_tf_1.command.json`.
- Read-only independent CPU policy audit identified coarse MLP group coupling,
  separate paired prefill weights, hard-coded generator KV admission and packed
  GDN correction constraints. Implementing explicit per-group policy plumbing;
  material changes will receive refreshed measurements.
- Candidate matrix includes canonical accuracy, BFP8 fidelity alternatives,
  inner-layer BFP4 MLP and gate/up LoFi/HiFi2, BFP8 head, CCL and activation, and
  BF16/BFP4 cache comparisons. Candidate files are experimental, not selections.
- Added natural-text HF reference generator using contiguous real Dickens tokens.
  This is a raw-continuation numerical stress reference, not a chat quality verdict.

## Unfinished required work

Baseline repetitions; candidate smoke/full-model measurements; P1; selected-policy
propagation; capacity/capability checks; qualitative and resumed-prefill checks;
post-selection token-out; tables/plots; installed gate; independent stage review;
local stage checkpoint and SHA receipt.

## Harness and policy checks

- The initial reduced smoke and already-started full run reached traced decode,
  then failed solely while serializing native MeshTraceId objects. Their raw
  logs are `policy_smoke.log` and `baseline_full_1.log`. Fixed the report to
  stringify IDs after comparing native IDs, retaining the strict G-1 replay
  assertion. These failed reports are excluded from ranking.
- Natural-reference generation initially failed because the packaged book has
  a newline between the two opening clauses. Fixed the textual anchor to the
  first clause; `natural_reference_retry.log` records successful512-position HF
  scoring. The `.refpt.provenance.json` records the exact real-token passage.
- Independent read-only CPU audit confirmed baseline arithmetic preservation
  and active prefill/decode group routing. Fixed silent acceptance of unknown
  layer exception/kind keys, and declared the BF16 GDN norm-weight exception.
  Host policy validation rejects keys64, -1, foo and00; all16 policies parse.
- `run_sweep.py` serializes bounded1800-second full-model jobs and stops on any
  execution failure or a `stop` admission file. Numerical failures are evidence,
  never eligibility. The refreshed baseline is measured twice in separate
  processes, two warmed repeats per reference per process, before candidates.
- The candidate set also includes a BF16/HiFi4 higher-precision reference,
  retaining the sensitive corrected GDN input representation (two BFP8
  matrices) rather than claiming globally BF16 weights. Its full-context
  preparation must fit before it is offered to P1.

- After two baseline processes, paused queue admission using `stop` (no running
  device process terminated). Reordering the BF16/HiFi4 capacity check and P1
  calibration ahead of lower-precision candidates to catch long-reasoning
  protocol or baseline failures before the expensive matrix.

- Repeated baseline combined traced rates:30.873306,30.895465,30.882392,
  30.892162t/s/u. Median30.887277; spread0.022160. Exact calculation is in
  `baseline_spread.json`. AIME100 and natural512 both pass independently.
- `bf16_hifi4` passes full-context preparation and both references; its GDN
  input remains the optimized corrected-BFP8 representation. Added a stronger
  `bf16_all_hifi4` control with BF16 GDN input and KV for P1 calibration.
- Two host-only wait helpers watched a rewritten JSON receipt as though it were
  an append-only log and missed completion. Terminated only our idle helper
  PIDs1783317/1785236 after verifying the device job exited0 and closed the
  mesh. No device process was terminated and no reset occurred. Subsequent
  waiting uses process completion, not append-log matching on rewritten JSON.

- Strong BF16 primary weights/HiFi4 plus BF16 KV (`full_bf16_all_hifi4.json`)
  passes AIME100 at100% and natural512 at99.0234% top1; top5/top100100%.
  Full262144 preparation leaves2.296229GiB/device (308194752bytes/bank).
  This is the strongest fitting P1 reference; no advertised capability reduced.
- Started `p1_calibration.command.json` using four reasoning texts, unchanged
  6144-token generation cap and default statistical thresholds, against the
  baseline. The reduced text count is a sampling limitation, not a relaxed gate.
- Updated optional max-context acceptance memory accounting to consume actual
  selected KV tile size. Strengthened short-KV harness with runtime precision,
  actual cache dtype, unchanged trace-ID and exactly-one-decode-replay checks.

- Strong-reference propagation audit initially flagged UINT16 in layer0 CCL.
  Source inspection traced it to SamplingCCL top-k index transport sharing
  model.io; floating residual/projection payloads are still BF16. The checker
  now permits UINT16 only in layer0; `propagation_bf16_all.json` passes.
- A second independent fixed-policy audit found validation-only settings for
  activation/residual, recurrent-state and norm allocation dtypes. Preparing
  explicit construction wiring plus runtime observations. No model source is
  being changed while the P1 calibration process is live. Selection remains
  blocked on those changes and their refreshed evidence.

- Independent CPU review of prior `optimized_full_model/perf/runtime_matmul_policy.csv`
  and `layer_accounting.json` confirms coarse sweep priorities. Representative
  layer extrapolations: MLP9.59ms/token, GDN input5.77ms, GDN output1.37ms,
  head1.57ms. These are not new full-model timings. If broad candidates fail,
  isolate down4LoFi, GDN-output8LoFi/HiFi2, or head8HiFi4 before rejecting
  those groups. Combine individually passing material wins and remeasure.

- P1 generated4 accepted texts, zero drops; answer lengths6144,6144,6144,5085
  (23517 answer tokens). First three reached the configured cap; fourth ends
  with the worked answer42. This is a numerical reasoning-text probe, not a
  final-answer accuracy or population-level text-quality verdict. Prompt and
  decoded-output evidence is `p1_text_review.json`; original token cache is
  `../eval_readiness/precision_reasoning/text.jsonl`. Model-card sampling is
  temperature1/top_p0.95/top_k20, seeds1000–1003, chat thinking enabled.

- Initial P1 calibration passed all critical/advisory checks. Baseline vs
  BF16all: ΔNLL+0.00008, SE0.00019, KLx1000=0.47929, 23517positions;
  traced decode1024positions also passes. Reference vs HF6144positions:
  KLx1000=0.29102, top1agreement99.756%. Exact archived verdict:
  `p1_calibration_precision_reasoning.json`. No loss waiver is needed.
- After the calibration process exited0 and closed the mesh, applied the two
  fixed-policy patches and formatted them. New observations enforce actual
  activation/residual, norm, recurrent, sampler and KV-layout contracts.
  Started bounded two-layer KV4 dtype/nonaligned smoke; old calibration is
  preserved, and final P1 will be rerun on current source using cached texts.

- KV4 reduced smoke first failed because its diagnostic preparation omitted
  the normal128-token prefill trace. Added128 to the diagnostic warmup list;
  `kv4_dtype_smoke_retry.json` passes32/33-prefill and traced decode. This was
  a harness prerequisite error, not a KV dtype rejection. Both logs retained.
- `refresh3_bfp8_baseline.json` passes strict runtime propagation including
  actual residual/activation boundaries,48 recurrent states,16 K/V layouts
  and sampler tensors. AIME100%=100/100; natural97.8516%=501/512 top1,
  all top5/top100100%. Second fresh process and coarse matrix queued serially.

- Refreshed serial baselines3/4 both pass strict runtime propagation. New
  combined traced rates: 30.891737, 30.904718, 30.865728, 30.883792t/s/u.
  Median30.887764; spread0.038990. Use this current-source spread
  for ranking. Original receipt retained as `baseline_spread_initial.json`.

- Canonical `bfp8_accuracy` passes all references and strict propagation,
  traced teacher-forcing 30.459304t/s/u. AIME top1=99/100 and natural
  top1=503/512; all top5/top100100%. These small accuracy differences do not
  establish an advantage; the candidate is slower than baseline.

- The original `baseline_tf_1` used2048-token allocation and a cold initial
  TTFT, whereas ranking uses prepared262144-token allocation. The runtime
  selects SDPA K-chunk128 for capacity<=64pages and512 for the full allocation
  (`optimized_decoder._decode_attention_chunk`, TP4 policy). Keep these
  regimes separate; do not infer an optimization from the original99/100
  versus refreshed100/100 AIME top1 or compare their TTFTs directly.

### MLP BFP8 LoFi full-model result

`full_mlp8_lofi.json` and its strict propagation receipt pass. AIME teacher forcing is 99/100 top-1; natural-text teacher forcing is 495/512 top-1, with 100% top-5/top-100 on both. Weighted traced teacher-forcing throughput is 30.87297527 t/s/u; this does not beat the refreshed baseline beyond its 0.03898973 t/s/u noise spread. It is not a demonstrated performance gain. P1 remains unmeasured on current source.

### Attention BFP8 HiFi2

`full_attention8_hifi2.json`: pass, weighted traced TF 32.93696146 t/s/u. Strict actual-runtime propagation passed. P1 remains required before eligibility.

### Attention BFP8 LoFi comparison

`full_attention8_lofi.json` passes prefill/TF gates and strict runtime propagation. AIME TF top-1/5/100 is 100%; natural TF top-1 is 499/512 (97.4609375%), top-5/top-100 100%. Weighted traced TF is 32.97506053 t/s/u versus 32.93696146 for attention HiFi2. Their 0.03809907 t/s/u difference is below the established 0.03898973 noise spread: a tie, not a proven LoFi gain. Both remain P1-pending.

### Rejected broad BFP4 MLP + LoFi

`full_mlp4_lofi.json` executes correctly and passes strict runtime policy/trace propagation at 34.13107789 traced TF t/s/u, but fails the retained top-100=100% gate on natural teacher forcing (511/512). AIME TF top-1=98%, top-5/top-100=100%; natural TF top-1=471/512 (91.9921875%), top-5=504/512 (98.4375%), top-100=511/512 (99.8046875%). Numerical rejection is due specifically to top-100, not a runtime blocker or failed top-1/top-5. Inner-layer gate/up-only BFP4 restores down projections to baseline for the next full-model run.

### BFP4 gate/up + LoFi passes short-reference gates

`full_gateup4_lofi.json` passes all full-model prefill/TF thresholds and strict runtime propagation. Weighted traced TF is 33.10927104 t/s/u. AIME TF top-1/5/100=98/100/100%; natural TF top-1=480/512 (93.75%), top-5=509/512 (99.4140625%), top-100=512/512. This restores down projections to BFP8; its P1 reasoning gate remains unmeasured and mandatory before keeping BFP4. No selection is made from this short-reference pass.

### Broad BFP4 MLP HiFi2 comparison

`full_mlp4_hifi2.json` passes runtime propagation but repeats exactly the LoFi prefill/TF accuracy scores, including natural TF top-100=511/512. It is rejected by the same retained gate. Its weighted traced TF is 32.68854688 t/s/u, materially slower than BFP4+LoFi at 34.13107789. Higher compute fidelity did not repair this measured BFP4 weight-quantization loss.

### BFP4 gate/up HiFi2 comparison

`full_gateup4_hifi2.json` passes all full-model prefill/TF thresholds with exactly the measured accuracy of gate/up LoFi, but reaches only 32.09734909 traced TF t/s/u versus 33.10927104. LoFi is the demonstrated performance candidate for this BFP4 group; neither is selected until P1. Both material MLP BFP4 groups now have full-model LoFi and HiFi2 comparisons on real model weights.

### BFP8 head HiFi2

`full_head8_hifi2.json` passes all full-model gates and strict propagation at 31.53321056 traced TF t/s/u. AIME TF all metrics=100%; natural TF top-1=500/512 (97.65625%), top-5/top-100=100%. This is a >noise gain over baseline, and a candidate for combination with passing attention-fidelity improvements. P1 remains pending.

### BFP8 head LoFi and combination plan

`full_head8_lofi.json` passes full-model gates and propagation at 31.53136060 traced TF t/s/u, versus head HiFi2 31.53321056: well inside the 0.03898973 baseline noise band. Both have AIME TF 100/100/100% and natural TF 97.65625/100/100%. Prepared (not yet evaluated) `configs/attn8_hifi2_head8_hifi2.json` combines the material attention and head gains, preferring HiFi2 over tied LoFi. No combined performance or accuracy is inferred without a full-model run and P1.

### BFP8 CCL rejected for measured throughput

`full_ccl8.json` passes prefill/TF and actual-runtime propagation gates. AIME TF metrics=100%; natural TF top-1=497/512 (97.0703125%), top-5/top-100=100%. Weighted traced TF is 30.26589479 t/s/u, materially below baseline median 30.88776441. The exact scoped BFP8 transfer policy is therefore not a measured speed win; P1 is unnecessary for this slower unselected policy. No inferred kernel blocker is used to reject it.

### Preserve baseline full-attention fidelity in the combined candidate

Prepared `gdn_attn8_hifi2_head8_hifi2`: GDN attention input/output use BFP8+HiFi2; full-attention projections retain baseline BFP8+LoFi; head uses BFP8+HiFi2. The broad attention HiFi2 trial had also raised full-attention fidelity from baseline LoFi. Both broad attention trials passed full-model gates and tied in timing, so the new composite tests the useful GDN reduction while preserving the established full-attention LoFi path. The earlier broad `attn8_hifi2_head8_hifi2` config remains an unexecuted alternative, not measured evidence. No combined result is claimed yet.

### MLP projection activation BFP8

`full_mlp_activation8.json` passes numerical/propagation gates at 30.79699290 traced TF t/s/u. AIME TF all metrics=100%; natural TF top-1=502/512 (98.046875%), top-5/top-100=100%. Throughput is 0.09077151 t/s/u below baseline median, beyond its 0.03898973 spread. It is not kept as a performance optimization. General activation/residual boundaries remain BF16; the tested field is specifically `mlp_projection_activation_dtype`.

### BF16 KV control

`full_kv_bf16_control.json` passes full-model accuracy and strict propagation at 30.87531684 traced TF t/s/u, inside the baseline noise band. AIME TF all metrics=100%; natural TF top-1=500/512 (97.65625%), top-5/top-100=100%. The full 262144-token cache allocation and complete preparation pass; capacity.py records actual BF16 cache shape and prepared allocator memory in `doc/context_contract.json`. No speed advantage over the existing BFP8 cache is established.

### BFP4 KV and coarse-matrix completion

`full_kv4.json` passes numerical/strict propagation checks at 30.88183400 traced TF t/s/u, within baseline noise and not a speed gain. AIME TF top-1/5/100=99/100/100%; natural TF=485/512 (94.7265625%) top-1, 510/512 (99.609375%) top-5, 512/512 top-100. The BFP4 cache fits and prepares the full 262144-token allocation; actual memory is recorded in the context contract. BFP8 remains the safer tied cache choice, so the combined candidate retains it. The 17-policy coarse matrix completed without device failures/resets. Next full-model runs isolate inner-layer BFP4 down projections at LoFi, then combine GDN BFP8+HiFi2 attention and BFP8+HiFi2 head while preserving full-attention LoFi.

### Isolated BFP4 down projections + LoFi

`full_down4_lofi.json` passes full-model gates and strict propagation at 31.82973877 traced TF t/s/u. AIME TF metrics=100%; natural TF top-1=486/512 (94.921875%), top-5=511/512 (99.8046875%), top-100=512/512. Inner-layer gate/up weights/fidelities remain at baseline; only down weights/fidelity change, with first/last layer exceptions preserved. A HiFi2 comparison and P1 are needed before any BFP4 down selection.

### Combined BFP8 policy passes full-model gates

`full_gdn_attn8_hifi2_head8_hifi2.json` passes strict propagation and full-model gates at 33.72455305 traced TF t/s/u (9.1842% above the baseline median), the fastest accuracy-passing evaluated policy at this point. AIME TF top-1/5/100=99/100/100%; natural TF top-1=503/512 (98.2421875%), top-5/top-100=100%. No P1 eligibility is claimed yet. Next serial queue runs isolated down4 HiFi2, then `p1_primary` with the strongest fitting BF16 reference, baseline, combined BFP8, gate/up4 LoFi and down4 LoFi. Intermediate P1 skips the HF slice; the final selected P1 will run the HF slice and be the last P1 receipt.

### Isolated down4 HiFi2 and P1 launch

`full_down4_hifi2.json` passes all full-model gates and strict propagation with identical measured accuracy to down4 LoFi, but slower traced TF: 31.45713940 vs 31.82973877 t/s/u. The full-model matrix now contains 20 evaluated policies. `p1_primary.command.json` is running serially: strongest fitting BF16 reference, refreshed BFP8 baseline, combined BFP8 policy, gate/up4 LoFi and down4 LoFi. All candidates allocate/prepare the advertised 262144-token context. The cached four texts preserve the original sampled reasoning evidence; source/config hashes force fresh scores for changed model code.

### Final-check harness audit before hardware acceptance

The CPU audit corrected native MeshTraceId serialization after native equality checks in `check_kv_short.py`, validated control batch/source/layer inventory/non-KV policy/input IDs, verified per-user tensor shapes, and made numerical failures exit nonzero. B1 warms the required P128 trace branch; B32 warms the tested 32/33 branches. `run_selected_checks.py` now admits max-context execution on either KV dtype or layout change and validates recovery job names. `run_selected_kv_checks.py` builds a BF16-cache control from the exact eventual selected policy and serializes full-64-layer B1/B32 controls/comparisons. CPU fixtures passed; no hardware acceptance is inferred from them. These harness edits do not change TT runtime source or invalidate current P1 scores.

During `p1_primary`, direct read-only inspection of completed score caches found both baseline and `gdn_attn8_hifi2_head8_hifi2` within the runner numerical bounds over 23517 answer positions. This is provisional until the bounded runner finishes and its receipt is archived; no selected artifact was written.

### P1 gate/up result motivates combined-policy measurements

Read-only current score inspection found `gateup4_lofi` within noise against the full-context BF16 reference: ΔNLL approximately +0.0022 nats/token (SE 0.0005; bound0.004), KL×1000 approximately2.8 (limit5), 23517 answer positions. The final primary receipt remains pending. Prepared combined GDN-attention/head HiFi2 plus inner-layer gate/up, down, or both MLP BFP4 policies, each with LoFi and HiFi2 variants. These JSONs alone are not measurements. Combination layer exceptions restore only the changed MLP groups at layers0/63, preserving the independently measured attention/head policy. Full-model and P1 combination evidence is still required before selection.

### Combined gate/up BFP4 LoFi full-model pass

`full_gdn_attn8_head8_gateup4_lofi.json` passes all full-model accuracy/trace/policy gates at 36.4212279012 traced teacher-forcing t/s/u. Report minima: top1=0.94140625, top5=0.99609375, top100=1.0. This is a measured combined-policy gain over both baseline and the BFP8 attention/head combination. Its exact combined P1 check is pending; no final selection yet.

### P1 lifecycle failure and AutoFix

`p1_primary` returned selected-candidate pass but recorded `down4_lofi` scoring error: TRACE allocator exhausted on the fifth candidate. The baseline, attention/head combination, and gate/up BFP4 all pass measured P1; down4 is unscored, not numerically rejected. Isolated AutoDebug diagnosed missing `close()` because the upstream runner only calls `teardown()`. See `AUTODEBUG.md`, `AUTOFIX.md`, and the preserved primary log/JSON. A stage-local adapter now synchronizes/closes/asserts empty TRACE allocation and the driver rejects any candidate runtime error. Host validation passed; hardware validation is pending. No reset or device process termination occurred. Runtime source and existing numerical score caches remain unchanged.

### Combined both-MLP BFP4 LoFi passes short-reference gates

`full_gdn_attn8_head8_mlp4_lofi.json` passes at 37.6456317710 traced TF t/s/u, minima top1=0.91796875, top5=0.98828125, top100=1.0. Unlike standalone `mlp4_lofi`, this measured attention/head combination restores the missing natural top-100 hit. No numeric result is transferred between policies: the combined policy's P1 is mandatory and still pending. Its exact both-MLP HiFi2 comparison will also be measured before final selection.

### Combined down-only BFP4 LoFi

`full_gdn_attn8_head8_down4_lofi.json` passes all numerical/runtime gates at34.8458504398 traced TF t/s/u. Report minima top1=0.9453125, top5=0.998046875, top100=1. It is slower than gate/up-only36.4212279012 and both-MLP37.6456317710 combinations; P1 remains pending.

### Exact combined gate/up fidelity comparison

`full_gdn_attn8_head8_gateup4_hifi2.json` passes at35.2031845989 traced TF t/s/u with exactly the LoFi combination's reported top1/top5/top100 minima(0.94140625/0.99609375/1). Its LoFi counterpart is faster at36.4212279012, well beyond the0.0389897317 noise spread. The both-MLP combination's exact HiFi2 comparison is queued, followed serially by lifecycle hardware stress and combined P1.

### User stage-cap instruction

The user reached the stage time/token cap and directed: keep the best version that passes required gates; run only remaining required gates once; record untried ideas as hand-off notes; complete, allowing pass-with-carried for non-blocking items. No additional search will be admitted. The in-flight serial queue completes the exact both-MLP HiFi2 comparison, required P1 cleanup validation, and combined-candidate P1 gate. Selection then uses the fastest measured eligible policy. Remaining work is selected-default construction, token-out/32K/non-aligned/trace-allocation tracking/resumed-prefill/qualitative acceptance, final selected P1 with HF control, plots/docs, independent stage review and local commits. Optional unchanged-KV B1/B32 controls will not be run under the cap; selected policies retain existing BFP8 KV dtype/layout.

### Carried to next stage — untried ideas under the cap

Prepared but unevaluated policies are not claimed as wins: isolated GDN-output fidelity variants, head8 HiFi4, broad-attention-plus-head HiFi2, and the exact combined down-only BFP4 HiFi2 variant. First/last-layer BFP4 extension and additional activation/CCL combinations remain untried. The BFP4 KV candidate was not selected and its full B1/B32/max-context qualification remains unrun. Optional fresh matched BF16-cache controls for the unchanged selected BFP8 cache can be run with `run_selected_kv_checks.py`; they are not evidence of a changed-cache acceptance gate. None of these optional experiments may replace or waive required correctness gates.

The stop file now prevents admission after the in-flight fidelity measurement. No running device job is interrupted. Remaining gates will be launched explicitly; P1 is limited to policies that can still win selection. The slower down-only candidates need no new P1 scoring to select a faster passing policy, and their unmeasured long-answer status remains explicit. Cleanup itself still receives its required hardware validation.

### AutoFix device lifecycle validation passed

`p1_lifecycle.json` passes all five same-mesh representative-layer ownership cycles: TRACE bytes0 →1130496 →0 after close →0 after GC, with actual262144-context allocation for layers0/3. Exact bounded command is `p1_lifecycle.command.json`; exit0. Full-model contender P1 with HF control is running through the repaired adapter.

### Final precision selected after P1/HF pass

`p1_final_contenders_precision_reasoning.json` passes all critical/advisory checks, with no errors. Selected `gdn_attn8_head8_mlp4_lofi`: ΔNLL0.00248 (SE0.00056), KL×1000=3.85138, 23517 answer positions; traced decode top1=0.93555 versus reference0.93457 over1024 positions. The highest-precision full-context reference passes the HF control over6144 positions. Gate/up-only also passes P1 but is slower. Both models release22937600 TRACE bytes to0.

Wrote exact winning policy to `selected_precision_config.json`, now consumed by default Model/build_generator construction; regenerated25-row JSON/CSV, selected-red Pareto plots, and selected context contract. The policy uses inner-layer MLP BFP4+LoFi (first/last MLP exceptions BFP8), GDN attention BFP8+HiFi2, full-attention projections BFP8+LoFi, head BFP8+HiFi2, BF16 activation/residual/CCL/logits/sampling, and unchanged BFP8 KV. Default-path required acceptance checks are running once under the user cap. This is the last P1 run, naming the shipped config.

### Required stage-gate metadata correction

The first stage gate passed selected P1 but rejected the inherited `doc/eval_readiness/limitations.json` schema (`limitations` instead of required `items`). Converted the existing story-opening observation to the documented schema, preserved its content and `original_model: same`, added its probe/check/reason fields, and corrected evidence paths to model-relative existing files. No P1 waiver was added and no numerical gate changed. Original failure is retained as `stage_gate_initial.*`; the corrected gate is checked once afterward.

### Final required device checks recovered after interruption

All four jobs completed serially on2026-11-14 with exit0; no device job remained on resume2026-11-16, so no rerun was needed. `post_selection_token_out.json` passes at75.38076722994447 ms TTFT and37.81770435692664 token-out t/s/u, median three warmed127-step resident windows, B1/P128/G128/full262144 allocation. Same benchmark as stage7 (86.85081405565143 ms /30.99974964555567 t/s/u), separate from selection TF37.6456317710. Later serving comparisons must use the token-out result. Trace IDs/capture counts/program cache/persistent addresses remain stable, resident windows contain zero token readbacks. Non-aligned lengths33,127,128,129,511,512,513,1025,32769 pass, including eager/trace feedback and changed pages/tokens.

`selected_trace_tracking.json` passes with allocation tracking and program-cache allocations included; the ordinary-run generic unsafe-allocation warning is controlled by this dedicated audit. `selected_prefill_resume.json` passes all four default-threshold cases and required negative controls. `selected_qualitative.command.json` exits0; all12 TT and12 HF text outputs were read in full. The story opening matches HF exactly; all outputs remain relevant thinking under the common128-token cap. `qualitative_verdict.json` records this observational scope without population or final-answer claims; the inherited limitation now links current evidence. Exact selected policy equality in normal Model and build_generator paths is recorded in `selected_default_propagation.json`.

CPU inventory command: `python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/check_stage_artifacts.py --require-complete --output models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/stage_artifacts_check.json`:149 required checks pass;7 optional checks remain not_evaluated (156 inventory entries total). No new precision searches or optional hardware runs admitted under the cap. Independent stage review and local commits are next.

### Repository hooks and final source identity

Repository `pre-commit run` applied trailing-whitespace cleanup to historical `baseline_full_1.log`, Black23 formatting to `report.py`, and isort local-import ordering to `p1_runner.py`/`check_p1_lifecycle.py`. No TT model/runtime source, dtype config, scoring calculation, or cleanup behavior changed. P1_RUNNER_IDENTITY in executed logs correctly describes the pre-format adapter hash; the shipped adapter differs only in readiness/ttnn import ordering. Focused CPU validation after formatting: `python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/test_p1_runner.py -v`,11 tests pass in0.014s. Full hooks rerun passes (`precommit.log`); no hardware rerun justified by formatting.

### Independent review documentation corrections

The reviewer confirmed the inventory contains149 passing required checks and7 optional not_evaluated entries, not156 passes. Corrected the final README/work-log count and a stale sentence saying the default benchmark was queued. Completed raw measurements are unchanged; these are documentation corrections only.

### Stage review and completion

Fresh independent xhigh reviewer `/root/stage8_review1` returned **clean-pass** in `stage_review_round1.md`, with no required work or additional hardware runs. It rederived TF/token-out/P1 results and inspected source, runtime consumption and actual TT/HF outputs. The two documentation corrections are verified. All required gates pass, the source/policy is frozen, and `report.py --selected gdn_attn8_head8_mlp4_lofi --complete` finalizes the25-row report and Pareto artifacts. Optional seven unevaluated inventory entries are the five prepared alternatives and unchanged-KV B1/B32 controls already carried above. No vLLM work and no push. Local stage-owned checkpoint commits follow.

### Carried to next stage — reviewer integration note

Repeated generator construction must synchronize and call `close()`; the upstream P1 optional `teardown()` alone does not release this generator's traces. Retain the validated stage-local adapter for P1 until an upstream lifecycle change is separately handled. Its success does not prove recovery after a mid-capture allocation failure; it aborts on residual allocations. Serving construction must consume the selected artifact and unchanged262144 context through the normal builder, and compare performance against37.8177043569 token-out t/s/u rather than37.6456317710 teacher-forcing t/s/u. No serving implementation was added here.

### Local checkpoint SHAs

Repository `/home/ttuser/qwen36-27b/tt-metal`, branch `main`: implementation, selected policy and complete evidence committed as `6f71a42a58df13d6bc005b42902f232b15a2f1ae` (`Select validated Qwen3.6-27B datatype policy`). Repository pre-commit hooks pass, including formatting, file size and API checks. No other repository was modified and nothing was pushed. The only `git diff --cached --check` warnings were literal blank context lines inside the two archived `.patch` evidence files, preserved intentionally under the repository's patch whitespace exclusion.

`commit_receipt.json` records this SHA and final gate/performance status; a subsequent local documentation-only receipt commit records this log. Its SHA is obtained with `git log -1 --format=%H -- models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/commit_receipt.json`. No runtime, policy or measured evidence changed after the independent clean-pass.
