# AutoFix: additional layer34 long-stream accuracy

Status: **AutoFix failed to meet the unchanged accuracy gate with the tested repairs.** All device jobs closed cleanly. No candidate is promoted and the fused stage remains incomplete. This is not a claim that an exact device solution is impossible.

The failing contract is prefill/output/contribution PCC at least0.995 for the actual fused path, including a fresh128-token prefix,8192 sequential decodes,33-token continuation and three bit-equal trace repeats with allocation/program-cache tracking. The additional representative layer34 exposes296 failing positions in selected source8ac733; original functional and priorV4 controls fail too. Their failures do not waive the HF reference or acceptance threshold.

## Diagnosis and discriminators

The installed inspection-only AutoDebug launcher produced [the diagnosis](AUTODEBUG_v5_layer34.md). Immediate producer readbacks in [the stream probe](../../../tests/v5_layer_stream_probe.py) avoid the old state-alias and overwritten-decay defects. Canonical CPU recurrence on identical captured TT inputs matches the local TT decay, contraction, delta, state update and core. Small discrepancies are already present in incoming effective BF16 state; no new local recurrence or trace corruption is demonstrated at those captures.

Ordered Q/Z projection makes actual projection, convolution and normalizedQ/K inputs exact at sampled positions128/129/130/137/181/7944. It does not alone repair the model. Accurate BF16 first-residual addition and ordered postnorm together match the actual CPU residual and postnorm exactly. The earlier postnorm comparison had conflated CPU residual arithmetic with actual device residual; the updated observer captures the input explicitly.

The [same-operand gated-norm probe](v5_gated_norm_same_operand_device.json) shows that native RMSNorm on BF16 or FP32 input still changes hundreds of BF16 normalized values. ExplicitFP32 square/sum/mean/rsqrt/multiply matches normalized and gamma-product BF16 values for allfive tested captures. It leaves one finalSiLU/product code difference at128. Integrated explicit normalization reduces failures but does not clear the gate.

The [ordered output-projection probe](v5_ordered_outproj_device.json) gives exact BF16 results on all2048 actual captured181 output columns for K512 andK1024, while native differs335 codes. Both complete-decoder controls repair181 but fail201 stream positions.

The worst captured position7944 has an independent routing discrepancy. The actual HF logits and actual TT logits both contain equal boundary scores for experts64,74,202. CPU chooses202/64; native generalized gate chooses64/74. The [native alternative probe](v5_router_tie_device.json) shows direct BF16/FP32 topk and FP32softmax/topk choose74/202 on both exact input fixtures. None matches the frozen oracle, and all are slower as complete routing adapters. Full-stream alternatives also fail; their improved minimum PCC remains below0.995.

[The source and permutation proof](AUTOFIX_v5_router_tie_semantics.md) establishes the precise limitation: the inspected native APIs do not expose Torch CPU's value-dependent nth_element tie behavior. Permuting only lower logits changes CPU tied membership among allthree possible subsets, so a fixed expert-index priority cannot reproduce it. This is not a proof that an exact device implementation is impossible. No host runtime fallback, reference alteration, threshold waiver, shared/native edit or later-stage work was introduced.

## Reproducible evidence

[Measured source comparison](v5_layer34_repair_comparison.json) records complete raw result hashes, source hashes, failed-row counts and minimum PCC. Every snapshot has an adjacent provenance file with an AST scope or union proof. Commands and native close evidence are retained in the correspondingly named logs and JSON provenance. Diagnostic single-repeat cached-oracle runs are explicitly ineligible for final acceptance; the original selected-source failing run used fresh HF and three repeats.

Specific controls and complete commands are documented in the adjacent AUTOFIX_v5_ordered_postnorm, ordered_qz, ordered_postnorm_bf16_gate, ordered_qz_postnorm_bf16_gate, combined_accurate_residual, accurate_residual, combined_accurate_residual_explicit_gate_norm, ordered_outproj, router_tie_probe and ordered_outproj_router_compositions memos. Original functional control changes only its class declaration for the snapshot loader and is never a runtime fallback.

## State and limits

Production fused_decoder.py remains source8ac733. Its original layer0/3 repeated, watcher, context262144 and profiler evidence remains historical evidence for that source, not a passing qualification for the expanded representative gate. No context capability was reduced. Diagnostic precision candidates add resident weights and have no full-context or performance qualification; none is promoted.

The8-hour device-time or70%-roofline Stop Rule has not been claimed. The current repair is driven by a hard accuracy failure, independently of optimization stopping. Independent review and accepted stage commits remain open.

## Completed final controls and regression

The full precision+ordered-output+BF16topk combination fails203 positions (minimum contributionPCC0.946055341978); the corresponding FP32softmax/topk combination fails200 (minimum0.930944200802). Both still fail7944 at0.986205319036. BF16topk repairs181, whereas softmax/topk changes the tied selection and fails181 again. These complete compatible combinations close the remaining isolated-candidate question; neither is an acceptable runtime selection.

A new **fresh HF, three-repeat** run of actual production source8ac733 reproduces888 failures across296 unique positions, minimum contributionPCC0.912541466536, with bit-equal repeats. It is retained as [the required representative regression](../stress_representative_34_seed154.json), with the full bounded command in adjacent provenance and [native log](../logs/stress_representative_34_seed154.log). The validator and aggregate checker now require this case. [Aggregate hard gates](../hard_gates.json) explicitly report FAIL; [checker log](../logs/hard_gates_layer34_failure.log) identifies the failed representative artifact. A CPU negative control also rejects the earlier known failing capture. The stress harness, HF reference and numerical thresholds were not changed.

Two device diagnostic probes were formatted after their initial measurements. Their measured sources are preserved, full AST equality is recorded in [formatting provenance](probe_formatting_provenance.json), and both were rerun successfully as [formatted gated norm](v5_gated_norm_same_operand_device_formatted.json) and [formatted ordered output projection](v5_ordered_outproj_device_formatted.json). No runtime source changed.

An exact dynamic tie-compatible device selection implementation remains unimplemented; none of the inspected existing configurations guarantees the frozen CPU partition behavior. Further work must address that behavior without host fallback or a reference/threshold waiver, then re-run the full quality, capacity, watcher, timing and profiler gates before promotion. Existing approximate output/state precision may require further work too; tied selection is a proven independent failure, not an assertion that it explains every failed row.
