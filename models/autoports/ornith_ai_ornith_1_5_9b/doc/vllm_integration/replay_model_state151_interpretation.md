# Aligned replay-state interpretation

The [aligned151 result](replay_model_state151.json) passes all six structural/reconstruction checks. Host and device replay agree exactly for all48 captured tensors; inactive rows agree across all three comparisons. Prefix, consumed position151, physical mapping, finite values and shapes agree. The [log](replay_model_state151.log) records completion at line155 and device closure at160. Diagnostic preparation added28 programs before capture (847→875), then remained875 on its second pass; the rerun completes with the tracker enabled and program-cache checks retained. The first failed run remains under `replay_model_state151_before_slice_warmup/`.

The active uninterrupted-decode state differs from both identical replay states:

| State | Relative L2 across ranks | Maximum absolute difference across ranks |
| --- | --- | --- |
| Convolution history |0.128–0.185%|0.125–0.25|
| Recurrent state |1.240–1.620%|0.193–0.420|
| KV key, complete151-token prefix |0.673–0.770%|0.125–0.1875|
| KV value, complete151-token prefix |0.676–0.830%|0.03125–0.046875|

CPU inspection of the saved tensors sharpens the attribution: **every KV row0–128 is bitwise equal on every rank; the first difference is row129, and all22 decoded suffix rows differ.** Suffix-only relative L2 is1.787–2.089% for keys and1.763–2.435% for values; the whole-prefix figures above include129 exact rows. The JSON companion records each rank and the half-open ranges `[0,129)` and `[129,151)`. This analysis loads the saved CPU artifact with `torch.load(..., map_location="cpu", weights_only=True)` and compares those token-axis slices; it performs no model execution.

This demonstrates a computation-route difference: uninterrupted execution prefills129 tokens then decodes22; replay prefills the same151-token history. The model already has distinct prefill recurrence (`multichip_decoder.py::linear_attention`, chunked gated delta rule) and pooled decode recurrence (`pooled_linear_attention_decode_batch`, ordered recurrent path), so this is not an identical arithmetic graph. Correct alignment, exact original-prefix KV, equal sampler state, repeatable replay reconstruction and the separate [50-output greedy control](replay_greedy_follow50.json) provide no concrete remaining adapter ownership/history/position bug in this stimulus. No new plugin or adapter change is justified by these data.

**Strict forced-preemption token parity remains failed in all three original modes.** The151-token run follows only23 outputs and its token equality does not replace the100-output serving check. The separate50-output control reproduces greedy index49 (53 versus498). These measurements do not prove harmless rounding or an acceptable tolerance, nor exclude a systematic canonical kernel discrepancy. They cover layers0/3, B3, one active row, a fixed larger physical pool, zero inactive rows and only their first KV pages; they do not exercise live scheduler page replacement or full32 layers. No concrete unresolved implementation defect is identified; the precise canonical operator/precision source remains unisolated. Continue independent full-model acceptance/performance with this limitation explicit. If exact recomputation parity is required, the next discriminating check is matched-input/reference comparison at the first layer0 projection/conv/gate/recurrent difference, not another serving-interface change.
