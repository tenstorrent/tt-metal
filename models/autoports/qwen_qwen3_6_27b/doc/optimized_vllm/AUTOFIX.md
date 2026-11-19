# AutoFix: unaligned continuation cost

Failing performance contract: valid nonaligned chunked requests must remain supported without avoidable repeated full-layer work. Source diagnosis in AUTODEBUG.md confirms the current fallback executes each full-attention layer including its MLP per token until the next512 boundary. No current correctness failure is claimed. Baseline512 mixed long-prefill meets1000ms ITL (966.6ms); this aligned probe does not exercise the defect.

Hypotheses in priority order:

1. Existing tensor-start Q32/K512 SDPA supports tile-aligned starts inside a key block. Peel only to32, then retain512 maximum internal chunks and unchanged selected arithmetic. Discriminator: probe_sdpa_tile_start.py against dense causal reference built from quantized KV, including near-capacity and changed runtime buffers. If passing, compare reduced real layers[0,3,4] against the pinned old implementation with identical public boundaries; preserve prefix/other-slot cache and downstream GDN states.
2. If the primitive cannot express that contract, keep serial attention/cache update and vectorize token-independent finish/MLP. Discriminator: same real-model logits/state checks and serving mixed traffic. Not yet tried.
3. Secondary request-boundary overhead: repeated full-table max reductions and diagnostic page diffs when telemetry is disabled. Preserve reload semantics and runtime inputs; benchmark in the same serving configuration.

No model source changed yet. Independent CPU-only subagents authored the primitive and reduced continuation tests in distinct files. Device jobs remain serialized behind the baseline sweep; no profiling or resets.

## Discriminating results and adapted candidate

Primitive gate passes52cases, minimum whole-outputPCC0.999991917, minimum head-rowPCC0.999985884. Pinned SDPA tensor implementation supports the tested tile-aligned runtime offsets with unchangedQ32/K512; no shared kernel or plugin change.

Initial real-model peel candidate failed2/9cases. At start33, the first top1 difference was at logical token22, before the new vector segment. GDN0 recurrent state also changed: extra global internal boundaries changed GDN arithmetic. Removed the new global peel and restored the original512-boundary interval partition. The adapted candidate vectorizes full-attention only when the existing interval starts on a32-token cache tile. Arbitrary non32-aligned starts retain the accepted exact per-token path. This preserves valid prompts rather than restricting their lengths. The start33 rerun is now exact/passing.

The remaining start2016 diagnostic miss is a reference tie, not an attention failure: at token15 the old logits for IDs12 and220 are both6.03125. The candidate chooses220 (6.0625 versus6.0). The old comparison used torch.argmax, which arbitrarily requires the lowest tied index12. Existing full-model greedy contract explicitly accepts any equal maximum. Corrected the diagnostic to require candidate argmax membership in the exact reference maximum set, with no epsilon; retain strict-lowest-index agreement and every mismatch score in JSON. The numerical gates remain PCC>=.995 and top1/top5>=.98. A host positive/negative test covers exact ties and a strictly lower value. This is a test-definition correction backed by raw scores, not a precision waiver. See tile_continuation.json and tile_continuation_v2.json.

## Multi-row boundary candidate rejected
B3 uniform start2016/end2048 on real layers0/3/4 produced top1membership0.9791667 below0.98 (PCC0.9999918, top5=1, cacheguards pass); these were real lower-reference scores, not reference ties. The tile-offset vector optimization is restricted to single-row prefill. Existing full-batch512alignment arithmetic and all logical lengths remain supported. This is not a batched-prefill adoption. Recheck: tile_uniform_final.json must reproduce the old path exactly.
