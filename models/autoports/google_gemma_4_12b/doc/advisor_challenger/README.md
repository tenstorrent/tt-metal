# Advisor-challenger contribution: Gemma-4 12B decode batch 32

## Full-model estimate: 58,520.1 us -> 57,211.9 us, +/-55.5 us

The measured `$shard-advise` contribution is **1,308.2 us per estimated full decoder pass (2.24%)**. This is the sum of independently measured per-layer deltas scaled by the model-config counts: 40 sliding-attention and 8 full-attention layers. The uncertainty band is the two frozen-incumbent repeat spreads scaled by those counts and summed linearly.

| Layer kind | Frozen incumbent | Shipped candidate | Delta/layer | Layers | Delta/model |
|---|---:|---:|---:|---:|---:|
| sliding_attention | 1.241523 ms | 1.218789 ms | 22.734 us | 40 | 909.4 us |
| full_attention | 1.355874 ms | 1.306017 ms | 49.858 us | 8 | 398.9 us |

Every candidate repeat beats every corresponding frozen-incumbent repeat. Fresh-process confirmation produced 1.217706 ms sliding and 1.307295 ms full medians, again with complete repeat separation.

## Shipped placement

The decoder now keeps Q, K, and V in interleaved L1 across the per-head norms. Q is converted to the required height-sharded layout only at SDPA; K and V are converted only at cache update. The MLP fused multiply writes directly into the down-projection input layout. Full-attention layers additionally keep the concat/output-projection handoff in L1; this knob is intentionally off for sliding attention because its five-repeat distribution overlaps the control.

All losing or experimental knobs remain explicit. `GEMMA4_12B_ADVISOR_O_L1` defaults off for sliding layers and on for full layers. Capture-only graph declarations default off. The earlier rope-cache relayout experiments are retained as rejected evidence: checking the actual full-model path showed that the frozen control already uses row-major 2D caches, so those trials are not credited.

## Screening record

The original `reconcile.py --incumbent` ranking was followed across kinds using `advisor_removes_per_model_us`. On sliding attention, K, the extended Q/rotary/SDPA chain, V, and the MLP handoff each won independently with complete separation. The grouped concat/output-projection candidate overlapped and was rejected. On full attention, the individually resolvable Q and K chains won; the below-floor V/rotary/MLP chains were applied together first and won; the below-floor concat/output-projection pair was grouped and won. The product of the per-kind winners was then measured directly.

Each kept candidate has a bounded one-replay `tt-perf-report` CSV. Candidate timings use the unmodified protocol in `scripts/harness_template.py`, in a fresh process per configuration. The reconciliation JSON accounting and ranking are emitted by `scripts/reconcile.py`; `annotate_reconciliation.py` only joins measured screening fields onto that generated output and refuses non-generated input.

Both layer-kind captures use the model-filled copy of `scripts/capture_template.py` in this directory. They were recaptured at batch 32 after freezing the incumbent, with every candidate placement knob forced off; the reports record the executed incumbent policy, weight dtypes, advisor pin, template path, and capture timestamp. Reconciliation above was regenerated from those final reports with `--incumbent`.

The second profile/ranking pass is recorded as `reranked_*_iteration2.json`. It shows the MLP boundary eliminated and the remaining Q/K/V conversions reduced to required interleaved-to-sharded contract transitions. Attempting to leave norm inputs height-sharded raises `Height sharded inputs are not supported`; cache update and SDPA consume the required height-sharded forms. No further conversion removal is booked.

## Correctness

The shipped product passed real-checkpoint oracles for both layer kinds:

- sliding layer 0: prefill PCC 0.999613, decode PCC 0.999871;
- full layer 5: prefill PCC 0.999658, decode PCC 0.999659.

Both use actual `google/gemma-4-12B` checkpoint tensors and compare against Hugging Face `Gemma4TextDecoderLayer`. Isolated real-weight runs also covered the Q extension and full-only output-chain candidate. No dtype or fidelity changed.

## Accounting that is not contribution

The incumbent profiles close at 100%. `advised_boundaries.us_advisor_agrees` is 3.824 us/layer sliding and 3.983 us/layer full; it is real time but contributes zero here because the advisor retains those boundaries.

`model_estimate.layer_handoff` reports a DRAM-entry/L1-exit mismatch worth 47.619 us across sliding layers and 8.428 us across full layers. It is outside the one-layer advised graph, so it is reported upstream and never screened or credited.

The raw incumbent op reports show both `EmbeddingsDeviceOperation` and `SdpaDecodeDeviceOperation` entering and leaving in `DEV_0_DRAM_INTERLEAVED`; their `dram_resident` reconciliation rows are therefore already-shipped agreement, not de-sharding candidates. The soft pairing is corrected in the annotations without changing generated accounting.

The profile-only `ReshapeView` share is 47.650 us sliding and 87.233 us full. This is resolved from the authoritative IR: for example, the norm reshapes explicitly change `#ttnn_layout19` (height-sharded L1) to `#ttnn_layout21` (interleaved L1), and the reverse reshapes change the interleaved norm result back to the head layout. The same transitions are present in the advisor's final IR, so these are hidden advisor-agreed layout changes, not absent advisor boundaries and not contribution candidates. They remain in the generated JSON's `unresolved` bucket because `reconcile.py` deliberately cannot infer output layouts from the reduced profile; the IR resolution is recorded here as required. The advised core count was never treated as a speed recommendation: no sweep was bounded at or below it, and the shipped compatible layouts use the existing exactly-dividing 32-core head geometry or consumer-required grids.
