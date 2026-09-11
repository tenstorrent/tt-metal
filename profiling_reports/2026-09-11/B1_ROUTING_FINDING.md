# B1 routing follow-up: real skew exists, matched timing attribution remains open

The recovered real routing capture contains substantial expert and destination-device skew. It supports testing routing sensitivity, but cannot establish why the historical B1 timing layers differ: the routing and timing captures do not have matched inputs, context, revision, or topology.

## Provenance and validation

Source: `/data/kmabee/mistral4_caches/expert_routing_mistral4.safetensors`, with adjacent `expert_routing_mistral4.PROVENANCE.md`. That document records September 10 capture on `bh-glx-120-b03u02`, a local uncommitted capture patch, the real golden 56,320-token input, and ragged chunk index 4 (zero-based token interval [10,240, 15,360)). This is the first fully populated 5,120-token chunk. The rejected chunk-0 capture is 80% padding and was not used.

The standard-library-only analyzer verifies 35 layers (0–34), each int32 `[8,2560]`, exactly 20,480 routed token-expert assignments, valid IDs 0–127, and four distinct experts for every consecutive four-entry token group. Layer 35 is absent because the capture ran with a KV-only final layer. File SHA-256 and complete per-layer counts are in `b1_routing_evidence.json`.

The capture patch itself is unavailable locally. The loader documents reshaping into `[8,640,4]`; source→destination matrices consequently **assume source-major ordering**. Expert totals and destination totals do not depend on that assumption. Those matrices count routed token-expert assignments, not measured network bytes or packets: dispatch uses sparse multicast and can share payload writes.

`ExpertMapping.create_dispatch_table` in `models/demos/deepseek_v3_d_p/tt/moe/init_helpers.py:233` verifies contiguous placement: 128 experts / 4 TP columns / 8 devices gives 4 experts/device; one PP column gives 16 experts/device. The script computes TP destination as `(expert % 32) // 4`, column as `expert // 32`, and PP destination as `expert // 16`. **PP counts are a counterfactual remapping of TP-captured selections**, not observed PP execution or proof PP would select identical experts.

## Findings

| Routing statistic | L18 | L23 |
|---|---:|---:|
| Busiest expert / mean expert assignments | 11.4125× | 12.2625× |
| Busiest TP device / mean device assignments | 3.7625× | 3.6547× |
| Busiest TP column share | 42.9053% | 28.8037% |
| Counterfactual busiest PP device / mean | 1.7402× | 1.5254× |

L18 TP column totals are `[3701, 4382, 8787, 3610]`; a uniform column would receive 5,120 assignments. Counterfactual PP device totals are `[2372, 1329, 2020, 2362, 4455, 4332, 1949, 1661]` for L18 versus `[1936, 2010, 3905, 1884, 2256, 2590, 1996, 3903]` for L23.

Across the 35 captured layers, busiest-expert/mean ranges 2.1625×–24.5563×, TP busiest-device/mean 1.4219×–6.5781×, and counterfactual PP busiest-device/mean 1.1633×–2.2402×. Each minimum is L2 and maximum L9. L18 has the highest TP column share (42.9053%).

**The busiest-expert metric is actually larger for L23 than L18**, despite historical L18 having greater operation cost. That is a useful reason to test multiple load measures rather than assume expert skew alone explains latency; it is not a matched counterexample because these captures differ.

No timing correlation coefficient is reported. September 8 timing uses consecutive 5,120-token runner chunks and aggregates multiple contexts; this capture uses a September 10 ragged golden test and only one context. A layer-index join would manufacture a false appearance of paired observations.

## Discriminating next experiment

Collect routing and per-device Dispatch/Combine/FFN timings from the same execution, retaining layer, chunk/token range, valid-token count, input hash, revision, actual expert layout, source ordering, device mapping, router mode, and tracing mode. Cover at least L2, L9, L18, and L23, then all MoE layers. Measure destination assignments and sparse-multicast destination sets separately from actual transfer bytes. Compare hot-device timing with the mapped load within a fixed stage and context.

For a controlled replay, hold layer implementation, tensor shape, topology, payload values, capacity, dtype, and kernel configuration fixed; vary only valid routing indices with matching dispatch metadata. Compare captured L18/L23 patterns, a balanced control, and a deliberately skewed control, with repeated runs and correctness checks. A routing replay changes the model computation and establishes kernel sensitivity only; a production expert remapping must preserve expert weights, identities, and outputs and receive separate end-to-end validation.

## Reproduction

```bash
python3 profiling_reports/2026-09-11/analyze_routing_b1.py \
  --routing /data/kmabee/mistral4_caches/expert_routing_mistral4.safetensors \
  --output-dir profiling_reports/2026-09-11
```

Executed successfully without accelerator access or TTNN imports. Outputs: `b1_routing_evidence.json` and `b1_routing_per_layer.csv`. No hardware experiment, external post, or causal performance claim was made.
