# Full-prefill configuration decision

Select **paged Q64/K256 with HiFi2**, retaining Q128/K128 for the initial BF16 K/V chunk and BFP4 routed-expert gate/up and down weights. This configuration passes the maximum-context aggregate and endpoint criteria, and all 291 sampled rows exceed .995. Its warmed 4096-token host median is modestly lower than Q128/K128 HiFi2; overlapping timing ranges preclude a strong speedup claim.

This is an artifact-only summary of completed parent-owned runs. Exact values, artifact hashes, runtime hashes, gate provenance, and timing samples are in [long_prefill_config_results.json](long_prefill_config_results.json). No hardware or runtime edits were performed by this documentation task.

## Maximum-context accuracy

Layer 5, actual text-derived HF inputs, 262144 tokens, all causal K/V positions, and 291 sampled output rows; PCC threshold .995. Every completed accuracy run has the same input fixture hash `52d5a811ff0735478d3fff537be518fc10c098c812c8a34ffede705b0385352a` and BFP8 cache. Q/K below denotes paged attention chunks; the first chunk retains 128/128. Expert weights are BFP4/BFP4 unless shown otherwise.

| Configuration | Aggregate PCC | Original aggregate gate | Prefill 262142 | Prefill 262143 | Endpoint criterion | Diagnostic rows below .995 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| [128/128 LoFi, experts 4/4](validated_long_262144_layer5.json) | 0.991703279 | Fail | Not recorded | Not recorded | Not recorded | Not recorded |
| [128/128 LoFi, experts 4/8](actual_long_control_down8_layer5.json) | 0.992009353 | Fail | Not recorded | Not recorded | Not recorded | Not recorded |
| [128/128 LoFi, experts 8/4](actual_long_control_gate8_layer5.json) | 0.992018342 | Fail | Not recorded | Not recorded | Not recorded | Not recorded |
| [128/128 LoFi, experts 8/8](actual_long_control_both8_layer5.json) | 0.992332985 | Fail | Not recorded | Not recorded | Not recorded | Not recorded |
| [128/128 HiFi2](actual_long_attention_hifi2_layer5.json) | 0.998700577 | Pass | 0.998378523 | 0.997020468 | Pass* | 2 |
| [64/256 LoFi](actual_long_paged_q64k256_lofi_layer5.json) | 0.994555053 | Fail | 0.995549563 | 0.959289410 | Fail* | 108 |
| [64/256 HiFi2](actual_long_paged_q64k256_hifi2_layer5.json) | 0.999050538 | Pass | 0.999146879 | 0.998209209 | Pass* | 0 |
| [32/512 LoFi](actual_long_paged_q32k512_lofi_layer5.json) | 0.996133971 | Pass | 0.997212238 | 0.963909452 | Fail | 62 |

\* The endpoint criterion is evaluated here from saved row diagnostics for these earlier runs; it was not an enforced gate during their execution. The baseline and expert controls did not save row diagnostics.

The **original contract** required aggregate prefill PCC ≥ .995 plus every traced decode check ≥ .995. After Q64/K256 LoFi exposed prefill row 262143 at .959289 while decode at the same position passed, a **new targeted regression** was added: prefill rows 262142 and 262143 must individually meet the same .995 bar. The aggregate check remains; other sampled rows remain diagnostic, with no blanket per-row gate. `prefill_aggregate_passed` preserves the original criterion, while `passed` now also requires the two endpoint checks.

Q32/K512 LoFi demonstrates why the regression matters: its aggregate passes .996133971, but final-row PCC .963909452 fails. The [completed log](actual_long_paged_q32k512_lofi_layer5.log) ends in the accuracy assertion; this is not an allocation failure. The run has 62 diagnostic misses, including 14 among the final 33 rows. It is rejected despite the aggregate pass.

All completed maximum-context runs have identical passing decode PCCs: .997745272 at 262143, .999088389 at 262142, then .997745272 on repeated 262143. The two HiFi2 configurations pass the endpoint criterion. Q128/K128 HiFi2 retains two interior diagnostic misses; Q64/K256 HiFi2 passes all 291 samples (minimum .996061645). Sampling does not establish all-output parity.

The earlier shared-environment Q128/K256 trial failed before accuracy comparison: its first BF16 K/V SDPA allocated a CB region ending at 2012160 B, exceeding 1572864 B L1 ([log](actual_long_attention_k256_layer5.log)). The later paged-only probe keeps first-chunk geometry unchanged and executes Q64/K256 and Q32/K512 legally.

Prior controls use runtime `d6f4d858…`; Q32/K512 uses `365be219…` with an explicit LoFi override. Existing recorded precision-policy fields match the earlier LoFi control except paged geometry; newly recorded expert gate/down K-block fields both equal 22. This is not a sweep under one unchanged runtime hash. Full hashes remain in the JSON.

## Warmed host timing and selection

Both timing runs use the same actual 4096-token fixture (`85fef8da…`), runtime `365be219…`, HiFi2 prefill attention, 128 checked decode steps, and program-cache miss guard. The shared harness measures three complete-prefill host elapsed durations with device synchronization; these are not device-profiler durations.

| Paged geometry | Prefill median (µs) | Prefill range (µs) | Traced decode median (µs) | Prefill PCC | Minimum decode PCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| [128/128](prefill_choice_q128k128_hifi2.json) | 195613.065 | 195299.758–196188.591 | 1171.1347 | 0.999132067 | 0.995214121 |
| [64/256](prefill_choice_q64k256_hifi2.json) | 195009.555 | 194931.997–196242.232 | 1170.9036 | 0.999131800 | 0.995214121 |

The Q64/K256 prefill median is 603.510 µs (0.309%) lower; sample ranges overlap. Decode is effectively unchanged. The choice is primarily supported by clean maximum-context sampled accuracy, with a modest measured host-median advantage. No maximum-context speed or statistically established speedup is claimed.

These controls demonstrate sensitivity to prefill SDPA fidelity and geometry. They do not prove a specific kernel defect or require broader expert/cache precision increases. See [the source investigation](AUTODEBUG_long_prefill.md) for lowering, numerical-buffer, and L1 evidence. The parent owns implementation and final selected-default verification, including near-maximum nonaligned context.
