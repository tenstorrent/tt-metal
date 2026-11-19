# Final matched performance

Milliseconds;32tokens uses64decode/5prefill samples, longer lengths10decode/3prefill. Sources and legacy/contract statuses are in `final_performance_summary.json`. QSA1024 per-row diagnostic has independent HF assessment; raw failure remains visible.

| Layer | Logical prefill | Fused prefill | Optimized prefill | Fused traced decode | Optimized traced decode |
|---|---:|---:|---:|---:|---:|
| 0 | 32 | 78.284 | 58.806 | 5.0192 | 3.0518 |
| 0 | 33 | 83.628 | 62.436 | 5.0155 | 3.0611 |
| 0 | 129 | 303.250 | 236.838 | 5.0179 | 3.0590 |
| 0 | 1024 | 2302.439 | 1861.891 | 5.0178 | 3.0540 |
| 1 | 32 | 95.877 | 63.110 | 6.3490 | 4.2483 |
| 1 | 33 | 100.714 | 67.492 | 6.3452 | 4.2482 |
| 1 | 129 | 366.464 | 255.152 | 6.3599 | 4.2494 |
| 1 | 1024 | 2925.416 | 2004.284 | 6.3488 | 4.2863 |
| 3 | 32 | 184.062 | 149.206 | 9.5256 | 7.2960 |
| 3 | 33 | 193.293 | 156.793 | 9.5142 | 7.2955 |
| 3 | 129 | 749.398 | 603.927 | 9.5245 | 7.3072 |
| 3 | 1024 | 6092.845 | 4864.423 | 9.8400 | 7.6150 |

Final native interleaved BF16/HiFi4 GDN control3.0605ms versus selected DRAM3.0518ms. Final BFP8/LoFi separate experts best3.0766ms; native fused-SiLU separate3.0759ms but fails real-HF decode0.992139. Packed default3.0518ms passes.

The precision-locked27-geometry sweep reaches3.0514ms. Repeated interleaved default/down-Kblock20 comparison (`final_geometry_tie_repeat.json`) has medians3.052530/3.052430ms: a0.000100ms difference with overlapping samples and both directions across rounds. This is indistinguishable at this harness resolution; the existing default is retained. It reproduces the best measured correct geometry region, without claiming a significant win between tied knobs. Earlier BFP8/HiFi2 GDN3.0364ms is invalid on real-weight gates; BFP4 remains a stage8 candidate under the user-required BFP8 default.
