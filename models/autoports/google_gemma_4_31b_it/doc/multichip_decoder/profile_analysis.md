# Final profile interpretation

`performance.md` compares complete whole-layer device windows against the selected optimized single-chip profiles. The denominator includes all operations and internal device gaps; the merged tt-perf-report table is diagnostic and does not replace it.

Sliding B1 decode's five material matmuls are marked Optimized (BFP4 weights, LoFi, BF16 outputs). The representative table attributes about 206 us to matmuls, 56 us to RS/AG, 54 us to normalization, and 38 us to SDPA; these merged per-op totals are not whole-layer latency. Persistent buffers and Ring topology remain in the actual runtime rows, and post-reduction norms consume width-sharded L1.

Prefill matmul rows are marked SLOW at about 42–53% of the report's compute peak. The advice saying “No program_config specified” is a parser limitation for MinimalMatmulConfig: runtime attributes explicitly record M4/K8/N16, subblock 1×8 and 11×10 grid. BFP4/LoFi K14, K14/N8 smaller-grid and padded6144 controls did not improve the layer; K21 hit the recorded concrete L1 bound. These measurements support the selected configuration, not a claim of peak prefill utilization. Sliding B1 merged rows attribute about 3.42 ms to matmuls, 1.82 ms to normalization, 1.17 ms to CCL, 0.96 ms to typecasts and 0.76 ms to SDPA. Useful whole-layer FLOPs utilization is only 13.34%, reflecting these other operations and gaps.

The selected multichip prefill path retains conversions, including BFP8-to-BFP8 copies at collective boundaries. These remain visible in the final measured path; no unmeasured cleanup speedup is claimed. Raw logs and profiler captures stay local; compact JSON, TXT and CSV reports retain source hashes and exact capture/replay selection provenance.

Full-attention B1 decode likewise marks all five matmuls Optimized. Its representative merged rows attribute about 226 us to matmuls, 71 us to SDPA, 54 us to CCL and 53 us to normalization. The complete-window decode mean is 534.8 us; none of those diagnostic subtotals replaces that denominator. The full-attention baseline is the selected optimized K32 SDPA profile, not the older superseded profile.
