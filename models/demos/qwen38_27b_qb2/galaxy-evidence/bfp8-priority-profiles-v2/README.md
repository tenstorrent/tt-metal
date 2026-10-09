# BFP8 stage results and throughput projection

Projections anchored to completed native full-model sweep and eager stage/standalone trace differences; not optimized full-model or Galaxy throughput measurements.

| Batch/TP4, 32K ISL | Measured native TSU | Projected shared-Q/K TSU | Projected with epilogue | Ideal Galaxy output TPS with both |
|---|---:|---:|---:|---:|
| 16 | 11.75 | 14.6 | 14.8 | 1900 |
| 32 | 7.35 | 10.2 | 10.5 | 2692 |

Formula: native measured step minus 48 times the median-rank GDN-layer kernel-duration difference, then subtract 48 times the standalone DRAM epilogue trace saving. Attention stays unchanged. The epilogue is not model-integrated yet.

Assumptions:

- 48 equivalent GDN layers; measured per-layer difference transfers to full-model trace.
- Attention latency unchanged within the measured diagnostic variation.
- DRAM epilogue difference adds without overlap with the recurrence replacement.
- Eight independent TP4 replicas equally loaded for Galaxy aggregate extrapolation.
- B32 serving buckets and aggregate scaling require separate qualification.

All four B16/B32 profiles passed test, cleanup, repeatability and four-rank timing checks with matching input hashes and precision. The physical epilogue also passed; its result is linked in projection.json. Nested stage intervals are not additive, and no full-model/accuracy gate is implied.
