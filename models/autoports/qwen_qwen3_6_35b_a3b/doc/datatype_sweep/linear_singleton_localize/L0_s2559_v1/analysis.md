The same-input probe does not support a projection-grouping or singleton-engine repair. K512→K1024 leaves immediate attention outputs bit-identical. Switching to chunk-n1 produces a very small output difference with the BF16 boundary and none with FP32 carry. Boundary rounding has the larger local effect, but this does not establish better accuracy.

| Comparison | Raw recurrence max Δ | Final partial max Δ | Final partial RMS Δ | Changed final elements /8192 |
| --- | ---: | ---: | ---: | ---: |
| K512 vs K1024, BF16 boundary | 0 | 0 | 0 | 0 |
| K512 vs K1024, FP32 carry | 0 | 0 | 0 | 0 |
| Singleton vs chunk-n1, BF16 boundary | 7.15255737e-07 | 1.82539225e-06 | 1.64799997e-07 | 2048 |
| Singleton vs chunk-n1, FP32 carry | 7.15255737e-07 | 0 | 0 | 0 |
| BF16 boundary vs FP32 carry, singleton | 0.00313472748 | 0.000293366611 | 1.41674324e-05 | 8192 |

All four ranks have identical q/k/v/beta and initial-state values for the fixed-K1024 engine comparison, after undoing chunk head repetition/layout. Projection and conv updates are identical. The first observed difference therefore occurs inside recurrence/decay execution. The two paths evaluate decay differently; the saved operands do not isolate that from accumulation order.

With the BF16 boundary, the engine change crosses one raw-output BF16 rounding boundary on rank1, changes one gated value, and changes that rank’s2048 final projection partials. With FP32 carry, raw differences disappear at the output rounding boundary. Updated FP32 states still differ; rounding them to BF16 changes8 elements with the BF16 boundary and17 with FP32 carry. Future-token effects remain untested. K512→K1024 also leaves59 tiny differences in conv history despite identical immediate outputs.

These are one-layer, one-position controls, not a full-model gate or accuracy-reference comparison. Both paths use the same resumed state; fresh64-vs63+1 factorization was not reconstructed because only two final-chunk input rows were captured. Rank partials were concatenated for CPU float64 error measurement, not treated as a runtime collective. No supported production change follows from this evidence alone.

Evidence: [result.json](result.json), [operands_outputs.pt](operands_outputs.pt), and [analysis.json](analysis.json), which includes source-artifact hashes, stage metrics, operand checks, and limitations. Analysis used CPU PyTorch only; no additional device execution.
