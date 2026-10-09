# Physical GDN epilogue result

Completed Oct 9 at 22:26 UTC on one physical TP4. All eight batch/placement cases passed, including all-rank bit-identical outputs, normalization checks, cache rebinding, zero padding, changed-input trace replay and clean device closure. Model integration and full GPQA remain pending.

| Placement | Batch | Native us | Fused us | Native/fused |
|---|---:|---:|---:|---:|
| dram | 32 | 151.93 | 85.25 | 1.782x |
| dram | 16 | 92.24 | 64.72 | 1.425x |
| dram | 8 | 58.07 | 52.53 | 1.105x |
| dram | 1 | 35.58 | 46.38 | 0.767x |
| l1 | 32 | 108.51 | 83.07 | 1.306x |
| l1 | 16 | 69.20 | 62.82 | 1.102x |
| l1 | 8 | 46.76 | 51.22 | 0.913x |
| l1 | 1 | 34.49 | 46.16 | 0.747x |

B16/B32 improve in both placements. B1 regresses in both; B8 regresses in L1. Retain native paths for losing cases rather than enabling this globally. The whole model must verify savings with its actual mixed operand placement.

Five samples of 100 trace replays, bracketed native/fused/native controls, drift below 3% in every case. Each epilogue consumes only the single live FP32 recurrence row, keeps the BF16 intermediate rounding boundary, and writes caller-owned output. It does not reduce weight/KV/state precision.
