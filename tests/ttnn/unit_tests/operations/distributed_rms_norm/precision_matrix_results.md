
# distributed_rms_norm — precision matrix results

Last run: 2026-10-23 (Refinement 4), Blackhole 2x2 mesh, G = 2 along cluster_axis 1, Linear, bf16 TILE gamma.
Source: `test_distributed_rms_norm_precision_matrix.py` (worst device of the group per case; PCC asserted >= 0.99).

Skipped: `{float32, fp32_dest_acc_en=False}` — EXCLUSION (fp32 input needs fp32 accumulation): 64 cases.

## Summary (worst over 8 shapes x 2 distributions)

| dtype | fp32 DEST | fidelity | min PCC | max rel-RMS | max p99 abs err | max ULP (bf16) |
|---|---|---|---|---|---|---|
| BFLOAT16 | True | HiFi4 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | True | HiFi3 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | True | HiFi2 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | True | LoFi | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | False | HiFi4 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | False | HiFi3 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | False | HiFi2 | 0.999999 | 0 | 0 | 0 |
| BFLOAT16 | False | LoFi | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | True | HiFi4 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | True | HiFi3 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | True | HiFi2 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | True | LoFi | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | False | HiFi4 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | False | HiFi3 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | False | HiFi2 | 0.999999 | 0 | 0 | 0 |
| BFLOAT8_B | False | LoFi | 0.999999 | 0 | 0 | 0 |
| FLOAT32 | True | HiFi4 | 0.999999 | 0 | 0 | 0 |
| FLOAT32 | True | HiFi3 | 0.999999 | 0 | 0 | 0 |
| FLOAT32 | True | HiFi2 | 0.999999 | 0 | 0 | 0 |
| FLOAT32 | True | LoFi | 0.999999 | 0 | 0 | 0 |

## Full table

| shape | dtype | fidelity | fp32 DEST | dist | PCC | rel-RMS | median abs | p99 abs | max ULP | mean ULP | median ULP | p99 ULP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1x32x64 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x64 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x1792 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x64x1024 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x640x1792 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x2048x1280 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x32x50 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1x1x17x1280 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT16 | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | FLOAT32 | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi4 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi4 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi4 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi4 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi3 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi3 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi3 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi3 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi2 | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi2 | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi2 | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | HiFi2 | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | LoFi | True | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | LoFi | True | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | LoFi | False | rand | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2x1x100x100 | BFLOAT8_B | LoFi | False | randn | 0.999999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
