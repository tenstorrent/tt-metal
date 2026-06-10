# scaled_dot_product_attention — precision matrix results

- **Last run**: 2026-06-10 (Refinement 1 — numerical configurability expansion)
- **Test**: `test_scaled_dot_product_attention_precision_matrix.py` — 229 passed, 160 skipped
- **Shapes (B,H,S,D)**: (1,1,32,32), (1,1,128,64), (1,1,64,32), (1,4,256,64), (2,4,128,64),
  (1,1,128,256), (1,1,2048,64), (1,8,512,128) — all tile-aligned (non-aligned shapes are
  outside SUPPORTED until Refinement 4).
- **Gating**: normal input → PCC (floor 0.999 fp32, 0.99 bf16/bf8b, 0.99 LoFi cap);
  uniform input → max_abs (PCC degenerates when attention is near-uniform — the
  Refinement 3 pathology — while abs error stays small).

## Aggregated results (min PCC / max max_abs / max rel_RMS across the 8 shapes)

| dtype | fidelity | fp32_acc | dist | shapes | min PCC | max max_abs | max rel_RMS |
|---|---|---|---|---|---|---|---|
| FLOAT32 | HiFi4 | True | rand | 8 | 0.995026 | 0.00456 | 0.00397 |
| FLOAT32 | HiFi4 | True | randn | 8 | 0.999995 | 0.00628 | 0.00608 |
| FLOAT32 | HiFi3 | True | rand | 8 | 0.995020 | 0.00455 | 0.00420 |
| FLOAT32 | HiFi3 | True | randn | 8 | 0.999995 | 0.00710 | 0.00642 |
| FLOAT32 | HiFi2 | True | rand | 8 | 0.722166 | 0.02870 | 0.02406 |
| FLOAT32 | HiFi2 | True | randn | 8 | 0.999910 | 0.03373 | 0.02013 |
| FLOAT32 | LoFi | True | rand | 8 | 0.250229 | 0.14053 | 0.13662 |
| FLOAT32 | LoFi | True | randn | 8 | 0.992861 | 0.14289 | 0.20135 |
| BFLOAT16 | HiFi4 | False | rand | 8 | 0.877951 | 0.01573 | 0.00718 |
| BFLOAT16 | HiFi4 | False | randn | 8 | 0.999887 | 0.02336 | 0.01567 |
| BFLOAT16 | HiFi3 | False | rand | 8 | 0.877951 | 0.01573 | 0.00718 |
| BFLOAT16 | HiFi3 | False | randn | 8 | 0.999887 | 0.02336 | 0.01568 |
| BFLOAT16 | HiFi3 | True | rand | 8 | 0.933936 | 0.01675 | 0.01620 |
| BFLOAT16 | HiFi3 | True | randn | 8 | 0.999980 | 0.00784 | 0.01814 |
| BFLOAT16 | HiFi2 | False | rand | 8 | 0.690757 | 0.03095 | 0.02293 |
| BFLOAT16 | HiFi2 | False | randn | 8 | 0.999877 | 0.02668 | 0.02491 |
| BFLOAT16 | HiFi2 | True | rand | 8 | 0.713863 | 0.03346 | 0.02852 |
| BFLOAT16 | HiFi2 | True | randn | 8 | 0.999922 | 0.02385 | 0.02712 |
| BFLOAT16 | LoFi | False | rand | 8 | 0.251597 | 0.14239 | 0.12173 |
| BFLOAT16 | LoFi | False | randn | 8 | 0.994578 | 0.12043 | 0.18132 |
| BFLOAT16 | LoFi | True | rand | 8 | 0.245339 | 0.13845 | 0.13373 |
| BFLOAT16 | LoFi | True | randn | 8 | 0.992928 | 0.12824 | 0.19212 |
| BFLOAT8_B | HiFi3 | True | rand | 8 | 0.888956 | 0.01913 | 0.01692 |
| BFLOAT8_B | HiFi3 | True | randn | 8 | 0.999773 | 0.03764 | 0.02690 |
| BFLOAT8_B | HiFi2 | True | rand | 8 | 0.697746 | 0.03474 | 0.03140 |
| BFLOAT8_B | HiFi2 | True | randn | 8 | 0.999762 | 0.03764 | 0.03824 |
| BFLOAT8_B | LoFi | True | rand | 8 | 0.250540 | 0.13896 | 0.12455 |
| BFLOAT8_B | LoFi | True | randn | 8 | 0.992827 | 0.12043 | 0.18036 |

Low uniform-dist PCC rows have tiny absolute error (max_abs ≤ 0.035 at HiFi2+) — PCC
degenerates because near-uniform attention output has near-zero variance. LoFi error
(max_abs ≤ 0.14) is expected hardware fidelity tradeoff, not a kernel issue. Op defaults
(HiFi2 bf16/bf8b, HiFi4 fp32, fp32 DEST) sit on the precise rows.

## Skipped combinations

| combo (160 cells) | reason |
|---|---|
| HiFi4 + fp32_dest_acc_en + bf16/bf8b | Wormhole B0 K-accumulator known-bad (issue #38306) — entry point raises ValueError |
| fp32_dest_acc_en=False + fp32/bf8b | 16-bit DEST format pairing structurally unsupported (probe_008: fp32 pcc 0.008, bf8b NaN) — entry point raises NotImplementedError |
