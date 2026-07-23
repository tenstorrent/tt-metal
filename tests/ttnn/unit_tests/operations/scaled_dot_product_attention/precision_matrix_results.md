# scaled_dot_product_attention — precision matrix results

Last run: 2026-07-23 (Refinement 1 — Numerical configurability expansion).
Test: `test_scaled_dot_product_attention_precision_matrix.py` (40 passed, 8 skipped).

Config: `math_fidelity=HiFi4` requested (op clamps bf16/bf8b → HiFi2 internally,
issue #38306; fp32 keeps HiFi4). Reference = torch fused SDPA in fp32.

## PCC by (dtype, fp32_dest_acc_en, distribution)

Worst-case PCC across the four tile-aligned shapes
{(1,1,32,32), (1,1,128,64), (1,4,256,64), (2,4,128,128)}:

| dtype     | fp32_dest_acc_en | normal (worst) | uniform (worst) | gate (normal / uniform) |
|-----------|:----------------:|:--------------:|:---------------:|:-----------------------:|
| float32   | True             | ~0.99999       | ~0.99986        | 0.999 / 0.98            |
| float32   | False            | — (EXCLUDED)   | — (EXCLUDED)    | skipped                 |
| bfloat16  | True             | ~0.99997       | ~0.9956         | 0.995 / 0.98            |
| bfloat16  | False            | ~0.9999        | ~0.9874         | 0.99  / 0.98            |
| bfloat8_b | True             | ~0.99989       | ~0.9883         | 0.99  / 0.98            |
| bfloat8_b | False            | ~0.99985       | ~0.9815         | 0.99  / 0.98            |

Notes:
- **{float32, fp32_dest_acc_en=False}** is an op-side EXCLUSION (legal-but-lossy,
  mirrors softmax) — skipped in the matrix.
- `normal` (randn) is the golden-calibrated distribution; gate = golden TOLERANCES.
- `uniform` [0,1] is a stress distribution (small all-positive range flattens the
  softmax, amplifying relative error); gate is a 0.98 floor, not the tight randn
  tolerance. fp32 still lands ~0.9999.
- Large-head-dim shapes (D ∈ {256, 512, 1024}) are omitted — they OOM at these
  dtypes and are Refinement 2's L1-budget scope, not a precision question.
