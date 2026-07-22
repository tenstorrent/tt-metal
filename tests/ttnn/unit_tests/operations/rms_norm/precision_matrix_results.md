# rms_norm — precision matrix results

Source: `test_rms_norm_precision_matrix.py`
Last run: 2026-07-22 · Blackhole p150b
Result: **160 passed, 32 skipped** (all PCC gates held).

## Axes

- **dtype**: bfloat16, float32, bfloat8_b
- **fp32_dest_acc_en**: True, False
- **math_fidelity**: HiFi4, HiFi2
- **with_gamma**: True, False   (gamma RM; carried at bf16 for a bf8b input)
- **distribution**: uniform (`rand`), normal (`randn`)
- **shapes** (all tile-aligned so bf8b is exercised on every shape):
  `(32,64)`, `(2,64,128)`, `(2,4,128,512)`, `(1,1,128,4096)`

## Gate

Assert on **PCC only** (per `/numeric-formats-metal` §11), at the golden-suite
`TOLERANCES` thresholds: f32 → 0.999, bf16 → 0.995, bf8b → 0.99. rel-RMS +
allclose are printed for observability, not gated.

## Skips

- `{float32, fp32_dest_acc_en=False}` — the op EXCLUSION (fp32 activations with
  non-fp32 DEST accumulation is lossy). 32 cells skipped
  (4 shapes × 2 fidelity × 2 gamma × 2 distribution).

## Observed rel-RMS (max across all cells, per dtype)

| dtype | max rel-RMS | golden RMS gate | PCC gate | result |
|---|---|---|---|---|
| float32   | 0.0084 | 0.02 | 0.999 | pass |
| bfloat16  | 0.0561 | 0.04 (randn only) | 0.995 | pass (PCC) |
| bfloat8_b | 0.0567 | 0.10 | 0.99  | pass |

The bf16 max rel-RMS (0.056) is driven by the **uniform** distribution at HiFi2 /
bf16-acc — a corner the golden suite does not exercise (it uses `randn`, where
bf16 stays ≤ 0.04). PCC holds everywhere, which is the gated metric. rel-RMS
range across all 160 cells: **0.0006 – 0.0567**.
