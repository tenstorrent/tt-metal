# rms_norm — precision matrix results

**Last run**: 2026-06-19 (Refinement 2 — Numerical configurability expansion)
**Test**: `test_rms_norm_precision_matrix.py`
**Outcome**: 641 passed, 128 skipped, 0 failed.

## Axes swept

- **shapes** (tile-aligned only; ROW_MAJOR / non-aligned are Refinement 3):
  `32x32`, `32x64`, `64x128`, `128x512` (Regime B), `4x8x32x256` (rank-4),
  `1024x1024`, `256x2048`, `1x1x32x4096` (wide, Regime B).
- **dtype**: `bfloat16`, `float32`, `bfloat8_b`.
- **math_fidelity**: HiFi4, HiFi3, HiFi2, LoFi (not gated by the op).
- **fp32_dest_acc_en**: True, False.
- **gamma_mode**: gamma, no_gamma.
- **distribution**: uniform (`rand`), normal (`randn`).

## Skipped cells

- `{dtype=float32, fp32_dest_acc_en=False}` — 128 cells. Documented EXCLUSION:
  fp32 input mandates fp32 accumulation (no correct bf16-dest datapath for fp32
  data). The op raises a support refusal (`ExcludedCell`); verified by
  `test_rms_norm_precision_matrix_fp32_no_acc_refused`.

## Representative metrics (shape 128x512, fp32_dest_acc_en=True, normal dist)

| dtype | gamma | fidelity | PCC | Max ATOL | relRMS | p99_abs | inf/nan |
|-------|-------|----------|-----|----------|--------|---------|---------|
| bfloat16  | yes | HiFi4 | 0.999995 | 0.065 | 0.0039 | 0.0159 | 0/0 |
| bfloat16  | yes | LoFi  | 0.999788 | 0.522 | 0.0411 | 0.1567 | 0/0 |
| float32   | yes | HiFi4 | 0.9999997 | 0.019 | 0.0015 | 0.0057 | 0/0 |
| float32   | yes | LoFi  | 0.999780 | 0.595 | 0.0444 | 0.1708 | 0/0 |
| bfloat8_b | yes | HiFi4 | 0.999896 | 0.099 | 0.0146 | 0.0436 | 0/0 |
| bfloat8_b | yes | LoFi  | 0.999711 | 0.586 | 0.0358 | 0.1427 | 0/0 |
| bfloat16  | no  | HiFi4 | 0.999996 | 0.035 | 0.0036 | 0.0123 | 0/0 |
| float32   | no  | HiFi4 | 0.99999983 | 0.005 | 0.00062 | 0.0022 | 0/0 |
| bfloat8_b | no  | HiFi4 | 0.999944 | 0.065 | 0.0110 | 0.0305 | 0/0 |

Notes:
- LoFi shows the expected precision drop (smaller PCC, larger RMS) across all
  dtypes — hardware behavior, not a bug. Still well above the 0.98–0.99 floor.
- `Max RTOL = inf` for bfloat8_b is an artifact of near-zero reference elements
  in the relative-error denominator, not an output Inf (`inf=0` everywhere).
- No Inf / NaN in any cell after the Regime-B mcast partial-tile-bytes fix.

## PCC floors asserted (skill §11)

- bfloat16: 0.99
- float32:  0.99
- bfloat8_b: 0.98 (block-float across LoFi is inherently lower)
