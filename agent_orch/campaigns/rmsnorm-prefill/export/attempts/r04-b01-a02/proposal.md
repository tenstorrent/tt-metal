# r04-b01-a02: PRE at HiFi2: the per-tile x*x ELWMUL and the ones*S^T row-sum matmul run at MathFidelity::HiFi2 instead of the kernel's HiFi4; POST and the post-AG combine stay HiFi4

## Motivation
The AG can't start until the slowest worker's row stat is pushed, and that stat is produced by PRE: a DST-accumulated
`mul_tiles(x, x)` per input tile (HiFi4 ELWMUL), one pack of S, then `C = ones * S^T` (HiFi4 matmul), one pack.
- r01-b01-a04 / r03-b04-a02: PRE runs ~110-125 ns/tile and is per-core compute bound (unpack/math), roughly matched to
  the input read rate. When reads got faster (r03-b04-a02 waves) PRE became the gate (+2.4 µs).
- r03-b03-a02: the PRE tail after the last input block lands is ~0.7 µs on every shape, consistent with the last 4-tile
  block's HiFi4 math plus the S pack -> matmul -> pack hop.
- tt-metal#58723 (quoted in r03-b01-a03): BH ELWMUL math 82.6 cycles/tile at HiFi4 vs 34.6 at HiFi2.
- Suggested in r01-b01-a04, r01-b04-a03, r03-b01-a03, r03-b03-a02, r03-b04-a02, r04-b01-a01 (#2). Never tried.
r04-b01-a01 (parent) removed the dev-0 straggler, so every core now follows the same timeline and a per-core PRE cut
should map straight onto the AG start and the kernel end.

## Mechanism
Compute kernel only (`dit_rmsnorm_fused_compute.cpp`, resident whole-row PRE path, the one the campaign shapes use):
- `constexpr MathFidelity pre_fidelity = MathFidelity::HiFi2`.
- `mul_init(input_cb, input_cb)` -> same calls with an explicit fidelity: `state_configure`,
  `llk_math_eltwise_binary_init<ELWMUL, NONE, pre_fidelity>(..., acc_to_dest=true)`, `llk_unpack_AB_init`.
- `mul_tiles(x, x)` -> `llk_unpack_AB` + `llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, pre_fidelity>`.
- `matmul_init/matmul_tiles(ones, S)` -> the same LLK calls with `pre_fidelity` (and MM_THROTTLE as before).
  This is lossless: matmul maps in0 (the ones tile) to SrcB, and HiFi2 uses SrcB's top 7 mantissa bits, which hold
  1.0 exactly; SrcA (S, tf32) is used in full in both HiFi2 phases.
Every later op re-inits with the kernel's MATH_FIDELITY (HiFi4), so POST/combine are unchanged.

## Why this is not a repeat
No node changed math fidelity anywhere. The PRE nodes so far changed packing (r01-b01-a04, r01-b04-a04: DST
accumulation) and the stat hop (r02-b02-a02/a03 matmul, r03-b03-a02 SFPU column sum); none changed the per-tile math cost.
Siblings this round attack the push handshake (r04-b03-a01) and the drain (r04-b04-a01); orthogonal.

## Expected effect and risk
- PRE per-tile math ~halves; if PRE is math bound the stat is ready earlier on the slowest core -> AG starts earlier:
  expect -0.1..-0.4 µs per shape (more on wide shapes if PRE lags the read there). If PRE is unpack bound (2 bf16 tiles
  per mul), the gain is small: then this tells future nodes PRE math isn't the lever.
- Accuracy: HiFi2 drops SrcB's last bf16 mantissa bit: sum(x^2) biased ~-0.28% (r03-b01-a03 fid.py emulation), 1/rms
  +0.14%. Emulated max_abs 0.0156 -> 0.0226 (HW today 0.024 -> expect ~0.03 of the 0.05 gate), PCC unchanged
  (0.9999986). If it fails the gate -> accuracy_fail; HiFi3 is the fallback for a child.
- JIT compile errors possible (raw LLK calls); no hang risk (CB counts unchanged).
