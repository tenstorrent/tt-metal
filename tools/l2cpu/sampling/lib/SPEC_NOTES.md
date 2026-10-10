<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->
# x280s sampling: exact semantics and corner-case decisions

This file is the sampling specification of the library: greedy, temperature, top-k, top-p and a seeded
draw, with every corner case decided. Where a choice was open, the simplest deterministic rule was chosen. The C library (`x280s.c`) is the normative implementation;
`tests/np_ref.py` is an independent NumPy model of the same text.

## Inputs

| Item | Rule |
| --- | --- |
| dtype | `X280S_DTYPE_F32` (IEEE binary32) or `X280S_DTYPE_BF16` (u16 = top 16 bits of the binary32 pattern). bf16 -> fp32 is `bits << 16`, exact. Rounding fp32 to bf16 is the producer's business. |
| vocab V | Only elements `[0, V)` are read. Padding columns are never touched. |
| stride | Element `i` is at `base[i * stride]` (elements, not bytes); 1 = contiguous. |
| Errors | NULL `logits`/`params`, `V == 0`, `stride == 0`, `V > INT32_MAX` -> `-1`; unknown dtype -> `-2`; `work == NULL` on the sampling path -> `-1`. Stats are zeroed first in every call. |

## Step by step

1. **NaN.** Every NaN in `[0, V)` (any payload, either sign) is treated as `-inf` and counted in
   `stats.nan_count`, on both the greedy and the sampling path.
2. **Greedy** when `!(temperature > 0)`: `0`, `-0`, negative and NaN temperatures all select greedy.
   Result = lowest index of the maximum raw logit. `+0` and `-0` compare equal (lowest index wins).
   An all-`-inf` or all-NaN row returns 0.
3. **Scale.** `T = temperature`; `+inf` is replaced by `FLT_MAX` (so `inf / T` never makes a NaN).
   `x_i = logit_i / T` as one fp32 division (no reciprocal). Overflow to `+-inf` and underflow to
   `+-0` or subnormals are kept as IEEE produces them. A positive subnormal temperature is valid.
   After step 1 and this clamp no scaled value can be NaN.
4. **Top-k.** `K = top_k`; `top_k == 0` or `top_k > 1024` -> `K = K_MAX = 1024`; then `K = min(K, V)`.
   `stats.cap_applied = 1` exactly when the cap was requested (`top_k == 0` or `> 1024`) and `V > 1024`
   (candidates were really dropped by it). Kept: the K highest-ranked elements in the total order
   "value descending, then index ascending". `-inf` elements can be candidates when fewer than K
   elements are finite (they get weight 0, see 6).
5. **Sort.** Candidates in the same total order. Because indices are unique the order is strict, so
   the top-k set and its order are fully determined; any correct selection/sort algorithm (scalar heap,
   RVV prefilter) gives identical results.
6. **Weights.** `w_j = 1` if `x_j == x_max`, else `x280s_expf(x_j - x_max)`, where `x_max = x_0`.
   For finite `x_max` this equals the plain formula (`expf(0) = 1`). The rule makes two corners total:
   - `+inf` scaled values: every `+inf` candidate gets weight 1, everything else `expf(-inf) = 0`
     (the `+inf` candidates share the mass uniformly; lowest K indices among them if more than K).
   - all-`-inf` row on the sampling path: all weights 1, i.e. uniform over the first K indices, exactly
     like an all-equal row. (Greedy on the same row returns 0.)
   `S = sum w_j`, a sequential fp32 running sum `0 + w_0 + w_1 + ...` in sorted order. `1 <= S <= 1024`.
7. **Top-p.** `p = top_p`; if `!(p < 1)` then `p = 1` (so `top_p > 1` and NaN behave as 1).
   `thr = p * S` (one fp32 multiply). Keep the shortest prefix `0..n-1` whose sequential running sum
   `>= thr`. `top_p <= 0` gives `thr <= 0`, so exactly one candidate is kept. The prefix is always found
   (the full running sum equals S >= thr). With `p = 1` the prefix ends at the last candidate that still
   changes the fp32 running sum: trailing weights that are 0 or too small to change it are dropped.
   `kept_sum` = the running sum at the end of the prefix.
8. **Draw.** `r = SplitMix64(seed ^ ((uint64)user << 32) ^ step)` (one output step of SplitMix64:
   `z = x + 0x9E3779B97F4A7C15`, then the two xor-shift-multiply rounds with `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`); `u = (float)(uint32)(r >> 40) * 2^-24` (exact, `0 <= u <= 1 - 2^-24`);
   `target = u * kept_sum` (one fp32 multiply; it may round up to `kept_sum`). The token is the first
   kept candidate whose running sum (recomputed the same way, from the same stored weights) is
   strictly greater than `target`. If none is (only when `target` rounded up to `kept_sum`), the last kept
   candidate is returned and `stats.fallback = 1`. Zero-weight candidates can never be returned by the
   strict rule, and the last kept candidate always has a non-zero weight, so the fallback is safe too.
   Note: `user` and `step` are XORed into overlapping bits once `step >= 2^32`; the mapping is still a
   pure function of `(seed, user, step)`, which is all bit-exactness needs.

## Arithmetic rules (bit-exactness between x86-64 and rv64gcv)

- Only binary32 `+ - * /`, comparisons, integer ops and exact int<->fp32 conversions. No double, no
  long double, no libm, no FMA: every flavour is built with `-O2 -ffp-contract=off -fno-fast-math`
  (`objdump` of both RISC-V objects shows no `fmadd/fmsub`, the x86 `.so` no `vfmadd`).
- The entry points force round-to-nearest-even and restore the caller's state on exit: x86-64 loads
  MXCSR `0x1f80` (also clears FTZ and DAZ, which some host libraries set), RISC-V writes `frm = 0`.
  The firmware must have enabled the FPU (`mstatus.FS != 0`; and `VS` for the RVV build).
  `x280s_expf` called on its own does not change the FP environment.
- `x280s_expf` returns the canonical quiet NaN `0x7fc00000` for any NaN input: x86 propagates NaN
  payloads while RISC-V canonicalises them, so returning `x + x` differed (caught by the QEMU expf hash).
- RVV (`-DX280S_RVV`) is used only for exact operations: argmax (max reduction, compare, first-set),
  NaN counting, and the top-k prefilter compare `v > root_raw`. Division, `expf`, sums and the heap
  stay scalar and in the documented order. Rows with `stride != 1` always use the scalar code.

## x280s_expf algorithm

fp32 only: specials (`NaN`, `x > 88.75 -> +inf`, `x < -104 -> +0`); `n = rne(x * log2e)` through the
`1.5 * 2^23` shifter; Cody-Waite `r = (x - n*ln2_hi) - n*ln2_lo` with a 12-trailing-zero `ln2_hi`;
`exp(r) = 1 + r + r^2 * q(r)`, `q` = Taylor terms 1/2..1/5040 in Horner form; scale by `2^n` built
from exponent bits, in two steps for `n < -126` (gradual underflow, one rounding) and `n > 127`.
Exhaustive check over all 2,239,856,642 fp32 inputs in `[-104, 88.75]` against double `exp`:
99.19 % correctly rounded, 33 inputs above 1 ulp, max error 1.023 ulp (`make expf-sweep`).
