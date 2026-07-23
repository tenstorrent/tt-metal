# SFPU LLK Edge-Case Test-Coverage Audit

**Issue:** [tenstorrent/tt-metal#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
**Date:** 2026-07-23
**Scope:** All SFPU LLK kernels in `tt-metal/tt_metal/tt-llk`, audited through the tt-llk Python test infra (`tests/python_tests/`). Wormhole B0 and Blackhole share essentially the same SFPU kernel set (BH adds only `topk_xl`), so this audit treats them together and notes arch-specific test gaps inline.

---

## How to read this document

For every SFPU kernel/operation we list its plausible **edge cases** (domain boundaries, IEEE special values, format extremes, integer overflow, shift-amount limits, tie/ordering edges, etc.) and whether the test infra actually **drives and asserts** that edge.

**`Tested?` legend:**

| Value | Meaning |
|-------|---------|
| **Yes** | A test deliberately drives this edge value and asserts on it (cited). |
| **No** | The edge is neither excluded nor deliberately generated — the random sweep may or may not hit it, and behavior at it is not asserted. |
| **Excluded** | The stimulus domain deliberately clips this region out (via `sfpu_domains.py` `_OP_DOMAIN_REGISTRY` / `_SFPU_UNDEFINED_RANGES`, or a positive-only default). Behavior at/beyond the boundary is **not** verified — by design. |
| **Partial** | Some but not all of the edge is covered; see notes. |

---

## Executive summary — the coverage model and its systemic gaps

The SFPU test infra is a **random-sweep-within-safe-domains** system. It is excellent at *value-level correctness on benign inputs* across a large format × dest_acc × approx_mode matrix, but by construction it **avoids** the edges this issue is about. Four systemic findings dominate the table below:

1. **The unary "float sweep" (`test_eltwise_unary_sfpu_float`, the `ALL_MATHOPS` list) uses positive-only stimuli.** It passes **no `spec_A`**, so `generate_stimuli` falls back to `default_spec_for_format` = `uniform(0.1, 1.1)` for fp32/fp16/bf16 (and a positive `[0, ~2.9]` face for bfp). *Verified* at [test_sfpu_unary.py:628](tt_metal/tt-llk/tests/python_tests/test_sfpu_unary.py#L628) → [generator.py:257](tt_metal/tt-llk/tests/python_tests/helpers/stimuli_generator/generator.py#L257). Consequence: every op on `ALL_MATHOPS` (exp, log, sqrt, rsqrt, square, tanh, silu, gelu, gelu_tanh, sin, cos, abs, neg, celu, elu, hardsigmoid, floor/ceil/trunc/frac, relu_max, threshold, atanh, asinh, acosh, …) is **never fed a negative input** and never exercises its `x<0` branch, clip knee, saturation tail, or argument-reduction path in that sweep. The per-op signed domains in `sfpu_domains.py` are consumed **only** by the separate `test_eltwise_unary_sfpu_domain` (`DOMAIN_MATHOPS`) driver.

2. **The binary/ternary/scalar suites don't import `sfpu_domains.py` at all.** `test_sfpu_binary.py`, `test_sfpu_binary_bcast.py`, `test_sfpu_ternary.py`, `test_sfpu_binop_scalar.py` fall back to `default_spec_for_format` = positive `uniform(0.1, 1.1)` (floats) / non-negative half-range (ints). So div-by-zero, `pow` base≤0, `xlogy` y≤0, negative operands, ties, and all IEEE specials are avoided by the *default positive domain* — the registry's per-op holes are dormant for these tests.

3. **IEEE special values (±inf, NaN, ±0.0, denormals) are injected for exactly one family: the `isinf`/`isnan`/`isposinf`/`isneginf`/`isfinite` predicates.** No other op — not max/min, not comparisons, not div/pow, not typecast, not reduce, not where — is ever driven with inf/NaN/-0.0. Format extremes (fp max, denormals) and overflow-to-inf are likewise never driven for float ops.

4. **Integer sign/extreme edges are structurally excluded.** `_get_integer_bounds` uses `min+1`, so **INT32_MIN is never generated** anywhere, and default int stimuli are **non-negative** (`uniform(0, INT32_MAX//2-1)`). This silently removes negative-operand, sign-magnitude-Dst, overflow/wraparound, and unsigned-MSB coverage for the entire integer op family.

### Where edge coverage *is* genuinely good (the `Yes` highlights)
- **`isinf`/`isnan`/`isposinf`/`isneginf`/`isfinite`** — real ±inf/NaN injection (`test_eltwise_unary_sfpu_isinf_isnan`).
- **Integer shift ops (binary)** — a dedicated edge test drives shift amounts `{0..31, 32, 33, 40, 63, 100, 1000, -1, -5, -32, -1000}` against negative values, the sign bit and INT32_MAX; INT32_MIN is `Excluded` with a documenting `xfail`. **Wormhole-only** (skipped on Blackhole).
- **INT32 reduce** — `test_int32_reduce_extreme` genuinely injects INT32_MIN/INT32_MAX (Wormhole-only; BH still buggy, tt-metal#44750).
- **`unary_eq`/`unary_ne`/`logical_not`** — `test_eltwise_unary_sfpu_threshold` forces exact threshold hits (0.5 / 0.0).
- **`atan2`, `logsigmoid` (mid-range), `isclose`, float `eq`/`ne`, `rsub_int32` negative round-trip, float `mask`, `signbit`** — crafted stimuli hit the interesting branches.

### Biggest actionable gaps (recommended new edge tests)
- **Div/mod by zero:** `SfpuElwdiv`, `SfpuBinaryFmod`, `SfpuBinaryRemainder`, `SfpuDivInt32`, `SfpuRemainderInt32`, `SfpuFmodInt32`, `addcdiv`, `snake_beta` — the `b==0`/`c==0`→±inf/NaN branches are never exercised.
- **`reciprocal`/`rdiv` at/near 0** — the pole is `Excluded`; behavior at 0 is unverified.
- **Negative-operand integer paths** — signed vs unsigned max/min, `div_int32` vs `div_int32_floor`, `fmod_int32` vs `remainder_int32` (trunc-vs-floor distinction only shows for mixed signs), bitwise OR/XOR sign-magnitude fixup, gcd/lcm with 0/negatives, INT32_MIN for `abs_int32`.
- **Overflow / format extremes** — `mul_int32`, `add_int32`/`sub_int32` (perf-only, no functional assert), float overflow-to-inf, typecast float→int saturation/wrap and INT32_MIN/UINT32_MAX.
- **NaN/inf into generic ops** — max/min NaN propagation, `where` NaN branches, exp/log/pow non-finite inputs.
- **Exact-tie rounding** — `round` half-to-even ties, `floor`/`ceil`/`trunc` at exact integers, `sign`/comparison-to-zero at exact 0 / -0.0.
- **Unary shift ops** use a *fixed* shift of 3 with positive inputs — the ≥32 / negative / negative-value edges exist only in the *binary* shift test.
- **Entirely untested kernels** (header exists; no enum entry, no test): `welfords`, `dropout`, `quant`, `cumsum`, `reshuffle_rows`, `int_sum`, `tiled_prod`, `copy_dest_values`, `generalized_moe_gate_topk`, `max_pool_indices`, `rand`. `TopKLocalSort/Merge/Rebuild` are perf-only (no correctness test). `swiglu` is Quasar-only.

---

## 1. Unary — transcendental, trigonometric & special functions


**Framing note (applies to every row).** Two nightly-gated drivers in `test_sfpu_unary.py` cover this group, and they sweep very different domains:

- **`test_eltwise_unary_sfpu_float`** (ops in `ALL_MATHOPS`): calls `eltwise_unary_sfpu()` with **no `spec_A`**, so `generate_stimuli` falls back to `default_spec_for_format` (`helpers/stimuli_generator/generator.py:231`). For fp32/fp16/bf16 that is `StimuliSpec.uniform(low=0.1, high=1.1)`; for Bfp8_b/Bfp4_b it is a positive `[0, ~2.9]` face. **It never generates negatives and never uses the per-op `_OP_DOMAIN_REGISTRY` domains.** So for these ops the genuine domain/saturation/tail edges are effectively *not* reached by the sweep.
- **`test_eltwise_unary_sfpu_domain`** (ops in `DOMAIN_MATHOPS`): builds `spec_A = exclude_undefined(mathop, for_op(mathop, fmt).spec_A)`, i.e. the real per-op safe domain minus registered undefined holes. Covers the safe interior well but deliberately avoids the undefined boundaries.

Neither driver injects NaN/±inf/denormals/format-extremes for this group (only `test_eltwise_unary_sfpu_isinf_isnan` does, and it covers none of these ops). The one dedicated special-value test is `test_exponential_clamp_negative` (Exp only). Both drivers are `@pytest.mark.nightly`.

| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| exp (Exp) | e^x; approx/fast paths + `_sfpu_exp_accurate_`; `CLAMP_NEGATIVE` sanitizes negative tail near x≈-88.5 | Large negative / clamp boundary (x≈-88.5, -100…-10000) | Yes | `test_exponential_clamp_negative` drives x=-88.5,-100,-200,-1000,-10000 with `CLAMP_NEGATIVE` True/False, asserts ≤0 and isclose |
| exp (Exp) | " | Positive overflow → +inf | No | Float sweep only hits [0.1,1.1]; `_exp_spec` (high=80 fp32) would avoid overflow anyway; +inf never driven |
| exp (Exp) | " | NaN / ±inf input | No | No special-value injection for exp except the clamp test (finite) |
| exp (Exp) | " | Negative inputs generally | Partial | Only via `test_exponential_clamp_negative`; the `ALL_MATHOPS` float sweep is positive-only ([0.1,1.1]) |
| exp2 (Exp2) | 2^x via `_calculate_exp2_` | Overflow / large \|x\| | No | `ALL_MATHOPS` float sweep positive [0.1,1.1]; `_exp2_spec` (±100 fp32) unused by that driver; overflow never driven |
| exp2 (Exp2) | " | Negative inputs, NaN/±inf | No | Float sweep positive-only; no special values |
| exp_with_base (ExpWithBase) | exp(0.5·x), SCALE_EN | Positive amplification / error tail | Excluded | `_exp_with_base_spec` caps high=32 (arg≤16) precisely to keep error in rtol; larger args not tested |
| exp_with_base (ExpWithBase) | " | Negative reach (arg≈-50) / NaN / ±inf | No | Domain sweep [-100,32]; boundary at ~-88.5 not asserted; no special values |
| log (Log) | ln(x); poly on [1,2] + exp-correction; bf16 vs fp32 paths | x=0 → -inf | Excluded (float-sweep: No) | Kernel handles `in==0`→-inf; `ALL_MATHOPS` sweep only [0.1,1.1] so 0 never generated; `_SFPU_UNDEFINED_RANGES[Log]=(-inf,1e-6]` is only applied by the *domain* driver, which does not run Log |
| log (Log) | " | Negative x → NaN (bf16 path returns inf instead) | No | Kernel `in<0`→NaN; float sweep positive-only, never driven; bf16/fp32 divergence unverified |
| log (Log) | " | +inf→+inf, exp==128 (inf/NaN input)→NaN | No | Handled in `calculate_log_*_body` but never injected |
| log_with_base (LogWithBase) | log₂(x) (base-scaled log) | x≤0 boundary | Excluded | `for_op` domain is `LOG_UNIFORM [1e-4,1e3]` (positive only); no `_SFPU_UNDEFINED_RANGES` entry, so ≤0 simply never generated |
| log_with_base (LogWithBase) | " | Very small x (→ large negative) / NaN·inf | Partial / No | Domain reaches 1e-4 (safe interior covered); inf/NaN not injected. Also skipped under coverage (tt-llk#883) |
| log1p (Log1p) | log(1+x); Juffa bit-level 2^k·t reduction | x=-1 → -inf, x<-1 → NaN | Excluded | Kernel: `u<0`→NaN, `u==0`→-inf. `ALL_MATHOPS` float sweep [0.1,1.1] never reaches ≤-1; `_SFPU_UNDEFINED_RANGES[Log1p]=(-inf,-1+1e-6]` only applied by domain driver (Log1p not in it) |
| log1p (Log1p) | " | Near-zero (cancellation region) | Partial | Reformulation is stable near 0; float sweep [0.1,1.1] touches small positive but not negative side |
| log1p (Log1p) | " | +inf/NaN passthrough (`u>=inf`→u) | No | Handled but not injected |
| sqrt (Sqrt) | √x, `_calculate_sqrt_`; FAST_APPROX path | x<0 (undefined) | Excluded (float-sweep: No) | `_SFPU_UNDEFINED_RANGES[Sqrt]=(-inf,0]` applies only in domain driver (Sqrt not in it); float sweep positive [0.1,1.1] never negative |
| sqrt (Sqrt) | " | x=0 → 0 | No | `for_op` low=0.0 but float sweep uses [0.1,1.1]; exact 0 never deliberately driven |
| sqrt (Sqrt) | " | FAST_APPROX mode | Partial | In `SUPPORTED_FAST_MODE_OPS`; `_float_sweep_params` sweeps FastMode.Yes — but only over [0.1,1.1] |
| sqrt (Sqrt) | " | +inf/NaN | No | Not injected |
| sqrt_custom (SqrtCustom) | fast inv-sqrt seed + 2 Newton steps, ×val; `v_if(val!=0)` | x=0 → returns 0 branch | No | Domain `[0,100]` includes 0 but random uniform won't hit it; the `val!=0` branch not deliberately driven |
| sqrt_custom (SqrtCustom) | " | x<0 | No | No undefined-range entry; `for_op` domain is [0,100] so negatives never generated |
| rsqrt (Rsqrt) | 1/√x; FAST_APPROX path | x≤0 / x=0 → +inf | Excluded | `for_op` LOG_UNIFORM [1e-4,100] minus `_SFPU_UNDEFINED_RANGES[Rsqrt]=(-inf,1e-6]`. **But Rsqrt is in `ALL_MATHOPS`, not `DOMAIN_MATHOPS`**, so the float driver actually runs it over [0.1,1.1] and never applies the exclusion; either way ≤0 not driven |
| rsqrt (Rsqrt) | " | FAST_APPROX mode | Partial | In `SUPPORTED_FAST_MODE_OPS`; FastMode.Yes swept over [0.1,1.1] |
| rsqrt (Rsqrt) | " | Small x → very large output | No | Float sweep min is 0.1; the 1e-4 tail (registry) is not used by the ALL_MATHOPS driver |
| rsqrt_compat (RsqrtCompat) | legacy reciprocal-root approx | Small-x (large output) accuracy loss | Excluded | `for_op` LOG_UNIFORM `[1e-2,100]` intentionally tighter than accurate rsqrt to dodge the low-end error; not verified below 1e-2 |
| rsqrt_compat (RsqrtCompat) | " | x≤0 / x=0 branch | No | Domain positive-only; no undefined-range entry; `val==0` branch not driven |
| square (Square) | x² | Overflow (large \|x\|) | No | `ALL_MATHOPS` float sweep [0.1,1.1]; `_square_spec` (±1000 fp32) unused there and wouldn't overflow anyway |
| square (Square) | " | Negative / -0 sign path | No | Float sweep positive-only |
| cbrt (Cbrt) | ∛x, magic-constant + Newton; `setsgn` for sign | Negative inputs (sign path) | Yes | Domain driver, `for_op` UNIFORM [-27,27] spans both signs, exercising `setsgn(d,a)` |
| cbrt (Cbrt) | " | x=0 | No | 0 not deliberately generated within [-27,27] |
| cbrt (Cbrt) | " | NaN/±inf | No | Not injected |
| sigmoid (Sigmoid) | 1/(1+e^-x); exp_accurate/exp_21f + reciprocal | Saturation tails | Yes | Domain driver, `for_op` UNIFORM [-8,8] covers both saturation regions |
| sigmoid (Sigmoid) | " | Extreme \|x\| / inf / NaN | No | Beyond ±8 not generated; no special values |
| sigmoid_appx (SigmoidAppx) | 3-segment LUT + 0.5 | LUT knees | Yes | Domain [-5,5]; custom tol (0.13,0.05) in `DOMAIN_CUSTOM_TOLERANCES` acknowledges ~0.12 abs error at knees |
| sigmoid_appx (SigmoidAppx) | " | Saturation beyond ±5 | No | Domain capped at ±5 |
| silu (Silu) | x·sigmoid(x) | Saturation + linear regions | Partial | **In `ALL_MATHOPS`** → float driver runs [0.1,1.1] positive only; negative branch (the interesting part) and `for_op`'s [-5,5] domain are NOT used |
| silu (Silu) | " | Large negative tail (→0), NaN/inf | No | Never reached by positive-only float sweep |
| gelu (Gelu) | x·Φ(x), piecewise CDF: identity ≥2.78, core [-3.125,2.78), exp region, zero ≤-13.19 | Negative/exp-tail & zero-saturation regions | No | **In `ALL_MATHOPS`** → float sweep [0.1,1.1] hits only the core; identity, exp-tail (-13.19,-3.125), and zero-saturation (≤-13.19) branches never exercised (registry Gaussian std=3 unused here). Also Gelu skipped under coverage (tt-llk#883) |
| gelu (Gelu) | " | Identity region (x≥2.78) | No | Max float-sweep input 1.1 (bfp ~2.9) never reaches 2.78 |
| gelu_tanh (GeluTanh) | tanh approximation of gelu | Saturation tails / ±0 sign path | Partial | **In `ALL_MATHOPS`**, positive [0.1,1.1] only; registry Gaussian(std=3) domain unused; negative tail & sign path not reached |
| gelu_appx (GeluAppx) | LUT approx of gelu | LUT knees + tails | Partial | Domain driver, Gaussian(mean0,std3,clip[-5,5]); custom tol (0.13,0.05). Covers near-0 + partial tails; skipped under coverage |
| gelu_derivative (GeluDerivative) | Φ(x)+x·φ(x); regions: ≥3.17→1, core[-3,3.17], [-5,-3], (-13.375,-5] fused exp, ≤-13.375→0 | Deep negative tail & zero-saturation (≤-13.375) | Partial | Domain driver Gaussian(std3, clip[-5,5]) reaches only to -5; the (-13.375,-5] fused-exp branch and zero-saturation region not driven; skipped under coverage |
| gelu_derivative (GeluDerivative) | " | "hump" >1 region [0.77,3.16] + right saturation (≥3.17) | Yes | Within clip[-5,5]; both covered probabilistically |
| tanh (Tanh) | poly(bf16)/2·sigmoid(2x)-1(fp32); poly threshold at \|x\|=0.6; LUT approx path | Saturation (\|x\|>~3 → ±1) | Partial | **In `ALL_MATHOPS`**, [0.1,1.1] (+bfp ~2.9) barely reaches saturation; negatives never; the 0.6 poly-switch touched on positive side only |
| tanh (Tanh) | " | Approx-mode LUT path | No | `test_eltwise_unary_sfpu_float` does `pytest.skip` for Tanh+approx (line 238) — LUT tanh path untested |
| tanh (Tanh) | " | NaN/inf, negative saturation | No | Positive-only sweep; no special values |
| tanh_derivative (TanhDerivative) | sech²(x); core poly \|x\|<3, tail exp(-2\|x\|+ln4) to 45, 0 beyond | Deep tail & zero-saturation (\|x\|≥45) | No | Domain driver `for_op` UNIFORM [-5,5]; reaches only start of tail region; ≥45 zero-saturation never driven |
| tanh_derivative (TanhDerivative) | " | Core↔tail boundary (\|x\|=3) | Yes | [-5,5] spans the 3.0 `CORE_REGION_LIMIT` |
| tanh_derivative_lut (TanhDerivativeLut) | legacy 1-tanh²(x) via LUT; catastrophic cancellation for \|x\|>3.4 (Max ULP 15140) | Known-bad region \|x\|>3.4 | Excluded | `for_op` UNIFORM [-3,3] deliberately clipped inside the LUT's accurate region; the documented failure beyond 3.4 is NOT verified |
| sin (Sin) | 4-stage Cody-Waite reduction + odd-poly | Argument reduction (large \|x\|), negatives | No | **In `ALL_MATHOPS`** → float sweep [0.1,1.1]: less than one quadrant, positive only; registry's [-π,π] domain unused; the `j=round(v/π)` reduction path never exercised |
| sin (Sin) | " | NaN/±inf | No | Not injected |
| cos (Cos) | Cody-Waite reduction by π/2, quadrant parity | Argument reduction, negatives, full circle | No | **In `ALL_MATHOPS`**, [0.1,1.1] positive only; registry [-π,π] domain unused; quadrant-sign paths untested |
| tan (Tan) | Cody-Waite π/2 reduction + reciprocal-correction branch (`i<0`) near poles | Poles at ±π/2 | Excluded | Domain driver `for_op` UNIFORM [-1.3,1.3] stays inside the poles; pole behavior not verified |
| tan (Tan) | " | Large \|x\| arg reduction; NaN/inf | No | Domain only [-1.3,1.3]; reduction of multiple periods not driven |
| atan (Atan) | arctan; reciprocal branch for \|x\|>1; explicit NaN handling | Reciprocal branch (\|x\|>1) + tails | Yes | Domain driver [-10,10] spans both signs and crosses \|x\|=1 (uses `sfpu_reciprocal`) |
| atan (Atan) | " | NaN input (kernel returns NaN), ±inf (→±π/2) | No | Kernel handles `exp==255&&mant!=0`→NaN but not injected |
| asin (Asin) | asin, range-reduction at \|x\|=0.625, sqrt near ±1 | \|x\|>1 → NaN, \|x\|=1 boundary | Excluded | `for_op` UNIFORM [-0.99,0.99]; kernel guards `abs(v)<=1` else NaN but boundary/beyond not driven |
| asin (Asin) | " | Range-reduction knee (\|x\|=0.625) | Yes | [-0.99,0.99] spans 0.625 both signs |
| acos (Acos) | π/2 − asin | \|x\|>1 → NaN, \|x\|=1 boundary | Excluded | Same guard/domain as asin; boundary not driven |
| sinh (Sinh) | (e^x − e^-x)/2 via exp_21f | Overflow at large \|x\| | No | Domain [-5,5] keeps exp bounded; overflow tail not driven |
| sinh (Sinh) | " | Near-0 cancellation, NaN/inf | Partial / No | [-5,5] covers small x; no special values |
| cosh (Cosh) | (e^x + e^-x)/2 via exp_21f | Overflow at large \|x\| | No | Domain [-5,5]; overflow not driven |
| atanh (Atanh) | log1p reformulation; domain \|x\|<1 | \|x\|→1 → ±inf, \|x\|≥1 | Excluded (float-sweep gap below) | `_SFPU_UNDEFINED_RANGES[Atanh]=(-inf,-1+1e-6),(1-1e-6,inf)` applied only by domain driver — **but Atanh is in `ALL_MATHOPS`**, so float sweep runs [0.1,1.1] and even generates values in (1,1.1] (undefined) without any exclusion; ±1 boundary not deliberately asserted |
| atanh (Atanh) | " | Small-x region (old-form cancellation) | Partial | Reformulation stable; float sweep touches [0.1,1.1] positive only |
| asinh (Asinh) | arcsinh; all reals | Large \|x\|, negatives | No | **In `ALL_MATHOPS`**, [0.1,1.1] positive only; registry [-10,10] unused |
| acosh (Acosh) | arccosh; domain x≥1 | x<1 (undefined) & x=1 boundary | No / Excluded | `_SFPU_UNDEFINED_RANGES[Acosh]=(-inf,1.0)` applies only in domain driver — **but Acosh is in `ALL_MATHOPS`**, so float sweep runs [0.1,1.1], generating mostly x<1 (undefined → NaN) with no exclusion; genuine x≥1 range and x=1 boundary not deliberately tested |
| erf (Erf) | piecewise poly: 0/[0,1)/[1,3)/≥3→1 | Saturation (\|x\|≥3 →±1) + transition through 0 | Yes | Domain driver `for_op` UNIFORM [-3,3] spans both tails and 0; segment boundaries at 0,1,3 covered |
| erf (Erf) | " | Beyond ±3, NaN/inf | No | Domain capped at ±3 (saturates anyway); no special values |
| erfc (Erfc) | 1∓erf(x) | Tails + transition | Yes | Domain [-3,3]; both branches (`x<0`) covered |
| erfc (Erfc) | " | Beyond ±3, NaN/inf | No | Domain capped at ±3 |
| erfinv (Erfinv) | Winitzki approx via log(1-x²)+sqrt; explicit \|x\|=1→inf, >1→NaN | \|x\|=1 → ±inf, \|x\|>1 → NaN | Excluded | `for_op` UNIFORM [-0.99,0.99] minus `_SFPU_UNDEFINED_RANGES[Erfinv]` (open interval); kernel's ±1/beyond branches not driven |
| erfinv (Erfinv) | " | Interior incl. sign path | Yes | [-0.99,0.99] spans both signs (`setsgn`) |
| expm1 (Expm1) | e^x−1; Taylor \|x\|<0.4(bf16)/0.5(fp32), else exp-1 | Small-x cancellation region | Yes | Domain driver [-5,5] straddles the 0.4/0.5 Taylor↔exp switch, verifying cancellation avoidance |
| expm1 (Expm1) | " | Large positive (overflow), large negative (→-1), NaN/inf | No | Domain [-5,5]; overflow & extreme tails not driven |
| expm1_cw (Expm1Cw) | component-wise expm1 (golden = torch.expm1); shared CW helper clamps x to -87 | Underflow clamp (x≈-87) | No | `for_op` UNIFORM [-5,5] (same as Expm1); the `max(x,-87)` clamp never reached |
| expm1_cw (Expm1Cw) | " | Small-x cancellation switch | Yes | [-5,5] crosses the Taylor↔exp threshold |
| i0 (I0) | modified Bessel I₀, degree-10 poly in x² | Beyond poly-valid \|x\|>3.75 (asymptotic region) | Excluded | `for_op` UNIFORM [-3.75,3.75] clips exactly to the poly-valid range; asymptotic region not verified |
| i0 (I0) | " | x=0 (→1), NaN/inf | No | 0 not deliberately generated; no special values |
| i1 (I1) | modified Bessel I₁, x·poly(x²) | Beyond \|x\|>~3.75 | Excluded | `for_op` UNIFORM [-3.75,3.75]; asymptotic region not tested |
| i1 (I1) | " | Sign path (odd function) | Yes | [-3.75,3.75] spans both signs |
| lgamma (Lgamma) | Stirling + Bernoulli + reflection for x<0.5; poles at x≤0 | Poles at non-positive integers (x≤0) | Excluded | `for_op` UNIFORM [1,15] keeps well clear of poles |
| lgamma (Lgamma) | " | Reflection branch (x<0.5) | No | Domain starts at 1.0; the `in<0.5` reflection path never exercised |
| lgamma (Lgamma) | " | Bridge special-cases x=1,2 (→0) & x≈0.5 | No | In [1,15] but 1.0/2.0 not deliberately hit; boundary near 0.5 outside domain |
| digamma (Digamma) | ψ(x), 2-segment piecewise rational P/Q, LUT range [0.01,102] | Poles at x≤0 | Excluded | `for_op` UNIFORM [0.1,50] positive only; poles avoided |
| digamma (Digamma) | " | Outside LUT range (<0.01 or >102) | No | Domain [0.1,50] stays inside LUT fit; extrapolation not tested |
| digamma (Digamma) | " | Segment boundary (~51.005) | Partial | [0.1,50] just misses the second segment (>51) — high segment barely exercised |
| polygamma (Polygamma) | ψ⁽ⁿ⁾(x): exact 11-term sum + Euler-Maclaurin tail; poles at x≤0 | Poles at x≤0 | Excluded | `for_op` UNIFORM [0.5,10] positive only |
| polygamma (Polygamma) | " | Orders n≠1 (n is a runtime scale/param) | No | Registry comment fixes order 1 (trigamma); higher orders not swept |
| polygamma (Polygamma) | " | Small x near 0.5 (slow convergence), NaN/inf | Partial / No | [0.5,10] touches 0.5 lower edge; no special values |


## 2. Unary — activation, piecewise & rounding


**Cross-cutting finding (applies to every `ALL_MATHOPS` op below).** `test_eltwise_unary_sfpu_float` calls `eltwise_unary_sfpu(...)` with **no `spec_A`**, so stimuli come from `default_spec_for_format` (`helpers/stimuli_generator/generator.py:257`) = `uniform(low=0.1, high=1.1)` for Float32/Float16/Float16_b, and the Bfp8_b/Bfp4_b face defaults (`_default_bfp8b_face`/`_default_bfp4b_face`, lines 207–220) which are also **non-negative** (`[0, ~2.9]`). The signed per-op ranges in `_OP_DOMAIN_REGISTRY` are consumed **only** by `test_eltwise_unary_sfpu_domain` (`DOMAIN_MATHOPS`, Float16_b+Float32 only). Consequence: for `ALL_MATHOPS` ops the entire float sweep sees only positive inputs, so any `x<0` branch/knee is never exercised. No special values (+inf/-inf/NaN/-0.0) are injected for any op in this group (only `ISINF_ISNAN_MATHOPS` get them, none of which are here). dest_acc No/Yes are both swept everywhere (covers the bf16-dest fp16b-rounding path).

| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| Abs (`calculate_abs`) | \|x\| | Negative branch | No | In `ALL_MATHOPS` only → positive-only `[0.1,1.1]`; abs is identity there, the sign-flip path is never hit. Registry `Abs:[-10,10]` is unused (abs not in `DOMAIN_MATHOPS`). |
| Abs | \|x\| | -0.0, +inf/-inf, NaN | No | No special values injected. |
| Neg (`_calculate_negative_`) | -x | Negative input | No | `ALL_MATHOPS` positive-only; only positive→negative direction tested. |
| Neg | -x | -0.0 / +-inf / NaN | No | Not injected. |
| Celu (`calculate_celu`) | x≥0: x;  x<0: α(exp(x/α)-1) | exp branch (x<0) | No | `ALL_MATHOPS` positive-only `[0.1,1.1]` → only the `v>=0` pass-through runs; the `v<sfpi::vConst0` exp branch is never entered on the float sweep. |
| Celu | knee at x=0 | Exact 0 / -0.0 | No | Domain starts at 0.1; 0 never sampled. |
| Elu (`calculate_elu`→`expm1_cw_clamped`) | x≥0: x; x<0: α·expm1(x) | exp branch (x<0) | No | `ALL_MATHOPS` positive-only → `v_if(x>=0)` always true; expm1 negative branch untested. Approx-mode+bf8_b combo skipped (test line 251). |
| Selu (`calculate_selu`) | scale·(max(0,x)+min(0,α(eˣ-1))) | Both branches at x=0 | Yes (partial) | `DOMAIN_MATHOPS`, `Selu:[-5,5]` spans both `v>=0` and `v_else`. Only Float16_b/Float32; exact 0, special values not covered. |
| Softplus (`calculate_softplus`) | (1/β)ln(1+e^{βx}); linear above threshold | 4 poly/exp knees (-5, 4) + linear passthrough (thr=20) | Yes | `Softplus:[-5,30]` deliberately crosses the linear threshold 20 (registry comment) and the -5/4 branch splits. Both signs covered (Float16_b/Float32). |
| Softplus | β·x underflow / +-inf / NaN | No | Not injected. |
| Softsign (`calculate_softsign`) | x/(1+\|x\|) | Division safe (denom≥1) | Yes | `Softsign:[-5,5]` both signs; reciprocal arg always ≥1 so no pole. |
| Softsign | -0.0, large \|x\| overflow | No | Not injected; domain only ±5. |
| Softshrink (`calculate_softshrink`) | x∓λ outside ±λ(=0.5), else 0 | Zero band + both shrink branches | Yes | `Softshrink:[-5,5]` spans all three. |
| Softshrink | Exact ties x=±0.5 (v>λ strict) | No | Continuous uniform, measure-zero; boundary equality (`v>lambda` strict, so x=0.5→0) not asserted. |
| Hardshrink (`calculate_hardshrink`) | x if \|x\|>λ(0.5) else 0 | Both knees | Yes | `Hardshrink:[-4,4]` spans past ±0.5. |
| Hardshrink | Exact \|x\|=λ (`abs_v<=lambda`→0) | No | Boundary equality not driven (measure-zero). BF16-vs-FP32 λ pre-rounding path noted in header but not edge-tested. |
| Hardsigmoid (`ActivationImpl<Hardsigmoid>`, slope 1/6 +0.5, relu_max@1) | clamp((x+3)/6,0,1) | Clip knees at x=±3 | No | In `ALL_MATHOPS` → positive-only `[0.1,1.1]`; only the interior linear region runs. Registry `Hardsigmoid:[-4,4]` (which spans both knees) is unused. |
| Hardtanh (`calculate_hardtanh`) | clamp(x,-1,1) via SFPSWAP | Past both bounds | Yes | `Hardtanh:[-2,2]` spans past ±1 (Float16_b/Float32). |
| Hardtanh | Exact x=±1 boundary; NaN ordering in SFPSWAP | No | Boundary equality and NaN-min/max not driven. |
| Hardmish (`hardmish`) | x·clamp(0.5x+1,0,1) | Knees at x=-2 and x=0 | Yes | `Hardmish:[-4,4]` spans past both clamp knees. |
| Hardmish | x=-inf → (-inf)·0 = NaN (documented) | No | -inf not injected. |
| Mish (`calculate_mish`) | x·tanh(softplus(x)); ≈x for x≥8 | Saturation knee x≥8 | No | `Mish:[-5,5]` — never reaches the `x<SAT_HI(8)` cutover, so the `result=x` saturation branch is untested. |
| Mish | Transition/negative region | Yes | `[-5,5]` exercises the exp+reciprocal path both signs. |
| Relu (`_relu`) | max(0,x) | Any coverage on WH/BH | No | `MathOperation.Relu` absent from `ALL_MATHOPS` and `DOMAIN_MATHOPS`; only exercised in `quasar/test_eltwise_unary_sfpu_quasar.py` (uniform_spec). No WH/BH negative-branch/zero test. |
| ReluMax (`relu_max`, thr=5) | clamp(x,0,5) | Upper clamp (x>5) and lower (x<0) branches | No | `ALL_MATHOPS` positive-only `[0.1,1.1]` → all inputs in (0,5); neither the `a>threshold` nor the `a<0` branch fires — pure pass-through. |
| ReluMin (`relu_min`, thr=5) | max(x,5) | Whole op | No | **Skipped** unconditionally — `pytest.skip(...tt-llk#1120)` (test line 235). Untested. |
| Lrelu (`_calculate_lrelu_`, slope 0.1) | x≥0:x; x<0:0.1x | Negative (scaled) branch | Yes | `DOMAIN_MATHOPS`, `Lrelu:[-5,5]` spans both (Float16_b/Float32). |
| Lrelu | knee at exact 0 | No | 0 measure-zero; not asserted. |
| Prelu (`calculate_prelu`, slope 0.25) | x<0: 0.25x else x | Negative branch | Yes | `Prelu:[-5,5]` both signs. |
| Threshold (`_calculate_threshold_`, thr=5,val=10) | x≤thr→val else x | Pass-through branch (x>thr) | No | `ALL_MATHOPS` positive-only `[0.1,1.1]` — every input ≤5 → constant output 10; the `in>threshold` pass-through is never exercised (also makes output constant). |
| Clamp (`calculate_clamp`, [-1,1]) | min(max(x,-1),1) | Past both bounds | Yes | `Clamp:[-2,2]` spans past ±1 (Float16_b/Float32). |
| Clamp | Exact x=±1 / int32 clamp variant | No | Boundary equality and `calculate_clamp_int32` not driven by this test. |
| Xielu (`calculate_xielu`, β=0.5) | 4 regions: x>0 quad; [eps,0); (-0.5,0) Taylor; ≤-0.5 exp | x>0, moderate-neg, large-neg regions | Yes (partial) | `Xielu:[-5,5]` hits `x>0`, `x>-0.5` Taylor, and `x≤-0.5` neg-exp branches. |
| Xielu | "very small negative" region [eps=-1e-6, 0) | No | Interval width ~1e-6; random `[-5,5]` effectively never lands there → the `x>=vConstFloatPrgm1` branch untested. |
| Tanhshrink (`calculate_tanhshrink`) | x-tanh(x); small-x poly (\|x\|≤1) vs large-x (\|x\|>1) | Both paths | Partial | `ALL_MATHOPS` positive-only `[0.1,1.1]`: exercises the small-\|x\| poly path fully but only barely crosses into the `ax>1` path (1.0–1.1); negative (odd-function) side and the fp32 clamp-at-9 saturation tail untested. |
| Heaviside (`calculate_heaviside`, val=0.5) | x<0→0; x>0→1; x==0→val | x==0 branch | No | `Heaviside:[-5,5]` covers `<0`/`>0`; the exact-0 `v_else` (returns 0.5) never sampled. |
| Fill (`_fill`, const=5) | output := const (input ignored) | Input value irrelevant | Yes | HW ignores input (`Fill:[0,1]`); trivially covered. |
| Identity (`calculate_identity`) | pass-through | +-inf/NaN/-0.0 preservation | No | `Identity:[-10,10]`; no special values injected. `calculate_identity_uint` path not driven here. |
| Add1 (`calculate_add1`) | x+1 | Cancellation near x=-1 / +-inf | No | `Add1:[-10,10]` covers -1 region loosely but no exact/─inf test. |
| Floor (`_floor_body_`) | ⌊x⌋ | Negative-value branch (`v-1` correction) | No | `ALL_MATHOPS` positive-only `[0.1,1.1]` → `_floor_body_` negative correction path never runs; result only 0 or 1. |
| Floor | Exact integers, +-inf, NaN | No | No integer/tie/inf stimuli; header includes isinf/isnan but not driven. |
| Ceil (`_ceil_body_`) | ⌈x⌉ | Negative branch, exact integers | No | Positive-only sweep; `v+1` positive-correction only, no negative path, no exact-integer/inf test. |
| Trunc (`_trunc_body_`) | truncate toward 0 | Negative values, exact integers, exp<0 mask | Partial | Positive-only `[0.1,1.1]`: exercises the `exp<0`→0 mask (for 0.1–1.0) and passthrough (1.0–1.1) but never negative inputs, exact integers, or inf. |
| Frac (`_calculate_frac_`) | x-trunc(x) | Negative fractional (sign) | No | Positive-only; sign-of-x fractional path untested; no inf. |
| Round (`_round_even_`, decimals=0) | round-half-to-even | Exact half-ties (x.5) | No | `Round:[-10,10]` both signs, but continuous uniform never lands on representable `.5` ties → banker's-rounding tie behavior unverified. |
| Round | +-inf / NaN (golden returns x) | No | Not injected. Non-zero `decimals` (pow10 table, ±38/±45 clamps) not exercised (fixed decimals=0). |
| Sign (`_calculate_sign_`) | -1 / 0 / +1 | x==0 → 0 | No | `Sign:[-2,2]` covers ±; exact 0 (and -0.0) never sampled → the zero output branch unverified. |
| Signbit (`calculate_signbit`, shift>>31) | 1 if sign bit set | Both signs | Yes | Dedicated `test_eltwise_unary_sfpu_signbit` with intervals `[(-100,-0.5),(0.5,100)]`; also `DOMAIN_MATHOPS Signbit:[-2,2]`. |
| Signbit | -0.0 (should return 1) | No | Dedicated test **explicitly excludes** 0 ("sidestep -0.0/rounding ambiguity", test line 476) → signbit(-0.0)=1 path never verified. |
| Reciprocal (`calculate_reciprocal`) | 1/x | x=0 / near-zero pole | Excluded | `_SFPU_UNDEFINED_RANGES[Reciprocal]=(-1e-6,1e-6)` + registry intervals `[(-100,-0.1),(0.1,100)]` (domain path). Behavior at/near 0 not verified. |
| Reciprocal | Negative side (float sweep) | Partial | In `ALL_MATHOPS`→positive-only `[0.1,1.1]`; negatives covered only via the domain-registry path if `Reciprocal` ran there — it does not (`Reciprocal` is in `ALL_MATHOPS`, not `DOMAIN_MATHOPS`), so the float sweep tests positive x only. |
| Rdiv (`calculate_rdiv`, value=2.0) | 2/x | x=0 pole; negative divisor | No | `Rdiv:[1,8]` is **positive-only** → the reciprocal blow-up at 0 and negative-x path are never tested. |
| Rdiv | Trunc/Floor rounding-mode variants | No | Kernel has `RoundingMode::{Trunc,Floor}`; golden/test use plain (none) only. |
| Rpow (`calculate_rpow`→`_sfpu_binary_power_`, base=2.0) | 2^x | Negative-base / NaN paths | No | Base fixed positive (2.0) → the `base<0`/non-integer-power NaN handling and `base==0` NaN path in the shared power kernel are unreachable. Exponent `[-4,4]`. |
| UnaryPower (`calculate_unary_power`, exp=2.0) | x^2 | Neg-base × even power (→ +) | Yes (partial) | `UnaryPower:[-4,4]` includes negative bases → `base<0` sign-fixup with even integer power exercised. |
| UnaryPower | 0^negative NaN, non-integer power NaN, overflow clamp | No | Exponent fixed 2.0 (positive even int) → `IS_POSITIVE_EXPONENT` path only; the `abs_base==0 && pow<0`→NaN and non-integer-power→NaN branches, and the fp32 vs bf16 (`_61f_`/`_21f_`) accuracy split at boundaries, not edge-driven. |
| Fmod (`calculate_fmod`, divisor=2.0) | fmod(x,2) | Both signs | Yes | `Fmod:[-5,5]`; sign carried via `setsgn(v,val)`. |
| Fmod | Divisor s==0 → NaN | No | Divisor fixed 2.0 → the `v_if(s==0)` NaN branch unreachable. |
| Remainder (`calculate_remainder`, divisor=2.0) | remainder(x,2) | Both signs, sign-of-divisor fixups | Yes | `Remainder:[-5,5]` exercises `val<0` and `value_tmp<0` correction paths. |
| Remainder | Divisor 0 → NaN; negative divisor | No | Divisor fixed +2.0 → `s==0` NaN and negative-divisor `value_tmp<0` branch (value_tmp>0 here) untested. |
| CastFp32ToFp16a (`cast_fp32_to_fp16a`, SFP_STOCH_RND) | round mantissa fp32→fp16a (10-bit), exponent kept | Magnitudes > fp16 max (65504) preserved, not overflowed | Yes | `CastFp32ToFp16a:[-1e5,1e5]` spans well past 65504 → the "no exponent clamp" behavior is exercised. Round-half-to-even at bit level modeled in golden. |
| CastFp32ToFp16a | +-inf / NaN (bit-pattern passthrough), denormals | No | Non-finite inputs (golden special-cases exponent 0xFF) not injected; denormals/format-min not swept. |


## 3. Unary — comparison, predicate, integer, bitwise, shift & typecast


| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| equal_zero (EqualZero) | `x==0 ? 1:0` (float path `calculate_comp`, uses `_sfpu_is_fp16_zero_`) | exact `x==0.0` (the only "true" input) | No | `sfpu_domains.py:450` uniform(-2,2); continuous sweep never lands on 0.0, so the `=1` branch is essentially never driven. In `DOMAIN_MATHOPS` (`test_sfpu_unary.py:340`). |
| equal_zero | | `+0.0` vs `-0.0` (both should be zero) | No | Kernel uses fp16-zero magnitude test; golden `_equal_zero` uses `x==0.0` (both signs). Neither -0.0 nor +0.0 deliberately injected. |
| equal_zero | | NaN / ±inf input | No | No special values injected into comparison-to-zero sweep. |
| equal_zero | uint32/uint16 int paths (`calculate_eqz_uint32`, `calculate_comp_uint16`) | integer dest interpretation | No | Only float format sweep runs; the dedicated int/uint16 EqualZero kernel variants are never exercised. |
| not_equal_zero (NotEqualZero) | `x!=0 ? 1:0` | exact `x==0.0` (only "false" input) | No | `sfpu_domains.py:453` uniform(-2,2); 0.0 not hit, so the `=0` branch is never driven. ±0.0, NaN, inf not injected. |
| less_than_zero (LessThanZero) | `x<0 ? 1:0` | boundary `x==0` / `-0.0` | No | `sfpu_domains.py:456` uniform(-2,2) spans both signs (both branches fire) but exact 0 / -0.0 never hit. NaN/inf not injected. |
| greater_than_zero (GreaterThanZero) | `x>0 ? 1:0` | boundary `x==0` / `+0.0` | No | `sfpu_domains.py:459` uniform(-2,2); both signs covered, boundary and specials not. |
| less_than_equal_zero (LessThanEqualZero) | `x<=0 ? 1:0` | boundary `x==0` (inclusive branch) | No | `sfpu_domains.py:462` uniform(-2,2); exact 0 not hit so the `x==0 -> 1` inclusion edge is unverified. |
| greater_than_equal_zero (GreaterThanEqualZero) | `x>=0 ? 1:0` | boundary `x==0` (inclusive branch) | No | `sfpu_domains.py:465` uniform(-2,2); exact 0 not hit; specials not injected. |
| unary_gt (UnaryGt) | `x>0.5 ? 1:0` (`calculate_unary_gt`, threshold `_UNARY_COMP_THRESHOLD=0.5`) | exact `x==0.5` boundary | No | `sfpu_domains.py:396` uniform(-2,2); continuous sweep never lands on 0.5. Both output classes still produced. |
| unary_gt | | NaN input (`v>s` false for NaN) | No | No NaN injected in domain sweep. |
| unary_lt (UnaryLt) | `x<0.5 ? 1:0` | exact `x==0.5` boundary | No | `sfpu_domains.py:399` uniform(-2,2); 0.5 not hit. |
| unary_ge (UnaryGe) | `x>=0.5 ? 1:0` (built as `!(v<s)`) | exact `x==0.5` (inclusive branch) | No | `sfpu_domains.py:402` uniform(-2,2); the equality-inclusion edge at 0.5 is never driven. |
| unary_le (UnaryLe) | `x<=0.5 ? 1:0` (built as `!(v>s)`) | exact `x==0.5` (inclusive branch) | No | `sfpu_domains.py:405` uniform(-2,2); equality edge not driven. |
| unary_ne (UnaryNe) | `x!=0.5 ? 1:0` (float `calculate_unary_ne`) | exact `x==0.5` boundary | Yes | `test_eltwise_unary_sfpu_threshold` (`test_sfpu_unary.py:587`) with `_threshold_op_stimuli_spec`: `x[0::3]=0.5` forces guaranteed threshold hits so both branches fire. |
| unary_eq (UnaryEq) | `x==0.5 ? 1:0` (float `calculate_unary_eq`) | exact `x==0.5` boundary | Yes | Same threshold test (`test_sfpu_unary.py:566-577`) injects exact 0.5. Float32/Float16_b only; NaN/inf not injected. |
| unary_eq/unary_ne | int comparison path `calculate_comp_unary_int` (sign-mag→2's-comp) | integer eq/ne, sign-magnitude conversion | No | Tested ops route through the float threshold path only; the int `calculate_comp_unary_int` variant (incl. `sfpu_sign_mag_to_twos_comp`) is never exercised. |
| unary_max (UnaryMax) | `max(x,0.0)` via SFPSWAP (`calculate_unary_max_min`, value `_UNARY_MAX_MIN_VALUE=0.0`) | equal operand `x==0.0` | No | `sfpu_domains.py:409` uniform(-5,5); both sides of 0 covered but exact tie / -0.0 not driven. |
| unary_max | | NaN input into SFPSWAP min/max | No | No NaN injected; SFPSWAP NaN ordering unverified. |
| unary_min (UnaryMin) | `min(x,0.0)` via SFPSWAP | equal operand / NaN | No | `sfpu_domains.py:412` uniform(-5,5); tie and specials not driven. |
| unary_max_int32 (UnaryMaxInt32) | `max(x, 1000)` int, SFPSWAP with sign-mag inversion (`calculate_unary_max_min_int32`) | negative inputs / negative scalar (the SFPNOT invert branch) | No | `test_eltwise_unary_sfpu_int` (`test_sfpu_unary.py:311`), stimuli `uniform(0,2000)` straddling +1000 (`_int_unary_stimuli_spec`). Scalar fixed +1000, inputs positive-only, so the `msb==1` invert path (negative value/scalar) is never exercised. |
| unary_max_int32 | | INT32_MIN / INT32_MAX operands | No | Positive 0..2000 only; format extremes never driven. Golden `_unary_max_int32` exact-int (`golden_generators.py:2785`). |
| unary_min_int32 (UnaryMinInt32) | `min(x, 1000)` int | negative inputs / INT32_MIN / negative scalar branch | No | Same test; inputs 0..2000 positive, scalar +1000. Invert/negative branches unexercised. |
| unary_max_uint32 (UnaryMaxUint32) | `max(x,1000)` unsigned (`IS_UNSIGNED=true`) | full unsigned range incl. values with MSB set (>2^31) / UINT32_MAX | No | `test_eltwise_unary_sfpu_int` runs it under UInt32 (`_UINT32_INT_UNARY_OPS`, `test_sfpu_unary.py:288`) but stimuli only `uniform(0,2000)`. The high-bit unsigned branch and UINT32_MAX are never driven. Golden reuses `_unary_max_int32`. |
| unary_min_uint32 (UnaryMinUint32) | `min(x,1000)` unsigned | full unsigned range / UINT32_MAX / MSB-set values | No | Same: UInt32 format but 0..2000 stimuli only. |
| isinf (Isinf) | `is_inf(x) ? 1:0` | +inf and -inf inputs | Yes | `test_eltwise_unary_sfpu_isinf_isnan` (`test_sfpu_unary.py:523`); `_isinf_isnan_stimuli_spec` injects +inf/-inf/nan on regular strides. Float16_b+dest_acc=Yes skipped (bf16→fp32 unpack mangles -inf/nan). |
| isinf | | Float16 (A-exponent) NaN-remap of inf | No | Test uses only Float16_b/Float32; the A-exponent path (`handle_infinite_numbers`/`_torch_unary` inf→NaN) is not exercised for isinf. |
| isposinf (Isposinf) | `is_pos && is_inf` | +inf vs -inf discrimination | Yes | Same isinf/isnan test injects both ±inf; golden `_isposinf` (`golden_generators.py:2490`). |
| isneginf (Isneginf) | `is_neg && is_inf` | -inf, and -0.0/negative-finite not misclassified | Yes | Same test injects -inf; note Float16_b+dest_acc=Yes skipped because -inf not preserved through unpack. |
| isnan (Isnan) | `is_nan(x) ? 1:0` | NaN input (and inf not misclassified) | Yes | Same test injects nan (`x[2::7]=nan`). Only quiet NaN pattern from `float("nan")`; signaling-NaN bit patterns not covered. |
| isfinite (Isfinite) | `is_finite(x) ? 1:0` | ±inf / NaN → 0, finite → 1 | Yes | Same test carries all classes; golden `_isfinite` (`golden_generators.py:2499`). |
| add_int32 (AddInt32) | int32 add (unary wrapper `calculate_add_int32`; core `_add_int_`, sign-mag→2's-comp) | integer overflow / INT32_MIN/MAX operands | No | Perf-only: `perf_eltwise_unary_sfpu_int32.py:36-44` states no functional golden/assert (blocked by fast-tilize gap tt-llk #495). Not in `_INT_UNARY_OPS`/`DOMAIN_MATHOPS`. Integer core is *indirectly* exercised via binary `SfpuElwadd` (`_add_int_`) but the unary wrapper and its overflow edges are unverified. |
| sub_int32 (SubInt32) | int32 subtract (unary wrapper; core `_sub_int_`) | integer overflow / underflow / INT32_MIN | No | Perf-only, same note (`perf_eltwise_unary_sfpu_int32.py:43-44`). No functional golden/assert for the unary path. |
| abs_int32 (AbsInt32) | `\|x\|` int32 via `TTI_SFPABS` (`calculate_abs_int32`) | INT32_MIN (`abs(INT32_MIN)` overflow) | No | Perf-only (`perf_eltwise_unary_sfpu_int32.py:45`); no functional assert. INT32_MIN not representable in sign-magnitude Dst, so this critical edge is entirely unverified. |
| abs_int32 | | negative vs positive inputs | No | No functional sweep; kernel `SFPABS` behavior on sign-mag Dst untested. |
| left_shift (LeftShift, unary) | `x << 3` fixed (`calculate_left_shift`, `TT_SFPSHFT`) | shift ≥ 32 / negative shift | No | Shift amount is hard-fixed at 3 (`_int_shift_amount`, `golden_generators.py:2272`); the ≥32 and negative-shift edges only exist in the *binary* shift path (`test_sfpu_binary_int_shift_edge_cases`, `test_sfpu_binary.py:288`), not here. |
| left_shift | | overflow into sign bit / INT32_MIN | No | Stimuli `uniform(0, 1_000_000)` positive (`_int_unary_stimuli_spec`, `test_sfpu_unary.py:298`); `x<<3` ≤ ~8e6 stays in positive int32 by design (comment lines 295-297). No wrap/overflow driven. |
| left_shift | | negative input | No | Positive-only stimuli; arithmetic-vs-logical behavior on negatives untested (contrast binary edge test which drives negatives). |
| right_shift (RightShift, unary) | arithmetic `x >> 3` fixed (`calculate_right_shift`, sign-preserving) | negative input (arithmetic sign-extend) | No | Stimuli positive-only `uniform(0, 1e6)`; the negative-input sign-extension branch (`v_if(input<0)` in kernel) is never driven, even though golden `_right_shift` uses arithmetic `>>`. |
| right_shift | | shift ≥ 32 / negative shift / INT32_MIN | No | Shift fixed at 3; edge shifts and INT32_MIN only covered by binary shift test (`test_sfpu_binary.py:245-335`), not the unary op. |
| bitwise_not (BitwiseNot) | `~x` int32 (`calculate_bitwise_not`, `TTI_SFPNOT`) | any (INT32_MIN/MAX, sign-mag round-trip, negatives) | No | Perf-only (`perf_eltwise_unary_sfpu_int32.py:47`); no functional golden/assert. Not in any functional op list. |
| logical_not (LogicalNot / LogicalNotUnary) | `x==0 ? 1:0` (`calculate_logical_not`, SFPSETCC EQ0) | exact `x==0` (the "true" input) | Yes | `test_eltwise_unary_sfpu_threshold` (`test_sfpu_unary.py:587`) with threshold 0.0: `x[0::3]=0.0` forces zero hits so both branches fire. Golden `_logical_not` (`golden_generators.py:2502`). |
| logical_not | | NaN input (`nan!=0` → 0) / ±0.0 | No | Float32/Float16_b float (DEFAULT-layout) path only; NaN not injected and the LO16/INT32 kernel variants (`INSTRUCTION_MODE`) are not exercised. |
| typecast (Typecast) | dtype conversion over full ttnn matrix (`ckernel_sfpu_typecast.h`; golden `TypecastGolden`) | float→int overflow / saturation (clamp to dest range) | No | `test_eltwise_unary_typecast.py:171-179` uses whole-number floats `[0,201)` (`[0,16)` if bfp) and ints `[0,255]` (`[0,15]` if bfp). Golden clamps/wraps (`_to_integer`, `golden_generators.py:1793-1805`) but stimuli never reach the clamp boundaries, so saturation/overflow is unverified. |
| typecast | | INT32_MIN/MAX, UINT32_MAX, negative values | No | Stimuli are non-negative and bounded (≤255 / ≤200); sign-magnitude INT32_MIN round-trip and full-range values never driven. |
| typecast | | special values (±inf, NaN) through conversion | No | No inf/NaN in stimuli; float→int on non-finite and int→float exactness at large magnitudes untested. |
| typecast | | UInt8 low-byte wrap (`values % 256`) / two's-complement wrap | No | Golden models the wrap (`golden_generators.py:1796-1797`) but inputs ≤255 never exceed a byte, so the wrap path is not exercised. |
| typecast | | narrow block-float precision (Bfp8_b/Bfp4_b/Bfp2_b) | Partial | Full pair matrix (`TYPECAST_PAIRS`) is swept incl. bfp pairs, but only with tiny whole-number ranges (`[0,16)`) chosen so shared-exponent quant is near-exact; large/dense bfp inputs where quantization diverges are not stressed. |
| typecast | | full directed pair matrix coverage | Yes | `TYPECAST_PAIRS` (`test_eltwise_unary_typecast.py:84`) enumerates every directed pair over 8 formats minus same-dtype and `int32<->uint32`; dest_acc chosen per production rule (`_production_dest_acc`). Value-level correctness only, on benign inputs. |


## 4. Binary (float) SFPU & shift ops


**Cross-cutting finding (applies to every row below):** `test_sfpu_binary.py` / `test_sfpu_binary_bcast.py` do **not** import `sfpu_domains.py`; the `_OP_DOMAIN_REGISTRY` and `_SFPU_UNDEFINED_RANGES` entries (incl. `SfpuElwdiv` divisor hole `(-1e-6,1e-6)`, `SfpuXlogy` B `(-inf,1e-6)`, `SfpuElwpow` A `(-inf,1e-6)`) are consumed **only** by `test_sfpu_unary.py`. For the binary tests the stimuli come from `generate_stimuli(spec_A=None)` → `default_spec_for_format` = **uniform [0.1, 1.1]** for all float formats (positive, small, bounded away from 0), and **uniform [0, iinfo.max//2−1]** for Int32 / [0, 2³²−2] for UInt32. So most float edges are avoided *by the default positive domain*, not by the registry. `Excluded` below = deliberately kept out of range by that default domain (behaviour at the boundary is not verified); the registry entry, where present, is cited but is dormant for these tests.

| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| SfpuElwadd (ADD) | in0+in1, fp32 compute, RNE→bf16 on 16-bit dest | +inf/-inf/NaN operands, denormals, overflow to inf, -0.0 | No | `test_sfpu_binary_float`/`_bcast` use default [0.1,1.1]; no special values ever generated, none asserted |
| SfpuElwadd | " | happy path (float32/16/16b/bfp8, all bcast dims) | Yes | `test_sfpu_binary_float` sweeps 4 formats × {None,Row,Col,Scalar} bcast; also `test_sfpu_binary_bcast` (Row/Col) |
| SfpuElwadd | int32 add | int32 overflow / sign-magnitude Dst round-trip | No | `test_sfpu_binary_int` Int32 stimuli [0,2³⁰−1], non-negative, sum stays < 2³¹; overflow never hit |
| SfpuElwsub (SUB) | in0−in1 | +-inf/NaN/-0.0, catastrophic cancellation | No | default [0.1,1.1]; no IEEE specials |
| SfpuElwsub | " | happy path float + int32 + bcast | Yes | `test_sfpu_binary_float`, `test_sfpu_binary_int`, `test_sfpu_binary_bcast` |
| SfpuElwmul (MUL) | in0*in1; special 0*x=0 / x*0=0 for bf16 dest (`ckernel_sfpu_binary.h`) | operand exactly 0.0 (exercises the `v_if(in0==0||in1==0)` zero-clamp) | No | operands ∈ [0.1,1.1], never 0; the bf16 zero-clamp branch is never exercised |
| SfpuElwmul | " | +-inf/NaN, overflow to inf, denormals | No | default positive small domain |
| SfpuElwmul | " | happy path float + bcast (Col/Row use SFPMUL) | Yes | `test_sfpu_binary_float`, `test_sfpu_binary_bcast` |
| SfpuElwdiv (DIV) | in0 * recip(in1); `v_if(in1==0)`→±inf (0/0→NaN); `v_if(in0==in1)`→1.0 | divisor == 0 (±inf / NaN branch) | Excluded | Registry `SfpuElwdiv: B [(-1e-6,1e-6)]` (dormant) **and** default divisor ∈ [0.1,1.1] → the `in1==0` branch is never exercised. `test_sfpu_binary_div` |
| SfpuElwdiv | " | in0==in1 exact (result-forced-to-1.0 branch) | No | both operands independent uniform [0.1,1.1]; equality prob ≈ 0, branch untested |
| SfpuElwdiv | " | negative divisor / signed-zero inf, large-magnitude recip | No | default domain positive [0.1,1.1] only |
| SfpuElwrsub (RSUB) | in1 − in0 | +-inf/NaN/-0.0 | No | default [0.1,1.1]; happy path only (`test_sfpu_binary_float`) |
| SfpuElwpow (POW) | base**pow via 2^(pow·log2 base); NaN for base=0&pow<0 and neg-base&non-int pow (`ckernel_sfpu_binary_pow.h`) | base ≤ 0 (log of non-positive) | Excluded | Registry `SfpuElwpow: A [(-inf,1e-6)]` (dormant) + default base ∈ [0.1,1.1]; base≤0 never generated |
| SfpuElwpow | " | 0^0, negative base with fractional exponent (→NaN), base=0 & pow<0 (→NaN) | No | pow & base both ∈ [0.1,1.1] (positive); none of the NaN/sign special-case branches reached |
| SfpuElwpow | " | negative base, integer exponent (signed result) | No | base always positive |
| SfpuElwpow | " | happy path (positive base/exp), fp32 vs bf16 paths | Yes | `test_sfpu_binary_float` (float formats only; Bfp8_b explicitly skipped, line 106-110) |
| SfpuXlogy (XLOGY) | x·log(y); NaN for y<0/NaN, x·(−inf) for y=0 | y ≤ 0 (log of non-positive) | Excluded | Registry `SfpuXlogy: B [(-inf,1e-6)]` (dormant) + default y ∈ [0.1,1.1]; y≤0 never generated |
| SfpuXlogy | " | xlogy(0, 0) and xlogy(x, 0) (→ x·−inf), y=NaN | No | x,y ∈ [0.1,1.1]; zero/NaN operands never generated (golden `_xlogy` comment says non-finite edges "not consistently modelled") |
| SfpuXlogy | " | happy path (x≥0, y>0) | Yes | `test_sfpu_binary_float`; Bfp8_b skipped |
| SfpuAddTopRow | adds top row of two tiles; UInt32 masks to 32 bits (wraparound) | UInt32 wraparound / overflow | Partial | `test_sfpu_binary_add_top_row` runs UInt32 with default [0,2³²−2] so large sums can wrap; golden `_add_top_row` masks to 32 bits — wraparound path plausibly hit but not deterministically targeted |
| SfpuAddTopRow | " | Float32 / Int32 / UInt32 happy path | Yes | `test_sfpu_binary_add_top_row` (Int32/UInt32 require dest_acc=Yes) |
| SfpuAddTopRow | " | +-inf/NaN float operands | No | default float [0.1,1.1] |
| SfpuBinaryMax (MAX) | SFPSWAP min/max (`ckernel_sfpu_binary_max_min.h`); golden `torch.maximum` | NaN operand (SFPSWAP NaN propagation) | No | `test_sfpu_binary_float_extended` uses default [0.1,1.1]; no NaN generated |
| SfpuBinaryMax | " | equal operands (tie), +inf/-inf | No | operands independent uniform → equal ≈ never; no inf |
| SfpuBinaryMax | " | happy path (float32/16/16b/bfp8) | Yes | `test_sfpu_binary_float_extended` |
| SfpuBinaryMin (MIN) | SFPSWAP min; golden `torch.minimum` | NaN operand, equal operands, +-inf | No | same as MAX — default [0.1,1.1], edges never generated |
| SfpuBinaryMin | " | happy path | Yes | `test_sfpu_binary_float_extended` |
| SfpuBinaryFmod (FMOD) | a − trunc(a/b)·b via recip; `v_if(b==0)`→NaN, `v_if(a==b)`→0 (`ckernel_sfpu_binary_fmod.h`) | divisor b == 0 (NaN branch) | No | default b ∈ [0.1,1.1]; b==0 branch never exercised (Bfp8_b skipped, line 369) |
| SfpuBinaryFmod | " | a==b exact (→0 branch), sign of a negative, mixed signs | No | operands ∈ [0.1,1.1] positive; sign-correction branches (a<0 etc.) never reached |
| SfpuBinaryFmod | " | happy path (positive a,b) | Yes | `test_sfpu_binary_float_extended` (float32/16/16b) |
| SfpuBinaryRemainder (REMAINDER) | a − floor(a/b)·b; `v_if(b==0)`→NaN; sign follows b | divisor b == 0 (NaN branch) | No | default b ∈ [0.1,1.1]; never 0 |
| SfpuBinaryRemainder | " | negative b / mixed-sign sign-correction, magnitude-correction branches | No | positive [0.1,1.1] operands only; sign paths untested |
| SfpuBinaryRemainder | " | happy path | Yes | `test_sfpu_binary_float_extended` |
| SfpuAtan2 (ATAN2) | atan2(y=in0, x=in1) minimax; explicit (0,0)→±0, |x|=|y|→π/4, x<0→π−r, NaN→NaN (`ckernel_sfpu_atan2.h`) | signs / all four quadrants, |y|≥|x| and x<0 branches | Yes | `test_sfpu_binary_atan2` uniform [−5,5] both operands → mixed signs cover quadrants + branches (matched under PCC) |
| SfpuAtan2 | " | (0,0) exact (±0 special case) | No | continuous uniform [−5,5], (0,0) prob ≈ 0; the `min==0` branch not deterministically hit |
| SfpuAtan2 | " | NaN / ±inf operands (kernel has explicit NaN & inf==inf→π/4 handling) | No | no special values in [−5,5] stimuli |
| SfpuLogsigmoid (LOGSIGMOID) | −softplus(−x); poly for |−x|<4, −exp(−x) for −x<−4 i.e. x>4 (`ckernel_sfpu_logsigmoid.h`) | mid-range poly branch (−4<x<4) and large-negative passthrough (x≤−4) | Yes | `test_sfpu_binary_logsigmoid` x=linspace[−8, 3.9]; covers poly + passthrough (PCC) |
| SfpuLogsigmoid | " | x > 4 branch (uses device exp(−x) as in1) | No | stimuli capped at 3.9 by design — comment: "x>4 branch needs a device-computed exp(−x) operand the shared harness can't provide, left to a future driver" |
| SfpuLogsigmoid | " | +-inf/NaN | No | bounded linspace stimuli |
| SfpuIsclose (ISCLOSE) | \|a−b\| ≤ atol+rtol·\|b\| → 1/0, fixed rtol=1e-5 atol=1e-8, equal_nan=False | true (equal) and false (gap 2.0) both produced | Yes | `test_sfpu_binary_isclose` crafted paired stimuli: even p identical→1, odd p differ by 2.0→0 |
| SfpuIsclose | " | tolerance boundary (\|a−b\| ≈ atol+rtol·\|b\|) | No | comment: "2.0 gap dwarfs the tolerance so the decision is unambiguous" — boundary never probed |
| SfpuIsclose | " | NaN operands (equal_nan=False semantics), ±inf | No | crafted stimuli are small positive integers only |
| SfpuElwEq (EQ, float) | fp32 eq via `calculate_binary_comp_fp32`; explicit inf==inf bit-equal handling | equal branch (a==b→1) | Yes | `test_sfpu_binary_eq_ne` crafted paired stimuli ~50/50 equal/differ-by-1.0 |
| SfpuElwEq | " | inf==inf special case, NaN==NaN (→0) | No | no inf/NaN in stimuli; the `in0_bits==in1_bits && exp==0x7F800000` inf branch untested |
| SfpuElwNe (NE, float) | logical negation of eq | not-equal + equal branches | Yes | `test_sfpu_binary_eq_ne` |
| SfpuElwNe | " | NaN operands (NaN≠NaN→1) | No | no NaN in stimuli |
| SfpuElwLt / Gt / Le / Ge (float) | float relational → 1.0/0.0 | any float coverage | No | Commented out in `test_sfpu_binary_float` (lines 87-90: "failing due to very small differences in generated stimuli"); no other float driver — **float Lt/Gt/Le/Ge untested** |
| SfpuElwLt / Gt / Le / Ge (int32) | int32 relational via `calculate_binary_comp_int32` (sign-check, opposite-sign safe) | happy path over Int32 | Yes | `test_sfpu_binary_int` (Int32, default [0,2³⁰−1]) |
| SfpuElwLt / Gt / Le / Ge (int32) | " | equal operands (tie), negative operands, opposite-sign overflow path | No | default stimuli non-negative & independent → ties ≈ never, negatives never (kernel's opposite-sign branch designed for it but not driven) |
| SfpuElwLeftShift (LSHFT) | int32 `a << b`; amounts outside [0,31]→0 (kernel contract); INT32_2S_COMP mode | in-range amounts 0,1,2,7,15,16,30,31 | Yes | `test_sfpu_binary_int_shift_edge_cases` `_SHIFT_EDGE_AMOUNTS` (Wormhole only) |
| SfpuElwLeftShift | " | amount == 32 (first out-of-range → 0) | Yes | `_SHIFT_EDGE_AMOUNTS` includes 32; `_shift_reference`/golden `_left_shift` return 0 |
| SfpuElwLeftShift | " | amount > 32 (33,40,63,100,1000 → 0) | Yes | `_SHIFT_EDGE_AMOUNTS` |
| SfpuElwLeftShift | " | negative amount (−1,−5,−32,−1000 → 0) | Yes | `_SHIFT_EDGE_AMOUNTS` |
| SfpuElwLeftShift | " | negative value operand (0x40000000, −256, −0x55555555, etc.) | Yes | `_SHIFT_EDGE_VALUES` includes negatives & sign bit; golden `torch.bitwise_left_shift` |
| SfpuElwLeftShift | " | value == INT32_MAX (0x7FFFFFFF) | Yes | in `_SHIFT_EDGE_VALUES` |
| SfpuElwLeftShift | " | value/result == INT32_MIN (0x80000000) | Excluded | `_build_shift_edge_case_src` filters `v != INT32_MIN and result != INT32_MIN`; separately documented as HW-unsupported by xfail `test_sfpu_binary_int_shift_int32_min_unsupported` (sign-magnitude Dst can't round-trip −2³¹) |
| SfpuElwLeftShift | " | Blackhole coverage | No | `test_sfpu_binary_int_shift_edge_cases` skips BLACKHOLE (unmigrated TTI microcode diverges under INT32_2S_COMP; see SFPU_INT32_SHIFT.md) |
| SfpuElwRightShift (RSHFT) | int32 arithmetic `a >> b` (sign-extends); outside [0,31]→0 | in-range 0..31, ==32, >32, negative amount | Yes | `test_sfpu_binary_int_shift_edge_cases`; golden `_right_shift` guards <0/≥32→0 to match kernel (arithmetic shift would else sign-extend to −1) |
| SfpuElwRightShift | " | negative value → arithmetic sign extension | Yes | `_SHIFT_EDGE_VALUES` negatives; `torch.bitwise_right_shift` (arithmetic) |
| SfpuElwRightShift | " | INT32_MIN operand/result | Excluded | filtered in `_build_shift_edge_case_src`; xfail test documents unsupported |
| SfpuElwRightShift | " | Blackhole | No | edge test skips Blackhole |
| SfpuElwLogicalRightShift (LOGICAL_RSHFT) | unsigned `a >> b`; outside [0,31]→0 | in-range 0..31, ==32, >32, negative amount | Yes | `test_sfpu_binary_int_shift_edge_cases`; golden `_logical_right_shift` treats value as uint32 |
| SfpuElwLogicalRightShift | " | negative value → zero-fill (not sign-extend) | Yes | `_SHIFT_EDGE_VALUES` negatives; `_shift_reference` masks `&0xFFFFFFFF` then `>>` |
| SfpuElwLogicalRightShift | " | INT32_MIN operand/result | Excluded | filtered in `_build_shift_edge_case_src`; xfail test documents unsupported |
| SfpuElwLogicalRightShift | " | Blackhole | No | edge test skips Blackhole |
| Shift ops (all three), non-edge `test_sfpu_binary_int` | int32 shift over default stimuli | random large shift amounts | Partial | In `test_sfpu_binary_int` operand B (shift amount) is default [0,2³⁰−1] → nearly always ≥32 → degenerate (both kernel & golden = 0); real coverage is the dedicated edge test above |
| SFPU binary-bcast (ADD/SUB/MUL only) | col/row broadcast of one operand then eltwise (`ckernel_sfpu_binary_bcast.h`, ADD/SUB/MUL only) | Row & Col broadcast, Float32 & Float16_b | Yes | `test_sfpu_binary_bcast` (Blackhole/Wormhole; skip_for_quasar), default [0.1,1.1] |
| SFPU binary-bcast | " | +-inf/NaN, zero operand, negative operand | No | default positive [0.1,1.1] stimuli; DIV/POW/etc. not supported by bcast kernel (static_assert ADD/SUB/MUL) |


## 5. Binary — integer & bitwise ops

| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| SfpuElwmulInt (MUL) — `_mul_int_` (ckernel_sfpu_mul_int.h) | 16-bit integer multiply: splits u16→u8, casts to fp32, LO16 result | Exists in WH/BH binary suite at all | No | Not parametrized in `test_sfpu_binary.py` (grep: 0 hits). Golden maps `SfpuElwmulInt→self._mul` (golden_generators.py:3392) but no WH/BH test row. |
| SfpuElwmulInt (MUL) | u16 multiply | Any coverage (Quasar only) | Partial | Only `test_eltwise_binary_sfpu_int_quasar` (quasar/…:194), Int32, operands clamped to ±1000 via `_INT_OPS`. Quasar-only. |
| SfpuElwmulInt (MUL) | LO16 16-bit result | 16-bit overflow / wraparound (product ≥ 2^16) | No | Result stored as LO16; product > 65535 wraps. No WH/BH test; Quasar clamp ±1000 gives products > 2^16 but no dedicated wrap assertion. |
| SfpuElwmulInt (MUL) | LO16 (unsigned) operands | Negative operands | No | Kernel loads LO16 (unsigned 16-bit); negative-operand behavior untested. |
| SfpuMulInt32 (MUL_INT32) — `mul_int32` (ckernel_sfpu_mul_int32.h) | int32 multiply, low 32 bits via 11-bit fp32 chunks | Overflow / wraparound (product ≥ 2^31) | No | `_INT_BINARY_STIMULI[SfpuMulInt32]=(1,40000)` (test:418) keeps product < 2^31. Golden truncates to int32 (golden:3698) but wrap never exercised. |
| SfpuMulInt32 | int32 multiply | Negative operands / negative product | No | Kernel stores 2's-comp via plain INT32; harness sign-magnitude packer can't round-trip negatives, so test is positive-only by design (golden comment 3698-3703). |
| SfpuMulInt32 | int32 multiply | INT32_MAX operands, sign bit | No | Operands ≤ 40000; extremes never generated. |
| SfpuGtInt/LtInt/LeInt/GeInt (GT/LT/LE/GE_INT) — `calculate_binary_comp_int32` (ckernel_sfpu_binary_comp.h) | int32 relational via subtract + sign, with opposite-sign overflow-avoidance branch | Any coverage in WH/BH suite | No | Not parametrized in `test_sfpu_binary.py` (grep: 0 hits each). Goldens exist (`_gt_int`… golden:3369-3379). |
| SfpuGtInt/LtInt/LeInt/GeInt | int32 relational | Opposite-sign operands (overflow-avoidance branch) | Partial | Covered only on Quasar (`test_eltwise_binary_sfpu_int_quasar`), stimuli uniform full signed range `[iinfo.min, iinfo.max-1]` incl. mixed signs → exercises the `LREG_3==0` sign-XOR branch. Quasar-only. |
| SfpuGtInt/LtInt/LeInt/GeInt | int32 relational | INT32_MIN operand | Excluded | `_get_integer_bounds` uses `min+1` (generator utils.py:48-58); INT32_MIN never generated. |
| SfpuGtInt/LtInt/LeInt/GeInt | int32 relational | Equal operands (LE/GE equality path) | Partial | Quasar full-range random rarely hits exact equality; no crafted equal-pair stimuli for these ops. |
| SfpuEqInt/NeInt (EQ/NE_INT) — int32 dest bits | Exact integer equality → 0/1 | Equal branch actually hit | Yes | `test_sfpu_binary_eq_ne_int` uses `_eq_ne_stimuli_spec()` (~50% equal, values 1..9). |
| SfpuEqInt/NeInt | integer equality | Negative operands / INT32_MIN | No | Stimuli are positive small ints (1..9); sign bit / extremes never tested. |
| SfpuBitwiseAnd — `calculate_bitwise_and` | int32 AND | Negative operands / sign bit set | No | `test_sfpu_binary_bitwise`, default Int32 stimuli `uniform(0, INT32_MAX//2−1)` (generator.py:255) → operands non-negative, bit31 never set. |
| SfpuBitwiseOr — `calculate_bitwise_or` | int32 OR with sign-magnitude fixup (`v_if(res<0) setsgn`) | Sign-magnitude negative-result branch | No | Kernel has special `v_if(res>INT_MIN && res<0)` setsgn path; default non-negative stimuli make result always ≥0, so this branch is never exercised. |
| SfpuBitwiseXor — `calculate_bitwise_xor` | int32 XOR with same sign-magnitude fixup | Sign-magnitude negative-result branch | No | Same as OR: non-negative stimuli, setsgn branch untested. |
| SfpuBitwiseAnd/Or/Xor | int32 bitwise | INT32_MIN, full 32-bit patterns | No | INT32_MIN excluded; operands capped at ~2^30 (bit31=0). |
| SfpuDivInt32 (DIV_INT32) — `calculate_div_int32_trunc` | int32 division, round toward zero | Division by zero | No | `_INT_BINARY_STIMULI[SfpuDivInt32]=(1,8e6)` (test:411) — divisor ≥ 1, zero never generated. |
| SfpuDivInt32 vs SfpuDivInt32Floor | trunc vs floor rounding | Negative operands (the whole trunc/floor distinction) | No | Both use positive stimuli (1..8e6); for non-negatives trunc==floor, so the sign branch (`v_if(sign<0){ result=-result; if(floor) …-=1 }`) and the two ops' divergence are entirely untested. |
| SfpuDivInt32 / Floor | int32 division | INT32_MIN / −1 overflow | No | Extremes excluded; overflow case never hit. |
| SfpuDivInt32 / Floor | reciprocal-based division | Large operands (int→fp32 precision) | Partial | Kept < 2^24 (8e6) for exact int→fp32; behavior above 2^24 not tested. |
| SfpuGcd (GCD) — `calculate_sfpu_gcd` (binary-GCD, 31 iters) | gcd of int32 (abs both) | gcd(0,0) and gcd(x,0) | No | Stimuli strictly positive `(1,100000)` (test:414); init disables `a==0` lanes but zero operands never generated. |
| SfpuGcd | binary GCD | Negative operands | No | Kernel `SFPABS`es inputs, but stimuli positive-only; negative path untested. |
| SfpuGcd | binary GCD | INT32_MIN, large 31-bit inputs | No | Operands ≤ 1e5 (≪ 2^31); `max_input_bits=31` path near-boundary untested. |
| SfpuLcm (LCM) — `calculate_sfpu_lcm` (GCD@15 bits + u16×u16→u32) | lcm = a/gcd·b, assumes \|a\|,\|b\|<2^15 | lcm with 0 | No | Positive stimuli `(1,20000)` (test:416); zero never generated. |
| SfpuLcm | lcm | Negative operands | No | Kernel abs()es; stimuli positive-only. |
| SfpuLcm | lcm | Operands ≥ 2^15 (breaks kernel assumption) | No | 20000 < 32768; the `<2^15` precondition boundary and beyond untested. |
| SfpuLcm | u16×u16→u32 product | lcm overflow (result near/above 2^31–2^32) | No | Max operands 20000 → lcm < 2^30, well below u32 wrap; overflow never reached. |
| SfpuRsubInt32 (RSUB_INT32) — `calculate_rsub_int` | reverse subtract out = in1 − in0 (2's-comp add) | Negative result round-trip | Yes | `test_sfpu_binary_rsub_int32` with `twos_complement=True` (test:453). This is exactly the BH/WH sign-magnitude divergence documented in RSUB_INT32_BLACKHOLE_CI_FIX.md (BH mapped INT32_2S_COMP→SM32). |
| SfpuRsubInt32 | reverse subtract | INT32_MIN operand or result | Excluded | `_get_integer_bounds` drops INT_MIN (utils.py:48-58); sign-magnitude Dst treats 0x80000000 as "−0" and cannot round-trip (per RSUB doc). |
| SfpuRsubInt32 | reverse subtract | Negative operands (not just negative result) | No | Default stimuli non-negative (0..2^30); only the *result* goes negative. |
| SfpuRsubInt32 | reverse subtract | Overflow of t2−t1 beyond int32 | No | Golden widens to int64; operands ≤ 2^30 so difference always fits — wrap never tested. |
| SfpuMask (MASK) — `calculate_mask` (fp16-zero check) | zero data where mask==0, else passthrough | Float mask carries real zeros | Yes | `test_sfpu_binary_mask` (Float16_b/Float32), `_mask_stimuli_spec` zeroes ~1/3 of mask (test:477-488). |
| SfpuMask | integer mask path (`calculate_int_mask`) | Int32 mask | No | Test parametrizes only Float16_b/Float32 (test:492); the `mask==0` int variant is never driven. |
| SfpuMask | fp16-zero detection | mask = −0.0 / denormal / NaN | No | Stimuli are exact 0.0 / 1.0; `_sfpu_is_fp16_zero_` on −0.0/denorm/NaN untested. |
| SfpuMaxInt32/MinInt32 — `calculate_binary_max_min_int32<IS_UNSIGNED=false>` | signed int32 max/min via SFPSWAP + sign-fix | Negative operands (signed sign-fix branch) | No | Stimuli `(0,1e6)` non-negative (test:420-421); the `SFPSETCC LT0` correction branch for negatives never exercised. |
| SfpuMaxInt32/MinInt32 | signed max/min | INT32_MIN / INT32_MAX | No | Extremes never generated. |
| SfpuMaxInt32/MinInt32 | max/min | Equal operands | Partial | Random 0..1e6, exact ties essentially never; no crafted equal stimuli. |
| SfpuMaxUint32/MinUint32 — `calculate_binary_max_min_int32<IS_UNSIGNED=true>` | unsigned max/min (GTE0 swap variant) | Values with bit31 set (signed vs unsigned diverge) | No | Stimuli `(0,1e6)` under UInt32 (test:422-423, `_UINT32_BINARY_OPS`); all < 2^31 so signed==unsigned — the `IS_UNSIGNED` GTE0 branch is never distinguished from signed. |
| SfpuMaxUint32/MinUint32 | unsigned max/min | UINT32_MAX | No | Operands ≤ 1e6; max unsigned value never tested. |
| SfpuRemainderInt32 / RemainderUint32 (REMAINDER_INT32/UINT32) — `calculate_remainder_int32` | a − floor(a/b)·b (sign of b) | Modulo by zero | No | Divisor ≥ 1 (`(1,10000)`, test:426/428); zero never generated. |
| SfpuRemainderInt32/Uint32 | floor-remainder sign handling | Negative operands (floor convention, sign-of-b branch) | No | Non-negative stimuli; golden comment (golden:3688) states result is "convention-agnostic" — the `sign<0`/`a_signed<0` remainder-adjust branches untested. |
| SfpuRemainderInt32/Uint32 | reciprocal-based remainder | INT32_MIN, large operands > 2^24 | No / Partial | INT32_MIN excluded; operands kept < 10000 (≪ 2^24) for exact reciprocal — larger untested. |
| SfpuFmodInt32 (FMOD_INT32) — `calculate_fmod_int32` | a − trunc(a/b)·b (sign of a) | Modulo by zero | No | Divisor ≥ 1 (`(1,10000)`, test:427); zero never generated. |
| SfpuFmodInt32 | trunc-remainder sign handling | Negative dividend (sign-follows-a `r=−r` branch) | No | Non-negative stimuli; golden `int(t1)%int(t2)` (golden:3693) — the `v_if(a_signed<0) r=−r` branch untested. |
| SfpuFmodInt32 vs RemainderInt32 | trunc vs floor mod | Distinction between fmod and remainder | No | Only visible for mixed-sign operands, which are excluded; both goldens reduce to `a % b`, so the two ops are indistinguishable under current stimuli. |


## 6. Ternary, scalar-binop, reduce, top-k & other/untested kernels

| Kernel / Op | What it does | Edge case | Tested? | Evidence / gap notes |
|---|---|---|---|---|
| SfpuWhere / TTNNWhere (`ckernel_sfpu_where.h`) | `out = (cond==0) ? false : true`, per-element select via SFPSETCC EQ0 | predicate == 0 (select false branch) | Yes | `test_ttnn_where.py::test_ttnn_where` `test_case="all_zeros"` forces `cond=zeros`; `test_ttnn_where_mcw` uses alternating 0/1 pattern |
| SfpuWhere / TTNNWhere | " | predicate != 0 (select true branch) | Yes | `test_case="all_ones"` / `"mixed"`, and mcw alternating pattern |
| SfpuWhere / TTNNWhere | " | NaN/±inf in true/false branch values | No | Branch stimuli are `StimuliSpec.uniform(0,1)` (never special); golden's `torch_equal_nan` helper is defensive only and never driven — no NaN/inf ever generated |
| SfpuWhere / TTNNWhere | " | NaN / -0.0 as predicate (NaN≠0 → true; -0.0 EQ0 behavior) | No | cond stimuli uniform(0,1) or int pattern 0/1; -0.0 and NaN predicate never injected |
| SfpuWhere / TTNNWhere | " | Int32 / large-magnitude values | Partial | Int32 format swept (`input_output_formats([...Int32])`), but values still uniform(0,1); no INT32_MIN/MAX predicate |
| SfpuWhere / TTNNWhere | " | format coverage | Partial | Only Float16_b, Float32, Int32 with same in/out; UInt32 (allowed by kernel static_assert) not swept |
| SfpuAddcmul (`ckernel_sfpu_addcmul.h`) | `out = a + value*b*c` | generic values | Yes | `test_sfpu_ternary.py`, a,b,c uniform(-1,1), value=2.0 fixed; formats Float16_b/Float32/Bfp8_b, dest_acc No/Yes |
| SfpuAddcmul | " | NaN/±inf operands, format extremes/overflow | No | Inputs bounded uniform(-1,1); no special-value or saturation stimuli. Golden is dest_acc-agnostic (documented limitation, TernarySFPUGolden docstring) |
| SfpuAddcmul | " | scalar `value` edges (0, huge, negative) | No | `_SCALAR_VALUE=2.0` hard-coded; only one scalar ever tested |
| SfpuAddcdiv (`ckernel_sfpu_addcdiv.h`) | `out = a + value*b/c` | divide-by-zero (c=0) | No | Divisor spec is `uniform(1.0, 2.0)` (`_divide_by_c` branch in `_run_sfpu_ternary`) — c is deliberately kept in [1,2], so c=0 and near-zero c are never generated; behavior at c→0 unverified |
| SfpuAddcdiv | " | tiny \|c\| (near-pole amplification), negative c | No | c clamped to [1,2]; neither small-magnitude nor negative divisor exercised |
| SfpuAddcdiv | " | generic values | Yes | `test_sfpu_ternary.py` (Float16_b/Float32 only; Bfp8_b skipped for non-addcmul) |
| SfpuLerp (`ckernel_sfpu_lerp.h`) | `out = a + c*(b-a)` (c = weight) | weight = 0 and = 1 (exact endpoints) | No | weight `c` = `uniform(-1,1)`; exact 0.0/1.0 essentially never hit by continuous RNG, not asserted |
| SfpuLerp | " | weight outside [0,1] | Partial | c=uniform(-1,1) covers negative weights (<0) but never >1; upper-extrapolation side untested |
| SfpuLerp | " | NaN/inf operands | No | bounded uniform inputs only |
| SfpuSnakeBeta (`SfpuSnakeBeta`, ternary) | `out = a + sin(b*a)^2 / c` (a=x,b=alpha,c=beta) | beta (c) = 0 → divide-by-zero | No | c=`uniform(1.0,2.0)` (`_divide_by_c` includes SnakeBeta); beta never 0/near-0 |
| SfpuSnakeBeta | " | large \|b*a\| (sin arg reduction) | No | a,b∈(-1,1) so b*a∈(-1,1); large-argument sin reduction never exercised |
| SfpuSnakeBeta | " | generic values | Yes | `test_sfpu_ternary.py` (Float16_b/Float32; Bfp8_b skipped) |
| ScalarAdd (`binop_with_unary.h`) | `out = x + s` | generic | Yes | `test_sfpu_binop_scalar.py`, x=uniform(-1,1), s=2.0; Float16_b/Float32 |
| ScalarAdd | " | NaN/inf x, format max/overflow, -0.0 | No | bounded uniform(-1,1) inputs; single fixed scalar |
| ScalarSub (`binop_with_unary.h`) | `out = x - s` | generic / edges | Partial | Yes for generic; NaN/inf/overflow = No (bounded inputs, s=2.0 fixed) |
| ScalarMul (`binop_with_unary.h`) | `out = x * s` | generic / overflow-saturation | Partial | Yes generic; s=2.0 only, no large-magnitude x → no overflow/saturation coverage |
| ScalarDiv (`binop_with_unary.h`) | `out = x * (1/d)` — host inverts divisor, kernel multiplies | divide-by-zero (d=0) | No / N/A | `_DIVISOR=2.0` fixed; host computes `1/d` at compile time (`_SCALAR_BITS[ScalarDiv]=bits(1/2)`), kernel only multiplies, so on-device div-by-zero path does not exist and d=0 is never tested |
| ScalarDiv | " | reciprocal precision at small/large d | No | only d=2.0 (1/d=0.5) ever used |
| ScalarRsub (`binop_with_unary.h`) | `out = s - x` | generic / edges | Partial | Yes generic (s=2.0); NaN/inf/overflow untested |
| All ScalarBinop | " | format coverage / dest_acc | Partial | Only Float16_b & Float32 (same in/out); Float16_b+dest_acc=Yes and Float32+dest_acc=No skipped; no bf16-a/fp16/int/bfp formats |
| ReduceColumn (`ckernel_sfpu_reduce.h`) | column pool (Max/Min/Sum/Avg) over 32-row axis | single real row (extent=1), all-equal, sub-tile padding | Yes | `test_sfpu_reduce.py` `get_reduce_extents` sweeps extents {1,13,15,16,17,30,31,32} for Int32 (pad path); extent=1 ≈ single-element reduce |
| ReduceColumn | " | INT32_MIN / INT32_MAX operands | Yes | `test_int32_reduce_extreme` injects INT32_MIN & INT32_MAX (Min/Max), base ranges mixed/all-neg/all-pos; **Wormhole only** (`@skip_for_blackhole`, BH sign-magnitude path still buggy — tt-metal#44750) |
| ReduceColumn | " | NaN / ±inf float operands, float max-reduce overflow | No | float stimuli `uniform(min,max)` in ±1000; pad uses finite ±3e30 sentinel, never real inf/NaN; no format-max overflow injected |
| ReduceColumn | " | UInt16 Sum/Avg overflow | Yes (handled) | `get_reduce_formats` widens UInt16 Sum/Avg output to UInt32 to avoid overflow; bounds clipped to (0,1000) |
| ReduceRow (`ckernel_sfpu_reduce.h`) | row pool over column axis | Sum/Max/Min all formats; Avg float-only | Partial | `get_supported_reduce_axioms`: row Avg only Float32/Float16_b; UInt16 row Max/Min explicitly skipped (kernel garbage — documented) |
| ReduceRow | " | INT32_MIN/MAX | Yes | `test_int32_reduce_extreme` parametrizes ReduceRow too (WH only) |
| ReduceRow | " | NaN/inf, overflow | No | same bounded-stimuli gap as column |
| ReduceScalar (`ReduceDimension.Scalar`) | reduce all elements → single value | any edge | No (op path) | Golden supports scalar dim (`_reduce_scalar`) but `test_sfpu_reduce.py` only parametrizes ReduceColumn/ReduceRow; scalar reduce dim not exercised on device by this suite |
| Reduce (multi-dim) (`test_sfpu_reduce_multidim.py`) | chained column+row reduce, Int32 regression | Int32 sign/direction regression | Yes | Int32 is the stated regression target; formats Int32/Float32/UInt32/Float16_b, pools Max/Sum/Min(+Avg float) |
| Reduce (multi-dim) | " | NaN/inf, format extremes | No | bounds uniform/randint in ±range; no special values |
| Reduce (SDPA) (`test_sfpu_reduce_sdpa.py`) | 4×2 subblock column MAX (SDPA) | non-generic edges | No | Only Float16_b, only Max, single shape [512,64], plain uniform stimuli; no NaN/inf/extreme |
| ReduceBlockMax / reduce_custom (`ckernel_sfpu_reduce_custom.h`) | row-wise block max reduce | generic | Yes | `test_reduce_block_max.py` via `ReduceBlockMaxRowGolden`; validates column[0] |
| ReduceBlockMax / reduce_custom | " | NaN/inf, ties, extremes | No | not injected |
| TopK (whole op, `ckernel_sfpu_topk.h`) | per-row bitonic top-K values + indices | duplicate / tied values | Partial | `test_topk.py` stimuli uniform(0,1) (continuous — exact ties rare); `validate_topk_indices` tolerates tie index reordering but does not *force* ties; `stable_sort=True` path is **skipped** (LLK API broken, tt-metal#33492) |
| TopK | " | K ≠ 32 (k edges: 16, 64, k>width) | No | `K=[32]` only (explicit TODO in params to add 16/64); k boundary behavior untested |
| TopK | " | wide input 32×1024 | No (skipped) | `pytest.skip` for [32,1024] due to HW/golden discrepancy (issue #1344) |
| TopK | " | ascending vs descending | Yes | `sort_direction=[Descending, Ascending]` |
| TopK | " | NaN/inf values, negative values | No | stimuli uniform(0,1) only — no negatives, no special values |
| TopK | " | format coverage | Partial | Float16_b only |
| TopKLocalSort / TopKMerge / TopKRebuild | individual topk pipeline stages | any edge | No (untested) | Appear only in `perf_eltwise_unary_sfpu.py` (perf) and `sfpu_domains.py` (lines 522-530, perf domains); **no correctness test** drives the individual stages |
| SfpuSwiGLU (`ckernel_sfpu_swiglu.h`, Quasar) | swiglu(gate,up) binary SFPU | any edge | Partial (arch-limited) | Only `quasar/test_sfpu_swiglu_quasar.py` (1 C++ source, Quasar-only); no Wormhole/Blackhole correctness test, no special-value edges |
| ema (`ckernel_sfpu_ema.h`) | per-column EMA recurrence `alpha*prev+beta*x` | generic | Yes | `test_sfpu_ema.py`, Float16_b only, alpha=0.25 fixed, num_time_tiles {1,2,4} |
| ema | " | dest_acc=Yes (32-bit dest) | No (unsupported) | Kernel hardcoded 16-bit bf16 dest; dest_acc=Yes intentionally not swept (documented in test header) |
| ema | " | alpha edges (0, 1), NaN/inf x, long-sequence drift | No | single alpha=0.25; inputs uniform(-4,4); no boundary alpha or special values |
| welfords (`ckernel_sfpu_welfords.h`) | running mean/variance (Welford) | all edges | No (untested) | No MathOperation enum entry, no C++ test source, no Python test — kernel untested by infra |
| dropout (`ckernel_sfpu_dropout.h`) | random dropout mask/scale | all edges | No (untested) | No enum entry, no test source; untested |
| quant (`ckernel_sfpu_quant.h`) | quantize/dequantize | all edges | No (untested) | Header present; no dedicated SFPU-quant correctness test (grep "quant" hits are unrelated MX/quantize helpers) — untested |
| cumsum (`ckernel_sfpu_cumsum.h`) | cumulative sum | all edges | No (untested) | No enum entry, no test source; untested |
| reshuffle_rows (`ckernel_sfpu_reshuffle_rows.h`) | row permutation | all edges | No (untested) | No enum entry, no test source; untested |
| int_sum (`ckernel_sfpu_int_sum.h`) | integer sum reduction | all edges | No (untested) | No enum entry, no test source; untested |
| tiled_prod (`ckernel_sfpu_tiled_prod.h`) | tiled product | all edges | No (untested) | No enum entry, no test source; untested |
| copy_dest_values (`ckernel_sfpu_copy_dest_values.h`) | copy within Dest | all edges | No (untested) | No enum entry, no test source; untested |
| generalized_moe_gate_topk (`experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`) | MoE gate top-k | all edges | No (untested) | Experimental header (WH+BH); no enum entry, no test source; untested |
| max_pool_indices (`ckernel_sfpu_max_pool_indices.h`) | max-pool with argmax indices | all edges | No (untested) | No enum entry, no test source; untested |
| rand (`ckernel_sfpu_rand.h`) | RNG fill | all edges | No (untested) | No enum entry, no test source ("rand" grep hits are torch.rand/seeding); untested |
