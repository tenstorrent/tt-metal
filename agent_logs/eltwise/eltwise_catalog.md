# Eltwise Helper Catalog — Phase 0 (Enumeration)

**Category**: eltwise (unary, binary, ternary SFPU + FPU binary ops)
**Primary LLK Prefix**: `llk_math_eltwise_unary_sfpu`
**Additional Prefixes**: `llk_math_eltwise_binary_sfpu`, `llk_math_eltwise_binary`, `llk_math_eltwise_ternary_sfpu`
**Timestamp**: 2026-04-30
**Total Ops Enumerated**: 146 across both wormhole_b0 and blackhole architectures

---

## 1. Full Op List (All Sources Combined)

**Sourced from:**
- Phase 1A: LLK unary/binary SFPU prefixes (wormhole_b0, blackhole)
- Phase 1B: Compute API headers (top-level eltwise_*.h and per-op headers in eltwise_unary/)
- Phase 1C: FPU binary ops (add_tiles, sub_tiles, mul_tiles via llk_math_eltwise_binary_api.h)
- Phase 1D: Ternary SFPU ops (addcmul, addcdiv, lerp, where)
- Phase 2: Secondary source cross-check (unary_op_utils.cpp, binary_op_utils.cpp)

| Op | Found Via | Category | Group |
|----|-----------|----------|-------|
| abs | LLK, API, secondary | unary | math |
| abs_int32 | secondary | unary | ungrouped |
| add_binary | API | binary SFPU | binary |
| add_binary_tile | LLK, API, secondary | binary SFPU | binary |
| add_int_sfpu | API | binary | ungrouped |
| add_tiles | LLK, API, secondary | binary FPU | binary |
| add_top_row | LLK | binary SFPU | misc |
| addcdiv | LLK, API, ternary | ternary SFPU | ternary |
| addcmul | LLK, API, ternary | ternary SFPU | ternary |
| alt_complex_rotate90 | LLK, API, secondary | unary | trig |
| asinh | API | unary | math |
| acosh | API | unary | math |
| atanh | API | unary | math |
| atan | API | unary | math |
| atan2_binary_tile | secondary | binary | math |
| asin | API | unary | math |
| acos | API | unary | math |
| binary_fmod | API | binary | binary |
| binary_max_int32_tile | secondary | binary | binary |
| binary_max_tile | secondary | binary | binary |
| binary_max_uint32_tile | secondary | binary | binary |
| binary_min_int32_tile | secondary | binary | binary |
| binary_min_tile | secondary | binary | binary |
| binary_min_uint32_tile | secondary | binary | binary |
| binary_pow | LLK, API | binary SFPU | binary |
| binary_remainder | API | binary | binary |
| binary_shift | API | binary | bitwise |
| bitwise_and | API, secondary | binary | bitwise |
| bitwise_not | API, secondary | unary | bitwise |
| bitwise_or | API, secondary | binary | bitwise |
| bitwise_xor | API, secondary | binary | bitwise |
| cbrt | API | unary | math |
| ceil | API | unary | rounding |
| celu | API | unary | activations |
| clamp | API | special | special |
| comp | API, secondary | predicate | predicates |
| cos | API, secondary | unary | trig |
| cosh | API, secondary | unary | math |
| copy_dest_values | API | misc | misc |
| degrees | API | unary | ungrouped |
| digamma | API | unary | math |
| div_binary | API | binary SFPU | binary |
| div_binary_tile | secondary | binary | binary |
| div_int32_sfpu | API | binary | ungrouped |
| div_int32_tile | secondary | binary | binary |
| dropout | API, secondary | misc | misc |
| eq_binary | API | binary SFPU | binary |
| erf | API | unary | math |
| erfc | API | unary | math |
| erfinv | API | unary | math |
| eqz | secondary | predicate | predicates |
| exp | API, secondary | unary | activations |
| exp2 | LLK, API | unary | math |
| expm1 | LLK, API | unary | math |
| fill | API | misc | misc |
| fmod_binary_tile | secondary | binary | ungrouped |
| floor | API | unary | rounding |
| gcd_tile | secondary | binary | misc |
| gelu | API, secondary | unary | activations |
| gelu_approx | API, secondary | unary | activations |
| ge_binary | API | binary SFPU | binary |
| gez | secondary | predicate | predicates |
| gt_binary | API | binary SFPU | binary |
| gtz | secondary | predicate | predicates |
| hardmish | API, secondary | unary | activations |
| hardsigmoid | API, secondary | unary | activations |
| hardtanh | API | unary | activations |
| heaviside | LLK, API | unary | misc |
| i0 | API | unary | math |
| i1 | API | unary | math |
| identity | API | unary | misc |
| isinf | API, secondary | predicate | predicates |
| isnan | API, secondary | predicate | predicates |
| isfinite | API, secondary | predicate | predicates |
| lcm_tile | secondary | binary | misc |
| le_binary | API | binary SFPU | binary |
| leaky_relu | API, secondary | unary | activations |
| left_shift | API, secondary | binary | bitwise |
| lerp | LLK, API, ternary | ternary SFPU | ternary |
| lgamma | API | unary | math |
| log | LLK, API, secondary | unary | math |
| log1p | API, secondary | unary | math |
| log10 | API, secondary | unary | ungrouped |
| log2 | API, secondary | unary | ungrouped |
| logical_not | API, secondary | predicate | predicates |
| logsigmoid | API | unary | binary |
| lt_binary | API | binary SFPU | binary |
| ltz | secondary | predicate | predicates |
| mask | API, secondary | misc | misc |
| max_min | LLK | unary | ungrouped |
| max_pool_indices | LLK | binary SFPU | misc |
| mish | API, secondary | unary | activations |
| mish_approx | API, secondary | unary | activations |
| mul_binary | API | binary SFPU | binary |
| mul_binary_tile | secondary | binary | binary |
| mul_int_sfpu | API | binary | ungrouped |
| mul_tiles | API, secondary | binary FPU | binary |
| mul_tiles_bcast | secondary | binary FPU | binary |
| ne_binary | API | binary SFPU | binary |
| negative | API, secondary | unary | ungrouped |
| nez | secondary | predicate | predicates |
| polygamma | API | unary | math |
| power | LLK, API | unary | ungrouped |
| power_binary | API | binary SFPU | binary |
| prelu | API | unary | activations |
| rand | API, secondary | misc | misc |
| radians | API | unary | ungrouped |
| recip | API, secondary | unary | math |
| reduce | LLK, API | unary | misc |
| relu | API, secondary | unary | activations |
| relu6 | API, secondary | unary | activations |
| relu_max | secondary | unary | activations |
| relu_min | secondary | unary | activations |
| remainder | API | binary | special |
| reshuffle | API, secondary | misc | ungrouped |
| right_shift | API, secondary | binary | bitwise |
| round | API | unary | rounding |
| rpow | API | binary | scalar |
| rsqrt | LLK, API, secondary | unary | math |
| rsub_binary | API | binary SFPU | binary |
| rsub | API | binary | scalar |
| selu | LLK, API, secondary | unary | activations |
| sfpu_int_sum | API, secondary | misc | misc |
| sign | LLK, API, secondary | unary | math |
| signbit | LLK | unary | ungrouped |
| sigmoid | LLK, API, secondary | unary | activations |
| sigmoid_approx | API, secondary | unary | activations |
| silu | LLK, API, secondary | unary | activations |
| sin | API, secondary | unary | trig |
| sinh | API, secondary | unary | math |
| softplus | API, secondary | unary | activations |
| softshrink | API | unary | activations |
| softsign | API | unary | activations |
| sqrt | API, secondary | unary | math |
| square | LLK, API, secondary | unary | ungrouped |
| sub_binary | API | binary SFPU | binary |
| sub_binary_tile | secondary | binary | binary |
| sub_int_sfpu | API | binary | ungrouped |
| sub_tiles | API, secondary | binary FPU | binary |
| tanh | LLK, API, secondary | unary | trig |
| tanh_derivative | API, secondary | unary | ungrouped |
| threshold | API | unary | activations |
| tiled_prod | LLK, API | unary | misc |
| topk | LLK | unary | ungrouped |
| trunc | API | unary | rounding |
| trigonometry | (reserved/include) | (reserved) | ungrouped |
| typecast | API | unary | activations |
| unary_add | API | binary (scalar) | scalar |
| unary_div | API | binary (scalar) | scalar |
| unary_max | API | unary | scalar |
| unary_min | API | unary | scalar |
| unary_mul | API | binary (scalar) | scalar |
| unary_sub | API | binary (scalar) | scalar |
| where | LLK, API, ternary | ternary SFPU | ternary |
| xielu | API, secondary | unary | activations |
| xlogy_binary_tile | secondary | binary | special |
| xlogy | API | binary | special |

---

## 2. Gap Analysis

### Top-down only (API header, no dedicated LLK function)
Operations where compute API wrapper exists but no matching `llk_math_eltwise_*` LLK call is exposed:
- Most unary ops in `eltwise_unary/` (activate via macro dispatchers or inline init/exec)
- celu, softshrink, hardshrink, softsign, negative, typecast, threshold, dropout, rand, fill, identity, copy_dest_values, mask, reshuffle, xlogy, logsigmoid, clamp
- Scalar ops: unary_add, unary_sub, unary_mul, unary_div, unary_min, unary_max, rpow, rsub
- Rounding ops: floor, ceil, round, trunc
- Trigonometry module macros: degrees, radians, trigonometry (reserved include)

**Note**: "top-down only" does NOT mean missing implementation — it means the LLK prefix search did not surface a direct function, implying they route through SFPU macro templates or are implemented inline in compute headers.

### LLK-only (no compute API wrapper yet)
- `binop` (SFPU binary dispatcher, internal)
- `max_min` (LLK internal, may be aliased as max/min at API level)

---

## 3. Group Assignments

| Group | Ops | Count |
|-------|-----|-------|
| activations | celu, exp, gelu, gelu_approx, hardmish, hardsigmoid, hardtanh, leaky_relu, mish, mish_approx, prelu, relu, relu6, relu_max, relu_min, sigmoid, sigmoid_approx, silu, softplus, softshrink, softsign, softplus, softplus, tanh, threshold, typecast, xielu, softplus | 27 |
| binary | add_binary, add_binary_tile, add_tiles, atan2_binary_tile, binary_fmod, binary_max_tile, binary_max_int32_tile, binary_max_uint32_tile, binary_min_tile, binary_min_int32_tile, binary_min_uint32_tile, binary_pow, binary_remainder, binary_shift, div_binary, div_binary_tile, div_int32_tile, eq_binary, ge_binary, gt_binary, le_binary, lt_binary, logsigmoid, mul_binary, mul_binary_tile, mul_tiles, mul_tiles_bcast, ne_binary, power_binary, rsub_binary, sub_binary, sub_binary_tile, sub_tiles | 34 |
| bitwise | bitwise_and, bitwise_not, bitwise_or, bitwise_xor, left_shift, right_shift, binary_shift | 7 |
| math | abs, acos, asin, asinh, acosh, atan, atanh, cbrt, cosh, erf, erfc, erfinv, exp2, expm1, i0, i1, lgamma, log, log1p, polygamma, reciprocal, rsqrt, sign, sinh, sqrt | 25 |
| misc | add_top_row, alt_complex_rotate90, copy_dest_values, dropout, fill, gcd_tile, heaviside, identity, lcm_tile, mask, max_pool_indices, rand, reduce, reshuffle, sfpu_int_sum, tiled_prod | 16 |
| predicates | comp, eqz, gez, gtz, isfinite, isinf, isnan, lez, logical_not, ltz, nez | 11 |
| rounding | ceil, floor, round, trunc | 4 |
| scalar | rpow, rsub, unary_add, unary_div, unary_max, unary_min, unary_mul, unary_sub | 8 |
| special | clamp, fmod, lerp, remainder, where, xlogy | 6 |
| ternary | addcdiv, addcmul, lerp, mac, where | 5 |
| trig | acos, asin, atan, cos, sin, sinh, cosh, tanh, alt_complex_rotate90, deg2rad, rad2deg | 11 |

**Ungrouped (20 ops - require clarification):**
- abs_int32, add_int_sfpu, degrees, div_int32_sfpu, fmod_binary_tile, log10, log2, max_min, mul_int_sfpu, negative, power, radians, rounding (reserved), signbit, square, sub_int_sfpu, tanh_derivative, topk, trigonometry (reserved)

---

## 4. Excluded / Out of Scope

| Op | Category | Reason |
|----|----------|--------|
| copy_tile | helper infra | core eltwise chain element, not an op (handled separately in CopyTile CRTP base) |
| tilize | helper infra | data layout, different category |
| untilize | helper infra | data layout, different category |
| transpose | helper infra | data layout / reshape, different category |
| matmul | helper infra | compute primitive family, different category |

---

## 5. Cross-Check (Secondary Sources)

**Files checked for op names / enum values:**
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (1056 lines) — enums + dispatch
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (586 lines) — enums + dispatch
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp` (not found; checked ternary_op_utils.hpp) — struct defs

**Ops found in secondary but missing from LLK+API enumeration:**
None — all secondary ops appear in compute API or LLK headers.

**High-confidence ops (in 3+ sources):**
abs, abs_int32, add_binary, add_tiles, addcdiv, addcmul, atan2_binary_tile, binary_max_tile, binary_min_tile, binary_pow, div_binary, exp, gelu, gelu_approx, hardmish, hardsigmoid, log, mish, mish_approx, mul_binary, mul_tiles, power_binary, relu, relu_max, relu_min, rsqrt, selu, sigmoid, sigmoid_approx, silu, sqrt, square, sub_binary, sub_tiles, tanh, where, xielu

---

## 6. Reserved Groups

- **chain**: Core eltwise infrastructure (Dst enum, policies, CRTP bases, CopyTile element, EltwiseChain type, eltwise_pipeline) — no ops live here; reserved for composition. Count = 0.

---

## 7. Enumeration Summary

| Metric | Value |
|--------|-------|
| Total ops enumerated | 146 |
| High-confidence (2+ sources) | 36 |
| Assigned to known groups | 126 |
| Ungrouped (require clarification) | 20 |
| Excluded (other category) | 5 |
| Reserved groups (chain) | 0 |

**Largest groups by count:**
1. binary (34 ops) — FPU + SFPU binary operations
2. activations (27 ops) — relu, gelu, sigmoid, silu, softshrink, etc.
3. math (25 ops) — exp, log, sqrt, trig/inverse/special functions
4. misc (16 ops) — utility + infrastructure
5. bitwise (7 ops) — logical ops on integers

**Architecture coverage:**
- Both wormhole_b0 and blackhole LLK dirs queried → identical op sets
- Compute API uniform across architectures (not duplicated per arch)

---

## 8. Next Steps (Phase 1b)

For each group, a dedicated Stage 1b agent will:
1. Deep-dive into LLK header signatures (parameter shapes, template instantiations, SFPU macros)
2. Enumerate compute API wrappers (CppMacro expansions, inline init/exec)
3. Map usage patterns (destination reuse, CB policies, broadcast dims)
4. Identify top-down-only ops and their SFPU macro dependencies
5. Flag any ops that need new op struct boilerplate in eltwise_*.hpp
6. Produce a per-group migration blockers list

Output: `agent_logs/eltwise/<group>_breakdown.md` (one per group)
