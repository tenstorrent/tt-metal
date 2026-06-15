# SFPU Op Coverage by Architecture

Source: per-op wrapper files in `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/llk_math_eltwise_{unary,binary,ternary}_sfpu_<op>.h`. Quasar's metal-API layer is skeletal, so the QSR column distinguishes:

- `[x]` — metal-API wrapper present (op is fully wired)
- `[~]` — per-op kernel only in `tt_metal/tt-llk/tt_llk_<arch>/common/inc/sfpu/ckernel_sfpu_<op>.h` (no metal wrapper). On WH/BH this means the kernel is callable directly via `ckernel::sfpu::_calculate_*_` but has no dedicated `llk_math_eltwise_*_sfpu_<op>.h` wrapper.
- `[ ]` — not present

WH/BH metal-API presence implies both the core LLK and the metal API are wired.

Path conventions: LLK-layer kernels live under `tt_metal/tt-llk/tt_llk_<arch>/common/inc/sfpu/`. Metal-layer wrappers (and a parallel set of metal-layer `ckernel_sfpu_*.h` kernels — see line 179) live under `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/`. Within this file the shorthand `common/inc/sfpu/` always refers to the LLK-layer path.

Markers used on Quasar rows:
- `†` — kernel lives in `tt_metal/tt-llk/tt_llk_quasar/common/inc/experimental/` (not `common/inc/sfpu/`)
- `§` — present as a variant inside another kernel file (e.g. `lrelu`/`relu_min`/`relu_max` all in `ckernel_sfpu_relu.h`; `stochround` referenced only by enum + `llk_math_eltwise_unary_sfpu_common.h`, no dedicated kernel file)
- `‡` — kernel exists in WH/BH `common/inc/sfpu/` but no dedicated metal-API wrapper (called via low-level `ckernel::sfpu::_calculate_*_`)

---

## Unary SFPU Ops

| Op | WH | BH | QSR |
|---|:--:|:--:|:--:|
| abs | [x] | [x] | [~]† |
| activations | [x] | [x] | [ ] |
| add1 | [x] | [x] | [ ] |
| alt_complex_rotate90 | [x] | [x] | [ ] |
| binop_with_scalar | [x] | [x] | [ ] |
| cast_fp32_to_fp16a | [x] | [x] | [ ] |
| cbrt | [x] | [x] | [ ] |
| clamp | [x] | [x] | [ ] |
| cumsum | [x] | [x] | [ ] |
| exp / exponential | [ ]‡ | [ ]‡ | [~] |
| fill | [ ]‡ | [ ]‡ | [~]† |
| exp2 | [x] | [x] | [ ] |
| expm1 | [x] | [x] | [ ] |
| gelu | [ ]‡ | [ ]‡ | [~] |
| lrelu | [ ] | [ ] | [~]§ |
| hardmish | [x] | [x] | [ ] |
| hardtanh | [x] | [x] | [ ] |
| heaviside | [x] | [x] | [ ] |
| log | [x] | [x] | [ ] |
| mask | [x] | [x] | [ ] |
| max_min | [x] | [x] | [ ] |
| power | [x] | [x] | [ ] |
| rdiv | [x] | [x] | [ ] |
| recip / reciprocal | [ ]‡ | [ ]‡ | [~] |
| reduce | [x] | [x] | [ ] |
| relu | [ ]‡ | [ ]‡ | [~] |
| relu_min | [ ] | [ ] | [~]§ |
| relu_max | [ ] | [ ] | [~]§ |
| reshuffle_rows | [x] | [x] | [ ] |
| rpow | [x] | [x] | [ ] |
| rsqrt | [x] | [x] | [~] |
| rsub_int32 | [x] | [x] | [ ] |
| selu | [x] | [x] | [ ] |
| sigmoid | [x] | [x] | [x] |
| sigmoid_appx | [x] | [x] | [ ] |
| sign | [x] | [x] | [ ] |
| signbit | [x] | [x] | [ ] |
| silu | [x] | [x] | [x] |
| sqrt | [ ]‡ | [ ]‡ | [~] |
| square | [x] | [x] | [~] |
| stochround | [ ] | [ ] | [ ]§ |
| swiglu | [ ] | [ ] | [~]† |
| tanh | [x] | [x] | [~] |
| tanh_derivative | [x] | [x] | [ ] |
| threshold | [x] | [x] | [ ] |
| tiled_prod | [x] | [x] | [ ] |
| topk | [x] | [x] | [ ] |
| typecast | [x] | [x] | [~] |
| where (unary on QSR) | — | — | [~]† |

---

## Binary SFPU Ops

| Op | WH | BH | QSR |
|---|:--:|:--:|:--:|
| add | [ ] | [ ] | [~] |
| add_int | [x] | [x] | [ ] |
| add_top_row | [x] | [x] | [ ] |
| atan2 | [x] | [x] | [ ] |
| binary_bcast | [x] | [x] | [ ] |
| binary_comp | [x] | [x] | [ ] |
| binary_fmod | [x] | [x] | [ ] |
| binary_pow | [x] | [x] | [ ] |
| binary_remainder | [x] | [x] | [ ] |
| binop | [x] | [x] | [ ] |
| bitwise | [x] | [x] | [ ] |
| copy_dest_values | [x] | [x] | [ ] |
| div_int32 | [x] | [x] | [ ] |
| div_int32_floor | [x] | [x] | [ ] |
| gcd | [x] | [x] | [ ] |
| lcm | [x] | [x] | [ ] |
| logsigmoid | [x] | [x] | [ ] |
| max_min | [x] | [x] | [ ] |
| max_pool_indices | [x] | [x] | [ ] |
| mul_int | [x] | [x] | [ ] |
| mul_int32 | [ ] | [x] | [ ] |
| quant | [x] | [x] | [ ] |
| rsub_int | [x] | [x] | [ ] |
| shift | [x] | [x] | [ ] |
| sub_int | [x] | [x] | [ ] |
| xlogy | [x] | [x] | [ ] |

---

## Ternary SFPU Ops

| Op | WH | BH | QSR |
|---|:--:|:--:|:--:|
| addcdiv | [x] | [x] | [ ] |
| addcmul | [x] | [x] | [ ] |
| lerp | [x] | [x] | [ ] |
| where (ternary on WH/BH) | [x] | [x] | — |

---

---

## Other SFPU Categories (outside unary/binary/ternary)

These wrappers don't follow the `llk_math_eltwise_{unary,binary,ternary}_sfpu_*` pattern, so they aren't in the tables above. All exist on WH+BH; none on QSR yet.

### Welford's (multi-tile mean/variance reduction)

Dedicated scaffolding in core LLK: `tt_llk_<arch>/llk_lib/llk_math_welfords_sfpu.h` + `..._params.h`. Metal wrapper: `llk_math_welfords_sfpu_entry.h`. APIs: `_init`, `_clear_previous_mean_and_m2`, `_reinit`, `_calculate_welfords_tile_`, `_calculate_welfords_partial_tile_`, `_store_mean_m2_to_dst`, `_load_mean_m2_from_dst`, `_store_mean_var_to_dst_row`, `_store_mean_var_to_dst_raw`.

| Op | WH | BH | QSR |
|---|:--:|:--:|:--:|
| welfords | [x] | [x] | [ ] |

### EMA (exponential moving average)

Metal wrapper only: `llk_math_ema_sfpu_entry.h`. Internally piggybacks on the ternary scaffolding (`_llk_math_eltwise_ternary_sfpu_init_<SfpuType::unused>()`). APIs: `_init`, `_load_alpha_beta`, `_clear_previous_output`, `_tile`.

| Op | WH | BH | QSR |
|---|:--:|:--:|:--:|
| ema | [x] | [x] | [ ] |

### lgamma / polygamma (mixed-arity bundles)

Use the arity-less wrapper file `llk_math_eltwise_sfpu_{lgamma,polygamma}.h` — they expose entry points across multiple arity classes:

| Op variant | Arity | WH | BH | QSR |
|---|---|:--:|:--:|:--:|
| lgamma_stirling (unary) | unary | [x] | [x] | [ ] |
| lgamma_stirling (binary) | binary | [x] | [x] | [ ] |
| lgamma_adjusted | ternary | [x] | [x] | [ ] |
| polygamma | unary | [x] | [x] | [ ] |

---

## Wrappers That Bundle Multiple Op Variants

Several wrappers in the main tables expose more than one entry point. Open the file to see the full set; here are the notable bundles:

| Wrapper | Family | Variants exposed |
|---|---|---|
| `max_min` | unary | `unary_max`, `unary_min`, `unary_{max,min}_int32`, `unary_{max,min}_uint32` |
| `max_min` | binary | `binary_max`, `binary_min`, `binary_{max,min}_int32`, `binary_{max,min}_uint32` |
| `binary_comp` | binary | `eq_fp32`, `ne_fp32`, `lt_fp32`, `le_fp32`, `gt_fp32`, `ge_fp32`, `lt_int`, `le_int`, `gt_int`, `ge_int` |
| `binop` | binary | `binop`, `binop_mul`, `binop_div` |
| `binary_bcast` | binary | `add_bcast_{row,col}`, `mul_bcast_{row,col}`, `sub_bcast_{row,col}` |
| `quant` | binary | `quant_int32`, `requant_int32`, `dequant_int32` |
| `shift` | binary | `left_shift`, `right_shift`, `logical_right_shift` |
| `binary_fmod` | binary | `binary_fmod`, `fmod_int32` |
| `binary_remainder` | binary | `binary_remainder`, `remainder_int32` |
| `div_int32` | binary | `div_int32`, `div_int32_trunc` |
| `binop_with_scalar` | unary | `binop_with_scalar`, `binop_with_scalar_{add,sub}_int32` |
| `abs` | unary | `abs`, `abs_int32` |
| `clamp` | unary | `clamp`, `clamp_int32` |
| `mask` | unary | `mask`, `mask_posinf` |
| `log` | unary | `log`, `log_with_base` |
| `power` | unary | `power`, `power_iterative` |
| `signbit` | unary | `signbit`, `signbit_int32` |
| `topk` | unary | `topk_local_sort`, `topk_merge`, `topk_rebuild` |
| `activations` | unary | `celu`, `hardshrink`, `hardsigmoid`, `softshrink`, `softsign` |

Additional kernels exist with no dedicated `llk_math_eltwise_*_sfpu_<op>.h` wrapper — callers reach them through low-level `ckernel::sfpu::_calculate_*_` directly. These split across two paths:

- **LLK-layer kernels** in `tt_metal/tt-llk/tt_llk_<arch>/common/inc/sfpu/`: `elu`, `dropout`, `trigonometry` (plus the `[ ]‡` and `[~]‡` ops marked in the tables above).
- **Metal-layer kernels** in `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` (same directory as the wrappers, but no matching wrapper file): `prelu`, `xielu`, `softplus`, `softshrink`, `softsign`, `hardshrink`, `digamma`, `i0`, `i1`, `erf`, `erfc`, `erfinv`, `log1p`, `identity`, `sqrt_custom`, `rand`, `conversions`, `piecewise_rational`, `celu`, `int_sum`, `logical_not`, `unary_comp`, `unary_max_min`, `unary_power`, `binary_max_min`, `binop_with_unary`, `bitwise_{and,not,or,xor}`, `fmod`, `remainder`, `left_shift`, `right_shift`, `mul_int32` (WH only — BH has a wrapper). Some of these (`celu`, `softshrink`, `softsign`, `hardshrink`) are also exposed through the `activations` bundle; the rest are reachable only via the low-level path.

Quasar has a parallel metal-layer kernel directory at `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/` containing `ckernel_sfpu_{relu,sigmoid,silu}.h` (alongside the `sigmoid`/`silu` wrappers).

---

## Summary

| Family | WH | BH | QSR (wrapped) | QSR (kernel only) |
|---|:--:|:--:|:--:|:--:|
| Unary | 37 | 37 | 2 | 15 |
| Binary | 24 | 25 | 0 | 1 |
| Ternary | 4 | 4 | 0 | 0 (no scaffolding at all) |

Notes (verified by grep on 2026-05-18):

- BH-only delta: `binary/mul_int32` (WH has `mul_int` but not `mul_int32` as a separate wrapper).
- Quasar wrapped ops (`sigmoid`, `silu`) live at `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/`.
- Quasar kernel-only ops in `tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu/`: `add` (binary), `exp`, `gelu`, `recip`, `relu`, `rsqrt`, `sigmoid`, `silu`, `sqrt`, `square`, `tanh`, `typecast_fp16b_uint16`, `typecast_int32_fp32`. Plus relu variants `lrelu`/`relu_min`/`relu_max` colocated in `ckernel_sfpu_relu.h`.
- Quasar **experimental** kernels in `tt_metal/tt-llk/tt_llk_quasar/common/inc/experimental/`: `abs`, `fill`, `swiglu`, `where`. All four Quasar SFPU tests (`sfpu_abs_quasar_test.cpp`, `sfpu_fill_quasar_test.cpp`, `sfpu_swiglu_quasar_test.cpp`, `sfpu_where_quasar_test.cpp`) include from `experimental/` and route through `llk_math_eltwise_unary_sfpu_common.h` — so all four are **unary** on Quasar.
- **`where` arity differs by arch**: ternary on WH/BH (`SfpuType::where` via `llk_math_eltwise_ternary_sfpu_where.h`), unary on Quasar (`experimental/ckernel_sfpu_where.h` via unary scaffolding).
- **No ternary scaffolding on Quasar**: `find tt_metal/tt-llk/tt_llk_quasar -name 'llk_math_eltwise_ternary*'` returns nothing. Only `llk_math_eltwise_unary_sfpu_common.h` and `llk_math_eltwise_binary_sfpu.h` exist.
- **Quasar uses non-templated init**: tests call `_llk_math_eltwise_unary_sfpu_init_()` and `_llk_math_eltwise_binary_sfpu_init_()` without an `<SfpuType::X>` template argument (unlike WH/BH). Grep for `_llk_math_eltwise_*_sfpu_init_<SfpuType::` against Quasar paths returns zero matches.
- Quasar `SfpuType` enum (`tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h:74-96`) — 20 enumerators verified verbatim: `tanh, gelu, exponential, reciprocal, sqrt, rsqrt, relu, lrelu, relu_min, relu_max, stochround, typecast, add, square, sigmoid, silu, abs, fill, swiglu, where`. Note: `exponential` (not `exp`), `reciprocal` (not `recip`).
- Active `SfpuType::X` template uses across Quasar code (grep tally): `gelu`(5), `silu`(4), `sigmoid`(4), `tanh`(3), `sqrt`(3), `relu`(3), `reciprocal`(3), `exponential`(3). Everything else (`abs`, `where`, `fill`, `swiglu`, `add`, `square`, `typecast`, `lrelu`, `relu_min`, `relu_max`, `stochround`) is referenced only in the enum or scaffolding — not in any concrete dispatch site.
- Quasar `BinaryOp` enum (`tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h:98-103`) only has `ADD`, `SUB`, `MUL`. WH/BH `BinaryOp` (`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h:244-257`) also has `DIV`, `RSUB`, `POW`, `XLOGY`, `RSHFT`, `LSHFT`, `LOGICAL_RSHFT`, `ADD_TOP_ROW`.
