# Reference Operation Selection for swish

## Target Operation
- **Name**: swish
- **Definition**: x / (1 + exp(-x))
- **Equivalent form**: x * sigmoid(x)  (since sigmoid(x) = 1 / (1 + exp(-x)))
- **Component operations identified**: exp(-x), negation (-x), addition (1 + ...), reciprocal (1 / ...), multiplication (x * ...) — or equivalently: sigmoid(x) followed by multiply-by-x

## Key Observation

Swish is mathematically identical to SiLU (Sigmoid Linear Unit): `swish(x) = x * sigmoid(x) = silu(x)`.
The codebase already implements `SILU` in `UnaryOpType` with its SFPU kernel at
`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h`.
The implementor should treat `silu` as the definitive structural template.

## Selected References (ranked by relevance)

### 1. silu
- **UnaryOpType**: `SILU`
- **SFPU kernel**: `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h`
- **Registration**: `REGISTER_UNARY_OPERATION(silu, SILU)` in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:149`
- **Why selected**: Mathematically identical to swish. The actual kernel: loads `val = dst_reg[0]`, computes `result = abs(val)`, applies `_sigmoid_piecewise_linear_positive_(result)` (piecewise linear via `POLYVAL5`), conditionally flips for negative x (`1.0f - result`), then stores `val * result`. This is exactly what swish should do.
- **Relevance**: **high** — entire SFPU kernel body, `_sigmoid_piecewise_linear_positive_` helper, `#pragma GCC unroll 8` idiom, `dst_reg++` loop, and all registration macros transfer directly. The swish kernel is a rename of silu.

### 2. sigmoid
- **UnaryOpType**: `SIGMOID`
- **SFPU kernel**: `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sigmoid.h`
- **Why selected**: Swish = x * sigmoid(x). The sigmoid kernel uses a 6-piece LUT loaded via `_sfpu_load_imm32_` into LRegs 0–2, 4–6, then applies `lut2(val, ...) + 0.5f` per tile. The `_init_sigmoid_()` function sets up these LUT coefficients. Understanding this init/compute split is needed for any alternate implementation that calls sigmoid directly rather than the piecewise linear helper inside silu.
- **Relevance**: **high** — the `_sigmoid_piecewise_linear_positive_` helper inside silu is a faster approximation of sigmoid; this reference shows the LUT-based accurate path as the alternative.

### 3. elu
- **UnaryOpType**: `ELU`
- **SFPU kernel**: `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h`
- **Registration**: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(elu, ELU)` in `unary.hpp:189`
- **Why selected**: Uses `_sfpu_exp_21f_bf16_(v)` — the same exp primitive needed to compute exp(-x) in the denominator. Also demonstrates the `is_fp32_dest_acc_en` non-approx path with explicit `float_to_fp16b` rounding, and the `v_if (v < 0.0f)` conditional pattern. Shows how `#include "ckernel_sfpu_exp.h"` is used within an activation kernel.
- **Relevance**: **medium** — exp(-x) computation pattern, `_sfpu_exp_21f_bf16_` call, and fp16b rounding are directly relevant if swish implements the exact formula rather than the piecewise-linear shortcut.

### 4. hardswish
- **UnaryOpType**: `HARDSWISH`
- **SFPU kernel**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- **Why selected**: Structurally identical: `hardswish(x) = x * hardsigmoid(x)` — same "compute activation → multiply by x" idiom. The kernel shows: compute `hsigmoid = x * (1/6) + 0.5f`, clamp with `v_if / v_endif` to [0, 1], store `x * hsigmoid`. Also demonstrates the `SFPU_OP_HARDSWISH_INCLUDE` nonstandard macro path vs. the default path used by most operations.
- **Relevance**: **medium** — the `x * activation(x)` multiplication idiom and clamped-sigmoid piecewise implementation; the macro registration path difference is worth understanding.

### 5. logsigmoid
- **UnaryOpType**: `LOGSIGMOID`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logsigmoid_kernel.cpp`
- **Why selected**: Logsigmoid uses a custom (non-`eltwise_sfpu.cpp`) compute kernel that holds the input in two DST registers simultaneously — `copy_tile(cb_input, 0, 0)` and `copy_tile(cb_input, 0, 1)` — then negates DST[1], applies exp, and combines. This multi-register pattern is the reference for any swish implementation that wants to hold both `x` and `sigmoid(x)` in DST concurrently. Also shows sequencing of `negative_tile_init()` + `exp_tile_init()` + custom operation.
- **Relevance**: **medium** — multi-register DST pattern, custom compute kernel structure, negation+exp+sigmoid composition; relevant if swish is implemented as a custom kernel rather than a pure SFPU chain.

## Summary Table

| Rank | Operation | Key Pattern Borrowed |
|------|-----------|---------------------|
| 1 | silu | Complete structural template (mathematically identical) |
| 2 | sigmoid | LUT-based sigmoid init/compute, piecewise-linear alternative |
| 3 | elu | `_sfpu_exp_21f_bf16_` for exp(-x), fp16b rounding pattern |
| 4 | hardswish | `x * activation(x)` multiplication idiom, macro registration |
| 5 | logsigmoid | Multi-DST register approach, custom compute kernel structure |
