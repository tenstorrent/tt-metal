# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap), where cap is a float parameter
- **Component operations identified**: tanh (core sub-operation), divide by scalar (x / cap), multiply by scalar (cap *), float parameter injection into SFPU kernel

## Selected References (ranked by relevance)

### 1. atanh
- **Why selected**: The `atanh` ckernel (`ckernel_sfpu_atanh.h`) is the closest structural template in the active first-class SFPU kernel set. It uses an `_init_()` function to load polynomial coefficients into `vConstFloatPrgm0/1/2` — this exact mechanism is how the `cap` parameter should be loaded into programmable registers for softcap. The loop uses `sfpi::dst_reg[0]` / `sfpi::dst_reg++`, the same `#pragma GCC unroll 8`, the same `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` signature, and is registered with an init function in `llk_math_eltwise_unary_sfpu_atanh_init()`. The math structure (complex function built from primitives requiring setup constants) directly parallels what softcap needs.
- **Relevance**: High — the `_init_` + `vConstFloatPrgm` pattern is the primary mechanism for passing `cap` into the SFPU kernel; the full ckernel structure (init function, calculate function, namespace) is the exact template to follow.

### 2. sinh
- **Why selected**: `ckernel_sfpu_sinh.h` demonstrates a full SFPU implementation of a hyperbolic function using `exp_21f` as a helper — tanh can be expressed as `(exp(2x) - 1) / (exp(2x) + 1)` using the same `exp_21f` building block. It shows: a `constexpr float log2e` scalar multiply (`x * v_log2e`), clamping to prevent underflow with `v_if`/`v_endif`, computing `exp_21f` twice, scalar arithmetic (`(exp_pos - exp_neg) * v_half`), and a small-value Taylor series fallback. This pattern directly models the tanh implementation that softcap requires, and the scalar multiply for `1/(cap)` and `cap *` at the end mirror the `* v_half` and `* v_log2e` patterns.
- **Relevance**: High — `exp_21f` helper and hyperbolic function composition via exp arithmetic is exactly how a from-scratch tanh would be built; the scalar multiply pattern (`v_log2e`, `v_half`) shows how to apply the `cap` scaling.

### 3. swish
- **Why selected**: `ckernel_sfpu_swish.h` (in the active first-class ckernel set) implements `x * sigmoid(x)` entirely at the SFPU level — a composite function that computes a sub-function then multiplies the result by the original input. This is structurally identical to `cap * tanh(x/cap)`: compute `tanh(x/cap)` in a register, then multiply by `cap`. The file shows `v_if`/`v_elseif`/`v_endif` branching, multi-segment piecewise logic, final scalar multiply (`x * sig_pos`), and the absence of an `_init_` function (constants are `constexpr float`). This is a direct model for the multiply-by-scalar step.
- **Relevance**: High — the pattern of computing a function in `sig_pos` then multiplying by `x` is the closest structural match to the `cap * tanh(...)` output step.

### 4. tanhshrink
- **Why selected**: The `tanhshrink_sfpu_kernel.cpp` compute kernel shows the exact API sequence for calling `tanh_tile_init()` + `tanh_tile(idx)` followed by scalar arithmetic. `tanhshrink(x) = x - tanh(x)` — this is the only existing operation that calls `tanh_tile` directly in a kernel. The pattern of `copy_tile` into two destination registers, then `tanh_tile`, then `sub_binary_tile` is the template for: load x, compute `tanh(x/cap)`, then compose the result. The file also shows that a dedicated `.cpp` compute kernel file (not `eltwise_sfpu.cpp`) is used for operations that need non-standard tile manipulations.
- **Relevance**: Medium-High — directly shows how `tanh_tile` is invoked in a custom kernel; the multi-register tile manipulation pattern (copy to dst[0] and dst[1]) is what softcap will need for holding intermediate scaled values.

### 5. hardshrink (hardshrink_kernel_sfpu.cpp)
- **Why selected**: `hardshrink_kernel_sfpu.cpp` is the canonical example of a parameterized unary operation that receives a float runtime argument. The pattern `get_arg_val<uint32_t>(0)` + `reinterpret_cast<const float*>(&packed_scalar)` shows exactly how to decode the packed `cap` parameter. On the host side, `unary_program_factory.cpp` shows `packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())` and `SetRuntimeArgs(program, kernel_id, core, {packed_scalar1, packed_scalar2})`. `unary_op_utils.hpp` shows `is_parametrized_type` returning true for parameterized ops. Together these files form the complete pattern for adding a new single-float-parameter unary operation to the dispatch chain.
- **Relevance**: Medium-High — the runtime argument passing infrastructure (pack_scalar_runtime_arg, SetRuntimeArgs, get_arg_val decoding) is the exact mechanism needed for passing `cap` from Python to the SFPU kernel.
