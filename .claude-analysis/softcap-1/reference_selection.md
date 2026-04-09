# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: `cap * tanh(x / cap)`
- **Component operations identified**:
  - scalar divide: `x / cap` (equivalently, multiply by `1/cap`)
  - tanh: `tanh(scaled_x)`
  - scalar multiply: `result * cap`
  - runtime scalar parameter: `cap` must be passed as a runtime argument and applied as a scalar multiply/divide

## Selected References (ranked by relevance)

### 1. atanh
- **Why selected**: `atanh` is the most directly structurally relevant operation in the custom SFPU kernel layer. It is implemented as a full custom `ckernel_sfpu_atanh.h` + `llk_math_eltwise_unary_sfpu_atanh.h` + `api/compute/eltwise_unary/atanh.h` stack — exactly the three-layer pattern (`ckernel_sfpu_*.h`, `llk_math_eltwise_unary_sfpu_*.h`, `api/compute/eltwise_unary/*.h`) that softcap will need to follow. It also requires a custom `_init` function that stores polynomial coefficients into `vConstFloatPrgm{0,1,2}`, demonstrating how to use the programmable constant registers. The loop body pattern (`dst_reg[0] = ...; dst_reg++;`) matches what softcap will use.
- **Relevance**: high — provides the exact file structure, init pattern, loop skeleton, and `sfpi::dst_reg` iteration pattern for a new custom SFPU kernel that gets registered through `unary_op_utils.cpp` via the `SFPU_OP_ATANH_INCLUDE` define.

### 2. tanhshrink (kernel: tanhshrink_sfpu_kernel.cpp)
- **Why selected**: `tanhshrink(x) = x - tanh(x)` uses `tanh_tile()` directly as a building block and shows the DST-register multi-slot pattern needed for softcap: load input to a slot, apply tanh, then perform an arithmetic combination. The `tanhshrink_sfpu_kernel.cpp` version specifically demonstrates how to use `copy_tile_to_dst_init_short`, `copy_tile` into multiple DST slots, `tanh_tile`, and `sub_binary_tile` together — the same primitives softcap needs (copy to slot 0, fill scalar cap to slot 1, divide, tanh, multiply by cap). It also shows how to include `eltwise_binary_sfpu.h` alongside tanh for multi-operand in-DST arithmetic.
- **Relevance**: high — tanh is the core computation of softcap; this file shows how to drive tanh_tile in a real compute kernel alongside binary tile operations within the DST register space.

### 3. swish
- **Why selected**: `swish(x) = x * sigmoid(x)` is a composite operation of the same structural form as softcap: a non-linear activation (sigmoid) applied to a scaled version of x, then the result is multiplied back by x. `ckernel_sfpu_swish.h` is a self-contained custom SFPU kernel (no external init needed, no `_init` callback registered in `llk_math_eltwise_unary_sfpu_swish_init`), making it the simplest complete example of the full kernel file format. It demonstrates the `for (int d = 0; d < ITERATIONS; d++)` loop body, `sfpi::vFloat x = sfpi::dst_reg[0]`, result assignment, and `sfpi::dst_reg++` increment. It also shows the LLK wrapper pattern where `swish_init` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()` without a custom init callback — relevant if softcap stores cap via runtime args rather than `vConstFloatPrgm` registers.
- **Relevance**: high — cleanest example of a self-contained composite unary SFPU kernel that computes `f(x) * g(x)` in a single loop, matching softcap's `cap * tanh(x/cap)` structure.

### 4. hardshrink (kernel: hardshrink_kernel_sfpu.cpp)
- **Why selected**: `hardshrink` is the most complete example of a **parametrized** unary operation in this codebase: it reads a runtime scalar parameter via `get_arg_val<uint32_t>(0)`, reinterprets it as float via `reinterpret_cast<const float*>`, and passes it to `fill_tile(dst_slot, *lambd)` to create a scalar tile in DST. The `hardshrink_kernel_sfpu.cpp` version specifically uses `copy_tile_to_dst_init_short`, `fill_tile`, `add_binary_tile_init`, `mul_binary_tile_init`, and multi-DST-slot access — which is the same parameter-application pattern softcap will need for the `cap` scalar. The program factory (`unary_program_factory.cpp`) also shows exactly how `packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())` is wired for `HARDSHRINK`, providing the host-side template for registering softcap's `cap` parameter.
- **Relevance**: high — the definitive reference for how a single float runtime parameter is passed from host (`unary_program_factory.cpp`), received in the compute kernel (`get_arg_val`), and applied as a tile-scalar via `fill_tile` for use in DST-register arithmetic.

### 5. sinh
- **Why selected**: `sinh(x) = (exp(x) - exp(-x)) / 2` is a custom SFPU kernel that requires a non-trivial sub-function (`exp_21f`) and uses `sfpi::addexp`, `sfpi::setsgn`, and `sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(...))` — the more advanced sfpi primitives. It also demonstrates the `sinh_init` pattern where the init function is registered as a callback via `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>(sfpu::sinh_init<APPROXIMATE>)` but the init body is empty (no-op). This is useful as a contrast: softcap may not need custom polynomial constants at all if cap is passed as a runtime arg, and sinh shows what an empty init + non-trivial compute body looks like. Additionally, sinh shows `sfpi::setsgn(x, 0)` for abs value, which may be useful for testing cap sign handling.
- **Relevance**: medium — provides the complete "non-trivial custom kernel with no programmable constants in init" structural template, plus advanced sfpi intrinsic usage patterns that may be needed if softcap implements tanh natively rather than using the built-in `tanh_tile`.
