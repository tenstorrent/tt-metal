# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap), where cap is a float parameter (positive scalar, default 50.0)
- **Component operations identified**: scalar multiply (by cap), scalar divide (by 1/cap, i.e. multiply by 1/cap), tanh, scalar multiply (by cap), float parameter handling

## Selected References (ranked by relevance)

### 1. tanhshrink
- **Why selected**: Tanhshrink computes `x - tanh(x)`, making it the most structurally similar operation in the codebase that uses `tanh_tile_init()` + `tanh_tile()` in the SFPU compute kernel. The `tanhshrink_sfpu_kernel.cpp` shows exactly how to call `tanh_tile_init()` and `tanh_tile()` within the tile processing loop, copy tiles into DST, and then combine with a binary op. Softcap also applies tanh as the central transform and then multiplies the result — the kernel structure is nearly identical except softcap replaces the subtraction with scalar scaling.
- **Relevance**: high — directly shows `tanh_tile_init()` / `tanh_tile()` usage in a compute kernel, which is the core of softcap

### 2. swish
- **Why selected**: Swish (`x * sigmoid(x)`) is a composite operation that applies a non-linear function (sigmoid) and then multiplies the result by the original input. The `ckernel_sfpu_swish.h` implementation is the clearest existing example of a custom SFPU kernel that (a) approximates a hyperbolic-adjacent activation using polynomial/piecewise approach, (b) multiplies the activation output by a scalar (the input itself), and (c) follows the `#pragma GCC unroll 8` / `dst_reg[0]` / `dst_reg++` iteration pattern that softcap's ckernel will use. Its sfpi.h-based implementation style directly matches what a `ckernel_sfpu_softcap.h` file will need.
- **Relevance**: high — best existing ckernel_sfpu_*.h pattern for a composite activation with a final multiplication step; also uses the `llk_math_eltwise_unary_sfpu_*.h` wrapper pattern

### 3. hardshrink
- **Why selected**: Hardshrink is a parameterized unary operation (`REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER`) that receives a float scalar at runtime via `get_arg_val<uint32_t>(0)` / `reinterpret_cast<const float*>`. The `hardshrink_kernel.cpp` and its sfpu variant show exactly how to: (a) pack a float scalar as a runtime arg in the program factory (`pack_scalar_runtime_arg`), (b) retrieve it in the compute kernel, and (c) use `fill_tile()` + binary tile ops to apply scalar values inside the tile loop. Softcap needs the same float parameter handling for `cap`.
- **Relevance**: high — definitive reference for the float-parameter pipeline: program factory packing (unary_program_factory.cpp lines 129-130), kernel arg retrieval pattern, and `fill_tile()` for scalar operations

### 4. atanh
- **Why selected**: Atanh is the most mathematically related to softcap: `atanh(x) = 0.5 * ln((1+x)/(1-x))` is the inverse of tanh. The `ckernel_sfpu_atanh.h` implementation uses the `sfpi::vConstFloatPrgm0/1/2` programmable constant registers in `atanh_init()` and then applies them in `calculate_atanh()`. This shows how to store a precomputed scalar (like `1/cap`) into a programmable constant for efficient reuse across 8 SFPU lanes per iteration, which is the recommended pattern for a per-kernel scalar. The LLK wrapper `llk_math_eltwise_unary_sfpu_atanh.h` demonstrates an `_init()` function that sets constants, exactly matching what `softcap_init()` needs to do for `1/cap` and `cap`.
- **Relevance**: high — shows `atanh_init()` / programmable-constant pattern for storing the cap scalar; also demonstrates the unary SFPU registration pattern for ops that need a custom init function

### 5. sinh
- **Why selected**: Sinh (`(exp(x) - exp(-x)) / 2`) is a hyperbolic function implemented from scratch in `ckernel_sfpu_sinh.h` using the `exp_21f<>` helper and a Taylor fallback for small inputs. It demonstrates: (a) the full `calculate_*` + `*_init()` split with a no-op init, (b) use of `sfpi::vFloat` constants declared inside the function, (c) the `#pragma GCC unroll 0` pattern with explicit loop bounds, and (d) the `float_to_fp16b` rounding call at the end for deterministic bfloat16 output. For softcap, the sinh pattern for building a custom hyperbolic kernel from sfpi primitives and managing precision rounding is directly applicable.
- **Relevance**: medium — structural reference for writing a hyperbolic-function ckernel_sfpu_*.h from sfpi primitives, especially the `calculate_*` + `*_init()` template split and precision rounding
