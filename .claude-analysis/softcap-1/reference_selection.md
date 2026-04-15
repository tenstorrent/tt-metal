# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: scalar division (x / cap = x * (1/cap)), tanh nonlinearity, scalar multiply (result * cap), runtime scalar parameter (cap)

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: swish = x * sigmoid(x) is the structurally closest analogy in the codebase. Both swish and softcap compose a scalar multiply with a nonlinear activation in a single SFPU kernel. The `ckernel_sfpu_swish.h` implementation (wormhole_b0 custom) shows exactly the pattern needed: compute nonlinear(x) in a piecewise manner, then multiply by a scalar derived from x. The `SfpuType::swish` enum entry, `SFPU_OP_SWISH_INCLUDE` macro guard, `swish_tile_init()` / `swish_tile()` API, and `llk_math_eltwise_unary_sfpu_swish.h` wiring all directly mirror the full stack softcap needs to implement.
- **Relevance**: high — complete end-to-end reference for a custom SFPU kernel combining multiply + nonlinear activation, registered as a new SfpuType with its own include macro

### 2. atanh
- **Why selected**: atanh is a newly added custom SFPU operation in this codebase (not in the third-party LLK) that shows exactly how to add a new SfpuType with a custom `init()` function. `ckernel_sfpu_atanh.h` demonstrates loading polynomial constants into `vConstFloatPrgm0/1/2` in an `atanh_init<APPROXIMATION_MODE>()` function, then consuming them in `calculate_atanh()`. The `llk_math_eltwise_unary_sfpu_atanh_init()` calling `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>)` is the exact pattern for registering a custom init callback. Since softcap may need to bake `1/cap` or `cap` into a programmable constant at init time, this is the key registration reference.
- **Relevance**: high — shows how to add a new SfpuType entry with a custom init that loads scalar constants, and the full llk_math wiring

### 3. tanhshrink
- **Why selected**: tanhshrink = x - tanh(x) is implemented as a custom compute kernel (`tanhshrink_sfpu_kernel.cpp`) that calls `tanh_tile_init()` and `tanh_tile()` as a building block, then chains it with another operation. Since softcap requires calling tanh directly via the `tanh_tile` API followed by a multiply, this is the most direct reference for how to use tanh as a sub-operation inside a custom multi-step compute kernel. The kernel shows DST register slot management (loading x into slot 1, computing tanh, then operating with original x in slot 0).
- **Relevance**: high — direct reference for calling tanh_tile as a primitive inside a custom compute kernel with multiple DST slot operations

### 4. sinh
- **Why selected**: sinh = (exp(x) - exp(-x)) / 2. The `ckernel_sfpu_sinh.h` provides the `exp_21f<APPROXIMATION_MODE>()` helper (Moroz 2022 algorithm for 2^z) which may be reused if softcap implements tanh via the exponential formula tanh(x) = (e^2x - 1)/(e^2x + 1). It also shows the full SFPU kernel structure: clamping to prevent overflow/underflow, small-|x| Taylor fallback to avoid cancellation, and converting result to bfloat16 for deterministic rounding. The `SfpuType::sinh` registration and parameter-less `sinh_init()` pattern is the simplest form of a new SfpuType to contrast with atanh's init-with-constants pattern.
- **Relevance**: medium — provides `exp_21f` exponential helper potentially reusable in tanh implementation, and shows numerical stability patterns (overflow clamping, small-x fallback)

### 5. hardtanh
- **Why selected**: hardtanh takes two float parameters (min_val, max_val) passed as FP16B-packed uint32 runtime args. The `ckernel_sfpu_hardtanh.h` shows how to receive scalar parameters from the host via `uint32_t param0` / `param1` arguments and unpack them with `sfpi::s2vFloat16b(param0)` to get a `vFloat` constant broadcasted across all 32 SFPU lanes. Softcap has exactly one float parameter (`cap`) that must be delivered this way. This is the reference for the scalar parameter passing convention used in the existing SFPU infrastructure.
- **Relevance**: medium — shows the float parameter passing and unpacking pattern (s2vFloat16b) that softcap needs for its `cap` parameter, and demonstrates a two-parameter SFPU function signature
