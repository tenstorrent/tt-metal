# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: RReLU(x) = max(0, x) + a * min(0, x), where a = Uniform(lower, upper) in training mode and a = (lower + upper) / 2 in evaluation mode. Parameters: lower (float, default 0.125), upper (float, default 1/3), training (bool, default False).
- **Component operations identified**: conditional branch on sign of x (x >= 0 passthrough, x < 0 scaled), multiply-by-float-scalar, runtime float parameters (lower, upper), two-parameter registration

## Selected References (ranked by relevance)

### 1. hardshrink
- **Why selected**: Hardshrink is the closest structural match to rrelu. It uses the same `packed_scalar` float runtime argument pattern (bit-cast from `uint32_t` via `reinterpret_cast<const float*>`), and its dedicated compute kernel (`hardshrink_kernel.cpp` and `hardshrink_kernel_sfpu.cpp`) shows exactly the `fill_tile(float) + conditional (ltz_tile/gtz_tile) + mul` pipeline that rrelu requires. The program factory explicitly packs `ops_chain[0]` param 0 into `packed_scalar1` for hardshrink, which is the same mechanism rrelu will use to pass the slope `a`. The kernel also uses a temporary circular buffer (`cb_tmp0`) to stage intermediate results — exactly what rrelu eval-mode may need.
- **Relevance**: high — the entire conditional-multiply-by-scalar-param pattern is directly reusable; the kernel structure, CB setup, and runtime-arg passing in the program factory are the primary templates for rrelu.

### 2. swish
- **Why selected**: Swish (`ckernel_sfpu_swish.h`) is the clearest inline SFPU kernel example showing `v_if(x < 0.0f) { ... } v_endif;` conditional branching on the sign of x. It demonstrates the standard SFPU tile loop structure (`template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`, `sfpi::dst_reg[0]`, `sfpi::dst_reg++`), `v_if`/`v_endif` guard syntax, and ending with `sfpi::dst_reg[0] = x * sig_pos` (multiply result). This is the exact pattern for the rrelu SFPU kernel: load x, branch on sign, multiply negative branch by `a`, write result.
- **Relevance**: high — the SFPU ckernel file structure, `v_if(x < 0.0f)` conditional, and inline multiply directly inform the `ckernel_sfpu_rrelu.h` implementation; the LLK wrapper pattern in `llk_math_eltwise_unary_sfpu_swish.h` shows the no-custom-init-callback path.

### 3. hardtanh
- **Why selected**: Hardtanh takes two float parameters (min_val, max_val) and is registered in `is_parametrized_type` in `unary_op_utils.hpp`. It is invoked via `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}` — a two-float constructor. RReLU in eval mode similarly needs two float parameters (lower, upper), and in the full form may need both passed to compute `a = (lower + upper) / 2`. The `is_parametrized_type` registration and the two-float `UnaryWithParam` constructor in `unary.hpp` are the direct templates for adding rrelu's parameter declaration.
- **Relevance**: high — the two-float parameter registration infrastructure (`is_parametrized_type`, `get_op_init_and_func_parameterized`, two-float `UnaryWithParam` constructor, and the `get_op_init_and_func` dispatch) is exactly what rrelu must replicate for its lower/upper params.

### 4. frac
- **Why selected**: Frac (`ckernel_sfpu_frac.h`) demonstrates multi-branch `v_if`/`v_endif` conditional logic inside an SFPU tile loop based on a computed value (exponent compared to 0 and 23). It specifically uses `v_if(exp < 0)` — a sign-based branch — and shows how to compute a result and write it into `sfpi::dst_reg[0]` conditionally. The init pattern (no custom init callback, simple no-op) matches the expected rrelu init, and the LLK wrapper in `llk_math_eltwise_unary_sfpu_frac.h` shows the minimal wrapping needed for a simple no-init-param SFPU op.
- **Relevance**: medium — the `v_if(condition)` / default-path / `v_endif` branch structure for sign-based branching and the simple init-less LLK wrapper are directly applicable to rrelu's SFPU implementation.

### 5. where_tss
- **Why selected**: Where_tss is the only existing operation in the current codebase that uses two packed scalar runtime arguments (`packed_scalar1` and `packed_scalar2`) passed to the compute kernel. The program factory explicitly shows: `packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype()); packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());` and `SetRuntimeArgs(program, kernel_id, core, {packed_scalar1, packed_scalar2})`. RReLU in its full eval-mode form needs to pass two floats (lower and upper) or a pre-computed slope `a` — the two-param runtime arg infrastructure shown by where_tss is the reference for that. The `UnaryWithParam{UnaryOpType::WHERE_TSS, {t_true, t_false}}` two-float vector constructor is also the direct template for `UnaryWithParam{UnaryOpType::RRELU, {lower, upper}}`.
- **Relevance**: medium — the two-packed-scalar runtime arg infrastructure (program factory, SetRuntimeArgs with two scalars, kernel access via `get_arg_val<uint32_t>(0)` and `get_arg_val<uint32_t>(1)`) directly informs how rrelu passes lower and upper (or slope a) to the compute kernel.
