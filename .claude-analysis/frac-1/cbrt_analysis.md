# Analysis: cbrt (reference for frac)

## SFPU Kernel Pattern
- **File**: `ckernel_sfpu_cbrt.h`
- **Namespace**: `ckernel::sfpu`
- **Template**: `template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>`
- **Function**: `calculate_cube_root()` - extra template param `is_fp32_dest_acc_en`
- **Init function**: `cube_root_init()` sets `vConstFloatPrgm0/1/2`

## Key SFPI Patterns
- Uses `sfpi::int32_to_float()`, `sfpi::reinterpret<vInt>()`, `sfpi::reinterpret<vFloat>()`
- Bit manipulation: `sfpi::reinterpret<vInt>(f) << 8` (shift left)
- Uses `sfpi::abs()`, `sfpi::setsgn()`, `sfpi::addexp()`
- `sfpi::float_to_fp16b(y, 0)` for bfloat16 path
- Programmable constants: `sfpi::vConstFloatPrgm0/1/2`

## LLK Dispatch
- Init: `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`
- Compute: passes `fp32_dest_acc_en` as template param via `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`

## Key Takeaway for frac
- Shows advanced SFPI bit manipulation patterns (reinterpret, shift, exponent manipulation)
- Shows how to handle both fp32 and fp16b dest accumulator modes
- The `reinterpret<vInt>()` pattern is exactly what we need for frac's floor implementation
