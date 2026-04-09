# SFPU Kernel Analysis: `frac`

## 1. Operation Summary

| Property | Value |
|---|---|
| **Operation name** | `frac` |
| **Math definition** | `frac(x) = x - trunc(x)` (fractional part, truncation toward zero) |
| **PyTorch equivalent** | `torch.frac(x)` |
| **UnaryOpType enum** | `UnaryOpType::FRAC` |
| **SfpuType enum** | `SfpuType::frac` |
| **Parameterized** | No (non-parameterized unary op) |
| **Approximation mode** | Not used (`get_op_approx_mode` returns `false`) |
| **Supported dtypes** | BFLOAT16, BFLOAT8_B, FLOAT32 |
| **Has backward** | Yes (`frac_bw` — trivially passes gradient through) |

## 2. File Manifest (All Abstraction Layers)

### Layer 1: SFPU Kernel Function (`ckernel_sfpu_*.h`)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

### Layer 2: LLK Math Wrapper (`llk_math_eltwise_unary_sfpu_*.h`)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`

### Layer 3: Compute API Header (`frac.h`)
- `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`

### Layer 4: Split Include Gate (`sfpu_split_includes.h`)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

### Layer 5: SfpuType Enum (`llk_sfpu_types.h`)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

### Layer 6: UnaryOpType Enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — `FRAC` enum value

### Layer 7: Op Utils — Dispatch (init/func strings, macro defines)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

### Layer 8: C++ Registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — `REGISTER_UNARY_OPERATION(frac, FRAC)`

### Layer 9: Python Nanobind
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — `bind_unary_operation<"frac", &ttnn::frac>(...)`

### Layer 10: Composite Fallback (host-side)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.hpp` — declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp` — `frac(x) = subtract(x, trunc(x))`

### Layer 11: Tests
- `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`
- `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_frac.py`

### Backward Operation
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.hpp` — `frac_bw` declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` — `frac_bw` returns `grad` unchanged
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp` — Python binding

## 3. SFPU Kernel Deep Dive

### 3.1 Core Algorithm (`calculate_frac`)

**Location**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

The Wormhole B0 and Blackhole implementations are **identical**.

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat trunc_x = x;           // default: frac=0 for integers
        sfpi::vInt exp = sfpi::exexp(x);     // extract unbiased exponent

        v_if(exp < 0) { trunc_x = 0.0f; }   // |x| < 1: trunc=0
        v_endif;

        v_if(exp >= 0 && exp < 23) {
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}
```

### 3.2 SFPI Instructions Used

| SFPI Construct | Purpose | Notes |
|---|---|---|
| `sfpi::dst_reg[0]` | Read/write destination register tile element | Standard SFPU tile access |
| `sfpi::dst_reg++` | Advance to next tile element | Iterates over ITERATIONS elements |
| `sfpi::exexp(x)` | Extract unbiased exponent from float | Returns `int(floor(log2(|x|)))` — the biased exponent minus 127 |
| `sfpi::reinterpret<vInt>(x)` | Bitwise reinterpret float as integer | No conversion, raw bits |
| `sfpi::reinterpret<vFloat>(xi)` | Bitwise reinterpret integer as float | No conversion, raw bits |
| `sfpi::vInt(-1) << shift` | Create bitmask for truncation | Shifts `0xFFFFFFFF` left by (23-exp) bits |
| `v_if / v_endif` | Predicated (lane-masked) conditional execution | SIMD predication for per-element branching |
| `vFloat`, `vInt`, `vUInt` | SFPU vector register types | 32-bit lanes across SFPU vector width |

### 3.3 Algorithm Walkthrough

The kernel computes `frac(x) = x - trunc(x)` using IEEE 754 bit manipulation:

**Case analysis based on unbiased exponent `e = exexp(x)`:**

1. **`e < 0` (|x| < 1.0)**: The entire value is fractional. `trunc(x) = 0`, so `frac(x) = x`.

2. **`0 <= e < 23` (mixed integer+fractional bits)**: The float32 mantissa has 23 bits. For exponent `e`, the lower `(23 - e)` mantissa bits represent the fractional part. A bitmask `0xFFFFFFFF << (23 - e)` zeros out those fractional bits, producing `trunc(x)`. Then `frac(x) = x - trunc(x)`.

3. **`e >= 23` (pure integer)**: All 23 mantissa bits represent integer precision. The value has no fractional part, so `frac(x) = 0`. This is the default path — `trunc_x` is initialized to `x`, so `x - trunc_x = 0`.

**Key design choice**: Truncation toward zero (not floor), matching PyTorch semantics. For negative values like `-3.7`, `trunc(-3.7) = -3.0`, so `frac(-3.7) = -0.7` (sign preserved). This differs from `x - floor(x)` which would give `0.3`.

### 3.4 Template Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `APPROXIMATION_MODE` | (caller-provided) | **Not used** in this kernel — exact algorithm only |
| `ITERATIONS` | 8 | Number of elements per invocation (matches tile row) |

`APPROXIMATION_MODE` is accepted as a template parameter for interface conformance with the `_llk_math_eltwise_unary_sfpu_params_` dispatch template, but the kernel ignores it — the bit-manipulation algorithm is inherently exact for the representable float32 precision.

### 3.5 No Runtime Parameters

This operation takes **no runtime parameters** (no `param0`, no `param1`). It is a pure pointwise unary operation with no configurable behavior.

## 4. Dispatch Chain (Host to Kernel)

### 4.1 Compile Defines

When `UnaryOpType::FRAC` is selected:

1. `get_macro_definition(FRAC)` returns `"SFPU_OP_FRAC_INCLUDE"` — set to `"1"` as a compile define
2. `get_op_init_and_func_default(FRAC, idst)` returns:
   - Init: `"frac_tile_init();"`
   - Func: `"frac_tile({idst});"`
3. These are assigned to `SFPU_OP_INIT_0` and `SFPU_OP_FUNC_0`
4. The op chain macro `SFPU_OP_CHAIN_0` expands to `SFPU_OP_INIT_0 SFPU_OP_FUNC_0`

### 4.2 Include Guard Flow

```
SFPU_OP_FRAC_INCLUDE=1
    -> sfpu_split_includes.h: #if SFPU_OP_FRAC_INCLUDE -> #include "api/compute/eltwise_unary/frac.h"
        -> frac.h: #include "llk_math_eltwise_unary_sfpu_frac.h"
            -> llk_..._frac.h: #include "ckernel_sfpu_frac.h"
                -> calculate_frac<APPROX, ITERATIONS>()
```

### 4.3 Compute Kernel Entry Point

The standard `eltwise_sfpu.cpp` compute kernel:
```
cb_wait_front(c_0, 1) -> copy_tile(c_0, 0, 0) -> SFPU_OP_CHAIN_0 -> tile_regs_commit() -> ...
```

`SFPU_OP_CHAIN_0` expands to:
```cpp
frac_tile_init(); frac_tile(0);
```

Which calls:
```cpp
// frac_tile_init():
llk_math_eltwise_unary_sfpu_frac_init<APPROX>()
    -> llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROX>()

// frac_tile(idst):
llk_math_eltwise_unary_sfpu_frac<APPROX>(idst)
    -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_frac<APPROX, 8>, idst, VectorMode::RC)
```

### 4.4 Approximation Mode

`get_op_approx_mode(FRAC)` hits the `default` case, returning `false`. The `APPROX` template parameter is always `false`.

## 5. Composite Fallback Path

There is also a host-side composite implementation at `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp`:

```cpp
Tensor frac(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::subtract(x, ttnn::trunc(x, output_mem_config), std::nullopt, output_mem_config);
}
```

This composes `subtract` and `trunc` operations at the tensor level. The SFPU kernel provides a fused single-pass implementation that avoids the overhead of two separate kernel launches.

## 6. Backward Operation

```cpp
std::vector<Tensor> frac_bw(const Tensor& grad, const Tensor& /*input*/, ...) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
```

The derivative of `frac(x) = x - trunc(x)` is `1` almost everywhere (except at integer points where it's undefined). The backward simply passes the gradient through unchanged.

## 7. Test Coverage

### Unit Test (`test_frac.py`)
- **Basic test**: Shapes `[1,1,32,32]`, `[1,1,320,384]`, `[1,3,320,384]` with bfloat16; PCC >= 0.999, rtol=1.6e-2, atol=1e-2
- **Negative inputs**: Verifies sign preservation and output in (-1, 1) range
- **Integer inputs**: Verifies frac returns 0 for integer-valued inputs (atol=1e-2)

### Backward Test (`test_backward_frac.py`)
- Nightly test, verifies gradient passthrough with PCC comparison

## 8. Key Implementation Patterns for Reference

### Pattern: Bit-Manipulation on SFPU
The frac kernel demonstrates several reusable SFPI patterns:

1. **Exponent extraction**: `sfpi::exexp(x)` to get the unbiased exponent for range-based case analysis
2. **Bit reinterpretation**: `sfpi::reinterpret<vInt>(vFloat)` and back — zero-cost type punning between float and int views of the same register
3. **Dynamic bitmask construction**: `vInt(-1) << vUInt(shift)` to create per-element masks based on the exponent
4. **Predicated multi-case logic**: Two `v_if/v_endif` blocks with a default initialization handle three distinct exponent ranges without `v_elseif`
5. **Default-and-override pattern**: Initialize `trunc_x = x` (handles the `e >= 23` case), then override in the two conditional blocks. This avoids a third `v_if` and keeps the code compact.

### Pattern: Non-Parameterized Unary Op Registration
Frac follows the simplest unary SFPU registration pattern:
- No parameters in `UnaryWithParam`
- No custom `get_op_approx_mode` entry
- Uses `get_op_init_and_func_default` (not `_parameterized`)
- Uses the generic `eltwise_sfpu.cpp` compute kernel (no custom kernel path)
- Has its own `SFPU_OP_FRAC_INCLUDE` split-include gate

## 9. Architecture Differences

The Wormhole B0 and Blackhole implementations are **byte-for-byte identical** across all layers:
- `ckernel_sfpu_frac.h` — identical
- `llk_math_eltwise_unary_sfpu_frac.h` — identical

This operation uses no architecture-specific SFPI intrinsics — all constructs (`exexp`, `reinterpret`, `v_if`, bitwise ops, float subtraction) are portable across both architectures.
