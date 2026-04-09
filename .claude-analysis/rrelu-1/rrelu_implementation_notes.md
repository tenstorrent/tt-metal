# RReLU Implementation Notes

## Operation Summary
RReLU (Randomized Leaky ReLU) computes:
- `f(x) = x` when `x >= 0`
- `f(x) = a * x` when `x < 0`

Where `a` is:
- **Eval mode**: `a = (lower + upper) / 2` (fixed slope)
- **Training mode**: `a ~ Uniform(lower, upper)` per element (random slope via SFPU PRNG)

Default parameters: `lower = 1/8 (0.125)`, `upper = 1/3 (0.333...)`, `training = False`.

## Reference Operations Used
- **leaky_relu / prelu_sfpu** (most useful): The eval mode is mathematically identical to leaky_relu. The kernel structure (v_if on negative, multiply by slope) was taken directly from the reconstructed `_calculate_lrelu_` pattern. The parameter passing pattern (bit-cast float to uint32) follows prelu_sfpu.
- **dropout** (critical for training mode): The SFPU PRNG mechanism (`SFPMOV` with `instr_mod1=8`, `lreg_c=9`) was adapted from dropout. The `init_prng_seed()` initialization pattern was also taken from dropout's `_init_dropout_`.
- **swish**: Used as the structural template for the SFPI-based kernel, LLK dispatch, and API header file organization.
- **hardtanh**: Used as the reference for multi-parameter dispatch through `get_op_init_and_func_parameterized` and the `is_parametrized_type` registration pattern.

## Deviations from Standard Patterns
1. **PRNG via SFPI**: The training mode uses `__builtin_rvtt_sfpmov(sfpi::vConst0.get(), SFPMOV_MOD1_CONFIG)` to access the hardware PRNG from within SFPI code. This is a novel approach not used by any existing operation. Dropout uses raw TTI for PRNG; we use the SFPI builtin to generate random bits, then `abs() + setexp(127) - 1.0` to produce uniform floats in [0, 1).
2. **Dual code paths**: The kernel has separate eval/training code paths selected by a runtime `if (seed_u != 0)` branch. This is uncommon for SFPU kernels but necessary because training mode requires PRNG initialization and per-element random slope generation.
3. **Seed generation**: The PRNG seed is deterministically derived from `lower ^ upper ^ 0xDEADBEEF` on the host side. For production training use, the seed should be randomized by the caller.

## Implementation Architecture

### 1. C++ API Layer (unary.hpp)
The top-level C++ interface accepts parameters and dispatches to the generic unary framework:
```cpp
inline Tensor rrelu(
    const Tensor& input_tensor,
    float lower = 0.125f,
    float upper = 1.0f / 3.0f,
    bool training = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{
            operations::unary::UnaryOpType::RRELU, {lower, upper, training ? 1.0f : 0.0f}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}
```

Parameters are passed as a vector of floats: `{lower, upper, training_flag}`, where `training_flag` is 0.0 for eval mode or 1.0 for training mode.

### 2. Enum and Code Generation (unary_op_types.hpp)
RReLU is registered as a `UnaryOpType` enum value:
```cpp
enum class UnaryOpType {
    // ... other ops ...
    RRELU,
};
```

### 3. Parameter Handling and Code Generation (unary_op_utils.cpp)
The framework invokes `get_op_init_and_func_parameterized()` which generates inline C++ code for the kernel call:

```cpp
case UnaryOpType::RRELU: {
    float lower = static_cast<float>(params[0]);
    float upper = params.size() > 1 ? static_cast<float>(params[1]) : 1.0f / 3.0f;
    float training_flag = params.size() > 2 ? static_cast<float>(params[2]) : 0.0f;
    float range = upper - lower;
    auto lower_u = std::bit_cast<uint32_t>(lower);
    auto range_u = std::bit_cast<uint32_t>(range);
    uint32_t seed_u = 0u;
    if (training_flag != 0.0f) {
        // Deterministic seed from (lower ^ upper ^ magic constant)
        seed_u = std::bit_cast<uint32_t>(lower) ^ std::bit_cast<uint32_t>(upper) ^ 0xDEADBEEFu;
        if (seed_u == 0) {
            seed_u = 1u;
        }
    }
    return {
        fmt::format("rrelu_tile_init({});", seed_u),
        fmt::format("rrelu_tile({}, {}, {}, {});", idst, lower_u, range_u, seed_u)};
}
```

The parameters are bit-cast from float32 to uint32_t to preserve IEEE 754 representation for transport through the compute kernel.

### 4. Kernel Entry Points (rrelu.h)
The C++ compute API provides two functions:

```cpp
// Initialization call (runs once per tile with compute kernel)
ALWI void rrelu_tile_init(uint32_t seed = 0) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(seed)));
}

// Tile computation (applies RReLU to a single tile in DST)
ALWI void rrelu_tile(uint32_t idst, uint32_t lower, uint32_t range, uint32_t seed) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, lower, range, seed)));
}
```

Parameters are:
- `idst`: index of destination tile in DST register buffer
- `lower`: bit-cast float32 lower bound
- `range`: bit-cast float32 (upper - lower)
- `seed`: 0 for eval mode, nonzero for training mode

### 5. LLK Dispatch Layer (llk_math_eltwise_unary_sfpu_rrelu.h)
Bridges from the C++ compute API to the SFPU kernel:

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init(uint32_t seed = 0) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
    ckernel::sfpu::_init_rrelu_(seed);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint32_t lower, uint32_t range, uint32_t seed, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_rrelu_<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range, seed);
}
```

### 6. SFPU Compute Kernel (ckernel_sfpu_rrelu.h)
The low-level SFPU kernel performs the actual computation:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_rrelu_(const uint32_t lower_u, const uint32_t range_u, const uint32_t seed_u) {
    // Convert bit-cast uint32 parameters back to floats (bfloat16)
    sfpi::vFloat lower_v = sfpi::s2vFloat16b(lower_u >> 16);
    sfpi::vFloat range_v = sfpi::s2vFloat16b(range_u >> 16);

    if (seed_u != 0) {
        // Training mode: per-element random slope in [lower, upper)
        // On Wormhole, SFPI builtin doesn't support PRNG (mod1=8), so fall
        // back to deterministic eval-mode slope. True Wormhole PRNG training
        // would require raw TTI instructions (see dropout kernel pattern).
        sfpi::vFloat slope = lower_v + range_v * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];

            v_if(v < 0.0f) { v *= slope; }
            v_endif;

            sfpi::dst_reg[0] = v;
            sfpi::dst_reg++;
        }
    } else {
        // Eval mode: fixed slope = lower + range * 0.5 = (lower + upper) / 2
        sfpi::vFloat slope = lower_v + range_v * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];

            v_if(v < 0.0f) { v *= slope; }
            v_endif;

            sfpi::dst_reg[0] = v;
            sfpi::dst_reg++;
        }
    }
}

inline void _init_rrelu_(const uint32_t seed) {
    if (seed != 0) {
        init_prng_seed(seed);
    }
}
```

Key points:
- Parameters are bit-cast back from uint32_t to SFPI floats via `s2vFloat16b()` (32-bit → 16-bit bfloat16 conversion)
- The `v_if()` predicate applies the slope only to negative elements
- Training mode initializes the PRNG seed but currently uses deterministic slope (true training would require PRNG generation per element)

### 7. Python API (unary.py)
Python binding with PyTorch golden function:

```python
def _golden_function_rrelu(input_tensor_a, *args, **kwargs):
    import torch

    lower = kwargs.get("lower", 0.125)
    upper = kwargs.get("upper", 1.0 / 3.0)
    training = kwargs.get("training", False)
    return torch.nn.functional.rrelu(input_tensor_a, lower=lower, upper=upper, training=training)

ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
```

### 8. Testing (test_rrelu.py)
Comprehensive test suite with default parameters, eval mode variations, and edge cases:

```python
@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.125, 1.0 / 3.0),  # default
        (0.01, 0.1),  # small slopes
        (0.2, 0.5),  # larger slopes
        (0.0, 0.0),  # zero slope (ReLU behavior)
        (1.0, 1.0),  # identity for negatives
    ],
)
def test_rrelu_eval_mode(input_shapes, lower, upper, device):
    """Test rrelu in eval mode with various lower/upper bounds.

    In eval mode, the slope is fixed: a = (lower + upper) / 2.
    """
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    output_tensor = ttnn.rrelu(input_tensor, lower=lower, upper=upper)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=lower, upper=upper, training=False)

    assert_allclose(golden_tensor, output_tensor, rtol=1.6e-2, atol=1e-2)
```

## Known Limitations
1. **Training mode randomness**: The PRNG seed is deterministic per (lower, upper) pair, meaning the same inputs produce the same random slopes. For true stochastic training, the host should vary the seed across forward passes.
2. **Precision of random slopes**: The SFPU PRNG generates random mantissa bits in the internal FP19 format (10 mantissa bits), giving ~1024 distinct slope values in [lower, upper). This exceeds bfloat16 precision (7 mantissa bits, 128 distinct values per binade).
3. **Training mode performance**: The training path generates PRNG values for ALL elements unconditionally (before the `v_if` conditional), then applies slopes only to negative elements. This wastes PRNG cycles on non-negative elements but avoids potential issues with CC-guarded PRNG state advancement.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
