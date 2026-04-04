# swish (SiLU) Implementation Notes

## Overview
- **Operation**: swish / silu
- **Math definition**: x * sigmoid(x) = x / (1 + exp(-x))
- **UnaryOpType enum entry**: SILU (pre-existing)
- **Date**: 2026-04-04

## Key Design Decision

Unlike other SFPU operations (selu, cosh, cbrt, hardsigmoid, hardtanh) that required creating new SFPU kernel files from scratch, the silu/swish operation already has a complete SFPU kernel implementation in the upstream `tt_llk` third-party library. The implementation therefore focused on the **software stack integration** layer that was missing.

## What Was Already Present (Pre-existing)
1. SFPU kernel in `tt_llk` submodule (ckernel_sfpu_silu.h)
2. LLK dispatch header in `tt_llk` submodule (llk_math_eltwise_unary_sfpu_silu.h)
3. Include in `llk_math_unary_sfpu_api.h` (both wormhole_b0 and blackhole)
4. Compute API functions `silu_tile()` and `silu_tile_init()` in `compute_kernel_api.h`
5. `REGISTER_UNARY_OPERATION(silu, SILU)` in `unary.hpp`
6. `UnaryOpType::SILU` in `unary_op_types.hpp`
7. `unary_ng_op_utils.cpp` case for `UnaryOpType::SILU`

## What Was Implemented (New Changes)

### New Files

- `tests/ttnn/unit_tests/operations/eltwise/test_silu.py`

### Modified Files

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`

---

## Source Code Snippets

### New File: `tests/ttnn/unit_tests/operations/eltwise/test_silu.py`

```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_allclose,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_silu(device, input_shape, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    # Use a mix of positive and negative values
    torch_input = torch.randn(input_shape, dtype=torch_dtype) * 3.0

    torch_output = torch.nn.functional.silu(torch_input.float())
    expected = torch_output.to(torch_dtype)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.silu(tt_input)
    actual = ttnn.to_torch(tt_output).to(torch_dtype)

    assert_allclose(expected, actual, rtol=1.6e-2, atol=1e-2)
```

### Modified: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

Added `silu` to the SfpuType enum:

```cpp
enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardsigmoid,
    selu,
    hardtanh,
    silu,       // <-- ADDED
};
```

### Modified: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

Same change as wormhole_b0 (identical file).

### Modified: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Added SILU case to `get_op_init_and_func_default`:

```cpp
std::pair<std::string, std::string> get_op_init_and_func_default(
    UnaryOpType op_type, std::string idst, [[maybe_unused]] std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::IDENTITY: return {"identity_tile_init();", fmt::format("identity_tile({});", idst)};
        case UnaryOpType::DROPOUT: return {"dropout_tile_init();", fmt::format("dropout_tile({});", idst)};
        case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
        case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
        case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
        case UnaryOpType::SELU: return {"selu_tile_init();", fmt::format("selu_tile({});", idst)};
        case UnaryOpType::SILU: return {"silu_tile_init();", fmt::format("silu_tile({});", idst)};  // <-- ADDED
        default: TT_THROW("unexpected op type {}", op_type);
    };
}
```

Added SILU to `string_to_unary_with_param`:

```cpp
UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "cosh") {
        return UnaryWithParam(UnaryOpType::COSH);
    }
    if (name == "silu") {                              // <-- ADDED
        return UnaryWithParam(UnaryOpType::SILU);      // <-- ADDED
    }                                                  // <-- ADDED
    TT_THROW("Unknown unary op: {}", name);
}
```

### Modified: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

Added nanobind binding for silu:

```cpp
    bind_unary_operation<"silu", &ttnn::silu>(
        mod,
        R"doc(\text{silu}(x) = x \times \sigma(x) = \frac{x}{1 + e^{-x}})doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

### Modified: `ttnn/ttnn/operations/unary.py`

Added silu to the golden function map and the registration list:

```python
        name_to_golden_function = {
            "identity": torch.clone,
            "cbrt": torch_cbrt,
            "hardsigmoid": torch.nn.functional.hardsigmoid,
            "selu": lambda _x: torch.nn.functional.selu(_x.to(torch.float)),
            "silu": torch.nn.functional.silu,  # <-- ADDED
        }

TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.identity,
    ttnn.cbrt,
    ttnn.hardsigmoid,
    ttnn.selu,
    ttnn.silu,  # <-- ADDED
]
```

---

## Technical Notes

1. **No split includes needed**: silu uses the standard `compute_kernel_api.h` path (always included), not the conditional split-includes mechanism. Therefore, no `SFPU_OP_SILU_INCLUDE` macro or entry in `sfpu_split_includes.h` is needed.

2. **Approx mode**: silu uses `get_op_approx_mode` default (false). The upstream kernel handles approximation internally.

3. **DST_ACCUM_MODE**: The `silu_tile` function in `compute_kernel_api.h` passes `DST_ACCUM_MODE` as a template parameter to the LLK function, which differs from the split-include ops that use `SFPU_UNARY_NO_PARAM_KERNEL_FN`.

## Test Results

All 6 tests passed on first attempt:
- `test_silu[dtype=DataType.BFLOAT16-input_shape=[1, 1, 32, 32]]` -- PASS
- `test_silu[dtype=DataType.BFLOAT16-input_shape=[1, 1, 320, 384]]` -- PASS
- `test_silu[dtype=DataType.BFLOAT16-input_shape=[1, 3, 320, 384]]` -- PASS
- `test_silu[dtype=DataType.FLOAT32-input_shape=[1, 1, 32, 32]]` -- PASS
- `test_silu[dtype=DataType.FLOAT32-input_shape=[1, 1, 320, 384]]` -- PASS
- `test_silu[dtype=DataType.FLOAT32-input_shape=[1, 3, 320, 384]]` -- PASS

## Known Limitations
- None. The operation is a straightforward integration of an upstream kernel with no parameters.
