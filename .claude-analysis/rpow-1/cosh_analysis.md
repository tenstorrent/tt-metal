# Reference Analysis: cosh (COSH)

## Overview
cosh(x) = (exp(x) + exp(-x)) / 2. Non-parameterized operation using exponential computation.

## Key Patterns

### string_to_unary_with_param
```cpp
UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "cosh") {
        return UnaryWithParam(UnaryOpType::COSH);
    }
    TT_THROW("Unknown unary op: {}", name);
}
```

### Golden Function Pattern
```cpp
def _golden_function_cosh(input_tensor_a, *args, **kwargs):
    import torch
    return torch.cosh(input_tensor_a)

ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)
```

### Nanobind Registration Pattern
```cpp
bind_unary_operation<"cosh", &ttnn::cosh>(
    mod,
    R"doc(\mathrm{{output\_tensor}}_i = \cosh(\mathrm{{input\_tensor}}_i))doc",
    "[supported range -9 to 9]",
    R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

## Relevance to rpow
Shows the nanobind registration and golden function patterns. rpow needs a parameterized variant of these (similar to hardtanh but with 1 param instead of 2).
