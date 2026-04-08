# SFPU Unary Eltwise Operations Catalog

Pre-nuke snapshot of all UnaryOpType operations that route through SFPU kernels.

**Source files**:
- Enum: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- Parametrized check: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`

## Operations by SFPU Macro Group

### SFPU_OP_EXP_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| EXP | Yes | fast_and_approximate_mode param |

### SFPU_OP_GELU_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| GELU | Yes | fast_and_approximate_mode param |

### SFPU_OP_RECIP_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RECIP | No | |

### SFPU_OP_SQRT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| SQRT | Yes | |

### SFPU_OP_RSQRT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RSQRT | Yes | fast_and_approximate_mode param |

### SFPU_OP_CBRT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| CBRT | No | |

### SFPU_OP_ERFINV_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ERFINV | No | |

### SFPU_OP_ERF_ERFC_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ERF | Yes | fast_and_approximate_mode param |
| ERFC | Yes | fast_and_approximate_mode param |

### SFPU_OP_ELU_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ELU | Yes | alpha param |

### SFPU_OP_RELU_FAMILY_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RELU | No | Family head |
| RELU6 | No | |
| RELU_MAX | Yes | upper_limit param |
| RELU_MIN | Yes | lower_limit param |
| LEAKY_RELU | Yes | negative_slope param |

### SFPU_OP_BINOP_WITH_SCALAR_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ADD_UNARY_SFPU | Yes | scalar param |
| SUB_UNARY_SFPU | Yes | scalar param |
| MUL_UNARY_SFPU | Yes | scalar param |
| DIV_UNARY_SFPU | Yes | scalar param |

### SFPU_OP_ROUND_FAMILY_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| FLOOR | No | |
| CEIL | No | |
| TRUNC | No | |
| FRAC | No | |
| ROUND | Yes | decimals param |

### SFPU_OP_RSUB_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RSUB | Yes | scalar param |

### SFPU_OP_ISINF_ISNAN_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ISINF | No | |
| ISNAN | No | |
| ISNEGINF | No | |
| ISPOSINF | No | |
| ISFINITE | No | |

### SFPU_OP_LOGICAL_NOT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| LOGICAL_NOT_UNARY | No | |

### SFPU_OP_I0_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| I0 | No | Modified Bessel function, first kind, order 0 |

### SFPU_OP_I1_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| I1 | No | Modified Bessel function, first kind, order 1 |

### SFPU_OP_TRIG_FAMILY_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ACOSH | No | |
| COS | No | |
| COSH | No | |
| SINH | No | |
| SIN | No | |
| ASINH | No | |
| TAN | No | |
| ATANH | No | |

### SFPU_OP_NEG_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| NEG | No | |

### SFPU_OP_SOFTPLUS_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| SOFTPLUS | Yes | beta, threshold params |

### SFPU_OP_XIELU_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| XIELU | Yes | Tenstorrent custom activation |

### SFPU_OP_LOGSIGMOID_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| LOGSIGMOID | No | Also has custom kernel path |

### SFPU_OP_SELU_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| SELU | Yes | alpha, scale params |

### SFPU_OP_PRELU_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| PRELU_SFPU | Yes | weight param |

### SFPU_OP_TYPECAST_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| TYPECAST | Yes | target dtype param |

### SFPU_OP_BITWISE_XOR_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| BITWISE_XOR | Yes | scalar param |

### SFPU_OP_BITWISE_NOT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| BITWISE_NOT | No | |

### SFPU_OP_BITWISE_AND_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| BITWISE_AND | Yes | scalar param |

### SFPU_OP_BITWISE_OR_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| BITWISE_OR | Yes | scalar param |

### SFPU_OP_RIGHT_SHIFT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RIGHT_SHIFT | Yes | shift amount param |

### SFPU_OP_LEFT_SHIFT_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| LEFT_SHIFT | Yes | shift amount param |

### SFPU_OP_REMAINDER_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| REMAINDER | Yes | divisor param |

### SFPU_OP_FMOD_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| FMOD | Yes | divisor param |

### SFPU_OP_FILL_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| FILL | Yes | fill value param |

### SFPU_OP_LOG1P_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| LOG1P | Yes | |

### SFPU_OP_UNARY_COMP_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| UNARY_NE | Yes | scalar param |
| UNARY_EQ | Yes | scalar param |
| UNARY_GT | Yes | scalar param |
| UNARY_LT | Yes | scalar param |
| UNARY_GE | Yes | scalar param |
| UNARY_LE | Yes | scalar param |
| GTZ | No | |
| LTZ | No | |
| EQZ | No | |
| LEZ | No | |
| GEZ | No | |
| NEZ | No | |

### SFPU_OP_WHERE_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| WHERE_TSS | Yes | Also has custom kernel path |

### SFPU_OP_CLAMP_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| CLAMP_TSS | Yes | min, max params |

### SFPU_OP_ACTIVATIONS_INCLUDE (Family)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| SOFTSHRINK | Yes | lambda param |
| SOFTSIGN | No | |
| HARDSIGMOID | No | |
| CELU | Yes | alpha param |

### SFPU_OP_THRESHOLD_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| THRESHOLD | Yes | threshold, value params |

### SFPU_OP_HARDTANH_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| HARDTANH | Yes | min_val, max_val params |

### SFPU_OP_RPOW_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| RPOW | Yes | exponent param |

### SFPU_OP_HARDMISH_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| HARDMISH | No | Tenstorrent custom activation |

### SFPU_OP_LGAMMA_INCLUDE
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| LGAMMA | No | Data-type dependent kernel path |

### SFPU_OP_COMPUTE_KERNEL_API_INCLUDE (Default)
| Operation | Parametrized | Notes |
|-----------|-------------|-------|
| ABS | No | |
| ABS_INT32 | No | |
| SIGN | No | |
| SQUARE | No | |
| SIGMOID | Yes | fast_and_approximate_mode param |
| LOG | Yes | |
| LOG2 | Yes | |
| LOG10 | Yes | |
| POWER | Yes | exponent param |
| POWER_ITERATIVE | Yes | exponent param |
| DROPOUT | No | Internal infrastructure |
| MAXIMUM | Yes | scalar param |
| MINIMUM | Yes | scalar param |
| ALT_COMPLEX_ROTATE90 | No | |
| BITCAST | Yes | target dtype param |
| ASIN | No | |
| ACOS | No | |
| ATAN | No | |
| TANH | Yes | |
| EXP2 | No | |
| EXPM1 | No | |
| SILU | No | |
| RDIV | Yes | scalar param |
| SIGNBIT | No | |
| ZERO_POINT | No | Internal quantization |
| TILED_PROD | No | Internal reduction |
| HEAVISIDE | Yes | value param |

## Excluded from Nuke (Non-SFPU)

| Operation | Reason |
|-----------|--------|
| MISH | Custom kernel `mish_kernel.cpp` |
| IDENTITY | Custom kernel `eltwise_identity_kernel.cpp` |
| LOGIT | Custom kernel `logit_kernel.cpp` |

## Mixed Routing (Partial Nuke)

| Operation | SFPU Macro | Custom Kernel to Preserve |
|-----------|-----------|--------------------------|
| HARDSHRINK | SFPU_OP_ACTIVATIONS_INCLUDE (via default) | `hardshrink_kernel.cpp` |
| HARDSWISH | Default | `hardswish_kernel.cpp` |
| TANHSHRINK | Default | `tanhshrink_kernel.cpp` |
| LOGSIGMOID | SFPU_OP_LOGSIGMOID_INCLUDE | `logsigmoid_kernel.cpp` |
| WHERE_TSS | SFPU_OP_WHERE_INCLUDE | `where_tss_kernel.cpp` |
| LGAMMA | SFPU_OP_LGAMMA_INCLUDE | `lgamma_kernel.cpp`, `lgamma_fast_kernel.cpp` |

## Summary

- **Total SFPU operations**: 109
- **Operations with explicit SFPU macro**: 83
- **Operations using default SFPU macro**: 26
- **Excluded (non-SFPU)**: 3 (MISH, IDENTITY, LOGIT)
- **Excluded (infrastructure)**: 2 (DROPOUT, ZERO_POINT)
- **Mixed routing (partial nuke)**: 6
- **New operation to add**: RReLU
