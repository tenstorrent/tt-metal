# SFPU Program Factories

Comprehensive map of program factories for operations that use the SFPU (Special Function Processing Unit) in TTNN.

---

## 1. Unary Eltwise

**Device operation**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.hpp`

```cpp
program_factory_t = std::variant<UnaryProgramFactory, UnarySubCoreGridProgramFactory, UnaryShardedProgramFactory>
```

| Factory | File |
|---|---|
| `UnaryProgramFactory` | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` |
| `UnaryProgramFactory` (header) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` |
| `UnaryShardedProgramFactory` | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.cpp` |
| `UnaryShardedProgramFactory` (header) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.hpp` |

**Selection logic** (`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp`):
1. Input is sharded -> `UnaryShardedProgramFactory`
2. `sub_core_grids` specified -> `UnarySubCoreGridProgramFactory`
3. Default -> `UnaryProgramFactory`

All unary SFPU ops (exp, sqrt, relu, etc.) share these factories. The op type is passed as a compile-time define to the compute kernel.

---

## 2. Binary Eltwise

### Legacy Binary

**Device operation**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.hpp`

```cpp
program_factory_t = std::variant<
    ElementWiseMultiCore,
    ElementWiseMultiCoreSfpu,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
    BroadcastHeightMultiCoreSharded,
    BroadcastHeightMultiCoreShardedOptimized>
```

| Factory | File |
|---|---|
| `ElementWiseMultiCore` (FPU) | `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp` |
| `ElementWiseMultiCoreSfpu` | `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` |

**SFPU compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

**Selection logic** (`ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`):
1. Scalar operand -> `BroadcastHeightAndWidthMultiCore`
2. Shapes match -> `is_binary_sfpu_op()` check -> `ElementWiseMultiCoreSfpu` or `ElementWiseMultiCore`
3. Shape mismatch -> appropriate broadcast factory based on which dimension differs

`is_binary_sfpu_op()` recognizes: ADD, SUB, MUL, DIV, EQ, NE, LOGICAL_AND/OR/XOR, SQUARED_DIFFERENCE, LOGADDEXP, LOGADDEXP2, LDEXP, BIAS_GELU, HYPOT, RSUB, GT, LT, GE, LE, GCD, LCM, LEFT_SHIFT, RIGHT_SHIFT, LOGICAL_RIGHT_SHIFT, BITWISE_XOR/OR/AND, MAXIMUM, MINIMUM, XLOGY, FMOD, POWER.

### Next-Gen Binary (binary_ng)

**Device operation**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`

```cpp
program_factory_t = std::variant<ProgramFactory>
```

| Factory | File |
|---|---|
| `ProgramFactory` | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` |
| SFPU config/utils | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` |

Single unified factory. Uses an `is_sfpu` flag in `operation_attributes_t` to select SFPU vs FPU kernel paths internally. Additional SFPU ops in binary_ng: DIV_FLOOR, DIV_TRUNC, REMAINDER, FMOD, QUANT, REQUANT, DEQUANT, MAXIMUM, MINIMUM, XLOGY, POWER, WHERE_TST, WHERE_TTS.

---

## 3. Ternary Eltwise

**Device operation**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp`

```cpp
program_factory_t = std::variant<TernaryProgramFactory>
```

| Factory | File |
|---|---|
| `TernaryProgramFactory` | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp` |

**Selection logic** (`ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.cpp`):
Always returns `TernaryProgramFactory{}` -- no branching. The ternary op type (`where`, `addcmul`, `addcdiv`, etc.) is a compile-time define to the compute kernel.

---

## 4. Specialized Binary SFPU Operations (GCD, LCM, etc.)

These have **no dedicated factories**. They are binary SFPU ops routed through the binary pipeline. This section explains the full dispatch mechanism.

### How `is_binary_sfpu_op()` Gates FPU vs SFPU

**Legacy path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:21`

The function checks the `BinaryOpType` and both input `DataType`s to decide if the SFPU path is needed:

| Ops | Dtype Constraint |
|---|---|
| ADD, SUB, MUL, EQ, NE, LOGICAL_AND/OR/XOR, SQUARED_DIFFERENCE | `a == b` and one of: FLOAT32, INT32, UINT32, UINT16 |
| LOGADDEXP, LOGADDEXP2, LDEXP, BIAS_GELU, HYPOT | Both FLOAT32 |
| DIV, RSUB, GT, LT, GE, LE | Both FLOAT32 or both INT32 |
| GCD, LCM, LEFT_SHIFT, RIGHT_SHIFT, LOGICAL_RIGHT_SHIFT | Both INT32 or both UINT32 |
| BITWISE_XOR, BITWISE_OR, BITWISE_AND | Both INT32, UINT32, or UINT16 |
| MAXIMUM, MINIMUM, XLOGY, FMOD, POWER | Always SFPU (any dtype) |

**Binary_ng path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`

Similar logic but supports additional SFPU ops: DIV_FLOOR, DIV_TRUNC, REMAINDER, QUANT, REQUANT, DEQUANT, WHERE_TST, WHERE_TTS.

### Dispatch Path (using GCD as example)

```
ttnn::gcd(a, b)
  -> ExecuteGCD::invoke()                    # ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp:756
    -> BinaryOperationSfpu<BinaryOpType::GCD>::invoke()
      -> detail::invoke_binary_ng()          # routes to binary_ng path
        -> BinaryNgDeviceOperation           # ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp
          -> select_program_factory()        # returns ProgramFactory{}
            -> ProgramFactory::create()      # ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp
              -> OpConfig(BinaryOpType::GCD, std::in_place_type<SfpuBinaryOp>)
                -> binary_op = SfpuBinaryOp::GCD
              -> op_config.as_defines(dtype)
                -> get_sfpu_init_fn(SfpuBinaryOp::GCD, dtype)
                  -> returns {"gcd_tile_init();", "gcd_tile"}
                -> defines["BINARY_SFPU_INIT"] = "gcd_tile_init();"
                -> defines["BINARY_SFPU_OP"]   = "gcd_tile"
              -> CreateKernel(sfpu_compute_kernel, defines)
```

### Compile-Time Op Selection in the SFPU Kernel

**Legacy SFPU kernel**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

The kernel is a generic template that gets specialized at compile time via preprocessor macros. The host-side factory generates defines like `BINARY_SFPU_OP`, `BINOP_INIT`, `ADD_INT_INIT`, etc., and these are passed to the kernel compilation.

The kernel's inner loop (simplified):

```cpp
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp0, i, i * 2);       // input A -> DST[even]
    copy_tile(cb_inp1, i, i * 2 + 1);   // input B -> DST[odd]

    // Exactly ONE of these macros is defined per compilation:
    #ifdef BINOP_INIT       // FP SFPU ops (add_binary, sub_binary, etc.)
        BINOP_INIT
    #endif
    #ifdef ADD_INT_INIT     // Integer add
        ADD_INT_INIT
    #endif
    // ... more init macros for specific op categories ...

    #ifdef BINARY_SFPU_OP   // Generic SFPU op (gcd_tile, lcm_tile, etc.)
        BINARY_SFPU_OP
    #endif
    #ifdef SFPU_OP_INIT_0   // Chained unary post-processing
        SFPU_OP_INIT_0
        SFPU_OP_FUNC_0
    #endif

    pack_tile(i * 2, cb_out0);
}
```

The kernel also supports **pre-processing** of inputs via `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` macros (e.g., applying unary activations to inputs before the binary op).

### OpConfig and `get_sfpu_init_fn` (binary_ng)

**Config struct**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp:45`
**Init function map**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp:369`

The `OpConfig` struct holds a `std::variant<FpuBinaryOp, SfpuBinaryOp>`. In binary_ng, the `as_defines()` method calls `get_sfpu_init_fn()` which returns `{init_string, op_string}` pairs for each SFPU op. These become the `BINARY_SFPU_INIT` and `BINARY_SFPU_OP` kernel defines.

Full `SfpuBinaryOp` enum (29 ops):
```cpp
enum class SfpuBinaryOp {
    ADD, SUB, MUL, DIV, DIV_FLOOR, DIV_TRUNC, REMAINDER, FMOD,
    POWER, RSUB, GCD, LCM,
    LEFT_SHIFT, RIGHT_SHIFT, LOGICAL_RIGHT_SHIFT,
    BITWISE_AND, BITWISE_OR, BITWISE_XOR,
    QUANT, REQUANT, DEQUANT,
    MAXIMUM, MINIMUM, XLOGY,
    LT, GT, GE, LE, HYPOT, WHERE, EQ
};
```

Complete init/op function mapping:

| SfpuBinaryOp | Init Function | Tile Function | Dtype Notes |
|---|---|---|---|
| ADD | `add_int_tile_init()` / `add_binary_tile_init()` | `add_int_tile<Format>` / `add_binary_tile` | INT32/UINT32/UINT16 -> int path |
| SUB | `sub_int_tile_init()` / `sub_binary_tile_init()` | `sub_int_tile<Format>` / `sub_binary_tile` | INT32/UINT32/UINT16 -> int path |
| MUL | `mul_int_tile_init<Format>()` / `mul_binary_tile_init()` | `mul_int_tile<Format>` / `mul_binary_tile` | INT32/UINT32/UINT16 -> int path |
| DIV | `div_int32_tile_init()` / `div_binary_tile_init()` | `div_int32_tile` / `div_binary_tile` | INT32 -> int path |
| DIV_FLOOR | `div_int32_floor_tile_init()` | `div_int32_floor_tile` | INT32 only |
| DIV_TRUNC | `div_int32_trunc_tile_init()` | `div_int32_trunc_tile` | INT32 only |
| REMAINDER | `remainder_int32_tile_init()` | `remainder_int32_tile` | INT32 only |
| FMOD | `fmod_int32_tile_init()` / `fmod_binary_tile_init()` | `fmod_int32_tile` / `fmod_binary_tile` | INT32 -> int path |
| POWER | `power_binary_tile_init()` | `power_binary_tile` | |
| RSUB | `rsub_int_tile_init()` / `rsub_binary_tile_init()` | `rsub_int_tile<Format>` / `rsub_binary_tile` | INT32/UINT32/UINT16 -> int path |
| GCD | `gcd_tile_init()` | `gcd_tile` | INT32 only |
| LCM | `lcm_tile_init()` | `lcm_tile` | INT32 only |
| LEFT_SHIFT | `binary_shift_tile_init()` | `binary_left_shift_tile<Format>` | |
| RIGHT_SHIFT | `binary_shift_tile_init()` | `binary_right_shift_tile<Format>` | |
| LOGICAL_RIGHT_SHIFT | `binary_shift_tile_init()` | `binary_logical_right_shift_tile<Format>` | |
| BITWISE_AND | `binary_bitwise_tile_init()` | `bitwise_and_binary_tile<Format>` | |
| BITWISE_OR | `binary_bitwise_tile_init()` | `bitwise_or_binary_tile<Format>` | |
| BITWISE_XOR | `binary_bitwise_tile_init()` | `bitwise_xor_binary_tile<Format>` | |
| MAXIMUM | `binary_max_int32_tile_init()` / `binary_max_tile_init()` | `binary_max_int32_tile` / `binary_max_tile` | INT32/UINT32 -> int path |
| MINIMUM | `binary_min_int32_tile_init()` / `binary_min_tile_init()` | `binary_min_int32_tile` / `binary_min_tile` | INT32/UINT32 -> int path |
| QUANT | `quant_tile_init(zero_point)` | `quant_tile` | |
| REQUANT | `requant_tile_init(zero_point)` | `requant_tile` | |
| DEQUANT | `dequant_tile_init(zero_point)` | `dequant_tile` | |
| XLOGY | `xlogy_binary_tile_init()` | `xlogy_binary_tile` | |
| LT | `lt_int32_tile_init()` | `lt_int32_tile` | |
| GT | `gt_int32_tile_init()` | `gt_int32_tile` | |
| GE | `ge_int32_tile_init()` | `ge_int32_tile` | |
| LE | `le_int32_tile_init()` | `le_int32_tile` | |
| EQ | `eq_binary_tile_init()` | `eq_binary_tile` | |
| HYPOT | (not shown in get_sfpu_init_fn) | | |
| WHERE | `where_tile_init()` | `where_tile<Format>` | Format varies by dtype |

### Key Source Files

| Purpose | File |
|---|---|
| Legacy `is_binary_sfpu_op()` | `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` |
| Legacy SFPU program factory | `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` |
| Legacy SFPU compute kernel | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| GCD/LCM composite dispatch | `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp` |
| GCD/LCM registration | `ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp` |
| binary_ng `OpConfig` struct | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` |
| binary_ng `get_sfpu_init_fn()` | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` |
| binary_ng unified factory | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` |
| binary_ng SFPU kernel (no bcast) | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| binary_ng SFPU kernel (bcast) | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` |
| binary_ng SFPU kernel (row/col) | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_sfpu_row_col_bcast.cpp` |
| SFPU GCD implementation | `tt_metal/third_party/tt_llk/*/llk_sfpu_gcd.h` (via `api/compute/gcd.h`) |
| SFPU LCM implementation | `tt_metal/third_party/tt_llk/*/llk_sfpu_lcm.h` (via `api/compute/lcm.h`) |

---

## 5. Normalization Operations

### Softmax

**Device operation**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.hpp`

```cpp
program_factory_t = std::variant<
    SoftmaxProgramFactoryGeneralWSmall,
    SoftmaxProgramFactoryGeneralWLarge,
    SoftmaxProgramFactoryGeneralHSmall,
    SoftmaxProgramFactoryGeneralHLarge,
    SoftmaxProgramFactoryGeneralCLarge,
    SoftmaxShardedProgramFactoryAttentionOptimized,
    SoftmaxProgramFactoryAttentionOptimized>
```

| Factory | File |
|---|---|
| `SoftmaxProgramFactoryGeneralWSmall` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp` |
| `SoftmaxProgramFactoryGeneralWLarge` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp` |
| `SoftmaxProgramFactoryGeneralHSmall` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_small.cpp` |
| `SoftmaxProgramFactoryGeneralHLarge` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_large.cpp` |
| `SoftmaxProgramFactoryGeneralCLarge` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_c_large.cpp` |
| `SoftmaxProgramFactoryAttentionOptimized` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_attention_optimized.cpp` |
| `SoftmaxShardedProgramFactoryAttentionOptimized` | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_attention_optimized_sharded.cpp` |
| General base | `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general.cpp` |

**SFPU compute kernels**:
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_large_tensor.cpp`

**Selection logic** (`ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp`):
- In-place / scale-mask variants -> `SoftmaxShardedProgramFactoryAttentionOptimized` (if sharded config) or `SoftmaxProgramFactoryAttentionOptimized`
- General softmax on rank-4, dim W -> same sharded/non-sharded attention selection
- Generic softmax -> dimension-based general factory (W-small/large, H-small/large, C-large)

### Batch Norm

**Device operation**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp`

```cpp
program_factory_t = std::variant<BatchNormFactory>
```

| Factory | File |
|---|---|
| `BatchNormFactory` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp` |

**SFPU compute kernel**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`
(Used when `fp32_dest_acc_en` is true)

### Running Statistics (Batch Norm)

**Device operation**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/running_statistics_device_operation.hpp`

```cpp
program_factory_t = std::variant<RunningStatisticsProgramFactory>
```

| Factory | File |
|---|---|
| `RunningStatisticsProgramFactory` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/running_statistics_program_factory.cpp` |

**SFPU compute kernel**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_sfpu_kernel.cpp`
(Used when `fp32_dest_acc_en` is true)

### Group Norm

**Device operation**: `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.hpp`

```cpp
program_factory_t = std::variant<GroupNormShardedProgramFactory, GroupNormNoMcastProgramFactory, GroupNormMcastProgramFactory>
```

| Factory | File |
|---|---|
| `GroupNormShardedProgramFactory` | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp` |
| `GroupNormNoMcastProgramFactory` | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_no_mcast_program_factory.cpp` |
| `GroupNormMcastProgramFactory` | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp` |
| Program utils | `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.cpp` |

**Selection logic** (`ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.cpp`):
Sharded input -> `GroupNormShardedProgramFactory`; otherwise mcast/no-mcast variants.

Uses SFPU operations internally (exp, reciprocal_sqrt) within compute kernels.

---

## 6. Reduction Operations

### Generic Reduce

**Device operation**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_device_operation.hpp`

```cpp
program_factory_t = std::variant<
    ReduceSingleCoreHwProgramFactory,
    ReduceMultiCoreHProgramFactory,
    ReduceMultiCoreWProgramFactory>
```

| Factory | File |
|---|---|
| `ReduceSingleCoreHwProgramFactory` | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_single_core_hw_program_factory.cpp` |
| `ReduceSingleCoreHwProgramFactory` (header) | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_single_core_hw_program_factory.hpp` |
| `ReduceMultiCoreHProgramFactory` | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp` |
| `ReduceMultiCoreHProgramFactory` (header) | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.hpp` |
| `ReduceMultiCoreWProgramFactory` | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp` |
| `ReduceMultiCoreWProgramFactory` (header) | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.hpp` |

**Selection logic** (`ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_device_operation.cpp`):
Based on `get_parallelization_strategy()`:
- `MULTI_CORE_H` -> `ReduceMultiCoreHProgramFactory`
- `MULTI_CORE_W` -> `ReduceMultiCoreWProgramFactory`
- `MULTI_CORE_HW` / `SINGLE_CORE_HW` -> `ReduceSingleCoreHwProgramFactory`

### Other Reduction Operations

| Operation | Factory File |
|---|---|
| topk (single core) | `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_single_core_program_factory.cpp` |
| topk (multi core) | `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp` |
| argmax (single core) | `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_single_core_program_factory.cpp` |
| argmax (multi core) | `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_multi_core_program_factory.cpp` |
| prod (all) | `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_all_program_factory.cpp` |
| prod (NC) | `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_nc_program_factory.cpp` |
| moe | `ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp` |
| sampling | `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp` |
| accumulation | `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp` |
| ema | `ttnn/cpp/ttnn/operations/reduction/accumulation/ema/device/ema_program_factory.cpp` |
| manual_seed | `ttnn/cpp/ttnn/operations/reduction/manual_seed/device/manual_seed_program_factory.cpp` |

---

## 7. Moreh Operations

Moreh operations have their own program factories. Those that use SFPU internally (via compute kernels with exp, reciprocal, etc.):

| Operation | Factory File |
|---|---|
| moreh_softmax | `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp` |
| moreh_softmax_backward | `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.cpp` |
| moreh_group_norm | `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp` |
| moreh_group_norm_backward (gamma/beta grad) | `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_factory.cpp` |
| moreh_group_norm_backward (input grad) | `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/moreh_group_norm_backward_input_grad_factory.cpp` |
| moreh_norm (H) | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_h_other.cpp` |
| moreh_norm (NC) | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_nc_other.cpp` |
| moreh_norm (W) | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp` |
| moreh_norm_backward | `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/moreh_norm_backward_program_factory.cpp` |
| moreh_clip_grad_norm (step 1) | `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_program_factory.cpp` |
| moreh_clip_grad_norm (step 2) | `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_program_factory.cpp` |
| moreh_clip_grad_norm (step 3) | `ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_program_factory.cpp` |

---

## Architectural Pattern

The common pattern across all categories:
- **Device operation class** defines `program_factory_t` as a `std::variant` of factory structs
- **`select_program_factory()`** inspects tensor properties (sharding, shapes, dtypes) and op attributes to pick the right variant
- **Each factory's `create()` method** builds the full `tt::tt_metal::Program` with reader/compute/writer kernels and circular buffers
- **SFPU op type** is passed as a compile-time define to shared compute kernels, not as separate factory implementations per op
- **`fp32_dest_acc_en`** flag in `ComputeConfig` is a common indicator of SFPU usage, enabling FP32 accumulation in the destination register

## Notes

- **LayerNorm / RMSNorm**: These operations do not appear under `ttnn/cpp/ttnn/operations/normalization/` in the current codebase. They may use descriptor-based program construction or reside in a different subsystem.
- **Data movement (untilize)**: Untilize factories use SFPU for FP32 paths (`fp32_dest_acc_en`), but the primary purpose is data layout transformation, not SFPU compute.
- **Matmul, Conv, Pool, Embedding**: These do not have explicit SFPU program factories; they use the FPU (matrix engine) as their primary compute path.
