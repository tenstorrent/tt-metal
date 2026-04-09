# SFPU Kernel Analysis: tanhshrink

## 1. Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | tanhshrink |
| **UnaryOpType Enum** | `TANHSHRINK` (line 111 in `unary_op_types.hpp`) |
| **Math Definition** | `tanhshrink(x) = x - tanh(x)` |
| **Parameters** | None (parameterless unary op) |
| **PyTorch Equivalent** | `torch.nn.functional.tanhshrink(x)` |

## 2. Host-Side Registration & Dispatch Chain

### 2.1 C++ API Registration

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:158`

```cpp
REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)
```

This macro expands to an inline function `ttnn::tanhshrink(input_tensor, ...)` that calls `ttnn::detail::unary_impl()` with a `UnaryWithParam{UnaryOpType::TANHSHRINK}` (no parameters).

### 2.2 Dispatch Through `unary_op_utils.cpp`

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

#### `get_compute_kernel_path()` (line 167)

```cpp
std::string_view get_compute_kernel_path(UnaryOpType op_type, ...) {
    switch (op_type) {
        default: return "eltwise_sfpu.cpp";
    }
}
```

TANHSHRINK falls through to the default case, returning `"eltwise_sfpu.cpp"`. This means the standard SFPU dispatch path is used (via `SFPU_OP_CHAIN_0` macro expansion).

**CRITICAL ISSUE:** TANHSHRINK has NO entry in `get_op_init_and_func_default()` (line 46). The switch statement only handles FRAC, SWISH, ATANH, and SINH. Any call to `get_op_init_and_func()` for TANHSHRINK will hit the `default: TT_THROW("unexpected op type {}", op_type)` branch, causing a runtime error.

This means **tanhshrink cannot be dispatched through the standard `eltwise_sfpu.cpp` path** in its current state. The `SFPU_OP_CHAIN_0` define needed by `eltwise_sfpu.cpp` will never be generated for TANHSHRINK.

#### `get_macro_definition()` (line 18)

TANHSHRINK falls through to default, returning `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"`, the standard compute kernel API include. No special split-include macro is needed.

#### `is_parametrized_type()` (in `unary_op_utils.hpp:44`)

TANHSHRINK falls through to default → returns `false`. Correct; it has no parameters.

#### `get_op_approx_mode()` (line 73)

TANHSHRINK falls through to default → returns `false`. No approximate mode.

### 2.3 unary_ng Path

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

TANHSHRINK is **NOT** present in the unary_ng `get_op_init_and_func()` switch statement. The unary_ng path will also `TT_FATAL` if TANHSHRINK is dispatched through it.

### 2.4 Backward Operation

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:941`

```cpp
// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> tanhshrink_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}
```

The backward uses the derivative: `d/dx [x - tanh(x)] = 1 - sech^2(x) = tanh^2(x)`. This is implemented as `tanh(x)^2 * grad`, composing existing `ttnn::tanh` and `ttnn::square` operations.

## 3. Compute Kernel Analysis

### 3.1 Dedicated Kernel Files

TANHSHRINK has **two** dedicated compute kernel files, which is unusual — most unary SFPU operations use only the generic `eltwise_sfpu.cpp`.

#### Kernel Variant 1: FPU-based (`tanhshrink_kernel.cpp`)

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`

**Strategy:** Uses FPU `binary_dest_reuse_tiles` for the subtraction step.

```
Flow per tile:
1. cb_wait_front(cb_input, 1)         — wait for input tile
2. tile_regs_acquire()                — acquire DST register
3. copy_tile(cb_input, 0, 0)          — copy input tile to DST[0]
4. tanh_tile_init() / tanh_tile(0)    — compute tanh(x) in-place at DST[0]
5. binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)
   binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)
                                      — compute x - tanh(x) using FPU binary
                                        DEST_TO_SRCB moves DST[0] (tanh(x)) → SRCB
                                        unpacks cb_input tile → SRCA (x)
                                        computes SRCA - SRCB = x - tanh(x) → DST[0]
6. tile_regs_commit() / tile_regs_wait()
7. pack_tile(0, cb_output)            — pack result
8. tile_regs_release() / cb_pop_front(cb_input, 1)
```

**Key mechanism:** `EltwiseBinaryReuseDestType::DEST_TO_SRCB` — this avoids needing to store the original input separately. The tanh result in DST is moved to SRCB, and the original input is re-unpacked from the circular buffer into SRCA. The FPU then computes `SRCA - SRCB`.

**DST register usage:** 1 tile (DST[0] only).

#### Kernel Variant 2: SFPU-binary-based (`tanhshrink_sfpu_kernel.cpp`)

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`

**Strategy:** Uses SFPU binary subtraction (`sub_binary_tile`) instead of FPU binary.

```
Flow per tile:
1. cb_wait_front(cb_input, 1)          — wait for input tile
2. tile_regs_acquire()                 — acquire DST register
3. copy_tile(cb_input, 0, 1)           — copy input tile to DST[1]
4. tanh_tile_init() / tanh_tile(1)     — compute tanh(x) in-place at DST[1]
5. copy_tile_init(cb_input)
   copy_tile(cb_input, 0, 0)           — copy input tile again to DST[0]
6. sub_binary_tile_init()
   sub_binary_tile(0, 1, 0)            — compute DST[0] - DST[1] = x - tanh(x) → DST[0]
7. tile_regs_commit() / tile_regs_wait()
8. pack_tile(0, cb_output)             — pack result
9. tile_regs_release() / cb_pop_front(cb_input, 1)
```

**Key mechanism:** The input tile is copied to two DST registers (DST[0] and DST[1]). Tanh is applied to DST[1], and then SFPU binary subtract computes `DST[0] - DST[1]`.

**DST register usage:** 2 tiles (DST[0] and DST[1]).

### 3.2 Comparison of Kernel Variants

| Aspect | `tanhshrink_kernel.cpp` (FPU) | `tanhshrink_sfpu_kernel.cpp` (SFPU) |
|---|---|---|
| Subtraction method | FPU `binary_dest_reuse_tiles` | SFPU `sub_binary_tile` |
| DST registers used | 1 (DST[0]) | 2 (DST[0], DST[1]) |
| CB reads per tile | 2 (once for tanh, once for subtract via re-unpack) | 2 (two `copy_tile` calls from same CB slot) |
| Includes | `eltwise_binary.h` | `eltwise_binary_sfpu.h`, `copy_dest_values.h` |
| FPU unit usage | Yes (binary op) | No (all SFPU) |
| Precision | FPU path (higher fidelity) | SFPU path |

### 3.3 Circular Buffer Layout

Both kernels use the standard unary CB layout:

| CB | Index | Role |
|---|---|---|
| `cb_input` | `tt::CBIndex::c_0` | Input tiles from reader |
| `cb_output` | `tt::CBIndex::c_2` | Output tiles to writer |

No intermediate CBs are needed. Both kernels initialize with `init_sfpu(cb_input, cb_output)`.

### 3.4 Tile Processing Pattern

Both kernels follow the standard nested loop pattern:
- **Outer loop:** `per_core_block_cnt` blocks, each block reserves `per_core_block_dim` output tiles at the start and pushes them at the end.
- **Inner loop:** `per_core_block_dim` tiles per block, processing one tile at a time.

This is the standard **single-buffered** unary processing pattern.

## 4. Underlying SFPU/LLK Calls

### 4.1 `tanh_tile` (SFPU unary)

**Declared in:** `tt_metal/hw/inc/api/compute/compute_kernel_api.h:178`

```cpp
template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));
}
```

The tanh LLK implementation lives in the base firmware (not in the local `llk_sfpu/` directory). Both tanhshrink kernel variants call `tanh_tile` with default `fast_and_approx = false`, using the accurate tanh implementation.

### 4.2 `sub_binary_tile` (SFPU binary)

**Declared in:** `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:39`

```cpp
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::SUB>(idst0, idst1, odst)));
}
```

Operates entirely on DST registers. Takes two source tile indices and writes the result to an output tile index.

### 4.3 `binary_dest_reuse_tiles` (FPU binary)

**Declared in:** `tt_metal/hw/inc/api/compute/eltwise_binary.h:248`

```cpp
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK(...);
    MATH(...);
}
```

The `DEST_TO_SRCB` reuse mode moves the current DST tile into SRCB before unpacking the CB tile into SRCA. This allows a single DST register to hold the intermediate result while re-reading the original input.

## 5. Architecture Classification

tanhshrink is a **composite unary operation** that cannot be expressed as a single SFPU instruction. It requires:
1. An SFPU unary call (`tanh_tile`) for the tanh component
2. A subtraction step to compute `x - tanh(x)`

This places it in the "mixed routing" category — it needs more than just a simple SFPU op chain dispatch. The standard `eltwise_sfpu.cpp` kernel with `SFPU_OP_CHAIN_0` cannot express this operation because:
- `SFPU_OP_CHAIN_0` only supports sequential SFPU init+func pairs
- The subtraction `x - tanh(x)` requires access to both the original input `x` and the computed `tanh(x)` simultaneously
- This necessitates either FPU binary dest-reuse or SFPU binary operations with multiple DST registers

## 6. Current State Summary

### What exists:
- `UnaryOpType::TANHSHRINK` enum value
- `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` C++ API registration
- Two dedicated compute kernels: `tanhshrink_kernel.cpp` (FPU binary) and `tanhshrink_sfpu_kernel.cpp` (SFPU binary)
- Backward operation (`tanhshrink_bw`) implemented as a host-side composite
- Python tests and sweep configs
- Python nanobind for backward op

### What is missing/broken:
- **No `get_op_init_and_func` mapping** — dispatching TANHSHRINK through the standard unary path will `TT_THROW`
- **No `get_compute_kernel_path` routing** — the dedicated kernels exist but are not routed to; the default `eltwise_sfpu.cpp` is selected instead
- **No unary_ng support** — TANHSHRINK is absent from the unary_ng dispatch tables

### Implementation approach needed:
To re-implement tanhshrink, an implementor must either:
1. **Custom kernel path:** Add a `case UnaryOpType::TANHSHRINK:` in `get_compute_kernel_path()` pointing to one of the dedicated kernel files, and skip the `SFPU_OP_CHAIN_0` define generation (since the kernel handles everything internally)
2. **Composite approach:** Implement as `ttnn::subtract(input, ttnn::tanh(input))` at the host level (similar to the commented-out line in `unary_composite_op.cpp:501`)

The dedicated kernel approach (option 1) is preferred for performance since it avoids materializing an intermediate tensor.

## 7. Key Files Reference

| Layer | File | Role |
|---|---|---|
| Enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:111` | `TANHSHRINK` enum value |
| C++ API | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:158` | `ttnn::tanhshrink()` registration |
| Dispatch utils | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Kernel path, defines, init/func (MISSING for TANHSHRINK) |
| Program factory | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:140` | Builds compute kernel with path from dispatch |
| Compute kernel (FPU) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` | FPU binary dest-reuse approach |
| Compute kernel (SFPU) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp` | SFPU binary subtraction approach |
| Generic compute kernel | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Standard SFPU dispatch (not usable for TANHSHRINK) |
| Backward | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:941` | `tanhshrink_bw` host composite |
| Backward nanobind | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp:1364` | Python binding for backward |
| Key notes | `docs/sfpu_operations/key_notes/tanhshrink_key_notes.md` | Formula and PyTorch reference |
| Unit test | `tests/ttnn/unit_tests/operations/eltwise/test_activation.py:178` | Forward test |
| Backward test | `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_tanhshrink.py` | Backward test |
