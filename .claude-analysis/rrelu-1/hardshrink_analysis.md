# SFPU Kernel Analysis: hardshrink

## 1. Operation Overview

**Math Definition**: `hardshrink(x, lambda) = x if |x| > lambda, else 0`

Equivalent formulation used in kernels: `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)`

**Parameters**: One float parameter `lambda` (default: 0.5, common range: [0.1, 1.0])

**PyTorch equivalent**: `torch.nn.functional.hardshrink(input, lambd=0.5)`

## 2. Architecture Classification

Hardshrink is a **custom-kernel unary operation** — it does **NOT** use the standard SFPU_OP_CHAIN dispatch path through `eltwise_sfpu.cpp`. Instead, it has its own dedicated compute kernel files that implement the math using a combination of FPU binary operations and SFPU comparison operations.

**Key distinction**: Unlike typical SFPU unary ops (e.g., `exp`, `relu`, `sin`) that define `{op}_tile_init()` / `{op}_tile()` functions and get dispatched through the `SFPU_OP_CHAIN_0` macro, hardshrink requires multi-pass computation with an intermediate temporary buffer, making it unsuitable for the single-tile-in/single-tile-out SFPU dispatch pattern.

## 3. Abstraction Layers

### Layer 1: UnaryOpType Enum
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:114`
- **Entry**: `HARDSHRINK` in the `UnaryOpType` enum

### Layer 2: C++ API Registration
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:166`
- **Registration**: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(hardshrink, HARDSHRINK)`
- This macro generates an inline function `hardshrink(input_tensor, parameter, ...)` that creates a `UnaryWithParam{UnaryOpType::HARDSHRINK, parameter}` and routes through `unary_impl`.

### Layer 3: Compute Kernel Path Resolution
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:167-171`
- `get_compute_kernel_path()` currently returns `"eltwise_sfpu.cpp"` for all ops (post-nuke state). In the original code, HARDSHRINK would have returned its custom kernel path (e.g., `"hardshrink_kernel.cpp"` or `"hardshrink_kernel_sfpu.cpp"`).

### Layer 4: SFPU Init/Func Dispatch
- Hardshrink is **NOT** present in `get_op_init_and_func_default()` or `get_op_init_and_func_parameterized()` (both in `unary_op_utils.cpp`).
- It is **NOT** in `is_parametrized_type()` (`unary_op_utils.hpp:44-50`), despite having a parameter.
- This is because hardshrink uses its own complete compute kernel rather than the SFPU_OP_CHAIN tile-function dispatch.

### Layer 5: Scalar Packing
- **File (legacy)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:129-131`
- **File (ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:45`
- Lambda parameter is packed as a scalar runtime arg via `pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())` and passed as the first runtime arg to the compute kernel.

### Layer 6: Temporary Circular Buffer
- **File (ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:54`
- `needs_tmp0_cb(UnaryOpType t)` returns `true` only for `HARDSHRINK`.
- Hardshrink is the **only** unary operation that requires a temporary CB (`c_1` / `CBIndex::c_1`).

### Layer 7: Python Bindings
- Forward: Exposed via `ttnn.hardshrink(input, lambd=scalar)` through the `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER` macro path.
- Backward: `ttnn.hardshrink_bw` registered in `unary_backward_nanobind.cpp:1043-1050`, implemented in `unary_backward.cpp:714-721`.
- Golden function: `torch.nn.functional.hardshrink(input, lambd=alpha)`

## 4. Compute Kernel Analysis

Hardshrink has **two** compute kernel variants:

### 4a. `hardshrink_kernel.cpp` — Optimized FPU Binary Variant
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`

Uses `binary_dest_reuse_tiles` API for efficient in-dest-register binary operations.

**Includes**:
```cpp
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"          // binary_dest_reuse_tiles
#include "api/compute/tile_move_copy.h"          // copy_tile_to_dst_init_short, copy_tile
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/fill.h"      // fill_tile
#include "api/compute/eltwise_unary/comp.h"      // ltz_tile, gtz_tile
```

**Algorithm (per tile)**:

**Pass 1** — Compute `a * 1(a + lambda < 0)`:
1. `fill_tile(0, lambda)` — Fill dest register 0 with lambda
2. `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_input, 0, 0)` — Compute `a + lambda` (reusing dest as srcA, reading input from srcB via cb)
3. `ltz_tile(0)` — Apply less-than-zero: `1(a + lambda < 0)`
4. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_input, 0, 0)` — Multiply by input: `a * 1(a + lambda < 0)`
5. `pack_tile(0, cb_tmp0)` — Store intermediate result to tmp CB

**Pass 2** — Compute `a * 1(a - lambda > 0)` and add:
1. `fill_tile(0, lambda)` — Fill dest register 0 with lambda again
2. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` — Compute `a - lambda` (input as srcA, dest as srcB)
3. `gtz_tile(0)` — Apply greater-than-zero: `1(a - lambda > 0)`
4. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_input, 0, 0)` — Multiply by input: `a * 1(a - lambda > 0)`
5. `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_tmp0, 0, 0)` — Add pass-1 result from tmp CB
6. `pack_tile(0, cb_output)` — Write final result

### 4b. `hardshrink_kernel_sfpu.cpp` — Explicit Copy Variant
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`

Uses explicit `copy_tile` + separate `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile` calls.

**Additional Include** (vs kernel.cpp):
```cpp
#include "api/compute/eltwise_binary_sfpu.h"  // add_binary_tile, sub_binary_tile, mul_binary_tile
```

**Algorithm (per tile)** — Same math, different API:

**Pass 1**:
1. `fill_tile(0, lambda)` — Lambda into dest[0]
2. `copy_tile(cb_input, 0, 1)` — Copy input tile into dest[1]
3. `add_binary_tile(0, 1, 0)` — dest[0] = dest[0] + dest[1] = lambda + a
4. `ltz_tile(0)` — dest[0] = 1(a + lambda < 0)
5. `mul_binary_tile(0, 1, 0)` — dest[0] = dest[0] * dest[1] = a * 1(a + lambda < 0)
6. Pack to cb_tmp0

**Pass 2**:
1. `fill_tile(1, lambda)` — Lambda into dest[1]
2. `copy_tile(cb_input, 0, 0)` — Input into dest[0]
3. `sub_binary_tile(0, 1, 0)` — dest[0] = dest[0] - dest[1] = a - lambda
4. `gtz_tile(0)` — dest[0] = 1(a - lambda > 0)
5. `copy_tile(cb_input, 0, 1)` — Reload input into dest[1]
6. `mul_binary_tile(0, 1, 0)` — dest[0] = a * 1(a - lambda > 0)
7. `copy_tile(cb_tmp0, 0, 1)` — Load pass-1 result into dest[1]
8. `add_binary_tile(0, 1, 0)` — Final sum
9. Pack to cb_output

## 5. Circular Buffer Layout

| CB Index | Name | Purpose | Size |
|----------|------|---------|------|
| `c_0` | `cb_input` | Input tiles from reader | `num_input_tiles * tile_size` |
| `c_1` | `cb_tmp0` | Intermediate result (pass 1 output) | `num_input_tiles * tile_size` (only allocated for HARDSHRINK) |
| `c_2` | `cb_output` | Output tiles for writer | `num_output_tiles * tile_size` |

The temporary buffer `cb_tmp0` is essential because the two-pass algorithm cannot compute both partial results simultaneously — pass 1's result must be stored and reloaded during pass 2.

## 6. Data Flow

```
Reader -> cb_input (c_0)
                |
                v
    [Pass 1: fill(lambda), add, ltz, mul]
                |
                v
           cb_tmp0 (c_1)  [intermediate: a * 1(a+lambda<0)]
                |
                v
    [Pass 2: fill(lambda), sub, gtz, mul, add(tmp0)]
                |
                v
         cb_output (c_2) -> Writer
```

## 7. Kernel Arguments

**Compile-time args**:
- `per_core_block_cnt` — Number of tile blocks per core
- `per_core_block_dim` — Number of tiles per block

**Runtime args**:
- `arg[0]`: `packed_scalar` — Lambda value bit-cast to uint32_t

The lambda value is unpacked in the kernel via:
```cpp
const uint32_t packed_scalar = get_arg_val<uint32_t>(0);
const auto lambd = reinterpret_cast<const float*>(&packed_scalar);
```

## 8. CB Synchronization Pattern

The kernel uses a **two-pass tile-at-a-time** pattern with explicit CB synchronization:

```
Per block:
  cb_reserve_back(cb_output, per_core_block_dim)   // Reserve output space for entire block
  Per tile:
    cb_wait_front(cb_input, 1)                      // Wait for 1 input tile
    cb_reserve_back(cb_tmp0, 1)                     // Reserve tmp space
    [Pass 1 compute -> pack to cb_tmp0]
    cb_push_back(cb_tmp0, 1)                        // Signal tmp ready
    cb_wait_front(cb_tmp0, 1)                       // Wait for tmp tile
    [Pass 2 compute -> pack to cb_output]
    cb_pop_front(cb_input, 1)                       // Release input tile
    cb_pop_front(cb_tmp0, 1)                        // Release tmp tile
  cb_push_back(cb_output, per_core_block_dim)       // Signal entire block done
```

Notable: Output reservation is done at the block level, but tmp0 reservation is per-tile. This is because the output writer operates on blocks while the tmp buffer is purely internal to the compute kernel.

## 9. Key Design Patterns for Reimplementation

1. **Custom kernel, not SFPU_OP_CHAIN**: Hardshrink cannot be a simple `{op}_tile()` function because it requires intermediate storage and multiple FPU operations per tile.

2. **Temporary CB requirement**: The `needs_tmp0_cb()` function in the program factory must return `true` for hardshrink to allocate `CBIndex::c_1`.

3. **Scalar parameter via runtime arg**: Lambda is packed on the host via `pack_scalar_runtime_arg()` and unpacked in the kernel via `reinterpret_cast`.

4. **Two kernel variants**: The `binary_dest_reuse_tiles` variant (`hardshrink_kernel.cpp`) is more efficient (fewer tile copies) than the `_sfpu` variant. The `_sfpu` variant uses explicit copies and SFPU binary ops.

5. **FPU + SFPU hybrid**: The kernel uses FPU for binary arithmetic (add, sub, mul) and SFPU for comparisons (ltz, gtz) and fill operations.

6. **Not in `is_parametrized_type()`**: Despite having a float parameter, hardshrink is not listed in `is_parametrized_type()` because its parameter bypasses the SFPU_OP_CHAIN mechanism entirely.

## 10. Files Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | UnaryOpType::HARDSHRINK enum |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | C++ API via REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp` | Optimized compute kernel (binary_dest_reuse_tiles) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp` | Alternate compute kernel (explicit copy + SFPU binary) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Legacy program factory (CB setup, scalar packing) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.cpp` | Sharded variant |
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp` | Next-gen program factory |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Utility functions (scalar packing) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | is_parametrized_type (hardshrink NOT listed) |
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` | NG utilities (scalar packing, compute path) |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` | Backward implementation |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp` | Backward Python bindings |
| `tests/ttnn/unit_tests/operations/eltwise/test_activation.py` | Unit test (test_scalarB_hardshrink) |
| `tests/sweep_framework/sweeps/eltwise/unary/hardshrink/hardshrink.py` | Sweep test |

## 11. Comparison: hardshrink vs Standard SFPU Unary

| Aspect | Standard SFPU Unary (e.g., exp, relu) | hardshrink |
|--------|----------------------------------------|------------|
| Compute kernel | `eltwise_sfpu.cpp` (shared) | `hardshrink_kernel.cpp` (custom) |
| Dispatch | `SFPU_OP_CHAIN_0` macro expansion | Direct kernel code |
| SFPU tile functions | `{op}_tile_init()` + `{op}_tile()` | None (uses FPU binary + SFPU comparison) |
| Circular buffers | c_0 (in), c_2 (out) | c_0 (in), c_1 (tmp), c_2 (out) |
| Parameter passing | Via SFPU_OP_CHAIN defines | Via runtime arg (packed scalar) |
| `is_parametrized_type()` | Yes (if parameterized) | No |
| Compute passes | Single pass | Two passes with intermediate store |
| `get_op_init_and_func()` | Has entry | No entry |
| `get_compute_kernel_path()` | Returns `"eltwise_sfpu.cpp"` | Would return custom kernel path |

## 12. SFPU Instructions Used

The hardshrink kernels use these SFPU/FPU primitives:
- **`fill_tile(dst_idx, value)`** — SFPU fill: writes a scalar constant into all elements of a dest register tile
- **`ltz_tile(dst_idx)`** — SFPU comparison: element-wise less-than-zero, produces 1.0 or 0.0
- **`gtz_tile(dst_idx)`** — SFPU comparison: element-wise greater-than-zero, produces 1.0 or 0.0
- **`add_binary_tile(a, b, dst)`** / **`binary_dest_reuse_tiles<ELWADD>`** — FPU addition
- **`sub_binary_tile(a, b, dst)`** / **`binary_dest_reuse_tiles<ELWSUB>`** — FPU subtraction
- **`mul_binary_tile(a, b, dst)`** / **`binary_dest_reuse_tiles<ELWMUL>`** — FPU multiplication
- **`copy_tile(cb, idx, dst)`** — Move tile from CB to dest register
- **`pack_tile(dst, cb)`** — Move tile from dest register to CB
