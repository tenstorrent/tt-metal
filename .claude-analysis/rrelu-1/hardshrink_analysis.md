# SFPU Analysis: hardshrink

## 1. Operation Overview

**Math Definition**: `hardshrink(x, lambda) = x if |x| > lambda, 0 otherwise`

Equivalent decomposition used in the kernel: `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)`

**Parameters**: One float parameter `lambda` (default = 0.5)

**UnaryOpType Enum**: `UnaryOpType::HARDSHRINK` (line 114 in `unary_op_types.hpp`)

**Category**: Mixed-routing activation. Unlike pure SFPU ops that use `SFPU_OP_CHAIN_0` macro dispatch through `eltwise_sfpu.cpp`, hardshrink has its own dedicated compute kernels that perform the operation using a combination of FPU binary ops and SFPU comparison ops.

## 2. Abstraction Layer Map

### Layer 1: UnaryOpType Enum
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:114`
- **Value**: `HARDSHRINK`

### Layer 2: C++ API Registration
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:166`
- **Macro**: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(hardshrink, HARDSHRINK)`
- This creates `ttnn::hardshrink(input_tensor, float parameter, ...)` which calls `unary_impl()` with `UnaryWithParam{UnaryOpType::HARDSHRINK, parameter}`.

### Layer 3: Utils (op_init, op_func, kernel path, approx mode)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `get_compute_kernel_path(HARDSHRINK)` => returns default `"eltwise_sfpu.cpp"` (line 167-171)
- `get_op_approx_mode(HARDSHRINK)` => returns `false` (default case, line 73-77)
- `is_parametrized_type(HARDSHRINK)` => returns `false` (not listed in switch, line 44-51)
- `get_macro_definition(HARDSHRINK)` => returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (default case, line 18-26)

**IMPORTANT NOTE**: Despite `get_compute_kernel_path` returning `"eltwise_sfpu.cpp"`, this is the **default path used by the SFPU_OP_CHAIN dispatch**. Hardshrink does NOT use the standard SFPU_OP_CHAIN mechanism. Instead, it has dedicated custom compute kernels (`hardshrink_kernel.cpp` and `hardshrink_kernel_sfpu.cpp`) that implement the algorithm directly. These custom kernels are selected at a higher level (the program factory must override the path, or they exist alongside the eltwise_sfpu.cpp path for different code paths in the ng vs old factory).

### Layer 4: Program Factory - Scalar Packing
- **File (old)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:129-131`
- **File (ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:45`
- Scalar packing: `packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())`
- The lambda parameter is packed as a 32-bit runtime arg using bit-casting (float -> uint32_t).

### Layer 5: Program Factory - Circular Buffer Allocation
- **File (old)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:70-75`
- **File (ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:54`
- Hardshrink requires an **extra temporary circular buffer** `cb_tmp0` at `CBIndex::c_1`:
  ```cpp
  if (ops_chain[0].type() == UnaryOpType::HARDSHRINK) {
      CircularBufferConfig cb_tmp0_config = CircularBufferConfig(
          num_input_tiles * input_cb_page_size, {{tmp0_cb_index, cb_data_format}})
          .set_page_size(tmp0_cb_index, input_cb_page_size);
      CreateCircularBuffer(program, all_cores, cb_tmp0_config);
  }
  ```
- This is a distinguishing feature: most unary SFPU ops only need `c_0` (input) and `c_2` (output).

### Layer 6: Python Nanobind
- **Forward**: The nanobind registration for the forward path is currently missing from `unary_nanobind.cpp`. The C++ function exists via the `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER` macro but has no explicit Python binding in the current codebase state (likely nuked).
- **Backward**: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp:1043-1050`
  - Registered as `bind_unary_backward_float_with_default<"hardshrink_bw">` with `lambd` param, default 0.5.

### Layer 7: Python Golden Function
- **Forward**: Referenced via `ttnn.get_golden_function(ttnn.hardshrink)` in tests.
- **Backward**: `ttnn/ttnn/operations/unary_backward.py:162-167` — uses `torch.nn.functional.hardshrink`.

## 3. Circular Buffer Layout

| CB Index | Name | Role | Size | Notes |
|----------|------|------|------|-------|
| `c_0` | `cb_input` / `cb_src0` | Input tiles from reader | `num_input_tiles * tile_size` | Standard input CB |
| `c_1` | `cb_tmp0` | Intermediate results | `num_input_tiles * tile_size` | **Hardshrink-specific**: stores partial result of first half of computation |
| `c_2` | `cb_output` | Output tiles to writer | `num_output_tiles * tile_size` | Standard output CB |

The temporary CB `c_1` is needed because the hardshrink formula `a*1(a+lambda<0) + a*1(a-lambda>0)` computes two terms that must be added. The first term (`a*1(a+lambda<0)`) is computed, packed to `c_1`, then combined with the second term.

## 4. Compute Kernel Architecture

Hardshrink is a **mixed-routing** operation with **two compute kernel variants**:

### Variant 1: `hardshrink_kernel.cpp` (FPU binary_dest_reuse path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
- Uses `binary_dest_reuse_tiles` API from `eltwise_binary.h`
- This variant uses the FPU (matrix unit) for binary add/sub/mul operations with dest register reuse.

**Algorithm (Phase 1 — first term `a*1(a+lambda<0)`):**
1. `fill_tile(0, lambda)` — Fill dest tile 0 with lambda value
2. `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_input)` — Compute `a + lambda` (dest + src_a)
3. `ltz_tile(0)` — SFPU comparison: `1(result < 0)` on dest tile 0
4. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_input)` — Multiply indicator by `a`
5. Pack result to `cb_tmp0`

**Algorithm (Phase 2 — second term `a*1(a-lambda>0)` + combine):**
1. `fill_tile(0, lambda)` — Fill dest tile 0 with lambda
2. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input)` — Compute `a - lambda` (src_b - dest = a - lambda)
3. `gtz_tile(0)` — SFPU comparison: `1(result > 0)` on dest tile 0
4. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_input)` — Multiply indicator by `a`
5. `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(cb_tmp0)` — Add term from Phase 1
6. Pack result to `cb_output`

### Variant 2: `hardshrink_kernel_sfpu.cpp` (SFPU binary path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
- Uses `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile` from `eltwise_binary_sfpu.h`
- This variant uses the SFPU (vector unit) for binary operations.

**Algorithm (Phase 1 — first term):**
1. `fill_tile(0, lambda)` — Fill dest tile 0 with lambda
2. `copy_tile(cb_input, 0, 1)` — Copy input to dest tile 1
3. `add_binary_tile(0, 1, 0)` — `result[0] = lambda + a`
4. `ltz_tile(0)` — SFPU: `1(a+lambda < 0)` on dest tile 0
5. `mul_binary_tile(0, 1, 0)` — `result[0] = indicator * a`
6. Pack dest tile 0 to `cb_tmp0`

**Algorithm (Phase 2 — second term + combine):**
1. `fill_tile(1, lambda)` — Fill dest tile 1 with lambda
2. `copy_tile(cb_input, 0, 0)` — Copy input `a` to dest tile 0
3. `sub_binary_tile(0, 1, 0)` — `result[0] = a - lambda`
4. `gtz_tile(0)` — SFPU: `1(a-lambda > 0)`
5. `copy_tile(cb_input, 0, 1)` — Copy input `a` to dest tile 1 again
6. `mul_binary_tile(0, 1, 0)` — `result[0] = indicator * a`
7. `copy_tile(cb_tmp0, 0, 1)` — Copy Phase 1 result to dest tile 1
8. `add_binary_tile(0, 1, 0)` — `result[0] = term2 + term1`
9. Pack dest tile 0 to `cb_output`

## 5. SFPU Instructions Used

Hardshrink does **not** have its own dedicated SFPU kernel function. Instead, it uses existing SFPU primitives:

| SFPU API | Header | LLK Call | Purpose |
|----------|--------|----------|---------|
| `ltz_tile(dst)` | `eltwise_unary/comp.h` (nuked) | Comparison: less-than-zero | Generates indicator `1(x < 0)` |
| `gtz_tile(dst)` | `eltwise_unary/comp.h` (nuked) | Comparison: greater-than-zero | Generates indicator `1(x > 0)` |
| `fill_tile(dst, val)` | `eltwise_unary/fill.h` (nuked) | Fill dest tile with scalar | Loads lambda parameter into dest register |

For the SFPU variant (`hardshrink_kernel_sfpu.cpp`), it additionally uses:
| API | Header | LLK Call | Purpose |
|-----|--------|----------|---------|
| `add_binary_tile(a, b, o)` | `eltwise_binary_sfpu.h` | `llk_math_eltwise_binary_sfpu_binop<ADD>` | Element-wise add in SFPU |
| `sub_binary_tile(a, b, o)` | `eltwise_binary_sfpu.h` | `llk_math_eltwise_binary_sfpu_binop<SUB>` | Element-wise subtract in SFPU |
| `mul_binary_tile(a, b, o)` | `eltwise_binary_sfpu.h` | `llk_math_eltwise_binary_sfpu_binop_mul<MUL>` | Element-wise multiply in SFPU |

For the FPU variant (`hardshrink_kernel.cpp`), it uses:
| API | Header | LLK Call | Purpose |
|-----|--------|----------|---------|
| `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` | `eltwise_binary.h` | `llk_math_eltwise_binary<ELWADD>` | FPU element-wise add with dest reuse |
| `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` | `eltwise_binary.h` | `llk_math_eltwise_binary<ELWSUB>` | FPU element-wise sub with dest reuse |
| `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` | `eltwise_binary.h` | `llk_math_eltwise_binary<ELWMUL>` | FPU element-wise mul with dest reuse |

## 6. Compile-Time Arguments

| Index | Name | Source |
|-------|------|--------|
| 0 | `per_core_block_cnt` | Number of tile blocks per core |
| 1 | `per_core_block_dim` | Tiles per block (typically 1 for interleaved) |

## 7. Runtime Arguments

| Index | Name | Source |
|-------|------|--------|
| 0 | `packed_scalar` | Lambda parameter, bit-cast float->uint32_t via `pack_scalar_runtime_arg()` |

The lambda value is retrieved in the kernel via:
```cpp
const uint32_t packed_scalar = get_arg_val<uint32_t>(0);
const auto lambd = reinterpret_cast<const float*>(&packed_scalar);
```

## 8. Data Flow Summary

```
Reader Kernel (NoC0)
    |
    v
CB c_0 (input tiles)
    |
    v
Compute Kernel (hardshrink_kernel.cpp or hardshrink_kernel_sfpu.cpp)
    |
    +--- Phase 1: fill(lambda) -> add(a,lambda) -> ltz() -> mul(indicator,a) --> pack to CB c_1
    |
    +--- Phase 2: fill(lambda) -> sub(a,lambda) -> gtz() -> mul(indicator,a) -> add(term1) --> pack to CB c_2
    |
    v
CB c_2 (output tiles)
    |
    v
Writer Kernel (NoC1)
```

## 9. Key Design Patterns

### Mixed-Routing Pattern
Hardshrink is NOT a standard SFPU-chain operation. It does not use the `SFPU_OP_CHAIN_0` dispatch mechanism from `eltwise_sfpu.cpp`. Instead:
- It has dedicated compute kernels in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/`
- The program factory still uses `get_compute_kernel_path()` which returns `"eltwise_sfpu.cpp"` by default, but this is for the ng path; the old path constructs the path with `compute_root + path`
- The actual kernel selection mechanism for the dedicated kernel files vs eltwise_sfpu.cpp depends on which program factory is active

### Two-Phase Computation with Temporary CB
The algorithm splits `x * 1(|x| > lambda)` into two additive indicator terms to avoid needing `abs()`:
- Term 1: `x * 1(x + lambda < 0)` — handles the `x < -lambda` case
- Term 2: `x * 1(x - lambda > 0)` — handles the `x > lambda` case

This requires packing the Phase 1 result to `cb_tmp0 (c_1)` and reading it back in Phase 2.

### Dest Register Reuse
The FPU variant uses `binary_dest_reuse_tiles` with `DEST_TO_SRCA` or `DEST_TO_SRCB` modes, which avoid extra unpacking by treating the dest register content as one of the binary operands.

### Tile-by-Tile Processing
Unlike the standard SFPU chain (which can process an entire block before pushing to output), hardshrink processes one tile at a time within the inner loop:
- `cb_wait_front(cb_input, 1)` / `cb_pop_front(cb_input, 1)` — consume one tile
- `cb_reserve_back(cb_tmp0, 1)` / `cb_push_back(cb_tmp0, 1)` — produce/consume one intermediate tile
- Output is accumulated with `cb_reserve_back(cb_output, per_core_block_dim)` at block level

### SFPU vs FPU Binary Operations
The two kernel variants offer different execution paths:
- **FPU variant** (`hardshrink_kernel.cpp`): Uses matrix-unit binary ops with dest reuse. More register-efficient (fewer copy_tile calls) but uses the FPU pipeline.
- **SFPU variant** (`hardshrink_kernel_sfpu.cpp`): Uses SFPU binary ops with explicit tile copies to dest. Requires more copy_tile instructions but keeps everything on the SFPU pipeline.

## 10. Backward Operation

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:714-721`
- Implementation: Computes forward hardshrink, then uses `where(eqz(hardshrink_result), 0.0, grad)` — gradient is zero where hardshrink output is zero, otherwise passes through grad unchanged.
- This is a composite operation (calls `ttnn::hardshrink`, `ttnn::eqz`, `where`).

## 11. Test Coverage

| Test File | Type | What it Tests |
|-----------|------|---------------|
| `tests/ttnn/unit_tests/operations/eltwise/test_activation.py:338-350` | Unit | Forward with lambda=0.5 and 1.0, shape [64,128] |
| `tests/sweep_framework/sweeps/eltwise/unary/hardshrink/hardshrink.py` | Sweep | Multiple shapes, dtypes, memory configs |
| `tests/sweep_framework/sweeps/eltwise/unary/hardshrink/hardshrink_sharded.py` | Sweep | Sharded memory configurations |
| `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_hardshrink.py` | Nightly | Backward with various lambda values |
| `tests/sweep_framework/sweeps/eltwise/unary_backward/hardshrink_bw.py` | Sweep | Backward sweep tests |

## 12. File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:114` | Enum definition |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:166` | C++ API registration macro |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | Utils declarations |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:167-171` | Kernel path (default eltwise_sfpu.cpp) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:70-75,129-131` | Old program factory (CB + scalar packing) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.cpp:91,155` | Sharded program factory |
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:45,54` | NG program factory |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp` | Compute kernel (FPU binary_dest_reuse variant) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp` | Compute kernel (SFPU binary variant) |
| `tt_metal/hw/inc/api/compute/eltwise_binary.h` | FPU binary API (binary_dest_reuse_tiles) |
| `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` | SFPU binary API (add/sub/mul_binary_tile) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` | init_sfpu() |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:714-721` | Backward implementation |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp:1043-1050` | Backward nanobind |
| `ttnn/ttnn/operations/unary_backward.py:162-167` | Backward golden function |
| `docs/sfpu_operations/key_notes/hardshrink_key_notes.md` | Key notes |

## 13. Implementation Insights for New Operations

### What Makes Hardshrink Unique
1. **No dedicated SFPU kernel function**: Unlike operations like `sigmoid_tile()` or `exp_tile()`, hardshrink has no single SFPU instruction. It is composed from primitive SFPU and FPU operations.
2. **Temporary CB requirement**: The two-phase algorithm requires `cb_tmp0 (c_1)`, which must be explicitly allocated in the program factory with a conditional check.
3. **Runtime scalar parameter**: Lambda is passed as a runtime arg (not compile-time), packed as float->uint32_t bit-cast.
4. **Two kernel variants**: Both FPU and SFPU binary paths exist, giving hardware flexibility.
5. **Not in SFPU_OP_CHAIN dispatch**: Does not participate in the `get_op_init_and_func` / `SFPU_OP_CHAIN_0` mechanism. The `is_parametrized_type` check returns false for HARDSHRINK.

### Pattern Applicability to RReLU
RReLU (`max(0, x) + alpha * min(0, x)` where alpha is random during training, fixed during eval) shares key structural similarities:
- Conditional behavior based on sign of input (like hardshrink uses `ltz_tile`/`gtz_tile`)
- Requires a scalar parameter (alpha/negative_slope)
- Could use the same two-phase approach with temporary CB
- The `fill_tile` + comparison + multiply pattern is directly reusable
