# SFPU Analysis: where_tss

## 1. Operation Overview

**Operation**: `where_tss` (Tensor-Scalar-Scalar variant of `where`)
**UnaryOpType enum**: `UnaryOpType::WHERE_TSS` (defined in `unary_op_types.hpp:118`)
**Math definition**: `where(x > 0, true_val, false_val)` -- for each element, if the condition tensor value is nonzero, output `true_val`; otherwise output `false_val`.
**PyTorch equivalent**: `torch.where(condition, true_scalar, false_scalar)`

This is a **ternary operation exposed as a unary** with two scalar parameters. The condition comes from the input tensor; the two scalar outputs are passed as runtime arguments.

**Parameter count**: 2 (`t_true`, `t_false`)
**Parametrized**: Yes -- uses `UnaryWithParam{UnaryOpType::WHERE_TSS, {t_true, t_false}}`

---

## 2. Abstraction Layer Inventory

### Layer 1: C++ API (`ttnn::where_tss`)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:114-127`
- **Signature**: `Tensor where_tss(const Tensor& condition, float t_true, float t_false, ...)`
- Calls `ttnn::detail::unary_impl(condition, {UnaryWithParam{UnaryOpType::WHERE_TSS, {t_true, t_false}}}, ...)`
- **Declaration**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:265-272`

### Layer 2: Ternary Integration
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:186-196`
- The ternary `where(tensor, scalar, scalar)` overload delegates to `ttnn::where_tss(...)`.
- Also `ternary.cpp:228` for integer overloads.

### Layer 3: UnaryOpType Enum
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:118`
- `WHERE_TSS` entry in the `UnaryOpType` enum.

### Layer 4: Op Utils (Dispatch)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `get_macro_definition`: WHERE_TSS falls through to `default`, returning `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"`.
  - **Note**: The original code used `"SFPU_OP_WHERE_INCLUDE"` per documentation (`docs/sfpu_operations/unary_eltwise_sfpu_list.md:316`), but this has been nuked. The current codebase uses the generic include.
- `get_compute_kernel_path`: WHERE_TSS falls through to `default`, returning `"eltwise_sfpu.cpp"`.
  - **Note**: The original code returned `"where_tss_kernel.cpp"`. The custom kernel file still exists but is not referenced by the dispatch.
- `get_op_init_and_func_parameterized` / `get_op_init_and_func_default`: No WHERE_TSS cases present (nuked).
- `get_op_approx_mode`: Returns `false` (default).

### Layer 5: Program Factory (Interleaved)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:132-134`
- Packs **two** scalar runtime args:
  - `packed_scalar1 = pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())` -- true_val
  - `packed_scalar2 = pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype())` -- false_val
- Runtime args set at line 200: `SetRuntimeArgs(program, kernel_id, core, {packed_scalar1, packed_scalar2})`
- Compute kernel path resolved via `utils::get_compute_kernel_path(...)`.
- Identical scalar packing also at the repeat location (line 369-371) for the sub-core-grids variant.

### Layer 5b: Program Factory (Sharded)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.cpp:158-160`
- Same two-scalar packing pattern.

### Layer 5c: Program Factory (unary_ng)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:46-49`
- Same two-scalar packing in the `unary_ng` (next-gen) factory.

### Layer 6: Custom Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp`
- **This is the critical file.** Full analysis in Section 3.

### Layer 7: LLK Wrapper (ckernel API)
- **File (WH)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h`
- **File (BH)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h`
- `llk_math_eltwise_ternary_sfpu_where<APPROXIMATE, data_format>(dst0, dst1, dst2, odst)` calls `_calculate_where_`.
- `llk_math_eltwise_ternary_sfpu_where_init<APPROXIMATE>()` calls `_init_where_`.

### Layer 8: SFPU Kernel Function
- **File (WH)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h`
- **File (BH)**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h`
- Full analysis in Section 4.

---

## 3. Custom Compute Kernel Analysis

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp`

### 3.1 Includes
```cpp
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
```

This kernel includes `eltwise_binary_sfpu.h` (for the ternary-style SFPU where dispatch) and `fill.h` (to load scalar values into DST tiles).

### 3.2 Runtime Arguments
| Arg Index | Name | Type | Description |
|-----------|------|------|-------------|
| 0 | `packed_scalar1` | `uint32_t` | Bit-cast `true_val` (float or int32) |
| 1 | `packed_scalar2` | `uint32_t` | Bit-cast `false_val` (float or int32) |

### 3.3 Compile-Time Arguments
| Arg Index | Name | Description |
|-----------|------|-------------|
| 0 | `per_core_block_cnt` | Number of outer blocks (tile rows) |
| 1 | `per_core_block_dim` | Number of tiles per block |

### 3.4 Circular Buffers
| CB | Index | Role |
|----|-------|------|
| `cb_input` | `c_0` | Input condition tensor tiles |
| `cb_output` | `c_2` | Output tensor tiles |

### 3.5 Kernel Main Loop

```
init_sfpu(cb_input, cb_output)

for block in range(per_core_block_cnt):
    cb_reserve_back(cb_output, per_core_block_dim)
    for tile in range(per_core_block_dim):
        cb_wait_front(cb_input, 1)           # Wait for one input tile
        tile_regs_acquire()
        copy_tile_to_dst_init_short(cb_input)
        copy_tile(cb_input, 0, 0)            # Copy condition tile to DST[0]

        fill_tile_init()
        fill_tile(1, *true_value)            # Fill DST[1] with true_val scalar
        fill_tile(2, *false_value)           # Fill DST[2] with false_val scalar

        SFPU_OP_CHAIN_0                      # Execute where: select from DST[1]/DST[2] based on DST[0]
        tile_regs_commit()

        tile_regs_wait()
        pack_tile(0, cb_output)              # Pack result from DST[0]
        tile_regs_release()

        cb_pop_front(cb_input, 1)
    cb_push_back(cb_output, per_core_block_dim)
```

### 3.6 Data Format Handling

The kernel uses conditional compilation:
- `INP_INT32` / `INP_UINT32`: Uses `fill_tile_int<DataFormat::Int32>(dst_idx, packed_scalar)`.
- `INP_FLOAT` / `INP_FLOAT32`: Uses `fill_tile(dst_idx, float_value)`.

The packed scalars are reinterpreted as floats at the top of kernel_main via `reinterpret_cast<const float*>(&packed_scalar)`.

### 3.7 Key Design Pattern: "Ternary via Unary"

WHERE_TSS is conceptually ternary (condition, true_val, false_val) but packaged as unary because:
- The condition is the only tensor input (goes through the standard unary reader/writer).
- The two scalar values are passed as **runtime arguments to the compute kernel** (not as additional tensor inputs).
- The kernel constructs 3 DST tiles in-register: DST[0]=condition, DST[1]=true_val_tile, DST[2]=false_val_tile.
- Then `SFPU_OP_CHAIN_0` calls the ternary where SFPU function, which operates on all 3 DST tiles.

### 3.8 SFPU_OP_CHAIN_0 Expansion

`SFPU_OP_CHAIN_0` is a preprocessor macro defined by the program factory via `get_block_defines()`. For WHERE_TSS, it expands to:
```
SFPU_OP_CHAIN_0_INIT_0  SFPU_OP_CHAIN_0_FUNC_0
```
Where:
- `SFPU_OP_CHAIN_0_INIT_0` = the init function (e.g., `where_tile_init();`)
- `SFPU_OP_CHAIN_0_FUNC_0` = the compute function (e.g., `where_tile<DataFormat::Float16_b>(0);`)

**Note**: In the nuked codebase, these init/func strings are **not generated** because `get_op_init_and_func_parameterized` lacks a WHERE_TSS case. This is one of the components that needs restoration.

---

## 4. SFPU Kernel Function Analysis

**File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h`
(Blackhole version is structurally identical, differs only in address modifier constants: ADDR_MOD_7/ADDR_MOD_6 vs ADDR_MOD_3/ADDR_MOD_2.)

### 4.1 Function Signature

```cpp
template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const uint32_t dst_index_in0,  // condition tile
    const uint32_t dst_index_in1,  // true_val tile
    const uint32_t dst_index_in2,  // false_val tile
    const uint32_t dst_index_out)  // output tile
```

### 4.2 Supported Data Formats

Static assertion restricts to: `Float32`, `Float16_b`, `Int32`, `UInt32`.

### 4.3 Algorithm

The where operation is implemented using **SFPU conditional execution**:

1. **Load condition** from DST[in0] into LREG0.
2. **Load true_val** from DST[in1] into LREG1.
3. **SFPSETCC**: Set lane-enable flags based on `LREG0 == 0` (condition is zero/false).
4. **Load false_val** from DST[in2] into LREG1 -- this **overwrites only the lanes where condition == 0** (because SFPSETCC disabled the non-zero lanes).
5. **SFPENCC**: Re-enable all lanes.
6. **Store LREG1** to DST[out] -- LREG1 now contains the correct mix of true/false values.

The key insight: `SFPSETCC` with `LREG_EQ0` disables lanes where the condition is **nonzero** (true). The subsequent `SFPLOAD` of false_val into LREG1 only affects disabled lanes. After `SFPENCC`, LREG1 has true_val where condition!=0 and false_val where condition==0.

### 4.4 SFPI Instructions Used

| Instruction | Purpose |
|-------------|---------|
| `TT_SFPLOAD` / `TT_SFPLOADMACRO` | Load 32 values from DST register into SFPU local register |
| `TTI_SFPSETCC` | Set conditional execution flags (lane enable = condition==0) |
| `TTI_SFPENCC` | End conditional execution (re-enable all lanes) |
| `TT_SFPSTORE` | Store 32 values from SFPU local register back to DST |
| `TTI_SFPLOADI` | Load immediate for macro configuration |
| `TTI_SFPCONFIG` | Configure SFPLOADMACRO instruction scheduling |
| `lltt::record` / `lltt::replay` | Record/replay instruction sequences for efficient iteration |

### 4.5 SFPLOADMACRO Optimization

The non-fallback path uses **SFPLOADMACRO** for instruction-level parallelism across the SFPU's Load, Simple, and Store units:

**Case 1: `dst_out == dst_in0`** (output overwrites condition) -- **3 cycles per row**:
```
Load Unit               | Simple Unit                     | Store Unit
SFPLOAD L0=Dst[offset0] |                                 |
SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)  |
SFPLOAD L0=Dst[offset2] | SFPENCC (all lanes enabled)      |
(next SFPLOAD L0)       |                                 | SFPSTORE Dst[offset0]=L0
```

**Case 2: `dst_out != dst_in0`** -- **4 cycles per row**:
```
Load Unit               | Simple Unit                     | Store Unit
SFPLOAD L0=Dst[offset0] |                                 |
SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)  |
SFPLOAD L0=Dst[offset2] | SFPENCC (all lanes enabled)      |
-                       |                                 | SFPSTORE Dst[offset3]=L0
(next SFPLOAD L0)       |                                 |
```

### 4.6 Init Function

`_init_where_<APPROXIMATE>()` configures SFPLOADMACRO templates:
- **InstructionTemplate[0]**: `SFPSETCC` with `LREG_EQ0` mode (slot 12 in config space)
- **InstructionTemplate[1]**: `SFPENCC` (slot 13 in config space)
- **Macro 0**: For output==input0 case (3-cycle schedule with integrated store)
- **Macro 1**: For output!=input0 case (4-cycle schedule)
- **Macro 2**: Shared across both cases (sets up SFPENCC in pipeline)
- **Misc config**: `UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1`

### 4.7 Iteration Pattern

`ITERATIONS = 8` (hardcoded in template instantiation via the LLK wrapper). Each 32x32 tile has 32 rows; processing 4 rows per iteration over 8 iterations covers the full tile.

### 4.8 Data Format Handling

```cpp
constexpr uint32_t mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
```
- `Float16_b`: Uses `LO16` modifier (load/store lower 16 bits of each 32-bit DST register slot).
- `Float32`, `Int32`, `UInt32`: Uses `INT32` modifier (full 32-bit load/store).

### 4.9 Platform Differences (Wormhole vs Blackhole)

| Aspect | Wormhole B0 | Blackhole |
|--------|-------------|-----------|
| Address modifier (load) | `ADDR_MOD_3` | `ADDR_MOD_7` |
| Address modifier (store) | `ADDR_MOD_2` | `ADDR_MOD_6` |
| Replay buffer loading | Direct `lltt::record` + inline instructions | Lambda-based `load_replay_buf` helper |
| Algorithm | Identical | Identical |

---

## 5. Complete File Inventory

| File | Layer | Role |
|------|-------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:265-272` | API declaration | C++ header |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:114-127` | API implementation | Entry point |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:118` | Enum | WHERE_TSS value |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Dispatch | Macro def, kernel path, init/func (currently nuked) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp:132-134,369-371` | Program factory | Scalar arg packing |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_sharded_program_factory.cpp:158-160` | Sharded factory | Scalar arg packing |
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp:46-49` | NG factory | Scalar arg packing |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp` | Compute kernel | Custom kernel main |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` | LLK wrapper (WH) | Ternary SFPU LLK |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` | LLK wrapper (BH) | Ternary SFPU LLK |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h` | SFPU kernel (WH) | Low-level implementation |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h` | SFPU kernel (BH) | Low-level implementation |
| `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:186-196,228` | Ternary bridge | Delegates to where_tss |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Split includes | Needs SFPU_OP_WHERE_INCLUDE (nuked) |

---

## 6. What Was Nuked (Gaps in Current Codebase)

The following components are **missing from the current codebase** but are required for WHERE_TSS to work through the standard unary dispatch:

1. **`get_macro_definition`** (`unary_op_utils.cpp:18-26`): Missing `case UnaryOpType::WHERE_TSS: return "SFPU_OP_WHERE_INCLUDE";`
2. **`get_compute_kernel_path`** (`unary_op_utils.cpp:167-171`): Missing `case UnaryOpType::WHERE_TSS: return "where_tss_kernel.cpp";`
3. **`get_op_init_and_func_parameterized`** (`unary_op_utils.cpp:29-43`): Missing WHERE_TSS case that generates init/func strings.
4. **`sfpu_split_includes.h`** (`tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`): Missing `#if SFPU_OP_WHERE_INCLUDE` include block.
5. **Compute API header** (`api/compute/eltwise_unary/where.h` or similar): The `where_tile` / `where_tile_init` API functions that wrap `llk_math_eltwise_ternary_sfpu_where` are not present in the hw/inc/api headers (nuked or never existed as a unary API -- they exist in the ternary path).

### What Still Exists

- The **custom compute kernel** (`where_tss_kernel.cpp`) is intact.
- The **SFPU kernel functions** (`ckernel_sfpu_where.h`) are intact in both WH and BH.
- The **LLK wrappers** (`llk_math_eltwise_ternary_sfpu_where.h`) are intact.
- The **program factory scalar packing** code is intact in all three factories.
- The **UnaryOpType::WHERE_TSS enum value** is intact.
- The **C++ API** (`where_tss()`) and ternary bridge are intact.

---

## 7. Key Implementation Notes for Replication

### 7.1 "Ternary as Unary" Pattern
This operation demonstrates a pattern where a conceptually multi-input operation is implemented through the unary infrastructure by:
- Using `fill_tile()` to inject scalar constants into DST register slots (DST[1], DST[2]).
- Using the ternary SFPU kernel function that reads from multiple DST slots.
- Using a **custom compute kernel** (not `eltwise_sfpu.cpp`) to orchestrate the fill + SFPU call.

### 7.2 Custom Kernel vs Standard eltwise_sfpu.cpp
WHERE_TSS cannot use the standard `eltwise_sfpu.cpp` because:
- Standard unary kernels only copy one input tile to DST[0] and call SFPU_OP_CHAIN_0.
- WHERE_TSS needs 3 DST slots populated (condition + two scalars) before calling the SFPU function.
- The `fill_tile()` calls are specific to this kernel.

### 7.3 Two-Scalar Runtime Arg Pattern
The two-scalar packing in program factories is unique among unary ops (most have 0 or 1 scalar). The pattern is:
```cpp
case UnaryOpType::WHERE_TSS:
    packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
    packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());
    break;
```

### 7.4 No SFPI (Pure TTI/TT Instruction Style)
The SFPU kernel uses raw TTI/TT assembly-style instructions and SFPLOADMACRO scheduling, **not** the higher-level SFPI C++ DSL (`sfpi::vFloat`, etc.). This is because the where operation requires conditional lane execution (`SFPSETCC`/`SFPENCC`) and multi-source register manipulation that don't map cleanly to SFPI abstractions.

### 7.5 Relevance as a Reference for Other Operations
- **For operations needing conditional execution**: The `SFPSETCC`/`SFPENCC` pattern is directly applicable.
- **For operations with scalar parameters injected via fill_tile**: The custom kernel + fill_tile pattern is reusable.
- **For operations bridging ternary/unary dispatch**: The "TSS" pattern (one tensor, N scalars) is a model for similar operations (e.g., `CLAMP_TSS`).
