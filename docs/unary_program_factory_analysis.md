# Unary Program Factory — Detailed Analysis

How 13 of the Top 25 SFPU operations propagate through a single, shared program factory.

**Factory file**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

**Operations covered**: RELU, ABS, EXP, RECIP, SIGMOID, GELU, TANH, SQRT, LOG, SILU, ERFINV, ASIN, I0

---

## 1. Factory Architecture Overview

The unary program factory is a **single-path, define-driven** design. All 13 operations (and ~100+ other unary ops) share the exact same factory code. The factory never branches on `UnaryOpType` to change its structure — it delegates all op-specific behavior to a utility layer (`unary_op_utils.cpp`) that produces preprocessor defines. These defines are injected at kernel compile time so the device-side compute kernel instantiates the correct SFPU function.

There are two factory classes in the file:
- **`UnaryProgramFactory`** — the primary factory, uses `split_work_to_cores` for automatic core distribution
- **`UnarySubCoreGridProgramFactory`** — variant for user-specified sub-core grids, otherwise identical logic

This analysis focuses on `UnaryProgramFactory::create`.

---

## 2. Step-by-Step Factory Flow

### 2.1 Data Format and Page Size Resolution (lines 35–59)

```
input tensor → datatype_to_dataformat_converter → cb_data_format
output tensor → datatype_to_dataformat_converter → cb_data_format_output
```

The factory computes tile sizes from the data format. For row-major tensors it uses buffer page sizes instead. This stage is **completely op-agnostic** — all 13 operations pass through identically.

One exception: if `ops_chain[0].type() == BITCAST`, the input CB uses the output format to avoid unpacker conversion. None of the 13 Top 25 ops hit this path.

### 2.2 Core Grid Distribution (lines 46–49)

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```

Tiles (or row-major pages) are distributed across all available compute cores. Two core groups handle the uneven remainder: group 1 gets `ceil(num_pages/num_cores)` pages, group 2 gets `floor(num_pages/num_cores)`. **Identical for all 13 ops.**

### 2.3 Circular Buffer Setup (lines 50–84)

Three circular buffers are created:

| CB Index | Name | Size | Purpose |
|----------|------|------|---------|
| `c_0` | Input | 2 pages | Double-buffered input from DRAM |
| `c_1` | Temp | 2 pages | **Only** for HARDSHRINK, CBRT, LOGIT |
| `c_2` | Output | 2 pages | Double-buffered output to DRAM |

**For all 13 Top 25 ops**, only `c_0` and `c_2` are created. The temporary buffer `c_1` is never allocated because none of them are HARDSHRINK, CBRT, or LOGIT. This means all 13 ops have a minimal 2-CB memory footprint.

### 2.4 Dataflow Kernel Creation (lines 86–101)

Two fixed kernels are created, identical for all ops:

- **Reader**: `reader_unary_interleaved_start_id.cpp` — reads pages from DRAM one at a time into `c_0`
- **Writer**: `writer_unary_interleaved_start_id.cpp` — writes pages from `c_2` back to DRAM one at a time

Both use `TensorAccessorArgs` for compile-time buffer layout info. **Zero op-specific variation.**

### 2.5 Compute Configuration — The Differentiation Point (lines 108–168)

This is where the 13 ops diverge. Three mechanisms inject op-specific behavior:

#### 2.5.1 Approximation Mode (line 114–115)

```cpp
bool math_approx_mode = std::all_of(
    args.op_chain.begin(), args.op_chain.end(),
    [](const auto& u) { return utils::get_op_approx_mode(u.type()); });
```

`get_op_approx_mode()` currently returns `false` for **all** op types (default switch case). So `math_approx_mode = false` for all 13 ops. This is a no-op differentiation point today.

#### 2.5.2 Preprocessor Defines via `get_block_defines` (line 116)

```cpp
std::map<std::string, std::string> unary_defines =
    utils::get_block_defines(args.op_chain, "0", "0", input.dtype());
```

This is the **core differentiation mechanism**. For each op in the chain, it produces:

1. **An include macro** (via `get_macro_definition`) — controls which SFPU header gets `#include`d
2. **An init function string** (via `get_op_init_and_func`) — the `*_tile_init()` call
3. **A compute function string** (via `get_op_init_and_func`) — the `*_tile()` call

These are injected as `SFPU_OP_CHAIN_0_INIT_0`, `SFPU_OP_CHAIN_0_FUNC_0`, and `SFPU_OP_CHAIN_0` defines.

#### 2.5.3 Data Type Defines (lines 118–126)

An input type macro is set: `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT`. This tells the compute kernel the source data format.

#### 2.5.4 Compute Kernel Path (line 155)

```cpp
auto path = fmt::format("{}/{}",
    compute_root, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));
```

For all 13 Top 25 ops, `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` (the default case). Only a handful of specialized ops (MISH, TANHSHRINK, HARDSWISH, HARDSHRINK, CBRT, etc.) get custom kernel files.

### 2.6 Scalar Runtime Arguments (lines 128–153)

The factory packs up to 2 scalar runtime arguments. Only HARDSHRINK, WHERE_TSS, and LOGIT use this path. **None of the 13 Top 25 ops** trigger the special-case switch, so both `packed_scalar1` and `packed_scalar2` remain `0u`.

### 2.7 Per-Core Runtime Argument Assignment (lines 191–212)

Each core receives:
- **Reader**: `{src_buffer_address, num_pages_per_core, start_page_id}`
- **Writer**: `{dst_buffer_address, num_pages_per_core, start_page_id}`
- **Compute**: `{packed_scalar1=0, packed_scalar2=0}`

**Identical for all 13 ops.** The compute kernel receives the scalars but ignores them (they're zero).

---

## 3. Per-Operation Define Propagation

The table below traces each of the 13 ops through the three utility functions that produce their unique compile-time identity:

| Operation | `get_macro_definition` → Include Macro | `get_op_init_and_func` → Init | `get_op_init_and_func` → Func | Parametrized? |
|-----------|---------------------------------------|------|------|---|
| **RELU** | `SFPU_OP_RELU_FAMILY_INCLUDE` → `relu.h` | `relu_tile_init()` | `relu_tile(0)` | No |
| **ABS** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `abs_tile_init()` | `abs_tile(0)` | No |
| **EXP** | `SFPU_OP_EXP_INCLUDE` → `exp.h` | `exp_tile_init<P>()` | `exp_tile<P>(0)` | Yes (fast_and_approx flag) |
| **RECIP** | `SFPU_OP_RECIP_INCLUDE` → `recip.h` | `recip_tile_init<false>()` | `recip_tile<false>(0)` | No |
| **SIGMOID** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `sigmoid_tile_init<P1>()` | `sigmoid_tile<VecMode, P1>(0)` | Yes (VecMode, approx) |
| **GELU** | `SFPU_OP_GELU_INCLUDE` → `gelu.h` | `gelu_tile_init<P>()` | `gelu_tile<P>(0)` | Yes (approx flag) |
| **TANH** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `tanh_tile_init<P>()` | `tanh_tile<P>(0)` | Yes (approx flag) |
| **SQRT** | `SFPU_OP_SQRT_INCLUDE` → `sqrt.h` | `sqrt_tile_init()` | `sqrt_tile<P>(0)` | Yes (approx flag) |
| **LOG** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `log_tile_init<P>()` | `log_tile<P>(0)` | Yes (approx flag) |
| **SILU** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `silu_tile_init()` | `silu_tile(0)` | No |
| **ERFINV** | `SFPU_OP_ERFINV_INCLUDE` → `erfinv.h` | `erfinv_tile_init()` | `erfinv_tile(0)` | No |
| **ASIN** | `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` → `compute_kernel_api.h` | `asin_tile_init()` | `asin_tile(0)` | No |
| **I0** | `SFPU_OP_I0_INCLUDE` → `i0.h` | `i0_tile_init()` | `i0_tile(0)` | No |

### Include Macro Families

The 13 ops use two include strategies:

1. **Dedicated include macro** (6 ops): EXP, GELU, RECIP, SQRT, ERFINV, I0 — each gets its own `SFPU_OP_*_INCLUDE` that pulls in a single header via `sfpu_split_includes.h`
2. **Default include** (7 ops): RELU, ABS, SIGMOID, TANH, LOG, SILU, ASIN — use `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` which pulls the monolithic `compute_kernel_api.h`, or `SFPU_OP_RELU_FAMILY_INCLUDE` for the relu family

The split-include system is a **compile-time optimization**: by only including the header for the specific SFPU function, kernel compilation is faster and code size is smaller. Ops that fall through to the default include `compute_kernel_api.h` which contains everything.

---

## 4. The Generic Compute Kernel (`eltwise_sfpu.cpp`)

All 13 ops execute through the same 47-line compute kernel:

```cpp
void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // always 1

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);
            SFPU_OP_CHAIN_0              // <-- THIS IS WHERE THE OP HAPPENS
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_2);
            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

The `SFPU_OP_CHAIN_0` macro expands to the init + func defines, e.g., for EXP:
```cpp
exp_tile_init<0u>(); exp_tile<0u>(0);
```

For RELU:
```cpp
relu_tile_init(); relu_tile(0);
```

The tile processing loop is always:
1. `cb_wait_front(c_0, 1)` — wait for reader to push one tile
2. `copy_tile(c_0, 0, 0)` — copy tile from CB to DST register
3. `SFPU_OP_CHAIN_0` — apply the SFPU operation in-place on DST
4. `pack_tile(0, c_2)` — pack DST back to output CB
5. `cb_pop_front(c_0, 1)` — free the input slot

This means **every unary SFPU op, regardless of complexity (from trivial RELU to nested ERFINV), processes exactly one tile at a time through the same pipeline**. The difference in execution time is entirely inside the SFPU function itself — the factory and kernel wrapper contribute zero structural variation.

---

## 5. What Makes This Factory Unique

### 5.1 Complete Op-Agnosticism

Unlike the binary or ternary factories which branch on op type to select different kernel files, broadcast modes, or CB configurations, the unary factory has **zero conditional branches** based on the op type for the 13 Top 25 ops. The entire op-specific behavior is pushed into compile-time string defines.

### 5.2 The Op-Chain Mechanism

The factory supports **chaining multiple unary ops** in a single kernel invocation. The `args.op_chain` is a vector of `EltwiseUnaryWithParam`, and `get_block_defines` iterates over it to produce `SFPU_OP_CHAIN_0_INIT_0`, `SFPU_OP_CHAIN_0_FUNC_0`, `SFPU_OP_CHAIN_0_INIT_1`, `SFPU_OP_CHAIN_0_FUNC_1`, etc. This means a user can fuse e.g. EXP + RECIP into a single kernel launch without a separate SIGMOID op. The `SFPU_OP_CHAIN_0` macro concatenates all init+func pairs.

### 5.3 Double-Buffered But Single-Tile Processing

The CBs are sized for 2 pages (double-buffered), but `per_core_block_dim` is always set to `1`. This means the compute kernel processes one tile at a time within the inner loop. The double-buffering allows the reader to fill the next slot while compute processes the current one, but only at the granularity of single tiles.

### 5.4 Program Caching

The factory returns `cached_program_t` containing the program and shared variables (kernel handles, core counts). On subsequent calls with the same op configuration, `override_runtime_arguments` updates only the buffer addresses — the program structure and compiled kernels are reused.

---

## 6. Why All 13 Ops Share One Factory

The unary SFPU operations share a factory because they all follow the same structural pattern:

1. **Single input, single output** — no broadcast dimensions to handle
2. **Element-wise on tiles** — each tile is independent, no cross-tile dependencies
3. **No special CB requirements** — just input and output, no intermediate multi-tile accumulation
4. **No dataflow variation** — read tile, compute, write tile; the SFPU function is the only variable

The binary factory can't do this because broadcast modes (height, width, height+width, scalar) fundamentally change CB setup and kernel structure. The ternary factory can't do this because 3-input ops need different CB configurations and broadcast combinations. But all unary SFPU ops are structurally identical — the factory just needs to inject the right function name.

---

## 7. Parametrized vs Non-Parametrized Ops

Among the 13 ops, 6 are parametrized (carry compile-time or runtime parameters):

| Operation | Parameter | Effect |
|-----------|-----------|--------|
| EXP | `fast_and_approx` (uint32_t) | Template arg controlling approximation accuracy |
| SIGMOID | `VecMode` + `approx` | VecMode selects C or RC vector mode; approx controls accuracy |
| GELU | `approx` (uint32_t) | `0` = accurate, `1` = fast approximation |
| TANH | `approx` (uint32_t) | Template arg for approximation level |
| SQRT | `approx` (uint32_t) | Template arg for approximation level |
| LOG | `approx` (uint32_t) | Template arg for approximation level |

The remaining 7 (RELU, ABS, RECIP, SILU, ERFINV, ASIN, I0) are non-parametrized — they have a single fixed implementation.

Parameters are baked into the kernel at compile time via the define strings (e.g., `exp_tile_init<1u>()` vs `exp_tile_init<0u>()`). They do **not** flow through runtime arguments — the factory always passes `packed_scalar1 = 0, packed_scalar2 = 0` for all 13 ops. Only HARDSHRINK, WHERE_TSS, and LOGIT use the scalar runtime arg path.

---

## 8. Data Type Handling

The factory sets one of four input type defines:

| Condition | Define | Ops affected from Top 25 |
|-----------|--------|--------------------------|
| `input.dtype() == FLOAT32` | `INP_FLOAT32` | All 13 (when input is FP32) |
| `input.dtype() == INT32` | `INP_INT32` | RELU (has INT32 variant) |
| `input.dtype() == UINT32` | `INP_UINT32` | — |
| Otherwise (BF16) | `INP_FLOAT` | All 13 (typical BF16 path) |

Some ops also have dtype-specific tile function variants. For example, `relu_tile()` dispatches to `relu_tile_int32()` when the input is INT32. This branching happens in `get_op_init_and_func`, not in the factory itself.

---

## 9. FP32 Precision Control

Two flags control precision:

- **`fp32_dest_acc_en`**: Enables FP32 accumulation in the destination register (normally BF16)
- **`preserve_fp32_precision`**: Sets `UnpackToDestFp32` mode on input CBs, keeping full FP32 through unpack → DST → SFPU → pack

These are set by the caller (operation layer above the factory) and apply uniformly. They don't vary per-op within the factory.

---

## 10. Summary: The Op-Specific "Fingerprint"

For each of the 13 Top 25 unary ops, the factory produces a unique compilation by combining exactly three things:

1. **Include macro** → which SFPU header to compile
2. **Init string** → which `*_tile_init()` to call
3. **Func string** → which `*_tile()` to call

Everything else — CB setup, core distribution, dataflow kernels, runtime args, tile processing loop — is byte-for-byte identical across all 13 operations.
