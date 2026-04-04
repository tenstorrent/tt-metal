# Reference Operation Selection for hardtanh

## Target Operation
- **Name**: `hardtanh`
- **Definition**: `max(min_val, min(max_val, x))` where min_val=-1.0 (default), max_val=1.0 (default)
- **Component operations identified**:
  - Upper-bound clamp: `min(x, max_val)` → maps to `unary_min_tile` or `clamp_tile` upper param
  - Lower-bound clamp: `max(x, min_val)` → maps to `unary_max_tile` or `clamp_tile` lower param
  - Composed as: `clamp_tile(idst, min_val_bits, max_val_bits)` — a built-in SFPU two-parameter clamp
  - Two configurable float parameters (min_val, max_val) packed as packed_scalar1, packed_scalar2

## SFPU Implementation Strategy

`hardtanh` maps directly to the `clamp_tile(idst, param0, param1)` SFPU primitive found in
`api/compute/eltwise_unary/clamp.h`. The kernel will:
1. Get two packed float parameters from runtime args (`packed_scalar1 = min_val`, `packed_scalar2 = max_val`)
2. Call `clamp_tile_init();` then `clamp_tile(0, packed_scalar1, packed_scalar2);`
3. Use the standard unary_ng eltwise_sfpu.cpp kernel via `SFPU_OP_CHAIN_0` dispatch

The `unary_program_factory` must be extended to pack both `min_val` and `max_val` as separate
`packed_scalar1` / `packed_scalar2` runtime arguments (similar to the LOGIT and WHERE_TSS pattern).

---

## Selected References (ranked by relevance)

### 1. logit
- **Why selected**: The `logit_kernel.cpp` is the **only existing compute kernel that directly calls
  `clamp_tile(0, packed_scalar1, packed_scalar2)`** — identical to what `hardtanh` needs. It shows:
  - Reading two packed float scalars from runtime args (`get_arg_val<uint32_t>(0)` and `(1)`)
  - Calling `clamp_tile_init()` then `clamp_tile(0, packed_scalar1, packed_scalar2)` with exactly
    the `api/compute/eltwise_unary/clamp.h` include
  - Two-parameter packing in `unary_program_factory.cpp` (the `case UnaryOpType::LOGIT` block that
    sets `packed_scalar1` and `packed_scalar2` and passes them as runtime args per-core)
  - Custom kernel path (`logit_kernel.cpp`) selected via `get_compute_kernel_path()`
  - Temporary CB (`c_1`) creation for intermediate results
- **Relevance**: **High** — `hardtanh` is a strict subset of `logit`: the logit kernel conditionally
  applies a `clamp_tile` at its start. `hardtanh` will be a simpler version that **only** applies
  `clamp_tile` and returns, using both `packed_scalar1` (min_val) and `packed_scalar2` (max_val).

### 2. relu6
- **Why selected**: RELU6 is mathematically a special case of `hardtanh` with `min_val=0`,
  `max_val=6`. It is already implemented in `unary_ng_op_utils.cpp` as:
  ```cpp
  case UnaryOpType::RELU6:
      return {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
  ```
  This demonstrates the standard `SFPU_OP_CHAIN_0` dispatch path through `eltwise_sfpu.cpp` for
  two-sided clamp operations. It also shows how a "hardcoded-parameter" clamp is registered in
  `unary_ng_op_utils.cpp` — the reference for adapting to configurable parameters.
- **Relevance**: **High** — Direct structural predecessor. `hardtanh` is a generalization of RELU6
  with configurable min/max. The `relu_max_tile` → `clamp_tile` refactor and parameter handling are
  the core difference.

### 3. hardsigmoid
- **Why selected**: `hardsigmoid(x) = relu6(x + 3) / 6 = clamp(x+3, 0, 6) / 6`. It internally
  implements a two-sided clamp as a native SFPU tile function (`hardsigmoid_tile`). It is already
  registered in `unary_ng_op_utils.cpp`:
  ```cpp
  case UnaryOpType::HARDSIGMOID:
      return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
  ```
  Shows how a clamp-based operation with no user-visible parameters is wired through the unary_ng
  framework via `SFPU_OP_CHAIN_0`. Also referenced by `hardswish_kernel.cpp` in unary_ng.
- **Relevance**: **High** — Mathematical cousin (contains a clamp). The `hardsigmoid_tile` SFPU
  function is a fixed-parameter version of what `hardtanh` does. Understanding how hardsigmoid is
  wired informs how to wire `hardtanh` (same path, but with runtime scalar parameters).

### 4. hardshrink
- **Why selected**: `hardshrink(x, λ) = x if |x| > λ, else 0`. This is a one-parameter conditional
  SFPU operation with a fully implemented custom kernel (`hardshrink_kernel.cpp` and
  `hardshrink_kernel_sfpu.cpp`). Shows:
  - One float parameter packed as `packed_scalar1` in `unary_program_factory.cpp`
  - `case UnaryOpType::HARDSHRINK:` in the program factory's parameter packing switch
  - `needs_tmp0_cb()` returning true (requires CB c_1 for intermediate)
  - Custom kernel selection in `get_compute_kernel_path()`
  - The parameter received via `get_arg_val<uint32_t>(0)` in the kernel
  Although `hardshrink` uses a more complex conditional structure than `clamp_tile`, the
  **parameter passing pattern** (pack float → runtime arg → kernel reads it) is directly reusable.
- **Relevance**: **Medium-High** — Most mature example of a one-parameter unary SFPU operation with
  custom kernel. The two-parameter extension for `hardtanh` follows the same infrastructure path.

### 5. where_tss
- **Why selected**: `where_tss(cond, true_val, false_val)` is the **only** currently implemented
  operation that uses **two separate float scalars** (`packed_scalar1` and `packed_scalar2`) in
  `unary_program_factory.cpp`:
  ```cpp
  case UnaryOpType::WHERE_TSS:
      packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
      packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());
      break;
  ```
  The `where_tss_kernel.cpp` then reads both via `get_arg_val<uint32_t>(0)` and `(1)`. This is the
  exact two-parameter runtime argument pattern that `hardtanh` will use for `min_val` and `max_val`.
  Shows how a `BasicUnaryWithParam` with two float parameters is wired end-to-end.
- **Relevance**: **Medium-High** — The two-scalar runtime arg plumbing in `unary_program_factory.cpp`
  (and `unary_ng_program_factory.cpp`) is the structural template for `hardtanh`'s `min_val` /
  `max_val` parameter pair, even though the kernel computation is different.

---

## Key Files for Implementation

| Layer | File |
|-------|------|
| SFPU kernel include | `api/compute/eltwise_unary/clamp.h` (provides `clamp_tile_init()`, `clamp_tile()`) |
| Compute kernel | New `hardtanh_kernel.cpp` (or extend `logit_kernel.cpp` pattern) |
| Op utils (unary_ng) | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` |
| Program factory | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp` |
| Op type enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (`HARDTANH` already present) |
