# Original Developer Prompt
I need you to help me outline a task. I am contributing to a new “LLK Helper” API library which will aid AI kernel code generation by wrapping some of the more common and tricky kernel patterns in AI friendly calls.We already have some APIs built for tilize/untilize on this branch.  I need to implement them for 1D Matmul.You can reference the git log on this branch to see the tilize/untilize changes, and this page for an explanation of how we intend for this API to be used to produce AI generated operations:https://github.com/tenstorrent/tt-metal/blob/mstaletovic/NoPlanner/.claude/QUICK_START.mdFinally you can look at this page for an explanation of how 1D matmul works, and of course you can see how it works in the actual codebase too.For now we don’t need to actually complete this API implementation, I just want a task description/outline/plan that I can save in a ticket.  Keep in mind that Claude will be the one actually doing this so the outline needs to be human readable since I’ll put it in the ticket, but what really matters is that it convey’s the task instructions correctly to the future Claude instance which will do the actual work.

# Calude Generated Task Plan: Implement `matmul_1d` Helpers for the LLK Helper Library

## Background

The LLK Helper Library (`ttnn/cpp/ttnn/kernel_lib/`) wraps common but tricky kernel patterns in
AI-friendly, single-call helpers. It is used by the `create-op` AI pipeline (described in
`.claude/QUICK_START.md`) so that a future Claude instance can write correct, complete kernels
without needing to understand every LLK API detail. Existing helpers include
`tilize_helpers.hpp`, `untilize_helpers.hpp`, `reduce_helpers_compute.hpp`,
`binary_op_helpers.hpp`, and `copy_tile_helpers.hpp`.

All helpers follow the same structural pattern:
- A `.hpp` file — declarations, config enums, rich documentation comments
- A `.inl` file — implementation, included at the bottom of the `.hpp`
- Both live in `ttnn/cpp/ttnn/kernel_lib/`
- Compute helpers use namespace `compute_kernel_lib`; dataflow helpers use `dataflow_kernel_lib`

This task adds helpers for **single-core tiled matmul** (`C = A × B`). Multicore/mcast
variants are out of scope.

---

## What to Build

Four files:

1. `ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp` — compute helper declaration + enums
2. `ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.inl` — compute helper implementation
3. `ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp` — dataflow reader + writer helpers
4. `ttnn/cpp/ttnn/kernel_lib/docs/matmul_1d_reference.md` — LLM-targeted reference doc

Note: the dataflow helpers (file 3) follow the same `.hpp`-only pattern as
`reduce_helpers_dataflow.hpp` — no separate `.inl` is required if the implementation is
short enough to inline. Use judgment; if the implementations are long, split into a `.inl`.

---

## Reference Files (Study These First)

Before writing any code, read these files to understand the patterns to follow:

| File | What to learn |
|---|---|
| `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Enum declaration style, doc comment format, prerequisite docs, usage examples |
| `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl` | `static_assert` pattern, WaitMode branching, InitUninitMode branching, reconfig pattern |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | Dataflow helper declaration style, `FORCE_INLINE`, `dataflow_kernel_lib` namespace |
| `ttnn/cpp/ttnn/kernel_lib/docs/tilize_untilize_reference.md` | Reference doc format and content density |
| `tt_metal/hw/inc/api/compute/matmul.h` | The underlying LLK APIs: `mm_init`, `matmul_tiles` |
| `tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp` | Canonical single-core reader pattern |
| `tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp` | Canonical single-core writer pattern |
| `tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp` | Canonical compute kernel being wrapped |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp` | Alternate compute reference (TTNN version of same pattern) |

---

## File 1: `matmul_1d_helpers.hpp` (Compute)

### Includes and namespace

```cpp
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {
namespace matmul_1d_config { ... }
// function declaration
}  // namespace compute_kernel_lib
#include "matmul_1d_helpers.inl"
```

### Config enums (in `compute_kernel_lib::matmul_1d_config`)

Follow the exact structure used in `tilize_config` in `tilize_helpers.hpp`.

**`InitUninitMode`**:
- `InitAndUninit` (default) — calls `mm_init` at the start; use for standalone calls
- `InitOnly` — calls `mm_init` only; use as the first op in a chained sequence
- `UninitOnly` — no-op (there is no `mm_uninit` in the LLK API); document this explicitly
  in a comment; included for API symmetry with tilize/untilize
- `Neither` — skips init entirely; for middle calls in a chain

**`WaitMode`**:
- `WaitPerTile` (default) — `cb_wait_front` for 1 in0 tile + 1 in1 tile on each Kt iteration
- `WaitUpfront` — wait for all Kt tiles of in0 and all Kt×Nt tiles of in1 before the Nt
  loop, then pop them all after all Nt outputs for that Mt row are packed; use when the
  reader pre-loads a full block before compute begins
- `NoWait` — caller manages all CB synchronization; skip `cb_wait_front` and `cb_pop_front`

**`ReconfigureRegisterDatatypeMode`**:
- `NoReconfigure`
- `UnpackReconfigure`
- `PackReconfigure`
- `UnpackAndPackReconfigure` (default)

Same semantics as in `tilize_config`. Reconfiguration is applied before `mm_init`.

### Function signature

```cpp
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    matmul_1d_config::InitUninitMode init_uninit_mode = matmul_1d_config::InitUninitMode::InitAndUninit,
    matmul_1d_config::WaitMode wait_mode = matmul_1d_config::WaitMode::WaitPerTile,
    matmul_1d_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        matmul_1d_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>
ALWI void matmul_1d(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch = 1);
```

### Doc comment (required — place above the declaration)

Follow the exact format from `tilize_helpers.hpp`. Must include:

- **What it does**: wraps `C = A × B` tile-by-tile; reads from `in0_cb` (A tiles, MK layout)
  and `in1_cb` (B tiles, KN layout), writes to `out_cb` (C tiles, MN layout).
- **PREREQUISITE**: `compute_kernel_hw_startup(in0_cb, in1_cb, out_cb)` must be called at the
  start of the kernel before using this function. Use the three-argument form because srcA
  and srcB are different CBs.
- Template param descriptions (one line each).
- Runtime param descriptions.
- **Usage examples** — at minimum these five:
  1. Basic usage (all defaults)
  2. Register reconfiguration when transitioning from a different op mode
  3. `WaitUpfront` mode (tiles pre-loaded by reader)
  4. `NoWait` mode (caller manages synchronization)
  5. Back-to-back matmul calls using `InitOnly` / `Neither` / `UninitOnly`

---

## File 2: `matmul_1d_helpers.inl` (Compute Implementation)

### Static asserts (compile-time)
```cpp
static_assert(in0_cb != out_cb, "...");
static_assert(in1_cb != out_cb, "...");
static_assert(in0_cb < 32, "...");
static_assert(in1_cb < 32, "...");
static_assert(out_cb < 32, "...");
```

### Runtime asserts
```cpp
ASSERT(Mt > 0);
ASSERT(Nt > 0);
ASSERT(Kt > 0);
ASSERT(batch > 0);
PACK(ASSERT(get_cb_num_pages(out_cb) >= 1));
```

### Data format reconfiguration

Copy the reconfig pattern from `tilize_helpers.inl` directly. Applied before init:
- `UnpackReconfigure` or `UnpackAndPackReconfigure`: call `reconfig_data_format_srca(in0_cb)`
  and `reconfig_data_format_srcb(in1_cb)`
- `PackReconfigure` or `UnpackAndPackReconfigure`: call `pack_reconfig_data_format(out_cb)`

### Init

```cpp
if constexpr (init_uninit_mode == InitAndUninit || init_uninit_mode == InitOnly) {
    mm_init(in0_cb, in1_cb, out_cb);
}
// UninitOnly and Neither are no-ops — there is no mm_uninit in the LLK API.
```

### Main loop

Loop order: `batch × Mt × Nt × Kt`. This must match the CB production order from the reader
helper (see File 3).

```
for batch:
    for mt in Mt:
        [WaitUpfront: cb_wait_front(in0_cb, Kt), cb_wait_front(in1_cb, Kt * Nt)]
        for nt in Nt:
            acquire_dst()
            for kt in Kt:
                [WaitPerTile: cb_wait_front(in0_cb, 1), cb_wait_front(in1_cb, 1)]
                matmul_tiles(in0_cb, in1_cb, 0, 0, 0)
                [WaitPerTile: cb_pop_front(in0_cb, 1), cb_pop_front(in1_cb, 1)]
            cb_reserve_back(out_cb, 1)
            pack_tile(0, out_cb)
            cb_push_back(out_cb, 1)
            release_dst()
        [WaitUpfront: cb_pop_front(in0_cb, Kt), cb_pop_front(in1_cb, Kt * Nt)]
```

**WaitUpfront note**: in0_cb holds `Kt` tiles for the current mt row; in1_cb holds `Kt * Nt`
tiles (the full B block for this mt row). All Nt output tiles are computed before any input
tiles are popped. The program factory must size CBs accordingly: in0_cb ≥ Kt pages,
in1_cb ≥ Kt * Nt pages.

**Tile index note (WaitPerTile)**: `matmul_tiles(..., 0, 0, 0)` always uses index 0 because
tiles are popped immediately after use. The reader re-pushes the same A[mt, kt] tile once
per Nt iteration. See the reader helper for how this is handled.

---

## File 3: `matmul_1d_dataflow_helpers.hpp` (Reader + Writer)

Provides helpers for both reader and writer kernels. All functions in namespace
`dataflow_kernel_lib`, using `FORCE_INLINE`. Follow the style of `reduce_helpers_dataflow.hpp`.

### Includes

```cpp
#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib { ... }
```

If implementations are long, split into `matmul_1d_dataflow_helpers.inl` and include it at
the bottom of the `.hpp`.

### Reader helper: `read_matmul_tiles`

Wraps `reader_single_core_mm.cpp`. Reads A and B tiles from DRAM into their CBs in the order
the compute helper consumes them: for each (batch, mt, nt, kt), push 1 A tile then 1 B tile.

```cpp
template <uint32_t in0_cb, uint32_t in1_cb>
FORCE_INLINE void read_matmul_tiles(
    uint32_t in0_tensor_addr,
    uint32_t in1_tensor_addr,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t batch = 1,
    bool bcast_B = false);
```

Implementation:

```
Set up TensorAccessor s0 for in0, s1 for in1 using TensorAccessorArgs chaining:
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, in0_tensor_addr, get_tile_size(in0_cb));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, in1_tensor_addr, get_tile_size(in1_cb));

for b in batch:
    for mt in Mt:
        for nt in Nt:
            for kt in Kt:
                // A tile at (b, mt, kt)
                a_tile_index = b * Mt * Kt + mt * Kt + kt
                cb_reserve_back(in0_cb, 1)
                noc_async_read_tile(a_tile_index, s0, get_write_ptr(in0_cb))
                noc_async_read_barrier()
                cb_push_back(in0_cb, 1)

                // B tile at (b, kt, nt) — b dimension skipped if bcast_B
                b_tile_index = (bcast_B ? 0 : b * Kt * Nt) + kt * Nt + nt
                cb_reserve_back(in1_cb, 1)
                noc_async_read_tile(b_tile_index, s1, get_write_ptr(in1_cb))
                noc_async_read_barrier()
                cb_push_back(in1_cb, 1)
```

`bcast_B = true` means B is not batched (shared across all batch slices). Matches the
`bcast_B` parameter in `reader_bmm_tile_layout.cpp`.

The `TensorAccessorArgs` chaining pattern is required for compatibility with the TTNN program
factory, which inserts tensor accessor compile-time args in chained order. Do not use fixed
offsets.

### Writer helper: `write_matmul_tiles`

Wraps `writer_single_core_mm.cpp`.

```cpp
template <uint32_t out_cb>
FORCE_INLINE void write_matmul_tiles(
    uint32_t out_tensor_addr,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t batch = 1);
```

Implementation:

```
Set up TensorAccessor s for out_cb.

for b in batch:
    for mt in Mt:
        for nt in Nt:
            tile_index = b * Mt * Nt + mt * Nt + nt
            cb_wait_front(out_cb, 1)
            noc_async_write_tile(tile_index, s, get_read_ptr(out_cb))
            noc_async_write_barrier()
            cb_pop_front(out_cb, 1)
```

Use `noc_async_write_barrier()` per tile for simplicity and correctness. The doc comment
should note that `noc_async_write_flushed()` + a single final `noc_async_write_barrier()` is
a valid throughput optimization, but the per-tile barrier form is preferred here.

### Doc comments

Each function needs a doc comment specifying:
- What it does
- Template params (CB indices)
- Runtime params (addresses, dimensions)
- CB sizing requirements
- TensorAccessor compile-time arg layout

---

## File 4: `ttnn/cpp/ttnn/kernel_lib/docs/matmul_1d_reference.md` (LLM Reference)

A dense, factual reference for a future LLM writing kernels with these helpers. Model it on
`docs/tilize_untilize_reference.md` — short sentences, no padding, code examples.

Required sections:

### Includes and prerequisites
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"          // compute kernels
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp" // reader/writer kernels
```
Compute kernels must call `compute_kernel_hw_startup(in0_cb, in1_cb, out_cb)` before any
helper call. Use the three-argument form — srcA and srcB are different CBs.

### Dimension notation
- `Mt = M / 32`, `Kt = K / 32`, `Nt = N / 32`
- M, K, N must all be multiples of 32
- A: shape `[batch, Mt, Kt]` in tiles — tile at (b, mt, kt) has linear index `b*Mt*Kt + mt*Kt + kt`
- B: shape `[batch, Kt, Nt]` in tiles — tile at (b, kt, nt) has linear index `b*Kt*Nt + kt*Nt + nt`
- C: shape `[batch, Mt, Nt]` in tiles — tile at (b, mt, nt) has linear index `b*Mt*Nt + mt*Nt + nt`

### CB setup requirements
```
in0_cb: tile-sized pages, >= 1 page (WaitPerTile) or >= Kt pages (WaitUpfront)
in1_cb: tile-sized pages, >= 1 page (WaitPerTile) or >= Kt * Nt pages (WaitUpfront)
out_cb: tile-sized pages, >= 1 page
```
All CBs use tiled data format (not row-major).

### Full single-core matmul kernel skeleton

Show all three kernels using the helpers:

```cpp
// reader kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"
void kernel_main() {
    uint32_t in0_addr  = get_arg_val<uint32_t>(0);
    uint32_t in1_addr  = get_arg_val<uint32_t>(1);
    uint32_t Mt        = get_arg_val<uint32_t>(2);
    uint32_t Kt        = get_arg_val<uint32_t>(3);
    uint32_t Nt        = get_arg_val<uint32_t>(4);
    uint32_t batch     = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    dataflow_kernel_lib::read_matmul_tiles<cb_in0, cb_in1>(in0_addr, in1_addr, Mt, Nt, Kt, batch);
}

// writer kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);
    uint32_t Nt       = get_arg_val<uint32_t>(2);
    uint32_t batch    = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    dataflow_kernel_lib::write_matmul_tiles<cb_out>(out_addr, Mt, Nt, batch);
}

// compute kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"
void kernel_main() {
    uint32_t Mt    = get_arg_val<uint32_t>(0);
    uint32_t Kt    = get_arg_val<uint32_t>(1);
    uint32_t Nt    = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
}
```

### TensorAccessor compile-time arg layout

The reader helper creates two `TensorAccessor` objects using chained `TensorAccessorArgs`.
The program factory must insert their compile-time args in order after any named CB args:

```
CTA[0 .. N-1]:  accessor args for in0  (s0_args = TensorAccessorArgs<0>())
CTA[N .. M-1]:  accessor args for in1  (s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>())
```

### WaitMode trade-offs
- `WaitPerTile` (default): CB depth 1 sufficient for in0 and in1. Reader and compute
  naturally pipeline. Use for all standard cases.
- `WaitUpfront`: CB must hold the full Mt-row block. Reader must also pre-load the full
  block before compute begins — the reader helper does not support this mode natively;
  a hand-written reader is required when using `WaitUpfront`.

### InitUninitMode use cases
- `InitAndUninit`: standalone matmul kernel. Most common case.
- `InitOnly`: matmul followed by an eltwise op in the same kernel; init matmul first, then
  init the eltwise op.
- `UninitOnly` / `Neither`: no practical use for matmul since there is no `mm_uninit`.
  Included for API symmetry.

---

## Testing

Testing proceeds in three phases: implement and smoke-test the helpers, generate a full TTNN
op using the `/create-op` AI pipeline, then run regression tests against existing tests.

---

### Phase 1 — Smoke test: migrate the programming example

The `tt_metal/programming_examples/matmul/matmul_single_core/` example uses exactly the
patterns being wrapped. Replacing its three kernels is the fastest way to confirm the helpers
compile and produce correct results.

Migrate:
- `kernels/compute/mm.cpp` → call `compute_kernel_lib::matmul_1d`
- `kernels/dataflow/reader_single_core_mm.cpp` → call `dataflow_kernel_lib::read_matmul_tiles`
- `kernels/dataflow/writer_single_core_mm.cpp` → call `dataflow_kernel_lib::write_matmul_tiles`

Validate by running the example binary from the build output and checking its golden
comparison output. This example does its own correctness check internally.

---

### Phase 2 — End-to-end: AI-generated TTNN op via `/create-op`

Use the `create-op` pipeline to generate a full TTNN operation (`matmul_sc`) that uses the
new helpers end-to-end. This is the primary validation that the helpers are AI-usable with no
prior context.

Before invoking, confirm that `ttnn/cpp/ttnn/kernel_lib/docs/matmul_1d_reference.md` is
present and that the architect's discovery step scans `ttnn/cpp/ttnn/kernel_lib/docs/`.

Invoke:
```
/create-op matmul_sc: single-core tiled matrix multiplication C = A × B.
  Inputs: two rank-2 bfloat16 tiled interleaved tensors A [M, K] and B [K, N].
  Output: rank-2 bfloat16 tiled interleaved tensor C [M, N].
  Constraint: M, K, N must all be multiples of 32. Single-core only.
```

Expected TDD stages (the architect determines these, but they should be approximately):
1. `data_pipeline` — reader + writer passing A through to output; compute is a no-op copy
2. `matmul_compute` — full `matmul_1d` compute with reader and writer using the helpers

The generated op will land at `ttnn/ttnn/operations/matmul_sc/`.

---

### Phase 3 — Regression testing

After Phase 2 produces a passing `matmul_sc` op, run the following to confirm nothing is
broken and output is numerically correct:

```bash
# Validate matmul_sc output against torch.matmul (PCC >= 0.999)
pytest tests/ttnn/unit_tests/operations/matmul_sc/test_matmul_sc.py -v

# Broader TTNN matmul regression (catches any regressions in shared infrastructure)
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py \
    -k "test_tutorial_matmul or test_matmul_does_dot_product or test_matmul_with_matched_width_height" \
    -v

# GTest integration suite
./build/tests/tt_metal/unit_tests_common --gtest_filter="*matmul*"
```

`test_matmul_sc.py` is generated by the `/create-op` pipeline in Phase 2. It should cover
at minimum: a small tile-aligned shape (e.g. 64×64×64), a larger shape (e.g. 512×256×128),
and a batched case (e.g. batch=2, 128×64×64).

---

## What Is Out of Scope for This Task

- **Multicore / mcast variants** — the 1D mcast factory
  (`matmul_multicore_reuse_mcast_1d_program_factory`) and any work distribution logic.
- **Fused bias or activation** — post-matmul fused operations.
- **Transpose** — `matmul_tiles` supports a transpose flag for B; not exposed in v1.
- **Dataflow (reader/writer) helpers** — removed per PR feedback; DM helpers are a
  separate initiative since they are not tightly coupled to the compute pattern.

---

## PR Feedback and Revisions (2026-03-23)

The original task plan was revised based on PR feedback:

### Naming: `matmul_1d` → `matmul_tile`
"matmul 1d" in TTNN refers to how work is split across cores (1D row distribution),
not the LLK-level matmul pattern. LLK has no such concept. Renamed to `matmul_tile`
to describe what the helper actually wraps: tile-by-tile matmul using `mm_init` +
`matmul_tiles`.

### Dataflow helpers removed
Reader/writer helpers were removed from scope. They are generic tile-moving patterns
not tightly coupled to matmul compute. The reduce scaler DM helper was the exception
because scaler generation is unique to reduce. Dataflow helpers will be a separate
initiative.

### `matmul_block` helper added (was previously out of scope)
A `matmul_block` compute helper was implemented wrapping the sub-blocked matmul pattern
from `bmm_large_block_zm.cpp`, including spill/reload of partial results. This is the
performance variant used in production TTNN matmul kernels.

### Standalone helper tests added
GTest-based standalone tests were added for the `matmul_tile` helper, exercising it
directly with hand-written reader/writer kernels (no TTNN op overhead). This is the
first such test for the helper library.

### Current file locations (post-revision)
- `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp` — tile-at-a-time compute helper
- `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.inl` — implementation
- `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` — sub-blocked compute helper
- `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl` — implementation
- `ttnn/cpp/ttnn/kernel_lib/docs/matmul_tile_reference.md` — LLM reference doc
- `tests/tt_metal/tt_metal/integration/matmul/test_matmul_tile_helper.cpp` — standalone test

## PR Feedback and Revisions (2026-03-24)

### `matmul_block` argument structs
PR feedback identified that `matmul_block`'s 12 positional `uint32_t` arguments are
unreadable from the caller side and error-prone for AI code generation. Refactored
into three descriptive structs in `matmul_block_config` namespace:

- `In0BlockParams` — A-matrix block parameters (block_w, num_subblocks, block_num_tiles, subblock_num_tiles)
- `In1BlockParams` — B-matrix block parameters (num_subblocks, block_num_tiles, per_core_w)
- `OutSubblockParams` — output sub-block dimensions (h, w, num_tiles)

Callers now use designated initializers for self-documenting calls:
```cpp
compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
    {.block_w = in0_block_w, .num_subblocks = in0_num_subblocks,
     .block_num_tiles = in0_block_num_tiles, .subblock_num_tiles = in0_subblock_num_tiles},
    {.num_subblocks = in1_num_subblocks, .block_num_tiles = in1_block_num_tiles,
     .per_core_w = in1_per_core_w},
    num_blocks,
    {.h = out_subblock_h, .w = out_subblock_w, .num_tiles = out_subblock_num_tiles},
    batch);
```

### `bmm.cpp` migration tested
The TTNN production matmul compute kernel (`bmm.cpp`) was migrated to use
`matmul_tile` helper. Verified with the full TTNN matmul Python test suite:
588 passed, 104 skipped, 0 failures.

### Test results (2026-03-24)
| Test suite | Result | Validates |
|---|---|---|
| TTNN matmul Python (test_matmul.py) | 588 passed, 104 skipped | bmm.cpp with matmul_tile helper |
| Matmul tile helper GTest (4 cases) | 4/4 passed, PCC > 0.997 | matmul_tile helper directly |
| matmul_multicore_reuse example | Passed, PCC = 0.999 | matmul_block helper with struct API |
| Integration GTests (large block) | 13/13 passed | Raw LLK matmul_block regression |
