# Matmul Init Refactoring — Implementation Guide

**Issue**: [#22946](https://github.com/tenstorrent/tt-metal/issues/22946) (under [#22219](https://github.com/tenstorrent/tt-metal/issues/22219))

## Context

The compute API cleanup (#22219) introduced `compute_kernel_hw_startup()` — a single,
operation-agnostic function that performs all HW configuration (MMIO writes) once at
kernel start. After that, each operation uses a lightweight "short init" that only sets
operation-specific state (no MMIO).

This was already completed for reduce (#22787) and tilize (#22850). Matmul was blocked
because it previously had its own `llk_unpack_AB_matmul_hw_configure_disaggregated()`
that collided with the generic `llk_unpack_hw_configure()`. That blocker was resolved
by commit `cb1d67d5820` — there is now only one `llk_unpack_hw_configure`.

## The matmul operand swap problem

Matmul maps operands in reverse: `in0 → srcB`, `in1 → srcA`. The current `mm_init()`
handles this by calling `llk_unpack_hw_configure(in1, in0)` — note the swap. But
`compute_kernel_hw_startup()` uses standard order `(icb0, icb1)`, so if you call
`compute_kernel_hw_startup(in0, in1, out)` the unpack data format registers end up
configured as `srcA=in0, srcB=in1` — backwards for matmul.

### What registers are actually wrong?

`llk_unpack_AB_matmul_init()` (the matmul-specific init) does NOT touch data format
registers. It only sets:
- Transpose mode flag (`THCON_SEC0_REG2_Haloize_mode`)
- Address counter X-end values (face dimensions)
- `KT_DIM` GPR
- Replay buffer for address increment patterns

So after a standard-order `compute_kernel_hw_startup()`, these data format registers
have the wrong values for matmul:

| Register | Has | Needs |
|----------|-----|-------|
| `THCON_SEC0_REG0[3:0]` (unpA/srcA input format) | `unpack_src_format[in0]` | `unpack_src_format[in1]` |
| `THCON_SEC0_REG2[3:0]` (unpA/srcA output format) | `unpack_dst_format[in0]` | `unpack_dst_format[in1]` |
| `THCON_SEC1_REG0[3:0]` (unpB/srcB input format) | `unpack_src_format[in1]` | `unpack_src_format[in0]` |
| `THCON_SEC1_REG2[3:0]` (unpB/srcB output format) | `unpack_dst_format[in1]` | `unpack_dst_format[in0]` |
| `THCON_SEC0/1_REG1` FP8 e4m3 flags | based on wrong CB | based on correct CB |
| `UNP0/1_ADDR_CTRL` Z-strides | based on wrong dst format | based on correct dst format |
| `TILE_SIZE_A/B` GPRs | wrong CB tile size | correct CB tile size |

On the math side: `llk_math_reconfig_data_format` with default `to_from_int8=false`
is a **complete no-op** — zero register writes. Only matters for Int8/Int32 formats.

### Key insight: when `in0` and `in1` have the same data format, the swap doesn't matter

All the "wrong" registers get identical values regardless of order. This is the common
case (e.g., both bf16, or both bfp8_b). The swap only produces wrong values when the
two inputs have different formats (e.g., bf16 + bfp8_b mixed precision).

## The solution: `compute_kernel_lib::matmul_init`

A new helper in `ttnn/cpp/ttnn/kernel_lib/matmul_helpers.hpp` that:

1. **Checks at compile time** whether `in0` and `in1` have different data formats using
   the JIT-generated `unpack_src_format[]` arrays (same technique as
   `tilize_helpers.inl::has_supported_fast_tilize_format()`).

2. **Only emits `reconfig_data_format(in1, in0)`** when formats actually differ.
   When both inputs share the same format, the `if constexpr` eliminates the call
   entirely — zero cost.

3. The reconfig cost when formats DO differ: ~8 register writes (4 RMW for format bits,
   2 FP8 flags, 2 SETDMAREG for tile sizes). The math-side reconfig is a no-op.

```cpp
template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb,
          bool transpose = false,
          matmul_config::ReconfigurePackMode pack_reconfig_mode = ...>
ALWI void matmul_init() {
    if constexpr (detail::needs_format_swap<in0_cb, in1_cb>()) {
        reconfig_data_format(in1_cb, in0_cb);
    }
    if constexpr (pack_reconfig_mode == PackReconfigure) {
        pack_reconfig_data_format(out_cb);
    }
    mm_init_short(in0_cb, in1_cb, transpose);
}
```

## What a refactored kernel looks like

```cpp
// Op-agnostic — standard CB order, works for any first operation
compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
compute_kernel_lib::matmul_init<cb_in0, cb_in1, cb_out, transpose>();

// ... after untilize ...
compute_kernel_lib::matmul_init<cb_in0, cb_in1, cb_out, transpose>();

// ... after tilize (which also changed pack format) ...
compute_kernel_lib::matmul_init<cb_in0, cb_in1, cb_out, transpose,
    ReconfigurePackMode::PackReconfigure>();
```

## What needs to happen (step by step)

### Step 1: Add `matmul_helpers.hpp`
- Create `ttnn/cpp/ttnn/kernel_lib/matmul_helpers.hpp`
- Implementation is in `matmul_helpers.hpp` in this directory

### Step 2: Refactor `transformer_attn_matmul.cpp` as pilot
- File: `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp`
- Replace `mm_init(...)` → `compute_kernel_hw_startup(...)` + `matmul_init<...>()`
- Replace `mm_init_short_with_dt(...)` → `matmul_init<...>()`
- Replace `mm_block_init_short_with_both_dt(...)` → `matmul_init<..., PackReconfigure>()`
- **Tests**: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py`
- Already validated with explicit reconfig (step 3 below), all 19 tests pass on Blackhole

### Step 3: Rename in `matmul.h` (the big API change)
- Strip HW config calls from `mm_init()` → make it equivalent to `mm_init_short()`
- Strip HW config calls from `mm_block_init()` → make it equivalent to `mm_block_init_short()`
- Delete: `mm_init_short`, `mm_init_short_with_dt`, `mm_block_init_short`,
  `mm_block_init_short_with_dt`, `mm_block_init_short_with_both_dt`
- ~70 kernel files need updating (add `compute_kernel_hw_startup` before first `mm_init`)

## Files in this directory

| File | Description |
|------|-------------|
| `1_before_kernel_lib.cpp` | Original kernel — raw init/uninit/reconfig ceremony |
| `2_current_with_kernel_lib.cpp` | Current main — tilize/untilize wrapped, matmul still raw |
| `3_with_hw_startup.cpp` | Tested — hw_startup + explicit reconfig (all 19 tests pass) |
| `4_proposed_with_matmul_init.cpp` | End goal — matmul_init hides everything |
| `matmul_helpers.hpp` | The helper implementation |
