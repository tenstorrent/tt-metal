# How to Add a New Eltwise SFPU Operator

This document describes how to add a new eltwise SFPU operator (unary, binary or ternary) in `tt-metal` using
the static op-struct dispatch layer in `llk_math_eltwise_sfpu_op.h`. It replaces both the earlier per-op
`llk_math_eltwise_unary_sfpu_<op>.h` wrapper headers and the `SFPU_UNARY_CALL` / `SFPU_UNARY_INIT*` macro
family that followed them.

## How dispatch works

`tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/llk_math_eltwise_sfpu_op.h` (one copy per arch, same
class surface) defines three CRTP bases:

| Base | `calculate(...)` | `init(...)` |
| --- | --- | --- |
| `SfpuUnaryOp<Derived, DST_SYNC, DST_ACCUM, TILE_SHAPE>` | `(dst_index, vector_mode, args...)` | shared SFPU init, then `Derived::init_kernel(args...)` |
| `SfpuBinaryOp<Derived, ...>` | `(in0, in1, out, vector_mode, args...)` | same |
| `SfpuTernaryOp<Derived, ...>` | `(in0, in1, in2, out, vector_mode, args...)` | same |

`calculate()` checks the dest indices (and, on Quasar, the vector mode) and then runs
`_llk_math_eltwise_<arity>_sfpu_params_` with `Derived::kernel` as the per-face body. `init()` runs the shared
SFPU init `_llk_math_eltwise_sfpu_init_()` (SFPU config register, `ADDR_MOD_7 = {0,0,0}`, counter reset;
identical on every arch) and then `Derived::init_kernel(args...)`. `SfpuOpBase` supplies a no-op
`init_kernel()`; an op that needs more state defines its own, which hides the base one through ordinary CRTP
name lookup. There is no op enumeration and no central init table: the base classes know nothing about
individual ops. Everything is `static` and always-inlined; there is no object and no runtime cost over a
hand-written call.

`DST_SYNC` and `DST_ACCUM` are ordinary template parameters. The dispatch layer never reads
`DST_SYNC_MODE` / `DST_ACCUM_MODE`; the compute API entry point defaults its own
`bool is_fp32_dest_acc_en = DST_ACCUM_MODE` parameter and passes it down, so the kernel-wide define is
consulted in exactly one place.

## Step-by-step: adding a new op

### 1. Implement the kernel

Write the per-face body in `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` (or in
tt-llk, with a thin metal header including it), as before:

```cpp
namespace ckernel::sfpu {
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_negative() { ... }
}  // namespace ckernel::sfpu
```

### 2. Add the op struct next to the kernel

In the same header, derive a struct from the matching base. Its template parameters are *all* of the op's
compile-time configuration, in this order: op-specific parameters first, then `DstSync DST_SYNC, bool
DST_ACCUM`, then anything that can carry a default.

```cpp
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

// Backs negative_tile / negative_tile_init.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Negative : SfpuUnaryOp<Negative<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_negative<APPROXIMATION_MODE, ITERATIONS>(); }
    // No init_kernel: init() is the shared SFPU init only.
};

}  // namespace ckernel::sfpu
```

Rules:

- If the kernel has its own fp32-dest-accumulation parameter, feed `DST_ACCUM` into it; do not add a second bool.
- Provide `static void init_kernel(runtime_args...)` only when the op needs state beyond the shared init:
  its own init function (`sfpu::<op>_init<...>()`, LUT/constant programming, replay buffers) or a dest
  auto-increment on `ADDR_MOD_6`. The op owns `ADDR_MOD_6`; program it inside `init_kernel`, e.g.
  `addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);` (see
  `ZeroComp`, `BinaryComp`, `Typecast`). Otherwise omit `init_kernel` and the inherited no-op is used.
- If one kernel serves several modes (comparisons, trigonometry, isinf/isnan...), define a per-family
  `enum class` next to the kernel (`ZeroCompMode`, `UnaryCompMode`, `IsInfNanMode`, `BinaryCompMode`, `TrigOp`)
  and make it a template parameter, instead of writing near-duplicate structs. There is no global op enum.
- Binary/ternary kernels receive the dest indices as their leading parameters; copy the kernel's signature
  into `kernel(...)`.
- Add the struct to **every arch that has the kernel**, with an identical name and template list. An arch
  whose kernel takes fewer arguments simply ignores the extra parameters. Per-arch differences in kernel
  template lists belong here, not in the compute API header.

### 3. Write the compute API header

`tt_metal/hw/inc/api/compute/eltwise_unary/<op>.h` includes only the kernel header (which pulls in the
dispatch layer) and calls the struct:

```cpp
#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_negative.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
    ALWI void negative_tile_init() {
    MATH((sfpu::Negative<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
    }

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
    ALWI void negative_tile(uint32_t idst) {
    MATH((sfpu::Negative<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC)));
}

}  // namespace ckernel
```

Note the double parentheses inside `MATH((...))`: the template argument list contains commas. Keep
`is_fp32_dest_acc_en` as the last template parameter so existing callers are unaffected.

## Kernels without an op struct

Test harnesses and one-off kernels in downstream code can dispatch a fully specialised kernel directly
through the generic adapters, without writing a struct:

```cpp
using MyOp = SfpuUnaryFn<sfpu::my_face_fn<APPROX, 8>, DST_SYNC_MODE, DST_ACCUM_MODE, sfpu::my_init<APPROX>>;
MyOp::init();                                   // shared init, then sfpu::my_init<APPROX>()
MyOp::calculate(idst, VectorMode::RC, arg);
SfpuBinaryFn<sfpu::my_binary_fn, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(in0, in1, out, VectorMode::RC);
SfpuBinaryFn<sfpu::my_binary_fn, DST_SYNC_MODE, DST_ACCUM_MODE>::init();   // shared init only
```

Prefer a named op struct for anything exposed through the compute API.

## Migration notes

- Do not create `llk_math_eltwise_unary_sfpu_<op>.h` wrapper headers.
- Do not add new preprocessor dispatch macros; the `SFPU_*_CALL` / `SFPU_*_INIT*` family has been removed.
- Do not add an op enumeration: the former global op enum and the `llk_math_eltwise_<arity>_sfpu_init<op>`
  wrappers have been removed. An op's identity is its struct; its init is `init_kernel`.
- The struct for an op lives with its kernel, one per arch; the compute API header is arch-neutral apart
  from `#ifndef ARCH_QUASAR` guards for ops an arch does not implement.
