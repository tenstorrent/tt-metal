# Experimental APIs

Features that are not part of the stable Metal API and may change or be
removed without notice.

## Named Kernel-Args (Blaze-Only)

> **Temporary feature.** This will be deleted when Blaze migrates to the
> Metal 2.0 `args::` system.  Do not use in new non-Blaze code.
> Removal is tracked by https://github.com/tenstorrent/tt-metal/issues/50953.

### What

Named kernel-args let a kernel descriptor specify runtime and compile-time
arguments by name (`"my_op.num_tiles"`) instead of by positional index.  The
host merges named values after positional ones, and a JIT-generated header
(`named_args_generated.h`) emits `blaze_ct_args::<ns>::<field>` constants and
`blaze_rt_args::Arg` / `blaze_rt_args::ArrayArg` descriptors so kernel code can write:

```cpp
uint32_t n = blaze_rt_args::get<blaze_ct_args::my_op::num_tiles>();
```

### Why experimental

This feature is used by the Blaze codebase but is not part of the stable
Metal API.  It adds types and code paths to the host and device pipelines
that will be superseded by the Metal 2.0 `args::` system.  Everything
related to it is quarantined in `namespace tt::tt_metal::experimental::blaze`
(host) and an opt-in device header so that it can be removed in one pass.

### Where

| Layer | Location | How to use |
|-------|----------|------------|
| Device | `tt_metal/hw/inc/experimental/blaze_named_args.h` | `#include "experimental/blaze_named_args.h"` in kernel source |
| Host C++ | `tt_metal/api/tt-metalium/experimental/blaze/named_kernel_args.hpp` | `experimental::blaze::NamedKernelArgs` on `KernelDescriptor` |
| Host impl | `tt_metal/impl/experimental/blaze/named_kernel_args.cpp` | `experimental::blaze::process_named_args()` + `emit_named_args_header()` |
| Python | `_ttnn.experimental` submodule | `blaze_named_*` property bindings on `KernelDescriptor` |

### Core runtime touch-points

A small number of core files still carry presence-gated hooks (default
no-op).  Each is fenced with `// Blaze-only experimental named args` banners
citing issue #50953 so they are easy to find and remove:

- `jit_build/jit_build_settings.hpp` — 2 virtuals + 4 type aliases
- `jit_build/genfiles.cpp` — `write_named_args_generated_header()`
- `impl/kernels/kernel.hpp` / `kernel.cpp` — 2 fields + 2 setters + 2 overrides
- `impl/program/program.cpp` — gated call to `experimental::blaze::process_named_args()`
- `impl/emulation/emulated_program_runner.cpp` — data plumbing + call to `experimental::blaze::emit_named_args_header()`
- `api/tt-metalium/program_descriptors.hpp` — `experimental::blaze::NamedKernelArgs blaze_named_args` member
