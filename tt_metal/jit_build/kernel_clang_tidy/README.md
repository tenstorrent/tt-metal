<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kernel clang-tidy (post-hoc, from real JIT compiles)

clang-tidy configuration for device (kernel) code. The flow is documented in
[`tech_reports/code-indexing/kernel-clang-tidy.md`](../../../tech_reports/code-indexing/kernel-clang-tidy.md)
and driven by [`scripts/build_kernel_clang_tidy_commands.py`](../../../scripts/build_kernel_clang_tidy_commands.py).

| File | Used by |
| --- | --- |
| `codechecker.json` | CI. Checker selection, check options and scoping. |
| `skiplist` | CI, via `codechecker.json`. Paths excluded from analysis and reporting. |
| `review_status.yaml` | CI, at parse time. Per-path exceptions to a checker. |
| `.clang-tidy` | Local `--run` only. Mirrors the check options from `codechecker.json`. |

## Where are the mock SFPI headers?

There are none, on purpose. An earlier attempt
(PR [#37252](https://github.com/tenstorrent/tt-metal/pull/37252)) hand-wrote
stubs faking SFPI internals, generated `chlkc_*` content and compile-time args
so host clang could parse anything at all. Two things make that unnecessary:

1. **Real generated content, captured rather than synthesized.** A normal test
   run leaves every kernel's real generated files (`chlkc_unpack/math/pack.cpp`,
   `chlkc_descriptors.h`, `kernel_includes.hpp`, `defines_generated.h`) and the
   real compile command in the durable tt-metal-cache directory. Replaying those
   invocations through clang gives the analysis the same program facts the
   device build used.

2. **SFPI ships its own analysis-tool fallback.** Since SFPI 7.x,
   `<sfpi>/include/tensix_builtins.h` detects a non-SFPI compiler (no
   `__riscv_xtttensix*` predefine), derives the extension macro from the
   `ARCH_WORMHOLE`/`ARCH_BLACKHOLE`/`ARCH_QUASAR` define every kernel compile
   already carries, defines the `__xtt_vector` types, and includes
   `tensix_builtins.def` — machine-generated declarations of every
   `__builtin_rvtt_*` intrinsic. Its own header comment says it is "provided so
   that code analysis tools will not just give up in the face of an unknown
   function call". Clang parses the real `sfpi.h` stack unaided. Verified
   against the pinned SFPI release (7.73.0, see `tt_metal/sfpi-version`).

The remaining GCC/clang gaps are handled by the capture script as flag
translation, not fakes: `--target`/`-march` mapping, `-std=c++17` + `-ftt-*` →
`-std=c++20`, SFPI's own newlib/libstdc++ headers via `-isystem`, and the
riscv32 `int32_t`-is-`long` type-model overrides that `sfpi.h` static_asserts.
