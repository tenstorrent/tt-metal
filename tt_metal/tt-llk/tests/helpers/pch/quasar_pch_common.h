// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// Precompiled-header (PCH) base for the Quasar `--compile-producer` build.
//
// This header aggregates the *variant-independent* prefix that every Quasar
// kernel translation unit parses before any generated per-variant header. A
// single Quasar compile job fans out ~400k `riscv-tt-elf-g++` invocations, and
// without a PCH each one re-parses this same ~2600 lines of stable headers from
// scratch (ckernel_ops.h alone is ~1256 lines of constexpr register-op
// wrappers). Precompiling it once per job run and force-including the result
// with `-include` eliminates that repeated front-end work.
//
// CORRECTNESS INVARIANT — only headers whose transitive `#include` closure is
// byte-identical across every variant in a run may live here. Concretely, this
// prefix must NOT (even transitively) reach the per-variant generated header
// `build.h` (pulled in via the static `helpers/include/params.h` wrapper). The
// closure of the headers below was verified to bottom out only at static,
// checked-in or toolchain-provided files:
//
//   <cstdint>            - libstdc++, stable
//   ckernel.h            - -> ckernel_ops.h / ckernel_include.h / tensix.h ...
//                          (does NOT reach cfg_defines.h / build.h / params.h)
//   llk_defs.h           - stable enums/defs
//   llk_memory_checks.h  - -> core_config.h / dev_mem_map.h (static hw headers)
//
// Notably `cfg_defines.h` (hw/inc/internal/tt-2xx/quasar/cfg_defines.h) IS
// static per-arch, so headers that reach it are still PCH-safe; only `build.h`
// (format inference emitted per variant) and its `params.h` wrapper are
// per-variant and are therefore deliberately absent from this prefix.
//
// This header is force-included at top-of-TU via `-include`, so every include
// below is guarded (`#pragma once`) and the source `#include "ckernel.h"` etc.
// in the kernel .cpp become no-ops. The per-variant `params.h`/`build.h` are
// still parsed normally, AFTER this prefix, exactly as before.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
