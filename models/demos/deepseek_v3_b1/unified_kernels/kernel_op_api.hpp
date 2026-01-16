// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// Core type detection - constexpr bools for use with constexpr if
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
inline constexpr bool is_ncrisc = true;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = false;
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#elif defined(COMPILE_FOR_BRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = true;
inline constexpr bool is_trisc = false;
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#elif defined(COMPILE_FOR_TRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = true;
#include "tools/profiler/kernel_profiler.hpp"
#include "compute_kernel_api/blank.h"
#endif

// ============================================================================
// Unified kernel entry point macro
// ============================================================================
#if defined(COMPILE_FOR_TRISC)
#define KERNEL_ENTRY      \
    namespace NAMESPACE { \
    void MAIN
#define KERNEL_END }
#else
#define KERNEL_ENTRY void kernel_main()
#define KERNEL_END
#endif

// ============================================================================
// Type helper: Select type based on current core
// Usage: using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;
// Note: ReaderConfigDescriptor -> NCRISC, WriterConfigDescriptor -> BRISC
// ============================================================================
namespace unified_kernels {

template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV = std::conditional_t<is_ncrisc, Reader, std::conditional_t<is_brisc, Writer, Compute>>;

}  // namespace unified_kernels
