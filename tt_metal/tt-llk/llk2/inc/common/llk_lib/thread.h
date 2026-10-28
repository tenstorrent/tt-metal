// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib's thread split, the same on every arch. Every TRISC compiles the same kernel body, so every thread meets
// every Compute API call. llk-lib's per-thread calls (unpack, matmul, ...) take the thread they run on (On) as a
// template parameter: they plan on every thread, and configure and execute only where current_thread == On.
// Elsewhere that branch is discarded and never instantiated, so each TRISC's build carries only its own thread's
// code (tt-llk-lab llk-tensor, report 4, section 3).

#pragma once

#include "tensor/tensor.h"

namespace llk2
{

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
inline constexpr Thread current_thread = Thread::Unpack;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
inline constexpr Thread current_thread = Thread::Math;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
inline constexpr Thread current_thread = Thread::Pack;
#else
inline constexpr Thread current_thread = Thread::None; // host builds: plan only, nothing executes
#endif

template <Thread On>
inline constexpr bool runs_on = current_thread == On;

} // namespace llk2
