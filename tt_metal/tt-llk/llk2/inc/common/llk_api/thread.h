// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The thread split. Every TRISC compiles the same kernel body; a per-thread call takes the thread it runs on (On)
// as a template parameter and executes only where current_thread == On. Elsewhere the execute branch is discarded
// and never instantiated (report 4, section 3).

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
