// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/sync.h"
#include "hal/utils/issue.h"

// How llk-lib's own instruction words reach the instruction buffer, and its scoped wait. The HAL issues its words
// itself; this is for the words llk-lib encodes because the HAL has no descriptor for them (platform/sfpu.h).
// With the HAL's compiler-visible issue path (hal/utils/issue.h) these go the same way as the HAL's own words.
namespace llk2::platform
{

/** @brief Issue a compile-time instruction word. */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void issue()
{
    ::hal::detail::issue<Word>();
}

/** @brief Issue a run-time instruction word through the instruction buffer. */
inline __attribute__((always_inline)) void issue_buffered(const std::uint32_t word)
{
    ::hal::detail::issue_buffered(word);
}

/**
 * @brief Install a compile-time STALLWAIT, then issue the instructions it is for.
 *
 * @tparam Targets: Instruction classes held back.
 * @tparam Conditions: Conditions that must all hold before they proceed.
 * @param guarded: Callable issuing the instructions the wait is for; put only those inside (not the instruction
 *        that later consumes what they write).
 * @note The hardware sees a plain STALLWAIT: it holds back everything issued after it, not only `guarded`. A
 *       compiler that tracks Tensix state is also told where the guarded range ends, so it can delete the wait when
 *       none of the guarded instructions is left or needs it.
 * @note Do not nest scopes, and keep a mutex acquire out of the scope: the compiler keeps a wait that holds an ATGETM.
 */
template <hal::sync::StallTarget Targets, hal::sync::StallCondition Conditions, typename Callable>
inline __attribute__((always_inline, flatten)) void stall_scope(Callable&& guarded)
{
    HAL_TTI_STALLWAIT_SCOPED(hal::to_underlying(Targets), hal::to_underlying(Conditions));
    static_cast<Callable&&>(guarded)();
    HAL_STALL_SCOPE_END();
}

} // namespace llk2::platform
