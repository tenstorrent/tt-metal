// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/sync.h"

// How llk-lib's own instruction words reach the instruction buffer, and its scoped wait. The HAL issues its words
// itself; this is for the words llk-lib encodes because the HAL has no descriptor for them (platform/sfpu.h).
namespace llk2::platform
{

/** @brief Issue a compile-time instruction word. */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void issue()
{
    TTI_INSN(Word);
}

/** @brief Issue a run-time instruction word through the instruction buffer. */
inline __attribute__((always_inline)) void issue_buffered(const std::uint32_t word)
{
    TT_INSN(word);
}

/**
 * @brief Install a compile-time STALLWAIT, then issue the instructions it is for.
 *
 * @tparam Targets: Instruction classes held back.
 * @tparam Conditions: Conditions that must all hold before they proceed.
 * @param guarded: Callable issuing the instructions the wait is for; put only those inside (not the instruction
 *        that later consumes what they write).
 * @note The hardware sees a plain STALLWAIT: it holds back everything issued after it, not only `guarded`. The
 *       scope marks, in the source, which writes the wait is for.
 */
template <hal::sync::StallTarget Targets, hal::sync::StallCondition Conditions, typename Callable>
inline __attribute__((always_inline, flatten)) void stall_scope(Callable&& guarded)
{
    hal::sync::wait::stall<Targets, Conditions>();
    static_cast<Callable&&>(guarded)();
}

} // namespace llk2::platform
