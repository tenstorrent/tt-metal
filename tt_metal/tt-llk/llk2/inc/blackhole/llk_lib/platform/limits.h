// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

// HAL gaps: two values the Blackhole HAL keeps to itself or does not have.
namespace llk2::platform
{

/** @brief Largest MOP Template 1 loop count, outer or inner: the expander reads 10 bits of each (MOPExpander.md, Blackhole). */
inline constexpr std::uint32_t mop_max_loop_count = (1u << 10) - 1u;

/**
 * @brief Return the unpack thread's software copy of the config context its next UNPACRs use (0 or 1).
 * @note The copy is ckernel's (ckernel_globals.h), one per unpack-thread program: code on LLK 2.0 and code on the
 * older LLKs read and write the same one. The hardware's context is set separately.
 */
[[nodiscard]] inline __attribute__((always_inline)) std::uint32_t& unpack_config_context()
{
    return ::unp_cfg_context;
}

} // namespace llk2::platform
