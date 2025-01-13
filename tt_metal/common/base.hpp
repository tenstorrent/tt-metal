// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** \file base.hpp
 * The basic enums and data structures used by the rest of code base.
 */
#pragma once

#include <cstdint>

// DO NOT ADD MORE CODE TO THIS FILE
// THIS FILE POLLUTES ALL TRANSLATION UNITS - tt_metal, ttnn, programming examples, tests, customer code

// FIXME: At least put this in tt namespace
inline constexpr uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

namespace tt {
/**
 * @brief Specifies the target devices on which the graph can be run.
 */
enum class TargetDevice : std::uint8_t {
    Silicon = 0,
    Simulator = 1,
    Invalid = 0xFF,
};
}  // namespace tt
