// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Reached via add_helpers_tree(<tree>) -> add_include_dirs(<tree>/include).

#pragma once

#include <cstdint>

#define OOT_EXPECTED_MARKER    0xA5A5u
#define OOT_EXPECTED_SRC_VALUE 0x5A5Au

constexpr std::uint32_t oot_helpers_marker()
{
    return OOT_EXPECTED_MARKER;
}
