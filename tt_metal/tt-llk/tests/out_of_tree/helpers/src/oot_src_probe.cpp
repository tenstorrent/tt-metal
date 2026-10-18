// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Reached via add_helpers_tree(<tree>) -> add_src_include_dirs(<tree>/src).
// This is the tests/helpers/src role: pulled in with #include <foo.cpp> from
// the driver, not compiled as its own translation unit. Everything here is
// constexpr/inline so repeated inclusion across the three TRISC TUs is safe.

#pragma once

#include <cstdint>

#include "oot_helpers.h"

constexpr std::uint32_t oot_src_probe_value()
{
    return OOT_EXPECTED_SRC_VALUE;
}
