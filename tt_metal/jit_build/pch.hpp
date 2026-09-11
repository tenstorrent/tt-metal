// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace tt::jit_build {

// Relative to the tt-metal root; installed alongside the other JIT inputs.
inline constexpr std::string_view PCH_UMBRELLA = "tt_metal/hw/firmware/src/pch.h";

// Return the cached header path for -include, or empty on failure.
// Match the consuming compile's flags and optimization level for GCC PCH compatibility.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::string& root,
    const std::string& pch_root);

}  // namespace tt::jit_build
