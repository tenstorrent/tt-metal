// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace tt::jit_build {

// Relative to the tt-metal root; installed alongside the other JIT inputs.
inline constexpr std::string_view PCH_UMBRELLA = "tt_metal/hw/inc/internal/pch.h";

// Throw if the umbrella cannot be read; return the cached header path or empty if PCH setup fails.
// The STL-only umbrella uses the compiler's default include paths.
// Match the consuming compile's flags and optimization level for GCC PCH compatibility.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::filesystem::path& umbrella,
    const std::filesystem::path& pch_root);

}  // namespace tt::jit_build
