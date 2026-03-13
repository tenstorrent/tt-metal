// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::precompiled {

// Look for a pre-compiled firmware directory for the given build hash.
// Checks whether <root>/pre-compiled/<hash>/ exists on disk.
// Returns the directory path (with trailing slash) if found, std::nullopt otherwise.
std::optional<std::string> find_precompiled_firmware_dir(const std::string& root, uint64_t hash);

// Compatibility wrapper for existing firmware callers.
std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash);

// Look for a pre-compiled kernel directory.
// Checks whether <root>/<kernel_name>/<compile_hash>/ exists on disk.
// Returns the directory path (with trailing slash) if found, std::nullopt otherwise.
std::optional<std::string> find_precompiled_kernel_dir(
    const std::string& root, const std::string& kernel_name, size_t compile_hash);

}  // namespace tt::tt_metal::precompiled
