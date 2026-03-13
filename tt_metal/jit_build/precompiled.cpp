// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "precompiled.hpp"

#include <filesystem>

namespace tt::tt_metal::precompiled {

std::optional<std::string> find_precompiled_firmware_dir(const std::string& root, uint64_t hash) {
    auto path = std::filesystem::path(root) / "tt_metal" / "pre-compiled" / std::to_string(hash);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        // TODO: validate the dir contains all binaries we need
        path /= "";
        return path.string();
    }
    return std::nullopt;
}

std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash) {
    return find_precompiled_firmware_dir(root, hash);
}

std::optional<std::string> find_precompiled_kernel_dir(
    const std::string& root, const std::string& kernel_name, size_t compile_hash) {
    auto path = std::filesystem::path(root) / kernel_name / std::to_string(compile_hash);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        path /= "";
        return path.string();
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::precompiled
