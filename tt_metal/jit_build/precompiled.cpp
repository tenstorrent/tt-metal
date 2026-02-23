// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "precompiled.hpp"

#include <filesystem>
#include <fmt/format.h>

namespace tt::tt_metal::precompiled {

std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash) {
    auto path = fmt::format("{}pre-compiled/{}/", root, hash);
    if (std::filesystem::is_directory(path)) {
        return path;
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::precompiled
