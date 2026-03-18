// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "precompiled.hpp"

#include <filesystem>
#include <fmt/format.h>
#include "common/filesystem_utils.hpp"

namespace tt::tt_metal::precompiled {

std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash) {
    auto path = fmt::format("{}tt_metal/pre-compiled/{}/", root, hash);
    if (tt::filesystem::safe_is_directory(path).value_or(false)) {
        // TODO: validate the dir contains all binaries we need
        return path;
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::precompiled
