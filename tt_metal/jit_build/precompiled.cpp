// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "precompiled.hpp"

#include <filesystem>
#include <fmt/format.h>

namespace tt::tt_metal::precompiled {

std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash) {
    auto path = fmt::format("{}tt_metal/pre-compiled/{}/", root, hash);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        // TODO: validate the dir contains all binaries we need
        return path;
    }
    log_error(tt::LogBuildKernels, "Pre-compiled directory {} does not exist.", path);
    return std::nullopt;
}

}  // namespace tt::tt_metal::precompiled
