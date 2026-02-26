// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::precompiled {

// Look for a pre-compiled binary directory for the given hash.
// Checks whether <root>/pre-compiled/<hash>/ exists on disk.
// Returns the directory path (with trailing slash) if found, std::nullopt otherwise.
std::optional<std::string> find_precompiled_dir(const std::string& root, uint64_t hash);

}  // namespace tt::tt_metal::precompiled
