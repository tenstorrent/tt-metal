// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace tt::tt_fabric::visualizer {

// The commit tt-metal was built from, as the version header CMake refreshes on every build records it.
struct BuildIdentity {
    // Empty when the source tree is neither a git checkout nor a `git archive` export.
    std::string_view git_sha;
    std::string_view git_sha_short;
    // The version string, such as "v0.62.0-rc1-123-gabcdef1".
    std::string_view version;
    // The CMake build type, such as "Release".
    std::string_view build_type;
    // Whether tracked files had uncommitted changes.
    bool git_dirty = false;
};

BuildIdentity get_build_identity();

}  // namespace tt::tt_fabric::visualizer
