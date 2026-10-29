// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace tt::tt_fabric::manifest {

// The commit tt-metal was built from, as the version header CMake refreshes on every build records it.
struct BuildIdentity {
    // Empty when the source tree is neither a git checkout nor a `git archive` export.
    std::string_view git_sha;
    // Whether tracked files had uncommitted changes.
    bool git_dirty = false;
};

BuildIdentity get_build_identity();

}  // namespace tt::tt_fabric::manifest
