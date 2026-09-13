// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "jit_build/jit_build_utils.hpp"
#include "jit_build/pch.hpp"

namespace tt::jit_build {
namespace {

TEST(SharedPch, CacheRootsAndRemovedArtifactsAreIndependent) {
    namespace fs = std::filesystem;
    const fs::path root = utils::FileRenamer::generate_temp_path(fs::temp_directory_path() / "tt_metal_pch");
    fs::create_directories(root / "tt_metal/hw/inc/internal");
    std::ofstream(root / "tt_metal/hw/inc/internal/pch.h") << "#include <cstdint>\n";
    const auto compiler = root / "compiler";
    // This test checks cache ownership and artifact lifetime, not GCC PCH acceptance.
    std::ofstream(compiler) << "#!/bin/sh\nwhile [ $# -gt 0 ]; do\n"
                               "if [ \"$1\" = -o ]; then shift; printf pch > \"$1\"; exit; fi\n"
                               "shift\ndone\nexit 1\n";
    fs::permissions(compiler, fs::perms::owner_all);
    const auto build = [&](const fs::path& cache) {
        return ensure_pch(compiler.string(), "O2", "", root / PCH_UMBRELLA, cache);
    };
    const auto first = build(root / "first");
    const auto second = build(root / "second");
    EXPECT_FALSE(first.empty());
    EXPECT_FALSE(second.empty());
    EXPECT_NE(first, second);
    EXPECT_TRUE(fs::exists(first + ".gch"));
    EXPECT_TRUE(fs::exists(second + ".gch"));
    fs::remove_all(root / "first");
    EXPECT_EQ(build(root / "second"), second);
    EXPECT_EQ(build(root / "first"), first);
    EXPECT_TRUE(fs::exists(first));
    EXPECT_TRUE(fs::exists(first + ".gch"));
    fs::remove_all(root);
}

}  // namespace
}  // namespace tt::jit_build
