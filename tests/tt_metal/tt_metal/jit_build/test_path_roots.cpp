// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "jit_build/path_roots.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::jit_build {

TEST(JitPathRootsTest, PathIsWrittenRelativeToItsLongestRoot) {
    PathRoots roots;
    roots.add("APP", "/work");
    roots.add("ROOT", "/work/tt-metal/");

    EXPECT_EQ(roots.relocatable("/work/tt-metal/hw/a.h"), "${ROOT}/hw/a.h");
    EXPECT_EQ(roots.relocatable("/work/ops/k.cpp"), "${APP}/ops/k.cpp");
    EXPECT_EQ(roots.relocatable("/work"), "${APP}");
    EXPECT_EQ(roots.relocatable("/workspace/k.cpp"), "/workspace/k.cpp");

    EXPECT_EQ(roots.resolve("${ROOT}/hw/a.h"), "/work/tt-metal/hw/a.h");
    EXPECT_EQ(roots.resolve("${OTHER}/a.h"), "${OTHER}/a.h");
}

TEST(JitPathRootsTest, FlagStringPathsAreRelocated) {
    PathRoots roots;
    roots.add("ROOT", "/work/tt-metal");

    EXPECT_EQ(
        roots.relocatable_text("-O2 -I. -I/work/tt-metal/hw/inc -Wl,--just-symbols=/work/tt-metal/fw.elf -L/opt/lib "),
        "-O2 -I. -I${ROOT}/hw/inc -Wl,--just-symbols=${ROOT}/fw.elf -L/opt/lib ");
}

TEST(JitPathRootsTest, NoRootsLeavePathsAsTheyAre) {
    const PathRoots roots;
    EXPECT_TRUE(roots.empty());
    EXPECT_EQ(roots.relocatable("/work/k.cpp"), "/work/k.cpp");
    EXPECT_EQ(roots.relocatable_text("-I/work/inc"), "-I/work/inc");
}

// The mode belongs to the cache directory: asking for it marks the directory, and a marked
// directory is relocatable for a process that asks for nothing.
TEST(JitPathRootsTest, CacheDirectoryIsRelocatableOnceMarked) {
    namespace fs = std::filesystem;
    auto pattern = (fs::temp_directory_path() / "jit_path_roots_XXXXXX").string();
    const char* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    const fs::path cache_dir = directory;

    const char* previous = std::getenv("TT_METAL_CACHE");
    const std::optional<std::string> previous_cache = previous ? std::optional<std::string>(previous) : std::nullopt;
    setenv("TT_METAL_CACHE", cache_dir.c_str(), 1);
    const auto roots_when = [](std::optional<bool> requested) {
        tt::llrt::RunTimeOptions rtoptions;
        rtoptions.set_jit_relocatable_cache(requested);
        rtoptions.set_jit_path_roots("APP=/work");
        return PathRoots::from_rtoptions(rtoptions);
    };

    EXPECT_TRUE(roots_when(std::nullopt).empty());
    EXPECT_FALSE(fs::exists(cache_dir / RELOCATABLE_CACHE_MARKER));

    const PathRoots requested = roots_when(true);
    EXPECT_TRUE(fs::exists(cache_dir / RELOCATABLE_CACHE_MARKER));
    EXPECT_EQ(requested.relocatable((cache_dir / "1234/kernels/k.elf").string()), "${CACHE}/1234/kernels/k.elf");
    EXPECT_EQ(requested.relocatable("/work/ops/k.cpp"), "${APP}/ops/k.cpp");

    EXPECT_FALSE(roots_when(std::nullopt).empty());
    EXPECT_TRUE(roots_when(false).empty());

    previous_cache ? setenv("TT_METAL_CACHE", previous_cache->c_str(), 1) : unsetenv("TT_METAL_CACHE");
    fs::remove_all(cache_dir);
}

}  // namespace tt::jit_build
