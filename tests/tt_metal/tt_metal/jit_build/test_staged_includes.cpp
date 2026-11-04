// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "jit_build/staged_includes.hpp"

namespace tt::jit_build {
namespace {

namespace fs = std::filesystem;

std::string read(const fs::path& path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

// A tt-metal root with a host and a device observability/api folder, and a cache root, both private to the test.
class StagedIncludesTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_ = fs::temp_directory_path() / ("staged_includes_test_" + std::to_string(::getpid()) + "_" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(base_);
        root_ = base_ / "root";
        cache_ = base_ / "cache";
    }
    void TearDown() override { fs::remove_all(base_); }

    fs::path host(const std::string& name) const { return root_ / OBSERVABILITY_API_HOST_DIR / name; }
    fs::path device(const std::string& name) const { return root_ / OBSERVABILITY_API_DEVICE_DIR / name; }
    fs::path staged(const std::string& include_root, const std::string& name) const {
        return fs::path(include_root) / "observability/api" / name;
    }

    fs::path base_, root_, cache_;
};

TEST_F(StagedIncludesTest, DeviceHeaderReplacesTheHostHeaderOfTheSameName) {
    write(host("a.h"), "#pragma once\n// host a\n");
    write(host("b.h"), "#pragma once\n// host b\n");
    write(device("b.h"), "#pragma once\n// device b\n");
    write(device("c.h"), "#pragma once\n// device c\n");
    write(device("README.md"), "# not a header\n");

    const std::string dir = stage_observability_api(root_, cache_);
    EXPECT_EQ(fs::path(dir).parent_path(), fs::absolute(cache_ / "include"));
    EXPECT_NE(read(staged(dir, "a.h")).find("// host a"), std::string::npos);
    EXPECT_NE(read(staged(dir, "b.h")).find("// device b"), std::string::npos);
    EXPECT_EQ(read(staged(dir, "b.h")).find("// host b"), std::string::npos);
    EXPECT_NE(read(staged(dir, "c.h")).find("// device c"), std::string::npos);
    EXPECT_FALSE(fs::exists(staged(dir, "README.md")));
}

TEST_F(StagedIncludesTest, CopyOpensWithALineNamingItsSource) {
    write(host("a.h"), "#pragma once\nint a;\n");
    write(device("b.h"), "#pragma once\nint b;\n");
    const std::string dir = stage_observability_api(root_, cache_);

    const std::string a = fs::absolute(host("a.h")).lexically_normal().string();
    EXPECT_EQ(
        read(staged(dir, "a.h")),
        "// Staged by the JIT build from " + a + "; edit that file, not this copy.\n#line 1 \"" + a +
            "\"\n#pragma once\nint a;\n");
    const std::string b = fs::absolute(device("b.h")).lexically_normal().string();
    EXPECT_NE(read(staged(dir, "b.h")).find("#line 1 \"" + b + "\"\n"), std::string::npos);
}

TEST_F(StagedIncludesTest, CompileErrorNamesTheSourceLine) {
    write(host("broken.h"), "#pragma once\n\n#error broken on purpose\n");
    const std::string dir = stage_observability_api(root_, cache_);

    const fs::path log = base_ / "compile.log";
    const std::string cmd =
        "c++ -fsyntax-only -x c++ -I" + dir + " -include observability/api/broken.h /dev/null 2>" + log.string();
    EXPECT_NE(std::system(cmd.c_str()), 0);
    const std::string source = fs::absolute(host("broken.h")).lexically_normal().string();
    EXPECT_NE(read(log).find(source + ":3:"), std::string::npos) << read(log);
}

TEST_F(StagedIncludesTest, FolderIsKeyedByContent) {
    write(host("a.h"), "#pragma once\n// one\n");
    const std::string first = stage_observability_api(root_, cache_);
    EXPECT_EQ(stage_observability_api(root_, cache_), first);

    write(host("a.h"), "#pragma once\n// two\n");
    const std::string second = stage_observability_api(root_, cache_);
    EXPECT_NE(second, first);
    EXPECT_NE(read(staged(first, "a.h")).find("// one"), std::string::npos);  // a published folder never changes
    EXPECT_NE(read(staged(second, "a.h")).find("// two"), std::string::npos);
}

TEST_F(StagedIncludesTest, NothingToStageThrows) {
    fs::create_directories(root_ / OBSERVABILITY_API_HOST_DIR);
    EXPECT_ANY_THROW(stage_observability_api(root_, cache_));
}

}  // namespace
}  // namespace tt::jit_build
