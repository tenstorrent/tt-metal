// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "jit_build/jit_build_utils.hpp"
#include "jit_build/pch.hpp"

namespace tt::jit_build {
namespace {

class CompilerVersionTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "compiler_version_XXXXXX").string();
        const auto* dir = mkdtemp(pattern.data());
        ASSERT_NE(dir, nullptr);
        root_ = dir;
        compiler_ = root_ / "compiler";
    }
    void TearDown() override { std::filesystem::remove_all(root_); }
    static void write(const std::filesystem::path& path, const std::string& text) {
        std::ofstream file(path);
        file << text;
        file.close();
        ASSERT_FALSE(file.fail());
    }
    static std::string read(const std::filesystem::path& path) {
        std::ifstream file(path);
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
    void script(const std::filesystem::path& path, const std::string& body) {
        write(path, "#!/bin/sh\n" + body);
        std::filesystem::permissions(path, std::filesystem::perms::owner_all);
    }
    std::filesystem::path root_;
    std::filesystem::path compiler_;
};

TEST_F(CompilerVersionTest, CompleteFirstLineCachedAcrossThreads) {
    const std::string version = "SFPI " + std::string(512, 'x') + "\n";
    script(compiler_, "echo probe >> \"$0.count\"\nprintf '" + version + "second line\\n'\n");
    std::vector<std::string> results(8);
    std::vector<std::thread> threads;
    threads.reserve(results.size());
    for (auto& result : results) {
        threads.emplace_back([&result, this] { result = utils::compiler_version(compiler_.string()); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& result : results) {
        EXPECT_EQ(result, version);
    }
    EXPECT_EQ(read(compiler_.string() + ".count"), "probe\n");
}

TEST_F(CompilerVersionTest, WrapperAndShellMetacharactersAreLiteralArguments) {
    const auto wrapper = root_ / "ccache";
    const auto marker = root_ / "injected";
    // Model ccache's config argument followed by the compiler command.
    script(wrapper, "[ \"$1\" = 'sloppiness=pch_defines,time_macros' ] || exit 9\nshift\nexec \"$@\"\n");
    script(compiler_, "printf '%s\\n' \"$@\"\n");
    const std::string literal = "$(touch${IFS}" + marker.string() + ")";
    EXPECT_EQ(
        utils::compiler_version(
            wrapper.string() + " sloppiness=pch_defines,time_macros " + compiler_.string() + " " + literal),
        literal + "\n");
    EXPECT_FALSE(std::filesystem::exists(marker));
}

TEST_F(CompilerVersionTest, FailedAndEmptyProbesDoNotBecomeCachedIdentities) {
    EXPECT_THROW(utils::compiler_version(""), std::runtime_error);
    EXPECT_THROW(utils::compiler_version(compiler_.string()), std::runtime_error);
    script(compiler_, "echo misleading-version\nexit 1\n");
    EXPECT_THROW(utils::compiler_version(compiler_.string()), std::runtime_error);
    script(compiler_, "echo stderr-is-not-a-version >&2\n");
    EXPECT_THROW(utils::compiler_version(compiler_.string()), std::runtime_error);
    script(compiler_, "printf 'working-version\\n'\n");
    EXPECT_EQ(utils::compiler_version(compiler_.string()), "working-version\n");
}

using CompilerVersionDeathTest = CompilerVersionTest;

TEST_F(CompilerVersionDeathTest, PchIdentityChangesAfterSamePathCompilerUpgrade) {
    const auto umbrella = root_ / "pch.h";
    write(umbrella, "#include <array>\n");
    // A recording compiler isolates cache identity from a particular installed toolchain.
    script(compiler_, R"(if [ "$1" = --version ]; then
    cat "$0.version"
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then
        shift
        printf artifact > "$1"
        exit 0
    fi
    shift
done
exit 2
)");
    write(compiler_.string() + ".version", "SFPI version one\n");
    // Each death-test child starts before this compiler's identity is memoized. The
    // parent's cache stays empty, modeling independent JIT processes across an upgrade.
    const auto build_in_child = [&](const std::string& result) {
        const auto path = ensure_pch(compiler_.string(), "O2", "", umbrella, root_ / "pch");
        if (path.empty() || !std::filesystem::exists(path + ".gch")) {
            std::_Exit(1);
        }
        write(root_ / result, path);
        std::_Exit(0);
    };
    ASSERT_EXIT(build_in_child("before"), ::testing::ExitedWithCode(0), "");
    ASSERT_EXIT(build_in_child("warm"), ::testing::ExitedWithCode(0), "");
    EXPECT_EQ(read(root_ / "before"), read(root_ / "warm"));
    write(compiler_.string() + ".version", "SFPI version two\n");
    ASSERT_EXIT(build_in_child("after"), ::testing::ExitedWithCode(0), "");
    EXPECT_NE(read(root_ / "before"), read(root_ / "after"));
    EXPECT_TRUE(std::filesystem::exists(read(root_ / "before") + ".gch"));
    EXPECT_TRUE(std::filesystem::exists(read(root_ / "after") + ".gch"));
}

}  // namespace
}  // namespace tt::jit_build
