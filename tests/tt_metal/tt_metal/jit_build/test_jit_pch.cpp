// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/pch.hpp"
#include "jit_test_tools.hpp"

namespace tt::jit_build {

// Compile writable headers with SFPI and verify PCH consumption without a device.
class JitPchTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "jit_pch_test_XXXXXX").string();
        const char* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        scratch_ = directory;
        source_ = scratch_ / "source";
        consumer_ = scratch_ / "consumer";
        std::filesystem::create_directories(source_);
        std::filesystem::create_directories(consumer_);

        compiler_ = test::JitTestTools{}.compiler();
        write(source_ / "umbrella.h", "#pragma once\n#include \"value.h\"\n");
        write(source_ / "value.h", "#pragma once\nconstexpr int value = 1;\n");
    }

    void TearDown() override {
        pch_cache_clear();
        clear_file_hash_cache();
        if (!scratch_.empty()) {
            std::filesystem::remove_all(scratch_);
        }
    }

    static std::string read(const std::filesystem::path& path) {
        std::ifstream file(path);
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }

    static void write(const std::filesystem::path& path, const std::string& content) {
        std::ofstream file(path);
        file << content;
        file.close();
        ASSERT_FALSE(file.fail()) << path;
    }

    std::string includes() const { return "-I. -I.. -I" + source_.string() + extra_includes_; }

    std::string ensure(const std::string& cache = "cache") const {
        return ensure_pch(
            compiler_,
            source_.string(),
            (scratch_ / cache).string(),
            "blackhole",
            "umbrella.h",
            "O0",
            "-mcpu=tt-bh-tensix -std=c++17 -MMD",
            includes(),
            {"-DFORCE_INLINE=inline"});
    }

    void consume(
        const std::string& pch, const std::string& body, const std::vector<std::string>& extra_defines = {}) const {
        const auto source = consumer_ / "consumer.cpp";
        const std::string object = (consumer_ / "consumer.o").string();
        const std::string dep = (consumer_ / "consumer.d").string();
        const std::string log = (consumer_ / "compile.log").string();
        ASSERT_NO_FATAL_FAILURE(write(source, body));
        std::filesystem::remove(log);
        std::vector<std::string> defines = {"-DFORCE_INLINE=inline"};
        if (!pch.empty()) {
            defines.insert(defines.end(), {"-include", pch});
        }
        defines.insert(defines.end(), extra_defines.begin(), extra_defines.end());
        const auto args = utils::build_gpp_argv(
            compiler_,
            "O0",
            "-mcpu=tt-bh-tensix -std=c++17 -MMD -H -Werror=invalid-pch",
            includes(),
            defines,
            source.string(),
            utils::GppAction::Compile,
            object,
            dep);
        ASSERT_TRUE(utils::exec_command(args, consumer_.string(), log)) << read(log);
        if (!pch.empty()) {
            ASSERT_NE(read(log).find("! " + pch + ".gch"), std::string::npos) << read(log);
            merge_pch_deps_into_kernel_d(dep, object, pch + ".d");
        }
        write_dependency_hashes(consumer_.string(), object, object + ".dephash");
        ASSERT_TRUE(dependencies_up_to_date_file(object + ".dephash"));
    }

    bool consumer_up_to_date() const {
        return dependencies_up_to_date_file((consumer_ / "consumer.o.dephash").string());
    }

    void use_named_api() {
        extra_includes_ = " -I" + test::JitTestTools{}.hw_include_dir();
        write(source_ / "umbrella.h", "#include \"api/compile_time_args.h\"\n");
    }

    std::filesystem::path scratch_;
    std::filesystem::path source_;
    std::filesystem::path consumer_;
    std::string compiler_;
    std::string extra_includes_;
};

TEST_F(JitPchTest, ReusesUnchangedPchAndMergesItsDependencies) {
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1);\n"));
    const auto sentinel = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24);
    std::filesystem::last_write_time(pch + ".gch", sentinel);
    EXPECT_EQ(ensure(), pch);
    EXPECT_EQ(std::filesystem::last_write_time(pch + ".gch"), sentinel);
    const std::string hashes = read(consumer_ / "consumer.o.dephash");
    EXPECT_NE(hashes.find((source_ / "value.h").string()), std::string::npos);
    EXPECT_NE(hashes.find((source_ / "umbrella.h").string()), std::string::npos);
}

TEST_F(JitPchTest, RebuildsAfterTransitiveHeaderEdit) {
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1);\n"));
    write(source_ / "value.h", "#pragma once\nconstexpr int value = 22;\n");
    ASSERT_FALSE(consumer_up_to_date());
    ASSERT_EQ(ensure(), pch);
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 22);\n"));
}

TEST_F(JitPchTest, RebuildsAfterOriginalUmbrellaEdit) {
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1);\n"));
    write(source_ / "umbrella.h", "#pragma once\n#include \"value.h\"\nconstexpr int added = 21;\n");
    ASSERT_FALSE(consumer_up_to_date());
    ASSERT_EQ(ensure(), pch);
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1 && added == 21);\n"));
}

TEST_F(JitPchTest, SeparatesOutputRootsInOneProcess) {
    const auto first = ensure("first");
    const auto second = ensure("second");
    ASSERT_FALSE(first.empty());
    ASSERT_FALSE(second.empty());
    ASSERT_NE(first, second);
    ASSERT_NO_FATAL_FAILURE(consume(first, "static_assert(value == 1);\n"));
    ASSERT_NO_FATAL_FAILURE(consume(second, "static_assert(value == 1);\n"));
}

TEST_F(JitPchTest, CacheClearDropsCompletedEntries) {
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    std::filesystem::remove(pch + ".gch");
    pch_cache_clear();
    clear_file_hash_cache();
    ASSERT_EQ(ensure(), pch);
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1);\n"));
}

TEST_F(JitPchTest, FailedPchBuildRequestsTextualFallback) {
    write(source_ / "umbrella.h", "#include \"missing_generated_header.h\"\n");
    EXPECT_TRUE(ensure().empty());
    ASSERT_NO_FATAL_FAILURE(consume({}, "static_assert(1 + 1 == 2);\n"));
}

TEST_F(JitPchTest, StrictValidationRequiresTheExpectedPch) {
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    const auto log = (consumer_ / "compile.log").string();
    ASSERT_NO_FATAL_FAILURE(consume({}, "static_assert(1 + 1 == 2);\n"));
    EXPECT_THROW(require_pch_consumed(log, pch), std::runtime_error);
    EXPECT_THROW(require_pch_consumed(log, {}), std::runtime_error);
    ASSERT_NO_FATAL_FAILURE(consume(pch, "static_assert(value == 1);\n"));
    EXPECT_NO_THROW(require_pch_consumed(log, pch));
    EXPECT_THROW(require_pch_consumed(log, pch + ".other"), std::runtime_error);
    write(log, "x " + pch + ".gch\n");
    EXPECT_THROW(require_pch_consumed(log, pch), std::runtime_error);
    std::filesystem::remove(log);
    EXPECT_THROW(require_pch_consumed(log, pch), std::runtime_error);
}

TEST_F(JitPchTest, DistinctNamedMapsShareAnAcceptedPch) {
    ASSERT_NO_FATAL_FAILURE(use_named_api());
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    for (const unsigned value : {7u, 21u}) {
        const auto map_header = consumer_ / "named.h";
        write(map_header, utils::format_named_ct_arg_map_header({{"n", value}}));
        ASSERT_EQ(ensure(), pch);
        ASSERT_NO_FATAL_FAILURE(consume(
            pch,
            "#include \"api/compile_time_args.h\"\n#include \"api/named_compile_time_args.h\"\n"
            "static_assert(get_named_compile_time_arg_val(\"n\") == " +
                std::to_string(value) + ");\n",
            {"-include", map_header.string()}));
    }
}

TEST_F(JitPchTest, PublicIncludeExposesNamedApiWithAndWithoutPch) {
    ASSERT_NO_FATAL_FAILURE(use_named_api());
    const std::string body =
        "#define KERNEL_COMPILE_TIME_ARG_MAP {\"n\",7}\n"
        "#include \"api/compile_time_args.h\"\n"
        "static_assert(get_named_compile_time_arg_val(\"n\") == 7);\n";
    ASSERT_NO_FATAL_FAILURE(consume({}, body));
    const auto pch = ensure();
    ASSERT_FALSE(pch.empty());
    ASSERT_NO_FATAL_FAILURE(consume(pch, body));
}

}  // namespace tt::jit_build
