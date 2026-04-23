// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "common/filesystem_utils.hpp"
#include "jit_build/jit_build_utils.hpp"

namespace {

namespace fs = std::filesystem;

class ExecCommandTest : public ::testing::Test {
protected:
    void SetUp() override {
        tt::filesystem::set_nfs_safety(false);
        auto temp_template = (fs::temp_directory_path() / "jit_build_utils_test_XXXXXX").string();
        auto* temp_dir = ::mkdtemp(temp_template.data());
        ASSERT_NE(temp_dir, nullptr);
        temp_dir_ = fs::path(temp_dir);
    }

    void TearDown() override {
        tt::filesystem::set_nfs_safety(false);
        std::error_code ec;
        fs::remove_all(temp_dir_, ec);
    }

    static std::string read_text(const fs::path& path) {
        std::ifstream file(path);
        EXPECT_TRUE(file.is_open()) << path;
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    static std::string trim_newlines(std::string text) {
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        return text;
    }

    static void expect_equivalent_path(const std::string& actual_path, const fs::path& expected_path) {
        ASSERT_FALSE(actual_path.empty());
        std::error_code ec;
        EXPECT_TRUE(fs::equivalent(fs::path(actual_path), expected_path, ec));
        EXPECT_FALSE(ec) << ec.message();
    }

    fs::path temp_dir_;
};

TEST_F(ExecCommandTest, EmptyArgsReturnsFalse) { EXPECT_FALSE(tt::jit_build::utils::exec_command({}, temp_dir_, {})); }

TEST_F(ExecCommandTest, NonEmptyWorkingDirChangesChildCwd) {
    fs::path working_dir = temp_dir_ / "child_cwd";
    ASSERT_TRUE(fs::create_directories(working_dir));

    fs::path log_file = temp_dir_ / "child_cwd.log";
    ASSERT_TRUE(tt::jit_build::utils::exec_command({"/bin/pwd"}, working_dir, log_file));

    expect_equivalent_path(trim_newlines(read_text(log_file)), working_dir);
}

TEST_F(ExecCommandTest, EmptyWorkingDirInheritsParentCwd) {
    fs::path log_file = temp_dir_ / "inherit_cwd.log";
    ASSERT_TRUE(tt::jit_build::utils::exec_command({"/bin/pwd"}, {}, log_file));

    expect_equivalent_path(trim_newlines(read_text(log_file)), fs::current_path());
}

TEST_F(ExecCommandTest, LogFileCapturesStdoutAndStderrAndAppends) {
    fs::path log_file = temp_dir_ / "exec_command.log";
    {
        std::ofstream file(log_file);
        ASSERT_TRUE(file.is_open());
        file << "prefix\n";
    }

    ASSERT_TRUE(tt::jit_build::utils::exec_command(
        {"/bin/sh", "-c", R"(printf 'stdout_one\n'; printf 'stderr_one\n' >&2)"}, {}, log_file));
    ASSERT_TRUE(tt::jit_build::utils::exec_command({"/bin/sh", "-c", R"(printf 'stdout_two\n')"}, {}, log_file));

    const std::string contents = read_text(log_file);
    ASSERT_GE(contents.size(), std::string("prefix\n").size());
    EXPECT_EQ(contents.find("prefix\n"), 0U);
    EXPECT_NE(contents.find("stdout_one\n"), std::string::npos);
    EXPECT_NE(contents.find("stderr_one\n"), std::string::npos);
    EXPECT_NE(contents.find("stdout_two\n"), std::string::npos);
}

TEST_F(ExecCommandTest, MissingExecutableReturnsFalse) {
    EXPECT_FALSE(
        tt::jit_build::utils::exec_command({"__definitely_missing_exec_command_test_binary__"}, temp_dir_, {}));
}

TEST_F(ExecCommandTest, InvalidLogPathReturnsFalse) {
    fs::path invalid_log_path = temp_dir_ / "missing_parent" / "exec.log";
    EXPECT_FALSE(tt::jit_build::utils::exec_command({"/bin/pwd"}, {}, invalid_log_path));
    EXPECT_FALSE(fs::exists(invalid_log_path));
}

}  // namespace
