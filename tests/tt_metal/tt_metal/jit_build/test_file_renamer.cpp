// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "jit_build/jit_build_utils.hpp"

namespace tt::jit_build::utils {
namespace {

// FileRenamer::unique_id_ is initialized once per process, so a fork()ed child
// inherits the parent's value. Without mixing in the pid, parent and child derive
// byte-identical temp paths -- and because JitBuildState::compile() removes those
// temp objects once linking finishes, two processes sharing a temp path can delete
// it out from under each other's LTO link.
//
// This matters in practice for any harness that forks around a shared kernel cache
// (e.g. pytest --forked), where it showed up as
//   "Failed to generate binaries for <kernel> ... cannot copy file: File exists".
//
// Run the child's generation in a fork so the inherited-id path is what is
// exercised; the id must already be initialized in the parent beforehand.
std::string generate_in_forked_child(const std::filesystem::path& target) {
    int fds[2];
    if (pipe(fds) != 0) {
        return {};
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return {};
    }
    if (pid == 0) {
        close(fds[0]);
        const std::string path = FileRenamer::generate_temp_path(target);
        const ssize_t written = write(fds[1], path.data(), path.size());
        close(fds[1]);
        _exit(written == static_cast<ssize_t>(path.size()) ? 0 : 1);
    }
    close(fds[1]);
    std::string out;
    char buf[1024];
    ssize_t n = 0;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return out;
}

}  // namespace

TEST(FileRenamerForkSafety, ForkedChildDoesNotReuseParentTempPath) {
    const std::filesystem::path target = "trisck.o";

    // Initialize unique_id_ in the parent *before* forking, so the child inherits it.
    const std::string parent_path = FileRenamer::generate_temp_path(target);
    ASSERT_FALSE(parent_path.empty());

    const std::string child_path = generate_in_forked_child(target);
    ASSERT_FALSE(child_path.empty()) << "forked child produced no temp path";

    EXPECT_NE(parent_path, child_path)
        << "forked child reused the parent's temp path (" << parent_path
        << "); two processes would then contend for, and remove, the same temp object";
}

TEST(FileRenamerForkSafety, TempPathIsStableWithinAProcess) {
    // Same process must keep deriving the same temp path, otherwise the
    // write-temp-then-rename pattern in JitBuildState::compile() breaks.
    const std::filesystem::path target = "trisck.o";
    EXPECT_EQ(FileRenamer::generate_temp_path(target), FileRenamer::generate_temp_path(target));
}

TEST(FileRenamerForkSafety, TempPathKeepsExtensionAndDiffersFromTarget) {
    const std::filesystem::path with_ext = "trisck.o";
    const std::string temp = FileRenamer::generate_temp_path(with_ext);
    EXPECT_NE(temp, with_ext.string());
    EXPECT_TRUE(temp.ends_with(".o")) << temp;

    const std::filesystem::path no_ext = "trisck";
    const std::string temp_no_ext = FileRenamer::generate_temp_path(no_ext);
    EXPECT_NE(temp_no_ext, no_ext.string());
    EXPECT_TRUE(temp_no_ext.starts_with("trisck.")) << temp_no_ext;
}

TEST(FileRenamerForkSafety, RenamesTempToTargetOnDestruction) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("file_renamer_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path target = dir / "out.o";

    {
        FileRenamer renamer(target.string());
        EXPECT_NE(renamer.path(), target.string());
        std::ofstream out(renamer.path());
        out << "payload";
    }

    EXPECT_TRUE(std::filesystem::exists(target)) << "temp file was not renamed onto the target";
    std::filesystem::remove_all(dir);
}

}  // namespace tt::jit_build::utils
