// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>

#ifdef __linux__
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <tt-logger/tt-logger.hpp>
#include "common/filesystem_utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::jit_build::utils {

#ifdef __linux__
// Spawn /bin/sh -c <cmd> via posix_spawnp.
//
// Unlike std::system(), posix_spawn uses vfork+exec on Linux (glibc) and
// does NOT trigger pthread_atfork handlers.  This avoids the Executor
// being torn down and recreated on every JIT compile/link invocation.
//
// The child is placed in its own process group (POSIX_SPAWN_SETPGROUP)
// so that the entire subprocess tree can be cleaned up with a single
// kill(-pgid, SIGKILL) if the parent is interrupted.
static int spawn_shell(const std::string& cmd, const std::filesystem::path& log_file, bool verbose) {
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t file_actions;
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_init(&file_actions);

    // Put child in its own process group for clean teardown.
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);

    int log_fd = -1;
    if (!verbose) {
        log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (log_fd >= 0) {
            posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDOUT_FILENO);
            posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDERR_FILENO);
            posix_spawn_file_actions_addclose(&file_actions, log_fd);
        }
    }

    const char* argv[] = {"/bin/sh", "-c", cmd.c_str(), nullptr};

    pid_t pid = -1;
    int err = posix_spawnp(&pid, "/bin/sh", &file_actions, &attr, const_cast<char**>(argv), environ);

    if (log_fd >= 0) {
        close(log_fd);
    }

    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&file_actions);

    if (err != 0) {
        return -1;
    }

    int status = 0;
    while (waitpid(pid, &status, 0) == -1) {
        if (errno != EINTR) {
            return -1;
        }
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}
#endif  // __linux__

bool run_command(const std::string& cmd, const std::filesystem::path& log_file, bool verbose) {
    static std::mutex io_mutex;

    if (verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
    }

#ifdef __linux__
    return spawn_shell(cmd, log_file, verbose) == 0;
#else
    std::string full_cmd = verbose ? cmd : (cmd + " >> " + log_file.string() + " 2>&1");
    return std::system(full_cmd.c_str()) == 0;
#endif
}

void create_file(const std::string& file_path_str) {
    namespace fs = std::filesystem;

    fs::path file_path(file_path_str);
    tt::filesystem::safe_create_directories(file_path.parent_path());

    std::ofstream ofs(file_path);
    ofs.close();
}

uint64_t FileRenamer::unique_id_ = []() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> distr;
    return distr(rd);
}();

std::filesystem::path FileRenamer::generate_temp_path(const std::filesystem::path& target_path) {
    std::filesystem::path path(target_path);
    if (path.has_extension()) {
        path.replace_extension(fmt::format("{}{}", unique_id_, path.extension().string()));
        return path;
    }
    return std::filesystem::path(fmt::format("{}.{}", target_path.string(), unique_id_));
}

FileRenamer::FileRenamer(const std::filesystem::path& target_path) :
    temp_path_(generate_temp_path(target_path)), target_path_(target_path) {}

FileRenamer::~FileRenamer() {
    if (target_path_.empty()) {
        return;
    }
    if (!tt::filesystem::safe_rename(temp_path_, target_path_)) {
        log_error(
            tt::LogBuildKernels, "Failed to rename temporary file {} to target file {}", temp_path_, target_path_);
    }
}

}  // namespace tt::jit_build::utils
