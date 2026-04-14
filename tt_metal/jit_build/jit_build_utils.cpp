// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/fmt.hpp>
#include "common/filesystem_utils.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::filesystem::path& log_file, bool verbose) {
    // ZoneScoped;
    // ZoneText( cmd.c_str(), cmd.length());
    int ret;
    static std::mutex io_mutex;

    if (verbose) {
        {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "===== RUNNING SYSTEM COMMAND:\n";
            std::cout << cmd << "\n" << std::endl;
        }
        ret = system(cmd.c_str());
    } else {
        std::string redirected_cmd = cmd + " >> " + log_file.string() + " 2>&1";
        ret = system(redirected_cmd.c_str());
    }

    return (ret == 0);
}

std::vector<std::string> tokenize_flags(const std::string& flags) {
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < flags.size()) {
        while (i < flags.size() && std::isspace(static_cast<unsigned char>(flags[i]))) {
            ++i;
        }
        if (i >= flags.size()) {
            break;
        }
        std::size_t start = i;
        while (i < flags.size() && !std::isspace(static_cast<unsigned char>(flags[i]))) {
            ++i;
        }
        tokens.emplace_back(flags, start, i - start);
    }
    return tokens;
}

bool exec_command(const std::vector<std::string>& args, const std::string& working_dir, const std::string& log_file) {
    if (args.empty()) {
        return false;
    }

    // Build a null-terminated argv array for posix_spawn.
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    int log_fd = -1;
    if (!log_file.empty()) {
        log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (log_fd < 0) {
            posix_spawn_file_actions_destroy(&file_actions);
            return false;
        }
        posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&file_actions, log_fd, STDERR_FILENO);
    }

    if (!working_dir.empty()) {
        posix_spawn_file_actions_addchdir_np(&file_actions, working_dir.c_str());
    }

    pid_t pid = 0;
    int spawn_ret =
        posix_spawnp(&pid, argv[0], &file_actions, nullptr, const_cast<char* const*>(argv.data()), ::environ);

    if (log_fd >= 0) {
        close(log_fd);
    }
    posix_spawn_file_actions_destroy(&file_actions);

    if (spawn_ret != 0) {
        log_error(tt::LogBuildKernels, "posix_spawnp failed for '{}': {}", argv[0], std::strerror(spawn_ret));
        return false;
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            log_error(tt::LogBuildKernels, "waitpid failed for '{}': {}", argv[0], std::strerror(errno));
            return false;
        }
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

void create_file(const std::filesystem::path& file_path) {
    tt::filesystem::safe_create_directories(file_path.parent_path());

    // just making sure the file is there. Don't need to worry about the state
    [[maybe_unused]] std::error_code open_ec;
    [[maybe_unused]] auto _ = tt::filesystem::retry_on_estale_ec(
        [&](std::error_code& ec) {
            std::ofstream file(file_path);
            if (!file.is_open() || file.fail()) {
                ec.assign(errno, std::system_category());
                return false;
            }
            file.close();
            return true;
        },
        open_ec);
}

uint64_t FileRenamer::unique_id_ = []() {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> distr;
    return distr(rd);
}();

std::filesystem::path FileRenamer::generate_temp_path(const std::filesystem::path& target_path) {
    // stem() gives you the filename without the last extension, and extension() is empty when
    // there isn't one, so this covers both cases:
    // foo.txt -> foo.42.txt
    // foo -> foo.42

    std::filesystem::path filename = target_path.stem();
    filename += ".";
    filename += std::to_string(unique_id_);
    filename += target_path.extension();
    return target_path.parent_path() / filename;
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
