// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_utils.hpp"

#include <algorithm>
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

namespace {

class SpawnFileActionsGuard {
public:
    SpawnFileActionsGuard() = default;
    SpawnFileActionsGuard(const SpawnFileActionsGuard&) = delete;
    SpawnFileActionsGuard& operator=(const SpawnFileActionsGuard&) = delete;

    ~SpawnFileActionsGuard() {
        close_log_fd();
        destroy_file_actions();
    }

    int init() {
        int ret = ::posix_spawn_file_actions_init(&file_actions_);
        initialized_ = ret == 0;
        return ret;
    }

    ::posix_spawn_file_actions_t* actions() { return &file_actions_; }

    int log_fd() const { return log_fd_; }
    int& mutable_log_fd() { return log_fd_; }

    void cleanup_after_spawn() {
        close_log_fd();
        destroy_file_actions();
    }

private:
    void close_log_fd() {
        if (log_fd_ >= 0) {
            ::close(log_fd_);
            log_fd_ = -1;
        }
    }

    void destroy_file_actions() {
        if (initialized_) {
            ::posix_spawn_file_actions_destroy(&file_actions_);
            initialized_ = false;
        }
    }

    ::posix_spawn_file_actions_t file_actions_{};
    int log_fd_ = -1;
    bool initialized_ = false;
};

}  // namespace

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
        ret = ::system(cmd.c_str());
    } else {
        std::string redirected_cmd = cmd + " >> " + log_file.string() + " 2>&1";
        ret = ::system(redirected_cmd.c_str());
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

bool exec_command(
    const std::vector<std::string>& args,
    const std::filesystem::path& working_dir,
    const std::filesystem::path& log_file) {
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

    SpawnFileActionsGuard spawn_guard;
    int file_actions_ret = spawn_guard.init();
    if (file_actions_ret != 0) {
        log_error(
            tt::LogBuildKernels,
            "posix_spawn_file_actions_init failed for '{}': {}",
            argv[0],
            std::strerror(file_actions_ret));
        return false;
    }

    if (!log_file.empty()) {
        const bool opened_log_file = tt::filesystem::retry_on_estale([&]() {
            errno = 0;
            spawn_guard.mutable_log_fd() = ::open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
            return spawn_guard.log_fd() >= 0;
        });
        if (!opened_log_file) {
            const int open_errno = errno;
            log_error(tt::LogBuildKernels, "open failed for log file '{}': {}", log_file, std::strerror(open_errno));
            return false;
        }
        int dup_stdout_ret =
            ::posix_spawn_file_actions_adddup2(spawn_guard.actions(), spawn_guard.log_fd(), STDOUT_FILENO);
        if (dup_stdout_ret != 0) {
            log_error(
                tt::LogBuildKernels,
                "posix_spawn_file_actions_adddup2 failed for '{}' -> stdout (log file '{}'): {}",
                argv[0],
                log_file,
                std::strerror(dup_stdout_ret));
            return false;
        }
        int dup_stderr_ret =
            ::posix_spawn_file_actions_adddup2(spawn_guard.actions(), spawn_guard.log_fd(), STDERR_FILENO);
        if (dup_stderr_ret != 0) {
            log_error(
                tt::LogBuildKernels,
                "posix_spawn_file_actions_adddup2 failed for '{}' -> stderr (log file '{}'): {}",
                argv[0],
                log_file,
                std::strerror(dup_stderr_ret));
            return false;
        }
    }

    if (!working_dir.empty()) {
        int chdir_ret = ::posix_spawn_file_actions_addchdir_np(spawn_guard.actions(), working_dir.c_str());
        if (chdir_ret != 0) {
            log_error(
                tt::LogBuildKernels,
                "posix_spawn_file_actions_addchdir_np failed for '{}' (working dir '{}'): {}",
                argv[0],
                working_dir,
                std::strerror(chdir_ret));
            return false;
        }
    }

    pid_t pid = 0;
    int spawn_ret =
        ::posix_spawnp(&pid, argv[0], spawn_guard.actions(), nullptr, const_cast<char* const*>(argv.data()), ::environ);

    spawn_guard.cleanup_after_spawn();

    if (spawn_ret != 0) {
        log_error(tt::LogBuildKernels, "posix_spawnp failed for '{}': {}", argv[0], std::strerror(spawn_ret));
        return false;
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            log_error(tt::LogBuildKernels, "waitpid failed for '{}': {}", argv[0], std::strerror(errno));
            return false;
        }
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read file: " + path);
    }
    std::streampos pos = file.tellg();
    if (pos == std::streampos(-1)) {
        throw std::runtime_error("Cannot determine size of file: " + path);
    }
    auto byte_count = static_cast<std::streamsize>(pos);
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(byte_count));
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    if (file.gcount() != byte_count || (!file && !file.eof())) {
        throw std::runtime_error(
            fmt::format("Failed to read file '{}' fully (expected {} bytes, got {})", path, byte_count, file.gcount()));
    }
    return data;
}

std::vector<tt::jit_build::GeneratedFile> read_directory_files(
    const std::filesystem::path& dir, std::span<const std::string> extensions) {
    namespace fs = std::filesystem;
    std::vector<tt::jit_build::GeneratedFile> files;
    if (!fs::is_directory(dir)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!extensions.empty() &&
            std::find(extensions.begin(), extensions.end(), entry.path().extension().string()) == extensions.end()) {
            continue;
        }
        files.push_back({entry.path().filename().string(), read_file_bytes(entry.path().string())});
    }
    return files;
}

bool create_file(const std::filesystem::path& file_path) {
    tt::filesystem::safe_create_directories(file_path.parent_path());

    std::error_code open_ec;
    return tt::filesystem::retry_on_estale_ec(
        [&](std::error_code& ec) {
            errno = 0;
            std::ofstream file(file_path);
            if (!file.is_open() || file.fail()) {
                const int open_errno = errno;
                if (open_errno != 0) {
                    ec.assign(open_errno, std::system_category());
                } else {
                    ec = std::make_error_code(std::errc::io_error);
                }
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
    std::string leaf = "." + std::to_string(unique_id_) + target_path.extension();
    filename.concat(leaf);
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
