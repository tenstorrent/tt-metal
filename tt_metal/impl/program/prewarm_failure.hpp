// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-only prewarm execution policy. Kept independent of MetalContext so CPU tests exercise
// exactly the subprocess, accounting and persistent-failure path used by the offline tool.
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace tt::tt_metal::kernel_prewarm::failure {

inline constexpr size_t diagnostic_half = 8192;

struct CommandResult {
    size_t limit = diagnostic_half;
    int exit_code = -1;
    bool interrupted = false;
    bool timed_out = false;
    bool semantic_error = false;
    bool infrastructure_error = false;
    size_t bytes = 0;
    std::string head;
    std::string tail;
    std::string scan_tail;

    void consume(std::string_view chunk) {
        bytes += chunk.size();
        const size_t take = std::min(limit - head.size(), chunk.size());
        head.append(chunk.substr(0, take));
        tail.append(chunk.substr(take));
        if (tail.size() > limit) {
            tail.erase(0, tail.size() - limit);
        }
        // Scan ALL bytes, including dropped output, with overlap for split diagnostics.
        std::string scan = scan_tail + std::string(chunk);
        semantic_error |= scan.find(": error:") != std::string::npos;
        for (const char* text : {"fatal error:", "internal compiler error", "No space left", "Permission denied",
                                 "Cannot allocate memory", "cannot execute", "Killed signal", "Input/output error"}) {
            infrastructure_error |= scan.find(text) != std::string::npos;
        }
        scan_tail = scan.substr(scan.size() > 128 ? scan.size() - 128 : 0);
    }

    bool success() const { return exit_code == 0 && !interrupted && !timed_out; }
    bool definite_compile_failure() const {
        return exit_code == 1 && !interrupted && !timed_out && semantic_error && !infrastructure_error;
    }
    std::string diagnostic() const {
        return head + (bytes > head.size() + tail.size() ? "\n[compiler output truncated]\n" : "") + tail;
    }
};

// Never spool compiler output to disk. Drain a nonblocking pipe into a fixed head/tail window.
// A private process group lets the timeout reap this invocation and its children only.
inline CommandResult command(
    const std::vector<std::string>& args,
    const std::string& cwd,
    std::chrono::milliseconds timeout = std::chrono::minutes(5),
    size_t half_limit = diagnostic_half) {
    CommandResult result;
    result.limit = half_limit;
    if (args.empty()) {
        return result;
    }
    int pipes[2];
    if (::pipe2(pipes, O_CLOEXEC) != 0) {
        return result;
    }
    std::vector<char*> argv;
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    int error = posix_spawn_file_actions_init(&actions);
    if (error != 0) {
        ::close(pipes[0]);
        ::close(pipes[1]);
        return result;
    }
    error = posix_spawnattr_init(&attr);
    if (error == 0) {
        error = posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO);
        if (!error) error = posix_spawn_file_actions_adddup2(&actions, pipes[1], STDERR_FILENO);
        if (!error) error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        if (!error && !cwd.empty()) error = posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
        if (!error) error = posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        if (!error) error = posix_spawnattr_setpgroup(&attr, 0);
        pid_t pid = -1;
        if (!error) error = posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), ::environ);
        posix_spawnattr_destroy(&attr);
        posix_spawn_file_actions_destroy(&actions);
        ::close(pipes[1]);
        if (error) {
            ::close(pipes[0]);
            return result;
        }
        ::fcntl(pipes[0], F_SETFL, O_NONBLOCK);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        bool eof = false;
        bool reaped = false;
        int status = 0;
        while (!eof || !reaped) {
            if (std::chrono::steady_clock::now() >= deadline) {
                result.timed_out = true;
                ::kill(-pid, SIGKILL);
                break;
            }
            std::array<char, 4096> buffer{};
            const auto n = ::read(pipes[0], buffer.data(), buffer.size());
            if (n > 0) {
                result.consume(std::string_view(buffer.data(), n));
            } else if (n == 0) {
                eof = true;
            } else if (errno != EAGAIN && errno != EINTR) {
                result.interrupted = true;
                ::kill(-pid, SIGKILL);
                break;
            }
            if (!reaped) {
                const auto waited = ::waitpid(pid, &status, WNOHANG);
                reaped = waited == pid;
                if (waited < 0 && errno != EINTR) {
                    result.interrupted = true;
                    break;
                }
            }
            if (n <= 0 && (!eof || !reaped)) {
                pollfd fd{eof ? -1 : pipes[0], POLLIN, 0};
                ::poll(&fd, 1, 10);
            }
        }
        ::close(pipes[0]);
        if (!reaped) {
            while (::waitpid(pid, &status, 0) < 0) {
                if (errno != EINTR) {
                    result.interrupted = true;
                    break;
                }
            }
        }
        if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
        if (WIFSIGNALED(status)) result.interrupted = true;
        return result;
    }
    posix_spawn_file_actions_destroy(&actions);
    ::close(pipes[0]);
    ::close(pipes[1]);
    return result;
}

class CompileFailure : public std::runtime_error {
public:
    bool definite;
    explicit CompileFailure(const CommandResult& result, bool compiling) :
        std::runtime_error(
            "exit=" + std::to_string(result.exit_code) + " interrupted=" + std::to_string(result.interrupted) +
            " timeout=" + std::to_string(result.timed_out) + " bytes=" + std::to_string(result.bytes) + "\n" +
            result.diagnostic()),
        definite(compiling && result.definite_compile_failure()) {}
};

// The kernel lock lives outside the target directory that a failed build removes. Nonblocking:
// another prewarm owns this kernel, so defer instead of waiting while holding a pool thread.
class KernelLock {
    int fd_ = -1;
public:
    explicit KernelLock(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("cannot open prewarm lock");
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int error = errno;
            ::close(fd_);
            fd_ = -1;
            if (error != EWOULDBLOCK && error != EAGAIN) throw std::runtime_error("cannot lock prewarm kernel");
        }
    }
    bool acquired() const { return fd_ >= 0; }
    ~KernelLock() { if (fd_ >= 0) ::close(fd_); }
    KernelLock(const KernelLock&) = delete;
    KernelLock& operator=(const KernelLock&) = delete;
};

inline void save(const std::filesystem::path& path, const std::string& context) {
    std::string name = path.string() + ".XXXXXX";
    const int fd = ::mkstemp(name.data());
    if (fd < 0) throw std::runtime_error("cannot create prewarm failure state");
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    const std::string data = "prewarm-failure-v1 " + context + "\n";
    size_t offset = 0;
    bool ok = true;
    while (offset < data.size()) {
        const auto n = ::write(fd, data.data() + offset, data.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { ok = false; break; }
        offset += n;
    }
    if (::fsync(fd) != 0) ok = false;
    if (::close(fd) != 0) ok = false;
    if (ok && ::rename(name.c_str(), path.c_str()) == 0) return;
    ::unlink(name.c_str());
    throw std::runtime_error("cannot publish prewarm failure state");
}

inline bool matches(const std::filesystem::path& path, const std::string& context) {
    std::ifstream input(path, std::ios::binary);
    const std::string expected = "prewarm-failure-v1 " + context + "\n";
    std::string contents(expected.size() + 1, '\0');
    input.read(contents.data(), contents.size());
    contents.resize(input.gcount());
    return contents == expected;
}

enum class Outcome { Successful, Failed, PreviouslyFailed };
struct TargetResult {
    Outcome outcome;
    std::string diagnostic;
};

// Caller holds KernelLock through generated-file writes and all target jobs. No manifest edit.
// Missing/incomplete context, state I/O errors and infrastructure faults always remain retryable.
inline TargetResult target(
    const std::filesystem::path& state,
    bool retry,
    const std::function<std::optional<std::string>()>& fingerprint,
    const std::function<void()>& build,
    const std::function<void()>& cleanup) {
    std::optional<std::string> before;
    try { before = fingerprint(); } catch (const std::exception&) { }
    if (!retry && before && matches(state, *before)) return {Outcome::PreviouslyFailed, {}};
    std::error_code ec;
    std::filesystem::remove(state, ec);
    try {
        build();
        return {Outcome::Successful, {}};
    } catch (const std::exception& e) {
        const std::string detail = std::string(e.what()).substr(0, 2 * diagnostic_half + 256);
        bool persist = false;
        const auto* failure = dynamic_cast<const CompileFailure*>(&e);
        if (failure != nullptr && failure->definite && before) {
            try { persist = fingerprint() == before; } catch (const std::exception&) { }
        }
        std::string state_error;
        try { cleanup(); } catch (const std::exception&) { state_error = " (cleanup failed)"; }
        if (persist && state_error.empty()) {
            try { save(state, *before); } catch (const std::exception&) { state_error = " (state not saved; will retry)"; }
        }
        return {Outcome::Failed, detail + state_error};
    }
}

}  // namespace tt::tt_metal::kernel_prewarm::failure
