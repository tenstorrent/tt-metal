// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tt::tt_metal::detail {
// Opt-in: unchanged defaults for other workloads.
inline unsigned host_executor_threads() {
    const unsigned available = std::max(1u, std::thread::hardware_concurrency());
    const char* value = std::getenv("TT_METAL_EXECUTOR_THREADS");
    if (!value) {
        return available;
    }
    unsigned count = 0;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || count > 4096) {
            throw std::runtime_error("TT_METAL_EXECUTOR_THREADS must be an integer in [1, 4096]");
        }
        count = count * 10 + (*p - '0');
    }
    if (count == 0 || count > 4096) {
        throw std::runtime_error("TT_METAL_EXECUTOR_THREADS must be an integer in [1, 4096]");
    }
    return std::min(count, available);
}

// One optional host-local lock shared by all MPI ranks. Hold only around a
// compiler subprocess, never around tasks that may wait for another build task.
class HostJitCommandLock {
    int fd_ = -1;

public:
    HostJitCommandLock() {
        const char* path = std::getenv("TT_METAL_JIT_LOCK_PATH");
        if (!path) {
            return;
        }
        if (path[0] != '/') {
            throw std::runtime_error("TT_METAL_JIT_LOCK_PATH must be absolute");
        }
        fd_ = open(path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open TT_METAL_JIT_LOCK_PATH");
        }
        struct stat st{};
        if (fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid()) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("JIT lock must be a regular file owned by this user");
        }
        while (flock(fd_, LOCK_EX) != 0) {
            if (errno == EINTR) {
                continue;
            }
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("Cannot acquire TT_METAL_JIT_LOCK_PATH");
        }
    }
    ~HostJitCommandLock() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    HostJitCommandLock(const HostJitCommandLock&) = delete;
    HostJitCommandLock& operator=(const HostJitCommandLock&) = delete;
};
}  // namespace tt::tt_metal::detail
