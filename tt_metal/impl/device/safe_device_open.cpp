// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "safe_device_open.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

constexpr std::string_view kDirtyShmPrefix = "/TT_METAL_DEVICE_DIRTY.mesh-";
constexpr std::string_view kMutexNamePrefix = "tt-metal-mesh-";

// Canonical string for a set of chip IDs: sorted, joined with '-'.
// e.g. {3,0,2,1} -> "0-1-2-3"
std::string mesh_key(const std::vector<tt::ChipId>& ids) {
    std::vector<tt::ChipId> sorted(ids);
    std::sort(sorted.begin(), sorted.end());
    std::string key;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i > 0) {
            key += '-';
        }
        key += std::to_string(sorted[i]);
    }
    return key;
}

std::string mutex_name_for(const std::vector<tt::ChipId>& ids) {
    return fmt::format("{}{}", kMutexNamePrefix, mesh_key(ids));
}

std::string dirty_shm_name_for(const std::vector<tt::ChipId>& ids) {
    return fmt::format("{}{}", kDirtyShmPrefix, mesh_key(ids));
}

void verify_tt_smi_available() {
    if (std::system("which tt-smi > /dev/null 2>&1") != 0) {
        TT_THROW(
            "TT_METAL_SAFE_DEVICE_OPEN=1 requires 'tt-smi' on PATH for device recovery. "
            "Install it with: pip install tt-smi");
    }
}

}  // namespace

SafeDeviceGuard::SafeDeviceGuard(const std::vector<tt::ChipId>& device_ids)
    : device_ids_(device_ids), mutex_(mutex_name_for(device_ids)) {
    verify_tt_smi_available();
    try {
        mutex_.initialize();
        mutex_.lock();
        locked_ = true;

        dirty_ = open_dirty_shm(dirty_shm_name_for(device_ids));

        if (*dirty_.ptr != 0) {
            run_tt_smi_reset();
        }
        // Pessimistic: assume this process will dirty the mesh. Cleared in destructor on graceful
        // exit; persists if the process is killed before we get there.
        *dirty_.ptr = 1;
    } catch (...) {
        if (dirty_.ptr != nullptr) {
            *dirty_.ptr = 0;
            close_dirty_shm(dirty_);
        }
        if (locked_) {
            try {
                mutex_.unlock();
            } catch (...) {
            }
            locked_ = false;
        }
        throw;
    }
}

SafeDeviceGuard::~SafeDeviceGuard() {
    if (dirty_.ptr != nullptr) {
        // If on_hang() fired, leave dirty=1 so the next process double-checks via tt-smi -r.
        // Otherwise this was a graceful run — clear the bit.
        if (!hang_handled_.load(std::memory_order_acquire)) {
            *dirty_.ptr = 0;
        }
        close_dirty_shm(dirty_);
    }
    if (locked_) {
        try {
            mutex_.unlock();
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "SafeDeviceGuard: mesh unlock failed: {}", e.what());
        }
        locked_ = false;
    }
}

void SafeDeviceGuard::on_hang() {
    bool expected = false;
    if (!hang_handled_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;  // already handled
    }
    // Mark dirty before the reset so a crash mid-reset still leaves the next acquirer with a
    // definitive "must reset" signal.
    if (dirty_.ptr != nullptr) {
        *dirty_.ptr = 1;
    }
    run_tt_smi_reset();
}

SafeDeviceGuard::DirtyShm SafeDeviceGuard::open_dirty_shm(std::string_view shm_name) {
    DirtyShm s;
    const std::string name(shm_name);

    auto old_umask = umask(0);
    s.fd = shm_open(name.c_str(), O_RDWR | O_CREAT, 0666);
    umask(old_umask);

    if (s.fd == -1) {
        TT_THROW("shm_open failed for {} errno: {}", name, errno);
    }

    struct stat sb {};
    if (fstat(s.fd, &sb) != 0) {
        const int e = errno;
        ::close(s.fd);
        s.fd = -1;
        TT_THROW("fstat failed for {} errno: {}", name, e);
    }
    if (sb.st_size < static_cast<off_t>(sizeof(uint8_t))) {
        if (ftruncate(s.fd, sizeof(uint8_t)) != 0) {
            const int e = errno;
            ::close(s.fd);
            s.fd = -1;
            TT_THROW("ftruncate failed for {} errno: {}", name, e);
        }
    }

    void* addr = mmap(nullptr, sizeof(uint8_t), PROT_READ | PROT_WRITE, MAP_SHARED, s.fd, 0);
    if (addr == MAP_FAILED) {
        const int e = errno;
        ::close(s.fd);
        s.fd = -1;
        TT_THROW("mmap failed for {} errno: {}", name, e);
    }
    s.ptr = static_cast<uint8_t*>(addr);
    return s;
}

void SafeDeviceGuard::close_dirty_shm(DirtyShm& s) {
    if (s.ptr != nullptr) {
        if (munmap(s.ptr, sizeof(uint8_t)) != 0) {
            log_warning(tt::LogMetal, "munmap failed for dirty shm errno: {}", errno);
        }
        s.ptr = nullptr;
    }
    if (s.fd != -1) {
        if (::close(s.fd) != 0) {
            log_warning(tt::LogMetal, "close failed for dirty shm errno: {}", errno);
        }
        s.fd = -1;
    }
}

void SafeDeviceGuard::run_tt_smi_reset() {
    log_warning(tt::LogMetal, "SafeDeviceGuard: mesh dirty, running: tt-smi -r");
    int rc = std::system("tt-smi -r");
    if (rc != 0) {
        log_warning(
            tt::LogMetal,
            "SafeDeviceGuard: 'tt-smi -r' returned non-zero exit code: {} (is tt-smi on PATH?)",
            WEXITSTATUS(rc));
    } else {
        log_info(tt::LogMetal, "SafeDeviceGuard: mesh reset complete");
    }
}

}  // namespace tt::tt_metal
