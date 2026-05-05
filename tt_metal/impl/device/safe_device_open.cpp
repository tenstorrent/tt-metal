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

constexpr const char* kDirtyShmPrefix = "TT_METAL_DEVICE_DIRTY.";
constexpr const char* kMutexNamePrefix = "tt-metal-device-";

std::string mutex_name_for(tt::ChipId id) { return fmt::format("{}{}", kMutexNamePrefix, id); }

std::string dirty_shm_name_for(tt::ChipId id) { return fmt::format("/{}{}", kDirtyShmPrefix, id); }

}  // namespace

SafeDeviceGuard::PerDevice::PerDevice(tt::ChipId id_in) : id(id_in), mutex(mutex_name_for(id_in)) {}

SafeDeviceGuard::DirtyShm SafeDeviceGuard::open_dirty_shm(tt::ChipId id) {
    DirtyShm s;
    const std::string name = dirty_shm_name_for(id);

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

void SafeDeviceGuard::run_tt_smi_reset(tt::ChipId id) {
    // tt-smi -r <int> resets by UMD logical ID — same id space tt-metal's Cluster exposes.
    const std::string cmd = fmt::format("tt-smi -r {}", id);
    log_warning(tt::LogMetal, "SafeDeviceGuard: device {} dirty, running: {}", id, cmd);
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        log_warning(
            tt::LogMetal,
            "SafeDeviceGuard: '{}' returned non-zero exit code: {} (is tt-smi on PATH?)",
            cmd,
            WEXITSTATUS(rc));
    } else {
        log_info(tt::LogMetal, "SafeDeviceGuard: device {} reset complete", id);
    }
}

SafeDeviceGuard::SafeDeviceGuard(const std::vector<tt::ChipId>& device_ids) {
    per_device_.reserve(device_ids.size());
    try {
        for (auto id : device_ids) {
            auto pd = std::make_unique<PerDevice>(id);

            // Acquire the cross-process mutex first, then act on the dirty bit under the lock.
            pd->mutex.initialize();
            pd->mutex.lock();
            pd->locked = true;

            pd->dirty = open_dirty_shm(pd->id);

            if (*pd->dirty.ptr != 0) {
                run_tt_smi_reset(pd->id);
            }
            // Pessimistic: assume this process will dirty the device. Cleared in destructor on
            // graceful exit; persists if the process is killed before we get there.
            *pd->dirty.ptr = 1;

            per_device_.push_back(std::move(pd));
        }
    } catch (...) {
        // Roll back partial acquisitions before propagating.
        for (auto& pd : per_device_) {
            if (pd->dirty.ptr != nullptr) {
                *pd->dirty.ptr = 0;
                close_dirty_shm(pd->dirty);
            }
            if (pd->locked) {
                try {
                    pd->mutex.unlock();
                } catch (...) {
                    // best-effort
                }
                pd->locked = false;
            }
        }
        per_device_.clear();
        throw;
    }
}

SafeDeviceGuard::~SafeDeviceGuard() {
    for (auto& pd : per_device_) {
        if (pd->dirty.ptr != nullptr) {
            // If on_hang() fired, leave dirty=1 so the next process double-checks via tt-smi -r.
            // Otherwise this was a graceful run — clear the bit.
            if (!hang_handled_.load(std::memory_order_acquire)) {
                *pd->dirty.ptr = 0;
            }
            close_dirty_shm(pd->dirty);
        }
        if (pd->locked) {
            try {
                pd->mutex.unlock();
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "SafeDeviceGuard: unlock failed for device {}: {}", pd->id, e.what());
            }
            pd->locked = false;
        }
    }
}

void SafeDeviceGuard::on_hang() {
    bool expected = false;
    if (!hang_handled_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;  // already handled
    }
    for (auto& pd : per_device_) {
        // Mark dirty in shmem first so any handler crash before the reset completes still leaves
        // the next acquirer with a definitive "must reset" signal.
        if (pd->dirty.ptr != nullptr) {
            *pd->dirty.ptr = 1;
        }
        run_tt_smi_reset(pd->id);
    }
}

}  // namespace tt::tt_metal
