// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <tt-metalium/device_types.hpp>
#include <umd/device/utils/robust_mutex.hpp>

namespace tt::tt_metal {

// Cooperative cross-process device guard.
//
// On construction, acquires a per-device cross-process mutex (RobustMutex via /dev/shm) and
// checks a sibling /dev/shm "dirty" bit. If the previous holder did not release cleanly
// (process killed mid-workload, or the dispatch timeout handler tagged the device on hang),
// shells out to `tt-smi -r <chip_id>` to recover before proceeding. Sets the dirty bit for
// the duration of this guard's lifetime.
//
// Replaces the flock + dirty-file + tt-smi shellout choreography in scripts/run_safe_pytest.sh,
// moving it inside tt-metal device open. This is a userspace stopgap until the KMD lock-63
// pattern (auto-release on FD close) becomes viable.
//
// Gated externally on TT_METAL_SAFE_DEVICE_OPEN=1.
class SafeDeviceGuard {
public:
    // Acquires the lock for each device id. Reset-on-dirty happens before this returns.
    explicit SafeDeviceGuard(const std::vector<tt::ChipId>& device_ids);

    // Clears the dirty bit and releases the locks. If on_hang() was called, leaves the dirty
    // bit set so the next process resets again as a belt-and-suspenders measure.
    ~SafeDeviceGuard();

    SafeDeviceGuard(const SafeDeviceGuard&) = delete;
    SafeDeviceGuard& operator=(const SafeDeviceGuard&) = delete;

    // Called by MetalContext::on_dispatch_timeout_detected. Runs `tt-smi -r <id>` synchronously
    // for each held device. Idempotent — safe to call from multiple threads; second call is a no-op.
    void on_hang();

private:
    struct DirtyShm {
        // Backed by /dev/shm/TT_METAL_DEVICE_DIRTY.<id>, a single byte mmap'd as uint8_t.
        // Set to 1 before workload, 0 on graceful release. Persists across crash/SIGKILL.
        uint8_t* ptr = nullptr;
        int fd = -1;
    };

    struct PerDevice {
        tt::ChipId id;
        tt::umd::RobustMutex mutex;
        DirtyShm dirty;
        bool locked = false;

        explicit PerDevice(tt::ChipId id);
    };

    static DirtyShm open_dirty_shm(tt::ChipId id);
    static void close_dirty_shm(DirtyShm& s);
    static void run_tt_smi_reset(tt::ChipId id);

    std::vector<std::unique_ptr<PerDevice>> per_device_;
    std::atomic<bool> hang_handled_{false};
};

}  // namespace tt::tt_metal
