// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <string_view>
#include <vector>

#include <tt-metalium/device_types.hpp>
#include <umd/device/utils/robust_mutex.hpp>

namespace tt::tt_metal {

// Cooperative cross-process device guard.
//
// On construction, acquires a single cross-process mutex covering the full mesh (all device IDs
// together) and checks a sibling /dev/shm dirty bit. If the previous holder did not release
// cleanly (process killed mid-workload, or the dispatch timeout handler tagged the mesh on hang),
// shells out to `tt-smi -r` to reset all devices before proceeding. Sets the dirty bit for the
// duration of this guard's lifetime.
//
// A single mesh-scoped mutex is correct for multi-device setups: acquisition is all-or-nothing for
// the whole set of chips, avoiding the partial-acquisition races that per-chip locks introduce.
// Using `tt-smi -r` with no device ID also sidesteps the BH GLX /dev/tenstorrent/<N> shuffle that
// can occur after a per-device reset.
//
// The mutex name and dirty shm are derived from the sorted set of chip IDs, so two processes
// sharing the same mesh contend on the same lock, while non-overlapping meshes run in parallel.
//
// Replaces the flock + dirty-file + tt-smi shellout choreography in scripts/run_safe_pytest.sh.
// This is a userspace stopgap until the KMD lock-63 pattern (auto-release on FD close) is viable.
//
// Gated externally on TT_METAL_SAFE_DEVICE_OPEN=1.
class SafeDeviceGuard {
public:
    // Acquires the mesh-scoped lock. Reset-on-dirty happens before this returns.
    explicit SafeDeviceGuard(const std::vector<tt::ChipId>& device_ids);

    // Clears the dirty bit and releases the lock. If on_hang() was called, leaves the dirty bit
    // set so the next process resets again as a belt-and-suspenders measure.
    ~SafeDeviceGuard();

    SafeDeviceGuard(const SafeDeviceGuard&) = delete;
    SafeDeviceGuard& operator=(const SafeDeviceGuard&) = delete;

    // Called by MetalContext::on_dispatch_timeout_detected. Runs `tt-smi -r` synchronously.
    // Idempotent — safe to call from multiple threads; second call is a no-op.
    void on_hang();

private:
    struct DirtyShm {
        // Single byte in /dev/shm, mmap'd as uint8_t.
        // Set to 1 before workload, 0 on graceful release. Persists across crash/SIGKILL.
        uint8_t* ptr = nullptr;
        int fd = -1;
    };

    static DirtyShm open_dirty_shm(std::string_view shm_name);
    static void close_dirty_shm(DirtyShm& s);
    static void run_tt_smi_reset();

    std::vector<tt::ChipId> device_ids_;  // for logging
    tt::umd::RobustMutex mutex_;
    DirtyShm dirty_;
    bool locked_ = false;
    std::atomic<bool> hang_handled_{false};
};

}  // namespace tt::tt_metal
