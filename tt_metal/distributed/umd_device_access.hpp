// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace tt::umd {
class Cluster;
}

namespace tt::tt_metal::distributed {

/**
 * Lightweight UMD-only device access for cross-process socket connectors.
 *
 * Opens the UMD driver for a single chip without going through MetalContext
 * or tt-metal's Cluster (no ethernet firmware init, no RISC reset).
 * Provides a pcie_writer callable suitable for H2D/D2H socket connectors.
 */
class UmdDeviceAccess {
public:
    /**
     * @param device_id Physical chip ID to open.
     * @param virtual_core_x Pre-resolved virtual (translated) core X coordinate.
     * @param virtual_core_y Pre-resolved virtual (translated) core Y coordinate.
     */
    UmdDeviceAccess(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y);
    ~UmdDeviceAccess();

    UmdDeviceAccess(const UmdDeviceAccess&) = delete;
    UmdDeviceAccess& operator=(const UmdDeviceAccess&) = delete;
    UmdDeviceAccess(UmdDeviceAccess&&) noexcept;
    UmdDeviceAccess& operator=(UmdDeviceAccess&&) noexcept;

    /**
     * Returns a writer function: (void* data, uint32_t num_bytes, uint64_t device_addr).
     * Uses UMD's write_to_device with dynamic TLB reconfiguration per write.
     */
    std::function<void(void*, uint32_t, uint64_t)> get_pcie_writer() const;

private:
    std::unique_ptr<tt::umd::Cluster> umd_cluster_;
    uint32_t device_id_ = 0;
    uint32_t virtual_core_x_ = 0;
    uint32_t virtual_core_y_ = 0;
};

}  // namespace tt::tt_metal::distributed
