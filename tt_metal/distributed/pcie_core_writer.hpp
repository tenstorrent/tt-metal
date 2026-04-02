// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tt::umd {
class Cluster;
}

namespace tt::tt_metal::distributed {

/**
 * Lightweight UMD-only device access for cross-process socket connectors.
 *
 * Opens the UMD driver for a single chip without going through MetalContext
 * or tt-metal's Cluster (no ethernet firmware init, no RISC reset).
 * Caches the UMD Cluster per device_id so topology discovery runs only once.
 */
class PCIeCoreWriter {
public:
    PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y);

    PCIeCoreWriter(const PCIeCoreWriter&) = delete;
    PCIeCoreWriter& operator=(const PCIeCoreWriter&) = delete;
    PCIeCoreWriter(PCIeCoreWriter&&) noexcept = default;
    PCIeCoreWriter& operator=(PCIeCoreWriter&&) noexcept = default;

    std::function<void(void*, uint32_t, uint64_t)> get_pcie_writer() const;

private:
    static tt::umd::Cluster* get_or_create_cluster(uint32_t device_id);

    static std::mutex cluster_cache_mutex_;
    static std::unordered_map<uint32_t, std::unique_ptr<tt::umd::Cluster>> cluster_cache_;

    uint32_t device_id_ = 0;
    uint32_t virtual_core_x_ = 0;
    uint32_t virtual_core_y_ = 0;
};

}  // namespace tt::tt_metal::distributed
