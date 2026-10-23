// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
 * Opens the UMD driver for the local host without going through MetalContext
 * or tt-metal's Cluster (no ethernet firmware init, no RISC reset).
 *
 * A single `tt::umd::Cluster` is shared across all PCIeCoreWriters in this process to
 * avoid the file-descriptor exhaustion of building one cluster per chip.
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
    static tt::umd::Cluster* get_or_create_cluster();

    static std::mutex cluster_mutex_;
    static std::unique_ptr<tt::umd::Cluster> shared_cluster_;

    uint32_t device_id_ = 0;
    uint32_t virtual_core_x_ = 0;
    uint32_t virtual_core_y_ = 0;
};

}  // namespace tt::tt_metal::distributed
