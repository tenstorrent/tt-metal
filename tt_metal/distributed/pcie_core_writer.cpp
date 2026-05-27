// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/pcie_core_writer.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::distributed {

std::mutex PCIeCoreWriter::cluster_mutex_;
std::unique_ptr<tt::umd::Cluster> PCIeCoreWriter::shared_cluster_;

tt::umd::Cluster* PCIeCoreWriter::get_or_create_cluster() {
    std::lock_guard lock(cluster_mutex_);
    if (!shared_cluster_) {
        // Single cluster for the whole process — see header comment.
        // No `target_devices` set: that field is documented as a no-op for
        // SILICON, so leaving it empty avoids implying constraints UMD does
        // not honour. The Cluster ctor discovers all visible PCIe chips.
        shared_cluster_ = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{});
    }
    return shared_cluster_.get();
}

PCIeCoreWriter::PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y) :
    device_id_(device_id), virtual_core_x_(virtual_core_x), virtual_core_y_(virtual_core_y) {
    // Force-init on first construction so the topology-discovery cost (and
    // its fd usage) is paid up front in a known place rather than lazily
    // inside `get_pcie_writer()` during the first send.
    get_or_create_cluster();
}

std::function<void(void*, uint32_t, uint64_t)> PCIeCoreWriter::get_pcie_writer() const {
    auto* cluster = get_or_create_cluster();
    auto chip_id = static_cast<int>(device_id_);
    auto vx = virtual_core_x_;
    auto vy = virtual_core_y_;
    tt::umd::CoreCoord core(vx, vy, CoreType::TENSIX, CoordSystem::TRANSLATED);
    return [cluster, chip_id, core](void* data, uint32_t num_bytes, uint64_t device_addr) {
        cluster->write_to_device(data, num_bytes, chip_id, core, device_addr);
    };
}

}  // namespace tt::tt_metal::distributed
