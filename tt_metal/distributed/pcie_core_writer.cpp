// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/pcie_core_writer.hpp"

#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::distributed {

std::mutex PCIeCoreWriter::cluster_cache_mutex_;
std::unordered_map<uint32_t, std::unique_ptr<tt::umd::Cluster>> PCIeCoreWriter::cluster_cache_;

tt::umd::Cluster* PCIeCoreWriter::get_or_create_cluster(uint32_t device_id) {
    std::lock_guard lock(cluster_cache_mutex_);
    auto it = cluster_cache_.find(device_id);
    if (it != cluster_cache_.end()) {
        return it->second.get();
    }
    auto cluster = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
        .target_devices = {static_cast<int>(device_id)},
    });
    auto* ptr = cluster.get();
    cluster_cache_.emplace(device_id, std::move(cluster));
    return ptr;
}

PCIeCoreWriter::PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y) :
    device_id_(device_id), virtual_core_x_(virtual_core_x), virtual_core_y_(virtual_core_y) {
    get_or_create_cluster(device_id);
}

std::function<void(void*, uint32_t, uint64_t)> PCIeCoreWriter::get_pcie_writer() const {
    auto* cluster = get_or_create_cluster(device_id_);
    auto chip_id = static_cast<int>(device_id_);
    auto vx = virtual_core_x_;
    auto vy = virtual_core_y_;
    tt::umd::CoreCoord core(vx, vy, CoreType::TENSIX, CoordSystem::TRANSLATED);
    return [cluster, chip_id, core](void* data, uint32_t num_bytes, uint64_t device_addr) {
        cluster->write_to_device(data, num_bytes, chip_id, core, device_addr);
    };
}

}  // namespace tt::tt_metal::distributed
